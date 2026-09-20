//! Anthropic Messages provider.
//!
//! Maps Clypeus completion requests to `POST /v1/messages` (buffered and
//! streamed) and parses the responses, including `thinking` blocks,
//! `tool_use` blocks with `input_json_delta` fragments, and usage. The model
//! catalog comes from `GET /v1/models` with the `anthropic-version` header.

use std::time::{Duration, Instant};

use clypeus_core::models::{ChatMessage, ChatRole, ReasoningLevel, TokenUsage, ToolCall, ToolSpec};
use clypeus_core::provider::{
    AssistantOutcome, CompletionRequest, ModelCapability, ModelCatalog, PayloadEvent, ProbeReport,
    Provider, ProviderConfig, ProviderError, ProviderStream, RoundDelta, StreamAccumulator,
    StreamDecoder, extract_error_message, validate_base_url,
};
use futures_util::StreamExt;
use serde_json::{Map, Value, json};

const MODEL_CATALOG_TIMEOUT: Duration = Duration::from_secs(7);
const PROBE_TIMEOUT: Duration = Duration::from_secs(7);
const UPSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Messages provider.
#[derive(Clone)]
pub struct AnthropicProvider {
    http: reqwest::Client,
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("AnthropicProvider")
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicProvider {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
        }
    }

    pub fn with_client(http: reqwest::Client) -> Self {
        Self { http }
    }

    fn endpoint(&self, config: &ProviderConfig) -> Result<String, ProviderError> {
        let base = validate_base_url(&config.base_url, config.allow_private_targets)?;
        Ok(format!("{base}/v1/messages"))
    }

    fn headers(
        &self,
        request: reqwest::RequestBuilder,
        config: &ProviderConfig,
    ) -> reqwest::RequestBuilder {
        request
            .header("x-api-key", config.api_key.expose())
            .header("anthropic-version", ANTHROPIC_VERSION)
    }

    async fn start_stream(
        &self,
        url: &str,
        config: &ProviderConfig,
        payload: &Value,
    ) -> Result<reqwest::Response, ProviderError> {
        tokio::time::timeout(
            UPSTREAM_CONNECT_TIMEOUT,
            self.headers(self.http.post(url), config)
                .header("content-type", "application/json")
                .header("accept", "text/event-stream")
                .json(payload)
                .send(),
        )
        .await
        .map_err(|_| ProviderError::Timeout)?
        .map_err(map_send_error)
    }
}

#[async_trait::async_trait]
impl Provider for AnthropicProvider {
    fn id(&self) -> &'static str {
        "anthropic"
    }

    async fn catalog(&self, config: &ProviderConfig) -> Result<ModelCatalog, ProviderError> {
        let base = validate_base_url(&config.base_url, config.allow_private_targets)?;
        let url = format!("{base}/v1/models");
        let response = self
            .headers(self.http.get(&url).timeout(MODEL_CATALOG_TIMEOUT), config)
            .header("accept", "application/json")
            .send()
            .await
            .map_err(map_send_error)?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| ProviderError::Transport(error.to_string()))?;
        if !status.is_success() {
            return Err(ProviderError::Upstream {
                status: status.as_u16(),
                detail: extract_error_message(&body)
                    .unwrap_or_else(|| status.canonical_reason().unwrap_or("error").to_string()),
            });
        }
        let root: Value = serde_json::from_str(&body)
            .map_err(|error| ProviderError::InvalidPayload(error.to_string()))?;
        Ok(parse_anthropic_catalog(&root))
    }

    async fn probe(&self, config: &ProviderConfig) -> ProbeReport {
        let started = Instant::now();
        let result = async {
            let base = validate_base_url(&config.base_url, config.allow_private_targets)?;
            let url = format!("{base}/v1/models");
            let response = self
                .headers(self.http.get(&url).timeout(PROBE_TIMEOUT), config)
                .header("accept", "application/json")
                .send()
                .await
                .map_err(map_send_error)?;
            Ok::<_, ProviderError>(response)
        }
        .await;
        let elapsed_ms = started.elapsed().as_millis() as i64;
        match result {
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                if status.is_success() {
                    let count = serde_json::from_str::<Value>(&body).ok().and_then(|value| {
                        value
                            .get("data")
                            .or_else(|| value.get("models"))
                            .and_then(Value::as_array)
                            .map(|array| array.len() as i32)
                    });
                    ProbeReport {
                        succeeded: true,
                        model_count: count,
                        elapsed_ms,
                        checked_at_utc: chrono::Utc::now(),
                        error: None,
                    }
                } else {
                    ProbeReport {
                        succeeded: false,
                        model_count: None,
                        elapsed_ms,
                        checked_at_utc: chrono::Utc::now(),
                        error: extract_error_message(&body)
                            .or_else(|| Some(format!("HTTP {}", status.as_u16()))),
                    }
                }
            }
            Err(error) => ProbeReport {
                succeeded: false,
                model_count: None,
                elapsed_ms,
                checked_at_utc: chrono::Utc::now(),
                error: Some(if error == ProviderError::Timeout {
                    "timeout".to_string()
                } else {
                    error.safe_message()
                }),
            },
        }
    }

    async fn complete(
        &self,
        config: &ProviderConfig,
        request: CompletionRequest,
    ) -> Result<AssistantOutcome, ProviderError> {
        let url = self.endpoint(config)?;
        let mut payload = build_payload(&request, false);
        let response = self
            .headers(self.http.post(&url), config)
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .json(&payload)
            .timeout(config.timeout)
            .send()
            .await
            .map_err(map_send_error)?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| ProviderError::Transport(error.to_string()))?;

        if status.is_success() {
            let value: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            let outcome = extract_outcome(&value);
            if !outcome.is_empty() {
                return Ok(outcome);
            }
            if request.reasoning.is_some() {
                strip_reasoning(&mut payload);
                let retry = self
                    .headers(self.http.post(&url), config)
                    .header("content-type", "application/json")
                    .header("accept", "application/json")
                    .json(&payload)
                    .timeout(config.timeout)
                    .send()
                    .await
                    .map_err(map_send_error)?;
                let retry_status = retry.status();
                let retry_body = retry.text().await.unwrap_or_default();
                if retry_status.is_success() {
                    let value: Value = serde_json::from_str(&retry_body).unwrap_or(Value::Null);
                    let outcome = extract_outcome(&value);
                    if !outcome.is_empty() {
                        return Ok(outcome);
                    }
                }
                return Err(ProviderError::InvalidPayload(
                    "upstream returned no response text".into(),
                ));
            }
            return Err(ProviderError::EmptyResponse);
        }

        if request.reasoning.is_some()
            && ProviderError::is_retryable_reasoning_failure(status.as_u16())
        {
            strip_reasoning(&mut payload);
            let retry = self
                .headers(self.http.post(&url), config)
                .header("content-type", "application/json")
                .header("accept", "application/json")
                .json(&payload)
                .timeout(config.timeout)
                .send()
                .await
                .map_err(map_send_error)?;
            let retry_status = retry.status();
            let retry_body = retry.text().await.unwrap_or_default();
            if retry_status.is_success() {
                let value: Value = serde_json::from_str(&retry_body).unwrap_or(Value::Null);
                let outcome = extract_outcome(&value);
                if !outcome.is_empty() {
                    return Ok(outcome);
                }
                return Err(ProviderError::InvalidPayload(
                    "upstream returned no response text on retry".into(),
                ));
            }
            return Err(ProviderError::Upstream {
                status: retry_status.as_u16(),
                detail: extract_error_message(&retry_body)
                    .unwrap_or_else(|| "upstream error on retry".into()),
            });
        }

        Err(ProviderError::Upstream {
            status: status.as_u16(),
            detail: extract_error_message(&body)
                .unwrap_or_else(|| status.canonical_reason().unwrap_or("upstream error").into()),
        })
    }

    async fn stream(
        &self,
        config: &ProviderConfig,
        request: CompletionRequest,
    ) -> Result<ProviderStream, ProviderError> {
        let url = self.endpoint(config)?;
        let mut payload = build_payload(&request, true);
        let response = match self.start_stream(&url, config, &payload).await? {
            response if response.status().is_success() => response,
            response
                if request.reasoning.is_some()
                    && ProviderError::is_retryable_reasoning_failure(
                        response.status().as_u16(),
                    ) =>
            {
                drop(response);
                strip_reasoning(&mut payload);
                let retry = self.start_stream(&url, config, &payload).await?;
                if !retry.status().is_success() {
                    let status = retry.status().as_u16();
                    let body = retry.text().await.unwrap_or_default();
                    return Err(ProviderError::Upstream {
                        status,
                        detail: extract_error_message(&body)
                            .unwrap_or_else(|| "upstream stream error".into()),
                    });
                }
                retry
            }
            response => {
                let status = response.status().as_u16();
                let body = response.text().await.unwrap_or_default();
                return Err(ProviderError::Upstream {
                    status,
                    detail: extract_error_message(&body)
                        .unwrap_or_else(|| "upstream stream error".into()),
                });
            }
        };
        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(std::io::Error::other))
            .boxed();
        Ok(ProviderStream::new(
            stream,
            Box::new(AnthropicStreamDecoder::default()),
        ))
    }
}

fn map_send_error(error: reqwest::Error) -> ProviderError {
    if error.is_timeout() {
        ProviderError::Timeout
    } else {
        ProviderError::Transport(error.to_string())
    }
}

/// Builds the Messages payload.
pub fn build_payload(request: &CompletionRequest, stream: bool) -> Value {
    let mut payload = Map::new();
    payload.insert("model".into(), Value::String(request.model.clone()));
    payload.insert(
        "messages".into(),
        Value::Array(
            request
                .messages
                .iter()
                .filter(|message| message.role != ChatRole::System)
                .map(anthropic_message)
                .collect(),
        ),
    );
    if let Some(system) = request
        .messages
        .iter()
        .find(|message| message.role == ChatRole::System)
    {
        payload.insert("system".into(), Value::String(system.content.clone()));
    }
    if !request.tools.is_empty() {
        payload.insert(
            "tools".into(),
            Value::Array(request.tools.iter().map(anthropic_tool).collect()),
        );
        payload.insert(
            "tool_choice".into(),
            json!({"type": match request.tool_choice {
                clypeus_core::provider::ToolChoice::None => "none",
                clypeus_core::provider::ToolChoice::Required => "any",
                clypeus_core::provider::ToolChoice::Auto => "auto",
            }}),
        );
    }
    payload.insert(
        "max_tokens".into(),
        Value::Number(serde_json::Number::from(request.max_output_tokens)),
    );
    if stream {
        payload.insert("stream".into(), Value::Bool(true));
    }
    if let Some(reasoning) = request
        .reasoning
        .filter(|level| *level != ReasoningLevel::Default)
    {
        payload.insert(
            "thinking".into(),
            json!({
                "type": "enabled",
                "budget_tokens": anthropic_budget_tokens(reasoning)
            }),
        );
    }
    Value::Object(payload)
}

fn strip_reasoning(payload: &mut Value) {
    if let Some(object) = payload.as_object_mut() {
        object.remove("thinking");
    }
}

/// Explicit thinking budget for each reasoning level.
pub fn anthropic_budget_tokens(level: ReasoningLevel) -> u32 {
    match level {
        ReasoningLevel::Minimal | ReasoningLevel::Low => 1_024,
        ReasoningLevel::Medium | ReasoningLevel::Default => 4_096,
        ReasoningLevel::High => 8_192,
        ReasoningLevel::XHigh => 16_000,
    }
}

fn anthropic_tool(tool: &ToolSpec) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    })
}

fn anthropic_message(message: &ChatMessage) -> Value {
    match message.role {
        ChatRole::Tool => json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": message.tool_call_id.clone().unwrap_or_default(),
                "content": message.content,
            }]
        }),
        ChatRole::Assistant
            if message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty()) =>
        {
            let mut blocks = Vec::new();
            if !message.content.is_empty() {
                blocks.push(json!({"type": "text", "text": message.content}));
            }
            for call in message
                .tool_calls
                .as_ref()
                .expect("tool calls checked above")
            {
                blocks.push(json!({
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }));
            }
            json!({"role": "assistant", "content": blocks})
        }
        _ => json!({"role": message.role, "content": message.content}),
    }
}

/// Parses one buffered Messages payload.
pub fn extract_outcome(value: &Value) -> AssistantOutcome {
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    let usage = value.get("usage").and_then(parse_usage);
    let Some(blocks) = value.get("content").and_then(Value::as_array) else {
        return AssistantOutcome::default();
    };
    let mut text_parts = Vec::new();
    let mut reasoning_parts = Vec::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("thinking") => {
                if let Some(text) = block.get("thinking").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    reasoning_parts.push(text);
                }
            }
            Some("tool_use") => {
                let Some(id) = block.get("id").and_then(Value::as_str) else {
                    continue;
                };
                let Some(name) = block.get("name").and_then(Value::as_str) else {
                    continue;
                };
                tool_calls.push(ToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments: block.get("input").cloned().unwrap_or(Value::Null),
                });
            }
            _ => {
                if let Some(text) = block.get("text").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    text_parts.push(text);
                }
            }
        }
    }
    AssistantOutcome {
        content: text_parts.join("\n"),
        reasoning: (!reasoning_parts.is_empty()).then(|| reasoning_parts.join("\n")),
        model,
        usage,
        tool_calls,
    }
}

fn parse_usage(value: &Value) -> Option<TokenUsage> {
    let prompt = value.get("input_tokens").and_then(Value::as_i64);
    let completion = value.get("output_tokens").and_then(Value::as_i64);
    let usage = TokenUsage {
        prompt_tokens: prompt.and_then(|value| value.try_into().ok()),
        completion_tokens: completion.and_then(|value| value.try_into().ok()),
        total_tokens: match (prompt, completion) {
            (Some(prompt), Some(completion)) => (prompt + completion).try_into().ok(),
            _ => None,
        },
        cached_tokens: value
            .get("cache_read_input_tokens")
            .and_then(Value::as_i64)
            .and_then(|value| value.try_into().ok()),
        reasoning_tokens: None,
    };
    (!usage.is_empty()).then_some(usage)
}

/// Incremental decoder for Messages SSE payloads.
#[derive(Default)]
pub struct AnthropicStreamDecoder {
    accumulator: StreamAccumulator,
}

impl StreamDecoder for AnthropicStreamDecoder {
    fn consume(&mut self, payload: &str) -> PayloadEvent {
        if payload.is_empty() {
            return PayloadEvent::Data(RoundDelta::default());
        }
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            self.accumulator.push_content(payload);
            return PayloadEvent::Data(RoundDelta {
                content: Some(payload.to_string()),
                ..RoundDelta::default()
            });
        };
        let mut round = RoundDelta::default();
        match value.get("type").and_then(Value::as_str) {
            Some("error") => {
                let detail = value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("provider stream error")
                    .to_string();
                return PayloadEvent::Error(detail);
            }
            Some("content_block_start") => {
                let Some(block) = value.get("content_block") else {
                    return PayloadEvent::Data(round);
                };
                if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                    let index = value.get("index").and_then(Value::as_u64).unwrap_or(0);
                    let id = block.get("id").and_then(Value::as_str).unwrap_or_default();
                    let name = block
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    self.accumulator.open_tool_use(index, id, name);
                    if let Some(input) = block.get("input") {
                        self.accumulator.set_tool_initial_input(index, input);
                    }
                }
            }
            Some("content_block_delta") => {
                let Some(delta) = value.get("delta") else {
                    return PayloadEvent::Data(round);
                };
                match delta.get("type").and_then(Value::as_str) {
                    Some("thinking_delta") => {
                        if let Some(thinking) = delta.get("thinking").and_then(Value::as_str) {
                            self.accumulator.push_reasoning(thinking);
                            round.reasoning = Some(thinking.to_string());
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(partial) = delta.get("partial_json").and_then(Value::as_str) {
                            let index = value.get("index").and_then(Value::as_u64).unwrap_or(0);
                            self.accumulator.push_tool_input(index, partial);
                        }
                    }
                    _ => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            self.accumulator.push_content(text);
                            round.content = Some(text.to_string());
                        }
                    }
                }
            }
            Some("message_start") => {
                if let Some(usage) = value.pointer("/message/usage").and_then(parse_usage) {
                    self.accumulator.merge_usage(usage.clone());
                    round.usage = Some(usage);
                }
            }
            Some("message_delta") => {
                if let Some(usage) = value.get("usage").and_then(parse_usage) {
                    self.accumulator.merge_usage(usage.clone());
                    round.usage = Some(usage);
                }
            }
            Some("message_stop") => return PayloadEvent::Done,
            _ => {}
        }
        PayloadEvent::Data(round)
    }

    fn outcome(&self) -> AssistantOutcome {
        self.accumulator.outcome()
    }
}

impl std::fmt::Debug for AnthropicStreamDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("AnthropicStreamDecoder")
    }
}

/// Parses a model catalog from the Messages models endpoint.
pub fn parse_anthropic_catalog(root: &Value) -> ModelCatalog {
    let array = root
        .get("data")
        .or_else(|| root.get("models"))
        .and_then(Value::as_array);
    let Some(items) = array else {
        return ModelCatalog::default();
    };
    ModelCatalog {
        models: items
            .iter()
            .filter_map(|item| {
                let model = item
                    .get("id")
                    .or_else(|| item.get("name"))
                    .and_then(Value::as_str)?
                    .trim()
                    .to_string();
                if model.is_empty() {
                    return None;
                }
                Some(ModelCapability {
                    model,
                    reasoning_levels: Vec::new(),
                    default_reasoning_level: String::new(),
                })
            })
            .collect(),
    }
}
