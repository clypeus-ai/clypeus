//! OpenAI-compatible Chat Completions provider.
//!
//! Maps Clypeus completion requests to `POST /v1/chat/completions` (buffered
//! and streamed) and parses the responses, including reasoning deltas,
//! `tool_calls` fragments, and usage. The provider also reads the model
//! catalog from `GET /v1/models`, tolerating the reasoning metadata shapes
//! OpenAI-compatible gateways commonly expose.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use clypeus_core::models::{ChatMessage, ChatRole, TokenUsage, ToolCall, ToolSpec};
use clypeus_core::provider::{
    AssistantOutcome, CompletionRequest, ModelCapability, ModelCatalog, PayloadEvent, ProbeReport,
    Provider, ProviderConfig, ProviderError, ProviderStream, RoundDelta, StreamDecoder,
    extract_error_message, validate_base_url, walk_path,
};
use futures_util::StreamExt;
use serde_json::{Map, Value, json};

const MODEL_CATALOG_TIMEOUT: Duration = Duration::from_secs(7);
const PROBE_TIMEOUT: Duration = Duration::from_secs(7);
/// Upper bound for the provider to answer and start streaming.
const UPSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// OpenAI-compatible provider.
#[derive(Clone)]
pub struct OpenAiProvider {
    http: reqwest::Client,
}

impl std::fmt::Debug for OpenAiProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("OpenAiProvider")
    }
}

impl Default for OpenAiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiProvider {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
        }
    }

    pub fn with_client(http: reqwest::Client) -> Self {
        Self { http }
    }
}

impl OpenAiProvider {
    fn endpoint(&self, config: &ProviderConfig, path: &str) -> Result<String, ProviderError> {
        let base = validate_base_url(&config.base_url, config.allow_private_targets)?;
        Ok(format!("{base}{path}"))
    }

    fn auth(
        &self,
        request: reqwest::RequestBuilder,
        config: &ProviderConfig,
    ) -> reqwest::RequestBuilder {
        request.bearer_auth(config.api_key.expose())
    }
}

#[async_trait::async_trait]
impl Provider for OpenAiProvider {
    fn id(&self) -> &'static str {
        "openai"
    }

    async fn catalog(&self, config: &ProviderConfig) -> Result<ModelCatalog, ProviderError> {
        let url = self.endpoint(config, "/v1/models")?;
        let response = self
            .auth(self.http.get(&url).timeout(MODEL_CATALOG_TIMEOUT), config)
            .header("accept", "application/json")
            .send()
            .await
            .map_err(map_send_error)?;
        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
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
        if !is_jsonish(&content_type, &body) {
            return Err(ProviderError::InvalidPayload(
                "Provider returned a non-JSON catalog; check the provider base URL.".into(),
            ));
        }
        let root: Value = serde_json::from_str(&body)
            .map_err(|error| ProviderError::InvalidPayload(error.to_string()))?;
        Ok(parse_model_catalog(&root))
    }

    async fn probe(&self, config: &ProviderConfig) -> ProbeReport {
        let started = Instant::now();
        let result = async {
            let url = self.endpoint(config, "/v1/models")?;
            let response = self
                .auth(self.http.get(&url).timeout(PROBE_TIMEOUT), config)
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
                    let model_count = serde_json::from_str::<Value>(&body).ok().and_then(|value| {
                        value
                            .get("data")
                            .or_else(|| value.get("models"))
                            .and_then(Value::as_array)
                            .map(|array| array.len() as i32)
                    });
                    ProbeReport {
                        succeeded: true,
                        model_count,
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
        let url = self.endpoint(config, "/v1/chat/completions")?;
        let mut payload = build_payload(&request, false);
        let response = self
            .auth(self.http.post(&url), config)
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
            let outcomes = extract_outcomes(&value);
            if !outcomes.is_empty() {
                return Ok(outcomes
                    .into_iter()
                    .find(|outcome| !outcome.is_empty())
                    .unwrap_or_default());
            }
            if request.reasoning.is_some() {
                strip_reasoning(&mut payload);
                let retry = self
                    .auth(self.http.post(&url), config)
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
                    if let Some(outcome) = extract_outcomes(&value)
                        .into_iter()
                        .find(|outcome| !outcome.is_empty())
                    {
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
                .auth(self.http.post(&url), config)
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
                if let Some(outcome) = extract_outcomes(&value)
                    .into_iter()
                    .find(|outcome| !outcome.is_empty())
                {
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
        let url = self.endpoint(config, "/v1/chat/completions")?;
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
            Box::new(OpenAiStreamDecoder::default()),
        ))
    }
}

impl OpenAiProvider {
    async fn start_stream(
        &self,
        url: &str,
        config: &ProviderConfig,
        payload: &Value,
    ) -> Result<reqwest::Response, ProviderError> {
        tokio::time::timeout(
            UPSTREAM_CONNECT_TIMEOUT,
            self.auth(self.http.post(url), config)
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

fn map_send_error(error: reqwest::Error) -> ProviderError {
    if error.is_timeout() {
        ProviderError::Timeout
    } else {
        ProviderError::Transport(error.to_string())
    }
}

fn is_jsonish(content_type: &str, body: &str) -> bool {
    if content_type.contains("json") {
        return true;
    }
    let trimmed = body.trim_start();
    trimmed.starts_with('{') || trimmed.starts_with('[')
}

/// Builds the Chat Completions payload.
pub fn build_payload(request: &CompletionRequest, stream: bool) -> Value {
    let mut payload = Map::new();
    payload.insert("model".into(), Value::String(request.model.clone()));
    payload.insert(
        "messages".into(),
        Value::Array(request.messages.iter().map(openai_message).collect()),
    );
    if !request.tools.is_empty() {
        payload.insert(
            "tools".into(),
            Value::Array(request.tools.iter().map(openai_tool).collect()),
        );
        payload.insert(
            "tool_choice".into(),
            Value::String(request.tool_choice.as_wire().to_string()),
        );
    }
    payload.insert(
        "max_tokens".into(),
        Value::Number(serde_json::Number::from(request.max_output_tokens)),
    );
    if stream {
        payload.insert("stream".into(), Value::Bool(true));
        payload.insert("stream_options".into(), json!({"include_usage": true}));
    }
    if let Some(reasoning) = request.reasoning
        && reasoning != clypeus_core::models::ReasoningLevel::Default
    {
        payload.insert(
            "reasoning_effort".into(),
            Value::String(reasoning.as_wire().to_string()),
        );
    }
    Value::Object(payload)
}

fn strip_reasoning(payload: &mut Value) {
    if let Some(object) = payload.as_object_mut() {
        object.remove("reasoning_effort");
        object.remove("reasoning");
    }
}

fn openai_tool(tool: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
    })
}

fn openai_message(message: &ChatMessage) -> Value {
    match message.role {
        ChatRole::Tool => json!({
            "role": "tool",
            "tool_call_id": message.tool_call_id.clone().unwrap_or_default(),
            "content": message.content,
        }),
        ChatRole::Assistant
            if message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty()) =>
        {
            let calls: Vec<Value> = message
                .tool_calls
                .as_ref()
                .expect("tool calls checked above")
                .iter()
                .map(|call| {
                    json!({
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments.to_string(),
                        }
                    })
                })
                .collect();
            let content = if message.content.is_empty() {
                Value::Null
            } else {
                Value::String(message.content.clone())
            };
            json!({"role": "assistant", "content": content, "tool_calls": calls})
        }
        _ => json!({"role": message.role, "content": message.content}),
    }
}

/// Parses every assistant choice out of a buffered payload.
pub fn extract_outcomes(value: &Value) -> Vec<AssistantOutcome> {
    let mut outcomes = Vec::new();
    let root_model = value
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    let root_usage = value.get("usage").and_then(parse_usage);
    let Some(choices) = value.get("choices").and_then(Value::as_array) else {
        return outcomes;
    };
    for choice in choices {
        let message = choice.get("message");
        let content = message
            .and_then(|message| message.get("content"))
            .and_then(|content| match content {
                Value::String(text) => Some(text.clone()),
                Value::Array(parts) => Some(
                    parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join(""),
                ),
                _ => None,
            })
            .unwrap_or_default();
        let reasoning = message
            .and_then(|message| {
                message
                    .get("reasoning_content")
                    .or_else(|| message.get("reasoning"))
            })
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .map(str::to_string);
        let model = choice
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| root_model.clone());
        let usage = choice
            .get("usage")
            .and_then(parse_usage)
            .or_else(|| root_usage.clone());
        let tool_calls = parse_tool_calls(message);
        if content.is_empty() && reasoning.is_none() && tool_calls.is_empty() {
            continue;
        }
        outcomes.push(AssistantOutcome {
            content,
            reasoning,
            model,
            usage,
            tool_calls,
        });
    }
    outcomes
}

fn parse_tool_calls(message: Option<&Value>) -> Vec<ToolCall> {
    let Some(calls) = message
        .and_then(|message| message.get("tool_calls"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    calls
        .iter()
        .filter_map(|call| {
            let id = call.get("id").and_then(Value::as_str)?;
            let function = call.get("function")?;
            let name = function.get("name").and_then(Value::as_str)?;
            let arguments = match function.get("arguments") {
                Some(Value::String(text)) => {
                    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone()))
                }
                Some(value) => value.clone(),
                None => Value::Null,
            };
            Some(ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments,
            })
        })
        .collect()
}

fn parse_usage(value: &Value) -> Option<TokenUsage> {
    let usage = TokenUsage {
        prompt_tokens: value
            .get("prompt_tokens")
            .and_then(Value::as_i64)
            .and_then(|value| value.try_into().ok()),
        completion_tokens: value
            .get("completion_tokens")
            .and_then(Value::as_i64)
            .and_then(|value| value.try_into().ok()),
        total_tokens: value
            .get("total_tokens")
            .and_then(Value::as_i64)
            .and_then(|value| value.try_into().ok()),
        cached_tokens: value
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_i64)
            .and_then(|value| value.try_into().ok()),
        reasoning_tokens: value
            .pointer("/completion_tokens_details/reasoning_tokens")
            .and_then(Value::as_i64)
            .and_then(|value| value.try_into().ok()),
    };
    (!usage.is_empty()).then_some(usage)
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Incremental decoder for Chat Completions SSE payloads.
#[derive(Default)]
pub struct OpenAiStreamDecoder {
    content: String,
    reasoning: String,
    usage: Option<TokenUsage>,
    tool_calls: BTreeMap<u64, PartialToolCall>,
}

impl StreamDecoder for OpenAiStreamDecoder {
    fn consume(&mut self, payload: &str) -> PayloadEvent {
        if payload.is_empty() {
            return PayloadEvent::Data(RoundDelta::default());
        }
        if payload == "[DONE]" {
            return PayloadEvent::Done;
        }
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            // Non-JSON providers stream plain text deltas.
            self.content.push_str(payload);
            return PayloadEvent::Data(RoundDelta {
                content: Some(payload.to_string()),
                ..RoundDelta::default()
            });
        };
        if let Some(detail) = value
            .get("error")
            .and_then(|error| {
                error
                    .get("message")
                    .and_then(Value::as_str)
                    .or_else(|| error.as_str())
            })
            .filter(|detail| !detail.trim().is_empty())
        {
            return PayloadEvent::Error(detail.to_string());
        }
        let mut round = RoundDelta::default();
        if let Some(usage) = value.get("usage").and_then(parse_usage) {
            self.merge_usage(usage.clone());
            round.usage = Some(usage);
        }
        for choice in value
            .get("choices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(usage) = choice.get("usage").and_then(parse_usage) {
                self.merge_usage(usage.clone());
                round.usage = Some(usage);
            }
            let Some(delta) = choice.get("delta") else {
                continue;
            };
            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                self.content.push_str(content);
                round
                    .content
                    .get_or_insert_with(String::new)
                    .push_str(content);
            }
            let reasoning = delta
                .get("reasoning_content")
                .and_then(Value::as_str)
                .or_else(|| delta.get("reasoning").and_then(Value::as_str));
            if let Some(reasoning) = reasoning {
                self.reasoning.push_str(reasoning);
                round
                    .reasoning
                    .get_or_insert_with(String::new)
                    .push_str(reasoning);
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    self.consume_tool_call(call);
                }
            }
        }
        PayloadEvent::Data(round)
    }

    fn outcome(&self) -> AssistantOutcome {
        let tool_calls = self
            .tool_calls
            .values()
            .map(|partial| {
                let arguments = if partial.arguments.trim().is_empty() {
                    Value::Object(Map::new())
                } else {
                    serde_json::from_str(&partial.arguments)
                        .unwrap_or_else(|_| Value::String(partial.arguments.clone()))
                };
                ToolCall {
                    id: partial.id.clone(),
                    name: partial.name.clone(),
                    arguments,
                }
            })
            .collect();
        AssistantOutcome {
            content: self.content.clone(),
            reasoning: (!self.reasoning.is_empty()).then(|| self.reasoning.clone()),
            model: None,
            usage: self.usage.clone(),
            tool_calls,
        }
    }
}

impl OpenAiStreamDecoder {
    fn consume_tool_call(&mut self, call: &Value) {
        let index = call
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or(self.tool_calls.len() as u64);
        let entry = self.tool_calls.entry(index).or_default();
        if let Some(id) = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            entry.id = id.to_string();
        }
        let Some(function) = call.get("function") else {
            return;
        };
        if let Some(name) = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        {
            entry.name = name.to_string();
        }
        if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
            entry.arguments.push_str(arguments);
        }
    }

    fn merge_usage(&mut self, incoming: TokenUsage) {
        let merged = self.usage.get_or_insert_with(TokenUsage::default);
        macro_rules! merge_field {
            ($field:ident) => {
                if incoming.$field.is_some() {
                    merged.$field = incoming.$field;
                }
            };
        }
        merge_field!(prompt_tokens);
        merge_field!(completion_tokens);
        merge_field!(total_tokens);
        merge_field!(cached_tokens);
        merge_field!(reasoning_tokens);
    }
}

/// Parses a model catalog from the many shapes OpenAI-compatible gateways use.
pub fn parse_model_catalog(root: &Value) -> ModelCatalog {
    let array = root
        .get("data")
        .or_else(|| root.get("Data"))
        .or_else(|| root.get("models"))
        .or_else(|| root.get("Models"))
        .and_then(Value::as_array);
    let Some(items) = array else {
        return ModelCatalog::default();
    };
    ModelCatalog {
        models: items.iter().filter_map(parse_model_capability).collect(),
    }
}

fn parse_model_capability(item: &Value) -> Option<ModelCapability> {
    let model = item
        .get("id")
        .or_else(|| item.get("Id"))
        .or_else(|| item.get("name"))
        .or_else(|| item.get("Name"))
        .or_else(|| item.get("model"))
        .or_else(|| item.get("Model"))
        .or_else(|| item.get("slug"))
        .or_else(|| item.get("Slug"))
        .and_then(Value::as_str)?
        .trim()
        .to_string();
    if model.is_empty() {
        return None;
    }
    let levels = if reasoning_supported(item) == Some(false) {
        Vec::new()
    } else {
        extract_reasoning_levels(item)
    };
    let default = extract_default_reasoning_level(item)
        .filter(|value| levels.iter().any(|level| level == value))
        .or_else(|| levels.first().cloned())
        .unwrap_or_default();
    Some(ModelCapability {
        model,
        reasoning_levels: levels,
        default_reasoning_level: default,
    })
}

fn reasoning_supported(item: &Value) -> Option<bool> {
    item.get("reasoning")
        .and_then(|value| value.get("supported"))
        .and_then(Value::as_bool)
        .or_else(|| {
            item.get("capabilities")
                .and_then(|value| value.get("reasoning"))
                .and_then(Value::as_bool)
        })
}

fn extract_default_reasoning_level(item: &Value) -> Option<String> {
    let candidates = [
        ["default_reasoning_effort"].as_slice(),
        &["reasoning", "default"],
        &["metadata", "default_reasoning_effort"],
        &["metadata", "reasoning", "default"],
    ];
    for path in &candidates {
        if let Some(value) = walk_path(item, path).and_then(Value::as_str) {
            let lower = value.trim().to_ascii_lowercase();
            if !lower.is_empty() && lower != "null" {
                return Some(lower);
            }
        }
    }
    None
}

fn extract_reasoning_levels(item: &Value) -> Vec<String> {
    let mut found = Vec::new();
    let candidates = [
        ["supported_reasoning_levels"].as_slice(),
        &["supported_reasoning_efforts"],
        &["reasoning", "supported_efforts"],
        &["reasoning", "supported_levels"],
        &["reasoning", "levels"],
        &["reasoning", "efforts"],
        &["reasoning_levels"],
        &["reasoning_efforts"],
        &["metadata", "supported_reasoning_levels"],
        &["metadata", "supported_reasoning_efforts"],
        &["metadata", "reasoning", "supported_efforts"],
        &["metadata", "reasoning", "levels"],
    ];
    for path in &candidates {
        if let Some(array) = walk_path(item, path).and_then(Value::as_array) {
            for value in array {
                if let Some(level) = value.as_str() {
                    let lower = level.trim().to_ascii_lowercase();
                    if !lower.is_empty() && !found.contains(&lower) {
                        found.push(lower);
                    }
                }
            }
        }
    }
    found
}

impl std::fmt::Debug for OpenAiStreamDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiStreamDecoder")
            .field("content_len", &self.content.len())
            .field("tool_calls", &self.tool_calls.len())
            .finish()
    }
}
