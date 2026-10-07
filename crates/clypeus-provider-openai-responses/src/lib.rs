//! OpenAI-compatible Responses provider.
//!
//! Maps Clypeus completion requests to `POST /v1/responses` (buffered and
//! streamed) and reads the typed output items, including reasoning summaries,
//! `function_call` items with `function_call_arguments` fragments, and usage.
//! The model catalog comes from `GET /v1/models`, the same shape the Chat
//! Completions adapter reads.

use std::collections::{BTreeMap, BTreeSet};
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

/// OpenAI-compatible Responses provider.
#[derive(Clone)]
pub struct OpenAiResponsesProvider {
    http: reqwest::Client,
}

impl std::fmt::Debug for OpenAiResponsesProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("OpenAiResponsesProvider")
    }
}

impl Default for OpenAiResponsesProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiResponsesProvider {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
        }
    }

    pub fn with_client(http: reqwest::Client) -> Self {
        Self { http }
    }
}

impl OpenAiResponsesProvider {
    fn endpoint(&self, config: &ProviderConfig, path: &str) -> Result<String, ProviderError> {
        let base = validate_base_url(&config.base_url, config.allow_private_targets)?;
        Ok(format!("{base}{path}"))
    }

    fn auth(
        &self,
        request: reqwest::RequestBuilder,
        config: &ProviderConfig,
    ) -> reqwest::RequestBuilder {
        config.apply_headers(request.bearer_auth(config.api_key.expose()))
    }
}

#[async_trait::async_trait]
impl Provider for OpenAiResponsesProvider {
    fn id(&self) -> &'static str {
        "openai_responses"
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
        let url = self.endpoint(config, "/v1/responses")?;
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
            match extract_outcome(&value) {
                // A 2xx whose status is not `completed` is a prefix of a
                // document, not an answer. Refusing it here keeps the caller
                // from parsing half a document as a whole one.
                Err(error) => return Err(error),
                Ok(outcome) if !outcome.is_empty() => return Ok(outcome),
                Ok(_) => {}
            }
            if has_reasoning(&request) {
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
                    match extract_outcome(&value) {
                        Ok(outcome) if !outcome.is_empty() => return Ok(outcome),
                        Err(error) => return Err(error),
                        Ok(_) => {}
                    }
                }
                return Err(ProviderError::InvalidPayload(
                    "upstream returned no response text".into(),
                ));
            }
            return Err(ProviderError::EmptyResponse);
        }

        if has_reasoning(&request) && ProviderError::is_retryable_reasoning_failure(status.as_u16())
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
                match extract_outcome(&value) {
                    Ok(outcome) if !outcome.is_empty() => return Ok(outcome),
                    Err(error) => return Err(error),
                    Ok(_) => {}
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
        let url = self.endpoint(config, "/v1/responses")?;
        let mut payload = build_payload(&request, true);
        let response = match self.start_stream(&url, config, &payload).await? {
            response if response.status().is_success() => response,
            response
                if has_reasoning(&request)
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
            Box::new(OpenAiResponsesStreamDecoder::default()),
        ))
    }
}

impl OpenAiResponsesProvider {
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

/// True when the request carries a reasoning override worth retrying without.
fn has_reasoning(request: &CompletionRequest) -> bool {
    clypeus_core::provider::reasoning_override(request.reasoning.as_deref()).is_some()
}

/// Builds the Responses payload.
pub fn build_payload(request: &CompletionRequest, stream: bool) -> Value {
    let mut payload = Map::new();
    payload.insert("model".into(), Value::String(request.model.clone()));
    let instructions = request
        .messages
        .iter()
        .filter(|message| message.role == ChatRole::System)
        .map(|message| message.content.as_str())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if !instructions.is_empty() {
        payload.insert("instructions".into(), Value::String(instructions));
    }
    payload.insert(
        "input".into(),
        Value::Array(request.messages.iter().flat_map(response_input).collect()),
    );
    if !request.tools.is_empty() {
        payload.insert(
            "tools".into(),
            Value::Array(request.tools.iter().map(response_tool).collect()),
        );
        payload.insert(
            "tool_choice".into(),
            Value::String(request.tool_choice.as_wire().to_string()),
        );
    }
    // Always strict: a schema a provider is free to ignore is a schema that turns a
    // caller's hard requirement into a suggestion, and the failure would surface far from
    // here as a parse error on an answer that looked successful.
    if let clypeus_core::provider::OutputFormat::JsonSchema { name, schema } = &request.output {
        payload.insert(
            "text".into(),
            json!({
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "strict": true,
                    "schema": schema,
                },
            }),
        );
    }
    payload.insert(
        "max_output_tokens".into(),
        Value::Number(serde_json::Number::from(request.max_output_tokens)),
    );
    if stream {
        payload.insert("stream".into(), Value::Bool(true));
    }
    if let Some(reasoning) =
        clypeus_core::provider::reasoning_override(request.reasoning.as_deref())
    {
        payload.insert("reasoning".into(), json!({"effort": reasoning}));
    }
    Value::Object(payload)
}

fn strip_reasoning(payload: &mut Value) {
    if let Some(object) = payload.as_object_mut() {
        object.remove("reasoning");
    }
}

/// Maps one chat message onto `input` items.
///
/// Tool results are `function_call_output` items and prior assistant tool
/// calls are `function_call` items, because those correlators (`call_id`) are
/// what the protocol matches to each other. System messages never appear here;
/// they were folded into `instructions`.
fn response_input(message: &ChatMessage) -> Vec<Value> {
    match message.role {
        ChatRole::System => Vec::new(),
        ChatRole::Tool => vec![json!({
            "type": "function_call_output",
            "call_id": message.tool_call_id.clone().unwrap_or_default(),
            "output": message.content,
        })],
        ChatRole::Assistant
            if message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty()) =>
        {
            let calls = message
                .tool_calls
                .as_ref()
                .expect("tool calls checked above");
            let mut items = Vec::new();
            if !message.content.is_empty() {
                items.push(json!({"role": "assistant", "content": message.content}));
            }
            for call in calls {
                items.push(json!({
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments.to_string(),
                }));
            }
            items
        }
        _ => vec![json!({"role": message.role, "content": message.content})],
    }
}

fn response_tool(tool: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
    })
}

/// Parses one buffered Responses payload.
///
/// The answer text sits in a `message` item that the API places after any
/// `reasoning` items, so output items are read by kind, never by position. A
/// response whose `status` is not `completed` is refused: its text is a prefix
/// of a document, and returning it would present the prefix as the answer.
pub fn extract_outcome(value: &Value) -> Result<AssistantOutcome, ProviderError> {
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if status != "completed" {
        return Err(ProviderError::InvalidPayload(non_completed_detail(
            value, status,
        )));
    }
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);
    let usage = value.get("usage").and_then(parse_usage);
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    for item in value
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => append_message_text(item, &mut content),
            Some("reasoning") => append_reasoning_text(item, &mut reasoning),
            Some("function_call") => {
                if let Some(call) = parse_function_call(item) {
                    tool_calls.push(call);
                }
            }
            _ => {}
        }
    }
    Ok(AssistantOutcome {
        content,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
        model,
        usage,
        tool_calls,
    })
}

/// Appends every text part of a `message` output item.
fn append_message_text(item: &Value, target: &mut String) {
    for part in item
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            target.push_str(text);
        } else if let Some(refusal) = part.get("refusal").and_then(Value::as_str) {
            target.push_str(refusal);
        }
    }
}

/// Appends every text part of a `reasoning` output item: `summary` for the
/// summarized form and `content` for the raw one.
fn append_reasoning_text(item: &Value, target: &mut String) {
    for key in ["summary", "content"] {
        for part in item
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                target.push_str(text);
            }
        }
    }
}

fn parse_function_call(item: &Value) -> Option<ToolCall> {
    let name = item.get("name").and_then(Value::as_str)?;
    let id = item
        .get("call_id")
        .and_then(Value::as_str)
        .or_else(|| item.get("id").and_then(Value::as_str))?;
    let arguments = match item.get("arguments") {
        Some(Value::String(text)) if text.trim().is_empty() => Value::Object(Map::new()),
        Some(Value::String(text)) => {
            serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone()))
        }
        Some(value) => value.clone(),
        None => Value::Object(Map::new()),
    };
    Some(ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments,
    })
}

/// Explains why a response that is not `completed` cannot be an answer.
fn non_completed_detail(value: &Value, status: &str) -> String {
    let status = if status.is_empty() { "unknown" } else { status };
    let reason = value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .pointer("/incomplete_details/reason")
                .and_then(Value::as_str)
        });
    match reason {
        Some(reason) => format!("upstream response status is '{status}': {reason}"),
        None => format!("upstream response status is '{status}'"),
    }
}

/// Explains one terminal stream failure event.
fn event_error_detail(value: &Value) -> String {
    if let Some(message) = value
        .pointer("/response/error/message")
        .and_then(Value::as_str)
    {
        return message.to_string();
    }
    if let Some(reason) = value
        .pointer("/response/incomplete_details/reason")
        .and_then(Value::as_str)
    {
        return format!("response incomplete: {reason}");
    }
    if let Some(message) = value.get("message").and_then(Value::as_str) {
        return message.to_string();
    }
    "provider stream error".to_string()
}

fn parse_usage(value: &Value) -> Option<TokenUsage> {
    let usage = TokenUsage {
        prompt_tokens: value
            .get("input_tokens")
            .and_then(Value::as_i64)
            .and_then(|value| value.try_into().ok()),
        completion_tokens: value
            .get("output_tokens")
            .and_then(Value::as_i64)
            .and_then(|value| value.try_into().ok()),
        total_tokens: value
            .get("total_tokens")
            .and_then(Value::as_i64)
            .and_then(|value| value.try_into().ok()),
        cached_tokens: value
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(Value::as_i64)
            .and_then(|value| value.try_into().ok()),
        reasoning_tokens: value
            .pointer("/output_tokens_details/reasoning_tokens")
            .and_then(Value::as_i64)
            .and_then(|value| value.try_into().ok()),
    };
    (!usage.is_empty()).then_some(usage)
}

/// `output_index` of an SSE event, or zero when the event omits it.
fn output_index(value: &Value) -> u64 {
    value
        .get("output_index")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Incremental decoder for Responses SSE payloads.
#[derive(Default)]
pub struct OpenAiResponsesStreamDecoder {
    content: String,
    reasoning: String,
    usage: Option<TokenUsage>,
    model: Option<String>,
    tool_calls: BTreeMap<u64, PartialToolCall>,
    /// Output indices whose text arrived as deltas, so a later `done` event
    /// carrying the full text does not append it a second time.
    streamed_text: BTreeSet<u64>,
    streamed_reasoning: BTreeSet<u64>,
    streamed_arguments: BTreeSet<u64>,
}

impl StreamDecoder for OpenAiResponsesStreamDecoder {
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
        let mut round = RoundDelta::default();
        match value.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                    self.content.push_str(delta);
                    round.content = Some(delta.to_string());
                    self.streamed_text.insert(output_index(&value));
                }
            }
            Some("response.output_text.done") => {
                let index = output_index(&value);
                if !self.streamed_text.contains(&index)
                    && let Some(text) = value.get("text").and_then(Value::as_str)
                {
                    self.content.push_str(text);
                    round.content = Some(text.to_string());
                }
            }
            Some("response.content_part.done") => {
                let index = output_index(&value);
                if !self.streamed_text.contains(&index)
                    && let Some(text) = value.pointer("/part/text").and_then(Value::as_str)
                {
                    self.content.push_str(text);
                    round.content = Some(text.to_string());
                }
            }
            Some("response.reasoning_summary_text.delta")
            | Some("response.reasoning_text.delta") => {
                if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                    self.reasoning.push_str(delta);
                    round.reasoning = Some(delta.to_string());
                    self.streamed_reasoning.insert(output_index(&value));
                }
            }
            Some("response.reasoning_summary_text.done") | Some("response.reasoning_text.done") => {
                let index = output_index(&value);
                if !self.streamed_reasoning.contains(&index)
                    && let Some(text) = value.get("text").and_then(Value::as_str)
                {
                    self.reasoning.push_str(text);
                    round.reasoning = Some(text.to_string());
                }
            }
            Some("response.output_item.added") => {
                if let Some(item) = value.get("item")
                    && item.get("type").and_then(Value::as_str) == Some("function_call")
                {
                    self.open_tool_call(item, output_index(&value));
                }
            }
            Some("response.function_call_arguments.delta") => {
                if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                    let index = output_index(&value);
                    self.tool_calls
                        .entry(index)
                        .or_default()
                        .arguments
                        .push_str(delta);
                    self.streamed_arguments.insert(index);
                }
            }
            Some("response.function_call_arguments.done") => {
                let index = output_index(&value);
                if !self.streamed_arguments.contains(&index)
                    && let Some(arguments) = value.get("arguments").and_then(Value::as_str)
                {
                    self.tool_calls
                        .entry(index)
                        .or_default()
                        .arguments
                        .push_str(arguments);
                }
            }
            Some("response.output_item.done") => {
                if let Some(item) = value.get("item") {
                    let index = output_index(&value);
                    match item.get("type").and_then(Value::as_str) {
                        Some("message") => {
                            if !self.streamed_text.contains(&index) {
                                let mut text = String::new();
                                append_message_text(item, &mut text);
                                if !text.is_empty() {
                                    self.content.push_str(&text);
                                    round.content = Some(text);
                                }
                            }
                        }
                        Some("reasoning") => {
                            if !self.streamed_reasoning.contains(&index) {
                                let mut text = String::new();
                                append_reasoning_text(item, &mut text);
                                if !text.is_empty() {
                                    self.reasoning.push_str(&text);
                                    round.reasoning = Some(text);
                                }
                            }
                        }
                        Some("function_call") => {
                            self.open_tool_call(item, index);
                            if !self.streamed_arguments.contains(&index)
                                && let Some(arguments) =
                                    item.get("arguments").and_then(Value::as_str)
                            {
                                self.tool_calls
                                    .entry(index)
                                    .or_default()
                                    .arguments
                                    .push_str(arguments);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some("response.completed") => {
                let response = value.get("response").unwrap_or(&Value::Null);
                if let Some(status) = response.get("status").and_then(Value::as_str)
                    && status != "completed"
                {
                    return PayloadEvent::Error(non_completed_detail(response, status));
                }
                if let Some(usage) = response.get("usage").and_then(parse_usage) {
                    self.merge_usage(usage.clone());
                    round.usage = Some(usage);
                }
                if let Some(model) = response.get("model").and_then(Value::as_str) {
                    self.model = Some(model.to_string());
                }
                return PayloadEvent::Done;
            }
            Some("response.failed") | Some("response.incomplete") => {
                return PayloadEvent::Error(event_error_detail(&value));
            }
            Some("response.error") => {
                let detail = value
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("provider stream error");
                return PayloadEvent::Error(detail.to_string());
            }
            _ => {}
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
            model: self.model.clone(),
            usage: self.usage.clone(),
            tool_calls,
        }
    }
}

impl OpenAiResponsesStreamDecoder {
    /// Records the identity of a `function_call` item. The `call_id` is the
    /// correlator a later `function_call_output` must carry, so it becomes the
    /// tool call id; the item id only fills in when a gateway omits it.
    fn open_tool_call(&mut self, item: &Value, index: u64) {
        let entry = self.tool_calls.entry(index).or_default();
        let id = item
            .get("call_id")
            .and_then(Value::as_str)
            .or_else(|| item.get("id").and_then(Value::as_str))
            .filter(|id| !id.is_empty());
        if let Some(id) = id {
            entry.id = id.to_string();
        }
        if let Some(name) = item
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        {
            entry.name = name.to_string();
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

impl std::fmt::Debug for OpenAiResponsesStreamDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiResponsesStreamDecoder")
            .field("content_len", &self.content.len())
            .field("tool_calls", &self.tool_calls.len())
            .finish()
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
            let trimmed = value.trim();
            if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("null") {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn extract_reasoning_levels(item: &Value) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
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
                    let trimmed = level.trim();
                    if !trimmed.is_empty()
                        && !found.iter().any(|existing| existing.as_str() == trimmed)
                    {
                        found.push(trimmed.to_string());
                    }
                }
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use clypeus_core::provider::{DEFAULT_REASONING_LEVEL, OutputFormat, ToolChoice};

    fn request(output: OutputFormat, reasoning: Option<&str>) -> CompletionRequest {
        CompletionRequest {
            model: "ubi-model".into(),
            messages: vec![ChatMessage::text(ChatRole::User, "hi")],
            reasoning: reasoning.map(str::to_string),
            tools: Vec::new(),
            tool_choice: ToolChoice::None,
            max_output_tokens: 128,
            output,
        }
    }

    fn schema() -> OutputFormat {
        OutputFormat::JsonSchema {
            name: "answer".into(),
            schema: json!({
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": false
            }),
        }
    }

    #[test]
    fn payload_sends_a_strict_text_format_for_a_schema() {
        let payload = build_payload(&request(schema(), None), false);
        let format = &payload["text"]["format"];
        assert_eq!(format["type"], "json_schema");
        assert_eq!(format["name"], "answer");
        assert_eq!(format["strict"], true);
        assert_eq!(format["schema"]["required"][0], "ok");
    }

    #[test]
    fn payload_without_a_schema_carries_no_text_format() {
        let payload = build_payload(&request(OutputFormat::Text, None), false);
        assert!(
            payload.get("text").is_none(),
            "free-text requests must not carry a format"
        );
    }

    #[test]
    fn payload_maps_system_to_instructions_and_tools_to_input_items() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "echo".into(),
            arguments: json!({"message": "ping"}),
        };
        let mut request = request(OutputFormat::Text, None);
        request.messages = vec![
            ChatMessage::text(ChatRole::System, "be brief"),
            ChatMessage::text(ChatRole::User, "hi"),
            ChatMessage::assistant_tool_calls("", vec![call]),
            ChatMessage::tool_result("call_1", "pong"),
        ];
        let payload = build_payload(&request, false);
        assert_eq!(payload["instructions"], "be brief");
        let input = payload["input"].as_array().expect("input array");
        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[1]["name"], "echo");
        assert_eq!(input[1]["arguments"], "{\"message\":\"ping\"}");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[2]["output"], "pong");
    }

    #[test]
    fn payload_sends_tools_and_tool_choice_only_when_tools_exist() {
        let mut with_tools = request(OutputFormat::Text, None);
        with_tools.tools = vec![ToolSpec {
            name: "echo".into(),
            description: "echoes".into(),
            input_schema: json!({"type": "object"}),
        }];
        with_tools.tool_choice = ToolChoice::Required;
        let payload = build_payload(&with_tools, false);
        assert_eq!(payload["tools"][0]["type"], "function");
        assert_eq!(payload["tools"][0]["name"], "echo");
        assert_eq!(payload["tools"][0]["parameters"]["type"], "object");
        assert_eq!(payload["tool_choice"], "required");

        let payload = build_payload(&request(OutputFormat::Text, None), false);
        assert!(payload.get("tools").is_none());
        assert!(payload.get("tool_choice").is_none());
    }

    #[test]
    fn payload_sends_reasoning_effort_verbatim() {
        for level in ["minimal", "none", "max", "xhigh", "custom-id"] {
            let payload = build_payload(&request(OutputFormat::Text, Some(level)), false);
            assert_eq!(payload["reasoning"]["effort"], level, "level {level}");
        }
    }

    #[test]
    fn payload_omits_sentinel_and_empty_reasoning() {
        for level in [
            None,
            Some(""),
            Some("   "),
            Some("default"),
            Some(" default "),
        ] {
            let payload = build_payload(&request(OutputFormat::Text, level), false);
            assert!(
                payload.get("reasoning").is_none(),
                "reasoning must be absent for {level:?}"
            );
        }
        assert_eq!(
            build_payload(&request(OutputFormat::Text, None), false)["max_output_tokens"],
            128
        );
        assert_eq!(
            build_payload(&request(OutputFormat::Text, None), true)["stream"],
            true
        );
    }

    #[test]
    fn buffered_outcome_reads_the_message_after_reasoning() {
        let value = json!({
            "status": "completed",
            "model": "ubi-model",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "thought first"}]
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "{\"ok\": true}"}]
                }
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 5}
            }
        });
        let outcome = extract_outcome(&value).expect("completed response parses");
        assert_eq!(outcome.content, "{\"ok\": true}");
        assert_eq!(outcome.reasoning.as_deref(), Some("thought first"));
        assert_eq!(outcome.model.as_deref(), Some("ubi-model"));
        let usage = outcome.usage.expect("usage");
        assert_eq!(usage.prompt_tokens, Some(11));
        assert_eq!(usage.completion_tokens, Some(7));
        assert_eq!(usage.total_tokens, Some(18));
        assert_eq!(usage.cached_tokens, Some(3));
        assert_eq!(usage.reasoning_tokens, Some(5));
    }

    #[test]
    fn buffered_outcome_refuses_a_non_completed_status() {
        let incomplete = json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "model": "ubi-model",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "{\"ok\""}]
            }]
        });
        let error = extract_outcome(&incomplete).unwrap_err();
        assert_eq!(error.code(), "provider_invalid_response");
        let message = format!("{error}");
        assert!(message.contains("incomplete"), "message: {message}");
        assert!(message.contains("max_output_tokens"), "message: {message}");

        let failed = json!({
            "status": "failed",
            "error": {"code": "server_error", "message": "model overloaded"}
        });
        let error = extract_outcome(&failed).unwrap_err();
        assert!(format!("{error}").contains("model overloaded"));

        let missing = json!({"output": []});
        assert!(extract_outcome(&missing).is_err());
    }

    #[test]
    fn buffered_outcome_parses_function_calls_by_call_id() {
        let value = json!({
            "status": "completed",
            "model": "ubi-model",
            "output": [{
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "{\"message\":\"ping\"}"
            }]
        });
        let outcome = extract_outcome(&value).expect("parses");
        assert_eq!(outcome.content, "");
        assert_eq!(outcome.tool_calls.len(), 1);
        assert_eq!(outcome.tool_calls[0].id, "call_1");
        assert_eq!(outcome.tool_calls[0].name, "echo");
        assert_eq!(outcome.tool_calls[0].arguments, json!({"message": "ping"}));
    }

    const RECORDED_STREAM: &[&str] = &[
        r#"{"type":"response.created","response":{"status":"in_progress","model":"ubi-model"}}"#,
        r#"{"type":"response.in_progress","response":{"status":"in_progress"}}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}"#,
        r#"{"type":"response.output_item.added","output_index":1,"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}"#,
        r#"{"type":"response.content_part.added","output_index":1,"content_index":0,"part":{"type":"output_text","text":""}}"#,
        r#"{"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"{\"ok\": true"}"#,
        r#"{"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"}"}"#,
        r#"{"type":"response.content_part.done","output_index":1,"content_index":0,"part":{"type":"output_text","text":"{\"ok\": true}"}}"#,
        r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"{\"ok\": true}"}]}}"#,
        r#"{"type":"response.completed","response":{"status":"completed","model":"ubi-model","usage":{"input_tokens":4,"output_tokens":6,"total_tokens":10,"output_tokens_details":{"reasoning_tokens":2}}}}"#,
    ];

    #[test]
    fn stream_decoder_replays_a_recorded_response() {
        let mut decoder = OpenAiResponsesStreamDecoder::default();
        let mut deltas = Vec::new();
        for payload in RECORDED_STREAM {
            match decoder.consume(payload) {
                PayloadEvent::Data(round) => {
                    if let Some(content) = round.content {
                        deltas.push(content);
                    }
                    if let Some(usage) = round.usage {
                        assert_eq!(usage.total_tokens, Some(10));
                    }
                }
                PayloadEvent::Done => break,
                PayloadEvent::Error(detail) => panic!("unexpected stream error: {detail}"),
            }
        }
        assert_eq!(deltas.concat(), "{\"ok\": true}");
        let outcome = decoder.outcome();
        assert_eq!(outcome.content, "{\"ok\": true}");
        assert_eq!(outcome.model.as_deref(), Some("ubi-model"));
        assert_eq!(outcome.usage.and_then(|usage| usage.total_tokens), Some(10));
    }

    #[test]
    fn stream_decoder_refuses_failed_and_incomplete_events() {
        let mut decoder = OpenAiResponsesStreamDecoder::default();
        let failed = r#"{"type":"response.failed","response":{"status":"failed","error":{"message":"overloaded"}}}"#;
        assert_eq!(
            decoder.consume(failed),
            PayloadEvent::Error("overloaded".into())
        );
        let incomplete = r#"{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}"#;
        assert_eq!(
            decoder.consume(incomplete),
            PayloadEvent::Error("response incomplete: max_output_tokens".into())
        );
        let error = r#"{"type":"response.error","message":"bad request"}"#;
        assert_eq!(
            decoder.consume(error),
            PayloadEvent::Error("bad request".into())
        );
    }

    /// The tool-call event names and the `function_call` item shape come from
    /// the OpenAI Responses specification; none of them could be observed
    /// against the live service because no tools were offered there.
    #[test]
    fn stream_decoder_accumulates_function_call_arguments() {
        let events = [
            r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"echo","arguments":""}}"#,
            r#"{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"{\"message\":"}"#,
            r#"{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"\"ping\"}"}"#,
            r#"{"type":"response.function_call_arguments.done","output_index":0,"item_id":"fc_1","arguments":"{\"message\":\"ping\"}"}"#,
            r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"echo","arguments":"{\"message\":\"ping\"}"}}"#,
            r#"{"type":"response.completed","response":{"status":"completed","model":"ubi-model","usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}"#,
        ];
        let mut decoder = OpenAiResponsesStreamDecoder::default();
        for (position, payload) in events.iter().enumerate() {
            let event = decoder.consume(payload);
            if position + 1 == events.len() {
                assert_eq!(event, PayloadEvent::Done);
            } else {
                assert!(
                    matches!(event, PayloadEvent::Data(_)),
                    "unexpected event: {event:?}"
                );
            }
        }
        let outcome = decoder.outcome();
        assert_eq!(outcome.tool_calls.len(), 1);
        assert_eq!(outcome.tool_calls[0].id, "call_1");
        assert_eq!(outcome.tool_calls[0].name, "echo");
        assert_eq!(outcome.tool_calls[0].arguments, json!({"message": "ping"}));
        assert_eq!(outcome.usage.and_then(|usage| usage.total_tokens), Some(3));
    }

    #[test]
    fn stream_decoder_falls_back_to_done_text_without_deltas() {
        let mut decoder = OpenAiResponsesStreamDecoder::default();
        let part = r#"{"type":"response.content_part.done","output_index":1,"content_index":0,"part":{"type":"output_text","text":"whole answer"}}"#;
        let event = decoder.consume(part);
        assert_eq!(
            event,
            PayloadEvent::Data(RoundDelta {
                content: Some("whole answer".into()),
                ..RoundDelta::default()
            })
        );
        assert_eq!(decoder.outcome().content, "whole answer");
    }

    #[test]
    fn catalog_parses_reasoning_levels() {
        let root = json!({
            "data": [{
                "id": "ubi-model",
                "reasoning": {
                    "supported": true,
                    "levels": ["default", "none", "max"],
                    "default": "max"
                }
            }]
        });
        let catalog = parse_model_catalog(&root);
        assert_eq!(catalog.models.len(), 1);
        assert_eq!(
            catalog.models[0].reasoning_levels,
            vec!["default", "none", "max"]
        );
        assert_eq!(catalog.models[0].default_reasoning_level, "max");
    }

    #[test]
    fn retry_without_reasoning_ignores_the_sentinel() {
        assert!(has_reasoning(&request(OutputFormat::Text, Some("max"))));
        assert!(!has_reasoning(&request(
            OutputFormat::Text,
            Some(DEFAULT_REASONING_LEVEL)
        )));
        assert!(!has_reasoning(&request(OutputFormat::Text, None)));
    }

    #[derive(Default)]
    struct HeaderCapture {
        requests: std::sync::Mutex<Vec<(String, std::collections::BTreeMap<String, String>)>>,
    }

    async fn capture_request(
        axum::extract::State(capture): axum::extract::State<std::sync::Arc<HeaderCapture>>,
        axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
        headers: axum::http::HeaderMap,
    ) -> axum::Json<Value> {
        let mut captured = std::collections::BTreeMap::new();
        for (name, value) in &headers {
            captured.insert(
                name.as_str().to_string(),
                value.to_str().unwrap_or_default().to_string(),
            );
        }
        capture
            .requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((uri.path().to_string(), captured));
        axum::Json(json!({
            "data": [{"id": "captured", "reasoning": {"levels": ["low"]}}],
            "status": "completed",
            "model": "captured",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "captured answer"}]
            }]
        }))
    }

    async fn spawn_capture() -> (String, std::sync::Arc<HeaderCapture>) {
        let capture = std::sync::Arc::new(HeaderCapture::default());
        let router = axum::Router::new()
            .route("/v1/models", axum::routing::get(capture_request))
            .route("/v1/responses", axum::routing::post(capture_request))
            .with_state(std::sync::Arc::clone(&capture));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("mock provider binds");
        let address = listener.local_addr().expect("mock address");
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        (format!("http://{address}"), capture)
    }

    #[tokio::test]
    async fn configured_headers_reach_the_catalog_and_the_completion() {
        let (base_url, capture) = spawn_capture().await;
        let config = ProviderConfig::new(base_url, "test-key")
            .allow_private_targets(true)
            .with_header("x-session-id", "stable-session")
            .with_header("x-static", "fixed");
        let provider = OpenAiResponsesProvider::new();
        provider.catalog(&config).await.expect("catalog");
        provider
            .complete(&config, request(OutputFormat::Text, None))
            .await
            .expect("completion");

        let requests = capture
            .requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            requests.len(),
            2,
            "catalog and completion must both be sent"
        );
        assert_eq!(requests[0].0, "/v1/models");
        assert_eq!(requests[1].0, "/v1/responses");
        for (path, headers) in requests.iter() {
            assert_eq!(
                headers.get("x-session-id").map(String::as_str),
                Some("stable-session"),
                "session header missing on {path}"
            );
            assert_eq!(
                headers.get("x-static").map(String::as_str),
                Some("fixed"),
                "static header missing on {path}"
            );
            assert_eq!(
                headers.get("authorization").map(String::as_str),
                Some("Bearer test-key"),
                "bearer auth missing on {path}"
            );
        }
    }

    #[test]
    fn provider_kind_spelling_matches_the_registry_key() {
        assert_eq!(
            OpenAiResponsesProvider::new().id(),
            clypeus_core::models::ProviderKind::OpenaiResponses.as_wire()
        );
    }
}
