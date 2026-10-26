//! Open Responses provider (openresponses.org).
//!
//! Speaks the open standard rather than any company's dialect of it. The
//! request goes to `POST {base}/v1/responses` with typed input items and
//! `store: false`; the stream is a sequence of semantic events that ends with
//! a terminal response event (`response.completed`, `response.failed` or
//! `response.incomplete`) and the spec's `[DONE]` sentinel is accepted but
//! never required. Errors are the standard's typed envelope, so a refusal
//! carries a machine-readable `type` beside the human-readable `message`.
//!
//! The standard is built on the OpenAI Responses API's shape, so servers
//! behind this provider usually accept that dialect too. What this provider
//! does not do is assume it: an item without a `type`, a stream that ends
//! without a terminal event and a data payload that is not JSON are protocol
//! violations here, not something to guess around.
//!
//! The model catalog comes from `GET /v1/models`. The standard's core defines
//! no catalog route; every server implementing it exposes the OpenAI-compatible
//! list, and the capability metadata the standard does not define (reasoning
//! levels in particular) is deliberately not guessed at — a caller that asks
//! for a reasoning level the model does not have gets the provider's
//! retry-without-reasoning path, which is the honest answer either way.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use clypeus_core::models::{ChatMessage, ChatRole, TokenUsage, ToolCall, ToolSpec};
use clypeus_core::provider::{
    AssistantOutcome, CompletionRequest, ModelCapability, ModelCatalog, PayloadEvent, ProbeReport,
    Provider, ProviderConfig, ProviderError, ProviderStream, RoundDelta, StreamDecoder,
    extract_error_message, response_error, retry_after_from_headers, validate_base_url,
};
use futures_util::StreamExt;
use serde_json::{Map, Value, json};

const MODEL_CATALOG_TIMEOUT: Duration = Duration::from_secs(7);
const PROBE_TIMEOUT: Duration = Duration::from_secs(7);
/// Upper bound for the provider to answer and start streaming.
const UPSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Open Responses provider.
#[derive(Clone)]
pub struct OpenResponsesProvider {
    http: reqwest::Client,
}

impl std::fmt::Debug for OpenResponsesProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("OpenResponsesProvider")
    }
}

impl Default for OpenResponsesProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenResponsesProvider {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
        }
    }

    pub fn with_client(http: reqwest::Client) -> Self {
        Self { http }
    }

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
impl Provider for OpenResponsesProvider {
    fn id(&self) -> &'static str {
        "open_responses"
    }

    async fn catalog(&self, config: &ProviderConfig) -> Result<ModelCatalog, ProviderError> {
        let url = self.endpoint(config, "/v1/models")?;
        let response = self
            .auth(self.http.get(&url).timeout(MODEL_CATALOG_TIMEOUT), config)
            .header("accept", "application/json")
            .send()
            .await
            .map_err(ProviderError::transport_error)?;
        let status = response.status();
        // Headers are read before the body: consuming the body moves the
        // response, and the wait a rate limit states is in a header.
        let retry_after = retry_after_from_headers(response.headers());
        let body = response
            .text()
            .await
            .map_err(ProviderError::transport_error)?;
        if !status.is_success() {
            return Err(response_error(
                status.as_u16(),
                retry_after,
                &body,
                None,
                status.canonical_reason().unwrap_or("error"),
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
                .map_err(ProviderError::transport_error)?;
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
            .map_err(ProviderError::transport_error)?;
        let status = response.status();
        let retry_after = retry_after_from_headers(response.headers());
        let body = response
            .text()
            .await
            .map_err(ProviderError::transport_error)?;

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
                return self
                    .retry_without_reasoning(config, &url, &mut payload, &request)
                    .await;
            }
            return Err(ProviderError::EmptyResponse);
        }

        if has_reasoning(&request) && ProviderError::is_retryable_reasoning_failure(status.as_u16())
        {
            return self
                .retry_without_reasoning(config, &url, &mut payload, &request)
                .await;
        }

        Err(response_error(
            status.as_u16(),
            retry_after,
            &body,
            Some(&request.model),
            status.canonical_reason().unwrap_or("upstream error"),
        ))
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
                    let retry_hint = retry_after_from_headers(retry.headers());
                    let body = retry.text().await.unwrap_or_default();
                    return Err(response_error(
                        status,
                        retry_hint,
                        &body,
                        Some(&request.model),
                        "upstream stream error",
                    ));
                }
                retry
            }
            response => {
                let status = response.status().as_u16();
                let retry_hint = retry_after_from_headers(response.headers());
                let body = response.text().await.unwrap_or_default();
                return Err(response_error(
                    status,
                    retry_hint,
                    &body,
                    Some(&request.model),
                    "upstream stream error",
                ));
            }
        };
        let stream = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(std::io::Error::other))
            .boxed();
        Ok(ProviderStream::new(
            stream,
            Box::new(OpenResponsesStreamDecoder::default()),
        ))
    }
}

impl OpenResponsesProvider {
    /// Repeats the request without the reasoning override.
    ///
    /// A reasoning level is an extension the standard does not define, so a
    /// conforming server may reject or ignore it. Asking once more without it
    /// is the difference between "the model has no levels" and a failed
    /// generation.
    async fn retry_without_reasoning(
        &self,
        config: &ProviderConfig,
        url: &str,
        payload: &mut Value,
        request: &CompletionRequest,
    ) -> Result<AssistantOutcome, ProviderError> {
        strip_reasoning(payload);
        let retry = self
            .auth(self.http.post(url), config)
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .json(payload)
            .timeout(config.timeout)
            .send()
            .await
            .map_err(ProviderError::transport_error)?;
        let retry_status = retry.status();
        let retry_hint = retry_after_from_headers(retry.headers());
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
        Err(response_error(
            retry_status.as_u16(),
            retry_hint,
            &retry_body,
            Some(&request.model),
            "upstream error on retry",
        ))
    }

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
        .map_err(ProviderError::transport_error)
    }
}

/// True when the request carries a reasoning override worth retrying without.
fn has_reasoning(request: &CompletionRequest) -> bool {
    clypeus_core::provider::reasoning_override(request.reasoning.as_deref()).is_some()
}

/// Builds the Open Responses request body.
///
/// Input items are typed the way the standard requires: a `message` item with
/// a discriminated content part, never a bare string. `store: false` states
/// that the server keeps no response state, which is what a gateway pass-through
/// wants: nothing to retain, nothing to leak, nothing to garbage-collect.
pub fn build_payload(request: &CompletionRequest, stream: bool) -> Value {
    let mut payload = Map::new();
    payload.insert("model".into(), Value::String(request.model.clone()));
    payload.insert("store".into(), Value::Bool(false));
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
        Value::Array(request.messages.iter().flat_map(input_items).collect()),
    );
    if !request.tools.is_empty() {
        payload.insert(
            "tools".into(),
            Value::Array(request.tools.iter().map(function_tool).collect()),
        );
        payload.insert(
            "tool_choice".into(),
            Value::String(request.tool_choice.as_wire().to_string()),
        );
    }
    // The strict JSON-document shape is an extension the standard's base API
    // defines. A conforming server that does not implement it ignores the
    // field; one that does enforces the schema, and a schema merely suggested
    // would turn a hard requirement into a coin toss.
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
    // The reasoning effort is the same non-standard extension; sent verbatim
    // when asked for, absent otherwise.
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

/// Maps one chat message onto typed `input` items.
///
/// Tool results are `function_call_output` items and prior assistant tool
/// calls are `function_call` items, because those correlators (`call_id`) are
/// what the standard matches to each other. System messages never appear here;
/// they were folded into `instructions`.
fn input_items(message: &ChatMessage) -> Vec<Value> {
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
                items.push(assistant_message(&message.content));
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
        ChatRole::Assistant => vec![assistant_message(&message.content)],
        ChatRole::User => vec![json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": message.content}],
        })],
    }
}

/// An assistant message item, round-tripped with the content kind a model
/// output originally had.
fn assistant_message(content: &str) -> Value {
    json!({
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": content}],
    })
}

fn function_tool(tool: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
    })
}

/// Parses one buffered Open Responses payload.
///
/// The answer text sits in a `message` item that the API places after any
/// `reasoning` items, so output items are read by kind, never by position. A
/// response whose `status` is not `completed` is refused: `incomplete` means
/// the token budget ran out and `failed` means the turn died, and either way
/// the text is a prefix of a document rather than an answer.
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
            // Unknown and implementor-prefixed item types are the standard's
            // extension surface; a client that is not the implementor ignores
            // them and still reconstructs the canonical response.
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

/// Explains one terminal failure: a `response.failed` event, a truncated
/// `response.incomplete` one, or the standard's standalone `error` envelope.
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
    if let Some(message) = value.pointer("/error/message").and_then(Value::as_str) {
        return message.to_string();
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

/// The catalog every Open Responses server exposes: the OpenAI-compatible
/// `GET /v1/models` list, read by id. The standard defines no capability
/// metadata, so none is invented here.
fn parse_model_catalog(root: &Value) -> ModelCatalog {
    let models = root
        .get("data")
        .or_else(|| root.get("models"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let model = item.get("id").and_then(Value::as_str)?.trim().to_string();
            (!model.is_empty()).then_some(ModelCapability {
                model,
                reasoning_levels: Vec::new(),
                default_reasoning_level: String::new(),
            })
        })
        .collect();
    ModelCatalog { models }
}

/// `output_index` of an SSE event, or zero when the event omits it.
fn output_index(value: &Value) -> u64 {
    value
        .get("output_index")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

/// `content_index` of an SSE event, or zero when the event omits it.
fn content_index(value: &Value) -> u64 {
    value
        .get("content_index")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Incremental decoder for Open Responses SSE payloads.
#[derive(Default)]
pub struct OpenResponsesStreamDecoder {
    content: String,
    reasoning: String,
    usage: Option<TokenUsage>,
    model: Option<String>,
    tool_calls: BTreeMap<u64, PartialToolCall>,
    /// Content parts whose text arrived as deltas, keyed by the item and part
    /// the spec addresses them by, so a later `done` event carrying the full
    /// part does not append it a second time — and a sibling part of the same
    /// item, which streamed nothing, still lands.
    streamed_text: BTreeSet<(u64, u64)>,
    streamed_reasoning: BTreeSet<u64>,
    streamed_arguments: BTreeSet<u64>,
}

impl std::fmt::Debug for OpenResponsesStreamDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenResponsesStreamDecoder")
            .field("content_bytes", &self.content.len())
            .field("reasoning_bytes", &self.reasoning.len())
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl StreamDecoder for OpenResponsesStreamDecoder {
    fn consume(&mut self, payload: &str) -> PayloadEvent {
        if payload.is_empty() {
            // A keep-alive or an event-name-only block carries no data.
            return PayloadEvent::Data(RoundDelta::default());
        }
        // The spec allows the sentinel after the terminal event; a stream that
        // ends after the terminal event without it is equally valid, and the
        // core stops reading either way.
        if payload == "[DONE]" {
            return PayloadEvent::Done;
        }
        let value: Value = match serde_json::from_str(payload) {
            Ok(value) => value,
            // Data that is not JSON is a protocol violation, not text to
            // guess around: the standard requires JSON-encoded data objects.
            Err(error) => {
                return PayloadEvent::Error(format!(
                    "open responses stream carried a non-JSON event: {error}"
                ));
            }
        };
        let mut round = RoundDelta::default();
        match value.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                    self.content.push_str(delta);
                    round.content = Some(delta.to_string());
                    self.streamed_text
                        .insert((output_index(&value), content_index(&value)));
                }
            }
            Some("response.output_text.done") => {
                let key = (output_index(&value), content_index(&value));
                if !self.streamed_text.contains(&key)
                    && let Some(text) = value.get("text").and_then(Value::as_str)
                {
                    self.content.push_str(text);
                    round.content = Some(text.to_string());
                }
            }
            Some("response.content_part.done") => {
                let key = (output_index(&value), content_index(&value));
                if !self.streamed_text.contains(&key)
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
                    // The done event carried the whole argument text; a later
                    // item event repeating it must not append it again.
                    self.streamed_arguments.insert(index);
                }
            }
            Some("response.output_item.done") => {
                if let Some(item) = value.get("item") {
                    let index = output_index(&value);
                    match item.get("type").and_then(Value::as_str) {
                        Some("message") => {
                            let mut text = String::new();
                            for (part_index, part) in item
                                .get("content")
                                .and_then(Value::as_array)
                                .into_iter()
                                .flatten()
                                .enumerate()
                            {
                                let key = (index, part_index as u64);
                                if self.streamed_text.contains(&key) {
                                    continue;
                                }
                                if let Some(part_text) = part.get("text").and_then(Value::as_str) {
                                    text.push_str(part_text);
                                }
                            }
                            if !text.is_empty() {
                                self.content.push_str(&text);
                                round.content = Some(text);
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
            // The standard's terminal failures. `incomplete` means the token
            // budget ran out mid-item: the text collected so far is a prefix,
            // and a prefix presented as an answer is a wrong answer.
            Some("response.failed") | Some("response.incomplete") => {
                return PayloadEvent::Error(event_error_detail(&value));
            }
            // The standard's standalone error envelope; the spec sequences it
            // before the terminal `response.failed`, and the first error is
            // the one that names the cause.
            Some("error") => {
                return PayloadEvent::Error(event_error_detail(&value));
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

impl OpenResponsesStreamDecoder {
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

#[cfg(test)]
mod tests {
    use super::*;
    use clypeus_core::provider::{DEFAULT_REASONING_LEVEL, OutputFormat, ToolChoice};

    fn request(output: OutputFormat, reasoning: Option<&str>) -> CompletionRequest {
        CompletionRequest {
            model: "free-model".into(),
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
    fn payload_sends_typed_items_and_declares_no_retention() {
        let payload = build_payload(&request(OutputFormat::Text, None), false);
        assert_eq!(payload["model"], "free-model");
        assert_eq!(payload["store"], false);
        assert!(
            payload.get("stream").is_none(),
            "a buffered request must not ask for a stream"
        );
        let item = &payload["input"][0];
        assert_eq!(item["type"], "message");
        assert_eq!(item["role"], "user");
        assert_eq!(item["content"][0]["type"], "input_text");
        assert_eq!(item["content"][0]["text"], "hi");
    }

    #[test]
    fn payload_folds_system_messages_into_instructions() {
        let mut completion = request(OutputFormat::Text, None);
        completion
            .messages
            .insert(0, ChatMessage::text(ChatRole::System, "be brief"));
        let payload = build_payload(&completion, true);
        assert_eq!(payload["instructions"], "be brief");
        assert_eq!(payload["stream"], true);
        // System text must not also travel as an input item: the standard's
        // message items are user and assistant turns, and a duplicated system
        // prompt is a doubled instruction.
        assert_eq!(
            payload["input"]
                .as_array()
                .expect("input is an array")
                .len(),
            1
        );
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
    fn payload_sends_the_reasoning_override_only_when_asked() {
        let with_level = build_payload(&request(OutputFormat::Text, Some("high")), false);
        assert_eq!(with_level["reasoning"]["effort"], "high");
        let sentinel = build_payload(
            &request(OutputFormat::Text, Some(DEFAULT_REASONING_LEVEL)),
            false,
        );
        assert!(
            sentinel.get("reasoning").is_none(),
            "the default sentinel is no override"
        );
        let none = build_payload(&request(OutputFormat::Text, None), false);
        assert!(none.get("reasoning").is_none());
    }

    #[test]
    fn payload_correlates_tool_calls_and_outputs_by_call_id() {
        let mut completion = request(OutputFormat::Text, None);
        let call = ToolCall {
            id: "call_1".into(),
            name: "lookup".into(),
            arguments: json!({"city": "berlin"}),
        };
        completion.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: String::new(),
            tool_calls: Some(vec![call]),
            tool_call_id: None,
        });
        completion.messages.push(ChatMessage {
            role: ChatRole::Tool,
            content: "17C".into(),
            tool_calls: None,
            tool_call_id: Some("call_1".into()),
        });
        completion.tools = vec![ToolSpec {
            name: "lookup".into(),
            description: "weather".into(),
            input_schema: json!({"type": "object"}),
        }];
        completion.tool_choice = ToolChoice::Auto;
        let payload = build_payload(&completion, false);
        let input = payload["input"].as_array().expect("input is an array");
        // The caller's own user message is the first item; the tool exchange
        // follows it.
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[2]["output"], "17C");
        assert_eq!(payload["tools"][0]["type"], "function");
        assert_eq!(payload["tool_choice"], "auto");
    }

    #[test]
    fn an_assistant_message_round_trips_as_output_text() {
        let mut completion = request(OutputFormat::Text, None);
        completion
            .messages
            .push(ChatMessage::text(ChatRole::Assistant, "prior answer"));
        let payload = build_payload(&completion, false);
        let item = &payload["input"][1];
        assert_eq!(item["role"], "assistant");
        assert_eq!(item["content"][0]["type"], "output_text");
        assert_eq!(item["content"][0]["text"], "prior answer");
    }

    #[test]
    fn buffered_outcome_reads_items_by_kind() {
        let value = json!({
            "status": "completed",
            "model": "free-model",
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thought first"}]},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "{\"ok\""},
                    {"type": "output_text", "text": ": true}"}
                ]}
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
        assert_eq!(outcome.model.as_deref(), Some("free-model"));
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
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "{\"ok\""}]}]
        });
        let error = extract_outcome(&incomplete).unwrap_err();
        assert_eq!(error.code(), "provider_invalid_response");
        let message = format!("{error}");
        assert!(message.contains("incomplete"), "message: {message}");
        assert!(message.contains("max_output_tokens"), "message: {message}");

        // The standard's typed envelope lands in the explanation, so the
        // operator reads the server's own cause, not just a status word.
        let failed = json!({
            "status": "failed",
            "error": {"type": "server_error", "code": "overloaded", "message": "model overloaded"}
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

    #[test]
    fn unknown_item_types_are_ignored_not_fatal() {
        let value = json!({
            "status": "completed",
            "output": [
                {"type": "acme:trace_event", "id": "tc_1", "status": "completed"},
                {"type": "message", "content": [{"type": "output_text", "text": "kept"}]}
            ]
        });
        let outcome = extract_outcome(&value).expect("extension items do not break parsing");
        assert_eq!(outcome.content, "kept");
    }

    const RECORDED_STREAM: &[&str] = &[
        r#"{"type":"response.created","response":{"status":"in_progress","model":"free-model"}}"#,
        r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}}"#,
        r#"{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"plan: "}"#,
        r#"{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"answer shortly"}"#,
        r#"{"type":"response.reasoning_summary_text.done","output_index":0,"text":"plan: answer shortly"}"#,
        r#"{"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"plan: answer shortly"}]}}"#,
        r#"{"type":"response.output_item.added","output_index":1,"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}"#,
        r#"{"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"{\"ok\""}"#,
        r#"{"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":": true}"}"#,
        r#"{"type":"response.output_item.done","output_index":1,"item":{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"{\"ok\": true}"}]}}"#,
        r#"{"type":"response.completed","response":{"status":"completed","model":"free-model","usage":{"input_tokens":4,"output_tokens":6,"total_tokens":10,"output_tokens_details":{"reasoning_tokens":2}}}}"#,
    ];

    fn decoded(payloads: &[&str]) -> OpenResponsesStreamDecoder {
        let mut decoder = OpenResponsesStreamDecoder::default();
        for payload in payloads {
            decoder.consume(payload);
        }
        decoder
    }

    #[test]
    fn a_recorded_stream_assembles_the_answer_once() {
        let decoder = decoded(RECORDED_STREAM);
        let outcome = decoder.outcome();
        assert_eq!(outcome.content, "{\"ok\": true}");
        // The summary deltas streamed; the done and item events must not
        // repeat the same text.
        assert_eq!(outcome.reasoning.as_deref(), Some("plan: answer shortly"));
        assert_eq!(outcome.model.as_deref(), Some("free-model"));
        let usage = outcome
            .usage
            .expect("usage arrives with the terminal event");
        assert_eq!(usage.completion_tokens, Some(6));
        assert_eq!(usage.reasoning_tokens, Some(2));
    }

    #[test]
    fn the_terminal_event_ends_the_stream_and_the_sentinel_is_optional() {
        let mut decoder = OpenResponsesStreamDecoder::default();
        assert!(matches!(
            decoder.consume(RECORDED_STREAM.last().expect("terminal event")),
            PayloadEvent::Done
        ));
        // The spec allows `[DONE]` after the terminal event; it is accepted
        // as an end marker, not required to reach one.
        assert_eq!(decoder.consume("[DONE]"), PayloadEvent::Done);
    }

    #[test]
    fn sibling_parts_of_one_item_land_exactly_once() {
        // Part 0 streamed as deltas, part 1 arrived only with the item's done
        // event: the dedupe is per part, so part 1 is appended and part 0 is
        // not repeated.
        let decoder = decoded(&[
            r#"{"type":"response.output_text.delta","output_index":3,"content_index":0,"delta":"first "}"#,
            r#"{"type":"response.output_item.done","output_index":3,"item":{"type":"message","content":[{"type":"output_text","text":"first "},{"type":"output_text","text":"second"}]}}"#,
        ]);
        assert_eq!(decoder.outcome().content, "first second");
    }

    #[test]
    fn content_part_done_fills_a_part_that_streamed_no_deltas() {
        let decoder = decoded(&[
            r#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"a"}"#,
            r#"{"type":"response.content_part.done","output_index":0,"content_index":0,"part":{"type":"output_text","text":"a"}}"#,
            r#"{"type":"response.content_part.done","output_index":0,"content_index":1,"part":{"type":"output_text","text":"b"}}"#,
        ]);
        assert_eq!(decoder.outcome().content, "ab");
    }

    #[test]
    fn function_call_fragments_assemble_into_one_call() {
        let decoder = decoded(&[
            r#"{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","name":"lookup","call_id":"call_9"}}"#,
            r#"{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"city\":"}"#,
            r#"{"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"berlin\"}"}"#,
        ]);
        let outcome = decoder.outcome();
        assert_eq!(outcome.tool_calls.len(), 1);
        assert_eq!(outcome.tool_calls[0].id, "call_9");
        assert_eq!(outcome.tool_calls[0].name, "lookup");
        assert_eq!(outcome.tool_calls[0].arguments, json!({"city": "berlin"}));
    }

    #[test]
    fn arguments_done_fills_a_call_that_streamed_no_deltas() {
        let decoder = decoded(&[
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","name":"ping","call_id":"call_1"}}"#,
            r#"{"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"x\":1}"}"#,
            r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","name":"ping","call_id":"call_1","arguments":"{\"x\":1}"}}"#,
        ]);
        let outcome = decoder.outcome();
        assert_eq!(outcome.tool_calls.len(), 1);
        assert_eq!(outcome.tool_calls[0].arguments, json!({"x": 1}));
    }

    #[test]
    fn terminal_failures_are_errors_not_answers() {
        let mut decoder = OpenResponsesStreamDecoder::default();
        let event = decoder.consume(
            r#"{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}"#,
        );
        let PayloadEvent::Error(detail) = event else {
            panic!("incomplete must be an error");
        };
        assert!(detail.contains("max_output_tokens"), "detail: {detail}");

        let mut decoder = OpenResponsesStreamDecoder::default();
        let event = decoder.consume(
            r#"{"type":"response.failed","response":{"status":"failed","error":{"message":"provider exploded"}}}"#,
        );
        let PayloadEvent::Error(detail) = event else {
            panic!("failed must be an error");
        };
        assert_eq!(detail, "provider exploded");

        // The standalone envelope the spec sequences before the terminal
        // event; its `message` is the cause.
        let mut decoder = OpenResponsesStreamDecoder::default();
        let event = decoder.consume(
            r#"{"type":"error","status":400,"error":{"code":"invalid_request","message":"input too long","param":"input"}}"#,
        );
        let PayloadEvent::Error(detail) = event else {
            panic!("error envelope must be an error");
        };
        assert_eq!(detail, "input too long");
    }

    #[test]
    fn data_that_is_not_json_is_a_protocol_violation() {
        let mut decoder = OpenResponsesStreamDecoder::default();
        let event = decoder.consume("plainly not json");
        let PayloadEvent::Error(detail) = event else {
            panic!("non-JSON data must be an error, not text");
        };
        assert!(detail.contains("non-JSON"), "detail: {detail}");
    }

    #[test]
    fn empty_and_unknown_events_are_inert() {
        let mut decoder = OpenResponsesStreamDecoder::default();
        assert_eq!(
            decoder.consume(""),
            PayloadEvent::Data(RoundDelta::default())
        );
        // Implementor-prefixed extensions must be ignorable, per the spec.
        assert_eq!(
            decoder.consume(r#"{"type":"acme:trace_event","sequence_number":1}"#),
            PayloadEvent::Data(RoundDelta::default())
        );
        assert_eq!(decoder.outcome().content, "");
    }

    #[test]
    fn a_completed_event_with_a_non_completed_status_is_an_error() {
        let mut decoder = OpenResponsesStreamDecoder::default();
        let event = decoder.consume(
            r#"{"type":"response.completed","response":{"status":"failed","error":{"message":"turn died"}}}"#,
        );
        let PayloadEvent::Error(detail) = event else {
            panic!("a terminal status mismatch must be an error");
        };
        assert!(detail.contains("turn died"), "detail: {detail}");
    }

    /// Binds a loopback stub that answers every request on one route with the
    /// same status, content type and body.
    async fn spawn_stub(
        route: &str,
        status: u16,
        body: &str,
        content_type: &'static str,
    ) -> String {
        let code = axum::http::StatusCode::from_u16(status).expect("valid stub status");
        let body = body.to_owned();
        let router = axum::Router::new().route(
            route,
            axum::routing::any(move || {
                let body = body.clone();
                async move {
                    (
                        code,
                        [(axum::http::header::CONTENT_TYPE, content_type)],
                        body,
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("the stub binds loopback");
        let base_url = format!(
            "http://{}",
            listener.local_addr().expect("the stub has an address")
        );
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        base_url
    }

    async fn spawn_json_stub(route: &str, status: u16, body: &Value) -> String {
        spawn_stub(route, status, body.to_string().as_str(), "application/json").await
    }

    fn config(base_url: String) -> ProviderConfig {
        ProviderConfig::new(base_url, "test-key").allow_private_targets(true)
    }

    #[tokio::test]
    async fn the_catalog_reads_model_ids() {
        let base_url = spawn_json_stub(
            "/v1/models",
            200,
            &json!({"data": [{"id": "free-a"}, {"id": "free-b"}]}),
        )
        .await;
        let catalog = OpenResponsesProvider::new()
            .catalog(&config(base_url))
            .await
            .expect("catalog reads");
        assert_eq!(
            catalog
                .models
                .iter()
                .map(|m| m.model.as_str())
                .collect::<Vec<_>>(),
            vec!["free-a", "free-b"]
        );
    }

    #[tokio::test]
    async fn an_empty_catalog_is_an_empty_list_and_a_refusal_keeps_its_class() {
        let base_url = spawn_json_stub("/v1/models", 200, &json!({"data": []})).await;
        let catalog = OpenResponsesProvider::new()
            .catalog(&config(base_url))
            .await
            .expect("an empty list is a list");
        assert!(catalog.models.is_empty());

        let base_url = spawn_json_stub(
            "/v1/models",
            429,
            &json!({"error": {"type": "too_many_requests", "message": "slow down"}}),
        )
        .await;
        let error = OpenResponsesProvider::new()
            .catalog(&config(base_url))
            .await
            .expect_err("a rate limit is a rate limit");
        assert_eq!(error.code(), "provider_rate_limited");
    }

    #[tokio::test]
    async fn a_catalog_with_a_non_json_body_is_an_invalid_payload() {
        let base_url = spawn_stub("/v1/models", 200, "<html>gateway</html>", "text/html").await;
        let error = OpenResponsesProvider::new()
            .catalog(&config(base_url))
            .await
            .expect_err("HTML is not a catalog");
        assert_eq!(error.code(), "provider_invalid_response");
    }

    #[tokio::test]
    async fn the_probe_counts_models_and_reports_a_refusal() {
        let base_url = spawn_json_stub(
            "/v1/models",
            200,
            &json!({"data": [{"id": "a"}, {"id": "b"}]}),
        )
        .await;
        let report = OpenResponsesProvider::new().probe(&config(base_url)).await;
        assert!(report.succeeded);
        assert_eq!(report.model_count, Some(2));

        let base_url = spawn_json_stub(
            "/v1/models",
            401,
            &json!({"error": {"message": "Missing API key"}}),
        )
        .await;
        let report = OpenResponsesProvider::new().probe(&config(base_url)).await;
        assert!(!report.succeeded);
        assert_eq!(report.error.as_deref(), Some("Missing API key"));
    }

    #[tokio::test]
    async fn buffered_completion_reads_the_response_object() {
        let base_url = spawn_json_stub(
            "/v1/responses",
            200,
            &json!({
                "status": "completed",
                "model": "free-model",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
            }),
        )
        .await;
        let outcome = OpenResponsesProvider::new()
            .complete(&config(base_url), request(OutputFormat::Text, None))
            .await
            .expect("completion reads");
        assert_eq!(outcome.content, "answer");
    }

    #[tokio::test]
    async fn a_stream_start_refusal_keeps_the_standard_error_type() {
        let base_url = spawn_json_stub(
            "/v1/responses",
            400,
            &json!({
                "type": "error",
                "error": {"type": "ModelProtocolUnsupported", "message": "not on this protocol"}
            }),
        )
        .await;
        let error = OpenResponsesProvider::new()
            .stream(&config(base_url), request(OutputFormat::Text, None))
            .await
            .expect_err("a refused stream start is an error");
        assert_eq!(error.code(), "provider_protocol_not_available");
    }

    #[tokio::test]
    async fn a_reasoning_override_the_server_refuses_is_retried_without_it() {
        use axum::response::IntoResponse as _;
        let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let router = axum::Router::new().route(
            "/v1/responses",
            axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
                let attempts = std::sync::Arc::clone(&attempts);
                async move {
                    let retry = attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 0;
                    if !retry && body.get("reasoning").is_some() {
                        return axum::http::StatusCode::BAD_REQUEST.into_response();
                    }
                    axum::Json(json!({
                        "status": "completed",
                        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]
                    }))
                    .into_response()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("the stub binds loopback");
        let base_url = format!(
            "http://{}",
            listener.local_addr().expect("the stub has an address")
        );
        tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        let outcome = OpenResponsesProvider::new()
            .complete(&config(base_url), request(OutputFormat::Text, Some("high")))
            .await
            .expect("the retry without reasoning answers");
        assert_eq!(outcome.content, "ok");
    }

    #[tokio::test]
    async fn an_empty_answer_is_an_empty_response_error() {
        let base_url = spawn_json_stub(
            "/v1/responses",
            200,
            &json!({"status": "completed", "output": []}),
        )
        .await;
        let error = OpenResponsesProvider::new()
            .complete(&config(base_url), request(OutputFormat::Text, None))
            .await
            .expect_err("no output is not an answer");
        assert_eq!(error.code(), "provider_empty_response");
    }

    #[test]
    fn an_empty_base_url_is_refused_before_any_request() {
        let error = OpenResponsesProvider::new()
            .endpoint(&ProviderConfig::new("", "key"), "/v1/responses")
            .expect_err("an empty base URL is refused");
        assert_eq!(error.code(), "provider_invalid_base_url");
    }
}
