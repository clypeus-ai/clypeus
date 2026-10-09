//! Provider abstraction: model catalogs, completions, and incremental streams.
//!
//! The core owns the transport-independent machinery: the SSE splitter, the
//! stream accumulator, retry policy for unsupported reasoning parameters, and
//! base-URL validation. Each provider crate owns its request mapping and its
//! payload decoder.

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

use crate::models::{ChatMessage, ProviderKind, TokenUsage, ToolCall, ToolSpec};
use crate::secrets::SecretString;

/// Fixed error code returned when a provider does not recognize a reasoning
/// level value.
pub const REASONING_NOT_AVAILABLE_CODE: &str = "provider_reasoning_not_available";

/// Fixed error code returned when a provider cannot constrain a model's output to a
/// schema. A caller that needs a document branches on this rather than on prose.
pub const OUTPUT_NOT_AVAILABLE_CODE: &str = "provider_output_not_available";

/// Fixed error code returned when a provider does not serve a model on the
/// requested protocol. A caller may act on it by asking the same service for
/// the other protocol.
pub const PROTOCOL_NOT_AVAILABLE_CODE: &str = "provider_protocol_not_available";

/// Which layer of the connection to a provider broke.
///
/// The fixes belong to different people: a name that does not resolve is a base
/// URL or a DNS problem, a refused or unreachable connection is a host or
/// firewall problem, and a failed handshake is a certificate or proxy problem.
/// One word for all three sends an operator to check all of them, so the
/// classification happens here, where the transport error still carries its
/// cause chain, and not at a call site that receives only a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportKind {
    /// The host name could not be resolved.
    Dns,
    /// The connection could not be established.
    Connect,
    /// The TLS handshake or certificate verification failed.
    Tls,
    /// Any other transport failure: a reset, or a body read that was cut off.
    Other,
}

impl TransportKind {
    /// Stable machine-readable code for this layer.
    pub fn code(self) -> &'static str {
        match self {
            Self::Dns => "provider_dns_error",
            Self::Connect => "provider_connect_error",
            Self::Tls => "provider_tls_error",
            Self::Other => "provider_unreachable",
        }
    }
}

/// Transport-level provider failure. Only [`ProviderError::code`] and
/// [`ProviderError::safe_message`] may leave the process.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ProviderError {
    #[error("Invalid provider base URL: {0}")]
    InvalidBaseUrl(String),
    #[error("Provider timed out")]
    Timeout,
    #[error("Provider returned status {status}: {detail}")]
    Upstream { status: u16, detail: String },
    #[error("Provider transport failure ({kind:?}): {detail}")]
    Transport {
        /// The layer that broke; see [`TransportKind`].
        kind: TransportKind,
        /// The full cause chain, outermost first.
        detail: String,
    },
    #[error("Provider returned an unparseable payload: {0}")]
    InvalidPayload(String),
    #[error("Provider stream failed: {0}")]
    Stream(String),
    #[error("Provider completed the turn without an answer")]
    EmptyResponse,
    /// The provider turned a request away because it is busy. Distinct from
    /// [`Self::Upstream`] because it is the one refusal worth waiting out, and
    /// it usually states for how long.
    #[error("Provider is rate limiting requests")]
    RateLimited { retry_after_secs: Option<u64> },
    #[error("Provider does not recognize reasoning level '{value}' for model '{model}'")]
    UnsupportedReasoning { model: String, value: String },
    #[error("Provider cannot be asked for a JSON document (model '{model}')")]
    UnsupportedOutput { model: String },
    /// The provider does not serve this model on the protocol the request used.
    /// A gateway states this as a structured error type, so a caller learns it
    /// can ask the other protocol instead of treating a configuration guess as
    /// permanent.
    #[error("Provider does not serve model '{model}' on this protocol")]
    UnsupportedProtocol { model: String },
}

impl ProviderError {
    /// Stable machine-readable code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidBaseUrl(_) => "provider_invalid_base_url",
            Self::Timeout => "provider_timeout",
            Self::Upstream { .. } => "provider_unavailable",
            Self::Transport { kind, .. } => kind.code(),
            Self::InvalidPayload(_) => "provider_invalid_response",
            Self::Stream(_) => "provider_stream_error",
            Self::EmptyResponse => "provider_empty_response",
            Self::RateLimited { .. } => "provider_rate_limited",
            Self::UnsupportedReasoning { .. } => REASONING_NOT_AVAILABLE_CODE,
            Self::UnsupportedOutput { .. } => OUTPUT_NOT_AVAILABLE_CODE,
            Self::UnsupportedProtocol { .. } => PROTOCOL_NOT_AVAILABLE_CODE,
        }
    }

    /// Safe client-facing message. The detail of `InvalidBaseUrl` stays visible
    /// because it describes the administrator's own input; for the same reason
    /// the transport cause chain, the status and the provider's own sentence
    /// stay visible too. A gateway reported as merely "unreachable" when the
    /// real answer was a DNS failure or `HTTP 401: missing key` is a gateway an
    /// operator cannot act on, and none of those strings carries a credential:
    /// a request header never appears in a `reqwest` error, and provider text
    /// is bounded by [`MAX_DETAIL_CHARS`].
    pub fn safe_message(&self) -> String {
        match self {
            Self::InvalidBaseUrl(detail) => detail.clone(),
            Self::Timeout => "The provider timed out.".to_string(),
            Self::Upstream { status, detail } => format!(
                "The provider returned HTTP {status}: {}",
                bounded_detail(detail)
            ),
            Self::Transport { detail, .. } => bounded_detail(detail),
            Self::InvalidPayload(detail) => format!(
                "The provider returned an invalid response: {}",
                bounded_detail(detail)
            ),
            Self::Stream(detail) => {
                format!("The provider stream failed: {}", bounded_detail(detail))
            }
            Self::EmptyResponse => "The provider completed the turn without an answer.".to_string(),
            Self::RateLimited { retry_after_secs } => match retry_after_secs {
                Some(seconds) => {
                    format!("The provider is rate limiting requests; retry in {seconds} seconds.")
                }
                None => "The provider is rate limiting requests; try again later.".to_string(),
            },
            Self::UnsupportedReasoning { model, value } => format!(
                "The provider does not support reasoning level '{value}' for model '{model}'."
            ),
            Self::UnsupportedOutput { model } => {
                format!("The provider cannot be asked for a JSON document with model '{model}'.")
            }
            Self::UnsupportedProtocol { model } => {
                format!("The provider does not serve model '{model}' on this protocol.")
            }
        }
    }

    /// A transport failure recorded from a cause that is already text.
    ///
    /// Used where the failure is not a `reqwest` send: a stream dies mid-read, a
    /// store fails, a provider is missing from the registry. The layer is read
    /// from the text, and text that names none is [`TransportKind::Other`].
    pub fn transport(detail: impl Into<String>) -> Self {
        let detail = detail.into();
        let kind = classify_transport_markers(&detail);
        Self::Transport { kind, detail }
    }

    /// Classifies a `reqwest` failure at the layer it broke.
    ///
    /// A deadline becomes [`Self::Timeout`]. Everything else keeps the full
    /// cause chain, because the top-level Display of a `reqwest` error is
    /// "error sending request for url (...)" and names nothing: DNS, a refused
    /// connection and a failed handshake are distinguishable only in the
    /// sources beneath it.
    pub fn transport_error(error: reqwest::Error) -> Self {
        if error.is_timeout() {
            return Self::Timeout;
        }
        let detail = error_chain(&error);
        let named = classify_transport_markers(&detail);
        let kind = if named == TransportKind::Other && error.is_connect() {
            TransportKind::Connect
        } else {
            named
        };
        Self::Transport { kind, detail }
    }

    /// Upstream HTTP statuses that suggest a reasoning parameter was rejected.
    /// Providers retry once without reasoning on these.
    pub fn is_retryable_reasoning_failure(status: u16) -> bool {
        matches!(status, 400 | 404 | 405 | 415 | 422 | 502 | 503)
    }
}

/// Longest provider-supplied text one safe message may carry.
///
/// A transport chain and a gateway's own error body are written for machines,
/// and a problem document is not a log. The value that constructed the error
/// keeps the full text; only what leaves the process is bounded.
const MAX_DETAIL_CHARS: usize = 512;

/// Trims a provider-supplied detail and bounds it at [`MAX_DETAIL_CHARS`].
fn bounded_detail(detail: &str) -> String {
    let trimmed = detail.trim();
    if trimmed.chars().count() <= MAX_DETAIL_CHARS {
        return trimmed.to_owned();
    }
    let mut bounded: String = trimmed.chars().take(MAX_DETAIL_CHARS).collect();
    bounded.push('…');
    bounded
}

/// The full cause chain of a failure, outermost first.
///
/// `reqwest::Error`'s own Display names only the URL; the layer is in the
/// sources: "dns error: error resolving DNS: failed to lookup address
/// information: Name or service not known".
fn error_chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut parts = vec![error.to_string()];
    let mut source = error.source();
    while let Some(cause) = source {
        parts.push(cause.to_string());
        source = cause.source();
    }
    parts.join(": ")
}

/// The resolver and connector phrases a DNS failure carries in its Display.
///
/// The first entry is the hyper connector's own layer name and is what holds
/// across platforms; the rest are the libc messages underneath it, kept so the
/// chain names the OS reason even when the middle layer changes.
const DNS_MARKERS: [&str; 7] = [
    "dns error",
    "error resolving dns",
    "failed to lookup address information",
    "name or service not known",
    "nodename nor servname provided",
    "no such host",
    "temporary failure in name resolution",
];

/// The rustls phrases a failed handshake or a rejected certificate carries.
///
/// A TLS failure reaches the chain as a `tokio-rustls`-wrapped `std::io::Error`
/// whose message is rustls's own Display verbatim (measured: a plaintext server
/// answers a ClientHello with "received corrupt message of type
/// InvalidContentType" and nothing else), and the wrapped error is not reachable
/// for a downcast. The test builds real `rustls::Error` values through the
/// variants a peer can trigger, so the list is checked against the type that
/// emits it.
const TLS_MARKERS: [&str; 14] = [
    "received corrupt message",
    "received unexpected message",
    "invalid peer certificate",
    "invalid certificate revocation list",
    "peer sent no certificates",
    "received fatal alert",
    "handshake not complete",
    "peer is incompatible",
    "peer misbehaved",
    "cannot decrypt peer's message",
    "encrypted client hello failure",
    "peer sent excess record size",
    "presented server name type wasn't supported",
    "peer doesn't support any known protocol",
];

/// Names the failure layer from the Display text of a cause chain.
///
/// DNS is checked first: its leaviest message ("failed to lookup address
/// information") is what the resolver prints, while a TLS failure never names a
/// resolver. Text that names neither is [`TransportKind::Other`]; whether that
/// becomes a connect error is for [`ProviderError::transport_error`] to decide
/// from `reqwest::Error::is_connect`, not for this function to guess.
fn classify_transport_markers(detail: &str) -> TransportKind {
    let lowered = detail.to_ascii_lowercase();
    if DNS_MARKERS.iter().any(|marker| lowered.contains(marker)) {
        TransportKind::Dns
    } else if TLS_MARKERS.iter().any(|marker| lowered.contains(marker)) {
        TransportKind::Tls
    } else {
        TransportKind::Other
    }
}

/// Provider connection settings for one scope.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key: SecretString,
    pub timeout: Duration,
    pub max_output_tokens: i32,
    /// When false (default), base URLs resolving to private, loopback or
    /// link-local addresses are rejected.
    pub allow_private_targets: bool,
    /// Headers every request to this provider must carry, in insertion order.
    /// A gateway that refuses traffic without one (a session header, for
    /// example) otherwise looks like a provider with no models, because the
    /// catalog request fails the same way the completions do.
    pub headers: Vec<(String, SecretString)>,
}

impl ProviderConfig {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: SecretString::new(api_key),
            timeout: Duration::from_secs(60),
            max_output_tokens: 1_200,
            allow_private_targets: false,
            headers: Vec::new(),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_max_output_tokens(mut self, tokens: i32) -> Self {
        self.max_output_tokens = tokens;
        self
    }

    pub fn allow_private_targets(mut self, allow: bool) -> Self {
        self.allow_private_targets = allow;
        self
    }

    /// Adds a header every request to this provider carries. The value is held
    /// as a secret because a header a provider requires in order to accept
    /// traffic is usually a credential, and this struct derives `Debug`.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), SecretString::new(value)));
        self
    }

    /// Applies every configured header to an outgoing request. Adapters call
    /// this on each request, the model list included: a catalog that fails for
    /// a missing header is indistinguishable from a provider with no models.
    pub fn apply_headers(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        self.headers.iter().fold(request, |request, (name, value)| {
            request.header(name.as_str(), value.expose())
        })
    }
}

/// Tool selection policy sent to the provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
}

impl ToolChoice {
    pub fn as_wire(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::None => "none",
            Self::Required => "required",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "none" => Some(Self::None),
            "required" => Some(Self::Required),
            _ => None,
        }
    }
}

/// Reasoning-level sentinel that means "no explicit override": the provider's
/// own default applies.
pub const DEFAULT_REASONING_LEVEL: &str = "default";

/// Normalizes a caller-supplied reasoning level. Absent, empty and the
/// `"default"` sentinel mean "no override"; every other value is returned
/// trimmed and verbatim so it reaches the provider exactly as requested.
pub fn reasoning_override(value: Option<&str>) -> Option<&str> {
    value
        .map(str::trim)
        .filter(|level| !level.is_empty() && *level != DEFAULT_REASONING_LEVEL)
}

/// What shape the model's answer has to be in.
///
/// A schema is not a stronger prompt. It is a constraint the provider itself enforces
/// while the model decodes, so a caller that needs a document gets one or gets an error,
/// rather than prose that a parser downstream has to guess at. Providers that cannot
/// constrain decoding say so through [`ProviderError::UnsupportedOutput`] instead of
/// accepting the request and dropping the constraint.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputFormat {
    /// Free text, which is what a model produces unless it is told otherwise.
    #[default]
    Text,
    /// A JSON document conforming to `schema`, which every provider that supports this
    /// is asked for strictly: a schema that is merely suggested is a schema the model may
    /// ignore, and a caller who needed the document would not learn that it had been.
    JsonSchema {
        /// The name the schema is registered under. Providers that require a name use
        /// it in their logs and error messages; it never reaches the model.
        name: String,
        /// A JSON Schema the answer must validate against.
        schema: serde_json::Value,
    },
}

/// One provider completion request.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Open reasoning level. `None` (or the `"default"` sentinel) sends no
    /// override; any other value is forwarded to the provider verbatim.
    pub reasoning: Option<String>,
    pub tools: Vec<ToolSpec>,
    pub tool_choice: ToolChoice,
    pub max_output_tokens: i32,
    /// The shape the answer must take. [`OutputFormat::Text`] unless the caller needs a
    /// document.
    pub output: OutputFormat,
}

/// Everything a provider answer carries beyond the text itself.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AssistantOutcome {
    pub content: String,
    pub reasoning: Option<String>,
    pub model: Option<String>,
    pub usage: Option<TokenUsage>,
    pub tool_calls: Vec<ToolCall>,
}

impl AssistantOutcome {
    pub fn is_empty(&self) -> bool {
        self.content.trim().is_empty() && self.tool_calls.is_empty()
    }
}

/// One model advertised by a provider catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelCapability {
    pub model: String,
    /// Reasoning level names the model advertises, verbatim. Empty when the
    /// model accepts no explicit reasoning override.
    pub reasoning_levels: Vec<String>,
    /// Level the provider uses when a request names none. Empty or `"default"`
    /// when the provider has no explicit default.
    pub default_reasoning_level: String,
}

/// Catalog of models advertised by a provider.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelCatalog {
    pub models: Vec<ModelCapability>,
}

impl ModelCatalog {
    pub fn find(&self, model: &str) -> Option<&ModelCapability> {
        self.models.iter().find(|entry| entry.model == model)
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

/// Result of a credential/connectivity probe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProbeReport {
    pub succeeded: bool,
    pub model_count: Option<i32>,
    pub elapsed_ms: i64,
    pub checked_at_utc: chrono::DateTime<chrono::Utc>,
    pub error: Option<String>,
}

/// One incremental delta destined for a client stream.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamDelta {
    Content(String),
    Reasoning(String),
}

/// Result of consuming one provider SSE payload.
#[derive(Debug, Clone, PartialEq)]
pub enum PayloadEvent {
    Data(RoundDelta),
    Done,
    Error(String),
}

/// Incremental pieces of one provider chunk, in arrival order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoundDelta {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub usage: Option<TokenUsage>,
}

/// Provider-specific SSE payload decoder. Implementations accumulate tool call
/// fragments and usage; core forwards text/reasoning deltas to the client.
pub trait StreamDecoder: Send {
    fn consume(&mut self, payload: &str) -> PayloadEvent;
    /// Accumulated outcome. Valid once the stream is exhausted.
    fn outcome(&self) -> AssistantOutcome;
}

#[derive(Debug, Clone, Default)]
struct PartialToolCall {
    id: String,
    name: String,
    /// Non-streamed input seen in one block (Anthropic `content_block_start`).
    initial_input: String,
    /// Fragments streamed after the block started.
    input_buffer: String,
}

/// Provider-agnostic accumulation of one streamed round: answer text,
/// reasoning trace, usage, and tool-call fragments. Provider decoders feed it
/// from their own payload shapes.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    content: String,
    reasoning: String,
    usage: Option<TokenUsage>,
    tool_calls: BTreeMap<u64, PartialToolCall>,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_content(&mut self, text: &str) {
        self.content.push_str(text);
    }

    pub fn push_reasoning(&mut self, text: &str) {
        self.reasoning.push_str(text);
    }

    pub fn merge_usage(&mut self, usage: TokenUsage) {
        let merged = self.usage.get_or_insert_with(TokenUsage::default);
        macro_rules! merge_field {
            ($field:ident) => {
                if usage.$field.is_some() {
                    merged.$field = usage.$field;
                }
            };
        }
        merge_field!(prompt_tokens);
        merge_field!(completion_tokens);
        merge_field!(total_tokens);
        merge_field!(cached_tokens);
        merge_field!(reasoning_tokens);
    }

    /// Applies one OpenAI `tool_calls` delta fragment.
    pub fn push_tool_call_fragment(
        &mut self,
        index: u64,
        id: Option<&str>,
        name: Option<&str>,
        arguments: Option<&str>,
    ) {
        let entry = self.tool_calls.entry(index).or_default();
        if let Some(id) = id.filter(|id| !id.is_empty()) {
            entry.id = id.to_string();
        }
        if let Some(name) = name.filter(|name| !name.is_empty()) {
            entry.name = name.to_string();
        }
        if let Some(arguments) = arguments {
            entry.input_buffer.push_str(arguments);
        }
    }

    /// Applies an Anthropic `tool_use` block opening.
    pub fn open_tool_use(&mut self, index: u64, id: &str, name: &str) {
        let entry = self.tool_calls.entry(index).or_default();
        entry.id = id.to_string();
        entry.name = name.to_string();
    }

    /// Applies an Anthropic `input_json_delta` fragment.
    pub fn push_tool_input(&mut self, index: u64, partial_json: &str) {
        self.tool_calls
            .entry(index)
            .or_default()
            .input_buffer
            .push_str(partial_json);
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn reasoning(&self) -> Option<&str> {
        (!self.reasoning.is_empty()).then_some(self.reasoning.as_str())
    }

    pub fn usage(&self) -> Option<&TokenUsage> {
        self.usage.as_ref()
    }

    pub fn tool_calls(&self) -> Vec<ToolCall> {
        self.tool_calls
            .values()
            .map(|partial| {
                let raw = if partial.input_buffer.is_empty() {
                    partial.initial_input.as_str()
                } else {
                    partial.input_buffer.as_str()
                };
                let arguments = if raw.trim().is_empty() {
                    Value::Object(serde_json::Map::new())
                } else {
                    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
                };
                ToolCall {
                    id: partial.id.clone(),
                    name: partial.name.clone(),
                    arguments,
                }
            })
            .collect()
    }

    /// Sets the non-streamed tool input captured at block start. Used when a
    /// provider sends the full `input` object instead of JSON deltas.
    pub fn set_tool_initial_input(&mut self, index: u64, input: &Value) {
        let entry = self.tool_calls.entry(index).or_default();
        if input.as_object().is_some_and(|object| !object.is_empty()) {
            entry.initial_input = input.to_string();
        }
    }

    pub fn outcome(&self) -> AssistantOutcome {
        AssistantOutcome {
            content: self.content.clone(),
            reasoning: self.reasoning().map(str::to_string),
            model: None,
            usage: self.usage.clone(),
            tool_calls: self.tool_calls(),
        }
    }
}

type ByteStream = BoxStream<'static, Result<Bytes, std::io::Error>>;

/// Incremental reader over one streamed provider round.
pub struct ProviderStream {
    upstream: ByteStream,
    splitter: SseSplitter,
    decoder: Box<dyn StreamDecoder>,
    finished: bool,
    error: Option<ProviderError>,
}

impl std::fmt::Debug for ProviderStream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderStream")
            .field("finished", &self.finished)
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl ProviderStream {
    pub fn new(upstream: ByteStream, decoder: Box<dyn StreamDecoder>) -> Self {
        Self {
            upstream,
            splitter: SseSplitter::default(),
            decoder,
            finished: false,
            error: None,
        }
    }

    /// Reads until the next incremental delta or the end of the round.
    pub async fn next(&mut self) -> Option<Result<Vec<StreamDelta>, ProviderError>> {
        if let Some(error) = self.error.take() {
            return Some(Err(error));
        }
        if self.finished {
            return None;
        }
        loop {
            match self.upstream.next().await {
                Some(Ok(chunk)) => {
                    let mut deltas = Vec::new();
                    for block in self.splitter.push(&chunk) {
                        deltas.extend(self.consume(&block));
                    }
                    if !deltas.is_empty() {
                        return Some(Ok(deltas));
                    }
                    if let Some(error) = self.error.take() {
                        return Some(Err(error));
                    }
                    if self.finished {
                        return None;
                    }
                }
                Some(Err(error)) => {
                    self.finished = true;
                    return Some(Err(ProviderError::transport(error.to_string())));
                }
                None => {
                    self.finished = true;
                    if let Some(tail) = self.splitter.flush() {
                        let deltas = self.consume(&tail);
                        if !deltas.is_empty() {
                            return Some(Ok(deltas));
                        }
                    }
                    if let Some(error) = self.error.take() {
                        return Some(Err(error));
                    }
                    return None;
                }
            }
        }
    }

    fn consume(&mut self, block: &[u8]) -> Vec<StreamDelta> {
        match self.decoder.consume(&event_payload(block)) {
            PayloadEvent::Data(round) => {
                let mut deltas = Vec::new();
                if let Some(reasoning) = round.reasoning {
                    deltas.push(StreamDelta::Reasoning(reasoning));
                }
                if let Some(content) = round.content {
                    deltas.push(StreamDelta::Content(content));
                }
                deltas
            }
            PayloadEvent::Done => {
                self.finished = true;
                Vec::new()
            }
            PayloadEvent::Error(detail) => {
                self.finished = true;
                self.error = Some(ProviderError::Stream(detail));
                Vec::new()
            }
        }
    }

    /// The accumulated round result.
    pub fn into_outcome(self) -> AssistantOutcome {
        self.decoder.outcome()
    }
}

/// A provider backend implementation.
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Stable provider identifier (`openai`, `anthropic`,
    /// `openai_responses`).
    fn id(&self) -> &'static str;

    /// Fetches the model catalog.
    async fn catalog(&self, config: &ProviderConfig) -> Result<ModelCatalog, ProviderError>;

    /// Tests credentials and reachability.
    async fn probe(&self, config: &ProviderConfig) -> ProbeReport;

    /// Runs one buffered completion.
    async fn complete(
        &self,
        config: &ProviderConfig,
        request: CompletionRequest,
    ) -> Result<AssistantOutcome, ProviderError>;

    /// Starts one streamed completion.
    async fn stream(
        &self,
        config: &ProviderConfig,
        request: CompletionRequest,
    ) -> Result<ProviderStream, ProviderError>;
}

/// Registry of provider implementations, keyed by [`ProviderKind`].
#[derive(Default)]
pub struct ProviderRegistry {
    providers: BTreeMap<ProviderKind, Arc<dyn Provider>>,
}

impl std::fmt::Debug for ProviderRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderRegistry")
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(mut self, kind: ProviderKind, provider: Arc<dyn Provider>) -> Self {
        self.providers.insert(kind, provider);
        self
    }

    pub fn get(&self, kind: ProviderKind) -> Option<&Arc<dyn Provider>> {
        self.providers.get(&kind)
    }

    pub fn kinds(&self) -> Vec<ProviderKind> {
        self.providers.keys().copied().collect()
    }
}

/// Splits raw upstream bytes into complete SSE event blocks.
#[derive(Debug, Default)]
pub struct SseSplitter {
    buffer: Vec<u8>,
}

impl SseSplitter {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<Vec<u8>> {
        self.buffer.extend_from_slice(chunk);
        let mut blocks = Vec::new();
        while let Some(boundary) = find_event_boundary(&self.buffer) {
            let block = self.buffer[..boundary].to_vec();
            self.buffer.drain(..boundary);
            blocks.push(block);
        }
        blocks
    }

    pub fn flush(&mut self) -> Option<Vec<u8>> {
        if self.buffer.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.buffer))
    }
}

fn find_event_boundary(buffer: &[u8]) -> Option<usize> {
    if buffer.is_empty() {
        return None;
    }
    let mut index = 0;
    while index + 1 < buffer.len() {
        if buffer[index] == b'\n' && buffer[index + 1] == b'\n' {
            return Some(index + 2);
        }
        if index + 3 < buffer.len()
            && buffer[index] == b'\r'
            && buffer[index + 1] == b'\n'
            && buffer[index + 2] == b'\r'
            && buffer[index + 3] == b'\n'
        {
            return Some(index + 4);
        }
        index += 1;
    }
    None
}

/// Extracts the `data:` payload of one SSE event block.
pub fn event_payload(block: &[u8]) -> String {
    let text = String::from_utf8_lossy(block);
    let mut payload = String::new();
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.strip_prefix(' ').unwrap_or(data);
        if !payload.is_empty() {
            payload.push('\n');
        }
        payload.push_str(data);
    }
    payload
}

fn ensure_routable(addr: &IpAddr) -> Result<(), ProviderError> {
    let blocked = match addr {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => is_blocked_v6(v6),
    };
    if blocked {
        Err(ProviderError::InvalidBaseUrl(format!(
            "base URL resolves to a non-routable address ({addr})"
        )))
    } else {
        Ok(())
    }
}

fn is_blocked_v4(addr: &Ipv4Addr) -> bool {
    addr.is_loopback()
        || addr.is_private()
        || addr.is_link_local()
        || addr.is_broadcast()
        || addr.is_multicast()
        || addr.is_unspecified()
        || addr.octets()[0] == 0
        || matches!(addr.octets(), [169, 254, ..])
        || matches!(addr.octets(), [100, byte, ..] if (64..=127).contains(&byte))
}

fn is_blocked_v6(addr: &Ipv6Addr) -> bool {
    addr.is_loopback()
        || addr.is_unspecified()
        || addr.is_multicast()
        || (addr.segments()[0] & 0xfe00) == 0xfc00
        || (addr.segments()[0] & 0xffc0) == 0xfe80
        || addr.to_ipv4_mapped().is_some_and(|v| is_blocked_v4(&v))
}

/// Validates and canonicalizes a provider base URL.
pub fn validate_base_url(raw: &str, allow_private_targets: bool) -> Result<String, ProviderError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(ProviderError::InvalidBaseUrl("base URL is required".into()));
    }
    if trimmed.len() > 512 {
        return Err(ProviderError::InvalidBaseUrl(
            "base URL exceeds 512 characters".into(),
        ));
    }
    let parsed = url::Url::parse(trimmed)
        .map_err(|error| ProviderError::InvalidBaseUrl(error.to_string()))?;
    match parsed.scheme() {
        "http" | "https" => {}
        other => {
            return Err(ProviderError::InvalidBaseUrl(format!(
                "scheme must be http(s), got {other}"
            )));
        }
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(ProviderError::InvalidBaseUrl(
            "base URL must not include a query string or fragment".into(),
        ));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| ProviderError::InvalidBaseUrl("base URL must include a host".into()))?;
    if host.is_empty() {
        return Err(ProviderError::InvalidBaseUrl(
            "base URL must include a host".into(),
        ));
    }

    if !allow_private_targets {
        if let Ok(addr) = host.parse::<IpAddr>() {
            ensure_routable(&addr)?;
        } else {
            let probe = format!("{host}:{}", parsed.port_or_known_default().unwrap_or(443));
            if let Ok(iter) = probe.to_socket_addrs() {
                for addr in iter {
                    ensure_routable(&addr.ip())?;
                }
            }
        }
    }

    Ok(parsed.to_string().trim_end_matches('/').to_string())
}

/// Extracts the first provider error message found in a payload.
pub fn extract_error_message(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    for path in &[
        ["message"].as_slice(),
        &["detail"],
        &["error", "message"],
        &["error", "detail"],
    ] {
        if let Some(text) = walk_path(&value, path).and_then(Value::as_str)
            && !text.trim().is_empty()
        {
            return Some(text.trim().to_string());
        }
    }
    None
}

/// The error type an OpenAI-compatible gateway uses to say it does not serve a
/// model on the requested protocol.
///
/// Measured against OpenCode Go:
/// `{"type":"error","error":{"type":"ModelProtocolUnsupported","message":"Model
/// does not support this protocol."}}`. The message names no model, so the
/// request's own model is what the refusal has to carry.
pub const MODEL_PROTOCOL_UNSUPPORTED_TYPE: &str = "ModelProtocolUnsupported";

/// Extracts the structured error type a provider names, when it names one.
///
/// A gateway that can say `ModelProtocolUnsupported` is giving a caller a fact
/// to branch on; the prose beside it is written for a person, may be reworded,
/// and on the measured gateway says the same sentence for every model. Reading
/// the type is what keeps a caller's behaviour from depending on that sentence.
pub fn extract_error_type(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    for path in &[["error", "type"].as_slice(), &["error", "code"], &["type"]] {
        if let Some(text) = walk_path(&value, path).and_then(Value::as_str) {
            let trimmed = text.trim();
            if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("error") {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Reads the wait a provider asked for from response headers.
///
/// The header only, never the prose in the body: the same number is often
/// repeated in a sentence, and reading a sentence to get a duration is how a
/// provider's wording becomes this process's contract. A header is the part a
/// provider writes for a program to read. Only the integer-seconds form is
/// parsed; an HTTP date or an absent header means the provider did not state a
/// wait, which callers already handle.
pub fn retry_after_from_headers(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok())
}

/// Classifies a non-success provider response.
///
/// A `429` is a rate limit whatever else the body says, and the wait the header
/// states is what makes it actionable. A structured
/// `ModelProtocolUnsupported` type is a protocol refusal; `model` is absent
/// where the request named none (a model list), and a protocol refusal without
/// a model has nothing to carry. Everything else is an upstream failure with the
/// provider's own message, or `detail_fallback` when the body names none.
pub fn response_error(
    status: u16,
    retry_after_secs: Option<u64>,
    body: &str,
    model: Option<&str>,
    detail_fallback: &str,
) -> ProviderError {
    if status == 429 {
        return ProviderError::RateLimited { retry_after_secs };
    }
    if let Some(model) = model
        && extract_error_type(body).is_some_and(|kind| kind == MODEL_PROTOCOL_UNSUPPORTED_TYPE)
    {
        return ProviderError::UnsupportedProtocol {
            model: model.to_string(),
        };
    }
    ProviderError::Upstream {
        status,
        detail: extract_error_message(body).unwrap_or_else(|| detail_fallback.to_string()),
    }
}

/// Walks a JSON path.
pub fn walk_path<'a>(root: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cursor = root;
    for key in path {
        cursor = cursor.get(*key)?;
    }
    Some(cursor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_validation_rejects_malformed_inputs() {
        for invalid in [
            "",
            "ftp://example.com",
            "https://example.com?x=1",
            "https://example.com#frag",
            "https://",
        ] {
            assert!(
                validate_base_url(invalid, false).is_err(),
                "{invalid} must be rejected"
            );
        }
        assert_eq!(
            validate_base_url("https://api.example.com/", false).unwrap(),
            "https://api.example.com"
        );
    }

    #[test]
    fn base_url_validation_blocks_private_targets_unless_allowed() {
        assert!(validate_base_url("http://127.0.0.1:8080", false).is_err());
        assert!(validate_base_url("http://10.0.0.5:8080", false).is_err());
        assert!(validate_base_url("http://192.168.1.1", false).is_err());
        assert!(validate_base_url("http://169.254.1.1", false).is_err());
        assert!(validate_base_url("http://[::1]:9000", false).is_err());
        assert!(validate_base_url("http://127.0.0.1:8080", true).is_ok());
        assert!(validate_base_url("http://example.invalid", true).is_ok());
    }

    #[test]
    fn provider_config_collects_headers_in_order_without_logging_values() {
        let config = ProviderConfig::new("https://example.com", "key")
            .with_header("x-session-id", "stable-session")
            .with_header("x-other", "another");
        assert_eq!(config.headers.len(), 2);
        assert_eq!(config.headers[0].0, "x-session-id");
        assert_eq!(config.headers[0].1.expose(), "stable-session");
        assert_eq!(config.headers[1].0, "x-other");
        assert!(!format!("{config:?}").contains("stable-session"));
    }

    #[test]
    fn retryable_reasoning_statuses_match_the_contract() {
        for status in [400, 404, 405, 415, 422, 502, 503] {
            assert!(ProviderError::is_retryable_reasoning_failure(status));
        }
        for status in [401, 403, 429, 500, 501] {
            assert!(!ProviderError::is_retryable_reasoning_failure(status));
        }
    }

    #[test]
    fn sse_splitter_handles_crlf_and_partial_blocks() {
        let mut splitter = SseSplitter::default();
        assert!(splitter.push(b"data: {").is_empty());
        let blocks = splitter.push(b"\"a\":1}\r\n\r\n");
        assert_eq!(blocks.len(), 1);
        assert_eq!(event_payload(&blocks[0]), "{\"a\":1}");
        assert!(splitter.push(b"data: x\n").is_empty());
        assert_eq!(event_payload(&splitter.flush().unwrap()), "x");
    }

    #[test]
    fn error_message_extraction_walks_common_shapes() {
        assert_eq!(
            extract_error_message(r#"{"error":{"message":"bad key"}}"#).as_deref(),
            Some("bad key")
        );
        assert_eq!(
            extract_error_message(r#"{"detail":"nope"}"#).as_deref(),
            Some("nope")
        );
        assert!(extract_error_message("plain text").is_none());
    }

    #[test]
    fn protocol_refusal_is_read_from_the_structured_type_not_the_prose() {
        // The measured gateway body, verbatim. The message names no model; the
        // request's model is what the classifier carries.
        let measured = r#"{"type":"error","error":{"type":"ModelProtocolUnsupported","message":"Model does not support this protocol."}}"#;
        assert_eq!(
            extract_error_type(measured).as_deref(),
            Some("ModelProtocolUnsupported")
        );
        assert_eq!(
            response_error(400, None, measured, Some("longcat-2.0"), "bad request"),
            ProviderError::UnsupportedProtocol {
                model: "longcat-2.0".to_owned()
            }
        );

        // The same words without the structured type are ordinary prose and must
        // stay an upstream failure: a gateway is free to reword a sentence.
        let prose_only = r#"{"error":{"message":"ModelProtocolUnsupported: model does not support this protocol"}}"#;
        assert_eq!(extract_error_type(prose_only), None);
        assert_eq!(
            response_error(400, None, prose_only, Some("longcat-2.0"), "bad request"),
            ProviderError::Upstream {
                status: 400,
                detail: "ModelProtocolUnsupported: model does not support this protocol".to_owned(),
            }
        );

        // A protocol refusal with no model to carry stays upstream.
        assert_eq!(
            response_error(400, None, measured, None, "bad request"),
            ProviderError::Upstream {
                status: 400,
                detail: "Model does not support this protocol.".to_owned(),
            }
        );
    }

    #[test]
    fn a_429_is_a_rate_limit_with_the_wait_the_header_stated() {
        assert_eq!(
            response_error(
                429,
                Some(42),
                r#"{"error":{"message":"slow down"}}"#,
                Some("m"),
                "busy"
            ),
            ProviderError::RateLimited {
                retry_after_secs: Some(42)
            }
        );
        assert_eq!(
            response_error(429, None, "{}", Some("m"), "busy"),
            ProviderError::RateLimited {
                retry_after_secs: None
            }
        );
        assert_eq!(
            ProviderError::RateLimited {
                retry_after_secs: Some(42)
            }
            .code(),
            "provider_rate_limited"
        );
        assert!(
            ProviderError::RateLimited {
                retry_after_secs: Some(42)
            }
            .safe_message()
            .contains("42")
        );
    }

    #[test]
    fn retry_after_reading_ignores_unparsable_and_date_forms() {
        let mut headers = reqwest::header::HeaderMap::new();
        assert_eq!(retry_after_from_headers(&headers), None);
        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static(" 17 "),
        );
        assert_eq!(retry_after_from_headers(&headers), Some(17));
        headers.insert(
            reqwest::header::RETRY_AFTER,
            reqwest::header::HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
        );
        assert_eq!(retry_after_from_headers(&headers), None);
    }

    #[test]
    fn new_provider_errors_keep_stable_codes_and_safe_messages() {
        assert_eq!(
            ProviderError::UnsupportedProtocol {
                model: "m".to_owned()
            }
            .code(),
            PROTOCOL_NOT_AVAILABLE_CODE
        );
        assert_eq!(
            ProviderError::UnsupportedProtocol {
                model: "m".to_owned()
            }
            .safe_message(),
            "The provider does not serve model 'm' on this protocol."
        );
    }

    #[test]
    fn an_upstream_failure_states_the_status_and_the_providers_own_sentence() {
        let error = ProviderError::Upstream {
            status: 401,
            detail: "Missing API key".to_owned(),
        };
        assert_eq!(
            error.safe_message(),
            "The provider returned HTTP 401: Missing API key"
        );
        assert_eq!(error.code(), "provider_unavailable");
    }

    #[test]
    fn a_transport_chain_names_its_layer_and_keeps_the_cause() {
        for (kind, detail) in [
            (
                TransportKind::Dns,
                "error sending request for url (http://x.invalid/v1/models): client error \
                 (Connect): dns error: error resolving DNS: failed to lookup address \
                 information: Name or service not known",
            ),
            (
                TransportKind::Tls,
                "error sending request for url (https://127.0.0.1:1/v1/models): client error \
                 (Connect): received corrupt message of type InvalidContentType",
            ),
            (
                TransportKind::Connect,
                "error sending request for url (http://127.0.0.1:1/v1/models): client error \
                 (Connect): tcp connect error: Connection refused (os error 111)",
            ),
        ] {
            let error = ProviderError::Transport {
                kind,
                detail: detail.to_owned(),
            };
            assert_eq!(error.code(), kind.code());
            assert!(error.safe_message().contains(detail), "{detail}");
        }
        assert_eq!(TransportKind::Dns.code(), "provider_dns_error");
        assert_eq!(TransportKind::Tls.code(), "provider_tls_error");
        assert_eq!(TransportKind::Connect.code(), "provider_connect_error");
        assert_eq!(TransportKind::Other.code(), "provider_unreachable");
    }

    #[test]
    fn the_tls_marker_list_covers_the_rustls_refusals_it_names() {
        // Real `rustls::Error` values, formatted by rustls itself: the wrapped
        // handshake error reaches the chain as this Display text and nothing
        // else, so the marker list is checked against the type that emits it.
        for error in [
            rustls::Error::InvalidCertificate(rustls::CertificateError::UnknownIssuer),
            rustls::Error::InvalidCertificate(rustls::CertificateError::Expired),
            rustls::Error::NoCertificatesPresented,
            rustls::Error::InvalidMessage(rustls::InvalidMessage::InvalidContentType),
            rustls::Error::DecryptError,
            rustls::Error::AlertReceived(rustls::AlertDescription::HandshakeFailure),
            rustls::Error::PeerIncompatible(rustls::PeerIncompatible::NoCipherSuitesInCommon),
            rustls::Error::PeerMisbehaved(rustls::PeerMisbehaved::IllegalMiddleboxChangeCipherSpec),
            rustls::Error::HandshakeNotComplete,
            rustls::Error::NoApplicationProtocol,
        ] {
            let text = error.to_string();
            assert_eq!(
                classify_transport_markers(&text),
                TransportKind::Tls,
                "{text}"
            );
        }
    }

    #[test]
    fn a_provider_detail_is_bounded_but_its_head_survives() {
        for error in [
            ProviderError::InvalidPayload("y".repeat(MAX_DETAIL_CHARS + 64)),
            ProviderError::Stream("z".repeat(MAX_DETAIL_CHARS + 64)),
        ] {
            let message = error.safe_message();
            assert!(message.contains('…'), "{message}");
            assert!(!message.contains(&"y".repeat(MAX_DETAIL_CHARS + 1)));
            assert!(!message.contains(&"z".repeat(MAX_DETAIL_CHARS + 1)));
        }
        assert_eq!(
            ProviderError::transport("  stated cause  ").safe_message(),
            "stated cause"
        );
    }

    /// Binds a listener, then drops it, leaving a port nothing accepts on.
    async fn closed_port() -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        drop(listener);
        address
    }

    /// A listener that accepts and then holds the connection, so a deadline or
    /// a handshake can be observed instead of a closed socket.
    async fn holding_listener(reply: Option<&'static [u8]>) -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("address");
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((mut socket, _)) = listener.accept().await {
                if let Some(reply) = reply {
                    use tokio::io::AsyncWriteExt;
                    let _ = socket.write_all(reply).await;
                }
                held.push(socket);
            }
        });
        address
    }

    #[tokio::test]
    async fn a_resolution_failure_is_named_dns() {
        let error = reqwest::Client::new()
            .get("http://a-host-that-does-not-resolve.invalid/v1/models")
            .send()
            .await
            .expect_err("must not resolve");
        match ProviderError::transport_error(error) {
            ProviderError::Transport { kind, detail } => {
                assert_eq!(kind, TransportKind::Dns);
                assert!(detail.to_lowercase().contains("dns"), "{detail}");
            }
            other => panic!("expected a transport failure, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_refused_connection_is_named_connect() {
        let address = closed_port().await;
        let error = reqwest::Client::new()
            .get(format!("http://{address}/v1/models"))
            .send()
            .await
            .expect_err("must refuse");
        match ProviderError::transport_error(error) {
            ProviderError::Transport { kind, detail } => {
                assert_eq!(kind, TransportKind::Connect);
                assert!(detail.to_lowercase().contains("connect"), "{detail}");
            }
            other => panic!("expected a transport failure, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_plaintext_answer_to_https_is_named_tls() {
        let address = holding_listener(Some(b"not tls at all\n")).await;
        let error = reqwest::Client::new()
            .get(format!("https://{address}/v1/models"))
            .send()
            .await
            .expect_err("must fail the handshake");
        match ProviderError::transport_error(error) {
            ProviderError::Transport { kind, detail } => {
                assert_eq!(kind, TransportKind::Tls, "{detail}");
            }
            other => panic!("expected a transport failure, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_deadline_is_named_timeout() {
        let address = holding_listener(None).await;
        let error = reqwest::Client::new()
            .get(format!("http://{address}/v1/models"))
            .timeout(Duration::from_millis(250))
            .send()
            .await
            .expect_err("must time out");
        assert!(matches!(
            ProviderError::transport_error(error),
            ProviderError::Timeout
        ));
    }
}
