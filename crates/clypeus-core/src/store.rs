//! Store contracts: scope settings, conversations, approvals, usage.
//!
//! Implementations live in the `clypeus-store-*` crates. Every method takes a
//! [`ScopeId`] and the subject, and every query is filtered by both: the core
//! treats the scope as opaque, but it is mandatory on every read and write.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::models::{ChatMessage, ProviderKind, TokenUsage};
use crate::principal::ScopeId;
use crate::profile::ProfileSelection;

/// Readiness probe contract. `Err` is a dependency failure with a safe detail
/// string.
#[async_trait::async_trait]
pub trait Readiness: Send + Sync {
    async fn check_ready(&self) -> Result<(), String>;
}

/// Store failure. `Validation` is a caller error (HTTP 400); everything else
/// is a server error and never reaches the client verbatim.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StoreError {
    #[error("not found")]
    NotFound,
    #[error("validation error: {0}")]
    Validation(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("store error: {0}")]
    Backend(String),
}

impl StoreError {
    pub fn backend(error: impl std::fmt::Display) -> Self {
        Self::Backend(error.to_string())
    }
}

// ---------------------------------------------------------------------------
// Scope settings
// ---------------------------------------------------------------------------

/// Provider settings for one scope. The API key itself lives in a
/// [`crate::secrets::SecretStore`]; this record only tracks whether one is
/// present.
#[derive(Debug, Clone, PartialEq)]
pub struct ScopeSettings {
    pub provider_kind: ProviderKind,
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub timeout_ms: i32,
    pub max_output_tokens: i32,
    pub api_key_present: bool,
    pub profile: ProfileSelection,
    /// Domain fields owned by the application.
    pub extensions: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

/// Partial update. `None` fields keep their stored value; `clear_api_key`
/// removes the stored key.
#[derive(Debug, Clone, Default)]
pub struct ScopeSettingsUpdate {
    pub provider_kind: Option<ProviderKind>,
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub timeout_ms: Option<i32>,
    pub max_output_tokens: Option<i32>,
    pub api_key_present: Option<bool>,
    pub profile: Option<ProfileSelection>,
    pub extensions: Option<Value>,
}

#[async_trait::async_trait]
pub trait ScopeSettingsStore: Send + Sync {
    async fn get(&self, scope: &ScopeId) -> Result<Option<ScopeSettings>, StoreError>;
    async fn upsert(
        &self,
        scope: &ScopeId,
        update: ScopeSettingsUpdate,
    ) -> Result<ScopeSettings, StoreError>;
}

// ---------------------------------------------------------------------------
// Conversation vocabulary
// ---------------------------------------------------------------------------

/// Lifecycle state of a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum MessageStatus {
    Pending,
    Streaming,
    AwaitingApproval,
    Complete,
    Error,
}

impl MessageStatus {
    pub fn as_wire(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Streaming => "streaming",
            Self::AwaitingApproval => "awaiting_approval",
            Self::Complete => "complete",
            Self::Error => "error",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "pending" => Some(Self::Pending),
            "streaming" => Some(Self::Streaming),
            "awaiting_approval" => Some(Self::AwaitingApproval),
            "complete" => Some(Self::Complete),
            "error" => Some(Self::Error),
            _ => None,
        }
    }
}

/// Lifecycle state of one tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    Requested,
    AwaitingApproval,
    Approved,
    Denied,
    Running,
    Succeeded,
    Failed,
    Expired,
}

impl ToolCallStatus {
    pub fn as_wire(self) -> &'static str {
        match self {
            Self::Requested => "requested",
            Self::AwaitingApproval => "awaiting_approval",
            Self::Approved => "approved",
            Self::Denied => "denied",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Expired => "expired",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "requested" => Some(Self::Requested),
            "awaiting_approval" => Some(Self::AwaitingApproval),
            "approved" => Some(Self::Approved),
            "denied" => Some(Self::Denied),
            "running" => Some(Self::Running),
            "succeeded" => Some(Self::Succeeded),
            "failed" => Some(Self::Failed),
            "expired" => Some(Self::Expired),
            _ => None,
        }
    }
}

/// Feedback rating on an assistant message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum FeedbackRating {
    Up,
    Down,
}

impl FeedbackRating {
    pub fn as_wire(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "up" => Some(Self::Up),
            "down" => Some(Self::Down),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct Feedback {
    pub message_id: Uuid,
    pub rating: FeedbackRating,
    pub comment: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Thread metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct Thread {
    pub id: Uuid,
    pub scope_id: String,
    pub subject: String,
    pub title: String,
    pub pinned: bool,
    pub model: Option<String>,
    pub active_leaf_message_id: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Default)]
pub struct CreateThread {
    pub title: Option<String>,
    pub model: Option<String>,
    pub pinned: bool,
}

/// Partial thread update. `model: Some(None)` clears it.
#[derive(Debug, Clone, Default)]
pub struct ThreadUpdate {
    pub title: Option<String>,
    pub pinned: Option<bool>,
    pub model: Option<Option<String>>,
}

/// Approval window attached to a parked call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalView {
    pub kind: String,
    pub expires_at: DateTime<Utc>,
    pub arguments_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirm_field: Option<String>,
    pub impact: String,
}

/// One tool call persisted with a message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct PersistedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
    pub risk: String,
    pub status: ToolCallStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approval: Option<ApprovalView>,
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
}

/// One version of a branched message. Siblings share a parent; the active
/// sibling is the one rendered on the current branch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct MessageVersion {
    pub id: Uuid,
    pub version: i32,
    pub is_active: bool,
    pub status: MessageStatus,
    pub created_at: DateTime<Utc>,
}

/// One persisted chat message with its tool calls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub id: Uuid,
    pub thread_id: Uuid,
    pub parent_message_id: Option<Uuid>,
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<PersistedToolCall>,
    pub tool_call_id: Option<String>,
    pub context_version: Option<String>,
    pub context_json: Option<Value>,
    pub model: Option<String>,
    pub reasoning_level: Option<String>,
    pub status: MessageStatus,
    pub error_detail: Option<String>,
    pub usage: Option<TokenUsage>,
    pub feedback: Option<Feedback>,
    /// 1-based position among siblings.
    pub version: i32,
    /// All versions in this message's sibling group, ordered by creation.
    #[serde(default)]
    pub versions: Vec<MessageVersion>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// Active branch of a thread.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadView {
    pub thread: Thread,
    pub messages: Vec<Message>,
    pub active_leaf_message_id: Option<Uuid>,
}

/// How a new turn attaches to the thread graph.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnTarget {
    /// Append a new turn.
    New,
    /// Replace a user message: a sibling with the same parent becomes active.
    Edit { message_id: Uuid },
    /// Regenerate an assistant answer: a sibling becomes active.
    Regenerate { message_id: Uuid },
}

/// Request to open a turn. Returns the user message and the pending assistant
/// placeholder.
#[derive(Debug, Clone)]
pub struct BeginTurn {
    pub thread_id: Option<Uuid>,
    pub target: TurnTarget,
    pub content: String,
    pub title: Option<String>,
    pub model: Option<String>,
    pub reasoning_level: Option<String>,
    pub context_version: Option<String>,
    pub context_json: Option<Value>,
}

/// Message pair created by [`ConversationStore::begin_turn`].
#[derive(Debug, Clone)]
pub struct StartedTurn {
    pub thread: Thread,
    pub user_message: Message,
    pub assistant_message: Message,
}

/// Terminal state written by the orchestrator.
#[derive(Debug, Clone)]
pub struct AssistantFinish<'a> {
    pub scope: &'a ScopeId,
    pub subject: &'a str,
    pub message_id: Uuid,
    pub content: &'a str,
    pub reasoning: Option<&'a str>,
    pub usage: Option<&'a TokenUsage>,
    pub status: MessageStatus,
    pub error_detail: Option<&'a str>,
}

/// Usage report for one thread.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct TurnUsage {
    pub total: TokenUsage,
    pub messages: Vec<UsageEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct UsageEntry {
    pub message_id: Uuid,
    pub model: Option<String>,
    pub reasoning_level: Option<String>,
    pub status: MessageStatus,
    pub usage: TokenUsage,
    pub created_at: DateTime<Utc>,
}

/// Conversation and settings surface.
#[async_trait::async_trait]
pub trait ConversationStore: Send + Sync {
    // Threads.
    async fn create_thread(
        &self,
        scope: &ScopeId,
        subject: &str,
        request: CreateThread,
    ) -> Result<Thread, StoreError>;
    async fn list_threads(&self, scope: &ScopeId, subject: &str)
    -> Result<Vec<Thread>, StoreError>;
    async fn get_thread(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<Option<Thread>, StoreError>;
    async fn update_thread(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
        update: ThreadUpdate,
    ) -> Result<Thread, StoreError>;
    async fn delete_thread(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<(), StoreError>;

    // Turns.
    async fn begin_turn(
        &self,
        scope: &ScopeId,
        subject: &str,
        request: BeginTurn,
    ) -> Result<StartedTurn, StoreError>;
    async fn finalize_assistant(&self, finish: AssistantFinish<'_>) -> Result<(), StoreError>;
    /// Marks every message still `pending`/`streaming` older than `cutoff` as
    /// `error` with `code`.
    async fn finalize_stale_turns(
        &self,
        cutoff: DateTime<Utc>,
        code: &str,
    ) -> Result<usize, StoreError>;
    async fn thread_view(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<ThreadView, StoreError>;
    async fn message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<Message, StoreError>;
    /// Switches the thread's active branch to `message` and everything below it.
    async fn activate_message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<(), StoreError>;
    async fn set_feedback(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
        rating: FeedbackRating,
        comment: Option<&str>,
    ) -> Result<Feedback, StoreError>;
    async fn clear_feedback(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<(), StoreError>;
    /// Provider history for one message: the parent chain from the root,
    /// annotated with wrapped tool results.
    async fn history_for_message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<Vec<ChatMessage>, StoreError>;
    async fn thread_usage(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<TurnUsage, StoreError>;
    /// Finds the user message anchoring the turn of an assistant message.
    async fn user_anchor_message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<Option<Uuid>, StoreError>;
}

// ---------------------------------------------------------------------------
// Approvals
// ---------------------------------------------------------------------------

/// Insert payload for one tool call row.
#[derive(Debug)]
pub struct NewToolCall<'a> {
    pub id: &'a str,
    pub thread_id: Uuid,
    pub message_id: Uuid,
    pub scope: &'a ScopeId,
    pub subject: &'a str,
    pub name: &'a str,
    pub risk: &'a str,
    pub arguments_json: &'a str,
    pub arguments_hash: &'a str,
    pub status: ToolCallStatus,
    pub auth_mode: Option<&'a str>,
    pub expires_at: Option<DateTime<Utc>>,
    pub approval_kind: Option<&'a str>,
    pub confirm_field: Option<&'a str>,
    pub impact: Option<&'a str>,
}

/// Insert payload for an approval decision row.
#[derive(Debug)]
pub struct NewToolApproval<'a> {
    pub id: &'a str,
    pub tool_call_id: &'a str,
    pub scope: &'a ScopeId,
    pub subject: &'a str,
    pub decision: &'a str,
    pub reason: Option<&'a str>,
    pub typed_confirm: Option<&'a str>,
    pub arguments_hash: &'a str,
    pub expires_at: DateTime<Utc>,
}

/// Tool call view used by the approval decision path and by one-shot
/// enforcement in the broker.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallSnapshot {
    pub id: String,
    pub scope_id: String,
    pub subject: String,
    pub thread_id: Uuid,
    pub message_id: Uuid,
    pub name: String,
    pub risk: String,
    pub status: String,
    pub approval_kind: Option<String>,
    pub approval_id: Option<String>,
    pub confirm_field: Option<String>,
    pub arguments: Value,
    pub arguments_hash: String,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Terminal tool call update.
#[derive(Debug)]
pub struct ToolCallCompletion<'a> {
    pub status: ToolCallStatus,
    pub result_json: Option<&'a Value>,
    pub error_code: Option<&'a str>,
    pub duration_ms: Option<i64>,
    pub approval_id: Option<&'a str>,
}

#[async_trait::async_trait]
pub trait ApprovalStore: Send + Sync {
    async fn record_tool_call(&self, call: NewToolCall<'_>) -> Result<(), StoreError>;
    async fn insert_tool_approval(&self, approval: NewToolApproval<'_>) -> Result<(), StoreError>;
    /// Transitions `awaiting_approval` -> `approved` and binds the approval id.
    /// Fails with `Conflict` when the row is not awaiting a decision (replay).
    async fn mark_tool_call_approved(&self, id: &str, approval_id: &str) -> Result<(), StoreError>;
    /// Transitions `awaiting_approval` -> `denied`. Fails with `Conflict` when
    /// the row is not awaiting a decision.
    async fn mark_tool_call_denied(&self, id: &str) -> Result<(), StoreError>;
    async fn mark_tool_call_running(
        &self,
        id: &str,
        auth_mode: Option<&str>,
        approval_id: Option<&str>,
    ) -> Result<(), StoreError>;
    async fn complete_tool_call(
        &self,
        id: &str,
        completion: ToolCallCompletion<'_>,
    ) -> Result<(), StoreError>;
    async fn get_tool_call(&self, id: &str) -> Result<Option<ToolCallSnapshot>, StoreError>;
    async fn count_recent_tool_calls(
        &self,
        scope: &ScopeId,
        subject: &str,
        name: &str,
        since: DateTime<Utc>,
    ) -> Result<u64, StoreError>;
    /// Marks parked calls whose window has passed as `expired`.
    async fn expire_tool_calls(&self, now: DateTime<Utc>) -> Result<usize, StoreError>;
}

/// In-memory settings map used by tests and by embedders that want to supply
/// static scope configuration.
#[derive(Debug, Default)]
pub struct StaticScopeSettings {
    entries: std::sync::Mutex<BTreeMap<String, ScopeSettings>>,
}

impl StaticScopeSettings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, scope: &ScopeId, settings: ScopeSettings) -> Self {
        self.entries
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(scope.to_string(), settings);
        self
    }
}

#[async_trait::async_trait]
impl ScopeSettingsStore for StaticScopeSettings {
    async fn get(&self, scope: &ScopeId) -> Result<Option<ScopeSettings>, StoreError> {
        Ok(self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(scope.as_str())
            .cloned())
    }

    async fn upsert(
        &self,
        scope: &ScopeId,
        update: ScopeSettingsUpdate,
    ) -> Result<ScopeSettings, StoreError> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let now = Utc::now();
        let entry = entries
            .entry(scope.to_string())
            .or_insert_with(|| ScopeSettings {
                provider_kind: ProviderKind::Openai,
                base_url: None,
                default_model: None,
                timeout_ms: 60_000,
                max_output_tokens: 1_200,
                api_key_present: false,
                profile: ProfileSelection::default(),
                extensions: Value::Object(serde_json::Map::new()),
                created_at: now,
                updated_at: None,
            });
        if let Some(kind) = update.provider_kind {
            entry.provider_kind = kind;
        }
        if let Some(base_url) = update.base_url {
            entry.base_url = (!base_url.trim().is_empty()).then_some(base_url);
        }
        if let Some(model) = update.default_model {
            entry.default_model = (!model.trim().is_empty()).then_some(model);
        }
        if let Some(timeout) = update.timeout_ms {
            entry.timeout_ms = timeout;
        }
        if let Some(tokens) = update.max_output_tokens {
            entry.max_output_tokens = tokens;
        }
        if let Some(present) = update.api_key_present {
            entry.api_key_present = present;
        }
        if let Some(profile) = update.profile {
            entry.profile = profile;
        }
        if let Some(extensions) = update.extensions {
            entry.extensions = extensions;
        }
        entry.updated_at = Some(now);
        Ok(entry.clone())
    }
}
