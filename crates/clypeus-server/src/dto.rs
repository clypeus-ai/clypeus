//! Wire DTOs and mapping from core types.

use chrono::{DateTime, Utc};
use clypeus_core::audit::AuditRecord;
use clypeus_core::models::{ChatMessage, ProviderKind, TokenUsage, ToolSpec};
use clypeus_core::provider::{ModelCatalog, ProbeReport};
use clypeus_core::store::{
    Feedback, Message, MessageVersion, PersistedToolCall, ScopeSettings, Thread, ThreadView,
    ToolCallStatus, TurnUsage,
};
use clypeus_core::tools::ToolError;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadDto {
    pub id: Uuid,
    pub scope_id: String,
    pub subject: String,
    pub title: String,
    pub pinned: bool,
    pub model: Option<String>,
    pub active_leaf_message_id: Option<Uuid>,
    pub created_at_utc: DateTime<Utc>,
    pub updated_at_utc: DateTime<Utc>,
}

impl From<Thread> for ThreadDto {
    fn from(thread: Thread) -> Self {
        Self {
            id: thread.id,
            scope_id: thread.scope_id,
            subject: thread.subject,
            title: thread.title,
            pinned: thread.pinned,
            model: thread.model,
            active_leaf_message_id: thread.active_leaf_message_id,
            created_at_utc: thread.created_at,
            updated_at_utc: thread.updated_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallErrorDto {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ApprovalDto {
    pub kind: String,
    pub expires_at_utc: DateTime<Utc>,
    pub arguments_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirm_field: Option<String>,
    pub impact: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallDto {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    pub risk: String,
    pub status: ToolCallStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ToolCallErrorDto>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approval: Option<ApprovalDto>,
    pub created_at_utc: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at_utc: Option<DateTime<Utc>>,
}

impl From<PersistedToolCall> for ToolCallDto {
    fn from(call: PersistedToolCall) -> Self {
        Self {
            id: call.id,
            name: call.name,
            arguments: call.arguments,
            risk: call.risk,
            status: call.status,
            result: call.result,
            error: call.error_code.map(|code| ToolCallErrorDto {
                message: ToolError::message_for_code(&code).to_string(),
                code,
            }),
            duration_ms: call.duration_ms,
            approval: call.approval.map(|approval| ApprovalDto {
                kind: approval.kind,
                expires_at_utc: approval.expires_at,
                arguments_hash: approval.arguments_hash,
                confirm_field: approval.confirm_field,
                impact: approval.impact,
            }),
            created_at_utc: call.created_at,
            completed_at_utc: call.completed_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct FeedbackDto {
    pub message_id: Uuid,
    pub rating: String,
    pub comment: Option<String>,
    pub created_at_utc: DateTime<Utc>,
    pub updated_at_utc: DateTime<Utc>,
}

impl From<Feedback> for FeedbackDto {
    fn from(feedback: Feedback) -> Self {
        Self {
            message_id: feedback.message_id,
            rating: feedback.rating.as_wire().to_string(),
            comment: feedback.comment,
            created_at_utc: feedback.created_at,
            updated_at_utc: feedback.updated_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct MessageVersionDto {
    pub id: Uuid,
    pub version: i32,
    pub is_active: bool,
    pub status: String,
    pub created_at_utc: DateTime<Utc>,
}

impl From<MessageVersion> for MessageVersionDto {
    fn from(version: MessageVersion) -> Self {
        Self {
            id: version.id,
            version: version.version,
            is_active: version.is_active,
            status: version.status.as_wire().to_string(),
            created_at_utc: version.created_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct MessageDto {
    pub id: Uuid,
    pub thread_id: Uuid,
    pub parent_message_id: Option<Uuid>,
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCallDto>,
    pub tool_call_id: Option<String>,
    pub context_version: Option<String>,
    pub context_json: Option<serde_json::Value>,
    pub model: Option<String>,
    /// Reasoning level applied to this turn; `null` when no explicit override
    /// was requested.
    pub reasoning_level: Option<String>,
    pub status: String,
    pub error_detail: Option<String>,
    pub usage: Option<TokenUsage>,
    pub feedback: Option<FeedbackDto>,
    pub version: i32,
    pub versions: Vec<MessageVersionDto>,
    pub created_at_utc: DateTime<Utc>,
    pub updated_at_utc: DateTime<Utc>,
    pub started_at_utc: Option<DateTime<Utc>>,
    pub completed_at_utc: Option<DateTime<Utc>>,
}

impl From<Message> for MessageDto {
    fn from(message: Message) -> Self {
        Self {
            id: message.id,
            thread_id: message.thread_id,
            parent_message_id: message.parent_message_id,
            role: message.role,
            content: message.content,
            reasoning_content: message.reasoning_content,
            tool_calls: message.tool_calls.into_iter().map(Into::into).collect(),
            tool_call_id: message.tool_call_id,
            context_version: message.context_version,
            context_json: message.context_json,
            model: message.model,
            reasoning_level: message.reasoning_level,
            status: message.status.as_wire().to_string(),
            error_detail: message.error_detail,
            usage: message.usage,
            feedback: message.feedback.map(Into::into),
            version: message.version,
            versions: message.versions.into_iter().map(Into::into).collect(),
            created_at_utc: message.created_at,
            updated_at_utc: message.updated_at,
            started_at_utc: message.started_at,
            completed_at_utc: message.completed_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadViewResponse {
    pub thread: ThreadDto,
    pub messages: Vec<MessageDto>,
    pub active_leaf_message_id: Option<Uuid>,
}

impl From<ThreadView> for ThreadViewResponse {
    fn from(view: ThreadView) -> Self {
        Self {
            thread: view.thread.into(),
            messages: view.messages.into_iter().map(Into::into).collect(),
            active_leaf_message_id: view.active_leaf_message_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ThreadListResponse {
    pub threads: Vec<ThreadDto>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct TurnResponse {
    pub thread: ThreadDto,
    pub user_message: MessageDto,
    pub assistant_message: MessageDto,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct CreateThreadRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub pinned: bool,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct UpdateThreadRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub pinned: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_explicit_option")]
    pub model: Option<Option<String>>,
}

fn deserialize_explicit_option<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    serde::Deserialize::deserialize(deserializer).map(Some)
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct PageContextDto {
    #[serde(default)]
    pub page: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub resource: Option<String>,
    #[serde(default)]
    pub status_text: Option<String>,
}

impl PageContextDto {
    pub fn to_value(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        if let Some(page) = self.page.as_deref().filter(|page| !page.is_empty()) {
            map.insert("page".into(), serde_json::Value::String(bound(page, 200)));
        }
        if let Some(path) = self.path.as_deref().filter(|path| !path.is_empty()) {
            map.insert("path".into(), serde_json::Value::String(bound(path, 500)));
        }
        if let Some(resource) = self
            .resource
            .as_deref()
            .filter(|resource| !resource.is_empty())
        {
            map.insert(
                "resource".into(),
                serde_json::Value::String(bound(resource, 200)),
            );
        }
        if let Some(status) = self
            .status_text
            .as_deref()
            .filter(|status| !status.is_empty())
        {
            map.insert(
                "statusText".into(),
                serde_json::Value::String(bound(status, 200)),
            );
        }
        serde_json::Value::Object(map)
    }
}

fn bound(value: &str, max: usize) -> String {
    value.chars().take(max).collect()
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct CreateTurnRequest {
    pub content: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub model: Option<String>,
    /// Open reasoning level. Must match one of the effective model's
    /// advertised levels verbatim; absent, empty, and `"default"` mean "no
    /// explicit override" and anything else is forwarded to the provider.
    #[serde(default)]
    pub reasoning_level: Option<String>,
    #[serde(default)]
    pub page_context: Option<PageContextDto>,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct EditTurnRequest {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub model: Option<String>,
    /// Open reasoning level. Must match one of the effective model's
    /// advertised levels verbatim; absent, empty, and `"default"` mean "no
    /// explicit override" and anything else is forwarded to the provider.
    #[serde(default)]
    pub reasoning_level: Option<String>,
    #[serde(default)]
    pub page_context: Option<PageContextDto>,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct RegenerateTurnRequest {
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub model: Option<String>,
    /// Open reasoning level. Must match one of the effective model's
    /// advertised levels verbatim; absent, empty, and `"default"` mean "no
    /// explicit override" and anything else is forwarded to the provider.
    #[serde(default)]
    pub reasoning_level: Option<String>,
    #[serde(default)]
    pub page_context: Option<PageContextDto>,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct SubmitApprovalRequest {
    pub decision: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub typed_confirm: Option<String>,
    pub arguments_hash: String,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct SetFeedbackRequest {
    pub rating: String,
    #[serde(default)]
    pub comment: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ScopeSettingsDto {
    pub scope_id: String,
    pub provider_kind: ProviderKind,
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub api_key_stored: bool,
    pub timeout_ms: i32,
    pub max_output_tokens: i32,
    pub profile: serde_json::Value,
    pub extensions: serde_json::Value,
    pub created_at_utc: DateTime<Utc>,
    pub updated_at_utc: Option<DateTime<Utc>>,
}

impl From<ScopeSettings> for ScopeSettingsDto {
    fn from(settings: ScopeSettings) -> Self {
        Self {
            scope_id: settings
                .extensions
                .get("scopeId")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string(),
            provider_kind: settings.provider_kind,
            base_url: settings.base_url,
            default_model: settings.default_model,
            api_key_stored: settings.api_key_present,
            timeout_ms: settings.timeout_ms,
            max_output_tokens: settings.max_output_tokens,
            profile: serde_json::to_value(&settings.profile).unwrap_or(serde_json::Value::Null),
            extensions: settings.extensions,
            created_at_utc: settings.created_at,
            updated_at_utc: settings.updated_at,
        }
    }
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct UpdateScopeSettingsRequest {
    #[serde(default)]
    pub provider_kind: Option<ProviderKind>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub timeout_ms: Option<i32>,
    #[serde(default)]
    pub max_output_tokens: Option<i32>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub clear_api_key: bool,
    /// Profile selection: `{"mode": "builtin"|"custom"|"disabled", ...}`.
    #[serde(default)]
    pub profile: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConnectionTestDto {
    pub succeeded: bool,
    pub model_count: Option<i32>,
    pub elapsed_ms: i64,
    pub checked_at_utc: DateTime<Utc>,
    pub error: Option<String>,
}

impl From<ProbeReport> for ConnectionTestDto {
    fn from(report: ProbeReport) -> Self {
        Self {
            succeeded: report.succeeded,
            model_count: report.model_count,
            elapsed_ms: report.elapsed_ms,
            checked_at_utc: report.checked_at_utc,
            error: report.error,
        }
    }
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelsPreviewRequest {
    pub base_url: String,
    pub api_key: String,
    pub provider_kind: ProviderKind,
}

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct CompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Open reasoning level forwarded to the provider verbatim; absent, empty,
    /// and `"default"` send no override.
    #[serde(default)]
    pub reasoning_level: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<i32>,
    #[serde(default)]
    pub tools: Option<Vec<ToolSpec>>,
    #[serde(default)]
    pub tool_choice: Option<String>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct AuditListResponse {
    pub entries: Vec<AuditRecord>,
    pub limit: i64,
    pub offset: i64,
    pub total: i64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct UsageResponse {
    pub total: TokenUsage,
    pub messages: Vec<UsageEntryDto>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct UsageEntryDto {
    pub message_id: Uuid,
    pub model: Option<String>,
    /// Reasoning level applied to this turn; `null` when no explicit override
    /// was requested.
    pub reasoning_level: Option<String>,
    pub status: String,
    pub usage: TokenUsage,
    pub created_at_utc: DateTime<Utc>,
}

impl From<TurnUsage> for UsageResponse {
    fn from(usage: TurnUsage) -> Self {
        Self {
            total: usage.total,
            messages: usage
                .messages
                .into_iter()
                .map(|entry| UsageEntryDto {
                    message_id: entry.message_id,
                    model: entry.model,
                    reasoning_level: entry.reasoning_level,
                    status: entry.status.as_wire().to_string(),
                    usage: entry.usage,
                    created_at_utc: entry.created_at,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelsResponse {
    pub models: Vec<clypeus_core::provider::ModelCapability>,
}

impl From<ModelCatalog> for ModelsResponse {
    fn from(catalog: ModelCatalog) -> Self {
        Self {
            models: catalog.models,
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ToolCatalogEntryDto {
    pub name: String,
    pub description: String,
    pub schema: serde_json::Value,
    pub required_scopes: Vec<String>,
    pub risk: String,
    pub approval: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirm_field: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ToolCatalogResponse {
    pub tools: Vec<ToolCatalogEntryDto>,
}
