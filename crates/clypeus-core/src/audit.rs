//! Append-only audit records and the audit store abstraction.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::principal::ScopeId;
use crate::store::StoreError;

/// What kind of audited item produced a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum AuditItemKind {
    Tool,
    Function,
}

impl AuditItemKind {
    pub fn as_wire(self) -> &'static str {
        match self {
            Self::Tool => "tool",
            Self::Function => "function",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "tool" => Some(Self::Tool),
            "function" => Some(Self::Function),
            _ => None,
        }
    }
}

/// One append-only audit record. Arguments are stored redacted; the hash binds
/// the record to the exact canonical arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct AuditRecord {
    pub id: Uuid,
    pub scope_id: String,
    pub subject: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    pub item_name: String,
    pub item_kind: AuditItemKind,
    pub risk: String,
    pub arguments_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments_redacted: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scopes_used: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub egress_service: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub egress_path_template: Option<String>,
    pub outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downstream_status: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_bytes: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approval_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Filter for audit reads.
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    pub limit: i64,
    pub offset: i64,
    pub item_kind: Option<AuditItemKind>,
    pub item_name: Option<String>,
    pub outcome: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
}

impl AuditQuery {
    pub fn with_limit(mut self, limit: i64) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_offset(mut self, offset: i64) -> Self {
        self.offset = offset;
        self
    }
}

/// One page of audit records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct AuditPage {
    pub entries: Vec<AuditRecord>,
    pub limit: i64,
    pub offset: i64,
    pub total: i64,
}

/// Append-only sink.
#[async_trait::async_trait]
pub trait AuditSink: Send + Sync {
    async fn append(&self, record: AuditRecord) -> Result<(), StoreError>;
}

/// Read/export/retention surface.
#[async_trait::async_trait]
pub trait AuditReader: Send + Sync {
    async fn page(&self, scope: &ScopeId, query: AuditQuery) -> Result<AuditPage, StoreError>;
    async fn export(
        &self,
        scope: &ScopeId,
        query: AuditQuery,
    ) -> Result<Vec<AuditRecord>, StoreError>;
    /// Deletes records strictly older than `cutoff`. The only delete path on
    /// the audit store.
    async fn purge_before(&self, cutoff: DateTime<Utc>) -> Result<usize, StoreError>;
}
