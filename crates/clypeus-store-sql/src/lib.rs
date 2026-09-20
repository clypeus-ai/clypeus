//! Shared SQL implementation of the Clypeus store traits.
//!
//! Both `clypeus-store-postgres` and `clypeus-store-sqlite` build on this
//! crate. Queries are written with `?` placeholders and rewritten to `$n` for
//! PostgreSQL; identifiers, timestamps, and JSON values are stored as text so
//! the two backends share one schema shape.
//!
//! Every read and write is filtered by scope and subject. Branching is a
//! parent-pointer graph: the thread points at one active leaf, and each
//! message records the parent it answered.

use std::collections::HashMap;

use chrono::{DateTime, SecondsFormat, Utc};
use clypeus_core::models::{ChatMessage, ChatRole, ProviderKind, TokenUsage, ToolCall};
use clypeus_core::principal::ScopeId;
use clypeus_core::profile::ProfileSelection;
use clypeus_core::store::{
    ApprovalStore, ApprovalView, AssistantFinish, BeginTurn, ConversationStore, CreateThread,
    Feedback, FeedbackRating, Message, MessageStatus, MessageVersion, NewToolApproval, NewToolCall,
    PersistedToolCall, ScopeSettings, ScopeSettingsStore, ScopeSettingsUpdate, StartedTurn,
    StoreError, Thread, ThreadUpdate, ThreadView, ToolCallCompletion, ToolCallSnapshot,
    ToolCallStatus, TurnTarget, TurnUsage, UsageEntry,
};
use serde_json::{Map, Value};
use sqlx::any::install_default_drivers;
use sqlx::any::{AnyPoolOptions, AnyRow};
use sqlx::{Any, AnyPool, Executor, Row};
use uuid::Uuid;

/// SQL backend dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlDialect {
    Postgres,
    Sqlite,
}

/// Shared SQL store. Each backend crate owns its migrations and passes the
/// migrator in at construction.
#[derive(Debug, Clone)]
pub struct SqlStore {
    pool: AnyPool,
    dialect: SqlDialect,
}

#[derive(Debug, Clone)]
enum SqlValue {
    Text(String),
    Int(i64),
    Bool(bool),
    /// Null for a text column. PostgreSQL needs a typed null per column, so
    /// text and integer nulls are distinct variants.
    NullText,
    /// Null for an integer column.
    NullInt,
}

impl From<String> for SqlValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for SqlValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<i64> for SqlValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<bool> for SqlValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<Option<String>> for SqlValue {
    fn from(value: Option<String>) -> Self {
        value.map_or(Self::NullText, Self::Text)
    }
}

impl From<Option<&str>> for SqlValue {
    fn from(value: Option<&str>) -> Self {
        value.map_or(Self::NullText, |value| Self::Text(value.to_string()))
    }
}

impl From<Option<i64>> for SqlValue {
    fn from(value: Option<i64>) -> Self {
        value.map_or(Self::NullInt, Self::Int)
    }
}

impl From<Option<i32>> for SqlValue {
    fn from(value: Option<i32>) -> Self {
        value.map_or(Self::NullInt, |value| Self::Int(i64::from(value)))
    }
}

impl From<i32> for SqlValue {
    fn from(value: i32) -> Self {
        Self::Int(i64::from(value))
    }
}

fn backend(error: sqlx::Error) -> StoreError {
    StoreError::Backend(error.to_string())
}

fn placeholders(query: &str, dialect: SqlDialect) -> String {
    match dialect {
        SqlDialect::Sqlite => query.to_string(),
        SqlDialect::Postgres => {
            let mut out = String::with_capacity(query.len() + 16);
            let mut index = 0usize;
            for character in query.chars() {
                if character == '?' {
                    index += 1;
                    out.push('$');
                    out.push_str(&index.to_string());
                } else {
                    out.push(character);
                }
            }
            out
        }
    }
}

async fn exec<'c, E>(
    executor: E,
    dialect: SqlDialect,
    query: &str,
    values: Vec<SqlValue>,
) -> Result<u64, StoreError>
where
    E: Executor<'c, Database = Any>,
{
    let text = placeholders(query, dialect);
    let mut statement = sqlx::query(&text);
    for value in values {
        statement = match value {
            SqlValue::Text(value) => statement.bind(value),
            SqlValue::Int(value) => statement.bind(value),
            SqlValue::Bool(value) => statement.bind(value),
            SqlValue::NullText => statement.bind(Option::<String>::None),
            SqlValue::NullInt => statement.bind(Option::<i64>::None),
        };
    }
    statement
        .execute(executor)
        .await
        .map(|result| result.rows_affected())
        .map_err(backend)
}

async fn fetch_all<'c, E>(
    executor: E,
    dialect: SqlDialect,
    query: &str,
    values: Vec<SqlValue>,
) -> Result<Vec<AnyRow>, StoreError>
where
    E: Executor<'c, Database = Any>,
{
    let text = placeholders(query, dialect);
    let mut statement = sqlx::query(&text);
    for value in values {
        statement = match value {
            SqlValue::Text(value) => statement.bind(value),
            SqlValue::Int(value) => statement.bind(value),
            SqlValue::Bool(value) => statement.bind(value),
            SqlValue::NullText => statement.bind(Option::<String>::None),
            SqlValue::NullInt => statement.bind(Option::<i64>::None),
        };
    }
    statement.fetch_all(executor).await.map_err(backend)
}

async fn fetch_optional<'c, E>(
    executor: E,
    dialect: SqlDialect,
    query: &str,
    values: Vec<SqlValue>,
) -> Result<Option<AnyRow>, StoreError>
where
    E: Executor<'c, Database = Any>,
{
    let text = placeholders(query, dialect);
    let mut statement = sqlx::query(&text);
    for value in values {
        statement = match value {
            SqlValue::Text(value) => statement.bind(value),
            SqlValue::Int(value) => statement.bind(value),
            SqlValue::Bool(value) => statement.bind(value),
            SqlValue::NullText => statement.bind(Option::<String>::None),
            SqlValue::NullInt => statement.bind(Option::<i64>::None),
        };
    }
    statement.fetch_optional(executor).await.map_err(backend)
}

// ---------------------------------------------------------------------------
// Row decoding helpers
// ---------------------------------------------------------------------------

fn column_text(row: &AnyRow, column: &str) -> Result<String, StoreError> {
    row.try_get::<String, _>(column)
        .map_err(|error| StoreError::Backend(format!("column {column}: {error}")))
}

fn column_opt_text(row: &AnyRow, column: &str) -> Result<Option<String>, StoreError> {
    row.try_get::<Option<String>, _>(column)
        .map_err(|error| StoreError::Backend(format!("column {column}: {error}")))
}

fn column_int(row: &AnyRow, column: &str) -> Result<i64, StoreError> {
    row.try_get::<i64, _>(column)
        .map_err(|error| StoreError::Backend(format!("column {column}: {error}")))
}

fn column_opt_int(row: &AnyRow, column: &str) -> Result<Option<i64>, StoreError> {
    row.try_get::<Option<i64>, _>(column)
        .map_err(|error| StoreError::Backend(format!("column {column}: {error}")))
}

fn parse_uuid(value: &str) -> Result<Uuid, StoreError> {
    Uuid::parse_str(value).map_err(|error| StoreError::Backend(format!("uuid: {error}")))
}

fn parse_opt_uuid(value: Option<String>) -> Result<Option<Uuid>, StoreError> {
    value.map(|value| parse_uuid(&value)).transpose()
}

fn timestamp(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn parse_timestamp(value: &str) -> Result<DateTime<Utc>, StoreError> {
    DateTime::parse_from_rfc3339(value)
        .map(|parsed| parsed.with_timezone(&Utc))
        .map_err(|error| StoreError::Backend(format!("timestamp: {error}")))
}

fn parse_opt_timestamp(value: Option<String>) -> Result<Option<DateTime<Utc>>, StoreError> {
    value.map(|value| parse_timestamp(&value)).transpose()
}

fn parse_json(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or(Value::Null)
}

fn parse_status(value: &str) -> MessageStatus {
    MessageStatus::parse(value).unwrap_or(MessageStatus::Error)
}

fn status_int(status: MessageStatus) -> bool {
    matches!(status, MessageStatus::Pending | MessageStatus::Streaming)
}

impl SqlStore {
    /// Connects, installs the sqlx `Any` drivers, and runs migrations.
    pub async fn connect(
        url: &str,
        dialect: SqlDialect,
        migrator: &sqlx::migrate::Migrator,
    ) -> Result<Self, StoreError> {
        install_default_drivers();
        // SQLite connections each see their own in-memory database unless the
        // URL uses a shared cache, and file databases serialize writers
        // anyway; one connection keeps both behaviors predictable.
        let max_connections = if dialect == SqlDialect::Sqlite { 1 } else { 10 };
        let pool = AnyPoolOptions::new()
            .max_connections(max_connections)
            .connect(url)
            .await
            .map_err(backend)?;
        migrator.run(&pool).await.map_err(|error| {
            tracing::error!(%error, "store migration failed");
            StoreError::Backend(format!("migration failed: {error}"))
        })?;
        Ok(Self { pool, dialect })
    }

    pub fn pool(&self) -> &AnyPool {
        &self.pool
    }

    pub fn dialect(&self) -> SqlDialect {
        self.dialect
    }

    async fn exec(&self, query: &str, values: Vec<SqlValue>) -> Result<u64, StoreError> {
        exec(&self.pool, self.dialect, query, values).await
    }

    async fn fetch_all(
        &self,
        query: &str,
        values: Vec<SqlValue>,
    ) -> Result<Vec<AnyRow>, StoreError> {
        fetch_all(&self.pool, self.dialect, query, values).await
    }

    async fn fetch_optional(
        &self,
        query: &str,
        values: Vec<SqlValue>,
    ) -> Result<Option<AnyRow>, StoreError> {
        fetch_optional(&self.pool, self.dialect, query, values).await
    }

    async fn load_thread_row(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<Option<Thread>, StoreError> {
        let row = self
            .fetch_optional(
                "SELECT * FROM clypeus_threads WHERE id = ? AND scope_id = ? AND subject = ?",
                vec![
                    thread.to_string().into(),
                    scope.to_string().into(),
                    subject.into(),
                ],
            )
            .await?;
        row.as_ref().map(thread_from_row).transpose()
    }

    /// Loads one message row plus its sibling versions, feedback, and tool
    /// calls.
    async fn load_message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message_id: Uuid,
    ) -> Result<Message, StoreError> {
        let row = self
            .fetch_optional(
                "SELECT * FROM clypeus_messages WHERE id = ? AND scope_id = ? AND subject = ?",
                vec![
                    message_id.to_string().into(),
                    scope.to_string().into(),
                    subject.into(),
                ],
            )
            .await?
            .ok_or(StoreError::NotFound)?;
        let mut message = self.message_from_row(&row).await?;
        message.versions = self.sibling_versions(scope, subject, &message).await?;
        Ok(message)
    }

    async fn message_from_row(&self, row: &AnyRow) -> Result<Message, StoreError> {
        let id = parse_uuid(&column_text(row, "id")?)?;
        let thread_id = parse_uuid(&column_text(row, "thread_id")?)?;
        let scope = ScopeId::new(column_text(row, "scope_id")?);
        let subject = column_text(row, "subject")?;
        let usage = column_opt_text(row, "usage_json")?
            .map(|raw| serde_json::from_str::<TokenUsage>(&raw))
            .transpose()
            .map_err(|error| StoreError::Backend(error.to_string()))?;
        let feedback = self.feedback_for(&scope, &subject, id).await?;
        let tool_calls = self.tool_calls_for(&scope, &subject, id).await?;
        Ok(Message {
            id,
            thread_id,
            parent_message_id: parse_opt_uuid(column_opt_text(row, "parent_message_id")?)?,
            role: column_text(row, "role")?,
            content: column_text(row, "content")?,
            reasoning_content: column_opt_text(row, "reasoning_content")?,
            tool_calls,
            tool_call_id: column_opt_text(row, "tool_call_id")?,
            context_version: column_opt_text(row, "context_version")?,
            context_json: column_opt_text(row, "context_json")?.map(|raw| parse_json(&raw)),
            model: column_opt_text(row, "model")?,
            reasoning_level: column_opt_text(row, "reasoning_level")?,
            status: parse_status(&column_text(row, "status")?),
            error_detail: column_opt_text(row, "error_detail")?,
            usage,
            feedback,
            version: column_int(row, "version")? as i32,
            versions: Vec::new(),
            created_at: parse_timestamp(&column_text(row, "created_at")?)?,
            updated_at: parse_timestamp(&column_text(row, "updated_at")?)?,
            started_at: parse_opt_timestamp(column_opt_text(row, "started_at")?)?,
            completed_at: parse_opt_timestamp(column_opt_text(row, "completed_at")?)?,
        })
    }

    async fn feedback_for(
        &self,
        scope: &ScopeId,
        subject: &str,
        message_id: Uuid,
    ) -> Result<Option<Feedback>, StoreError> {
        let row = self
            .fetch_optional(
                "SELECT * FROM clypeus_feedback WHERE message_id = ? AND scope_id = ? AND subject = ?",
                vec![
                    message_id.to_string().into(),
                    scope.to_string().into(),
                    subject.into(),
                ],
            )
            .await?;
        row.as_ref().map(feedback_from_row).transpose()
    }

    async fn tool_calls_for(
        &self,
        scope: &ScopeId,
        subject: &str,
        message_id: Uuid,
    ) -> Result<Vec<PersistedToolCall>, StoreError> {
        let rows = self
            .fetch_all(
                "SELECT * FROM clypeus_tool_calls WHERE message_id = ? AND scope_id = ? AND subject = ? ORDER BY created_at",
                vec![
                    message_id.to_string().into(),
                    scope.to_string().into(),
                    subject.into(),
                ],
            )
            .await?;
        rows.iter().map(tool_call_from_row).collect()
    }

    async fn sibling_versions(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: &Message,
    ) -> Result<Vec<MessageVersion>, StoreError> {
        let (query, values) = match message.parent_message_id {
            Some(parent) => (
                "SELECT id, version, status, created_at FROM clypeus_messages WHERE thread_id = ? AND parent_message_id = ? AND scope_id = ? AND subject = ? ORDER BY created_at",
                vec![
                    message.thread_id.to_string().into(),
                    parent.to_string().into(),
                    scope.to_string().into(),
                    subject.into(),
                ],
            ),
            None => (
                "SELECT id, version, status, created_at FROM clypeus_messages WHERE thread_id = ? AND parent_message_id IS NULL AND scope_id = ? AND subject = ? ORDER BY created_at",
                vec![
                    message.thread_id.to_string().into(),
                    scope.to_string().into(),
                    subject.into(),
                ],
            ),
        };
        let rows = self.fetch_all(query, values).await?;
        rows.iter()
            .map(|row| {
                let id = parse_uuid(&column_text(row, "id")?)?;
                Ok(MessageVersion {
                    id,
                    version: column_int(row, "version")? as i32,
                    is_active: id == message.id,
                    status: parse_status(&column_text(row, "status")?),
                    created_at: parse_timestamp(&column_text(row, "created_at")?)?,
                })
            })
            .collect()
    }
}

fn thread_from_row(row: &AnyRow) -> Result<Thread, StoreError> {
    Ok(Thread {
        id: parse_uuid(&column_text(row, "id")?)?,
        scope_id: column_text(row, "scope_id")?,
        subject: column_text(row, "subject")?,
        title: column_text(row, "title")?,
        pinned: column_int(row, "pinned")? != 0,
        model: column_opt_text(row, "model")?,
        active_leaf_message_id: parse_opt_uuid(column_opt_text(row, "active_leaf_message_id")?)?,
        created_at: parse_timestamp(&column_text(row, "created_at")?)?,
        updated_at: parse_timestamp(&column_text(row, "updated_at")?)?,
    })
}

fn feedback_from_row(row: &AnyRow) -> Result<Feedback, StoreError> {
    Ok(Feedback {
        message_id: parse_uuid(&column_text(row, "message_id")?)?,
        rating: FeedbackRating::parse(&column_text(row, "rating")?).unwrap_or(FeedbackRating::Up),
        comment: column_opt_text(row, "comment")?,
        created_at: parse_timestamp(&column_text(row, "created_at")?)?,
        updated_at: parse_timestamp(&column_text(row, "updated_at")?)?,
    })
}

fn tool_call_from_row(row: &AnyRow) -> Result<PersistedToolCall, StoreError> {
    let status =
        ToolCallStatus::parse(&column_text(row, "status")?).unwrap_or(ToolCallStatus::Failed);
    let approval_kind = column_opt_text(row, "approval_kind")?;
    let expires_at = parse_opt_timestamp(column_opt_text(row, "expires_at")?)?;
    let approval = approval_kind.map(|kind| ApprovalView {
        kind,
        expires_at: expires_at.unwrap_or_else(Utc::now),
        arguments_hash: column_text(row, "arguments_hash").unwrap_or_default(),
        confirm_field: column_opt_text(row, "confirm_field").unwrap_or_default(),
        impact: column_opt_text(row, "impact")
            .unwrap_or_default()
            .unwrap_or_default(),
    });
    Ok(PersistedToolCall {
        id: column_text(row, "id")?,
        name: column_text(row, "tool_name")?,
        arguments: parse_json(&column_text(row, "arguments_json")?),
        risk: column_text(row, "risk")?,
        status,
        result: column_opt_text(row, "result_json")?.map(|raw| parse_json(&raw)),
        error_code: column_opt_text(row, "error_code")?,
        duration_ms: column_opt_int(row, "duration_ms")?,
        approval,
        created_at: parse_timestamp(&column_text(row, "created_at")?)?,
        completed_at: parse_opt_timestamp(column_opt_text(row, "completed_at")?)?,
    })
}

#[async_trait::async_trait]
impl clypeus_core::store::Readiness for SqlStore {
    async fn check_ready(&self) -> Result<(), String> {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .map(|_| ())
            .map_err(|error| format!("database: {error}"))
    }
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl ScopeSettingsStore for SqlStore {
    async fn get(&self, scope: &ScopeId) -> Result<Option<ScopeSettings>, StoreError> {
        let row = self
            .fetch_optional(
                "SELECT * FROM clypeus_settings WHERE scope_id = ?",
                vec![scope.to_string().into()],
            )
            .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        let profile = profile_from_columns(
            &column_text(&row, "profile_mode")?,
            column_opt_text(&row, "profile_custom")?,
        );
        Ok(Some(ScopeSettings {
            provider_kind: ProviderKind::parse(&column_text(&row, "provider_kind")?)
                .unwrap_or(ProviderKind::Openai),
            base_url: column_opt_text(&row, "base_url")?,
            default_model: column_opt_text(&row, "default_model")?,
            timeout_ms: column_int(&row, "timeout_ms")? as i32,
            max_output_tokens: column_int(&row, "max_output_tokens")? as i32,
            api_key_present: column_int(&row, "api_key_present")? != 0,
            profile,
            extensions: column_opt_text(&row, "extensions")?
                .map(|raw| parse_json(&raw))
                .unwrap_or(Value::Object(Map::new())),
            created_at: parse_timestamp(&column_text(&row, "created_at")?)?,
            updated_at: parse_opt_timestamp(column_opt_text(&row, "updated_at")?)?,
        }))
    }

    async fn upsert(
        &self,
        scope: &ScopeId,
        update: ScopeSettingsUpdate,
    ) -> Result<ScopeSettings, StoreError> {
        let existing = self.get(scope).await?;
        let now = Utc::now();
        let mut settings = existing.unwrap_or(ScopeSettings {
            provider_kind: ProviderKind::Openai,
            base_url: None,
            default_model: None,
            timeout_ms: 60_000,
            max_output_tokens: 1_200,
            api_key_present: false,
            profile: ProfileSelection::default(),
            extensions: Value::Object(Map::new()),
            created_at: now,
            updated_at: None,
        });
        if let Some(kind) = update.provider_kind {
            settings.provider_kind = kind;
        }
        if let Some(base_url) = update.base_url {
            settings.base_url = (!base_url.trim().is_empty()).then_some(base_url);
        }
        if let Some(model) = update.default_model {
            settings.default_model = (!model.trim().is_empty()).then_some(model);
        }
        if let Some(timeout) = update.timeout_ms {
            settings.timeout_ms = timeout.clamp(1_000, 600_000);
        }
        if let Some(tokens) = update.max_output_tokens {
            settings.max_output_tokens = tokens.clamp(64, 200_000);
        }
        if let Some(present) = update.api_key_present {
            settings.api_key_present = present;
        }
        if let Some(profile) = update.profile {
            settings.profile = profile;
        }
        if let Some(extensions) = update.extensions {
            settings.extensions = extensions;
        }
        settings.updated_at = Some(now);
        self.exec(
            "INSERT INTO clypeus_settings (
                scope_id, provider_kind, base_url, default_model, timeout_ms,
                max_output_tokens, api_key_present, profile_mode, profile_custom,
                extensions, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_id) DO UPDATE SET
                provider_kind = excluded.provider_kind,
                base_url = excluded.base_url,
                default_model = excluded.default_model,
                timeout_ms = excluded.timeout_ms,
                max_output_tokens = excluded.max_output_tokens,
                api_key_present = excluded.api_key_present,
                profile_mode = excluded.profile_mode,
                profile_custom = excluded.profile_custom,
                extensions = excluded.extensions,
                updated_at = excluded.updated_at",
            vec![
                scope.to_string().into(),
                settings.provider_kind.as_wire().into(),
                settings.base_url.clone().into(),
                settings.default_model.clone().into(),
                i64::from(settings.timeout_ms).into(),
                i64::from(settings.max_output_tokens).into(),
                i64::from(settings.api_key_present).into(),
                profile_mode(&settings.profile).into(),
                profile_custom(&settings.profile).into(),
                serde_json::to_string(&settings.extensions)
                    .map_err(|error| StoreError::Backend(error.to_string()))?
                    .into(),
                timestamp(settings.created_at).into(),
                settings.updated_at.map(timestamp).into(),
            ],
        )
        .await?;
        Ok(settings)
    }
}

fn profile_mode(profile: &ProfileSelection) -> String {
    profile.mode_wire().to_string()
}

fn profile_custom(profile: &ProfileSelection) -> Option<String> {
    match profile {
        ProfileSelection::Builtin { custom } => custom.clone(),
        ProfileSelection::Custom(prompt) => Some(prompt.clone()),
        ProfileSelection::Disabled => None,
    }
}

fn profile_from_columns(mode: &str, custom: Option<String>) -> ProfileSelection {
    match mode {
        "builtin" => ProfileSelection::Builtin { custom },
        "custom" => ProfileSelection::Custom(custom.unwrap_or_default()),
        _ => ProfileSelection::Disabled,
    }
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl ConversationStore for SqlStore {
    async fn create_thread(
        &self,
        scope: &ScopeId,
        subject: &str,
        request: CreateThread,
    ) -> Result<Thread, StoreError> {
        let now = Utc::now();
        let thread = Thread {
            id: Uuid::new_v4(),
            scope_id: scope.to_string(),
            subject: subject.to_string(),
            title: request
                .title
                .filter(|title| !title.trim().is_empty())
                .unwrap_or_else(|| "New chat".to_string()),
            pinned: request.pinned,
            model: request.model.filter(|model| !model.trim().is_empty()),
            active_leaf_message_id: None,
            created_at: now,
            updated_at: now,
        };
        self.exec(
            "INSERT INTO clypeus_threads (
                id, scope_id, subject, title, pinned, model, active_leaf_message_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            vec![
                thread.id.to_string().into(),
                thread.scope_id.clone().into(),
                thread.subject.clone().into(),
                thread.title.clone().into(),
                i64::from(thread.pinned).into(),
                thread.model.clone().into(),
                thread
                    .active_leaf_message_id
                    .map(|id| id.to_string())
                    .into(),
                timestamp(thread.created_at).into(),
                timestamp(thread.updated_at).into(),
            ],
        )
        .await?;
        Ok(thread)
    }

    async fn list_threads(
        &self,
        scope: &ScopeId,
        subject: &str,
    ) -> Result<Vec<Thread>, StoreError> {
        let rows = self
            .fetch_all(
                "SELECT * FROM clypeus_threads WHERE scope_id = ? AND subject = ?
                 ORDER BY pinned DESC, updated_at DESC",
                vec![scope.to_string().into(), subject.into()],
            )
            .await?;
        rows.iter().map(thread_from_row).collect()
    }

    async fn get_thread(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<Option<Thread>, StoreError> {
        self.load_thread_row(scope, subject, thread).await
    }

    async fn update_thread(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
        update: ThreadUpdate,
    ) -> Result<Thread, StoreError> {
        let mut existing = self
            .load_thread_row(scope, subject, thread)
            .await?
            .ok_or(StoreError::NotFound)?;
        if let Some(title) = update.title.filter(|title| !title.trim().is_empty()) {
            existing.title = title;
        }
        if let Some(pinned) = update.pinned {
            existing.pinned = pinned;
        }
        if let Some(model) = update.model {
            existing.model = model.filter(|model| !model.trim().is_empty());
        }
        existing.updated_at = Utc::now();
        self.exec(
            "UPDATE clypeus_threads SET title = ?, pinned = ?, model = ?, updated_at = ?
             WHERE id = ? AND scope_id = ? AND subject = ?",
            vec![
                existing.title.clone().into(),
                i64::from(existing.pinned).into(),
                existing.model.clone().into(),
                timestamp(existing.updated_at).into(),
                thread.to_string().into(),
                scope.to_string().into(),
                subject.into(),
            ],
        )
        .await?;
        Ok(existing)
    }

    async fn delete_thread(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<(), StoreError> {
        if self
            .load_thread_row(scope, subject, thread)
            .await?
            .is_none()
        {
            return Err(StoreError::NotFound);
        }
        let scope_text = scope.to_string();
        let subject_text = subject.to_string();
        let thread_text = thread.to_string();
        let mut transaction = self.pool.begin().await.map_err(backend)?;
        for query in [
            "DELETE FROM clypeus_feedback WHERE scope_id = ? AND subject = ? AND message_id IN (SELECT id FROM clypeus_messages WHERE thread_id = ?)",
            "DELETE FROM clypeus_tool_approvals WHERE scope_id = ? AND subject = ? AND tool_call_id IN (SELECT id FROM clypeus_tool_calls WHERE thread_id = ?)",
            "DELETE FROM clypeus_tool_calls WHERE scope_id = ? AND subject = ? AND thread_id = ?",
            "DELETE FROM clypeus_messages WHERE scope_id = ? AND subject = ? AND thread_id = ?",
        ] {
            exec(
                &mut *transaction,
                self.dialect,
                query,
                vec![
                    scope_text.clone().into(),
                    subject_text.clone().into(),
                    thread_text.clone().into(),
                ],
            )
            .await?;
        }
        exec(
            &mut *transaction,
            self.dialect,
            "DELETE FROM clypeus_threads WHERE id = ? AND scope_id = ? AND subject = ?",
            vec![thread_text.into(), scope_text.into(), subject_text.into()],
        )
        .await?;
        transaction.commit().await.map_err(backend)?;
        Ok(())
    }

    async fn begin_turn(
        &self,
        scope: &ScopeId,
        subject: &str,
        request: BeginTurn,
    ) -> Result<StartedTurn, StoreError> {
        if !matches!(request.target, TurnTarget::Regenerate { .. })
            && request.content.trim().is_empty()
        {
            return Err(StoreError::Validation("content is required.".into()));
        }
        let scope_text = scope.to_string();
        let subject_text = subject.to_string();
        let now = Utc::now();

        let mut transaction = self.pool.begin().await.map_err(backend)?;

        // Resolve or create the thread.
        let mut thread = match request.thread_id {
            Some(thread_id) => {
                let row = fetch_optional(
                    &mut *transaction,
                    self.dialect,
                    "SELECT * FROM clypeus_threads WHERE id = ? AND scope_id = ? AND subject = ?",
                    vec![
                        thread_id.to_string().into(),
                        scope_text.clone().into(),
                        subject_text.clone().into(),
                    ],
                )
                .await?
                .ok_or(StoreError::NotFound)?;
                thread_from_row(&row)?
            }
            None => {
                let thread = Thread {
                    id: Uuid::new_v4(),
                    scope_id: scope_text.clone(),
                    subject: subject_text.clone(),
                    title: derive_title(&request.content),
                    pinned: false,
                    model: request.model.clone(),
                    active_leaf_message_id: None,
                    created_at: now,
                    updated_at: now,
                };
                exec(
                    &mut *transaction,
                    self.dialect,
                    "INSERT INTO clypeus_threads (
                        id, scope_id, subject, title, pinned, model, active_leaf_message_id,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    vec![
                        thread.id.to_string().into(),
                        thread.scope_id.clone().into(),
                        thread.subject.clone().into(),
                        thread.title.clone().into(),
                        i64::from(thread.pinned).into(),
                        thread.model.clone().into(),
                        SqlValue::NullText,
                        timestamp(thread.created_at).into(),
                        timestamp(thread.updated_at).into(),
                    ],
                )
                .await?;
                thread
            }
        };

        let (user_message, assistant_parent) = match &request.target {
            TurnTarget::New => {
                let parent = thread.active_leaf_message_id;
                let version =
                    sibling_version_in_tx(&mut transaction, self.dialect, thread.id, parent)
                        .await?;
                let message = new_user_message(&request, thread.id, parent, version, now);
                exec(
                    &mut *transaction,
                    self.dialect,
                    message_insert_sql(),
                    message_values(&message, &scope_text, &subject_text)?,
                )
                .await?;
                let parent = message.id;
                (message, parent)
            }
            TurnTarget::Edit { message_id } => {
                let row = fetch_optional(
                    &mut *transaction,
                    self.dialect,
                    "SELECT * FROM clypeus_messages WHERE id = ? AND thread_id = ? AND scope_id = ? AND subject = ?",
                    vec![
                        message_id.to_string().into(),
                        thread.id.to_string().into(),
                        scope_text.clone().into(),
                        subject_text.clone().into(),
                    ],
                )
                .await?
                .ok_or(StoreError::NotFound)?;
                let previous = message_base_from_row(&row)?;
                if previous.role != "user" {
                    return Err(StoreError::Validation(
                        "only user messages can be edited.".into(),
                    ));
                }
                let version = sibling_version_in_tx(
                    &mut transaction,
                    self.dialect,
                    thread.id,
                    previous.parent_message_id,
                )
                .await?;
                let message = new_user_message(
                    &request,
                    thread.id,
                    previous.parent_message_id,
                    version,
                    now,
                );
                exec(
                    &mut *transaction,
                    self.dialect,
                    message_insert_sql(),
                    message_values(&message, &scope_text, &subject_text)?,
                )
                .await?;
                let parent = message.id;
                (message, parent)
            }
            TurnTarget::Regenerate { message_id } => {
                let row = fetch_optional(
                    &mut *transaction,
                    self.dialect,
                    "SELECT * FROM clypeus_messages WHERE id = ? AND thread_id = ? AND scope_id = ? AND subject = ?",
                    vec![
                        message_id.to_string().into(),
                        thread.id.to_string().into(),
                        scope_text.clone().into(),
                        subject_text.clone().into(),
                    ],
                )
                .await?
                .ok_or(StoreError::NotFound)?;
                let previous = message_base_from_row(&row)?;
                if previous.role != "assistant" {
                    return Err(StoreError::Validation(
                        "only assistant messages can be regenerated.".into(),
                    ));
                }
                let anchor_id = previous
                    .parent_message_id
                    .ok_or_else(|| StoreError::Validation("message has no anchor.".into()))?;
                let anchor_row = fetch_optional(
                    &mut *transaction,
                    self.dialect,
                    "SELECT * FROM clypeus_messages WHERE id = ? AND thread_id = ? AND scope_id = ? AND subject = ?",
                    vec![
                        anchor_id.to_string().into(),
                        thread.id.to_string().into(),
                        scope_text.clone().into(),
                        subject_text.clone().into(),
                    ],
                )
                .await?
                .ok_or(StoreError::NotFound)?;
                let mut anchor = message_base_from_row(&anchor_row)?.into_message();
                anchor.tool_calls = Vec::new();
                anchor.feedback = None;
                anchor.versions = Vec::new();
                (anchor, anchor_id)
            }
        };

        let assistant_version = sibling_version_in_tx(
            &mut transaction,
            self.dialect,
            thread.id,
            Some(assistant_parent),
        )
        .await?;
        let assistant = Message {
            id: Uuid::new_v4(),
            thread_id: thread.id,
            parent_message_id: Some(assistant_parent),
            role: "assistant".into(),
            content: String::new(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            context_version: request.context_version.clone(),
            context_json: request.context_json.clone(),
            model: request.model.clone(),
            reasoning_level: request.reasoning_level.clone(),
            status: MessageStatus::Pending,
            error_detail: None,
            usage: None,
            feedback: None,
            version: assistant_version,
            versions: Vec::new(),
            created_at: now,
            updated_at: now,
            started_at: Some(now),
            completed_at: None,
        };
        exec(
            &mut *transaction,
            self.dialect,
            message_insert_sql(),
            message_values(&assistant, &scope_text, &subject_text)?,
        )
        .await?;

        exec(
            &mut *transaction,
            self.dialect,
            "UPDATE clypeus_threads SET active_leaf_message_id = ?, updated_at = ? WHERE id = ?",
            vec![
                assistant.id.to_string().into(),
                timestamp(now).into(),
                thread.id.to_string().into(),
            ],
        )
        .await?;
        transaction.commit().await.map_err(backend)?;

        thread.active_leaf_message_id = Some(assistant.id);
        thread.updated_at = now;
        Ok(StartedTurn {
            thread,
            user_message,
            assistant_message: assistant,
        })
    }

    async fn finalize_assistant(&self, finish: AssistantFinish<'_>) -> Result<(), StoreError> {
        if status_int(finish.status) && finish.status != MessageStatus::AwaitingApproval {
            return Err(StoreError::Validation(
                "assistant finalization requires a terminal status.".into(),
            ));
        }
        let usage = finish
            .usage
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| StoreError::Backend(error.to_string()))?;
        let now = Utc::now();
        let affected = self
            .exec(
                "UPDATE clypeus_messages SET
                    content = ?, reasoning_content = ?, status = ?, error_detail = ?,
                    usage_json = ?, updated_at = ?, completed_at = ?
                 WHERE id = ? AND scope_id = ? AND subject = ? AND role = 'assistant'",
                vec![
                    finish.content.into(),
                    finish.reasoning.into(),
                    finish.status.as_wire().into(),
                    finish.error_detail.into(),
                    usage.into(),
                    timestamp(now).into(),
                    timestamp(now).into(),
                    finish.message_id.to_string().into(),
                    finish.scope.to_string().into(),
                    finish.subject.into(),
                ],
            )
            .await?;
        if affected == 0 {
            return Err(StoreError::NotFound);
        }
        Ok(())
    }

    async fn finalize_stale_turns(
        &self,
        cutoff: DateTime<Utc>,
        code: &str,
    ) -> Result<usize, StoreError> {
        let now = Utc::now();
        let affected = self
            .exec(
                "UPDATE clypeus_messages SET status = 'error', error_detail = ?, updated_at = ?, completed_at = ?
                 WHERE role = 'assistant' AND status IN ('pending', 'streaming') AND updated_at < ?",
                vec![
                    code.into(),
                    timestamp(now).into(),
                    timestamp(now).into(),
                    timestamp(cutoff).into(),
                ],
            )
            .await?;
        Ok(affected as usize)
    }

    async fn thread_view(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<ThreadView, StoreError> {
        let thread = self
            .load_thread_row(scope, subject, thread)
            .await?
            .ok_or(StoreError::NotFound)?;
        let rows = self
            .fetch_all(
                "SELECT * FROM clypeus_messages WHERE thread_id = ? AND scope_id = ? AND subject = ? ORDER BY created_at",
                vec![
                    thread.id.to_string().into(),
                    scope.to_string().into(),
                    subject.into(),
                ],
            )
            .await?;
        let mut by_id: HashMap<Uuid, Message> = HashMap::new();
        for row in &rows {
            let message = self.message_from_row(row).await?;
            by_id.insert(message.id, message);
        }

        // Walk the active branch from the leaf to the root.
        let mut path = Vec::new();
        let mut cursor = thread.active_leaf_message_id;
        while let Some(id) = cursor {
            let Some(message) = by_id.get(&id) else {
                break;
            };
            path.push(id);
            cursor = message.parent_message_id;
        }
        path.reverse();

        let mut messages = Vec::with_capacity(path.len());
        for id in &path {
            let mut message = by_id.remove(id).ok_or(StoreError::NotFound)?;
            message.versions = self.sibling_versions(scope, subject, &message).await?;
            messages.push(message);
        }
        Ok(ThreadView {
            thread,
            messages,
            active_leaf_message_id: None,
        })
    }

    async fn message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<Message, StoreError> {
        self.load_message(scope, subject, message).await
    }

    async fn activate_message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<(), StoreError> {
        let target = self.load_message(scope, subject, message).await?;
        let thread = self
            .load_thread_row(scope, subject, target.thread_id)
            .await?
            .ok_or(StoreError::NotFound)?;
        let leaf = if target.role == "assistant" {
            target.id
        } else {
            // Activate the newest assistant child of the selected user message.
            let rows = self
                .fetch_all(
                    "SELECT * FROM clypeus_messages WHERE thread_id = ? AND parent_message_id = ? AND scope_id = ? AND subject = ? ORDER BY created_at DESC",
                    vec![
                        thread.id.to_string().into(),
                        target.id.to_string().into(),
                        scope.to_string().into(),
                        subject.into(),
                    ],
                )
                .await?;
            let mut newest_assistant = None;
            for row in &rows {
                let message = self.message_from_row(row).await?;
                if message.role == "assistant" {
                    newest_assistant = Some(message.id);
                    break;
                }
            }
            newest_assistant.unwrap_or(target.id)
        };
        self.exec(
            "UPDATE clypeus_threads SET active_leaf_message_id = ?, updated_at = ? WHERE id = ? AND scope_id = ? AND subject = ?",
            vec![
                leaf.to_string().into(),
                timestamp(Utc::now()).into(),
                thread.id.to_string().into(),
                scope.to_string().into(),
                subject.into(),
            ],
        )
        .await?;
        Ok(())
    }

    async fn set_feedback(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
        rating: FeedbackRating,
        comment: Option<&str>,
    ) -> Result<Feedback, StoreError> {
        let message_row = self.load_message(scope, subject, message).await?;
        if message_row.role != "assistant" {
            return Err(StoreError::Validation(
                "feedback applies to assistant messages only.".into(),
            ));
        }
        let now = Utc::now();
        self.exec(
            "INSERT INTO clypeus_feedback (message_id, scope_id, subject, rating, comment, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(message_id) DO UPDATE SET
                rating = excluded.rating,
                comment = excluded.comment,
                updated_at = excluded.updated_at",
            vec![
                message.to_string().into(),
                scope.to_string().into(),
                subject.into(),
                rating.as_wire().into(),
                comment.map(str::to_string).into(),
                timestamp(now).into(),
                timestamp(now).into(),
            ],
        )
        .await?;
        self.feedback_for(scope, subject, message)
            .await?
            .ok_or(StoreError::NotFound)
    }

    async fn clear_feedback(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<(), StoreError> {
        self.exec(
            "DELETE FROM clypeus_feedback WHERE message_id = ? AND scope_id = ? AND subject = ?",
            vec![
                message.to_string().into(),
                scope.to_string().into(),
                subject.into(),
            ],
        )
        .await?;
        Ok(())
    }

    async fn history_for_message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<Vec<ChatMessage>, StoreError> {
        let rows = self
            .fetch_all(
                "SELECT * FROM clypeus_messages WHERE scope_id = ? AND subject = ?",
                vec![scope.to_string().into(), subject.into()],
            )
            .await?;
        let mut by_id: HashMap<Uuid, Message> = HashMap::new();
        for row in &rows {
            let message = self.message_from_row(row).await?;
            by_id.insert(message.id, message);
        }
        let mut chain = Vec::new();
        let mut cursor = Some(message);
        while let Some(id) = cursor {
            let Some(row) = by_id.get(&id) else {
                break;
            };
            chain.push(row.clone());
            cursor = row.parent_message_id;
        }
        chain.reverse();

        let mut history = Vec::new();
        for row in chain {
            let Some(role) = ChatRole::parse(&row.role) else {
                continue;
            };
            let tool_calls: Vec<ToolCall> = row
                .tool_calls
                .iter()
                .map(|call| ToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .collect();
            if role == ChatRole::Assistant && row.content.trim().is_empty() && tool_calls.is_empty()
            {
                continue;
            }
            history.push(ChatMessage {
                role,
                content: row.content.clone(),
                tool_call_id: row.tool_call_id.clone(),
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            });
            for call in &row.tool_calls {
                let payload = match (&call.result, &call.error_code, call.status) {
                    (Some(result), _, _) => serde_json::json!({"ok": true, "data": result}),
                    (None, Some(code), _) => serde_json::json!({"ok": false, "code": code}),
                    (None, None, ToolCallStatus::AwaitingApproval) => {
                        serde_json::json!({"ok": false, "code": "tool_approval_required"})
                    }
                    (None, None, ToolCallStatus::Succeeded) => {
                        serde_json::json!({"ok": true, "data": Value::Null})
                    }
                    (None, None, _) => serde_json::json!({"ok": false, "code": "tool_internal"}),
                };
                history.push(ChatMessage::tool_result(
                    call.id.clone(),
                    clypeus_core::tools::wrap_untrusted_tool_result(payload).to_string(),
                ));
            }
        }
        Ok(history)
    }

    async fn thread_usage(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<TurnUsage, StoreError> {
        let rows = self
            .fetch_all(
                "SELECT id, model, reasoning_level, status, usage_json, created_at
                 FROM clypeus_messages
                 WHERE thread_id = ? AND scope_id = ? AND subject = ? AND role = 'assistant'
                 ORDER BY created_at",
                vec![
                    thread.to_string().into(),
                    scope.to_string().into(),
                    subject.into(),
                ],
            )
            .await?;
        let mut total = TokenUsage::default();
        let mut entries = Vec::new();
        for row in &rows {
            let usage = column_opt_text(row, "usage_json")?
                .and_then(|raw| serde_json::from_str::<TokenUsage>(&raw).ok())
                .unwrap_or_default();
            total.accumulate(&usage);
            entries.push(UsageEntry {
                message_id: parse_uuid(&column_text(row, "id")?)?,
                model: column_opt_text(row, "model")?,
                reasoning_level: column_opt_text(row, "reasoning_level")?,
                status: parse_status(&column_text(row, "status")?),
                usage,
                created_at: parse_timestamp(&column_text(row, "created_at")?)?,
            });
        }
        Ok(TurnUsage {
            total,
            messages: entries,
        })
    }

    async fn user_anchor_message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<Option<Uuid>, StoreError> {
        let rows = self
            .fetch_all(
                "SELECT id, parent_message_id, role FROM clypeus_messages WHERE scope_id = ? AND subject = ?",
                vec![scope.to_string().into(), subject.into()],
            )
            .await?;
        let mut by_id: HashMap<Uuid, (Option<Uuid>, String)> = HashMap::new();
        for row in &rows {
            by_id.insert(
                parse_uuid(&column_text(row, "id")?)?,
                (
                    parse_opt_uuid(column_opt_text(row, "parent_message_id")?)?,
                    column_text(row, "role")?,
                ),
            );
        }
        let mut cursor = Some(message);
        while let Some(id) = cursor {
            let Some((parent, role)) = by_id.get(&id) else {
                return Ok(None);
            };
            if role == "user" {
                return Ok(Some(id));
            }
            cursor = *parent;
        }
        Ok(None)
    }
}

/// Minimal message columns used inside a write transaction where loading the
/// full child graph would require a second connection.
#[derive(Debug, Clone)]
struct MessageBase {
    id: Uuid,
    thread_id: Uuid,
    parent_message_id: Option<Uuid>,
    role: String,
    content: String,
    version: i32,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    started_at: Option<DateTime<Utc>>,
    completed_at: Option<DateTime<Utc>>,
    context_json: Option<Value>,
}

impl MessageBase {
    fn into_message(self) -> Message {
        Message {
            id: self.id,
            thread_id: self.thread_id,
            parent_message_id: self.parent_message_id,
            role: self.role,
            content: self.content,
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            context_version: None,
            context_json: self.context_json,
            model: None,
            reasoning_level: None,
            status: MessageStatus::Complete,
            error_detail: None,
            usage: None,
            feedback: None,
            version: self.version,
            versions: Vec::new(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            started_at: self.started_at,
            completed_at: self.completed_at,
        }
    }
}

fn message_base_from_row(row: &AnyRow) -> Result<MessageBase, StoreError> {
    Ok(MessageBase {
        id: parse_uuid(&column_text(row, "id")?)?,
        thread_id: parse_uuid(&column_text(row, "thread_id")?)?,
        parent_message_id: parse_opt_uuid(column_opt_text(row, "parent_message_id")?)?,
        role: column_text(row, "role")?,
        content: column_text(row, "content")?,
        version: column_int(row, "version")? as i32,
        created_at: parse_timestamp(&column_text(row, "created_at")?)?,
        updated_at: parse_timestamp(&column_text(row, "updated_at")?)?,
        started_at: parse_opt_timestamp(column_opt_text(row, "started_at")?)?,
        completed_at: parse_opt_timestamp(column_opt_text(row, "completed_at")?)?,
        context_json: column_opt_text(row, "context_json")?.map(|raw| parse_json(&raw)),
    })
}

fn new_user_message(
    request: &BeginTurn,
    thread_id: Uuid,
    parent: Option<Uuid>,
    version: i32,
    now: DateTime<Utc>,
) -> Message {
    Message {
        id: Uuid::new_v4(),
        thread_id,
        parent_message_id: parent,
        role: "user".into(),
        content: request.content.clone(),
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        context_version: request.context_version.clone(),
        context_json: request.context_json.clone(),
        model: request.model.clone(),
        reasoning_level: request.reasoning_level.clone(),
        status: MessageStatus::Complete,
        error_detail: None,
        usage: None,
        feedback: None,
        version,
        versions: Vec::new(),
        created_at: now,
        updated_at: now,
        started_at: Some(now),
        completed_at: Some(now),
    }
}

fn message_insert_sql() -> &'static str {
    "INSERT INTO clypeus_messages (
        id, thread_id, scope_id, subject, parent_message_id, role, content,
        reasoning_content, tool_call_id, context_version, context_json, model,
        reasoning_level, status, error_detail, usage_json, version, created_at,
        updated_at, started_at, completed_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
}

async fn sibling_version_in_tx(
    connection: &mut sqlx::AnyConnection,
    dialect: SqlDialect,
    thread_id: Uuid,
    parent: Option<Uuid>,
) -> Result<i32, StoreError> {
    let count = match parent {
        Some(parent) => {
            let row = fetch_optional(
                connection,
                dialect,
                "SELECT COUNT(*) AS n FROM clypeus_messages WHERE thread_id = ? AND parent_message_id = ?",
                vec![thread_id.to_string().into(), parent.to_string().into()],
            )
            .await?;
            row.map(|row| column_int(&row, "n"))
                .transpose()?
                .unwrap_or(0)
        }
        None => {
            let row = fetch_optional(
                connection,
                dialect,
                "SELECT COUNT(*) AS n FROM clypeus_messages WHERE thread_id = ? AND parent_message_id IS NULL",
                vec![thread_id.to_string().into()],
            )
            .await?;
            row.map(|row| column_int(&row, "n"))
                .transpose()?
                .unwrap_or(0)
        }
    };
    Ok(i32::try_from(count + 1).unwrap_or(i32::MAX))
}

fn derive_title(content: &str) -> String {
    let line = content.lines().next().unwrap_or_default().trim();
    if line.is_empty() {
        "New chat".to_string()
    } else {
        line.chars().take(80).collect()
    }
}

fn message_values(
    message: &Message,
    scope: &str,
    subject: &str,
) -> Result<Vec<SqlValue>, StoreError> {
    Ok(vec![
        message.id.to_string().into(),
        message.thread_id.to_string().into(),
        scope.into(),
        subject.into(),
        message.parent_message_id.map(|id| id.to_string()).into(),
        message.role.clone().into(),
        message.content.clone().into(),
        message.reasoning_content.clone().into(),
        message.tool_call_id.clone().into(),
        message.context_version.clone().into(),
        message
            .context_json
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| StoreError::Backend(error.to_string()))?
            .into(),
        message.model.clone().into(),
        message.reasoning_level.clone().into(),
        message.status.as_wire().into(),
        message.error_detail.clone().into(),
        message
            .usage
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| StoreError::Backend(error.to_string()))?
            .into(),
        i64::from(message.version).into(),
        timestamp(message.created_at).into(),
        timestamp(message.updated_at).into(),
        message.started_at.map(timestamp).into(),
        message.completed_at.map(timestamp).into(),
    ])
}

// ---------------------------------------------------------------------------
// Approvals
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl ApprovalStore for SqlStore {
    async fn record_tool_call(&self, call: NewToolCall<'_>) -> Result<(), StoreError> {
        self.exec(
            "INSERT INTO clypeus_tool_calls (
                id, scope_id, subject, thread_id, message_id, tool_name, risk,
                arguments_json, arguments_hash, status, auth_mode, expires_at,
                approval_kind, confirm_field, impact, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING",
            vec![
                call.id.into(),
                call.scope.to_string().into(),
                call.subject.into(),
                call.thread_id.to_string().into(),
                call.message_id.to_string().into(),
                call.name.into(),
                call.risk.into(),
                call.arguments_json.into(),
                call.arguments_hash.into(),
                call.status.as_wire().into(),
                call.auth_mode.map(str::to_string).into(),
                call.expires_at.map(timestamp).into(),
                call.approval_kind.map(str::to_string).into(),
                call.confirm_field.map(str::to_string).into(),
                call.impact.map(str::to_string).into(),
                timestamp(Utc::now()).into(),
            ],
        )
        .await?;
        Ok(())
    }

    async fn insert_tool_approval(&self, approval: NewToolApproval<'_>) -> Result<(), StoreError> {
        self.exec(
            "INSERT INTO clypeus_tool_approvals (
                id, tool_call_id, scope_id, subject, decision, reason,
                typed_confirm, arguments_hash, expires_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            vec![
                approval.id.into(),
                approval.tool_call_id.into(),
                approval.scope.to_string().into(),
                approval.subject.into(),
                approval.decision.into(),
                approval.reason.map(str::to_string).into(),
                approval.typed_confirm.map(str::to_string).into(),
                approval.arguments_hash.into(),
                timestamp(approval.expires_at).into(),
                timestamp(Utc::now()).into(),
            ],
        )
        .await?;
        Ok(())
    }

    async fn mark_tool_call_approved(&self, id: &str, approval_id: &str) -> Result<(), StoreError> {
        let affected = self
            .exec(
                "UPDATE clypeus_tool_calls SET status = 'approved', approval_id = ?
                 WHERE id = ? AND status = 'awaiting_approval'",
                vec![approval_id.to_string().into(), id.into()],
            )
            .await?;
        if affected == 0 {
            return Err(StoreError::Conflict("call is not awaiting approval".into()));
        }
        Ok(())
    }

    async fn mark_tool_call_denied(&self, id: &str) -> Result<(), StoreError> {
        let affected = self
            .exec(
                "UPDATE clypeus_tool_calls SET status = 'denied', completed_at = ?
                 WHERE id = ? AND status = 'awaiting_approval'",
                vec![timestamp(Utc::now()).into(), id.into()],
            )
            .await?;
        if affected == 0 {
            return Err(StoreError::Conflict("call is not awaiting approval".into()));
        }
        Ok(())
    }

    async fn mark_tool_call_running(
        &self,
        id: &str,
        auth_mode: Option<&str>,
        approval_id: Option<&str>,
    ) -> Result<(), StoreError> {
        self.exec(
            "UPDATE clypeus_tool_calls SET status = 'running',
                auth_mode = COALESCE(?, auth_mode),
                approval_id = COALESCE(?, approval_id)
             WHERE id = ?",
            vec![
                auth_mode.map(str::to_string).into(),
                approval_id.map(str::to_string).into(),
                id.into(),
            ],
        )
        .await?;
        Ok(())
    }

    async fn complete_tool_call(
        &self,
        id: &str,
        completion: ToolCallCompletion<'_>,
    ) -> Result<(), StoreError> {
        let result = completion
            .result_json
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| StoreError::Backend(error.to_string()))?;
        self.exec(
            "UPDATE clypeus_tool_calls SET status = ?, result_json = ?,
                error_code = ?, duration_ms = ?,
                approval_id = COALESCE(?, approval_id),
                completed_at = ?
             WHERE id = ?",
            vec![
                completion.status.as_wire().into(),
                result.into(),
                completion.error_code.map(str::to_string).into(),
                completion.duration_ms.into(),
                completion.approval_id.map(str::to_string).into(),
                timestamp(Utc::now()).into(),
                id.into(),
            ],
        )
        .await?;
        Ok(())
    }

    async fn get_tool_call(&self, id: &str) -> Result<Option<ToolCallSnapshot>, StoreError> {
        let row = self
            .fetch_optional(
                "SELECT * FROM clypeus_tool_calls WHERE id = ?",
                vec![id.into()],
            )
            .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        Ok(Some(ToolCallSnapshot {
            id: column_text(&row, "id")?,
            scope_id: column_text(&row, "scope_id")?,
            subject: column_text(&row, "subject")?,
            thread_id: parse_uuid(&column_text(&row, "thread_id")?)?,
            message_id: parse_uuid(&column_text(&row, "message_id")?)?,
            name: column_text(&row, "tool_name")?,
            risk: column_text(&row, "risk")?,
            status: column_text(&row, "status")?,
            approval_kind: column_opt_text(&row, "approval_kind")?,
            approval_id: column_opt_text(&row, "approval_id")?,
            confirm_field: column_opt_text(&row, "confirm_field")?,
            arguments: parse_json(&column_text(&row, "arguments_json")?),
            arguments_hash: column_text(&row, "arguments_hash")?,
            expires_at: parse_opt_timestamp(column_opt_text(&row, "expires_at")?)?,
        }))
    }

    async fn count_recent_tool_calls(
        &self,
        scope: &ScopeId,
        subject: &str,
        name: &str,
        since: DateTime<Utc>,
    ) -> Result<u64, StoreError> {
        let row = self
            .fetch_optional(
                "SELECT COUNT(*) AS n FROM clypeus_tool_calls
                 WHERE scope_id = ? AND subject = ? AND tool_name = ? AND created_at >= ?
                   AND status NOT IN ('denied', 'expired')",
                vec![
                    scope.to_string().into(),
                    subject.into(),
                    name.into(),
                    timestamp(since).into(),
                ],
            )
            .await?;
        let count = row
            .map(|row| column_int(&row, "n"))
            .transpose()?
            .unwrap_or(0);
        Ok(u64::try_from(count).unwrap_or(0))
    }

    async fn expire_tool_calls(&self, now: DateTime<Utc>) -> Result<usize, StoreError> {
        let affected = self
            .exec(
                "UPDATE clypeus_tool_calls SET status = 'expired', completed_at = ?
                 WHERE status = 'awaiting_approval' AND expires_at IS NOT NULL AND expires_at < ?",
                vec![timestamp(now).into(), timestamp(now).into()],
            )
            .await?;
        Ok(affected as usize)
    }
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl clypeus_core::audit::AuditSink for SqlStore {
    async fn append(&self, record: clypeus_core::audit::AuditRecord) -> Result<(), StoreError> {
        self.exec(
            "INSERT INTO clypeus_audit (
                id, scope_id, subject, thread_id, message_id, tool_call_id,
                item_name, item_kind, risk, arguments_hash, arguments_redacted,
                scopes_used, decision, auth_mode, egress_service, egress_path_template,
                outcome, downstream_status, result_bytes, duration_ms, approval_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            vec![
                record.id.to_string().into(),
                record.scope_id.into(),
                record.subject.into(),
                record.thread_id.map(|id| id.to_string()).into(),
                record.message_id.map(|id| id.to_string()).into(),
                record.tool_call_id.into(),
                record.item_name.into(),
                record.item_kind.as_wire().into(),
                record.risk.into(),
                record.arguments_hash.into(),
                record
                    .arguments_redacted
                    .map(|value| value.to_string())
                    .into(),
                record.scopes_used.into(),
                record.decision.into(),
                record.auth_mode.into(),
                record.egress_service.into(),
                record.egress_path_template.into(),
                record.outcome.into(),
                record.downstream_status.map(i64::from).into(),
                record.result_bytes.map(i64::from).into(),
                record.duration_ms.into(),
                record.approval_id.into(),
                timestamp(record.created_at).into(),
            ],
        )
        .await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl clypeus_core::audit::AuditReader for SqlStore {
    async fn page(
        &self,
        scope: &ScopeId,
        query: clypeus_core::audit::AuditQuery,
    ) -> Result<clypeus_core::audit::AuditPage, StoreError> {
        let (filter, values) = audit_filter(scope, &query);
        let limit = if query.limit <= 0 {
            50
        } else {
            query.limit.min(500)
        };
        let offset = query.offset.max(0);

        let count_row = self
            .fetch_optional(
                &format!("SELECT COUNT(*) AS n FROM clypeus_audit WHERE {filter}"),
                values.clone(),
            )
            .await?;
        let total = count_row
            .map(|row| column_int(&row, "n"))
            .transpose()?
            .unwrap_or(0);

        let mut page_values = values;
        page_values.push(limit.into());
        page_values.push(offset.into());
        let rows = self
            .fetch_all(
                &format!(
                    "SELECT * FROM clypeus_audit WHERE {filter} ORDER BY created_at DESC LIMIT ? OFFSET ?"
                ),
                page_values,
            )
            .await?;
        let entries = rows
            .iter()
            .map(audit_from_row)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(clypeus_core::audit::AuditPage {
            entries,
            limit,
            offset,
            total,
        })
    }

    async fn export(
        &self,
        scope: &ScopeId,
        query: clypeus_core::audit::AuditQuery,
    ) -> Result<Vec<clypeus_core::audit::AuditRecord>, StoreError> {
        let (filter, values) = audit_filter(scope, &query);
        let rows = self
            .fetch_all(
                &format!("SELECT * FROM clypeus_audit WHERE {filter} ORDER BY created_at"),
                values,
            )
            .await?;
        rows.iter().map(audit_from_row).collect()
    }

    async fn purge_before(&self, cutoff: DateTime<Utc>) -> Result<usize, StoreError> {
        let affected = self
            .exec(
                "DELETE FROM clypeus_audit WHERE created_at < ?",
                vec![timestamp(cutoff).into()],
            )
            .await?;
        Ok(affected as usize)
    }
}

fn audit_filter(
    scope: &ScopeId,
    query: &clypeus_core::audit::AuditQuery,
) -> (String, Vec<SqlValue>) {
    let mut filter = String::from("scope_id = ?");
    let mut values: Vec<SqlValue> = vec![scope.to_string().into()];
    if let Some(kind) = query.item_kind {
        filter.push_str(" AND item_kind = ?");
        values.push(kind.as_wire().into());
    }
    if let Some(name) = query.item_name.as_deref().filter(|name| !name.is_empty()) {
        filter.push_str(" AND item_name = ?");
        values.push(name.into());
    }
    if let Some(outcome) = query
        .outcome
        .as_deref()
        .filter(|outcome| !outcome.is_empty())
    {
        filter.push_str(" AND outcome = ?");
        values.push(outcome.into());
    }
    if let Some(since) = query.since {
        filter.push_str(" AND created_at >= ?");
        values.push(timestamp(since).into());
    }
    if let Some(until) = query.until {
        filter.push_str(" AND created_at <= ?");
        values.push(timestamp(until).into());
    }
    (filter, values)
}

fn audit_from_row(row: &AnyRow) -> Result<clypeus_core::audit::AuditRecord, StoreError> {
    Ok(clypeus_core::audit::AuditRecord {
        id: parse_uuid(&column_text(row, "id")?)?,
        scope_id: column_text(row, "scope_id")?,
        subject: column_text(row, "subject")?,
        thread_id: parse_opt_uuid(column_opt_text(row, "thread_id")?)?,
        message_id: parse_opt_uuid(column_opt_text(row, "message_id")?)?,
        tool_call_id: column_opt_text(row, "tool_call_id")?,
        item_name: column_text(row, "item_name")?,
        item_kind: clypeus_core::audit::AuditItemKind::parse(&column_text(row, "item_kind")?)
            .unwrap_or(clypeus_core::audit::AuditItemKind::Tool),
        risk: column_text(row, "risk")?,
        arguments_hash: column_text(row, "arguments_hash")?,
        arguments_redacted: column_opt_text(row, "arguments_redacted")?.map(|raw| parse_json(&raw)),
        scopes_used: column_opt_text(row, "scopes_used")?,
        decision: column_opt_text(row, "decision")?,
        auth_mode: column_opt_text(row, "auth_mode")?,
        egress_service: column_opt_text(row, "egress_service")?,
        egress_path_template: column_opt_text(row, "egress_path_template")?,
        outcome: column_text(row, "outcome")?,
        downstream_status: column_opt_int(row, "downstream_status")?.map(|value| value as i32),
        result_bytes: column_opt_int(row, "result_bytes")?.map(|value| value as i32),
        duration_ms: column_opt_int(row, "duration_ms")?,
        approval_id: column_opt_text(row, "approval_id")?,
        created_at: parse_timestamp(&column_text(row, "created_at")?)?,
    })
}
