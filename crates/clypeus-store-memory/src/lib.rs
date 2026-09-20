//! In-memory store implementation.
//!
//! Implements every store trait over a single mutex-protected map. Intended
//! for unit tests, contract tests and embedded single-process deployments
//! that do not need durability.

use std::collections::BTreeMap;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use clypeus_core::audit::{AuditPage, AuditQuery, AuditReader, AuditRecord, AuditSink};
use clypeus_core::models::{ChatMessage, ProviderKind, TokenUsage};
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
use uuid::Uuid;

/// Tool call row held by the memory store.
#[derive(Debug, Clone)]
struct ToolCallRow {
    id: String,
    scope_id: String,
    subject: String,
    thread_id: Uuid,
    message_id: Uuid,
    name: String,
    risk: String,
    arguments: Value,
    arguments_hash: String,
    status: ToolCallStatus,
    auth_mode: Option<String>,
    expires_at: Option<DateTime<Utc>>,
    approval_kind: Option<String>,
    confirm_field: Option<String>,
    impact: Option<String>,
    approval_id: Option<String>,
    result: Option<Value>,
    error_code: Option<String>,
    duration_ms: Option<i64>,
    created_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
}

#[derive(Default)]
struct State {
    settings: BTreeMap<String, ScopeSettings>,
    threads: BTreeMap<Uuid, Thread>,
    messages: BTreeMap<Uuid, Message>,
    feedback: BTreeMap<Uuid, Feedback>,
    tool_calls: BTreeMap<String, ToolCallRow>,
    audit: Vec<AuditRecord>,
}

/// In-memory store.
#[derive(Default)]
pub struct MemoryStore {
    state: Mutex<State>,
}

impl std::fmt::Debug for MemoryStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        formatter
            .debug_struct("MemoryStore")
            .field("threads", &state.threads.len())
            .field("messages", &state.messages.len())
            .field("tool_calls", &state.tool_calls.len())
            .field("audit", &state.audit.len())
            .finish()
    }
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn thread_in_scope(
        state: &State,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<Thread, StoreError> {
        state
            .threads
            .get(&thread)
            .filter(|thread| thread.scope_id == scope.as_str() && thread.subject == subject)
            .cloned()
            .ok_or(StoreError::NotFound)
    }

    fn message_in_scope(
        state: &State,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<Message, StoreError> {
        state
            .messages
            .get(&message)
            .filter(|message| {
                state.threads.get(&message.thread_id).is_some_and(|thread| {
                    thread.scope_id == scope.as_str() && thread.subject == subject
                })
            })
            .cloned()
            .ok_or(StoreError::NotFound)
    }

    fn decorate(state: &State, mut message: Message) -> Message {
        message.tool_calls = state
            .tool_calls
            .values()
            .filter(|call| call.message_id == message.id)
            .map(tool_call_view)
            .collect();
        message.feedback = state.feedback.get(&message.id).cloned();
        message.versions = sibling_versions(state, &message);
        message
    }
}

fn tool_call_view(row: &ToolCallRow) -> PersistedToolCall {
    PersistedToolCall {
        id: row.id.clone(),
        name: row.name.clone(),
        arguments: row.arguments.clone(),
        risk: row.risk.clone(),
        status: row.status,
        result: row.result.clone(),
        error_code: row.error_code.clone(),
        duration_ms: row.duration_ms,
        approval: row.approval_kind.as_ref().map(|kind| ApprovalView {
            kind: kind.clone(),
            expires_at: row.expires_at.unwrap_or_else(Utc::now),
            arguments_hash: row.arguments_hash.clone(),
            confirm_field: row.confirm_field.clone(),
            impact: row.impact.clone().unwrap_or_default(),
        }),
        created_at: row.created_at,
        completed_at: row.completed_at,
    }
}

fn sibling_versions(state: &State, message: &Message) -> Vec<MessageVersion> {
    let mut siblings: Vec<&Message> = state
        .messages
        .values()
        .filter(|sibling| {
            sibling.thread_id == message.thread_id
                && sibling.parent_message_id == message.parent_message_id
        })
        .collect();
    siblings.sort_by_key(|sibling| sibling.created_at);
    siblings
        .into_iter()
        .enumerate()
        .map(|(index, sibling)| MessageVersion {
            id: sibling.id,
            version: i32::try_from(index + 1).unwrap_or(i32::MAX),
            is_active: sibling.id == message.id,
            status: sibling.status,
            created_at: sibling.created_at,
        })
        .collect()
}

fn derive_title(content: &str) -> String {
    let line = content.lines().next().unwrap_or_default().trim();
    if line.is_empty() {
        "New chat".to_string()
    } else {
        line.chars().take(80).collect()
    }
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

#[async_trait::async_trait]
impl clypeus_core::store::Readiness for MemoryStore {
    async fn check_ready(&self) -> Result<(), String> {
        Ok(())
    }
}

#[async_trait::async_trait]
impl ScopeSettingsStore for MemoryStore {
    async fn get(&self, scope: &ScopeId) -> Result<Option<ScopeSettings>, StoreError> {
        Ok(self.lock().settings.get(scope.as_str()).cloned())
    }

    async fn upsert(
        &self,
        scope: &ScopeId,
        update: ScopeSettingsUpdate,
    ) -> Result<ScopeSettings, StoreError> {
        let mut state = self.lock();
        let now = Utc::now();
        let settings = state
            .settings
            .entry(scope.to_string())
            .or_insert_with(|| ScopeSettings {
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
            settings.timeout_ms = timeout;
        }
        if let Some(tokens) = update.max_output_tokens {
            settings.max_output_tokens = tokens;
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
        Ok(settings.clone())
    }
}

#[async_trait::async_trait]
impl ConversationStore for MemoryStore {
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
        self.lock().threads.insert(thread.id, thread.clone());
        Ok(thread)
    }

    async fn list_threads(
        &self,
        scope: &ScopeId,
        subject: &str,
    ) -> Result<Vec<Thread>, StoreError> {
        let mut threads: Vec<Thread> = self
            .lock()
            .threads
            .values()
            .filter(|thread| thread.scope_id == scope.as_str() && thread.subject == subject)
            .cloned()
            .collect();
        threads.sort_by_key(|thread| {
            (
                std::cmp::Reverse(thread.pinned),
                std::cmp::Reverse(thread.updated_at),
            )
        });
        Ok(threads)
    }

    async fn get_thread(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<Option<Thread>, StoreError> {
        let state = self.lock();
        Ok(state
            .threads
            .get(&thread)
            .filter(|thread| thread.scope_id == scope.as_str() && thread.subject == subject)
            .cloned())
    }

    async fn update_thread(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
        update: ThreadUpdate,
    ) -> Result<Thread, StoreError> {
        let mut state = self.lock();
        let existing = state
            .threads
            .get_mut(&thread)
            .filter(|existing| existing.scope_id == scope.as_str() && existing.subject == subject)
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
        Ok(existing.clone())
    }

    async fn delete_thread(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<(), StoreError> {
        let mut state = self.lock();
        Self::thread_in_scope(&state, scope, subject, thread)?;
        state.threads.remove(&thread);
        state
            .messages
            .retain(|_, message| message.thread_id != thread);
        state.tool_calls.retain(|_, call| call.thread_id != thread);
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
        let mut state = self.lock();
        let now = Utc::now();

        let mut thread = match request.thread_id {
            Some(thread_id) => Self::thread_in_scope(&state, scope, subject, thread_id)?,
            None => {
                let thread = Thread {
                    id: Uuid::new_v4(),
                    scope_id: scope.to_string(),
                    subject: subject.to_string(),
                    title: derive_title(&request.content),
                    pinned: false,
                    model: request.model.clone(),
                    active_leaf_message_id: None,
                    created_at: now,
                    updated_at: now,
                };
                state.threads.insert(thread.id, thread.clone());
                thread
            }
        };

        let (user_message, assistant_parent) = match &request.target {
            TurnTarget::New => {
                let parent = thread.active_leaf_message_id;
                let version = next_sibling_version(&state, thread.id, parent);
                let message = new_user_message(&request, thread.id, parent, version, now);
                state.messages.insert(message.id, message.clone());
                let parent = message.id;
                (message, parent)
            }
            TurnTarget::Edit { message_id } => {
                let previous = Self::message_in_scope(&state, scope, subject, *message_id)?;
                if previous.role != "user" {
                    return Err(StoreError::Validation(
                        "only user messages can be edited.".into(),
                    ));
                }
                let version = next_sibling_version(&state, thread.id, previous.parent_message_id);
                let message = new_user_message(
                    &request,
                    thread.id,
                    previous.parent_message_id,
                    version,
                    now,
                );
                state.messages.insert(message.id, message.clone());
                let parent = message.id;
                (message, parent)
            }
            TurnTarget::Regenerate { message_id } => {
                let previous = Self::message_in_scope(&state, scope, subject, *message_id)?;
                if previous.role != "assistant" {
                    return Err(StoreError::Validation(
                        "only assistant messages can be regenerated.".into(),
                    ));
                }
                let anchor = previous
                    .parent_message_id
                    .ok_or_else(|| StoreError::Validation("message has no anchor.".into()))?;
                let anchor_message = Self::message_in_scope(&state, scope, subject, anchor)?;
                (anchor_message, anchor)
            }
        };

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
            version: next_sibling_version(&state, thread.id, Some(assistant_parent)),
            versions: Vec::new(),
            created_at: now,
            updated_at: now,
            started_at: Some(now),
            completed_at: None,
        };
        state.messages.insert(assistant.id, assistant.clone());
        if let Some(row) = state.threads.get_mut(&thread.id) {
            row.active_leaf_message_id = Some(assistant.id);
            row.updated_at = now;
        }
        thread.active_leaf_message_id = Some(assistant.id);
        thread.updated_at = now;
        Ok(StartedTurn {
            thread,
            user_message: Self::decorate(&state, user_message),
            assistant_message: Self::decorate(&state, assistant),
        })
    }

    async fn finalize_assistant(&self, finish: AssistantFinish<'_>) -> Result<(), StoreError> {
        if matches!(
            finish.status,
            MessageStatus::Pending | MessageStatus::Streaming
        ) {
            return Err(StoreError::Validation(
                "assistant finalization requires a terminal status.".into(),
            ));
        }
        let mut state = self.lock();
        let thread_id = state
            .messages
            .get(&finish.message_id)
            .map(|message| message.thread_id)
            .ok_or(StoreError::NotFound)?;
        let thread_matches = state.threads.get(&thread_id).is_some_and(|thread| {
            thread.scope_id == finish.scope.as_str() && thread.subject == finish.subject
        });
        if !thread_matches {
            return Err(StoreError::NotFound);
        }
        let message = state
            .messages
            .get_mut(&finish.message_id)
            .ok_or(StoreError::NotFound)?;
        if message.role != "assistant" {
            return Err(StoreError::NotFound);
        }
        message.content = finish.content.to_string();
        message.reasoning_content = finish.reasoning.map(str::to_string);
        message.status = finish.status;
        message.error_detail = finish.error_detail.map(str::to_string);
        message.usage = finish.usage.cloned();
        message.updated_at = Utc::now();
        message.completed_at = Some(Utc::now());
        Ok(())
    }

    async fn finalize_stale_turns(
        &self,
        cutoff: DateTime<Utc>,
        code: &str,
    ) -> Result<usize, StoreError> {
        let mut state = self.lock();
        let now = Utc::now();
        let mut affected = 0;
        for message in state.messages.values_mut() {
            if message.role == "assistant"
                && matches!(
                    message.status,
                    MessageStatus::Pending | MessageStatus::Streaming
                )
                && message.updated_at < cutoff
            {
                message.status = MessageStatus::Error;
                message.error_detail = Some(code.to_string());
                message.updated_at = now;
                message.completed_at = Some(now);
                affected += 1;
            }
        }
        Ok(affected)
    }

    async fn thread_view(
        &self,
        scope: &ScopeId,
        subject: &str,
        thread: Uuid,
    ) -> Result<ThreadView, StoreError> {
        let state = self.lock();
        let thread = Self::thread_in_scope(&state, scope, subject, thread)?;
        let mut path = Vec::new();
        let mut cursor = thread.active_leaf_message_id;
        while let Some(id) = cursor {
            let Some(message) = state.messages.get(&id) else {
                break;
            };
            path.push(id);
            cursor = message.parent_message_id;
        }
        path.reverse();
        let messages = path
            .into_iter()
            .filter_map(|id| state.messages.get(&id).cloned())
            .map(|message| Self::decorate(&state, message))
            .collect();
        Ok(ThreadView {
            active_leaf_message_id: thread.active_leaf_message_id,
            thread,
            messages,
        })
    }

    async fn message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<Message, StoreError> {
        let state = self.lock();
        let message = Self::message_in_scope(&state, scope, subject, message)?;
        Ok(Self::decorate(&state, message))
    }

    async fn activate_message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<(), StoreError> {
        let mut state = self.lock();
        let target = Self::message_in_scope(&state, scope, subject, message)?;
        let leaf = if target.role == "assistant" {
            target.id
        } else {
            let mut children: Vec<&Message> = state
                .messages
                .values()
                .filter(|candidate| {
                    candidate.thread_id == target.thread_id
                        && candidate.parent_message_id == Some(target.id)
                        && candidate.role == "assistant"
                })
                .collect();
            children.sort_by_key(|child| child.created_at);
            children.last().map_or(target.id, |child| child.id)
        };
        let thread = state
            .threads
            .get_mut(&target.thread_id)
            .ok_or(StoreError::NotFound)?;
        thread.active_leaf_message_id = Some(leaf);
        thread.updated_at = Utc::now();
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
        let mut state = self.lock();
        let target = Self::message_in_scope(&state, scope, subject, message)?;
        if target.role != "assistant" {
            return Err(StoreError::Validation(
                "feedback applies to assistant messages only.".into(),
            ));
        }
        let now = Utc::now();
        let created_at = state
            .feedback
            .get(&message)
            .map_or(now, |feedback| feedback.created_at);
        let feedback = Feedback {
            message_id: message,
            rating,
            comment: comment.map(str::to_string),
            created_at,
            updated_at: now,
        };
        state.feedback.insert(message, feedback.clone());
        Ok(feedback)
    }

    async fn clear_feedback(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<(), StoreError> {
        let mut state = self.lock();
        Self::message_in_scope(&state, scope, subject, message)?;
        state.feedback.remove(&message);
        Ok(())
    }

    async fn history_for_message(
        &self,
        scope: &ScopeId,
        subject: &str,
        message: Uuid,
    ) -> Result<Vec<ChatMessage>, StoreError> {
        let state = self.lock();
        let target = Self::message_in_scope(&state, scope, subject, message)?;
        let mut chain = Vec::new();
        let mut cursor = Some(target.id);
        while let Some(id) = cursor {
            let Some(row) = state.messages.get(&id) else {
                break;
            };
            chain.push(Self::decorate(&state, row.clone()));
            cursor = row.parent_message_id;
        }
        chain.reverse();

        let mut history = Vec::new();
        for row in chain {
            let Some(role) = clypeus_core::models::ChatRole::parse(&row.role) else {
                continue;
            };
            let tool_calls: Vec<clypeus_core::models::ToolCall> = row
                .tool_calls
                .iter()
                .map(|call| clypeus_core::models::ToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .collect();
            if role == clypeus_core::models::ChatRole::Assistant
                && row.content.trim().is_empty()
                && tool_calls.is_empty()
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
        let state = self.lock();
        Self::thread_in_scope(&state, scope, subject, thread)?;
        let mut messages: Vec<&Message> = state
            .messages
            .values()
            .filter(|message| message.thread_id == thread && message.role == "assistant")
            .collect();
        messages.sort_by_key(|message| message.created_at);
        let mut total = TokenUsage::default();
        let mut entries = Vec::new();
        for message in messages {
            let usage = message.usage.clone().unwrap_or_default();
            total.accumulate(&usage);
            entries.push(UsageEntry {
                message_id: message.id,
                model: message.model.clone(),
                reasoning_level: message.reasoning_level.clone(),
                status: message.status,
                usage,
                created_at: message.created_at,
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
        let state = self.lock();
        let target = Self::message_in_scope(&state, scope, subject, message)?;
        let mut cursor = Some(target.id);
        while let Some(id) = cursor {
            let Some(row) = state.messages.get(&id) else {
                return Ok(None);
            };
            if row.role == "user" {
                return Ok(Some(id));
            }
            cursor = row.parent_message_id;
        }
        Ok(None)
    }
}

fn next_sibling_version(state: &State, thread: Uuid, parent: Option<Uuid>) -> i32 {
    let count = state
        .messages
        .values()
        .filter(|message| message.thread_id == thread && message.parent_message_id == parent)
        .count();
    i32::try_from(count).unwrap_or(i32::MAX).saturating_add(1)
}

#[async_trait::async_trait]
impl ApprovalStore for MemoryStore {
    async fn record_tool_call(&self, call: NewToolCall<'_>) -> Result<(), StoreError> {
        let mut state = self.lock();
        state
            .tool_calls
            .entry(call.id.to_string())
            .or_insert_with(|| ToolCallRow {
                id: call.id.to_string(),
                scope_id: call.scope.to_string(),
                subject: call.subject.to_string(),
                thread_id: call.thread_id,
                message_id: call.message_id,
                name: call.name.to_string(),
                risk: call.risk.to_string(),
                arguments: serde_json::from_str(call.arguments_json).unwrap_or(Value::Null),
                arguments_hash: call.arguments_hash.to_string(),
                status: call.status,
                auth_mode: call.auth_mode.map(str::to_string),
                expires_at: call.expires_at,
                approval_kind: call.approval_kind.map(str::to_string),
                confirm_field: call.confirm_field.map(str::to_string),
                impact: call.impact.map(str::to_string),
                approval_id: None,
                result: None,
                error_code: None,
                duration_ms: None,
                created_at: Utc::now(),
                completed_at: None,
            });
        Ok(())
    }

    async fn insert_tool_approval(&self, _approval: NewToolApproval<'_>) -> Result<(), StoreError> {
        Ok(())
    }

    async fn mark_tool_call_approved(&self, id: &str, approval_id: &str) -> Result<(), StoreError> {
        let mut state = self.lock();
        let call = state.tool_calls.get_mut(id).ok_or(StoreError::NotFound)?;
        if call.status != ToolCallStatus::AwaitingApproval {
            return Err(StoreError::Conflict("call is not awaiting approval".into()));
        }
        call.status = ToolCallStatus::Approved;
        call.approval_id = Some(approval_id.to_string());
        Ok(())
    }

    async fn mark_tool_call_denied(&self, id: &str) -> Result<(), StoreError> {
        let mut state = self.lock();
        let call = state.tool_calls.get_mut(id).ok_or(StoreError::NotFound)?;
        if call.status != ToolCallStatus::AwaitingApproval {
            return Err(StoreError::Conflict("call is not awaiting approval".into()));
        }
        call.status = ToolCallStatus::Denied;
        call.completed_at = Some(Utc::now());
        Ok(())
    }

    async fn mark_tool_call_running(
        &self,
        id: &str,
        auth_mode: Option<&str>,
        approval_id: Option<&str>,
    ) -> Result<(), StoreError> {
        let mut state = self.lock();
        let call = state.tool_calls.get_mut(id).ok_or(StoreError::NotFound)?;
        call.status = ToolCallStatus::Running;
        if let Some(auth_mode) = auth_mode {
            call.auth_mode = Some(auth_mode.to_string());
        }
        if let Some(approval_id) = approval_id {
            call.approval_id = Some(approval_id.to_string());
        }
        Ok(())
    }

    async fn complete_tool_call(
        &self,
        id: &str,
        completion: ToolCallCompletion<'_>,
    ) -> Result<(), StoreError> {
        let mut state = self.lock();
        let call = state.tool_calls.get_mut(id).ok_or(StoreError::NotFound)?;
        call.status = completion.status;
        call.result = completion.result_json.cloned();
        call.error_code = completion.error_code.map(str::to_string);
        call.duration_ms = completion.duration_ms;
        if let Some(approval_id) = completion.approval_id {
            call.approval_id = Some(approval_id.to_string());
        }
        call.completed_at = Some(Utc::now());
        Ok(())
    }

    async fn get_tool_call(&self, id: &str) -> Result<Option<ToolCallSnapshot>, StoreError> {
        Ok(self.lock().tool_calls.get(id).map(|call| ToolCallSnapshot {
            id: call.id.clone(),
            scope_id: call.scope_id.clone(),
            subject: call.subject.clone(),
            thread_id: call.thread_id,
            message_id: call.message_id,
            name: call.name.clone(),
            risk: call.risk.clone(),
            status: call.status.as_wire().to_string(),
            approval_kind: call.approval_kind.clone(),
            approval_id: call.approval_id.clone(),
            confirm_field: call.confirm_field.clone(),
            arguments: call.arguments.clone(),
            arguments_hash: call.arguments_hash.clone(),
            expires_at: call.expires_at,
        }))
    }

    async fn count_recent_tool_calls(
        &self,
        scope: &ScopeId,
        subject: &str,
        name: &str,
        since: DateTime<Utc>,
    ) -> Result<u64, StoreError> {
        let count = self
            .lock()
            .tool_calls
            .values()
            .filter(|call| {
                call.scope_id == scope.as_str()
                    && call.subject == subject
                    && call.name == name
                    && call.created_at >= since
                    && !matches!(
                        call.status,
                        ToolCallStatus::Denied | ToolCallStatus::Expired
                    )
            })
            .count();
        Ok(count as u64)
    }

    async fn expire_tool_calls(&self, now: DateTime<Utc>) -> Result<usize, StoreError> {
        let mut state = self.lock();
        let mut affected = 0;
        for call in state.tool_calls.values_mut() {
            if call.status == ToolCallStatus::AwaitingApproval
                && call.expires_at.is_some_and(|expires| expires < now)
            {
                call.status = ToolCallStatus::Expired;
                call.completed_at = Some(now);
                affected += 1;
            }
        }
        Ok(affected)
    }
}

#[async_trait::async_trait]
impl AuditSink for MemoryStore {
    async fn append(&self, record: AuditRecord) -> Result<(), StoreError> {
        self.lock().audit.push(record);
        Ok(())
    }
}

#[async_trait::async_trait]
impl AuditReader for MemoryStore {
    async fn page(&self, scope: &ScopeId, query: AuditQuery) -> Result<AuditPage, StoreError> {
        let state = self.lock();
        let mut entries: Vec<AuditRecord> = state
            .audit
            .iter()
            .filter(|record| record.scope_id == scope.as_str())
            .filter(|record| audit_matches(record, &query))
            .cloned()
            .collect();
        entries.sort_by_key(|record| std::cmp::Reverse(record.created_at));
        let total = entries.len() as i64;
        let limit = if query.limit <= 0 {
            50
        } else {
            query.limit.min(500)
        };
        let offset = query.offset.max(0) as usize;
        let entries = entries
            .into_iter()
            .skip(offset)
            .take(limit as usize)
            .collect();
        Ok(AuditPage {
            entries,
            limit,
            offset: offset as i64,
            total,
        })
    }

    async fn export(
        &self,
        scope: &ScopeId,
        query: AuditQuery,
    ) -> Result<Vec<AuditRecord>, StoreError> {
        let state = self.lock();
        let mut entries: Vec<AuditRecord> = state
            .audit
            .iter()
            .filter(|record| record.scope_id == scope.as_str())
            .filter(|record| audit_matches(record, &query))
            .cloned()
            .collect();
        entries.sort_by_key(|record| record.created_at);
        Ok(entries)
    }

    async fn purge_before(&self, cutoff: DateTime<Utc>) -> Result<usize, StoreError> {
        let mut state = self.lock();
        let before = state.audit.len();
        state.audit.retain(|record| record.created_at >= cutoff);
        Ok(before - state.audit.len())
    }
}

fn audit_matches(record: &AuditRecord, query: &AuditQuery) -> bool {
    if let Some(kind) = query.item_kind
        && record.item_kind != kind
    {
        return false;
    }
    if let Some(name) = query.item_name.as_deref().filter(|name| !name.is_empty())
        && record.item_name != name
    {
        return false;
    }
    if let Some(outcome) = query
        .outcome
        .as_deref()
        .filter(|outcome| !outcome.is_empty())
        && record.outcome != outcome
    {
        return false;
    }
    if let Some(since) = query.since
        && record.created_at < since
    {
        return false;
    }
    if let Some(until) = query.until
        && record.created_at > until
    {
        return false;
    }
    true
}

/// Backwards-compatible alias used in documentation.
pub type InMemoryStore = MemoryStore;
