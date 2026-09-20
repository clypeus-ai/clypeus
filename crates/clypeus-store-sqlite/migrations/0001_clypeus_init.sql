-- Clypeus schema. Every table is partitioned by `scope_id` and `subject`;
-- queries in the store layer always filter by both.

CREATE TABLE IF NOT EXISTS clypeus_settings (
    scope_id TEXT PRIMARY KEY,
    provider_kind TEXT NOT NULL,
    base_url TEXT,
    default_model TEXT,
    timeout_ms INTEGER NOT NULL DEFAULT 60000,
    max_output_tokens INTEGER NOT NULL DEFAULT 1200,
    api_key_present INTEGER NOT NULL DEFAULT 0,
    profile_mode TEXT NOT NULL DEFAULT 'disabled',
    profile_custom TEXT,
    extensions TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS clypeus_threads (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    pinned INTEGER NOT NULL DEFAULT 0,
    model TEXT,
    active_leaf_message_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS clypeus_threads_scope_subject ON clypeus_threads (scope_id, subject);

CREATE TABLE IF NOT EXISTS clypeus_messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    parent_message_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    reasoning_content TEXT,
    tool_call_id TEXT,
    context_version TEXT,
    context_json TEXT,
    model TEXT,
    reasoning_level TEXT,
    status TEXT NOT NULL,
    error_detail TEXT,
    usage_json TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS clypeus_messages_thread ON clypeus_messages (thread_id, created_at);
CREATE INDEX IF NOT EXISTS clypeus_messages_siblings ON clypeus_messages (thread_id, parent_message_id);

CREATE TABLE IF NOT EXISTS clypeus_feedback (
    message_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    rating TEXT NOT NULL,
    comment TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clypeus_tool_calls (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    risk TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    arguments_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    auth_mode TEXT,
    expires_at TEXT,
    approval_kind TEXT,
    confirm_field TEXT,
    impact TEXT,
    approval_id TEXT,
    result_json TEXT,
    error_code TEXT,
    duration_ms INTEGER,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS clypeus_tool_calls_quota
    ON clypeus_tool_calls (scope_id, subject, tool_name, created_at);

CREATE TABLE IF NOT EXISTS clypeus_tool_approvals (
    id TEXT PRIMARY KEY,
    tool_call_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT,
    typed_confirm TEXT,
    arguments_hash TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS clypeus_tool_approvals_call
    ON clypeus_tool_approvals (tool_call_id);

CREATE TABLE IF NOT EXISTS clypeus_audit (
    id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    thread_id TEXT,
    message_id TEXT,
    tool_call_id TEXT,
    item_name TEXT NOT NULL,
    item_kind TEXT NOT NULL,
    risk TEXT NOT NULL,
    arguments_hash TEXT NOT NULL,
    arguments_redacted TEXT,
    scopes_used TEXT,
    decision TEXT,
    auth_mode TEXT,
    egress_service TEXT,
    egress_path_template TEXT,
    outcome TEXT NOT NULL,
    downstream_status INTEGER,
    result_bytes INTEGER,
    duration_ms INTEGER,
    approval_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS clypeus_audit_scope_time ON clypeus_audit (scope_id, created_at);
