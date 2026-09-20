//! Server-sent event contract.
//!
//! A turn stream closes with a terminal `turn_completed` event followed by
//! exactly one `data: [DONE]` sentinel. Text and reasoning flow as anonymous
//! `data:` chunks so any SSE client can render them; structured events use
//! named events:
//!
//! * `context` — the turn context snapshot, emitted before the first byte.
//! * `tool_call` — the model requested a tool call (`running` or
//!   `waiting_approval`).
//! * `tool_result` — the call finished, was denied, or expired.
//! * `turn_completed` — the persisted turn view.

use serde_json::{Map, Value, json};

use crate::context::TurnContext;
use crate::models::TokenUsage;
use crate::store::{Message, PersistedToolCall, Thread};

/// End-of-stream sentinel.
pub const DONE_EVENT: &[u8] = b"data: [DONE]\n\n";

/// Renders one named SSE event.
pub fn event(name: &str, payload: &Value) -> String {
    format!("event: {name}\ndata: {payload}\n\n")
}

/// Renders one anonymous `data:` chunk.
pub fn data(payload: &Value) -> String {
    format!("data: {payload}\n\n")
}

/// Incremental answer text.
pub fn content_delta(text: &str) -> String {
    data(&json!({"delta": {"content": text}}))
}

/// Incremental reasoning text.
pub fn reasoning_delta(text: &str) -> String {
    data(&json!({"delta": {"reasoning": text}}))
}

/// Cumulative token usage for the turn.
pub fn usage(usage: &TokenUsage) -> Option<String> {
    if usage.is_empty() {
        return None;
    }
    serde_json::to_value(usage)
        .ok()
        .map(|value| data(&json!({"usage": value})))
}

/// Turn context snapshot.
pub fn context(context: &TurnContext) -> String {
    event(
        "context",
        &json!({
            "schemaVersion": context.schema_version,
            "version": context.version,
            "fields": Value::Object(context.fields.clone()),
        }),
    )
}

/// Model-requested tool call.
pub fn tool_call(
    id: &str,
    name: &str,
    arguments: &Value,
    risk: &str,
    state: &str,
    approval: Option<&Value>,
) -> String {
    let mut payload = Map::new();
    payload.insert("id".into(), Value::String(id.to_string()));
    payload.insert("name".into(), Value::String(name.to_string()));
    payload.insert("arguments".into(), arguments.clone());
    payload.insert("risk".into(), Value::String(risk.to_string()));
    payload.insert("state".into(), Value::String(state.to_string()));
    if let Some(approval) = approval {
        payload.insert("approval".into(), approval.clone());
    }
    event("tool_call", &Value::Object(payload))
}

/// Tool call outcome.
pub fn tool_result(
    id: &str,
    name: &str,
    state: &str,
    result: Option<&Value>,
    error: Option<(&str, &str)>,
    duration_ms: u64,
) -> String {
    let mut payload = Map::new();
    payload.insert("id".into(), Value::String(id.to_string()));
    payload.insert("name".into(), Value::String(name.to_string()));
    payload.insert("state".into(), Value::String(state.to_string()));
    if let Some(result) = result {
        payload.insert("result".into(), result.clone());
    }
    if let Some((code, message)) = error {
        payload.insert("error".into(), json!({"code": code, "message": message}));
    }
    payload.insert("durationMs".into(), Value::from(duration_ms));
    event("tool_result", &Value::Object(payload))
}

/// Terminal persisted turn view.
pub fn turn_completed(thread: &Thread, user_message_id: uuid::Uuid, assistant: &Message) -> String {
    let payload = json!({
        "thread": thread,
        "userMessageId": user_message_id,
        "assistant": assistant,
    });
    event("turn_completed", &payload)
}

/// Stable error code for the client.
pub fn error(code: &str) -> String {
    data(&json!({"error": code}))
}

/// Serializes a persisted tool call for the approval card in a `tool_call`
/// event.
pub fn approval_value(call: &PersistedToolCall) -> Option<Value> {
    call.approval
        .as_ref()
        .and_then(|approval| serde_json::to_value(approval).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_events_render_the_expected_frame() {
        let rendered = tool_call("c1", "demo", &json!({}), "write", "running", None);
        assert!(rendered.starts_with("event: tool_call\ndata: "));
        assert!(rendered.ends_with("\n\n"));
        assert!(rendered.contains("\"state\":\"running\""));
    }

    #[test]
    fn usage_chunk_is_omitted_when_empty() {
        assert!(usage(&TokenUsage::default()).is_none());
        let rendered = usage(&TokenUsage {
            total_tokens: Some(7),
            ..TokenUsage::default()
        })
        .unwrap();
        assert!(rendered.contains("\"usage\""));
        assert!(rendered.contains("\"totalTokens\":7"));
    }

    #[test]
    fn done_sentinel_is_exact() {
        assert_eq!(DONE_EVENT, b"data: [DONE]\n\n");
    }
}
