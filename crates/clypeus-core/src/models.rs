//! Wire-level vocabulary shared by the API, the orchestrator and the stores.
//!
//! Names here are provider-agnostic: `ChatMessage` is the common shape both
//! the OpenAI-compatible and the Anthropic adapters translate to and from.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Provider backend a scope is configured to use.
#[derive(
    Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord, Serialize, Deserialize, ToSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Openai,
    Anthropic,
}

impl ProviderKind {
    pub fn as_wire(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "openai" | "open_ai" => Some(Self::Openai),
            "anthropic" | "claude" => Some(Self::Anthropic),
            _ => None,
        }
    }
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_wire())
    }
}

/// Chat message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    User,
    Assistant,
    System,
    Tool,
}

impl ChatRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Tool => "tool",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "user" => Some(Self::User),
            "assistant" => Some(Self::Assistant),
            "system" => Some(Self::System),
            "tool" => Some(Self::Tool),
            _ => None,
        }
    }
}

/// One chat message used in the request and response flow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatMessage {
    pub fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }

    pub fn assistant_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
            tool_call_id: None,
            tool_calls: Some(tool_calls),
        }
    }
}

/// One provider-requested tool call with structured arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Provider-facing tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// Strict JSON Schema (`additionalProperties: false`).
    pub input_schema: serde_json::Value,
}

/// Token accounting for one provider round or one whole turn.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
pub struct TokenUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<i32>,
}

impl TokenUsage {
    pub fn is_empty(&self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.total_tokens.is_none()
            && self.cached_tokens.is_none()
            && self.reasoning_tokens.is_none()
    }

    /// Adds another round's usage field by field. Multi-round tool turns bill
    /// every provider call, so usage is summed per round.
    pub fn accumulate(&mut self, other: &Self) {
        fn add(target: &mut Option<i32>, value: Option<i32>) {
            if let Some(value) = value {
                *target = Some(target.unwrap_or(0).saturating_add(value));
            }
        }
        add(&mut self.prompt_tokens, other.prompt_tokens);
        add(&mut self.completion_tokens, other.completion_tokens);
        add(&mut self.total_tokens, other.total_tokens);
        add(&mut self.cached_tokens, other.cached_tokens);
        add(&mut self.reasoning_tokens, other.reasoning_tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_kind_round_trips_and_parses_aliases() {
        for kind in [ProviderKind::Openai, ProviderKind::Anthropic] {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(
                ProviderKind::parse(json.trim_matches('"')),
                Some(kind),
                "wire value parses back"
            );
        }
        assert_eq!(ProviderKind::parse("claude"), Some(ProviderKind::Anthropic));
        assert_eq!(ProviderKind::parse("OPEN_AI"), Some(ProviderKind::Openai));
        assert_eq!(ProviderKind::parse("other"), None);
    }

    #[test]
    fn chat_message_omits_absent_tool_fields() {
        let plain = ChatMessage::text(ChatRole::User, "hi");
        let json = serde_json::to_string(&plain).unwrap();
        assert!(!json.contains("toolCallId"));
        assert!(!json.contains("toolCalls"));

        let call = ToolCall {
            id: "call_1".into(),
            name: "echo".into(),
            arguments: serde_json::json!({"message": "ping"}),
        };
        let assistant = ChatMessage::assistant_tool_calls("", vec![call]);
        let json = serde_json::to_string(&assistant).unwrap();
        assert!(json.contains("toolCalls"));
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, assistant);
    }

    #[test]
    fn token_usage_accumulates_across_rounds() {
        let mut total = TokenUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            cached_tokens: Some(2),
            reasoning_tokens: None,
        };
        total.accumulate(&TokenUsage {
            prompt_tokens: Some(20),
            completion_tokens: Some(7),
            total_tokens: Some(27),
            cached_tokens: None,
            reasoning_tokens: Some(4),
        });
        assert_eq!(total.prompt_tokens, Some(30));
        assert_eq!(total.completion_tokens, Some(12));
        assert_eq!(total.total_tokens, Some(42));
        assert_eq!(total.cached_tokens, Some(2));
        assert_eq!(total.reasoning_tokens, Some(4));
    }
}
