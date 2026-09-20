//! Secret strings and the secret store abstraction.

use serde::{Deserialize, Serialize};

use crate::principal::ScopeId;

/// A string that must never appear in logs, audit records, or error messages.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the underlying value. Call sites should be few and deliberate:
    /// provider requests, egress headers, and secret comparisons.
    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SecretString([redacted])")
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

/// Secret storage failure.
#[derive(Debug, Clone, thiserror::Error)]
#[error("secret store error: {message}")]
pub struct SecretError {
    pub message: String,
}

impl SecretError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Per-scope secret storage. Implementations must never log values.
#[async_trait::async_trait]
pub trait SecretStore: Send + Sync {
    async fn get(&self, scope: &ScopeId, key: &str) -> Result<Option<SecretString>, SecretError>;
    async fn put(&self, scope: &ScopeId, key: &str, value: SecretString)
    -> Result<(), SecretError>;
    async fn delete(&self, scope: &ScopeId, key: &str) -> Result<(), SecretError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_debug_is_redacted_and_round_trips_on_the_wire() {
        let secret = SecretString::new("sk-live-123");
        assert_eq!(format!("{secret:?}"), "SecretString([redacted])");
        assert!(!format!("{secret:?}").contains("123"));
        assert_eq!(secret.expose(), "sk-live-123");
        let json = serde_json::to_string(&secret).unwrap();
        assert_eq!(json, r#""sk-live-123""#);
        let back: SecretString = serde_json::from_str(&json).unwrap();
        assert_eq!(back.expose(), "sk-live-123");
    }
}
