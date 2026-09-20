//! System prompt profiles.
//!
//! A profile owns the system prompt an application sends before a turn. The
//! core ships no domain profile; applications register theirs (or use the
//! generic custom prompt). `ProfileSelection` is what a scope stores.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::principal::ScopeId;
use crate::store::StoreError;

/// How a scope selects its system prompt.
#[derive(Debug, Clone, PartialEq, Eq, Default, ToSchema)]
pub enum ProfileSelection {
    /// Use the registered built-in profile, optionally extended with
    /// application/operator instructions.
    Builtin { custom: Option<String> },
    /// Use the supplied prompt verbatim.
    Custom(String),
    /// Send no system prompt at all.
    #[default]
    Disabled,
}

#[derive(Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
enum ProfileMode {
    Builtin,
    Custom,
    Disabled,
}

#[derive(Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "camelCase")]
struct ProfileSelectionWire {
    mode: ProfileMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_instructions: Option<String>,
}

impl Serialize for ProfileSelection {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            Self::Builtin { custom } => ProfileSelectionWire {
                mode: ProfileMode::Builtin,
                custom_instructions: custom.clone(),
            },
            Self::Custom(prompt) => ProfileSelectionWire {
                mode: ProfileMode::Custom,
                custom_instructions: Some(prompt.clone()),
            },
            Self::Disabled => ProfileSelectionWire {
                mode: ProfileMode::Disabled,
                custom_instructions: None,
            },
        };
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ProfileSelection {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ProfileSelectionWire::deserialize(deserializer)?;
        Ok(match wire.mode {
            ProfileMode::Builtin => Self::Builtin {
                custom: wire.custom_instructions,
            },
            ProfileMode::Custom => Self::Custom(wire.custom_instructions.unwrap_or_default()),
            ProfileMode::Disabled => Self::Disabled,
        })
    }
}

impl ProfileSelection {
    pub fn builtin() -> Self {
        Self::Builtin { custom: None }
    }

    pub fn mode_wire(&self) -> &'static str {
        match self {
            Self::Builtin { .. } => "builtin",
            Self::Custom(_) => "custom",
            Self::Disabled => "disabled",
        }
    }
}

/// A versioned system prompt owned by an application.
pub trait PromptProfile: Send + Sync {
    /// Stable identifier.
    fn id(&self) -> &'static str;
    /// Prompt version. Bump whenever the prompt text changes.
    fn version(&self) -> u32;
    /// SHA-256 hex of the prompt text.
    fn prompt_hash(&self) -> String {
        crate::functions::sha256_hex(self.build(None).unwrap_or_default().as_bytes())
    }
    /// Renders the prompt. `None` means the profile is disabled.
    fn build(&self, custom: Option<&str>) -> Option<String>;
}

/// Loads and saves a scope's profile selection.
#[async_trait::async_trait]
pub trait ProfileStore: Send + Sync {
    async fn load(&self, scope: &ScopeId) -> Result<ProfileSelection, StoreError>;
    async fn save(&self, scope: &ScopeId, selection: &ProfileSelection) -> Result<(), StoreError>;
}

/// Profile store backed by an in-memory map.
#[derive(Debug, Default)]
pub struct StaticProfileStore {
    entries: std::sync::Mutex<std::collections::BTreeMap<String, ProfileSelection>>,
}

impl StaticProfileStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl ProfileStore for StaticProfileStore {
    async fn load(&self, scope: &ScopeId) -> Result<ProfileSelection, StoreError> {
        Ok(self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(scope.as_str())
            .cloned()
            .unwrap_or_default())
    }

    async fn save(&self, scope: &ScopeId, selection: &ProfileSelection) -> Result<(), StoreError> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(scope.to_string(), selection.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_selection_serializes_with_mode_tag() {
        let builtin = ProfileSelection::Builtin {
            custom: Some("be brief".into()),
        };
        let json = serde_json::to_value(&builtin).unwrap();
        assert_eq!(json["mode"], "builtin");
        assert_eq!(json["customInstructions"], "be brief");
        let back: ProfileSelection = serde_json::from_value(json).unwrap();
        assert_eq!(back, builtin);
        assert_eq!(ProfileSelection::Disabled.mode_wire(), "disabled");
    }
}
