//! File and environment secret store.
//!
//! Secrets are addressed by `(scope, key)`:
//!
//! * environment first: `CLYPEUS_SECRET_<SCOPE>_<KEY>` with every character
//!   outside `[A-Za-z0-9]` replaced by `_` and uppercased;
//! * then a file at `<dir>/<scope>/<key>`, read as UTF-8 and trimmed.
//!
//! Writes are supported so the settings API can store provider keys; when no
//! directory is configured, `put`/`delete` fail closed.

use std::path::PathBuf;

use clypeus_core::principal::ScopeId;
use clypeus_core::secrets::{SecretError, SecretStore, SecretString};

fn env_name(scope: &ScopeId, key: &str) -> String {
    let sanitize = |value: &str| {
        value
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character.to_ascii_uppercase()
                } else {
                    '_'
                }
            })
            .collect::<String>()
    };
    format!(
        "CLYPEUS_SECRET_{}_{}",
        sanitize(scope.as_str()),
        sanitize(key)
    )
}

/// Secret store backed by the process environment and an optional directory.
#[derive(Debug, Clone, Default)]
pub struct FileSecretStore {
    dir: Option<PathBuf>,
}

impl FileSecretStore {
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self { dir }
    }

    /// Reads `CLYPEUS_SECRET_DIR` when set.
    pub fn from_env() -> Self {
        Self {
            dir: std::env::var("CLYPEUS_SECRET_DIR")
                .ok()
                .filter(|dir| !dir.trim().is_empty())
                .map(PathBuf::from),
        }
    }

    fn path(&self, scope: &ScopeId, key: &str) -> Result<PathBuf, SecretError> {
        let dir = self
            .dir
            .as_ref()
            .ok_or_else(|| SecretError::new("secret directory is not configured"))?;
        if scope.as_str().contains("..") || key.contains("..") || key.contains('/') {
            return Err(SecretError::new("invalid secret path"));
        }
        Ok(dir.join(scope.as_str()).join(key))
    }
}

#[async_trait::async_trait]
impl SecretStore for FileSecretStore {
    async fn get(&self, scope: &ScopeId, key: &str) -> Result<Option<SecretString>, SecretError> {
        let name = env_name(scope, key);
        if let Ok(value) = std::env::var(&name)
            && !value.is_empty()
        {
            return Ok(Some(SecretString::new(value)));
        }
        let path = match self.path(scope, key) {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        match tokio::fs::read_to_string(&path).await {
            Ok(value) => Ok(Some(SecretString::new(value.trim().to_string()))),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(SecretError::new(error.to_string())),
        }
    }

    async fn put(
        &self,
        scope: &ScopeId,
        key: &str,
        value: SecretString,
    ) -> Result<(), SecretError> {
        let path = self.path(scope, key)?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| SecretError::new(error.to_string()))?;
        }
        tokio::fs::write(&path, value.expose())
            .await
            .map_err(|error| SecretError::new(error.to_string()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = std::fs::Permissions::from_mode(0o600);
            let _ = std::fs::set_permissions(&path, permissions);
        }
        Ok(())
    }

    async fn delete(&self, scope: &ScopeId, key: &str) -> Result<(), SecretError> {
        let path = self.path(scope, key)?;
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(SecretError::new(error.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_names_are_sanitized() {
        assert_eq!(
            env_name(&ScopeId::new("tenant-1"), "provider_api_key"),
            "CLYPEUS_SECRET_TENANT_1_PROVIDER_API_KEY"
        );
    }

    #[tokio::test]
    async fn files_round_trip_and_missing_returns_none() {
        let dir = std::env::temp_dir().join(format!("clypeus-secrets-{}", std::process::id()));
        let store = FileSecretStore::new(Some(dir.clone()));
        let scope = ScopeId::new("scope-a");
        assert!(store.get(&scope, "key").await.unwrap().is_none());
        store
            .put(&scope, "key", SecretString::new("value"))
            .await
            .unwrap();
        assert_eq!(
            store.get(&scope, "key").await.unwrap().unwrap().expose(),
            "value"
        );
        store.delete(&scope, "key").await.unwrap();
        assert!(store.get(&scope, "key").await.unwrap().is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn writes_fail_closed_without_a_directory() {
        let store = FileSecretStore::new(None);
        assert!(
            store
                .put(&ScopeId::new("s"), "k", SecretString::new("v"))
                .await
                .is_err()
        );
    }
}
