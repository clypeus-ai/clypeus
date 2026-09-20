//! Standalone server configuration, read from `CLYPEUS_*` variables.

use std::net::SocketAddr;
use std::path::PathBuf;

use clypeus_core::config::CoreConfig;
use clypeus_core::models::ProviderKind;
use clypeus_core::principal::ScopeId;

/// Which store backend to open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreKind {
    Memory,
    Sqlite,
    Postgres,
}

impl StoreKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "memory" | "mem" => Some(Self::Memory),
            "sqlite" => Some(Self::Sqlite),
            "postgres" | "postgresql" | "pg" => Some(Self::Postgres),
            _ => None,
        }
    }
}

/// Scope seeded at startup so a fresh deployment has a provider to talk to.
#[derive(Debug, Clone)]
pub struct SeedScope {
    pub scope: ScopeId,
    pub provider_kind: ProviderKind,
    pub base_url: String,
    pub api_key: String,
    pub default_model: Option<String>,
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: SocketAddr,
    pub store: StoreKind,
    pub database_url: Option<String>,
    pub sqlite_path: PathBuf,
    pub static_token: Option<String>,
    pub static_scope: ScopeId,
    pub static_subject: String,
    pub static_scopes: Vec<String>,
    pub admin_scopes: Vec<String>,
    pub jwks_path: Option<PathBuf>,
    pub jwks_scope_claim: String,
    pub jwks_subject_claim: String,
    pub jwks_scopes_claim: String,
    pub jwks_issuer: Option<String>,
    pub jwks_audience: Option<String>,
    pub secret_dir: Option<PathBuf>,
    pub seed: Option<SeedScope>,
    pub core: CoreConfig,
    pub stale_turn_seconds: u64,
}

fn env_string(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_list(name: &str) -> Vec<String> {
    env_string(name)
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

impl ServerConfig {
    pub fn from_env() -> Result<Self, String> {
        let core = CoreConfig::from_env();
        let store = env_string("CLYPEUS_STORE")
            .as_deref()
            .and_then(StoreKind::parse)
            .unwrap_or(StoreKind::Memory);
        let bind_addr = env_string("CLYPEUS_BIND_ADDR")
            .unwrap_or_else(|| "0.0.0.0:8080".to_string())
            .parse()
            .map_err(|error| format!("CLYPEUS_BIND_ADDR: {error}"))?;
        let provider_kind = env_string("CLYPEUS_SEED_PROVIDER")
            .as_deref()
            .and_then(ProviderKind::parse)
            .unwrap_or(ProviderKind::Openai);
        let seed = match (
            env_string("CLYPEUS_SEED_SCOPE"),
            env_string("CLYPEUS_SEED_BASE_URL"),
        ) {
            (Some(scope), Some(base_url)) => Some(SeedScope {
                scope: ScopeId::new(scope),
                provider_kind,
                base_url,
                api_key: env_string("CLYPEUS_SEED_API_KEY").unwrap_or_default(),
                default_model: env_string("CLYPEUS_SEED_MODEL"),
            }),
            _ => None,
        };
        Ok(Self {
            bind_addr,
            store,
            database_url: env_string("CLYPEUS_DATABASE_URL"),
            sqlite_path: env_string("CLYPEUS_SQLITE_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("clypeus.db")),
            static_token: env_string("CLYPEUS_STATIC_TOKEN"),
            static_scope: ScopeId::new(
                env_string("CLYPEUS_STATIC_SCOPE").unwrap_or_else(|| "default".to_string()),
            ),
            static_subject: env_string("CLYPEUS_STATIC_SUBJECT")
                .unwrap_or_else(|| "standalone".to_string()),
            static_scopes: env_list("CLYPEUS_STATIC_SCOPES"),
            admin_scopes: env_list("CLYPEUS_ADMIN_SCOPES"),
            jwks_path: env_string("CLYPEUS_JWKS_PATH").map(PathBuf::from),
            jwks_scope_claim: env_string("CLYPEUS_JWKS_SCOPE_CLAIM")
                .unwrap_or_else(|| "scope_id".to_string()),
            jwks_subject_claim: env_string("CLYPEUS_JWKS_SUBJECT_CLAIM")
                .unwrap_or_else(|| "sub".to_string()),
            jwks_scopes_claim: env_string("CLYPEUS_JWKS_SCOPES_CLAIM")
                .unwrap_or_else(|| "scopes".to_string()),
            jwks_issuer: env_string("CLYPEUS_JWKS_ISSUER"),
            jwks_audience: env_string("CLYPEUS_JWKS_AUDIENCE"),
            secret_dir: env_string("CLYPEUS_SECRET_DIR").map(PathBuf::from),
            seed,
            core,
            stale_turn_seconds: env_string("CLYPEUS_STALE_TURN_SECONDS")
                .and_then(|value| value.parse().ok())
                .unwrap_or(600),
        })
    }

    pub fn sqlite_url(&self) -> String {
        format!("sqlite://{}?mode=rwc", self.sqlite_path.to_string_lossy())
    }
}
