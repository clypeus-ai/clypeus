//! Application composition root.

use std::sync::Arc;
use std::time::Duration;

use clypeus_core::audit::{AuditReader, AuditSink};
use clypeus_core::broker::ToolBroker;
use clypeus_core::context::{EmptyTurnContextProvider, TurnContextProvider};
use clypeus_core::functions::FunctionRunner;
use clypeus_core::guard::{GuardPolicy, NeutralGuardPolicy};
use clypeus_core::models::ProviderKind;
use clypeus_core::orchestrator::{Orchestrator, TurnLimits};
use clypeus_core::principal::{
    PolicyEngine, PrincipalResolver, ResolverChain, StaticPolicy, StaticTokenResolver,
};
use clypeus_core::profile::PromptProfile;
use clypeus_core::provider::ProviderRegistry;
use clypeus_core::rate_limit::{InMemoryRateLimiter, RateLimitConfig, RateLimiter};
use clypeus_core::secrets::SecretStore;
use clypeus_core::store::{
    ApprovalStore, ConversationStore, Readiness, ScopeSettingsStore, ScopeSettingsUpdate,
};
use clypeus_provider_anthropic::AnthropicProvider;
use clypeus_provider_openai::OpenAiProvider;
use clypeus_secret_file::FileSecretStore;
use clypeus_store_memory::MemoryStore;

use crate::config::ServerConfig;
use crate::demo;

/// Shared application state.
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub providers: Arc<ProviderRegistry>,
    pub resolver: Arc<dyn PrincipalResolver>,
    pub policy: Arc<dyn PolicyEngine>,
    pub settings: Arc<dyn ScopeSettingsStore>,
    pub conversation: Arc<dyn ConversationStore>,
    pub approvals: Arc<dyn ApprovalStore>,
    pub audit_reader: Arc<dyn AuditReader>,
    pub readiness: Arc<dyn Readiness>,
    pub secrets: Arc<dyn SecretStore>,
    pub broker: Arc<ToolBroker>,
    pub orchestrator: Arc<Orchestrator>,
    pub function_runner: Arc<FunctionRunner>,
    pub functions: Arc<clypeus_core::functions::FunctionRegistry>,
    pub chat_limiter: Arc<dyn RateLimiter>,
    pub guard: Arc<dyn GuardPolicy>,
    /// Built-in prompt profile; `None` means `ProfileSelection::Builtin`
    /// contributes only its custom text.
    pub prompt_profile: Option<Arc<dyn PromptProfile>>,
    pub context_provider: Arc<dyn TurnContextProvider>,
    pub metrics: metrics_exporter_prometheus::PrometheusHandle,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AppState")
            .field("bind_addr", &self.config.bind_addr)
            .field("store", &self.config.store)
            .field("providers", &self.providers.kinds())
            .finish_non_exhaustive()
    }
}

impl AppState {
    /// Builds the state from configuration.
    pub async fn build(
        config: ServerConfig,
        metrics: metrics_exporter_prometheus::PrometheusHandle,
    ) -> Result<Arc<Self>, String> {
        let config = Arc::new(config);

        // Store backend. Each concrete store implements every store trait;
        // the trait objects below share one instance.
        let service_bases = tool_service_bases()?;
        type StoreHandles = (
            Arc<dyn ScopeSettingsStore>,
            Arc<dyn ConversationStore>,
            Arc<dyn ApprovalStore>,
            Arc<dyn AuditSink>,
            Arc<dyn AuditReader>,
            Arc<dyn Readiness>,
        );
        let (settings, conversation, approvals, audit_sink, audit_reader, readiness): StoreHandles =
            match config.store {
                crate::config::StoreKind::Memory => {
                    let store = Arc::new(MemoryStore::new());
                    (
                        store.clone(),
                        store.clone(),
                        store.clone(),
                        store.clone(),
                        store.clone(),
                        store,
                    )
                }
                crate::config::StoreKind::Sqlite => {
                    let store = Arc::new(
                        clypeus_store_sqlite::connect(&config.sqlite_url())
                            .await
                            .map_err(|error| format!("sqlite: {error}"))?,
                    );
                    (
                        store.clone(),
                        store.clone(),
                        store.clone(),
                        store.clone(),
                        store.clone(),
                        store,
                    )
                }
                crate::config::StoreKind::Postgres => {
                    let url = config
                        .database_url
                        .as_deref()
                        .ok_or("CLYPEUS_DATABASE_URL is required for the postgres store")?;
                    let store = Arc::new(
                        clypeus_store_postgres::connect(url)
                            .await
                            .map_err(|error| format!("postgres: {error}"))?,
                    );
                    (
                        store.clone(),
                        store.clone(),
                        store.clone(),
                        store.clone(),
                        store.clone(),
                        store,
                    )
                }
            };

        // Providers.
        let providers = Arc::new(
            ProviderRegistry::new()
                .register(ProviderKind::Openai, Arc::new(OpenAiProvider::new()))
                .register(ProviderKind::Anthropic, Arc::new(AnthropicProvider::new())),
        );

        // Principal resolution.
        let mut chain = ResolverChain::new();
        if let Some(token) = config.static_token.clone() {
            let principal = clypeus_core::principal::Principal::new(
                config.static_scope.clone(),
                config.static_subject.clone(),
            )
            .with_scopes(config.static_scopes.clone());
            chain = chain.push(Arc::new(StaticTokenResolver::new(token, principal)));
        }
        if let Some(path) = config.jwks_path.clone() {
            let raw = tokio::fs::read_to_string(&path)
                .await
                .map_err(|error| format!("JWKS {path:?}: {error}"))?;
            let resolver = clypeus_auth_jwks::JwksResolver::from_jwks_json(
                &raw,
                clypeus_auth_jwks::JwksResolverConfig {
                    scope_claim: config.jwks_scope_claim.clone(),
                    subject_claim: config.jwks_subject_claim.clone(),
                    scopes_claim: config.jwks_scopes_claim.clone(),
                    issuer: config.jwks_issuer.clone(),
                    audience: config.jwks_audience.clone(),
                    ..clypeus_auth_jwks::JwksResolverConfig::default()
                },
            )
            .map_err(|error| format!("JWKS: {error}"))?;
            chain = chain.push(Arc::new(resolver));
        }
        if config.static_token.is_none() && config.jwks_path.is_none() {
            tracing::warn!(
                "no principal resolver configured; set CLYPEUS_STATIC_TOKEN or CLYPEUS_JWKS_PATH"
            );
        }
        let resolver: Arc<dyn PrincipalResolver> = Arc::new(chain);

        let policy: Arc<dyn PolicyEngine> =
            Arc::new(StaticPolicy::new().with_admin_scopes(config.admin_scopes.clone()));

        // Secrets and rate limits.
        let secrets: Arc<dyn SecretStore> =
            Arc::new(FileSecretStore::new(config.secret_dir.clone()));
        let chat_limiter: Arc<dyn RateLimiter> =
            Arc::new(InMemoryRateLimiter::new(RateLimitConfig {
                max_requests: config.core.chat_rate_limit,
                window: config.core.chat_rate_window,
            }));
        let function_limiter: Arc<dyn RateLimiter> =
            Arc::new(InMemoryRateLimiter::new(RateLimitConfig {
                max_requests: config.core.function_rate_limit,
                window: config.core.chat_rate_window,
            }));
        let guard: Arc<dyn GuardPolicy> = Arc::new(NeutralGuardPolicy);

        // Tools and functions.
        let (tools, functions) = demo::registries();
        let broker = Arc::new(
            ToolBroker::new(
                Arc::clone(&tools),
                Arc::clone(&approvals),
                Arc::clone(&audit_sink),
                reqwest::Client::new(),
                service_bases,
                None,
                None,
                config.core.write_quota_per_day,
            )
            .map_err(|error| format!("tool broker: {error:?}"))?,
        );

        let orchestrator = Arc::new(Orchestrator::new(
            Arc::clone(&providers),
            Arc::clone(&conversation),
            Arc::clone(&broker),
            Arc::clone(&guard),
        ));

        let function_runner = Arc::new(
            FunctionRunner::new(
                Arc::clone(&providers),
                Arc::clone(&settings),
                Arc::clone(&secrets),
                Arc::clone(&audit_sink),
                Arc::clone(&function_limiter),
                Arc::clone(&guard),
            )
            .with_allow_private_targets(config.core.allow_private_providers),
        );

        let state = Arc::new(Self {
            config,
            providers,
            resolver,
            policy,
            settings,
            conversation: Arc::clone(&conversation),
            approvals: Arc::clone(&approvals),
            audit_reader: Arc::clone(&audit_reader),
            readiness,
            secrets,
            broker,
            orchestrator,
            function_runner,
            functions,
            chat_limiter,
            guard,
            prompt_profile: None,
            context_provider: Arc::new(EmptyTurnContextProvider),
            metrics,
        });

        state.seed().await?;
        state.spawn_maintenance();
        for identity in state.functions.identities() {
            tracing::info!(function = %identity, "registered AI function");
        }
        Ok(state)
    }

    /// Applies the startup seed for a fresh deployment.
    async fn seed(&self) -> Result<(), String> {
        let Some(seed) = self.config.seed.clone() else {
            return Ok(());
        };
        let mut update = ScopeSettingsUpdate {
            provider_kind: Some(seed.provider_kind),
            base_url: Some(seed.base_url.clone()),
            default_model: seed.default_model.clone(),
            ..ScopeSettingsUpdate::default()
        };
        if !seed.api_key.is_empty() {
            update.api_key_present = Some(true);
        }
        update.extensions = Some(serde_json::json!({
            "scopeId": seed.scope.to_string(),
        }));
        self.settings
            .upsert(&seed.scope, update)
            .await
            .map_err(|error| format!("seed settings: {error}"))?;
        if !seed.api_key.is_empty() {
            self.secrets
                .put(
                    &seed.scope,
                    clypeus_core::functions::PROVIDER_API_KEY,
                    clypeus_core::secrets::SecretString::new(seed.api_key.clone()),
                )
                .await
                .map_err(|error| format!("seed secret: {error}"))?;
        }
        tracing::info!(scope = %seed.scope, "seeded provider settings");
        Ok(())
    }

    /// Background maintenance: stale turns, expired approvals, audit retention.
    fn spawn_maintenance(self: &Arc<Self>) {
        let state = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                let stale_cutoff = chrono::Utc::now()
                    - chrono::Duration::seconds(
                        i64::try_from(state.config.stale_turn_seconds).unwrap_or(600),
                    );
                match state
                    .conversation
                    .finalize_stale_turns(stale_cutoff, "turn_incomplete")
                    .await
                {
                    Ok(0) => {}
                    Ok(count) => tracing::warn!(count, "finalized stale turns"),
                    Err(error) => tracing::error!(%error, "stale-turn sweep failed"),
                }
                match state.approvals.expire_tool_calls(chrono::Utc::now()).await {
                    Ok(0) => {}
                    Ok(count) => tracing::info!(count, "expired parked approvals"),
                    Err(error) => tracing::error!(%error, "approval expiry sweep failed"),
                }
                if state.config.core.audit_retention_days > 0 {
                    let cutoff = chrono::Utc::now()
                        - chrono::Duration::days(
                            i64::try_from(state.config.core.audit_retention_days).unwrap_or(90),
                        );
                    match state.audit_reader.purge_before(cutoff).await {
                        Ok(0) => {}
                        Ok(count) => tracing::info!(count, "purged expired audit records"),
                        Err(error) => tracing::error!(%error, "audit purge failed"),
                    }
                }
            }
        });
    }

    /// Turn limits for the orchestrator.
    pub fn turn_limits(&self) -> TurnLimits {
        self.config.core.limits
    }
}

/// Parses `CLYPEUS_EGRESS_SERVICES` (`name=https://base,name2=...`).
fn tool_service_bases() -> Result<Vec<(String, String)>, String> {
    let raw = std::env::var("CLYPEUS_EGRESS_SERVICES").unwrap_or_default();
    let mut services = Vec::new();
    for entry in raw.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let Some((name, url)) = entry.split_once('=') else {
            return Err(format!(
                "CLYPEUS_EGRESS_SERVICES entry is not name=url: {entry}"
            ));
        };
        services.push((name.trim().to_string(), url.trim().to_string()));
    }
    Ok(services)
}
