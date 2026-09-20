//! Tool execution broker.
//!
//! The broker is the only component that turns a provider tool request into an
//! upstream call. It authorizes against the caller's scopes, binds one fixed
//! egress target, attaches either the caller's token or a per-call minted
//! token, enforces the turn budget, and writes the operational row plus the
//! append-only audit record. It has no secret store, no SQL beyond the tool
//! tables, and no way to build an upstream URL from model arguments.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use url::Url;
use uuid::Uuid;

use crate::audit::{AuditItemKind, AuditRecord, AuditSink};
use crate::metrics;
use crate::models::ToolSpec;
use crate::principal::{Principal, ScopeId};
use crate::secrets::SecretString;
use crate::store::{
    ApprovalStore, NewToolCall, ToolCallCompletion, ToolCallSnapshot, ToolCallStatus,
};
use crate::tools::{
    Approval, Egress, EgressAuth, EgressResponse, Tool, ToolDefinition, ToolEgress, ToolError,
    ToolExecContext, ToolRegistry, canonicalize_arguments, redact_secrets,
};

/// Lifetime of a per-call minted token. One tool call is the whole lifetime.
pub const BROKER_TOKEN_TTL: Duration = Duration::from_secs(30);

/// How long a parked write-tool call may wait for a decision.
pub const APPROVAL_TTL: Duration = Duration::from_secs(300);

/// Rolling window of the per-tool write quota.
pub const WRITE_QUOTA_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);

/// How many times the exact same `(tool, canonical arguments)` pair may run in
/// one turn. The provider may legitimately retry a failed call once; a longer
/// loop is refused without touching upstream.
pub const MAX_IDENTICAL_CALLS_PER_TURN: usize = 2;

/// The approval a park decision waits for, with everything the UI needs to
/// render the confirmation card.
#[derive(Debug, Clone)]
pub struct ApprovalChallenge {
    pub kind: Approval,
    pub expires_at: DateTime<Utc>,
    pub arguments_hash: String,
    pub confirm_field: Option<String>,
    pub impact: String,
}

impl ApprovalChallenge {
    pub fn to_value(&self) -> Value {
        let mut approval = json!({
            "kind": self.kind.as_str(),
            "expiresAtUtc": self.expires_at.to_rfc3339(),
            "argumentsHash": self.arguments_hash,
            "impact": self.impact,
        });
        if let Some(field) = self.confirm_field.as_deref() {
            approval["confirmField"] = Value::String(field.to_string());
        }
        approval
    }
}

/// Proof that one parked call was approved. Execution fails closed unless the
/// hash of the arguments still matches the hash the caller approved.
#[derive(Debug, Clone)]
pub struct ApprovalGrant {
    pub approval_id: String,
    pub arguments_hash: String,
}

/// Shared per-turn budget: call count, projected result bytes, repeated-call
/// guard, and the turn deadline.
pub struct TurnToolBudget {
    max_calls: usize,
    max_result_bytes: usize,
    max_repeats: usize,
    deadline: Instant,
    calls: AtomicUsize,
    result_bytes: AtomicUsize,
    signatures: Mutex<HashMap<String, usize>>,
}

impl std::fmt::Debug for TurnToolBudget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TurnToolBudget")
            .field("max_calls", &self.max_calls)
            .field("max_result_bytes", &self.max_result_bytes)
            .field("expired", &self.expired())
            .finish_non_exhaustive()
    }
}

impl TurnToolBudget {
    pub fn new(max_calls: usize, max_result_bytes: usize, budget: Duration) -> Self {
        Self {
            max_calls,
            max_result_bytes,
            max_repeats: MAX_IDENTICAL_CALLS_PER_TURN,
            deadline: Instant::now() + budget,
            calls: AtomicUsize::new(0),
            result_bytes: AtomicUsize::new(0),
            signatures: Mutex::new(HashMap::new()),
        }
    }

    pub fn expired(&self) -> bool {
        self.deadline
            .saturating_duration_since(Instant::now())
            .is_zero()
    }

    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    fn try_reserve_call(&self) -> Result<(), ToolError> {
        if self.expired() {
            return Err(ToolError::Timeout);
        }
        let used = self.calls.fetch_add(1, Ordering::SeqCst);
        if used >= self.max_calls {
            return Err(ToolError::LimitExceeded);
        }
        Ok(())
    }

    /// Loop guard: identical calls are bounded per turn. The key is the
    /// canonical tool name plus the canonical argument hash.
    fn try_reserve_signature(&self, name: &str, arguments_hash: &str) -> Result<(), ToolError> {
        let key = format!("{name}:{arguments_hash}");
        let mut signatures = self
            .signatures
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let used = signatures.entry(key).or_insert(0);
        if *used >= self.max_repeats {
            return Err(ToolError::LimitExceeded);
        }
        *used += 1;
        Ok(())
    }

    fn try_reserve_bytes(&self, bytes: usize) -> Result<(), ToolError> {
        let used = self.result_bytes.fetch_add(bytes, Ordering::SeqCst);
        if used + bytes > self.max_result_bytes {
            return Err(ToolError::LimitExceeded);
        }
        Ok(())
    }
}

/// Identity, authority, and correlation for one tool call.
#[derive(Clone, Debug)]
pub struct Caller {
    pub principal: Principal,
    pub request_id: String,
    pub thread_id: Uuid,
    pub message_id: Uuid,
    pub budget: Arc<TurnToolBudget>,
}

impl Caller {
    pub fn new(
        principal: Principal,
        request_id: impl Into<String>,
        thread_id: Uuid,
        message_id: Uuid,
        budget: Arc<TurnToolBudget>,
    ) -> Self {
        Self {
            principal,
            request_id: request_id.into(),
            thread_id,
            message_id,
            budget,
        }
    }

    pub fn scope(&self) -> &ScopeId {
        &self.principal.scope
    }

    pub fn subject(&self) -> &str {
        &self.principal.subject
    }

    pub fn has_scope(&self, scope: &str) -> bool {
        self.principal.has_scope(scope)
    }

    pub fn deadline(&self) -> Instant {
        self.budget.deadline()
    }
}

/// One requested tool call with its provider id.
#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Result of one brokered call.
#[derive(Debug)]
pub enum ToolCallOutcome {
    Succeeded { result: Value, duration_ms: u64 },
    Failed { error: ToolError, duration_ms: u64 },
}

impl ToolCallOutcome {
    pub fn duration_ms(&self) -> u64 {
        match self {
            Self::Succeeded { duration_ms, .. } | Self::Failed { duration_ms, .. } => *duration_ms,
        }
    }

    pub fn error(&self) -> Option<&ToolError> {
        match self {
            Self::Failed { error, .. } => Some(error),
            Self::Succeeded { .. } => None,
        }
    }

    pub fn succeeded(&self) -> bool {
        matches!(self, Self::Succeeded { .. })
    }
}

/// Issues narrow tokens for tools that cannot pass the caller's own token.
#[async_trait::async_trait]
pub trait TokenMinter: Send + Sync {
    async fn mint(
        &self,
        principal: &Principal,
        scopes: &[&str],
        audience: &str,
    ) -> Result<SecretString, ToolError>;
}

/// Exchanges the caller's own token for a short-lived token scoped to one
/// downstream audience.
#[async_trait::async_trait]
pub trait UserTokenExchanger: Send + Sync {
    async fn exchange(
        &self,
        principal: &Principal,
        audience: &str,
        scope: &ScopeId,
    ) -> Result<SecretString, ToolError>;
}

/// Fixed egress allowlist. Every declared service maps to exactly one parsed
/// base URL at construction time; the scheme, host, and port can never be
/// influenced by model arguments.
#[derive(Clone, Debug)]
pub struct EgressAllowlist {
    bases: HashMap<String, Url>,
}

impl EgressAllowlist {
    /// Builds the allowlist from the deployment's service base URLs. Invalid
    /// or empty entries are hard errors: a misconfigured base must fail at
    /// startup, not degrade a tool call later.
    pub fn new<I, S>(service_bases: I) -> Result<Self, ToolError>
    where
        I: IntoIterator<Item = (S, String)>,
        S: Into<String>,
    {
        let mut bases = HashMap::new();
        for (service, base) in service_bases {
            let service = service.into();
            let service = service.trim().to_string();
            let trimmed = base.trim();
            if service.is_empty() || trimmed.is_empty() {
                tracing::error!(service, "empty tool egress base URL");
                return Err(ToolError::Internal);
            }
            let parsed = Url::parse(trimmed).map_err(|_| {
                tracing::error!(service, "tool egress base URL is not a valid URL");
                ToolError::Internal
            })?;
            if !matches!(parsed.scheme(), "http" | "https")
                || parsed.host_str().is_none()
                || parsed.query().is_some()
                || parsed.fragment().is_some()
            {
                tracing::error!(
                    service,
                    "tool egress base URL is not a plain http(s) origin"
                );
                return Err(ToolError::Internal);
            }
            bases.insert(service, parsed);
        }
        Ok(Self { bases })
    }

    /// Registered services, for diagnostics and tests.
    pub fn services(&self) -> Vec<&str> {
        let mut services: Vec<&str> = self.bases.keys().map(String::as_str).collect();
        services.sort_unstable();
        services
    }

    /// Resolves one declared egress into a concrete URL. The only inputs are
    /// the code-declared service/method/path template plus allowlisted path
    /// parameters and query pairs.
    pub fn resolve(
        &self,
        egress: Egress,
        path_params: &[(&str, &str)],
        query: &[(&str, String)],
    ) -> Result<Url, ToolError> {
        let base = self.bases.get(egress.service).ok_or_else(|| {
            tracing::warn!(
                service = egress.service,
                "tool egress service is not allowlisted"
            );
            ToolError::Internal
        })?;
        if !matches!(egress.method, "GET" | "POST") {
            return Err(ToolError::Internal);
        }
        if !is_safe_path_template(egress.path_template) {
            tracing::error!(
                path = egress.path_template,
                "tool egress path template is not a fixed relative path"
            );
            return Err(ToolError::Internal);
        }

        let mut path = egress.path_template.to_string();
        for (key, value) in path_params {
            if !is_safe_path_param(value) {
                metrics::record_injection_blocked("path_param");
                tracing::warn!(key, "tool path parameter is not an unreserved identifier");
                return Err(ToolError::InvalidArguments(format!(
                    "'{key}' is not a valid identifier"
                )));
            }
            path = path.replace(&format!("{{{key}}}"), value);
        }
        if path.contains('{') || path.contains('}') {
            return Err(ToolError::Internal);
        }

        let base_path = base.path().trim_end_matches('/');
        let mut url = base.clone();
        url.set_path(&format!("{base_path}{path}"));
        if query.is_empty() {
            url.set_query(None);
        } else {
            let mut pairs = url.query_pairs_mut();
            for (key, value) in query {
                pairs.append_pair(key, value);
            }
            drop(pairs);
        }

        if url.scheme() != base.scheme()
            || url.host_str() != base.host_str()
            || url.port_or_known_default() != base.port_or_known_default()
        {
            tracing::error!(
                service = egress.service,
                "resolved tool egress left its origin"
            );
            return Err(ToolError::Internal);
        }
        Ok(url)
    }
}

fn is_safe_path_template(template: &str) -> bool {
    template.starts_with('/')
        && !template.contains("//")
        && !template.contains("..")
        && !template.contains('\\')
        && !template.contains('?')
        && !template.contains('#')
        && !template.contains("://")
        && template.len() <= 512
        && !template.chars().any(char::is_control)
}

fn is_safe_path_param(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~'))
}

/// HTTP egress bound to one declared target and one credential.
pub struct HttpEgress {
    http: reqwest::Client,
    allowlist: EgressAllowlist,
    bound: Egress,
    auth: EgressAuthHeader,
    timeout: Duration,
}

#[derive(Debug, Clone)]
enum EgressAuthHeader {
    Bearer(String),
    None,
}

impl std::fmt::Debug for HttpEgress {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HttpEgress")
            .field("service", &self.bound.service)
            .field("method", &self.bound.method)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl ToolEgress for HttpEgress {
    async fn request(
        &self,
        egress: Egress,
        path_params: &[(&str, &str)],
        query: &[(&str, String)],
        body: Option<Value>,
    ) -> Result<EgressResponse, ToolError> {
        if egress != self.bound {
            tracing::error!(
                requested = egress.service,
                bound = self.bound.service,
                "tool requested an egress target other than its declaration"
            );
            return Err(ToolError::Internal);
        }
        let url = self.allowlist.resolve(egress, path_params, query)?;
        let mut request = match egress.method {
            "GET" => self.http.get(url),
            "POST" => self.http.post(url),
            _ => return Err(ToolError::Internal),
        }
        .timeout(self.timeout)
        .header("accept", "application/json");
        if let EgressAuthHeader::Bearer(token) = &self.auth {
            request = request.bearer_auth(token);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request.send().await.map_err(|error| {
            if error.is_timeout() {
                ToolError::Timeout
            } else {
                ToolError::UpstreamUnavailable
            }
        })?;
        let status = response.status();
        if status.is_client_error() && status.as_u16() == 404 {
            return Err(ToolError::NotFound);
        }
        if status.as_u16() == 429 {
            return Err(ToolError::RateLimited);
        }
        if status.is_server_error() {
            return Err(ToolError::UpstreamStatus {
                status: status.as_u16(),
            });
        }
        if !status.is_success() {
            return Err(ToolError::UpstreamStatus {
                status: status.as_u16(),
            });
        }
        let body = response.bytes().await.map_err(|error| {
            if error.is_timeout() {
                ToolError::Timeout
            } else {
                ToolError::UpstreamUnavailable
            }
        })?;
        Ok(EgressResponse {
            status: status.as_u16(),
            body,
        })
    }
}

/// The broker owns the registry, the stores, the egress allowlist, the
/// optional token minter/exchanger, and the write quota.
pub struct ToolBroker {
    registry: Arc<ToolRegistry>,
    approvals: Arc<dyn ApprovalStore>,
    audit: Arc<dyn AuditSink>,
    http: reqwest::Client,
    allowlist: EgressAllowlist,
    minter: Option<Arc<dyn TokenMinter>>,
    exchanger: Option<Arc<dyn UserTokenExchanger>>,
    write_quota: Option<u32>,
}

impl std::fmt::Debug for ToolBroker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ToolBroker")
            .field(
                "tools",
                &self
                    .registry
                    .list()
                    .iter()
                    .map(|t| t.name())
                    .collect::<Vec<_>>(),
            )
            .field("services", &self.allowlist.services())
            .field("write_quota", &self.write_quota)
            .finish_non_exhaustive()
    }
}

impl ToolBroker {
    /// `service_bases` is the complete egress allowlist by declared service
    /// name. A tool whose [`Egress::service`] is absent fails closed.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        registry: Arc<ToolRegistry>,
        approvals: Arc<dyn ApprovalStore>,
        audit: Arc<dyn AuditSink>,
        http: reqwest::Client,
        service_bases: Vec<(String, String)>,
        minter: Option<Arc<dyn TokenMinter>>,
        exchanger: Option<Arc<dyn UserTokenExchanger>>,
        write_quota: Option<u32>,
    ) -> Result<Self, ToolError> {
        let allowlist = EgressAllowlist::new(service_bases)?;
        Ok(Self {
            registry,
            approvals,
            audit,
            http,
            allowlist,
            minter,
            exchanger,
            write_quota,
        })
    }

    pub fn approval_of(&self, name: &str) -> Option<Approval> {
        self.registry.get(name).map(|tool| tool.approval())
    }

    pub fn requires_approval(&self, name: &str) -> bool {
        matches!(
            self.approval_of(name),
            Some(approval) if approval != Approval::Never
        )
    }

    pub fn tool_risk(&self, name: &str) -> Option<&'static str> {
        self.registry.get(name).map(|tool| tool.risk().as_str())
    }

    pub fn confirm_field_of(&self, name: &str) -> Option<&'static str> {
        self.registry
            .get(name)
            .and_then(|tool| tool.typed_confirm_field())
    }

    pub fn impact_of(&self, name: &str, args: &Value) -> Option<String> {
        self.registry
            .get(name)
            .map(|tool| tool.impact_summary(args))
    }

    pub fn catalog_for(&self, scopes: &[String]) -> Vec<ToolDefinition> {
        self.registry.definitions_for(scopes)
    }

    pub fn specs_for(&self, scopes: &[String]) -> Vec<ToolSpec> {
        self.catalog_for(scopes)
            .into_iter()
            .map(|definition| ToolSpec {
                name: definition.name,
                description: definition.description,
                input_schema: definition.input_schema,
            })
            .collect()
    }

    /// Parks one approval-gated call: scopes, arguments, and the write quota
    /// are validated first, the call row is persisted as `awaiting_approval`,
    /// and the challenge is returned. Nothing reaches upstream.
    pub async fn park(
        &self,
        caller: &Caller,
        mut request: ToolCallRequest,
    ) -> Result<ApprovalChallenge, ToolError> {
        let started = Instant::now();
        if let Err(error) = canonicalize_arguments(&mut request.arguments) {
            self.audit(
                caller,
                &request,
                None,
                "failed",
                "denied",
                None,
                None,
                None,
                None,
                started.elapsed(),
                None,
            )
            .await;
            return Err(error);
        }
        let Some(tool) = self.registry.get(&request.name).cloned() else {
            return Err(ToolError::Unknown(request.name.clone()));
        };
        if tool.approval() == Approval::Never {
            return Err(ToolError::Internal);
        }

        let missing: Vec<&str> = tool
            .required_scopes()
            .iter()
            .copied()
            .filter(|scope| !caller.has_scope(scope))
            .collect();
        if !missing.is_empty() {
            self.record_failed_call(
                caller,
                &request,
                Some(tool.as_ref()),
                ToolError::NotPermitted,
                Some(&missing.join(" ")),
                started.elapsed(),
            )
            .await;
            return Err(ToolError::NotPermitted);
        }

        if let Err(error) =
            crate::tools::validate_arguments(&tool.input_schema(), &request.arguments)
        {
            self.record_failed_call(
                caller,
                &request,
                Some(tool.as_ref()),
                error.clone(),
                None,
                started.elapsed(),
            )
            .await;
            return Err(error);
        }

        if let Err(error) = self.check_write_quota(caller, &request.name).await {
            self.record_failed_call(
                caller,
                &request,
                Some(tool.as_ref()),
                error.clone(),
                None,
                started.elapsed(),
            )
            .await;
            return Err(error);
        }

        let arguments_json = serde_json::to_string(&request.arguments).unwrap_or_default();
        let arguments_hash = arguments_hash(&arguments_json);
        let expires_at = Utc::now() + ChronoDuration::from_std(APPROVAL_TTL).unwrap_or_default();
        let confirm_field = tool.typed_confirm_field();
        let impact = tool.impact_summary(&request.arguments);

        if let Err(error) = self
            .approvals
            .record_tool_call(NewToolCall {
                id: &request.id,
                thread_id: caller.thread_id,
                message_id: caller.message_id,
                scope: caller.scope(),
                subject: caller.subject(),
                name: &request.name,
                risk: tool.risk().as_str(),
                arguments_json: &arguments_json,
                arguments_hash: &arguments_hash,
                status: ToolCallStatus::AwaitingApproval,
                auth_mode: Some(tool.auth_mode().as_str()),
                expires_at: Some(expires_at),
                approval_kind: Some(tool.approval().as_str()),
                confirm_field,
                impact: Some(&impact),
            })
            .await
        {
            tracing::error!(%error, tool = %request.name, "failed to park tool call");
            return Err(ToolError::Internal);
        }

        self.audit(
            caller,
            &request,
            Some(tool.as_ref()),
            "awaiting_approval",
            "requested",
            None,
            Some(tool.auth_mode().as_str()),
            None,
            None,
            started.elapsed(),
            None,
        )
        .await;
        metrics::record_tool_approval("requested");

        Ok(ApprovalChallenge {
            kind: tool.approval(),
            expires_at,
            arguments_hash,
            confirm_field: confirm_field.map(str::to_string),
            impact,
        })
    }

    /// Executes a call that a caller approved. The recorded arguments hash must
    /// still match the approved hash; scopes and the approval policy are
    /// re-checked so removal of either cannot be bypassed by replay.
    pub async fn execute_approved(
        &self,
        caller: &Caller,
        mut request: ToolCallRequest,
        grant: ApprovalGrant,
    ) -> ToolCallOutcome {
        let started = Instant::now();
        if let Err(error) = canonicalize_arguments(&mut request.arguments) {
            return ToolCallOutcome::Failed {
                error,
                duration_ms: elapsed_ms(started.elapsed()),
            };
        }
        let Some(tool) = self.registry.get(&request.name).cloned() else {
            return ToolCallOutcome::Failed {
                error: ToolError::Unknown(request.name.clone()),
                duration_ms: elapsed_ms(started.elapsed()),
            };
        };
        if tool.approval() == Approval::Never {
            return ToolCallOutcome::Failed {
                error: ToolError::Internal,
                duration_ms: elapsed_ms(started.elapsed()),
            };
        }
        let missing: Vec<&str> = tool
            .required_scopes()
            .iter()
            .copied()
            .filter(|scope| !caller.has_scope(scope))
            .collect();
        if !missing.is_empty() {
            let error = ToolError::NotPermitted;
            self.record_failed_call(
                caller,
                &request,
                Some(tool.as_ref()),
                error.clone(),
                Some(&missing.join(" ")),
                started.elapsed(),
            )
            .await;
            return ToolCallOutcome::Failed {
                error,
                duration_ms: elapsed_ms(started.elapsed()),
            };
        }

        let arguments_json = serde_json::to_string(&request.arguments).unwrap_or_default();
        let recomputed = arguments_hash(&arguments_json);
        if recomputed != grant.arguments_hash {
            let error = ToolError::ArgumentsChanged;
            if let Err(store_error) = self
                .approvals
                .complete_tool_call(
                    &request.id,
                    ToolCallCompletion {
                        status: ToolCallStatus::Denied,
                        result_json: None,
                        error_code: Some(error.code()),
                        duration_ms: Some(elapsed_ms_i64(started.elapsed())),
                        approval_id: Some(&grant.approval_id),
                    },
                )
                .await
            {
                tracing::warn!(%store_error, tool = %request.name, "failed to record changed arguments");
            }
            self.audit(
                caller,
                &request,
                Some(tool.as_ref()),
                "denied",
                "denied",
                None,
                Some(tool.auth_mode().as_str()),
                None,
                None,
                started.elapsed(),
                Some(&grant.approval_id),
            )
            .await;
            return ToolCallOutcome::Failed {
                error,
                duration_ms: elapsed_ms(started.elapsed()),
            };
        }

        // One-shot enforcement at the broker boundary: the row must still be
        // in the `approved` state bound to this exact approval. A replay (or a
        // stale grant) can never reach upstream.
        match self.approvals.get_tool_call(&request.id).await {
            Ok(Some(row))
                if row.status == ToolCallStatus::Approved.as_wire()
                    && row.approval_id.as_deref() == Some(grant.approval_id.as_str()) => {}
            Ok(_) => {
                self.audit(
                    caller,
                    &request,
                    Some(tool.as_ref()),
                    "denied",
                    "denied",
                    None,
                    Some(tool.auth_mode().as_str()),
                    None,
                    None,
                    started.elapsed(),
                    Some(&grant.approval_id),
                )
                .await;
                metrics::record_tool_approval("replayed");
                return ToolCallOutcome::Failed {
                    error: ToolError::ApprovalReplayed,
                    duration_ms: elapsed_ms(started.elapsed()),
                };
            }
            Err(error) => {
                tracing::error!(%error, tool = %request.name, "failed to load approved call");
                return ToolCallOutcome::Failed {
                    error: ToolError::Internal,
                    duration_ms: elapsed_ms(started.elapsed()),
                };
            }
        }

        if let Err(error) = self
            .approvals
            .mark_tool_call_running(
                &request.id,
                Some(tool.auth_mode().as_str()),
                Some(&grant.approval_id),
            )
            .await
        {
            tracing::error!(%error, tool = %request.name, "failed to mark approved call running");
            return ToolCallOutcome::Failed {
                error: ToolError::Internal,
                duration_ms: elapsed_ms(started.elapsed()),
            };
        }

        self.dispatch(
            caller,
            &request,
            &tool,
            started,
            Some(&grant.approval_id),
            "allowed",
        )
        .await
    }

    async fn check_write_quota(&self, caller: &Caller, tool_name: &str) -> Result<(), ToolError> {
        let Some(limit) = self.write_quota else {
            return Ok(());
        };
        let since = Utc::now() - ChronoDuration::from_std(WRITE_QUOTA_WINDOW).unwrap_or_default();
        let used = self
            .approvals
            .count_recent_tool_calls(caller.scope(), caller.subject(), tool_name, since)
            .await
            .map_err(|error| {
                tracing::error!(%error, tool = tool_name, "write quota check failed");
                ToolError::Internal
            })?;
        if used >= u64::from(limit) {
            tracing::warn!(
                tool = tool_name,
                used,
                limit,
                "daily write-tool quota exhausted"
            );
            return Err(ToolError::RateLimited);
        }
        Ok(())
    }

    /// Authorizes, executes, records, and audits one tool call. Approval-gated
    /// tools never pass this path.
    pub async fn execute(&self, caller: &Caller, mut request: ToolCallRequest) -> ToolCallOutcome {
        let started = Instant::now();
        if let Err(error) = canonicalize_arguments(&mut request.arguments) {
            let duration = started.elapsed();
            self.audit(
                caller, &request, None, "failed", "denied", None, None, None, None, duration, None,
            )
            .await;
            metrics::record_tool_call(&request.name, error.code(), duration);
            return ToolCallOutcome::Failed {
                error,
                duration_ms: elapsed_ms(duration),
            };
        }
        let Some(tool) = self.registry.get(&request.name).cloned() else {
            let duration = started.elapsed();
            self.audit(
                caller, &request, None, "unknown", "denied", None, None, None, None, duration, None,
            )
            .await;
            return ToolCallOutcome::Failed {
                error: ToolError::Unknown(request.name.clone()),
                duration_ms: elapsed_ms(duration),
            };
        };

        let missing: Vec<&str> = tool
            .required_scopes()
            .iter()
            .copied()
            .filter(|scope| !caller.has_scope(scope))
            .collect();
        if !missing.is_empty() {
            let duration = started.elapsed();
            self.record_failed_call(
                caller,
                &request,
                Some(tool.as_ref()),
                ToolError::NotPermitted,
                Some(&missing.join(" ")),
                duration,
            )
            .await;
            return ToolCallOutcome::Failed {
                error: ToolError::NotPermitted,
                duration_ms: elapsed_ms(duration),
            };
        }

        // Fail closed: an approval-gated tool can only run with an approval
        // grant supplied to `execute_approved`. The parked row (if any) is
        // left untouched: this path must never overwrite an approval window.
        if tool.approval() != Approval::Never {
            let duration = started.elapsed();
            self.audit(
                caller,
                &request,
                Some(tool.as_ref()),
                "denied",
                "denied",
                None,
                Some(tool.auth_mode().as_str()),
                None,
                None,
                duration,
                None,
            )
            .await;
            metrics::record_tool_call(&request.name, ToolError::ApprovalRequired.code(), duration);
            return ToolCallOutcome::Failed {
                error: ToolError::ApprovalRequired,
                duration_ms: elapsed_ms(duration),
            };
        }

        if let Err(error) = caller.budget.try_reserve_call() {
            let duration = started.elapsed();
            self.record_failed_call(
                caller,
                &request,
                Some(tool.as_ref()),
                error.clone(),
                None,
                duration,
            )
            .await;
            return ToolCallOutcome::Failed {
                error,
                duration_ms: elapsed_ms(duration),
            };
        }

        let auth_mode = tool.auth_mode();
        let arguments_json = serde_json::to_string(&request.arguments).unwrap_or_default();
        let arguments_hash = arguments_hash(&arguments_json);

        if let Err(error) = caller
            .budget
            .try_reserve_signature(&request.name, &arguments_hash)
        {
            metrics::record_tool_loop_blocked(&request.name);
            let duration = started.elapsed();
            self.record_failed_call(
                caller,
                &request,
                Some(tool.as_ref()),
                error.clone(),
                None,
                duration,
            )
            .await;
            return ToolCallOutcome::Failed {
                error,
                duration_ms: elapsed_ms(duration),
            };
        }

        if let Err(error) = self
            .approvals
            .record_tool_call(NewToolCall {
                id: &request.id,
                thread_id: caller.thread_id,
                message_id: caller.message_id,
                scope: caller.scope(),
                subject: caller.subject(),
                name: &request.name,
                risk: tool.risk().as_str(),
                arguments_json: &arguments_json,
                arguments_hash: &arguments_hash,
                status: ToolCallStatus::Running,
                auth_mode: Some(auth_mode.as_str()),
                expires_at: None,
                approval_kind: None,
                confirm_field: None,
                impact: None,
            })
            .await
        {
            tracing::error!(%error, tool = %request.name, "failed to record tool call");
            let duration = started.elapsed();
            return ToolCallOutcome::Failed {
                error: ToolError::Internal,
                duration_ms: elapsed_ms(duration),
            };
        }

        self.dispatch(caller, &request, &tool, started, None, "allowed")
            .await
    }

    async fn dispatch(
        &self,
        caller: &Caller,
        request: &ToolCallRequest,
        tool: &Arc<dyn Tool>,
        started: Instant,
        approval_id: Option<&str>,
        decision: &str,
    ) -> ToolCallOutcome {
        let auth_mode = tool.auth_mode();
        let egress = match self.build_egress(caller, tool).await {
            Ok(egress) => egress,
            Err(error) => {
                let duration = started.elapsed();
                self.record_failed_call(
                    caller,
                    request,
                    Some(tool.as_ref()),
                    error.clone(),
                    None,
                    duration,
                )
                .await;
                return ToolCallOutcome::Failed {
                    error,
                    duration_ms: elapsed_ms(duration),
                };
            }
        };

        // Principal attributes are the application's domain channel: they are
        // projected into the tool context as JSON metadata so tools can render
        // application identifiers (tenant keys, roles) without the core
        // interpreting them.
        let metadata = caller
            .principal
            .attributes
            .iter()
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect();
        let context = ToolExecContext {
            scope: caller.scope().clone(),
            subject: caller.subject().to_string(),
            scopes: caller.principal.scopes.clone(),
            request_id: caller.request_id.clone(),
            deadline: caller.deadline(),
            egress,
            metadata,
        };

        let outcome = match self
            .registry
            .execute(&request.name, request.arguments.clone(), context)
            .await
        {
            Ok(result) => {
                let result_bytes = serde_json::to_string(&result)
                    .map(|raw| raw.len())
                    .unwrap_or(0);
                match caller.budget.try_reserve_bytes(result_bytes) {
                    Ok(()) => Ok(result),
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error),
        };
        let duration = started.elapsed();

        let (status, result, error_code, downstream_status) = match &outcome {
            Ok(result) => (ToolCallStatus::Succeeded, Some(result), None, Some(200)),
            Err(error) => {
                let status = match error {
                    ToolError::NotPermitted => ToolCallStatus::Denied,
                    _ => ToolCallStatus::Failed,
                };
                let downstream = match error {
                    ToolError::UpstreamStatus { status } => Some(i32::from(*status)),
                    _ => None,
                };
                (status, None, Some(error.code()), downstream)
            }
        };

        self.complete_call(request, status, result, error_code, duration, approval_id)
            .await;
        let scopes_used = tool.required_scopes().join(" ");
        let result_bytes = result
            .and_then(|value| serde_json::to_string(value).ok())
            .map(|raw| raw.len().min(i32::MAX as usize) as i32);
        self.audit(
            caller,
            request,
            Some(tool.as_ref()),
            if error_code.is_none() {
                "succeeded"
            } else {
                "failed"
            },
            decision,
            Some(&scopes_used),
            Some(auth_mode.as_str()),
            downstream_status,
            result_bytes,
            duration,
            approval_id,
        )
        .await;

        let metric_outcome = if error_code.is_none() {
            "succeeded"
        } else {
            "failed"
        };
        metrics::record_tool_call(&request.name, metric_outcome, duration);
        if decision != "allowed" {
            metrics::record_tool_approval(decision);
        }

        match outcome {
            Ok(result) => ToolCallOutcome::Succeeded {
                result,
                duration_ms: elapsed_ms(duration),
            },
            Err(error) => ToolCallOutcome::Failed {
                error,
                duration_ms: elapsed_ms(duration),
            },
        }
    }

    /// Records a call that never reached upstream: operational row (when the
    /// tool is known), terminal status, audit, and metrics.
    async fn record_failed_call(
        &self,
        caller: &Caller,
        request: &ToolCallRequest,
        tool: Option<&dyn Tool>,
        error: ToolError,
        scopes_used: Option<&str>,
        duration: Duration,
    ) {
        if let Some(tool) = tool {
            let arguments_json = serde_json::to_string(&request.arguments).unwrap_or_default();
            let arguments_hash = arguments_hash(&arguments_json);
            let recorded = self
                .approvals
                .record_tool_call(NewToolCall {
                    id: &request.id,
                    thread_id: caller.thread_id,
                    message_id: caller.message_id,
                    scope: caller.scope(),
                    subject: caller.subject(),
                    name: &request.name,
                    risk: tool.risk().as_str(),
                    arguments_json: &arguments_json,
                    arguments_hash: &arguments_hash,
                    status: ToolCallStatus::Running,
                    auth_mode: Some(tool.auth_mode().as_str()),
                    expires_at: None,
                    approval_kind: None,
                    confirm_field: None,
                    impact: None,
                })
                .await;
            if recorded.is_ok() {
                let status = match error {
                    ToolError::NotPermitted => ToolCallStatus::Denied,
                    _ => ToolCallStatus::Failed,
                };
                self.complete_call(request, status, None, Some(error.code()), duration, None)
                    .await;
            } else if let Err(error) = recorded {
                tracing::error!(%error, tool = %request.name, "failed to record rejected tool call");
            }
        }
        self.audit(
            caller,
            request,
            tool,
            match &error {
                ToolError::NotPermitted => "denied",
                ToolError::LimitExceeded => "limit_exceeded",
                ToolError::RateLimited => "rate_limited",
                ToolError::Timeout => "timeout",
                _ => "failed",
            },
            "denied",
            scopes_used,
            tool.map(|tool| tool.auth_mode().as_str()),
            None,
            None,
            duration,
            None,
        )
        .await;
        metrics::record_tool_call(&request.name, error.code(), duration);
    }

    /// Records a denial decision on a parked call and audits it.
    pub async fn deny(&self, caller: &Caller, request: &ToolCallRequest) -> ToolCallOutcome {
        let started = Instant::now();
        if let Err(error) = self.approvals.mark_tool_call_denied(&request.id).await {
            tracing::warn!(%error, tool = %request.name, "failed to mark tool call denied");
        }
        let tool = self.registry.get(&request.name).cloned();
        self.audit(
            caller,
            request,
            tool.as_deref(),
            "denied",
            "denied",
            None,
            tool.as_deref().map(|tool| tool.auth_mode().as_str()),
            None,
            None,
            started.elapsed(),
            None,
        )
        .await;
        metrics::record_tool_approval("denied");
        ToolCallOutcome::Failed {
            error: ToolError::ApprovalDenied,
            duration_ms: elapsed_ms(started.elapsed()),
        }
    }

    async fn build_egress(
        &self,
        caller: &Caller,
        tool: &Arc<dyn Tool>,
    ) -> Result<Arc<dyn ToolEgress>, ToolError> {
        let auth = match tool.auth_mode() {
            EgressAuth::Passthrough => {
                // When an exchanger is configured, the caller's token is
                // exchanged for one scoped to the downstream audience while
                // preserving subject, scope, and scopes. Otherwise the caller's
                // own token is forwarded and the downstream enforces
                // membership.
                let token = if let Some(exchanger) = self.exchanger.as_ref() {
                    Some(
                        exchanger
                            .exchange(
                                &caller.principal,
                                tool.egress().service,
                                &caller.principal.scope,
                            )
                            .await?
                            .expose()
                            .to_string(),
                    )
                } else {
                    caller
                        .principal
                        .token
                        .as_ref()
                        .map(|token| token.expose().to_string())
                        .filter(|token| !token.is_empty())
                };
                match token {
                    Some(token) => EgressAuthHeader::Bearer(token),
                    None => EgressAuthHeader::None,
                }
            }
            EgressAuth::Minted { scopes, audience } => {
                let Some(minter) = self.minter.as_ref() else {
                    tracing::warn!(
                        tool = tool.name(),
                        "minted egress requested but no token minter configured"
                    );
                    return Err(ToolError::UpstreamUnavailable);
                };
                let token = minter.mint(&caller.principal, scopes, audience).await?;
                EgressAuthHeader::Bearer(token.expose().to_string())
            }
        };
        let timeout = tool
            .limits()
            .timeout
            .min(caller.deadline().saturating_duration_since(Instant::now()));
        if timeout.is_zero() {
            return Err(ToolError::Timeout);
        }
        Ok(Arc::new(HttpEgress {
            http: self.http.clone(),
            allowlist: self.allowlist.clone(),
            bound: tool.egress(),
            auth,
            timeout,
        }))
    }

    async fn complete_call(
        &self,
        request: &ToolCallRequest,
        status: ToolCallStatus,
        result: Option<&Value>,
        error_code: Option<&str>,
        duration: Duration,
        approval_id: Option<&str>,
    ) {
        let completion = ToolCallCompletion {
            status,
            result_json: result,
            error_code,
            duration_ms: Some(elapsed_ms_i64(duration)),
            approval_id,
        };
        if let Err(error) = self
            .approvals
            .complete_tool_call(&request.id, completion)
            .await
        {
            tracing::error!(%error, tool = %request.name, "failed to complete tool call");
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn audit(
        &self,
        caller: &Caller,
        request: &ToolCallRequest,
        tool: Option<&dyn Tool>,
        outcome: &str,
        decision: &str,
        scopes_used: Option<&str>,
        auth_mode: Option<&str>,
        downstream_status: Option<i32>,
        result_bytes: Option<i32>,
        duration: Duration,
        approval_id: Option<&str>,
    ) {
        let arguments_json = serde_json::to_string(&request.arguments).unwrap_or_default();
        let arguments_hash = arguments_hash(&arguments_json);
        let redacted = serde_json::to_string(&redact_secrets(request.arguments.clone())).ok();
        let egress = tool.map(|tool| tool.egress());
        let record = AuditRecord {
            id: Uuid::new_v4(),
            scope_id: caller.scope().to_string(),
            subject: caller.subject().to_string(),
            thread_id: Some(caller.thread_id),
            message_id: Some(caller.message_id),
            tool_call_id: Some(request.id.clone()),
            item_name: request.name.clone(),
            item_kind: AuditItemKind::Tool,
            risk: tool
                .map_or("unknown", |tool| tool.risk().as_str())
                .to_string(),
            arguments_hash,
            arguments_redacted: redacted.map(Value::String),
            scopes_used: scopes_used.map(str::to_string),
            decision: Some(decision.to_string()),
            auth_mode: auth_mode.map(str::to_string),
            egress_service: egress.map(|egress| egress.service.to_string()),
            egress_path_template: egress.map(|egress| egress.path_template.to_string()),
            outcome: outcome.to_string(),
            downstream_status,
            result_bytes,
            duration_ms: Some(elapsed_ms_i64(duration)),
            approval_id: approval_id.map(str::to_string),
            created_at: Utc::now(),
        };
        if let Err(error) = self.audit.append(record).await {
            tracing::error!(%error, tool = %request.name, "failed to append tool audit");
        }
    }

    /// Validates a decision against the stored call and returns the persisted
    /// snapshot for one-shot execution.
    pub async fn load_decision_target(
        &self,
        scope: &ScopeId,
        subject: &str,
        call_id: &str,
    ) -> Result<ToolCallSnapshot, ToolError> {
        match self.approvals.get_tool_call(call_id).await {
            Ok(Some(snapshot))
                if snapshot.scope_id == scope.as_str() && snapshot.subject == subject =>
            {
                Ok(snapshot)
            }
            Ok(Some(_)) => Err(ToolError::NotFound),
            Ok(None) => Err(ToolError::NotFound),
            Err(error) => {
                tracing::error!(%error, "failed to load tool call");
                Err(ToolError::Internal)
            }
        }
    }
}

fn elapsed_ms(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn elapsed_ms_i64(duration: Duration) -> i64 {
    duration.as_millis().min(i64::MAX as u128) as i64
}

/// SHA-256 hex of the canonical arguments JSON.
pub fn arguments_hash(arguments_json: &str) -> String {
    let digest = Sha256::digest(arguments_json.as_bytes());
    hex::encode(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::AuditSink;
    use crate::store::{
        ApprovalStore, NewToolApproval, NewToolCall, StoreError, ToolCallCompletion,
        ToolCallSnapshot,
    };
    use async_trait::async_trait;
    use std::sync::Mutex;

    #[test]
    fn arguments_hash_is_stable_sha256_hex() {
        let hash = arguments_hash(r#"{"a":1}"#);
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(hash, arguments_hash(r#"{"a":1}"#));
        assert_ne!(hash, arguments_hash(r#"{"a":2}"#));
    }

    #[test]
    fn turn_budget_blocks_repeated_identical_calls() {
        let budget = TurnToolBudget::new(8, 1024, Duration::from_secs(60));
        let hash = arguments_hash(r#"{"a":1}"#);
        assert!(budget.try_reserve_signature("demo", &hash).is_ok());
        assert!(budget.try_reserve_signature("demo", &hash).is_ok());
        assert!(budget.try_reserve_signature("demo", &hash).is_err());
        let other = arguments_hash(r#"{"a":2}"#);
        assert!(budget.try_reserve_signature("demo", &other).is_ok());
    }

    #[test]
    fn turn_budget_enforces_call_and_byte_limits() {
        let budget = TurnToolBudget::new(2, 10, Duration::from_secs(60));
        assert!(budget.try_reserve_call().is_ok());
        assert!(budget.try_reserve_call().is_ok());
        assert!(matches!(
            budget.try_reserve_call(),
            Err(ToolError::LimitExceeded)
        ));
        assert!(budget.try_reserve_bytes(6).is_ok());
        assert!(matches!(
            budget.try_reserve_bytes(6),
            Err(ToolError::LimitExceeded)
        ));
    }

    #[test]
    fn egress_allowlist_rejects_malformed_bases() {
        assert!(EgressAllowlist::new([("demo", "not a url".to_string())]).is_err());
        assert!(EgressAllowlist::new([("demo", String::new())]).is_err());
        assert!(EgressAllowlist::new([("demo", "ftp://example.com".to_string())]).is_err());
        let allowlist =
            EgressAllowlist::new([("demo", "https://api.example.com".to_string())]).unwrap();
        assert_eq!(allowlist.services(), vec!["demo"]);
    }

    #[test]
    fn egress_resolution_keeps_the_origin_and_rejects_traversal() {
        let allowlist =
            EgressAllowlist::new([("demo", "https://api.example.com".to_string())]).unwrap();
        let egress = Egress {
            service: "demo",
            method: "GET",
            path_template: "/v1/items/{id}",
        };
        let url = allowlist.resolve(egress, &[("id", "abc-1")], &[]).unwrap();
        assert_eq!(url.as_str(), "https://api.example.com/v1/items/abc-1");

        let traversal = allowlist.resolve(egress, &[("id", "../admin")], &[]);
        assert!(matches!(traversal, Err(ToolError::InvalidArguments(_))));

        let unknown_service = Egress {
            service: "other",
            method: "GET",
            path_template: "/v1/x",
        };
        assert!(matches!(
            allowlist.resolve(unknown_service, &[], &[]),
            Err(ToolError::Internal)
        ));
    }

    #[test]
    fn egress_resolution_rejects_unsafe_templates_and_methods() {
        let allowlist =
            EgressAllowlist::new([("demo", "https://api.example.com".to_string())]).unwrap();
        for template in ["relative", "/a/../b", "//host/x", "/x?y=1"] {
            let egress = Egress {
                service: "demo",
                method: "GET",
                path_template: template,
            };
            assert!(
                allowlist.resolve(egress, &[], &[]).is_err(),
                "{template} must be rejected"
            );
        }
        let egress = Egress {
            service: "demo",
            method: "DELETE",
            path_template: "/v1/x",
        };
        assert!(allowlist.resolve(egress, &[], &[]).is_err());
    }

    struct NoopMinter;

    #[async_trait]
    impl TokenMinter for NoopMinter {
        async fn mint(
            &self,
            _principal: &Principal,
            _scopes: &[&str],
            _audience: &str,
        ) -> Result<SecretString, ToolError> {
            Ok(SecretString::new("none"))
        }
    }

    #[derive(Default)]
    struct MemoryApprovals {
        calls: Mutex<HashMap<String, (ToolCallSnapshot, Value)>>,
    }

    #[async_trait]
    impl ApprovalStore for MemoryApprovals {
        async fn record_tool_call(&self, call: NewToolCall<'_>) -> Result<(), StoreError> {
            let snapshot = ToolCallSnapshot {
                id: call.id.to_string(),
                scope_id: call.scope.to_string(),
                subject: call.subject.to_string(),
                thread_id: call.thread_id,
                message_id: call.message_id,
                name: call.name.to_string(),
                risk: call.risk.to_string(),
                status: call.status.as_wire().to_string(),
                approval_kind: call.approval_kind.map(str::to_string),
                approval_id: None,
                confirm_field: call.confirm_field.map(str::to_string),
                arguments: serde_json::from_str(call.arguments_json).unwrap_or(Value::Null),
                arguments_hash: call.arguments_hash.to_string(),
                expires_at: call.expires_at,
            };
            self.calls
                .lock()
                .unwrap()
                .insert(call.id.to_string(), (snapshot, Value::Null));
            Ok(())
        }

        async fn insert_tool_approval(
            &self,
            _approval: NewToolApproval<'_>,
        ) -> Result<(), StoreError> {
            Ok(())
        }

        async fn mark_tool_call_approved(
            &self,
            id: &str,
            approval_id: &str,
        ) -> Result<(), StoreError> {
            let mut calls = self.calls.lock().unwrap();
            let (snapshot, _) = calls.get_mut(id).ok_or(StoreError::NotFound)?;
            if snapshot.status != ToolCallStatus::AwaitingApproval.as_wire() {
                return Err(StoreError::Conflict("already decided".into()));
            }
            snapshot.status = ToolCallStatus::Approved.as_wire().to_string();
            snapshot.approval_id = Some(approval_id.to_string());
            Ok(())
        }

        async fn mark_tool_call_denied(&self, id: &str) -> Result<(), StoreError> {
            let mut calls = self.calls.lock().unwrap();
            let (snapshot, _) = calls.get_mut(id).ok_or(StoreError::NotFound)?;
            if snapshot.status != ToolCallStatus::AwaitingApproval.as_wire() {
                return Err(StoreError::Conflict("already decided".into()));
            }
            snapshot.status = ToolCallStatus::Denied.as_wire().to_string();
            Ok(())
        }

        async fn mark_tool_call_running(
            &self,
            id: &str,
            _auth_mode: Option<&str>,
            _approval_id: Option<&str>,
        ) -> Result<(), StoreError> {
            let mut calls = self.calls.lock().unwrap();
            let (snapshot, _) = calls.get_mut(id).ok_or(StoreError::NotFound)?;
            snapshot.status = ToolCallStatus::Running.as_wire().to_string();
            Ok(())
        }

        async fn complete_tool_call(
            &self,
            id: &str,
            completion: ToolCallCompletion<'_>,
        ) -> Result<(), StoreError> {
            let mut calls = self.calls.lock().unwrap();
            let (snapshot, _) = calls.get_mut(id).ok_or(StoreError::NotFound)?;
            snapshot.status = completion.status.as_wire().to_string();
            Ok(())
        }

        async fn get_tool_call(&self, id: &str) -> Result<Option<ToolCallSnapshot>, StoreError> {
            Ok(self
                .calls
                .lock()
                .unwrap()
                .get(id)
                .map(|(snapshot, _)| snapshot.clone()))
        }

        async fn count_recent_tool_calls(
            &self,
            _scope: &ScopeId,
            _subject: &str,
            _name: &str,
            _since: DateTime<Utc>,
        ) -> Result<u64, StoreError> {
            Ok(0)
        }

        async fn expire_tool_calls(&self, _now: DateTime<Utc>) -> Result<usize, StoreError> {
            Ok(0)
        }
    }

    #[derive(Default)]
    struct MemoryAudit {
        records: Mutex<Vec<AuditRecord>>,
    }

    #[async_trait]
    impl AuditSink for MemoryAudit {
        async fn append(&self, record: AuditRecord) -> Result<(), StoreError> {
            self.records.lock().unwrap().push(record);
            Ok(())
        }
    }

    struct DemoWriteTool;

    #[async_trait]
    impl Tool for DemoWriteTool {
        fn name(&self) -> &'static str {
            "demo_write"
        }
        fn description(&self) -> &'static str {
            "writes a demo record"
        }
        fn input_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": false
            })
        }
        fn output_schema(&self) -> Value {
            json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": false})
        }
        fn required_scopes(&self) -> &'static [&'static str] {
            &["demo.write"]
        }
        fn risk(&self) -> crate::tools::Risk {
            crate::tools::Risk::Write
        }
        fn approval(&self) -> Approval {
            Approval::Required
        }
        fn egress(&self) -> Egress {
            Egress {
                service: "demo",
                method: "POST",
                path_template: "/v1/records",
            }
        }
        fn auth_mode(&self) -> EgressAuth {
            EgressAuth::Minted {
                scopes: &["demo.write"],
                audience: "demo",
            }
        }
        async fn execute(&self, _args: Value, _ctx: ToolExecContext) -> Result<Value, ToolError> {
            Ok(json!({"ok": true}))
        }
    }

    fn caller() -> Caller {
        Caller::new(
            Principal::new(ScopeId::new("scope-a"), "user-1").with_scopes(["demo.write"]),
            "req-1",
            Uuid::nil(),
            Uuid::nil(),
            Arc::new(TurnToolBudget::new(8, 4096, Duration::from_secs(60))),
        )
    }

    fn broker(approvals: Arc<dyn ApprovalStore>, audit: Arc<dyn AuditSink>) -> ToolBroker {
        let registry = Arc::new(ToolRegistry::new().register(DemoWriteTool));
        ToolBroker::new(
            registry,
            approvals,
            audit,
            reqwest::Client::new(),
            vec![("demo".to_string(), "https://api.example.com".to_string())],
            Some(Arc::new(NoopMinter)),
            None,
            None,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn approval_gated_tools_are_parked_and_executed_once() {
        let approvals = Arc::new(MemoryApprovals::default());
        let audit = Arc::new(MemoryAudit::default());
        let broker = broker(approvals.clone(), audit.clone());
        let caller = caller();
        let request = ToolCallRequest {
            id: "call-1".into(),
            name: "demo_write".into(),
            arguments: json!({"value": "x"}),
        };

        let challenge = broker.park(&caller, request.clone()).await.unwrap();
        assert_eq!(challenge.kind, Approval::Required);
        assert_eq!(challenge.arguments_hash.len(), 64);

        // A plain execute refuses an approval-gated tool.
        let outcome = broker.execute(&caller, request.clone()).await;
        assert!(matches!(
            outcome,
            ToolCallOutcome::Failed {
                error: ToolError::ApprovalRequired,
                ..
            }
        ));

        approvals
            .mark_tool_call_approved("call-1", "appr-1")
            .await
            .unwrap();
        let outcome = broker
            .execute_approved(
                &caller,
                request.clone(),
                ApprovalGrant {
                    approval_id: "appr-1".into(),
                    arguments_hash: challenge.arguments_hash.clone(),
                },
            )
            .await;
        assert!(matches!(outcome, ToolCallOutcome::Succeeded { .. }));

        // Replay is refused: the row is no longer `approved`.
        let replay = broker
            .execute_approved(
                &caller,
                request,
                ApprovalGrant {
                    approval_id: "appr-1".into(),
                    arguments_hash: challenge.arguments_hash.clone(),
                },
            )
            .await;
        assert!(matches!(
            replay,
            ToolCallOutcome::Failed {
                error: ToolError::ApprovalReplayed,
                ..
            }
        ));

        assert!(
            !audit.records.lock().unwrap().is_empty(),
            "audit records are written"
        );
    }

    #[tokio::test]
    async fn changed_arguments_are_refused_after_approval() {
        let approvals = Arc::new(MemoryApprovals::default());
        let audit = Arc::new(MemoryAudit::default());
        let broker = broker(approvals.clone(), audit);
        let caller = caller();
        let request = ToolCallRequest {
            id: "call-2".into(),
            name: "demo_write".into(),
            arguments: json!({"value": "x"}),
        };
        let challenge = broker.park(&caller, request.clone()).await.unwrap();
        approvals
            .mark_tool_call_approved("call-2", "appr-2")
            .await
            .unwrap();
        let mut changed = request;
        changed.arguments = json!({"value": "y"});
        let outcome = broker
            .execute_approved(
                &caller,
                changed,
                ApprovalGrant {
                    approval_id: "appr-2".into(),
                    arguments_hash: challenge.arguments_hash,
                },
            )
            .await;
        assert!(matches!(
            outcome,
            ToolCallOutcome::Failed {
                error: ToolError::ArgumentsChanged,
                ..
            }
        ));
    }

    struct MetadataReadTool {
        seen: Arc<Mutex<Option<Value>>>,
    }

    #[async_trait]
    impl Tool for MetadataReadTool {
        fn name(&self) -> &'static str {
            "demo_metadata"
        }

        fn description(&self) -> &'static str {
            "returns no data but records the tool context metadata"
        }

        fn input_schema(&self) -> Value {
            json!({"type": "object", "properties": {}, "required": [], "additionalProperties": false})
        }

        fn output_schema(&self) -> Value {
            json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": false})
        }

        fn required_scopes(&self) -> &'static [&'static str] {
            &[]
        }

        fn risk(&self) -> crate::tools::Risk {
            crate::tools::Risk::Read
        }

        fn approval(&self) -> Approval {
            Approval::Never
        }

        fn egress(&self) -> Egress {
            Egress {
                service: "demo",
                method: "GET",
                path_template: "/v1/metadata",
            }
        }

        fn auth_mode(&self) -> EgressAuth {
            EgressAuth::Passthrough
        }

        async fn execute(&self, _args: Value, ctx: ToolExecContext) -> Result<Value, ToolError> {
            *self.seen.lock().unwrap() = Some(Value::Object(ctx.metadata));
            Ok(json!({"ok": true}))
        }
    }

    #[tokio::test]
    async fn principal_attributes_reach_the_tool_context_as_metadata() {
        let seen = Arc::new(Mutex::new(None));
        let registry = Arc::new(ToolRegistry::new().register(MetadataReadTool {
            seen: Arc::clone(&seen),
        }));
        let broker = ToolBroker::new(
            registry,
            Arc::new(MemoryApprovals::default()),
            Arc::new(MemoryAudit::default()),
            reqwest::Client::new(),
            vec![("demo".to_string(), "https://api.example.com".to_string())],
            None,
            None,
            None,
        )
        .unwrap();
        let principal =
            Principal::new(ScopeId::new("scope-a"), "user-1").with_attribute("tenant_key", "acme");
        let caller = Caller::new(
            principal,
            "req-1",
            Uuid::nil(),
            Uuid::nil(),
            Arc::new(TurnToolBudget::new(8, 4096, Duration::from_secs(60))),
        );
        let outcome = broker
            .execute(
                &caller,
                ToolCallRequest {
                    id: "call-meta".into(),
                    name: "demo_metadata".into(),
                    arguments: json!({}),
                },
            )
            .await;
        assert!(matches!(outcome, ToolCallOutcome::Succeeded { .. }));
        let metadata = seen.lock().unwrap().clone().expect("tool context seen");
        assert_eq!(metadata["tenant_key"], json!("acme"));
    }

    #[tokio::test]
    async fn callers_without_the_required_scope_are_refused_at_park() {
        let approvals = Arc::new(MemoryApprovals::default());
        let audit = Arc::new(MemoryAudit::default());
        let broker = broker(approvals, audit);
        let outsider = Caller::new(
            Principal::new(ScopeId::new("scope-a"), "user-2"),
            "req-2",
            Uuid::nil(),
            Uuid::nil(),
            Arc::new(TurnToolBudget::new(8, 4096, Duration::from_secs(60))),
        );
        let error = broker
            .park(
                &outsider,
                ToolCallRequest {
                    id: "call-3".into(),
                    name: "demo_write".into(),
                    arguments: json!({"value": "x"}),
                },
            )
            .await
            .unwrap_err();
        assert_eq!(error, ToolError::NotPermitted);
    }
}
