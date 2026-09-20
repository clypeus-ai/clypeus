//! Principals, scopes and policy evaluation.
//!
//! The core never interprets identity. A [`Principal`] carries an opaque
//! [`ScopeId`] (the isolation partition), an opaque subject, opaque
//! authorization codes, free-form attributes for the application's own
//! templates, and an optional bearer token used for pass-through egress.
//! Authorization decisions are delegated to a [`PolicyEngine`] supplied by the
//! embedding application.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::secrets::SecretString;

/// Opaque isolation partition. Every stored record and every query is keyed by
/// a scope; the core never compares it to a path, a header, or a database
/// column it does not own.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScopeId(pub String);

impl ScopeId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ScopeId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// The authenticated caller of one request.
#[derive(Clone, Debug)]
pub struct Principal {
    /// Isolation partition the caller operates in.
    pub scope: ScopeId,
    /// Opaque subject identifier.
    pub subject: String,
    /// Opaque authorization codes held by the caller.
    pub scopes: Vec<String>,
    /// Domain attributes for application templates (roles, tenant keys,
    /// resource identifiers). The core only transports them.
    pub attributes: BTreeMap<String, String>,
    /// Bearer token for pass-through egress. Never persisted, never logged.
    pub token: Option<SecretString>,
}

impl Principal {
    pub fn new(scope: ScopeId, subject: impl Into<String>) -> Self {
        Self {
            scope,
            subject: subject.into(),
            scopes: Vec::new(),
            attributes: BTreeMap::new(),
            token: None,
        }
    }

    pub fn with_scopes<I, S>(mut self, scopes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.scopes = scopes.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(SecretString::new(token));
        self
    }

    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    pub fn attribute(&self, key: &str) -> Option<&str> {
        self.attributes.get(key).map(String::as_str)
    }

    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|held| held == scope)
    }
}

/// An authorization requirement checked against a principal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Requirement {
    /// The principal must hold this exact scope.
    Scope(String),
    /// The principal must hold every listed scope.
    AllOf(Vec<String>),
}

impl Requirement {
    pub fn scope(value: impl Into<String>) -> Self {
        Self::Scope(value.into())
    }

    pub fn all_of<I, S>(values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::AllOf(values.into_iter().map(Into::into).collect())
    }
}

/// Result of a policy check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny { code: &'static str },
}

impl Decision {
    pub fn is_allow(self) -> bool {
        matches!(self, Self::Allow)
    }
}

/// Authorization policy supplied by the application.
///
/// `is_admin` gates the administrative surface (settings, audit export,
/// cross-scope access). It defaults to denying, so an embedding application
/// must opt in explicitly.
pub trait PolicyEngine: Send + Sync {
    fn check(&self, principal: &Principal, requirement: &Requirement) -> Decision;

    fn is_admin(&self, _principal: &Principal) -> bool {
        false
    }
}

/// A policy that grants every scope check. Intended for single-tenant
/// deployments and tests; `is_admin` is still false.
#[derive(Debug, Default)]
pub struct AllowAllPolicy;

impl PolicyEngine for AllowAllPolicy {
    fn check(&self, _principal: &Principal, _requirement: &Requirement) -> Decision {
        Decision::Allow
    }
}

/// A policy driven by an explicit scope list. The principal's own scopes must
/// contain the requirement; `admin_scopes` additionally make a principal
/// administrative.
#[derive(Debug, Clone, Default)]
pub struct StaticPolicy {
    pub admin_scopes: Vec<String>,
}

impl StaticPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_admin_scopes<I, S>(mut self, scopes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.admin_scopes = scopes.into_iter().map(Into::into).collect();
        self
    }
}

impl PolicyEngine for StaticPolicy {
    fn check(&self, principal: &Principal, requirement: &Requirement) -> Decision {
        let held = |scope: &str| principal.has_scope(scope);
        match requirement {
            Requirement::Scope(scope) => {
                if held(scope) {
                    Decision::Allow
                } else {
                    Decision::Deny {
                        code: "not_permitted",
                    }
                }
            }
            Requirement::AllOf(scopes) => {
                if scopes.iter().all(|scope| held(scope)) {
                    Decision::Allow
                } else {
                    Decision::Deny {
                        code: "not_permitted",
                    }
                }
            }
        }
    }

    fn is_admin(&self, principal: &Principal) -> bool {
        self.admin_scopes
            .iter()
            .any(|scope| principal.has_scope(scope))
    }
}

/// The subset of a request a resolver may inspect.
#[derive(Debug)]
pub struct AuthRequest<'a> {
    pub headers: &'a http_headers::HeaderMap,
    /// Scope hinted by the request path, when the application routes by scope.
    pub path_scope_hint: Option<&'a str>,
}

/// Minimal header map surface, re-exported to avoid an `http` dependency in
/// downstream crates that only resolve principals.
pub mod http_headers {
    pub use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
}

/// Failure to resolve a principal. The code is safe to return to the caller;
/// the detail stays in the logs.
#[derive(Debug, Clone, thiserror::Error)]
#[error("authentication failed: {code}")]
pub struct AuthError {
    pub code: &'static str,
    pub detail: Option<String>,
}

impl AuthError {
    pub fn new(code: &'static str) -> Self {
        Self { code, detail: None }
    }

    pub fn with_detail(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: Some(detail.into()),
        }
    }

    pub fn missing_credentials() -> Self {
        Self::new("missing_credentials")
    }

    pub fn invalid_credentials() -> Self {
        Self::new("invalid_credentials")
    }
}

/// Resolves a request into a [`Principal`].
#[async_trait::async_trait]
pub trait PrincipalResolver: Send + Sync {
    async fn resolve(&self, request: &AuthRequest<'_>) -> Result<Principal, AuthError>;
}

/// Resolver for a fixed bearer token and a fixed principal. Useful for
/// single-tenant deployments, local development, and tests.
#[derive(Debug, Clone)]
pub struct StaticTokenResolver {
    token: SecretString,
    principal: Principal,
}

impl StaticTokenResolver {
    pub fn new(token: impl Into<String>, principal: Principal) -> Self {
        Self {
            token: SecretString::new(token),
            principal,
        }
    }

    pub fn principal(&self) -> &Principal {
        &self.principal
    }
}

#[async_trait::async_trait]
impl PrincipalResolver for StaticTokenResolver {
    async fn resolve(&self, request: &AuthRequest<'_>) -> Result<Principal, AuthError> {
        let header = request
            .headers
            .get(reqwest::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(AuthError::missing_credentials)?;
        let token = header
            .strip_prefix("Bearer ")
            .or_else(|| header.strip_prefix("bearer "))
            .ok_or_else(AuthError::invalid_credentials)?;
        if token != self.token.expose() {
            return Err(AuthError::invalid_credentials());
        }
        Ok(self.principal.clone())
    }
}

/// Resolver composition: the first resolver that accepts the request wins.
/// An empty chain rejects every request.
#[derive(Clone)]
pub struct ResolverChain {
    resolvers: Vec<Arc<dyn PrincipalResolver>>,
}

impl std::fmt::Debug for ResolverChain {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolverChain")
            .field("resolvers", &self.resolvers.len())
            .finish()
    }
}

impl ResolverChain {
    pub fn new() -> Self {
        Self {
            resolvers: Vec::new(),
        }
    }

    pub fn push(mut self, resolver: Arc<dyn PrincipalResolver>) -> Self {
        self.resolvers.push(resolver);
        self
    }
}

impl Default for ResolverChain {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl PrincipalResolver for ResolverChain {
    async fn resolve(&self, request: &AuthRequest<'_>) -> Result<Principal, AuthError> {
        let mut last = AuthError::missing_credentials();
        for resolver in &self.resolvers {
            match resolver.resolve(request).await {
                Ok(principal) => return Ok(principal),
                Err(error) => last = error,
            }
        }
        Err(last)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principal() -> Principal {
        Principal::new(ScopeId::new("scope-a"), "user-1")
            .with_scopes(["tool.read", "tool.write"])
            .with_attribute("role", "operator")
    }

    #[test]
    fn scope_id_is_opaque_and_transparent_on_the_wire() {
        let scope = ScopeId::new("tenant-42");
        assert_eq!(scope.as_str(), "tenant-42");
        assert_eq!(serde_json::to_string(&scope).unwrap(), r#""tenant-42""#);
        assert_eq!(scope.to_string(), "tenant-42");
    }

    #[test]
    fn static_policy_checks_single_and_all_of_requirements() {
        let policy = StaticPolicy::new().with_admin_scopes(["scope.admin"]);
        let user = principal();
        assert!(
            policy
                .check(&user, &Requirement::scope("tool.read"))
                .is_allow()
        );
        assert!(
            !policy
                .check(&user, &Requirement::scope("tool.delete"))
                .is_allow()
        );
        assert!(
            policy
                .check(&user, &Requirement::all_of(["tool.read", "tool.write"]))
                .is_allow()
        );
        assert!(
            !policy
                .check(&user, &Requirement::all_of(["tool.read", "tool.delete"]))
                .is_allow()
        );
        assert!(!policy.is_admin(&user));
        let admin = Principal::new(ScopeId::new("scope-a"), "admin").with_scopes(["scope.admin"]);
        assert!(policy.is_admin(&admin));
    }

    #[tokio::test]
    async fn static_token_resolver_requires_the_exact_bearer() {
        let resolver = StaticTokenResolver::new("s3cret", principal());
        let mut headers = http_headers::HeaderMap::new();
        headers.insert(
            http_headers::HeaderName::from_static("authorization"),
            http_headers::HeaderValue::from_static("Bearer s3cret"),
        );
        let resolved = {
            let request = AuthRequest {
                headers: &headers,
                path_scope_hint: None,
            };
            resolver.resolve(&request).await.unwrap()
        };
        assert_eq!(resolved.scope.as_str(), "scope-a");
        assert_eq!(resolved.subject, "user-1");

        headers.insert(
            http_headers::HeaderName::from_static("authorization"),
            http_headers::HeaderValue::from_static("Bearer nope"),
        );
        let request = AuthRequest {
            headers: &headers,
            path_scope_hint: None,
        };
        assert!(resolver.resolve(&request).await.is_err());
    }
}
