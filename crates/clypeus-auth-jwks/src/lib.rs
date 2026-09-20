//! JWKS-based principal resolution for standalone deployments.
//!
//! Validates an RS256 bearer JWT against a JSON Web Key Set, then maps its
//! claims onto an opaque [`Principal`]: a scope claim (the isolation
//! partition), a subject claim, a list of scope codes, and optional attribute
//! claims. Claim names are configuration, because the core does not define
//! what a subject or a scope means.

use std::collections::BTreeMap;

use clypeus_core::principal::{AuthError, AuthRequest, Principal, PrincipalResolver, ScopeId};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header, jwk::JwkSet};
use serde::Deserialize;
use serde_json::Value;

/// Static configuration of the resolver.
#[derive(Debug, Clone)]
pub struct JwksResolverConfig {
    /// Claim carrying the isolation scope. When absent, `default_scope` is
    /// used.
    pub scope_claim: String,
    /// Claim carrying the subject. Defaults to `sub`.
    pub subject_claim: String,
    /// Claim carrying the authorization codes. Defaults to `scopes`.
    pub scopes_claim: String,
    /// Additional claims copied into `Principal::attributes`.
    pub attribute_claims: Vec<String>,
    /// Scope used when the token carries no scope claim.
    pub default_scope: Option<ScopeId>,
    /// Required issuer, when set.
    pub issuer: Option<String>,
    /// Required audience, when set.
    pub audience: Option<String>,
    /// Rejects a request whose path scope hint differs from the token scope.
    pub enforce_path_scope: bool,
    /// Whether the `scope` claim may be a space-separated string.
    pub scopes_are_space_delimited: bool,
}

impl Default for JwksResolverConfig {
    fn default() -> Self {
        Self {
            scope_claim: "scope_id".to_string(),
            subject_claim: "sub".to_string(),
            scopes_claim: "scopes".to_string(),
            attribute_claims: Vec::new(),
            default_scope: None,
            issuer: None,
            audience: None,
            enforce_path_scope: false,
            scopes_are_space_delimited: false,
        }
    }
}

/// Resolver validating RS256 tokens against a JWKS document.
#[derive(Debug, Clone)]
pub struct JwksResolver {
    jwks: JwkSet,
    config: JwksResolverConfig,
}

#[derive(Debug, Deserialize)]
struct RawClaims {
    #[serde(flatten)]
    claims: BTreeMap<String, Value>,
}

impl JwksResolver {
    pub fn new(jwks: JwkSet, config: JwksResolverConfig) -> Self {
        Self { jwks, config }
    }

    /// Parses a JWKS document.
    pub fn from_jwks_json(raw: &str, config: JwksResolverConfig) -> Result<Self, String> {
        let jwks: JwkSet = serde_json::from_str(raw).map_err(|error| error.to_string())?;
        Ok(Self::new(jwks, config))
    }

    /// Reads the bearer token from the authorization header.
    pub fn bearer<'a>(request: &'a AuthRequest<'_>) -> Option<&'a str> {
        let header = request
            .headers
            .get(reqwest::header::AUTHORIZATION)?
            .to_str()
            .ok()?;
        header
            .strip_prefix("Bearer ")
            .or_else(|| header.strip_prefix("bearer "))
    }

    /// Verifies the token and returns its claims.
    pub fn verify(&self, token: &str) -> Result<BTreeMap<String, Value>, AuthError> {
        let header = decode_header(token)
            .map_err(|error| AuthError::with_detail("invalid_credentials", error.to_string()))?;
        let Some(kid) = header.kid.as_deref() else {
            return Err(AuthError::with_detail(
                "invalid_credentials",
                "token header has no kid",
            ));
        };
        let Some(jwk) = self.jwks.find(kid) else {
            return Err(AuthError::with_detail(
                "invalid_credentials",
                "no matching key",
            ));
        };
        let key = DecodingKey::from_jwk(jwk).map_err(|error| {
            AuthError::with_detail("invalid_credentials", format!("key: {error}"))
        })?;
        let mut validation = Validation::new(Algorithm::RS256);
        if let Some(issuer) = self.config.issuer.as_deref() {
            validation.set_issuer(&[issuer]);
        } else {
            validation.iss = None;
        }
        if let Some(audience) = self.config.audience.as_deref() {
            validation.set_audience(&[audience]);
        } else {
            validation.validate_aud = false;
        }
        let data = decode::<RawClaims>(token, &key, &validation)
            .map_err(|error| AuthError::with_detail("invalid_credentials", error.to_string()))?;
        Ok(data.claims.claims)
    }
}

fn claim_str<'a>(claims: &'a BTreeMap<String, Value>, name: &str) -> Option<&'a str> {
    claims.get(name).and_then(Value::as_str)
}

fn claim_string_list(
    claims: &BTreeMap<String, Value>,
    name: &str,
    space_delimited: bool,
) -> Vec<String> {
    match claims.get(name) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        Some(Value::String(text)) if space_delimited => {
            text.split_whitespace().map(str::to_string).collect()
        }
        Some(Value::String(text)) => vec![text.clone()],
        _ => Vec::new(),
    }
}

#[async_trait::async_trait]
impl PrincipalResolver for JwksResolver {
    async fn resolve(&self, request: &AuthRequest<'_>) -> Result<Principal, AuthError> {
        let token = Self::bearer(request).ok_or_else(AuthError::missing_credentials)?;
        let claims = self.verify(token)?;

        let scope = match claim_str(&claims, &self.config.scope_claim) {
            Some(scope) if !scope.trim().is_empty() => ScopeId::new(scope.trim()),
            _ => self.config.default_scope.clone().ok_or_else(|| {
                AuthError::with_detail("invalid_credentials", "scope claim missing")
            })?,
        };
        if self.config.enforce_path_scope
            && let Some(hint) = request.path_scope_hint
            && hint != scope.as_str()
        {
            return Err(AuthError::new("scope_mismatch"));
        }
        let subject = claim_str(&claims, &self.config.subject_claim)
            .filter(|subject| !subject.trim().is_empty())
            .ok_or_else(|| AuthError::with_detail("invalid_credentials", "subject claim missing"))?
            .to_string();
        let scopes = claim_string_list(
            &claims,
            &self.config.scopes_claim,
            self.config.scopes_are_space_delimited,
        );
        let attributes = self
            .config
            .attribute_claims
            .iter()
            .filter_map(|claim| {
                claims
                    .get(claim)
                    .and_then(|value| match value {
                        Value::String(text) => Some(text.clone()),
                        Value::Number(number) => Some(number.to_string()),
                        Value::Bool(flag) => Some(flag.to_string()),
                        _ => None,
                    })
                    .map(|value| (claim.clone(), value))
            })
            .collect();

        let mut principal = Principal {
            scope,
            subject,
            scopes,
            attributes,
            token: Some(clypeus_core::secrets::SecretString::new(token)),
        };
        principal
            .attributes
            .entry("token_present".to_string())
            .or_insert_with(|| "true".to_string());
        Ok(principal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_helpers_handle_arrays_and_scope_strings() {
        let mut claims = BTreeMap::new();
        claims.insert("scopes".into(), serde_json::json!(["a", "b"]));
        assert_eq!(claim_string_list(&claims, "scopes", false), vec!["a", "b"]);
        claims.insert("scope".into(), serde_json::json!("a b"));
        assert_eq!(claim_string_list(&claims, "scope", true), vec!["a", "b"]);
        assert_eq!(claim_string_list(&claims, "scope", false), vec!["a b"]);
        assert!(claim_string_list(&claims, "missing", true).is_empty());
    }

    #[tokio::test]
    async fn missing_authorization_is_rejected() {
        let resolver =
            JwksResolver::new(JwkSet { keys: Vec::new() }, JwksResolverConfig::default());
        let headers = reqwest::header::HeaderMap::new();
        let request = AuthRequest {
            headers: &headers,
            path_scope_hint: None,
        };
        let error = resolver.resolve(&request).await.unwrap_err();
        assert_eq!(error.code, "missing_credentials");
    }
}
