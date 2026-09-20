# Extension guide

The core is a library; an application adapts it by implementing a small set of
traits. This guide shows a minimal embedding.

## 1. Principal and policy

```rust
use clypeus_core::principal::{
    AuthError, AuthRequest, Decision, PolicyEngine, Principal, PrincipalResolver,
    Requirement, ScopeId,
};

struct MyResolver;

#[async_trait::async_trait]
impl PrincipalResolver for MyResolver {
    async fn resolve(&self, request: &AuthRequest<'_>) -> Result<Principal, AuthError> {
        // Verify the token however your platform does, then map claims:
        let token = request
            .headers
            .get(reqwest::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .ok_or_else(AuthError::missing_credentials)?;
        // ...validate `token`...
        Ok(Principal::new(ScopeId::new("scope-1"), "user-1")
            .with_scopes(["demo.write"])
            .with_token(token))
    }
}

struct MyPolicy;

impl PolicyEngine for MyPolicy {
    fn check(&self, principal: &Principal, requirement: &Requirement) -> Decision {
        match requirement {
            Requirement::Scope(scope) if principal.has_scope(scope) => Decision::Allow,
            Requirement::AllOf(scopes) if scopes.iter().all(|scope| principal.has_scope(scope)) => {
                Decision::Allow
            }
            _ => Decision::Deny { code: "not_permitted" },
        }
    }
}
```

## 2. A tool

```rust
use clypeus_core::tools::{Approval, Egress, EgressAuth, Risk, Tool, ToolError, ToolExecContext};

struct ListItems;

#[async_trait::async_trait]
impl Tool for ListItems {
    fn name(&self) -> &'static str { "list_items" }
    fn description(&self) -> &'static str { "Lists items visible to the caller." }
    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
            "required": [],
            "additionalProperties": false
        })
    }
    fn output_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
            "additionalProperties": false
        })
    }
    fn required_scopes(&self) -> &'static [&'static str] { &["items.read"] }
    fn risk(&self) -> Risk { Risk::Read }
    fn approval(&self) -> Approval { Approval::Never }
    fn egress(&self) -> Egress {
        Egress { service: "items", method: "GET", path_template: "/v1/items" }
    }
    fn auth_mode(&self) -> EgressAuth { EgressAuth::Passthrough }
    async fn execute(&self, args: serde_json::Value, ctx: ToolExecContext)
        -> Result<serde_json::Value, ToolError>
    {
        let limit = args.get("limit").and_then(|value| value.as_u64()).unwrap_or(20).to_string();
        let response = ctx.call(
            self.egress(),
            &[],
            &[("limit", limit)],
        ).await?;
        let body: serde_json::Value = serde_json::from_slice(&response.body)
            .map_err(|_| ToolError::UpstreamUnavailable)?;
        Ok(body)
    }
}
```

Register tools and declare the egress bases:

```rust
use std::sync::Arc;
use clypeus_core::tools::ToolRegistry;

let tools = Arc::new(ToolRegistry::new().register(ListItems));
```

## 3. Wiring

```rust
use clypeus_core::broker::ToolBroker;
use clypeus_core::context::EmptyTurnContextProvider;
use clypeus_core::guard::NeutralGuardPolicy;
use clypeus_core::orchestrator::Orchestrator;
use clypeus_core::provider::ProviderRegistry;

let broker = Arc::new(ToolBroker::new(
    tools,
    approvals,          // Arc<dyn ApprovalStore>
    audit,              // Arc<dyn AuditSink>
    reqwest::Client::new(),
    vec![("items".to_string(), "https://items.internal".to_string())],
    None,               // TokenMinter
    None,               // UserTokenExchanger
    Some(50),           // write quota per day
)?);
let orchestrator = Orchestrator::new(providers, conversations, broker, Arc::new(NeutralGuardPolicy));
```

Embedders that serve HTTP without the standalone binary can reuse
`clypeus-server::app` with a fully custom `AppState`; the standalone server is
a reference composition, not a requirement.

## 4. Conformance

Add to the adapter's integration tests:

```rust
#[tokio::test]
async fn store_passes_conformance() {
    let store = MyStore::new().await;
    clypeus_conformance::run_store_conformance(&store).await.unwrap();
}
```
