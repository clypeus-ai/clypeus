# Embedding Clypeus

Clypeus is designed to be embedded. The same library that backs the standalone
server can run inside an application process, with the application supplying
principals, policy, tools, functions, stores, and secrets.

## Two deployment shapes

| Shape | When to use it | Entry point |
|---|---|---|
| Standalone server | A dedicated gateway process that other services call over HTTP | `clypeus` binary (`clypeus-server`) |
| Embedded library | The gateway semantics run in-process, next to the application's own stores and auth | `clypeus-core` types composed in the application's service |

Both shapes share the core: providers, the tool broker, the approval engine,
the guard, the orchestrator, and the SSE contract.

## Embedding the library

Add the crates you need and pin an exact tag:

```toml
[dependencies]
clypeus-core = { git = "https://github.com/vitkuz573/clypeus", tag = "v1.0.0" }
clypeus-provider-openai = { git = "https://github.com/vitkuz573/clypeus", tag = "v1.0.0" }
clypeus-store-postgres = { git = "https://github.com/vitkuz573/clypeus", tag = "v1.0.0" }
```

The composition root is small by design:

1. **Storage.** Connect a store with `clypeus_store_postgres::connect` (applies
   the `clypeus_*` migrations) or `connect_existing` when the application owns
   the schema through its own migration tool. The returned `SqlStore`
   implements `ScopeSettingsStore`, `ConversationStore`, `ApprovalStore`,
   `AuditSink`, and `AuditReader`.
2. **Providers.** Register provider implementations in a `ProviderRegistry`.
   `clypeus-provider-openai` and `clypeus-provider-anthropic` cover the
   bundled shapes; a custom provider only implements `Provider`.
3. **Principal and policy.** Implement `PrincipalResolver` and `PolicyEngine`.
   The core passes opaque `ScopeId`, `subject`, and `scopes` through
   unchanged; the embedder decides what they mean.
4. **Tools and functions.** Register `Tool` implementations in a
   `ToolRegistry` and `AiFunction` implementations in a `FunctionRegistry`.
   A `ToolBroker` receives the registry, the `ApprovalStore`, the
   `AuditSink`, and the egress adapters (`ToolEgress`, `TokenMinter`,
   `UserTokenExchanger`).
5. **Prompting.** Implement `PromptProfile` and `TurnContextProvider` (or use
   the static helpers) to supply system text and per-turn context. Tool
   results are always wrapped as untrusted by the core.
6. **Orchestration.** Build an `Orchestrator` from the provider registry,
   conversation store, broker, and `GuardPolicy`. `spawn_streamed` returns a
   byte stream of the core SSE contract (`context`, `tool_call`,
   `tool_result`, `turn_completed`, `[DONE]`); `run_buffered` returns a
   `TurnOutcome` without streaming.

```rust,ignore
let orchestrator = Orchestrator::new(
    Arc::new(provider_registry),
    Arc::clone(&store) as Arc<dyn ConversationStore>,
    Arc::new(broker),
    Arc::new(MyGuardPolicy),
);
let stream = Arc::clone(&orchestrator).spawn_streamed(request);
```

The embedder is responsible for HTTP: authentication happens before the
orchestrator, and the core never reads the network for anything except
provider calls and broker egress.

## Extension points

The core defines a fixed, minimal trait set. See
[`docs/extension-guide.md`](extension-guide.md) for a worked adapter.

| Trait | Responsibility |
|---|---|
| `PrincipalResolver` | HTTP request → `Principal` |
| `PolicyEngine` | `Requirement` → allow/deny decision |
| `Provider` | Model catalog, probe, buffered and streamed completions |
| `Tool` | One tool: schema, risk, approval kind, fixed egress target, execution |
| `ToolEgress` | Executes an allowlisted egress request |
| `TokenMinter` | Mints a per-call token for a tool call |
| `UserTokenExchanger` | Exchanges the caller's token for a downstream audience |
| `AiFunction` | Server-side function: schema, prompt, validation, repair |
| `ConversationStore` | Threads, branched messages, feedback, usage |
| `ApprovalStore` | Parked tool calls, approvals, write quota |
| `ScopeSettingsStore` | Per-scope provider configuration |
| `AuditSink`, `AuditReader` | Append-only audit trail and reads |
| `SecretStore` | Provider credentials |
| `PromptProfile`, `ProfileStore` | System prompt selection |
| `TurnContextProvider` | Per-turn context snapshot |
| `GuardPolicy` | Refusal text and disclosure targets |
| `RateLimiter` | Token-bucket limiting keyed by scope/subject |

## Conformance for embedders

`clypeus-conformance` ships the contract suites the core uses against its own
adapters. Run them from an adapter's integration tests so a custom store or
broker fails the build when it drifts from the contract:

```rust,ignore
use clypeus_conformance::{run_broker_conformance, run_store_conformance};

#[tokio::test]
async fn my_store_meets_the_contract() {
    let store = my_store().await;
    run_store_conformance(&store).await.unwrap();
}

#[tokio::test]
async fn my_broker_meets_the_contract() {
    let approvals = Arc::new(my_approval_store().await);
    let audit = Arc::new(my_audit_sink().await);
    run_broker_conformance(approvals, audit).await.unwrap();
}
```

`run_store_conformance` covers settings, conversation, approval, and audit
behavior; `run_broker_conformance` covers approvals, replay refusal, quota,
and egress policy; `run_egress_conformance` and `run_guard_conformance` cover
the URL safety rules and the prompt guard.

## Versioning and pinning

Consumers pin the exact release tag and commit the resulting `Cargo.lock`. A
breaking core change bumps the major version and moves the HTTP surface to a
new prefix; the previous major receives fixes for one minor release. See
[`docs/release.md`](release.md).
