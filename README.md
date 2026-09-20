# Clypeus

Clypeus is a policy-first AI gateway. It sits between an application and one
or more model providers and owns four guarantees:

* **Isolation** — every stored record carries an opaque `ScopeId` and every
  query is filtered by the caller's scope. The core never interprets the
  scope; it only partitions by it.
* **Typed egress** — tools cannot build URLs. A tool declares a fixed target
  from an allowlist (service, method, path template) and the broker resolves
  it. Path parameters are restricted to identifiers.
* **Approvals** — write and destructive tools park before execution and run
  only against a hash-bound, expiring, one-shot grant with typed confirmation
  for destructive operations.
* **Untrusted tool data** — every tool result returned to a model is wrapped
  with an explicit untrusted marker and is never promoted to an instruction.

The core is a library (`clypeus-core`) with an optional standalone server
(`clypeus`). Applications embed the core and plug in their own principal
resolver, policy engine, tools, functions, and stores; the standalone server
bundles a JWKS/static resolver, a file secret store, OpenAI-compatible and
Anthropic providers, and memory/SQLite/PostgreSQL stores.

## Crates

| Crate | Purpose |
|---|---|
| `clypeus-core` | Principals, policy, providers, tools, broker, approvals, guard, orchestrator, SSE, rate limiting, metrics |
| `clypeus-server` | Standalone HTTP surface (binary `clypeus`), OpenAPI, health/readiness/metrics |
| `clypeus-provider-openai` | OpenAI-compatible Chat Completions provider |
| `clypeus-provider-anthropic` | Anthropic Messages provider |
| `clypeus-store-memory` | In-memory store for tests and embedded deployments |
| `clypeus-store-sqlite` | SQLite store (single node, edge, development) |
| `clypeus-store-postgres` | PostgreSQL store (production) |
| `clypeus-store-sql` | Shared SQL implementation used by the SQLite and PostgreSQL stores |
| `clypeus-auth-jwks` | JWKS/static bearer principal resolver |
| `clypeus-secret-file` | Environment/file secret store |
| `clypeus-conformance` | Reusable conformance suite for store and adapter implementations |

## Quick start

```bash
# Run the standalone server with the in-memory store.
export CLYPEUS_STATIC_TOKEN=dev-token
export CLYPEUS_STATIC_SCOPE=default
export CLYPEUS_SECRET_DIR=./secrets
export CLYPEUS_SEED_SCOPE=default
export CLYPEUS_SEED_BASE_URL=https://api.openai.com
export CLYPEUS_SEED_API_KEY=sk-...
export CLYPEUS_SEED_MODEL=gpt-4o-mini
cargo run -p clypeus-server --bin clypeus
```

Then:

```bash
curl -s localhost:8080/v1/threads \
  -H "Authorization: Bearer dev-token" \
  -H 'content-type: application/json' \
  -d '{"title":"Hi"}'
```

Or with Docker Compose (PostgreSQL + server):

```bash
CLYPEUS_SEED_API_KEY=sk-... docker compose up --build
```

## API

The standalone API is documented by `openapi/clypeus-v1.json`; regenerate it
with `clypeus --print-openapi`. Highlights:

* `POST /v1/completions` — buffered or streamed completion.
* `GET /v1/models`, `POST /v1/models/preview` — catalog and probe.
* Threads, branched messages, feedback, usage: `/v1/threads`, `/v1/messages`.
* Approvals: `POST /v1/tool-calls/{id}/approvals`.
* Tools and functions: `GET /v1/tools`, `GET|POST /v1/functions`.
* Audit: `GET /v1/audit`, `GET /v1/audit/export`.
* Administration: `/admin/v1/scopes/{scope}/settings`.
* Operations: `/healthz`, `/readyz`, `/metrics`, `/openapi/v1.json`.

The stream closes with a terminal `turn_completed` event followed by
`data: [DONE]`.

## Extension points

The core defines a small, fixed set of traits. See `docs/extension-guide.md`
for a worked adapter example.

* `PrincipalResolver`, `PolicyEngine` — authentication and authorization.
* `Provider` — a model backend.
* `Tool`, `ToolEgress`, `TokenMinter`, `UserTokenExchanger` — tool execution.
* `AiFunction` — server-side function registry.
* `ConversationStore`, `ApprovalStore`, `ScopeSettingsStore`, `AuditSink`,
  `AuditReader` — storage.
* `SecretStore`, `RateLimiter` — operational adapters.
* `PromptProfile`, `ProfileStore`, `TurnContextProvider` — prompting.

`clypeus-conformance` runs the contract suite against store and broker
implementations; run it from an adapter's integration tests.

## Development

```bash
./scripts/ci.sh                                                  # all gates
CLYPEUS_TEST_DATABASE_URL=postgres://... cargo test -p clypeus-store-postgres
```

`ci/github-ci.yml` is the canonical GitHub Actions workflow (isolation gate,
fmt/clippy/tests, PostgreSQL conformance, OpenAPI drift, cargo-deny). It is
installed at `.github/workflows/ci.yml` by a token that carries the GitHub
`workflow` scope:

```bash
mkdir -p .github/workflows && cp ci/github-ci.yml .github/workflows/ci.yml
```

## License

Dual-licensed under either of

* Apache License, Version 2.0 (`LICENSE-APACHE`)
* MIT license (`LICENSE-MIT`)

at your option. Contributions are accepted under the same terms.
