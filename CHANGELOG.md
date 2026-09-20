# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and the project adheres to Semantic Versioning.

## [0.1.0] - 2026-09-20

Initial release of the Clypeus core.

### Added

* `clypeus-core`: opaque `Principal`/`ScopeId`, pluggable `PolicyEngine`,
  provider registry, tool registry with strict JSON Schema validation,
  projection and redaction, tool broker with a typed egress allowlist,
  per-call token minting, approval engine (argument hashing, TTL, one-shot
  grants, typed confirmation, replay refusal, write quota), prompt-disclosure
  guard, bounded tool-loop orchestrator, token-bucket rate limiter,
  Prometheus metrics, and the SSE event contract.
* `clypeus-provider-openai` and `clypeus-provider-anthropic`: buffered and
  streamed completions, model catalogs with reasoning levels, tool calls and
  tool results, usage accounting, and retry-without-reasoning on providers
  that reject reasoning parameters.
* `clypeus-store-memory`, `clypeus-store-sqlite`, `clypeus-store-postgres`:
  scope settings, threads, branched messages with versions, feedback, usage,
  tool calls, approvals, and an append-only audit trail.
* `clypeus-auth-jwks`, `clypeus-secret-file`: standalone principal resolution
  and secrets.
* `clypeus-server` (binary `clypeus`): the standalone HTTP API, OpenAPI
  document, health/readiness/metrics probes, stale-turn and approval-expiry
  sweeps, and audit retention.
* `clypeus-conformance`: reusable store, egress, guard, and broker contract
  suites.
* CI: formatting, clippy with `-D warnings`, unit and integration tests,
  PostgreSQL conformance, OpenAPI drift, cargo-deny, and the isolation gate.
