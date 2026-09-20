# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and the project adheres to Semantic Versioning.

## [1.0.0] - 2026-09-20

First stable release. The public surface is the crate set listed in the
README, the `clypeus` binary, and `openapi/clypeus-v1.json`. Consumers pin an
exact tag; a breaking change moves the API to `/v2` and the crates to a new
major version.

### Added

* Embedding guide (`docs/embedding.md`): in-process library composition,
  standalone server deployment, extension points, and the conformance suite.
* Release guide (`docs/release.md`) and a signed image pipeline
  (`scripts/release.sh`) with a CycloneDX SBOM and cosign signature.

### Changed

* Crate version is `1.0.0`; repository metadata points at the canonical
  repository host.
* `PromptProfile`, `ProfileStore`, and `TurnContextProvider` are wired into
  every turn, so embedders can supply profile text and per-turn context.
* `Principal.attributes` are projected into the tool execution context.
* `connect` can own its schema migrations for embedders that manage the
  schema themselves.

### Fixed

* The streamed-turn guard now disarms after a terminal state is persisted,
  so completed replies are no longer rewritten as `turn_incomplete`. Earlier
  releases lost the reply content, reasoning, and usage on streamed turns.

## [0.1.3] - 2026-09-20

### Added

* Schema-owning connections for embedders with their own migration pipeline.

## [0.1.2] - 2026-09-20

### Added

* Prompt profile and turn context providers are applied to turns.

## [0.1.1] - 2026-09-20

### Added

* Principal attributes are projected into the tool execution context.

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
