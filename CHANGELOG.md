# Changelog

All notable changes to this project are documented here. The format follows
Keep a Changelog and the project adheres to Semantic Versioning.

## [Unreleased]

Callers can request a JSON document that conforms to a schema.

### Added

* `clypeus-core`: `OutputFormat` (`Text`, the default, or `JsonSchema { name,
  schema }`) carried on `CompletionRequest::output` and `TurnRequest::output`,
  and `ProviderError::UnsupportedOutput` with the stable code
  `provider_output_not_available` for a provider that cannot constrain
  decoding. `AiFunction` runs ask the provider for the function's own
  `output_schema()`.
* `clypeus-provider-openai`: sends a requested schema as a strict
  `response_format`:
  `{"type":"json_schema","json_schema":{"name":...,"strict":true,"schema":...}}`.
  `clypeus-provider-anthropic` returns `UnsupportedOutput`, because the
  Messages API has no field for it.
* The requested format is stored on the assistant message row
  (`clypeus_messages.output_format`, migration `0002_message_output_format.sql`,
  `NOT NULL DEFAULT '{"type":"text"}'` in both SQL stores), so a turn that
  parks for a tool approval is resumed with the format of the original
  request, which the resuming request never saw.
* `POST /v1/completions`, `POST /v1/threads/{id}/messages`,
  `PATCH /v1/messages/{id}` and `POST /v1/messages/{id}/regenerate` accept an
  `output` field (absent means free text); `MessageDto` exposes
  `outputFormat`.

## [1.2.0] - 2026-09-27

Cancelled turns become a first-class terminal state with an explicit stop.

### Added

* `MessageStatus::Stopped` (`stopped`): the terminal state of a turn that was
  cancelled or whose stream consumer disconnected. Partial content, reasoning,
  and usage are persisted exactly as streamed; the state is exposed through
  `MessageDto`, `UsageEntryDto`, and the thread usage row.
* `POST /v1/messages/{id}/stop`: stops a running assistant turn and returns the
  persisted `TurnResponse`. It is idempotent (a terminal turn is returned
  unchanged) and answers `404 not_found` for an unknown turn. Stopping a parked
  turn also refuses its pending tool decisions.
* `ConversationStore::stop_turn`: atomically finalizes a
  `pending`/`streaming`/`awaiting_approval` turn as `stopped` without touching
  its partial output, and leaves terminal turns unchanged.
* `Orchestrator::stop_turn`/`StopHandle`: a per-message cancellation registry
  for streamed turns; `StopHandle::wait` resolves once the terminal state is
  persisted. `spawn_streamed` and `spawn_resumed_streamed` register their
  turns automatically.
* `TurnOutcome::Stopped { completion, reason }` with `StopReason::Requested`
  and `StopReason::ClientDisconnected`.
* `turn_started` SSE event (`{ threadId, userMessageId, assistantMessageId }`),
  emitted before `context` so a consumer can address the stop route while the
  turn runs.

### Changed

* A dropped SSE/HTTP consumer now finalizes the turn as `stopped` with
  `errorDetail: "client_disconnected"` instead of `provider_unreachable`; the
  cancellation is selected against the in-flight provider read, so it does not
  wait for the next delta or the next tool result.
* Streamed stop requests cancel the provider read immediately and skip any
  remaining tool calls. The `turn_completed` view (with `status: "stopped"`)
  still closes the stream when the consumer is connected.
* `error_detail` is `null` for an explicit stop.
* Workspace version is `1.2.0`.

## [1.1.0] - 2026-09-27

Reasoning levels become open, catalog-validated strings.

### Changed

* `clypeus-core`: removed the closed `ReasoningLevel` enum. Reasoning levels
  are now plain strings: `CompletionRequest.reasoning` is `Option<String>`,
  and a value is valid when the selected model's catalog advertises it
  verbatim. Absent, empty, and `"default"` mean "no explicit override"; any
  other advertised value reaches the provider verbatim.
* `select_model` (turns and functions) validates requested levels against the
  effective model's `reasoning_levels` exactly and case-sensitively, still
  rejecting off-catalog values with `function_reasoning_not_available`. A
  model that advertises no levels accepts only `"default"`.
* `clypeus-provider-openai`: sends `reasoning_effort` verbatim, preserves the
  catalog's level spelling, and no longer converts through an enum. The
  retry-without-reasoning status set is unchanged.
* `clypeus-provider-anthropic`: maps known names to thinking budgets
  (`minimal`/`low` → 1024, `medium` → 4096, `high` → 8192, `xhigh` → 16000,
  `max` → 32000), omits thinking for `none`, and returns the typed
  `provider_reasoning_not_available` error naming the model and value for
  unrecognized names. The Anthropic catalog still advertises no levels.
* HTTP request fields (`reasoningLevel` on completions, turns, and functions)
  and response fields (`MessageDto`, usage, function diagnostics) carry the
  open string; the OpenAPI document documents the semantics.
* Workspace version is `1.1.0`.

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
