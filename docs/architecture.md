# Architecture

## Pipeline

```
client ── HTTP ──▶ clypeus-server (or an embedding application)
                     │  PrincipalResolver  → Principal { scope, subject, scopes }
                     │  PolicyEngine       → allow/deny scope checks
                     │  ProfileStore + TurnContextProvider → system message
                     ▼
                  Orchestrator ──▶ Provider (OpenAI-compatible, Anthropic, ...)
                     │                  ▲
                     │                  │  tool results (untrusted-wrapped)
                     ▼                  │
                  ToolBroker ──▶ ToolRegistry ──▶ ToolEgress (allowlist)
                     │                                └── TokenMinter / UserTokenExchanger
                     ▼
                  ConversationStore / ApprovalStore / AuditSink
```

One turn is a bounded loop:

1. The guard classifies the latest user message. A disclosure or override hit
   produces a refusal without a provider call.
2. The provider is asked with the scope-filtered tool catalog.
3. Requested tools are executed: approval-gated tools park the turn;
   independent read-only calls run concurrently.
4. Tool results are projected, redacted, wrapped as untrusted data, and fed
   back to the provider.
5. The loop stops on the first answer that requests no tools, on the turn
   budget, or on a parked approval.

Budgets: 4 tool rounds, 8 tool calls, 64 KiB of projected result bytes, and a
90-second turn budget by default (`CLYPEUS_*` overrides).

## Isolation model

`ScopeId` is an opaque partition. The core never compares it to a path or a
claim: applications resolve identity into a `Principal` and store adapters
filter every read and write by `scope_id` and `subject`. Cross-scope access
exists only through administrative routes, gated by `PolicyEngine::is_admin`.

## Egress model

A tool declares `Egress { service, method, path_template }` and an
`EgressAuth`. The broker resolves the target against an allowlist built at
startup and refuses unknown services, unsafe templates, and non-identifier
path parameters. The URL origin can never be influenced by model arguments.
Credentials are either the caller's own token (optionally exchanged for a
downstream audience) or a per-call minted token; nothing is cached.

## Approvals

Parking validates scopes, arguments, and the write quota, then records a call
with `awaiting_approval` and returns a challenge containing the SHA-256 of the
canonical arguments, an expiry, and a typed-confirmation field for destructive
tools. Execution requires:

* the recorded state `approved` bound to the presented approval id (one-shot),
* a recomputed arguments hash equal to the approved hash,
* the caller still holding every required scope,
* the approval window still open.

## Storage

`clypeus-store-memory` implements the contracts in memory;
`clypeus-store-sqlite` and `clypeus-store-postgres` share the SQL
implementation in `clypeus-store-sql` and own their migrations. Identifiers,
timestamps (RFC 3339 UTC), and JSON payloads are stored as text so both SQL
backends share one schema shape. Every query binds values; there is no string
interpolation of caller data.
