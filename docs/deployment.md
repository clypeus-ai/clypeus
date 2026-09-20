# Deployment

## Standalone server

The `clypeus` binary reads `CLYPEUS_*` environment variables.

| Variable | Meaning |
|---|---|
| `CLYPEUS_BIND_ADDR` | Listen address (default `0.0.0.0:8080`) |
| `CLYPEUS_STORE` | `memory`, `sqlite`, or `postgres` (default `memory`) |
| `CLYPEUS_SQLITE_PATH` | SQLite file (default `clypeus.db`) |
| `CLYPEUS_DATABASE_URL` | PostgreSQL URL (required for `postgres`) |
| `CLYPEUS_STATIC_TOKEN` | Fixed bearer token for single-tenant/simple setups |
| `CLYPEUS_STATIC_SCOPE`, `CLYPEUS_STATIC_SUBJECT`, `CLYPEUS_STATIC_SCOPES` | Principal assigned to the static token |
| `CLYPEUS_JWKS_PATH` | JWKS document; enables JWT resolution |
| `CLYPEUS_JWKS_SCOPE_CLAIM`, `CLYPEUS_JWKS_SUBJECT_CLAIM`, `CLYPEUS_JWKS_SCOPES_CLAIM` | Claim mapping (defaults `scope_id`, `sub`, `scopes`) |
| `CLYPEUS_ADMIN_SCOPES` | Comma-separated scope codes that make a principal administrative |
| `CLYPEUS_SECRET_DIR` | Directory backing the file secret store |
| `CLYPEUS_EGRESS_SERVICES` | `name=https://base,name2=...` tool egress allowlist |
| `CLYPEUS_ALLOW_PRIVATE_PROVIDERS` | Allow provider base URLs on private addresses |
| `CLYPEUS_SEED_SCOPE`, `CLYPEUS_SEED_BASE_URL`, `CLYPEUS_SEED_API_KEY`, `CLYPEUS_SEED_PROVIDER`, `CLYPEUS_SEED_MODEL` | Startup seed for a scope |
| `CLYPEUS_CHAT_RATE_LIMIT`, `CLYPEUS_CHAT_RATE_WINDOW_SECONDS`, `CLYPEUS_FUNCTION_RATE_LIMIT` | Rate limits |
| `CLYPEUS_TURN_BUDGET_SECONDS`, `CLYPEUS_MAX_TOOL_ROUNDS`, `CLYPEUS_MAX_TOOL_CALLS`, `CLYPEUS_MAX_TURN_RESULT_BYTES` | Turn budgets |
| `CLYPEUS_APPROVAL_TTL_SECONDS`, `CLYPEUS_WRITE_QUOTA_PER_DAY` | Approval and write policy |
| `CLYPEUS_AUDIT_RETENTION_DAYS` | Audit retention (`0` disables purging) |
| `CLYPEUS_STALE_TURN_SECONDS` | Age after which unfinished turns are failed |
| `RUST_LOG` | Tracing filter (default `info`) |

## Docker Compose

```bash
CLYPEUS_SEED_API_KEY=sk-... docker compose up --build
```

The compose file runs PostgreSQL and the server. Provider credentials live in
the `clypeus-secrets` volume mounted at `/var/lib/clypeus/secrets`; the
settings API writes provider keys there.

## Production notes

* Run PostgreSQL with regular backups; the store is the only state.
* Terminate TLS in front of the server; it speaks plain HTTP.
* Point `/metrics` at your Prometheus scraper and alert on
  `clypeus_injection_blocked_total`, `clypeus_tool_approvals_total{decision="replayed"}`,
  and provider failure rates.
* Set `CLYPEUS_ADMIN_SCOPES` deliberately: it gates settings and audit reads.
* Leave `CLYPEUS_ALLOW_PRIVATE_PROVIDERS` unset unless a provider genuinely
  lives on a private network.

## CI

The canonical workflow lives at `ci/github-ci.yml`; install it at
`.github/workflows/ci.yml` (a token with the `workflow` scope is required to
push workflow files) or run the identical gates locally with
`./scripts/ci.sh`.
