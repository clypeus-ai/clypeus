#!/usr/bin/env bash
# Runs the same gates as ci/github-ci.yml locally: formatting, clippy with
# warnings denied, the full test suite, OpenAPI drift, and the isolation gate.
set -euo pipefail

echo "==> cargo fmt"
cargo fmt --all -- --check

echo "==> cargo clippy"
cargo clippy --workspace --all-targets -- -D warnings

echo "==> cargo test"
cargo test --workspace

echo "==> OpenAPI drift"
cargo run -q -p clypeus-server --bin clypeus -- --print-openapi 2>/dev/null > /tmp/clypeus-v1.json
diff -u openapi/clypeus-v1.json /tmp/clypeus-v1.json

echo "==> isolation gate"
./scripts/check-isolation.sh

echo "==> cargo-deny"
if command -v cargo-deny >/dev/null 2>&1; then
  cargo deny check advisories licenses bans sources
else
  echo "cargo-deny is not installed; the CI workflow runs it on every change"
fi

echo "all gates passed"
