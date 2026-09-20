#!/usr/bin/env bash
# Isolation gate: the repository must not reference any other product,
# service, or database namespace. Run in CI and before every release.
#
# The gate excludes itself because the script necessarily contains the
# forbidden tokens it searches for.
set -euo pipefail

PATTERN='remote ?master|remotemaster|\brm[-_][a-z0-9]|sec_ai_|fleet-control|access-policy|host-gateway|relay_host|rm_perm|rm_tenant'

fail=0

# Working tree: code, docs, configuration, everything except .git.
if command -v rg >/dev/null 2>&1; then
  if rg -i -n --hidden -g '!.git' -g '!Cargo.lock' -g '!scripts/check-isolation.sh' "$PATTERN" .; then
    echo "isolation gate: forbidden token in the working tree" >&2
    fail=1
  fi
else
  if grep -rInE --exclude-dir=.git --exclude=check-isolation.sh -e "$PATTERN" .; then
    echo "isolation gate: forbidden token in the working tree" >&2
    fail=1
  fi
fi

# History: commit messages and authorship trailers.
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  if git log --format='%s%n%b' | grep -iE "$PATTERN"; then
    echo "isolation gate: forbidden token in git history" >&2
    fail=1
  fi
fi

if [ "$fail" -ne 0 ]; then
  exit 1
fi
echo "isolation gate: clean"
