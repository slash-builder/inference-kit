#!/usr/bin/env bash
# Fail if the crate that `cargo package` would build contains anything that must
# never ship: cargo config or registry state, credential files, env files, or
# build output. Only file NAMES are ever printed, never contents.
#
# Usage:
#   scripts/check-package.sh              check the real package list
#   scripts/check-package.sh --stdin      check a file list read from stdin (self-test)
#
# `--allow-dirty` is deliberate: the point is to see what WOULD be packaged even
# when the working tree holds stray files (which is exactly the failure this guards).
set -euo pipefail

if [ "${1:-}" = "--stdin" ]; then
  list=$(cat)
else
  list=$(cargo package --list --allow-dirty 2>/dev/null)
fi

if [ -z "$list" ]; then
  echo "check-package: 'cargo package --list' produced no files; refusing to pass." >&2
  exit 2
fi

# Anchored so that source files such as src/credentials.rs and
# src/credentials/conformance.rs do NOT match: only a file literally named
# credentials / credentials.toml, or anything under a .cargo/ directory.
bad=$(printf '%s\n' "$list" | grep -E \
  '(^|/)\.cargo/|(^|/)credentials(\.toml)?$|(^|/)\.env([./]|$)|(^|/)target[^/]*/|(^|/)\.netrc$|\.(pem|key|p12)$' || true)

if [ -n "$bad" ]; then
  echo "check-package: FAIL. The package would contain files that must never ship:" >&2
  printf '%s\n' "$bad" | head -20 >&2
  total=$(printf '%s\n' "$bad" | wc -l | tr -d ' ')
  [ "$total" -gt 20 ] && echo "  ... and $((total - 20)) more" >&2
  echo "Fix the manifest 'include' list and/or .gitignore; do not weaken this check." >&2
  exit 1
fi

echo "check-package: OK ($(printf '%s\n' "$list" | wc -l | tr -d ' ') files, none forbidden)."
