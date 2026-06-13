#!/usr/bin/env bash
set -uo pipefail

# Launch the Claude Code OAuth refresh DAEMON (resident process).
#
# It keeps the host's ``.credentials.json`` access token above MIN_TTL so
# Cage trial containers always copy a token that outlives a full trial.
# The daemon sleeps until the token is about to fall below MIN_TTL, then
# refreshes via platform.claude.com (through the proxy). Refresh is a pure
# OAuth exchange — zero inference / usage cost.
#
# Tunables (env vars):
#   CREDS                 path to .credentials.json
#   MIN_TTL_SECONDS       on-disk AT lifetime floor (>= trial timeout + margin)
#   HTTPS_PROXY           proxy to reach platform.claude.com
#   REQUEST_TIMEOUT       http timeout for the refresh POST
#   RETRY_BACKOFF_SECONDS wait after a failed refresh
#   MAX_SLEEP_SECONDS     cap on a single sleep (notice external changes)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

CREDS="${CREDS:-${HOME}/.claude_p/.credentials.json}"
MIN_TTL_SECONDS="${MIN_TTL_SECONDS:-9000}"          # 2.5h  >= 2h trial + margin
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-30}"
RETRY_BACKOFF_SECONDS="${RETRY_BACKOFF_SECONDS:-60}"
MAX_SLEEP_SECONDS="${MAX_SLEEP_SECONDS:-1800}"
# Only the refresh request needs the proxy. Leave unset unless your environment
# requires one, e.g. HTTPS_PROXY=http://127.0.0.1:7890.
export HTTPS_PROXY="${HTTPS_PROXY:-}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/cage-resume-loops}"
mkdir -p "${LOG_DIR}" || exit 1
LOG_FILE="${LOG_FILE:-${LOG_DIR}/refresh-claude-oauth.log}"

PY="${PY:-python3}"
SCRIPT="${SCRIPT:-${SCRIPT_DIR}/refresh_claude_oauth.py}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "missing ${SCRIPT}" >&2
  exit 127
fi

echo "[$(date -Iseconds)] launching refresh daemon: creds=${CREDS} min_ttl=${MIN_TTL_SECONDS}s proxy=${HTTPS_PROXY}" >>"${LOG_FILE}"

exec "${PY}" "${SCRIPT}" \
  --daemon \
  --creds "${CREDS}" \
  --min-ttl-seconds "${MIN_TTL_SECONDS}" \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --retry-backoff-seconds "${RETRY_BACKOFF_SECONDS}" \
  --max-sleep-seconds "${MAX_SLEEP_SECONDS}" \
  --http-proxy "${HTTPS_PROXY}" \
  >>"${LOG_FILE}" 2>&1
