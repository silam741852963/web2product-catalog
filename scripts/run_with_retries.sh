#!/usr/bin/env bash
set -uo pipefail   # no -e because we handle non-zero exits manually

# Configurable via env vars
RETRY_EXIT_CODE="${RETRY_EXIT_CODE:-17}"
OUT_DIR="${OUT_DIR:-outputs}"                           # must match --out-dir in run.py

# Strict mode config: only applied once all companies have been attempted at least once
STRICT_MIN_RETRY_SUCCESS_RATE="${STRICT_MIN_RETRY_SUCCESS_RATE:-0.1}"  # 10 percent
STRICT_MAX_RETRY_ITER="${STRICT_MAX_RETRY_ITER:-10}"                    # 10 strict retries

# Persistent retry history file (JSONL)
RETRY_HISTORY_FILE="${RETRY_HISTORY_FILE:-${OUT_DIR}/retry_history.jsonl}"
mkdir -p "$(dirname "$RETRY_HISTORY_FILE")"

log_history() {
  local iter="$1"
  local exit_code="$2"
  local current_retry_count="$3"
  local prev_retry_count="$4"
  local reason="$5"

  local ts
  ts="$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"

  printf '{"timestamp":"%s","iteration":%s,"exit_code":%s,' \
    "$ts" "$iter" "$exit_code" >> "$RETRY_HISTORY_FILE"
  printf '"current_retry_count":%s,"prev_retry_count":%s,' \
    "$current_retry_count" "$prev_retry_count" >> "$RETRY_HISTORY_FILE"
  printf '"reason":"%s"}\n' "$reason" >> "$RETRY_HISTORY_FILE"
}

PREV_RETRY_COUNT=0
ITER=1

STRICT_MODE="false"
STRICT_RETRY_COUNT=0

while :; do
  echo "[retry-wrapper] iteration ${ITER}: running crawler..."

  # Run the crawler; we expect RETRY_EXIT_CODE sometimes.
  python3 scripts/run.py "$@"
  EXIT_CODE=$?

  if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo "[retry-wrapper] run finished cleanly (exit 0); stopping."
    log_history "$ITER" "$EXIT_CODE" "0" "$PREV_RETRY_COUNT" "clean_exit"
    break
  fi

  if [[ "$EXIT_CODE" -ne "$RETRY_EXIT_CODE" ]]; then
    echo "[retry-wrapper] run exited with non-retry code ${EXIT_CODE}; stopping."
    log_history "$ITER" "$EXIT_CODE" "-1" "$PREV_RETRY_COUNT" "non_retry_exit"
    exit "$EXIT_CODE"
  fi

  RETRY_FILE="${OUT_DIR}/retry_companies.json"
  if [[ ! -f "$RETRY_FILE" ]]; then
    echo "[retry-wrapper] retry exit code but ${RETRY_FILE} not found; stopping."
    log_history "$ITER" "$EXIT_CODE" "-1" "$PREV_RETRY_COUNT" "retry_exit_missing_retry_file"
    exit 1
  fi

  # Read current retry count and all_attempted flag from JSON
  read -r CURRENT_RETRY_COUNT ALL_ATTEMPTED_FLAG <<EOF
$(python3 - "$RETRY_FILE" << 'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8"))
retry_companies = data.get("retry_companies") or []
all_attempted = bool(data.get("all_attempted"))
print(len(retry_companies), "true" if all_attempted else "false")
PYEOF
)
EOF

  echo "[retry-wrapper] ${CURRENT_RETRY_COUNT} companies need retry."
  echo "[retry-wrapper] all_attempted=${ALL_ATTEMPTED_FLAG}"

  reason="retry_exit_continue"

  # Enable strict mode once all companies in this run have been attempted at least once
  if [[ "$STRICT_MODE" == "false" && "$ALL_ATTEMPTED_FLAG" == "true" ]]; then
    STRICT_MODE="true"
    STRICT_RETRY_COUNT=0
    echo "[retry-wrapper] all companies have been attempted at least once; enabling strict retry policy (min success rate=${STRICT_MIN_RETRY_SUCCESS_RATE}, max strict retries=${STRICT_MAX_RETRY_ITER})."
  fi

  # In strict mode, once we have a previous retry count, require improvement
  if [[ "$STRICT_MODE" == "true" && "$PREV_RETRY_COUNT" -gt 0 ]]; then
    python3 - "$PREV_RETRY_COUNT" "$CURRENT_RETRY_COUNT" "$STRICT_MIN_RETRY_SUCCESS_RATE" << 'PYEOF'
import sys
prev = int(sys.argv[1])
cur = int(sys.argv[2])
thr = float(sys.argv[3])
succ = prev - cur
rate = (succ / prev) if prev else 0.0
print(f"[retry-wrapper] progress from last retry set: {succ}/{prev} ({rate:.1%})")
# Return non-zero to signal we should stop if improvement is below threshold
if cur == 0 or rate < thr:
    sys.exit(1)
PYEOF
    SHOULD_CONTINUE=$?
    if [[ "$SHOULD_CONTINUE" -ne 0 ]]; then
      if (( CURRENT_RETRY_COUNT == 0 )); then
        reason="retry_exit_stop_empty"
        echo "[retry-wrapper] no companies left to retry; stopping."
      else
        reason="retry_exit_stop_progress"
        echo "[retry-wrapper] progress below STRICT_MIN_RETRY_SUCCESS_RATE=${STRICT_MIN_RETRY_SUCCESS_RATE}; stopping."
      fi
    fi
  fi

  # In strict mode, also enforce a hard cap on number of strict retries
  if [[ "$STRICT_MODE" == "true" && "$reason" == "retry_exit_continue" ]]; then
    if (( STRICT_RETRY_COUNT >= STRICT_MAX_RETRY_ITER )); then
      reason="retry_exit_stop_max_iter"
      echo "[retry-wrapper] reached STRICT_MAX_RETRY_ITER=${STRICT_MAX_RETRY_ITER}; stopping."
    fi
  fi

  # Default behavior when not in strict mode:
  # - We do not require any improvement in retry count.
  # - We do not impose a retry limit.
  # So reason stays "retry_exit_continue" and we just loop.

  # Log this iteration's outcome
  log_history "$ITER" "$EXIT_CODE" "$CURRENT_RETRY_COUNT" "$PREV_RETRY_COUNT" "$reason"

  if [[ "$reason" != "retry_exit_continue" ]]; then
    break
  fi

  PREV_RETRY_COUNT=$CURRENT_RETRY_COUNT
  if [[ "$STRICT_MODE" == "true" ]]; then
    STRICT_RETRY_COUNT=$((STRICT_RETRY_COUNT + 1))
  fi
  ITER=$((ITER + 1))
done