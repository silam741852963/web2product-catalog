#!/usr/bin/env bash
set -uo pipefail   # no -e because we handle non-zero exits manually

# Configurable via env vars
RETRY_EXIT_CODE="${RETRY_EXIT_CODE:-17}"
MAX_RETRY_ITER="${MAX_RETRY_ITER:-10}"                    # max number of runs
MIN_RETRY_SUCCESS_RATE="${MIN_RETRY_SUCCESS_RATE:-0.3}"   # 30%
OUT_DIR="${OUT_DIR:-outputs}"                             # must match --out-dir in run.py

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

while :; do
  echo "[retry-wrapper] iteration ${ITER}: running crawler..."

  # Run the crawler; we *expect* non-zero (RETRY_EXIT_CODE) sometimes.
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

  # --- Read current retry count from JSON (as argv[1]) ---
  CURRENT_RETRY_COUNT=$(
    python3 - "$RETRY_FILE" << 'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8"))
print(len(data.get("retry_companies", [])))
EOF
  )

  echo "[retry-wrapper] ${CURRENT_RETRY_COUNT} companies need retry."

  reason="retry_exit_continue"

  if (( PREV_RETRY_COUNT > 0 )); then
    # --- Check progress between previous and current retry sets ---
    python3 - "$PREV_RETRY_COUNT" "$CURRENT_RETRY_COUNT" "$MIN_RETRY_SUCCESS_RATE" << 'EOF'
import sys
prev = int(sys.argv[1])
cur = int(sys.argv[2])
thr = float(sys.argv[3])
succ = prev - cur
rate = (succ/prev) if prev else 0.0
print(f"[retry-wrapper] progress from last retry set: {succ}/{prev} ({rate:.1%})")
if cur == 0 or rate < thr:
    sys.exit(1)  # stop auto-retries
EOF
    SHOULD_CONTINUE=$?
    if [[ "$SHOULD_CONTINUE" -ne 0 ]]; then
      # Differentiate "no companies left" vs "rate below threshold" via CURRENT_RETRY_COUNT
      if (( CURRENT_RETRY_COUNT == 0 )); then
        reason="retry_exit_stop_empty"
      else
        reason="retry_exit_stop_progress"
      fi
      echo "[retry-wrapper] progress below MIN_RETRY_SUCCESS_RATE=${MIN_RETRY_SUCCESS_RATE} or no companies left; stopping."
    fi
  fi

  if [[ "$reason" == "retry_exit_continue" ]] && (( ITER >= MAX_RETRY_ITER )); then
    echo "[retry-wrapper] reached MAX_RETRY_ITER=${MAX_RETRY_ITER}; stopping."
    reason="retry_exit_stop_max_iter"
  fi

  # Log this iteration's outcome
  log_history "$ITER" "$EXIT_CODE" "$CURRENT_RETRY_COUNT" "$PREV_RETRY_COUNT" "$reason"

  if [[ "$reason" != "retry_exit_continue" ]]; then
    break
  fi

  PREV_RETRY_COUNT=$CURRENT_RETRY_COUNT
  ITER=$((ITER+1))
done