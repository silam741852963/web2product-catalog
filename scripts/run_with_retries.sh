#!/usr/bin/env bash
set -uo pipefail   # no -e because we handle non-zero exits manually

# Configurable via env vars
RETRY_EXIT_CODE="${RETRY_EXIT_CODE:-17}"
OUT_DIR="${OUT_DIR:-outputs}"                           # must match --out-dir in run.py

# Strict mode config used in retry phase only
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

RETRY_FILE="${OUT_DIR}/retry_companies.json"

PREV_RETRY_COUNT=0
ITER=1

# Phase:
# - primary: focus on non retry companies, skipping known retry ids
# - retry: focus on retry list only, with strict limits
PHASE="primary"
STRICT_RETRY_COUNT=0

# Detect user supplied --retry-mode once
USER_RETRY_MODE=0
USER_RETRY_MODE_VALUE=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--retry-mode" ]]; then
    USER_RETRY_MODE=1
    if (( i + 1 < ${#args[@]} )); then
      USER_RETRY_MODE_VALUE="${args[$i+1]}"
    fi
    break
  fi
done
if [[ "$USER_RETRY_MODE" -eq 1 ]]; then
  echo "[retry-wrapper] user-specified --retry-mode=${USER_RETRY_MODE_VALUE} (wrapper will not override retry-mode)."
fi

while :; do
  echo "[retry-wrapper] iteration ${ITER} (phase=${PHASE})"

  # Decide how run.py should treat existing retry_companies.json
  if [[ "$USER_RETRY_MODE" -eq 0 ]]; then
    RETRY_COMPANY_MODE="all"
    if [[ "$PHASE" == "primary" ]]; then
      # After the first run, if we already have a retry file, skip those
      # companies so we move forward on the rest of the dataset.
      if [[ -f "$RETRY_FILE" ]]; then
        RETRY_COMPANY_MODE="skip-retry"
      fi
    else
      # In retry phase we only work on the retry list
      RETRY_COMPANY_MODE="only-retry"
    fi
    echo "[retry-wrapper] RETRY_COMPANY_MODE=${RETRY_COMPANY_MODE}"
  else
    echo "[retry-wrapper] using user-specified --retry-mode=${USER_RETRY_MODE_VALUE}"
  fi

  if [[ "$USER_RETRY_MODE" -eq 1 ]]; then
    python3 scripts/run.py "$@"
  else
    python3 scripts/run.py --retry-mode "$RETRY_COMPANY_MODE" "$@"
  fi
  EXIT_CODE=$?

  # --- Clean exit path ---------------------------------------------------
  if [[ "$EXIT_CODE" -eq 0 ]]; then
    # No new retry_ids were produced by this run.
    # We still check if an older retry file has pending companies.
    CURRENT_RETRY_COUNT=0
    if [[ -f "$RETRY_FILE" ]]; then
      CURRENT_RETRY_COUNT="$(python3 - "$RETRY_FILE" << 'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit
retry_companies = data.get("retry_companies") or []
print(len(retry_companies))
PYEOF
)"
    fi

    log_history "$ITER" "$EXIT_CODE" "0" "$PREV_RETRY_COUNT" "clean_exit"

    if [[ "$PHASE" == "primary" && "$CURRENT_RETRY_COUNT" -gt 0 ]]; then
      echo "[retry-wrapper] primary phase finished (no more non retry companies to process in this file);"
      echo "[retry-wrapper] switching to retry phase with ${CURRENT_RETRY_COUNT} stalled or timeout companies."
      PHASE="retry"
      PREV_RETRY_COUNT="$CURRENT_RETRY_COUNT"
      ITER=$((ITER + 1))
      continue
    fi

    if [[ "$PHASE" == "retry" && "$CURRENT_RETRY_COUNT" -gt 0 ]]; then
      # This should not normally happen since run.py exits with RETRY_EXIT_CODE
      # when retry_companies is non-empty, but be defensive.
      echo "[retry-wrapper] run exited with 0 but retry_companies.json still lists ${CURRENT_RETRY_COUNT} companies; stopping anyway."
    else
      echo "[retry-wrapper] run finished cleanly and no pending retry companies remain; stopping."
    fi
    break
  fi

  # --- Non retry exit code path -----------------------------------------
  if [[ "$EXIT_CODE" -ne "$RETRY_EXIT_CODE" ]]; then
    echo "[retry-wrapper] run exited with non retry code ${EXIT_CODE}; stopping."
    log_history "$ITER" "$EXIT_CODE" "-1" "$PREV_RETRY_COUNT" "non_retry_exit"
    exit "$EXIT_CODE"
  fi

  # --- RETRY_EXIT_CODE path ---------------------------------------------
  if [[ ! -f "$RETRY_FILE" ]]; then
    echo "[retry-wrapper] retry exit code but ${RETRY_FILE} not found; stopping."
    log_history "$ITER" "$EXIT_CODE" "-1" "$PREV_RETRY_COUNT" "retry_exit_missing_retry_file"
    exit 1
  fi

  # Read current retry count and bookkeeping fields from JSON
  read -r CURRENT_RETRY_COUNT ALL_ATTEMPTED_FLAG TOTAL_COMPANIES ATTEMPTED_TOTAL <<EOF
$(python3 - "$RETRY_FILE" << 'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8"))
retry_companies = data.get("retry_companies") or []
all_attempted = bool(data.get("all_attempted"))
total_companies = int(data.get("total_companies") or 0)
attempted_total = int(data.get("attempted_total") or 0)
print(len(retry_companies), "true" if all_attempted else "false", total_companies, attempted_total)
PYEOF
)
EOF

  echo "[retry-wrapper] ${CURRENT_RETRY_COUNT} companies need retry."
  echo "[retry-wrapper] all_attempted=${ALL_ATTEMPTED_FLAG} total_companies=${TOTAL_COMPANIES} attempted_total=${ATTEMPTED_TOTAL}"

  reason="retry_exit_continue"

  # ---------------- PHASE: primary ----------------
  if [[ "$PHASE" == "primary" ]]; then
    # In primary phase we want to ensure that all non retry companies in this
    # file are at least attempted once before we focus only on the retry list.

    if (( TOTAL_COMPANIES == 0 )); then
      echo "[retry-wrapper] primary phase run had no companies to process in this file; switching to retry phase."
      PHASE="retry"
    else
      if (( ATTEMPTED_TOTAL < TOTAL_COMPANIES )); then
        echo "[retry-wrapper] run did not attempt all primary companies (${ATTEMPTED_TOTAL}/${TOTAL_COMPANIES}); staying in primary phase."
      else
        echo "[retry-wrapper] all primary companies for this run were attempted at least once; switching to retry phase."
        PHASE="retry"
      fi
    fi

    log_history "$ITER" "$EXIT_CODE" "$CURRENT_RETRY_COUNT" "$PREV_RETRY_COUNT" "$reason"
    PREV_RETRY_COUNT="$CURRENT_RETRY_COUNT"
    ITER=$((ITER + 1))
    continue
  fi

  # ---------------- PHASE: retry (strict) ----------------

  # First retry phase entry: initialize baseline if needed
  if (( PREV_RETRY_COUNT == 0 )); then
    PREV_RETRY_COUNT="$CURRENT_RETRY_COUNT"
  fi

  if (( PREV_RETRY_COUNT > 0 )); then
    python3 - "$PREV_RETRY_COUNT" "$CURRENT_RETRY_COUNT" "$STRICT_MIN_RETRY_SUCCESS_RATE" << 'PYEOF'
import sys
prev = int(sys.argv[1])
cur = int(sys.argv[2])
thr = float(sys.argv[3])
succ = prev - cur
rate = (succ / prev) if prev else 0.0
print(f"[retry-wrapper] progress from last retry set: {succ}/{prev} ({rate:.1%})")
# Return non zero to signal we should stop if improvement is below threshold
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

  if [[ "$reason" == "retry_exit_continue" ]]; then
    if (( STRICT_RETRY_COUNT >= STRICT_MAX_RETRY_ITER )); then
      reason="retry_exit_stop_max_iter"
      echo "[retry-wrapper] reached STRICT_MAX_RETRY_ITER=${STRICT_MAX_RETRY_ITER}; stopping."
    fi
  fi

  log_history "$ITER" "$EXIT_CODE" "$CURRENT_RETRY_COUNT" "$PREV_RETRY_COUNT" "$reason"

  if [[ "$reason" != "retry_exit_continue" ]]; then
    break
  fi

  PREV_RETRY_COUNT="$CURRENT_RETRY_COUNT"
  STRICT_RETRY_COUNT=$((STRICT_RETRY_COUNT + 1))
  ITER=$((ITER + 1))
done