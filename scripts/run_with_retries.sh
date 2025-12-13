#!/usr/bin/env bash
set -uo pipefail   # no -e (we handle non-zero exits manually)

args=("$@")

# Retry exit code: prefer RETRY_EXIT_CODE, then DEEP_CRAWL_RETRY_EXIT_CODE, then default 17
: "${RETRY_EXIT_CODE:=}"
if [[ -z "$RETRY_EXIT_CODE" && -n "${DEEP_CRAWL_RETRY_EXIT_CODE:-}" ]]; then
  RETRY_EXIT_CODE="${DEEP_CRAWL_RETRY_EXIT_CODE}"
fi
if [[ -z "$RETRY_EXIT_CODE" ]]; then
  RETRY_EXIT_CODE=17
fi
export RETRY_EXIT_CODE
export DEEP_CRAWL_RETRY_EXIT_CODE="$RETRY_EXIT_CODE"

# Derive OUT_DIR from env or --out-dir argument
if [[ -n "${OUT_DIR:-}" ]]; then
  OUT_DIR="${OUT_DIR}"
else
  OUT_DIR="outputs"
  for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--out-dir" && $((i+1)) -lt ${#args[@]} ]]; then
      OUT_DIR="${args[$i+1]}"
      break
    fi
  done
fi
export OUT_DIR

RETRY_FILE="${OUT_DIR}/retry_companies.json"
GLOBAL_STATE_FILE="${OUT_DIR}/crawl_global_state.json"

# Expected improvement + retry caps (new names, with backward-compat)
EXPECTED_MIN_IMPROVEMENT_RATE="${EXPECTED_MIN_IMPROVEMENT_RATE:-${STRICT_MIN_RETRY_SUCCESS_RATE:-0.10}}"  # 10%
MAX_RETRY_ITER="${MAX_RETRY_ITER:-${STRICT_MAX_RETRY_ITER:-10}}"

# OOM auto restart config
OOM_RESTART_LIMIT="${OOM_RESTART_LIMIT:-100}"
OOM_RESTART_DELAY="${OOM_RESTART_DELAY:-10}"
OOM_RESTART_COUNT=0

# Persistent retry history file (JSONL)
RETRY_HISTORY_FILE="${RETRY_HISTORY_FILE:-${OUT_DIR}/retry_history.jsonl}"
mkdir -p "$(dirname "$RETRY_HISTORY_FILE")"

log_history() {
  local iter="$1"
  local exit_code="$2"
  local retry_mode="$3"
  local phase="$4"
  local current_retry_count="$5"
  local prev_retry_count="$6"
  local total_companies="$7"
  local attempted_total="$8"
  local reason="$9"

  local ts
  ts="$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"

  printf '{"timestamp":"%s","iteration":%s,"exit_code":%s,' \
    "$ts" "$iter" "$exit_code" >> "$RETRY_HISTORY_FILE"
  printf '"retry_mode":"%s","phase":"%s",' \
    "$retry_mode" "$phase" >> "$RETRY_HISTORY_FILE"
  printf '"current_retry_count":%s,"prev_retry_count":%s,' \
    "$current_retry_count" "$prev_retry_count" >> "$RETRY_HISTORY_FILE"
  printf '"total_companies":%s,"attempted_total":%s,' \
    "$total_companies" "$attempted_total" >> "$RETRY_HISTORY_FILE"
  printf '"reason":"%s"}\n' "$reason" >> "$RETRY_HISTORY_FILE"
}

read_retry_info() {
  # Prints: retry_count total_companies attempted_total
  if [[ ! -f "$RETRY_FILE" ]]; then
    echo "0 0 0"
    return 0
  fi
  python3 - "$RETRY_FILE" << 'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print("0 0 0")
    raise SystemExit
rc = data.get("retry_companies") or []
total = int(data.get("total_companies") or 0)
attempted = int(data.get("attempted_total") or 0)
print(len(rc), total, attempted)
PYEOF
}

read_in_progress_count() {
  if [[ ! -f "$GLOBAL_STATE_FILE" ]]; then
    echo "0"
    return 0
  fi
  python3 - "$GLOBAL_STATE_FILE" << 'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print(0)
    raise SystemExit
lst = data.get("in_progress_companies") or []
print(len(lst))
PYEOF
}

# Detect user supplied --retry-mode once (wrapper will not override)
USER_RETRY_MODE=0
USER_RETRY_MODE_VALUE=""
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

ITER=1
PHASE="primary"  # primary -> retry
RETRY_ITER=0
PREV_RETRY_COUNT=0

while :; do
  # Decide retry-mode for this iteration
  RETRY_MODE_TO_USE=""
  if [[ "$USER_RETRY_MODE" -eq 1 ]]; then
    RETRY_MODE_TO_USE="(user:${USER_RETRY_MODE_VALUE})"
  else
    read -r retry_count total_companies attempted_total <<< "$(read_retry_info)"
    if [[ "$PHASE" == "primary" ]]; then
      # first run uses --retry-mode all ONLY when retry file missing OR retry_companies is empty
      if [[ "$retry_count" -eq 0 ]]; then
        RETRY_MODE_TO_USE="all"
      else
        RETRY_MODE_TO_USE="skip-retry"
      fi
    else
      RETRY_MODE_TO_USE="only-retry"
    fi
  fi

  echo "[retry-wrapper] iteration ${ITER} phase=${PHASE} retry_mode=${RETRY_MODE_TO_USE}"

  # Run crawler
  if [[ "$USER_RETRY_MODE" -eq 1 ]]; then
    python3 scripts/run.py "${args[@]}"
  else
    python3 scripts/run.py --retry-mode "$RETRY_MODE_TO_USE" "${args[@]}"
  fi
  EXIT_CODE=$?

  # --- Clean exit path ---------------------------------------------------
  if [[ "$EXIT_CODE" -eq 0 ]]; then
    read -r retry_count total_companies attempted_total <<< "$(read_retry_info)"
    log_history "$ITER" "$EXIT_CODE" "$RETRY_MODE_TO_USE" "$PHASE" \
      "$retry_count" "$PREV_RETRY_COUNT" "$total_companies" "$attempted_total" "clean_exit"

    # Final “in_progress” cleanup pass:
    inprog="$(read_in_progress_count)"
    if [[ "$inprog" -gt 0 ]]; then
      echo "[retry-wrapper] clean exit but crawl_global_state.json has in_progress_companies=${inprog}"
      echo "[retry-wrapper] running finalize pass: force-mark markdown completed for those companies (no LLM)."

      # IMPORTANT: we intentionally do NOT override --retry-mode here; finalize flag filters by in_progress list.
      # We do force llm-mode none as requested, and finalize-in-progress-md.
      python3 scripts/run.py "${args[@]}" --finalize-in-progress-md --llm-mode none
      FINALIZE_EXIT=$?

      if [[ "$FINALIZE_EXIT" -ne 0 ]]; then
        echo "[retry-wrapper] finalize pass failed with exit_code=${FINALIZE_EXIT}; stopping."
        log_history "$ITER" "$FINALIZE_EXIT" "finalize" "finalize" \
          "-1" "$PREV_RETRY_COUNT" "$total_companies" "$attempted_total" "finalize_failed"
        exit "$FINALIZE_EXIT"
      fi

      echo "[retry-wrapper] finalize pass completed."
    fi

    echo "[retry-wrapper] finished."
    exit 0
  fi

  # --- Non-retry exit code path (handle OOM first) ------------------------
  if [[ "$EXIT_CODE" -ne "$RETRY_EXIT_CODE" ]]; then
    # Signal-kill -> 128 + signal
    if (( EXIT_CODE >= 128 )); then
      SIGNAL=$((EXIT_CODE - 128))
      if (( SIGNAL == 9 )); then
        # Likely OOM killer or kill -9
        if (( OOM_RESTART_COUNT < OOM_RESTART_LIMIT )); then
          OOM_RESTART_COUNT=$((OOM_RESTART_COUNT + 1))
          echo "[retry-wrapper] killed by signal 9 (likely OOM); restart ${OOM_RESTART_COUNT}/${OOM_RESTART_LIMIT} after ${OOM_RESTART_DELAY}s."
          read -r retry_count total_companies attempted_total <<< "$(read_retry_info)"
          log_history "$ITER" "$EXIT_CODE" "$RETRY_MODE_TO_USE" "$PHASE" \
            "$retry_count" "$PREV_RETRY_COUNT" "$total_companies" "$attempted_total" "oom_signal_restart"
          sleep "$OOM_RESTART_DELAY"
          ITER=$((ITER + 1))
          continue
        else
          echo "[retry-wrapper] repeated OOM; reached OOM_RESTART_LIMIT=${OOM_RESTART_LIMIT}; stopping."
          read -r retry_count total_companies attempted_total <<< "$(read_retry_info)"
          log_history "$ITER" "$EXIT_CODE" "$RETRY_MODE_TO_USE" "$PHASE" \
            "$retry_count" "$PREV_RETRY_COUNT" "$total_companies" "$attempted_total" "oom_signal_stop"
          exit "$EXIT_CODE"
        fi
      fi
    fi

    echo "[retry-wrapper] run exited with non-retry code ${EXIT_CODE}; stopping."
    read -r retry_count total_companies attempted_total <<< "$(read_retry_info)"
    log_history "$ITER" "$EXIT_CODE" "$RETRY_MODE_TO_USE" "$PHASE" \
      "$retry_count" "$PREV_RETRY_COUNT" "$total_companies" "$attempted_total" "non_retry_exit"
    exit "$EXIT_CODE"
  fi

  # --- RETRY_EXIT_CODE path ----------------------------------------------
  if [[ ! -f "$RETRY_FILE" ]]; then
    echo "[retry-wrapper] retry exit code but ${RETRY_FILE} not found; stopping."
    log_history "$ITER" "$EXIT_CODE" "$RETRY_MODE_TO_USE" "$PHASE" \
      "-1" "$PREV_RETRY_COUNT" "0" "0" "retry_exit_missing_retry_file"
    exit 1
  fi

  read -r CURRENT_RETRY_COUNT TOTAL_COMPANIES ATTEMPTED_TOTAL <<< "$(read_retry_info)"
  echo "[retry-wrapper] retry_companies=${CURRENT_RETRY_COUNT} total_companies=${TOTAL_COMPANIES} attempted_total=${ATTEMPTED_TOTAL}"

  REASON="retry_exit_continue"

  if [[ "$PHASE" == "primary" ]]; then
    # Switch to retry phase once this run’s population has all been attempted
    # (ATTEMPTED_TOTAL >= TOTAL_COMPANIES) and there is something to retry.
    if (( TOTAL_COMPANIES > 0 )) && (( ATTEMPTED_TOTAL >= TOTAL_COMPANIES )) && (( CURRENT_RETRY_COUNT > 0 )); then
      echo "[retry-wrapper] attempted_total>=total_companies; switching to retry phase (only-retry)."
      PHASE="retry"
      RETRY_ITER=0
      PREV_RETRY_COUNT="$CURRENT_RETRY_COUNT"
    else
      echo "[retry-wrapper] staying in primary phase (skip-retry on subsequent runs)."
    fi

    log_history "$ITER" "$EXIT_CODE" "$RETRY_MODE_TO_USE" "$PHASE" \
      "$CURRENT_RETRY_COUNT" "$PREV_RETRY_COUNT" "$TOTAL_COMPANIES" "$ATTEMPTED_TOTAL" "$REASON"
    PREV_RETRY_COUNT="$CURRENT_RETRY_COUNT"
    ITER=$((ITER + 1))
    continue
  fi

  # PHASE=retry: enforce expected improvement + max retry iterations
  if (( PREV_RETRY_COUNT == 0 )); then
    PREV_RETRY_COUNT="$CURRENT_RETRY_COUNT"
  fi

  if (( PREV_RETRY_COUNT > 0 )); then
    python3 - "$PREV_RETRY_COUNT" "$CURRENT_RETRY_COUNT" "$EXPECTED_MIN_IMPROVEMENT_RATE" << 'PYEOF'
import sys
prev = int(sys.argv[1])
cur = int(sys.argv[2])
thr = float(sys.argv[3])
succ = prev - cur
rate = (succ / prev) if prev else 0.0
print(f"[retry-wrapper] retry improvement: {succ}/{prev} ({rate:.1%})")
if cur == 0:
    sys.exit(2)  # done
if rate < thr:
    sys.exit(1)  # insufficient progress
sys.exit(0)
PYEOF
    IMPROVE_STATUS=$?
    if [[ "$IMPROVE_STATUS" -eq 2 ]]; then
      echo "[retry-wrapper] retry set is empty; stopping."
      REASON="retry_exit_stop_empty"
    elif [[ "$IMPROVE_STATUS" -ne 0 ]]; then
      echo "[retry-wrapper] progress below EXPECTED_MIN_IMPROVEMENT_RATE=${EXPECTED_MIN_IMPROVEMENT_RATE}; stopping."
      REASON="retry_exit_stop_progress"
    fi
  fi

  if [[ "$REASON" == "retry_exit_continue" ]]; then
    if (( RETRY_ITER >= MAX_RETRY_ITER )); then
      echo "[retry-wrapper] reached MAX_RETRY_ITER=${MAX_RETRY_ITER}; stopping."
      REASON="retry_exit_stop_max_iter"
    fi
  fi

  log_history "$ITER" "$EXIT_CODE" "$RETRY_MODE_TO_USE" "$PHASE" \
    "$CURRENT_RETRY_COUNT" "$PREV_RETRY_COUNT" "$TOTAL_COMPANIES" "$ATTEMPTED_TOTAL" "$REASON"

  if [[ "$REASON" != "retry_exit_continue" ]]; then
    exit 0
  fi

  PREV_RETRY_COUNT="$CURRENT_RETRY_COUNT"
  RETRY_ITER=$((RETRY_ITER + 1))
  ITER=$((ITER + 1))
done
