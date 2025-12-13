#!/usr/bin/env bash
set -uo pipefail   # no -e because we handle non-zero exits manually

args=("$@")

# Retry exit code: prefer RETRY_EXIT_CODE, then DEEP_CRAWL_RETRY_EXIT_CODE, else 17
: "${RETRY_EXIT_CODE:=}"
if [[ -z "$RETRY_EXIT_CODE" && -n "${DEEP_CRAWL_RETRY_EXIT_CODE:-}" ]]; then
  RETRY_EXIT_CODE="${DEEP_CRAWL_RETRY_EXIT_CODE}"
fi
if [[ -z "$RETRY_EXIT_CODE" ]]; then
  RETRY_EXIT_CODE=17
fi
export RETRY_EXIT_CODE
export DEEP_CRAWL_RETRY_EXIT_CODE="$RETRY_EXIT_CODE"

# OUT_DIR from env or --out-dir
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

# Strict retry controls (retry phase only)
STRICT_MIN_RETRY_SUCCESS_RATE="${STRICT_MIN_RETRY_SUCCESS_RATE:-0.1}"  # 10%
STRICT_MAX_RETRY_ITER="${STRICT_MAX_RETRY_ITER:-10}"

# OOM auto restart controls
OOM_RESTART_LIMIT="${OOM_RESTART_LIMIT:-100}"
OOM_RESTART_DELAY="${OOM_RESTART_DELAY:-10}"
OOM_RESTART_COUNT=0

# Persistent history file (JSONL)
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

# Detect user supplied --retry-mode once; if present, wrapper does NOT override it.
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

# Helper: read retry_companies count + totals
read_retry_fields() {
  # outputs: retry_count total_companies attempted_total all_attempted_flag
  python3 - "$RETRY_FILE" << 'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print("0 0 0 false")
    raise SystemExit(0)

retry_companies = data.get("retry_companies") or []
total_companies = int(data.get("total_companies") or 0)
attempted_total = int(data.get("attempted_total") or 0)
all_attempted = bool(data.get("all_attempted") or False)
print(len(retry_companies), total_companies, attempted_total, "true" if all_attempted else "false")
PYEOF
}

PREV_RETRY_COUNT=0
ITER=1
PHASE="primary"       # primary => skip-retry (after first iteration); retry => only-retry
STRICT_RETRY_COUNT=0

while :; do
  echo "[retry-wrapper] iteration ${ITER} (phase=${PHASE})"

  # Decide retry-mode for this iteration (only if wrapper controls it)
  if [[ "$USER_RETRY_MODE" -eq 0 ]]; then
    RETRY_COMPANY_MODE="skip-retry"

    if [[ "$PHASE" == "retry" ]]; then
      RETRY_COMPANY_MODE="only-retry"
    else
      # PHASE == primary
      if [[ "$ITER" -eq 1 ]]; then
        # First run: use "all" only if retry file missing OR retry_companies empty
        if [[ ! -f "$RETRY_FILE" ]]; then
          RETRY_COMPANY_MODE="all"
        else
          read -r _rc _tc _at _aa <<<"$(read_retry_fields)"
          if [[ "$_rc" -eq 0 ]]; then
            RETRY_COMPANY_MODE="all"
          else
            RETRY_COMPANY_MODE="skip-retry"
          fi
        fi
      else
        # Later runs: always skip-retry in primary phase
        RETRY_COMPANY_MODE="skip-retry"
      fi
    fi

    echo "[retry-wrapper] RETRY_COMPANY_MODE=${RETRY_COMPANY_MODE}"
  else
    echo "[retry-wrapper] using user-specified --retry-mode=${USER_RETRY_MODE_VALUE}"
  fi

  # Run
  if [[ "$USER_RETRY_MODE" -eq 1 ]]; then
    python3 scripts/run.py "${args[@]}"
  else
    python3 scripts/run.py --retry-mode "$RETRY_COMPANY_MODE" "${args[@]}"
  fi
  EXIT_CODE=$?

  # ---------------- Clean exit ----------------
  if [[ "$EXIT_CODE" -eq 0 ]]; then
    CURRENT_RETRY_COUNT=0
    TOTAL_COMPANIES=0
    ATTEMPTED_TOTAL=0
    ALL_ATTEMPTED_FLAG="false"

    if [[ -f "$RETRY_FILE" ]]; then
      read -r CURRENT_RETRY_COUNT TOTAL_COMPANIES ATTEMPTED_TOTAL ALL_ATTEMPTED_FLAG <<<"$(read_retry_fields)"
    fi

    log_history "$ITER" "$EXIT_CODE" "0" "$PREV_RETRY_COUNT" "clean_exit"

    # Defensive: if exit=0 but retry list exists, enter retry phase once.
    if [[ "$PHASE" == "primary" && "$CURRENT_RETRY_COUNT" -gt 0 ]]; then
      echo "[retry-wrapper] run exited cleanly but retry_companies.json has ${CURRENT_RETRY_COUNT}; switching to retry phase."
      PHASE="retry"
      PREV_RETRY_COUNT="$CURRENT_RETRY_COUNT"
      ITER=$((ITER + 1))
      continue
    fi

    echo "[retry-wrapper] run finished cleanly; stopping."
    break
  fi

  # ---------------- Non-retry exit (handle OOM signal 9 / 137 first) ----------------
  if [[ "$EXIT_CODE" -ne "$RETRY_EXIT_CODE" ]]; then
    if (( EXIT_CODE >= 128 )); then
      SIGNAL=$((EXIT_CODE - 128))
      if (( SIGNAL == 9 )); then
        if (( OOM_RESTART_COUNT < OOM_RESTART_LIMIT )); then
          OOM_RESTART_COUNT=$((OOM_RESTART_COUNT + 1))
          echo "[retry-wrapper] terminated by SIGKILL (likely OOM); restart ${OOM_RESTART_COUNT}/${OOM_RESTART_LIMIT} after ${OOM_RESTART_DELAY}s."
          log_history "$ITER" "$EXIT_CODE" "-1" "$PREV_RETRY_COUNT" "oom_signal_restart"
          sleep "$OOM_RESTART_DELAY"
          ITER=$((ITER + 1))
          continue
        else
          echo "[retry-wrapper] reached OOM_RESTART_LIMIT=${OOM_RESTART_LIMIT}; stopping."
          log_history "$ITER" "$EXIT_CODE" "-1" "$PREV_RETRY_COUNT" "oom_signal_stop"
          exit "$EXIT_CODE"
        fi
      fi
    fi

    echo "[retry-wrapper] run exited with non-retry code ${EXIT_CODE}; stopping."
    log_history "$ITER" "$EXIT_CODE" "-1" "$PREV_RETRY_COUNT" "non_retry_exit"
    exit "$EXIT_CODE"
  fi

  # ---------------- RETRY_EXIT_CODE path ----------------
  if [[ ! -f "$RETRY_FILE" ]]; then
    echo "[retry-wrapper] retry exit code but ${RETRY_FILE} not found; stopping."
    log_history "$ITER" "$EXIT_CODE" "-1" "$PREV_RETRY_COUNT" "retry_exit_missing_retry_file"
    exit 1
  fi

  read -r CURRENT_RETRY_COUNT TOTAL_COMPANIES ATTEMPTED_TOTAL ALL_ATTEMPTED_FLAG <<<"$(read_retry_fields)"
  echo "[retry-wrapper] retry_count=${CURRENT_RETRY_COUNT} total_companies=${TOTAL_COMPANIES} attempted_total=${ATTEMPTED_TOTAL} all_attempted=${ALL_ATTEMPTED_FLAG}"

  reason="retry_exit_continue"

  # ---- Primary phase: keep doing skip-retry until everything attempted at least once ----
  if [[ "$PHASE" == "primary" ]]; then
    # Switch condition: attempted_total >= total_companies (or all_attempted true)
    if [[ "$ALL_ATTEMPTED_FLAG" == "true" ]] || (( ATTEMPTED_TOTAL >= TOTAL_COMPANIES && TOTAL_COMPANIES > 0 )); then
      echo "[retry-wrapper] all primary companies attempted at least once; switching to retry phase (only-retry)."
      PHASE="retry"
      STRICT_RETRY_COUNT=0
      PREV_RETRY_COUNT="$CURRENT_RETRY_COUNT"
    else
      echo "[retry-wrapper] primary not finished yet (${ATTEMPTED_TOTAL}/${TOTAL_COMPANIES}); staying in primary phase."
      PREV_RETRY_COUNT="$CURRENT_RETRY_COUNT"
    fi

    log_history "$ITER" "$EXIT_CODE" "$CURRENT_RETRY_COUNT" "$PREV_RETRY_COUNT" "$reason"
    ITER=$((ITER + 1))
    continue
  fi

  # ---- Retry phase: enforce improvement + max iterations ----
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
print(f"[retry-wrapper] improvement: {succ}/{prev} ({rate:.1%})")
# exit 1 => stop
if cur == 0:
    sys.exit(1)
if rate < thr:
    sys.exit(1)
PYEOF
    SHOULD_CONTINUE=$?
    if [[ "$SHOULD_CONTINUE" -ne 0 ]]; then
      if (( CURRENT_RETRY_COUNT == 0 )); then
        reason="retry_exit_stop_empty"
        echo "[retry-wrapper] no companies left to retry; stopping."
      else
        reason="retry_exit_stop_progress"
        echo "[retry-wrapper] improvement below STRICT_MIN_RETRY_SUCCESS_RATE=${STRICT_MIN_RETRY_SUCCESS_RATE}; stopping."
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
