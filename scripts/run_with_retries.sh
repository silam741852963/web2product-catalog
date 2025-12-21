#!/usr/bin/env bash
set -uo pipefail  # no -e

ARGS=("$@")

# Retry exit code: prefer RETRY_EXIT_CODE, then DEEP_CRAWL_RETRY_EXIT_CODE, else 17
RETRY_EXIT_CODE="${RETRY_EXIT_CODE:-${DEEP_CRAWL_RETRY_EXIT_CODE:-17}}"
export RETRY_EXIT_CODE
export DEEP_CRAWL_RETRY_EXIT_CODE="$RETRY_EXIT_CODE"

# Derive OUT_DIR from env or --out-dir argument (default: outputs)
if [[ -n "${OUT_DIR:-}" ]]; then
  OUT_DIR="${OUT_DIR}"
else
  OUT_DIR="outputs"
  for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--out-dir" && $((i+1)) -lt ${#ARGS[@]} ]]; then
      OUT_DIR="${ARGS[$i+1]}"
      break
    fi
  done
fi

GLOBAL_STATE_FILE="${OUT_DIR}/crawl_global_state.json"

# OOM auto restart config (SIGKILL -> exit 137)
OOM_RESTART_LIMIT="${OOM_RESTART_LIMIT:-100}"
OOM_RESTART_DELAY="${OOM_RESTART_DELAY:-10}"
OOM_RESTART_COUNT=0

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

ITER=1
while :; do
  echo "[retry-wrapper] iteration=${ITER} out_dir=${OUT_DIR}"
  python3 scripts/run.py "${ARGS[@]}"
  EXIT_CODE=$?

  # Clean exit
  if [[ "$EXIT_CODE" -eq 0 ]]; then
    inprog="$(read_in_progress_count)"
    if [[ "$inprog" -gt 0 ]]; then
      echo "[retry-wrapper] clean exit but in_progress_companies=${inprog}; running finalize pass (markdown only)."
      python3 scripts/run.py "${ARGS[@]}" --finalize-in-progress-md --llm-mode none
      FINALIZE_EXIT=$?
      if [[ "$FINALIZE_EXIT" -ne 0 ]]; then
        echo "[retry-wrapper] finalize pass failed exit_code=${FINALIZE_EXIT}; stopping."
        exit "$FINALIZE_EXIT"
      fi
      echo "[retry-wrapper] finalize pass completed."
    fi

    echo "[retry-wrapper] finished."
    exit 0
  fi

  # OOM / SIGKILL handling (exit 128+9 = 137)
  if (( EXIT_CODE >= 128 )); then
    SIGNAL=$((EXIT_CODE - 128))
    if (( SIGNAL == 9 )); then
      if (( OOM_RESTART_COUNT < OOM_RESTART_LIMIT )); then
        OOM_RESTART_COUNT=$((OOM_RESTART_COUNT + 1))
        echo "[retry-wrapper] killed by signal 9 (likely OOM); restart ${OOM_RESTART_COUNT}/${OOM_RESTART_LIMIT} after ${OOM_RESTART_DELAY}s."
        sleep "$OOM_RESTART_DELAY"
        ITER=$((ITER + 1))
        continue
      else
        echo "[retry-wrapper] repeated OOM; reached OOM_RESTART_LIMIT=${OOM_RESTART_LIMIT}; stopping."
        exit "$EXIT_CODE"
      fi
    fi
  fi

  # Retry requested by scheduler (eligible retry-pending exists *now*)
  if [[ "$EXIT_CODE" -eq "$RETRY_EXIT_CODE" ]]; then
    echo "[retry-wrapper] retry exit_code=${EXIT_CODE}; restarting."
    ITER=$((ITER + 1))
    continue
  fi

  # Anything else: stop
  echo "[retry-wrapper] run exited with code=${EXIT_CODE}; stopping."
  exit "$EXIT_CODE"
done
