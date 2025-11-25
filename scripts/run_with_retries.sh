#!/usr/bin/env bash
set -uo pipefail   # no -e because we handle non-zero exits manually

# Configurable via env vars
RETRY_EXIT_CODE="${RETRY_EXIT_CODE:-17}"
MAX_RETRY_ITER="${MAX_RETRY_ITER:-10}"                    # max number of runs
MIN_RETRY_SUCCESS_RATE="${MIN_RETRY_SUCCESS_RATE:-0.3}"   # 30%
OUT_DIR="${OUT_DIR:-outputs}"                             # must match --out-dir in run.py

PREV_RETRY_COUNT=0
ITER=1

while :; do
  echo "[retry-wrapper] iteration ${ITER}: running crawler..."

  # Run the crawler; we *expect* non-zero (RETRY_EXIT_CODE) sometimes.
  python scripts/run.py "$@"
  EXIT_CODE=$?

  if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo "[retry-wrapper] run finished cleanly (exit 0); stopping."
    break
  fi

  if [[ "$EXIT_CODE" -ne "$RETRY_EXIT_CODE" ]]; then
    echo "[retry-wrapper] run exited with non-retry code ${EXIT_CODE}; stopping."
    exit "$EXIT_CODE"
  fi

  RETRY_FILE="${OUT_DIR}/retry_companies.json"
  if [[ ! -f "$RETRY_FILE" ]]; then
    echo "[retry-wrapper] retry exit code but ${RETRY_FILE} not found; stopping."
    exit 1
  fi

  # --- Read current retry count from JSON (as argv[1]) ---
  CURRENT_RETRY_COUNT=$(
    python - "$RETRY_FILE" << 'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8"))
print(len(data.get("retry_companies", [])))
EOF
  )

  echo "[retry-wrapper] ${CURRENT_RETRY_COUNT} companies need retry."

  if (( PREV_RETRY_COUNT > 0 )); then
    # --- Check progress between previous and current retry sets ---
    python - "$PREV_RETRY_COUNT" "$CURRENT_RETRY_COUNT" "$MIN_RETRY_SUCCESS_RATE" << 'EOF'
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
      echo "[retry-wrapper] progress below MIN_RETRY_SUCCESS_RATE=${MIN_RETRY_SUCCESS_RATE} or no companies left; stopping."
      break
    fi
  fi

  if (( ITER >= MAX_RETRY_ITER )); then
    echo "[retry-wrapper] reached MAX_RETRY_ITER=${MAX_RETRY_ITER}; stopping."
    break
  fi

  PREV_RETRY_COUNT=$CURRENT_RETRY_COUNT
  ITER=$((ITER+1))
done