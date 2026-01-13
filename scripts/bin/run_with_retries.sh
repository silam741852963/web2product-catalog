#!/usr/bin/env bash
set -uo pipefail  # no -e

ARGS=("$@")

# Retry exit code: prefer RETRY_EXIT_CODE, then DEEP_CRAWL_RETRY_EXIT_CODE, else 17
RETRY_EXIT_CODE="${RETRY_EXIT_CODE:-${DEEP_CRAWL_RETRY_EXIT_CODE:-17}}"
export RETRY_EXIT_CODE
export DEEP_CRAWL_RETRY_EXIT_CODE="$RETRY_EXIT_CODE"

# ---- allocator / fragmentation mitigations (glibc) ----
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
# export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

# Derive OUT_DIR from env or --out-dir/--output-dir argument (default: outputs)
if [[ -n "${OUT_DIR:-}" ]]; then
  OUT_DIR="${OUT_DIR}"
else
  OUT_DIR="outputs"

  for ((i=0; i<${#ARGS[@]}; i++)); do
    a="${ARGS[$i]}"

    # --out-dir VALUE / --output-dir VALUE
    if [[ "$a" == "--out-dir" || "$a" == "--output-dir" ]]; then
      if (( i + 1 < ${#ARGS[@]} )); then
        OUT_DIR="${ARGS[$((i+1))]}"
        break
      fi
    fi

    # --out-dir=VALUE / --output-dir=VALUE
    if [[ "$a" == --out-dir=* ]]; then
      OUT_DIR="${a#--out-dir=}"
      break
    fi
    if [[ "$a" == --output-dir=* ]]; then
      OUT_DIR="${a#--output-dir=}"
      break
    fi
  done
fi

# Resolve OUT_DIR to absolute path (matches run.py behavior)
OUT_DIR="$(python3 - <<'PYEOF' "$OUT_DIR"
import pathlib, sys
p = pathlib.Path(sys.argv[1]).expanduser().resolve()
print(str(p))
PYEOF
)"

GLOBAL_STATE_FILE="${OUT_DIR}/crawl_global_state.json"

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

# Ensure we're running from repo root (so relative paths and outputs match expectations).
# scripts/bin/run_with_retries.sh -> repo root is two levels up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ITER=1
while :; do
  echo "[retry-wrapper] iteration=${ITER} out_dir=${OUT_DIR} retry_exit_code=${RETRY_EXIT_CODE}"

  # Run as a module so it works after `pip install -e .`
  # shellcheck disable=SC2068
  python3 -m cli.crawl_extract.run ${ARGS[@]}
  EXIT_CODE=$?

  if [[ "$EXIT_CODE" -eq 0 ]]; then
    inprog="$(read_in_progress_count)"
    if [[ "$inprog" -gt 0 ]]; then
      echo "[retry-wrapper] clean exit but in_progress_companies=${inprog}; running finalize pass (markdown only)."
      # shellcheck disable=SC2068
      python3 -m cli.crawl_extract.run ${ARGS[@]} --finalize-in-progress-md --llm-mode none
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

  if [[ "$EXIT_CODE" -eq "$RETRY_EXIT_CODE" ]]; then
    echo "[retry-wrapper] retry exit_code=${EXIT_CODE}; restarting."
    ITER=$((ITER + 1))
    continue
  fi

  echo "[retry-wrapper] run exited with code=${EXIT_CODE}; stopping."
  exit "$EXIT_CODE"
done
