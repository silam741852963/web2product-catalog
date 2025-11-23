#!/usr/bin/env bash
set -euo pipefail

PYTHON=python3
ARGS="scripts/run.py --company-file data/test/test_16.csv --log-level DEBUG --company-concurrency 16 --enable-resource-monitor --max-pages 5"

while true; do
  echo "[wrapper] starting run.py at $(date -Iseconds)"
  $PYTHON $ARGS
  code=$?
  echo "[wrapper] run.py exited with code $code"

  if [ "$code" -eq 0 ]; then
    echo "[wrapper] Normal exit (0). Not restarting."
    break
  fi

  if [ "$code" -eq 3 ]; then
    echo "[wrapper] Stall (3) detected with healthy connectivity. Restarting in 10s..."
    sleep 10
    continue
  fi

  # Either restart on all non-zero codes or stop – choose what you like:
  echo "[wrapper] Non-zero exit ($code). Restarting in 10s..."
  sleep 10
done