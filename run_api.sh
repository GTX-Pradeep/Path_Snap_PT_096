#!/usr/bin/env bash
# Laptop 1: start the PathSnap API on port 8000.
cd "$(dirname "$0")/api"
export PATHSNAP_EVAL_DIR="${PATHSNAP_EVAL_DIR:-$(cd .. && pwd)/data/evaluator_test}"
echo "Evaluation trips: $PATHSNAP_EVAL_DIR"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
