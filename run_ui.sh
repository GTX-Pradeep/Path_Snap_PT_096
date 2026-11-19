#!/usr/bin/env bash
# Laptop 2: serve the evaluator UI on http://localhost:8080
cd "$(dirname "$0")/ui"
python -m http.server 8080
