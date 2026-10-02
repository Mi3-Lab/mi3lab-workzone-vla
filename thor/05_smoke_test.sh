#!/usr/bin/env bash
# Runs the smoke test with the project venv.  --quick: 20 s of regression.
source "$(dirname "$0")/config.sh"
PY="$VENV/bin/python"; [[ -x "$PY" ]] || PY=python3
exec "$PY" "$THOR_DIR/05_smoke_test.py" "$@"
