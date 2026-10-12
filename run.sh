#!/bin/zsh
# Retake API — local launcher.
#   ./run.sh                 → http://0.0.0.0:8080  (PORT env overrides)
#   RELOAD=1 ./run.sh        → autoreload for development
#   RETAKE_API_KEYS=k1,k2    → key mode (unset = open/dev mode)
#   RETAKE_DATA=/path        → jobs.db + jobs/<id>/ location (default ./var)
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "run.sh: .venv not found — create it first:  /usr/bin/python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 2
fi
if ! "$PY" -c "import uvicorn, fastapi" 2>/dev/null; then
  echo "run.sh: fastapi/uvicorn missing in .venv — run: .venv/bin/pip install -r requirements.txt" >&2
  exit 2
fi

ARGS=(api.main:app --host 0.0.0.0 --port "${PORT:-8080}" --log-level "${UVICORN_LOG_LEVEL:-warning}")
if [[ "${RELOAD:-0}" == "1" ]]; then
  ARGS+=(--reload --reload-dir api --reload-dir core)
fi
export HF_HUB_DISABLE_PROGRESS_BARS=1
exec "$PY" -m uvicorn "${ARGS[@]}"
