"""Retake API — hosted pronunciation QA for AI voiceover (FastAPI service).

Package layout (SPEC "HTTP API"):
  main.py        app factory, routers, CORS, error envelope, static ui/ mount
  settings.py    env-driven configuration (never holds secrets in logs)
  auth.py        X-Api-Key dependency (RETAKE_API_KEYS; open mode when unset)
  engine.py      thin, lazy bridge to the `core` package (503 when missing)
  jobs.py        SQLite job/fix store + single worker thread + queue
  schemas.py     pydantic models = the API contract
  routes_*.py    health, voices, lexicon, check_text, jobs, fix
"""

__version__ = "1.0.0"
