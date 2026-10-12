"""Retake API — app factory.

    uvicorn api.main:app --host 0.0.0.0 --port 8080

Wires: structured logging (no secrets), CORS from `RETAKE_CORS_ORIGINS`, the
`{error: {code, message}}` envelope for every failure, the /v1 routers, the
SQLite job store + worker thread (lifespan), Swagger (/docs) + ReDoc
(/redoc) with tags, and the static `ui/` SPA at `/` when the folder exists.
"""
import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import engine, settings
from .jobs import JobStore, Worker
from .routes_check_text import router as check_text_router
from .routes_fix import router as fix_router
from .routes_health import router as health_router
from .routes_jobs import router as jobs_router
from .routes_lexicon import router as lexicon_router

# ---------------------------------------------------------------------------
# logging: one JSON object per line, secrets scrubbed, no headers ever logged
# ---------------------------------------------------------------------------

_SECRET_RE = re.compile(r"(?i)(x-api-key|api[_-]?key|authorization|token)\s*[:=]\s*[^\s,;\"']+")
_RESERVED = set(logging.LogRecord(None, 0, "", 0, "", (), None).__dict__.keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
                   "level": record.levelname, "logger": record.name,
                   "msg": _SECRET_RE.sub(r"\1=[redacted]", record.getMessage())}
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)[-1500:]
        return _SECRET_RE.sub(r"\1=[redacted]", json.dumps(payload, ensure_ascii=False, default=str))


def setup_logging():
    root = logging.getLogger()
    if any(getattr(h, "_retake", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler._retake = True
    root.handlers = [handler]
    root.setLevel(settings.log_level())
    for noisy in ("uvicorn.access", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").propagate = True


log = logging.getLogger("retake")

TAGS = [
    {"name": "health", "description": "Liveness, auth mode, model warm state."},
    {"name": "jobs", "description": "Upload media (or render a script), poll progress, fetch results and files."},
    {"name": "fix", "description": "Re-render flagged words through WellSaid and verify the takes."},
    {"name": "check-text", "description": "Script pre-flight: heteronyms + intended sense, no audio."},
    {"name": "voices", "description": "WellSaid voice catalogue."},
    {"name": "lexicon", "description": "The heteronym lexicon Retake listens for."},
]

DESCRIPTION = """
**Retake** listens to AI voiceover and catches mispronounced heteronyms — *record* the *record*,
*live* from *live* — then fixes them through WellSaid.

Flow: `POST /v1/jobs` (media upload, or `{text, speaker_id, model}` to render first) → poll
`GET /v1/jobs/{id}` until `done` → read `result.occurrences` (flagged ones carry evidence + clips) →
`POST /v1/jobs/{id}/fix` → poll `GET /v1/jobs/{id}/fixes/{fix_id}` for verified takes.

Auth: send `X-Api-Key` when `GET /v1/health` reports `auth: "key"`. Errors always look like
`{"error": {"code", "message"}}`.
"""


@asynccontextmanager
async def lifespan(app):
    setup_logging()
    store = JobStore()
    worker = Worker(store)
    app.state.store, app.state.worker = store, worker
    worker.start()
    log.info("retake api up", extra={"event": "startup", "version": settings.VERSION, **settings.summary()})
    try:
        yield
    finally:
        worker.stop()
        store.close()
        log.info("retake api down", extra={"event": "shutdown"})


def _error_response(status, code, message, details=None):
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


_STATUS_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
                 405: "method_not_allowed", 409: "conflict", 413: "payload_too_large",
                 415: "unsupported_media_type", 422: "validation_error", 429: "rate_limited",
                 500: "internal_error", 502: "upstream_error", 503: "unavailable"}


def create_app():
    # type: () -> FastAPI
    app = FastAPI(title="Retake API", version=settings.VERSION, description=DESCRIPTION,
                  openapi_tags=TAGS, lifespan=lifespan, docs_url="/docs", redoc_url="/redoc",
                  contact={"name": "WellSaid Labs — Retake"}, swagger_ui_parameters={"persistAuthorization": True})

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins(), allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Request-Id"])

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        t0 = time.time()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled", extra={"event": "request_error", "rid": rid, "method": request.method,
                                              "path": request.url.path})
            response = _error_response(500, "internal_error", "Unexpected server error (ref %s)." % rid)
        response.headers["X-Request-Id"] = rid
        if not request.url.path.startswith("/v1/jobs/") or "/files/" not in request.url.path:
            log.info("request", extra={"event": "request", "rid": rid, "method": request.method,
                                       "path": request.url.path, "status": response.status_code,
                                       "ms": round((time.time() - t0) * 1000, 1)})
        return response

    # -- error envelope ------------------------------------------------------

    @app.exception_handler(StarletteHTTPException)
    async def http_exc(request, exc):
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            resp = _error_response(exc.status_code, detail["code"], detail.get("message", ""),
                                   detail.get("details"))
        else:
            resp = _error_response(exc.status_code, _STATUS_CODES.get(exc.status_code, "error"),
                                   str(detail) if detail else "Request failed.")
        for k, v in (getattr(exc, "headers", None) or {}).items():
            resp.headers[k] = v
        return resp

    @app.exception_handler(RequestValidationError)
    async def validation_exc(request, exc):
        return _error_response(422, "validation_error", "Request did not match the API schema.",
                               json.loads(json.dumps(exc.errors(), default=str)))

    @app.exception_handler(engine.CoreUnavailable)
    async def core_exc(request, exc):
        return _error_response(503, "core_unavailable", str(exc))

    @app.exception_handler(engine.UpstreamError)
    async def upstream_exc(request, exc):
        return _error_response(502, "upstream_error", str(exc))

    @app.exception_handler(Exception)
    async def any_exc(request, exc):
        log.exception("unhandled", extra={"event": "request_error", "path": request.url.path})
        return _error_response(500, "internal_error", "%s: %s" % (type(exc).__name__, engine._safe_err(exc)))

    # -- routers -------------------------------------------------------------

    for r in (health_router, jobs_router, fix_router, check_text_router, lexicon_router):
        app.include_router(r, prefix=settings.API_PREFIX)

    @app.get("/v1", include_in_schema=False)
    def v1_root():
        return {"name": "Retake API", "version": settings.VERSION, "docs": "/docs",
                "health": settings.API_PREFIX + "/health"}

    # -- static UI at / (when present) ----------------------------------------

    if settings.UI_DIR.is_dir() and (settings.UI_DIR / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(settings.UI_DIR), html=True), name="ui")
    else:
        @app.get("/", include_in_schema=False)
        def root():
            return RedirectResponse(url="/docs")

    return app


app = create_app()
