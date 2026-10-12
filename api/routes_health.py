"""GET /v1/health — liveness + capability report (always open, no key needed)."""
import time

from fastapi import APIRouter, Request

from . import engine, settings
from .schemas import HealthResponse, ModelsState

router = APIRouter(tags=["health"])
STARTED = time.time()


@router.get("/health", response_model=HealthResponse, summary="Service health and capabilities",
            description="Open endpoint (no key). `auth` tells you whether the server expects an "
                        "`X-Api-Key` header; `models` shows whether the heavy models are resident "
                        "(the first job warms them).")
def health(request: Request):
    worker = getattr(request.app.state, "worker", None)
    return HealthResponse(
        ok=True, version=settings.VERSION, auth=settings.auth_mode(),
        models=ModelsState(**engine.model_state()),
        wellsaid=engine.wellsaid_configured(), anthropic=settings.anthropic_configured(),
        core=engine.core_importable(), queue=worker.pending() if worker else 0,
        uptime_s=round(time.time() - STARTED, 1))
