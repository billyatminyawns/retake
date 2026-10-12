"""POST /v1/check-text — script pre-flight (no audio)."""
from fastapi import APIRouter, Body, Depends

from . import engine
from .auth import require_key
from .schemas import CheckTextRequest, CheckTextResponse, ErrorResponse

router = APIRouter(tags=["check-text"], dependencies=[Depends(require_key)])

_EXAMPLES = {
    "studio script": {
        "summary": "A short narration with four heteronyms",
        "value": {"text": "Please record the record. We present the present live, live from the studio."},
    },
    "compound": {
        "summary": "Prefix + plural compound (re-records)",
        "value": {"text": "The system automatically re-records the lesson when the policy changes.",
                  "engine": "spacy"},
    },
}


@router.post("/check-text", response_model=CheckTextResponse, summary="Find heteronyms in a script",
             responses={502: {"model": ErrorResponse, "description": "Intent engine unavailable"},
                        503: {"model": ErrorResponse, "description": "Engine not available"}},
             description="Scans the text for every heteronym (including re-/un-/pre-/mis- compounds "
                         "and plural tails) and decides the intended sense from context — Claude when "
                         "configured, spaCy POS otherwise. Use it before rendering to know which words "
                         "need a respelling, or to preview what a media check will listen for.")
def check_text(body: CheckTextRequest = Body(..., openapi_examples=_EXAMPLES)):
    return engine.check_text(body.text, engine=body.engine)
