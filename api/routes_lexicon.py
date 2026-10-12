"""GET /v1/voices and GET /v1/lexicon — reference data for the UI."""
from typing import List

from fastapi import APIRouter, Depends, Query

from . import engine
from .auth import require_key
from .schemas import ErrorResponse, LexiconResponse, Voice

router = APIRouter(dependencies=[Depends(require_key)])


@router.get("/voices", response_model=List[Voice], tags=["voices"], summary="WellSaid voices",
            responses={502: {"model": ErrorResponse, "description": "WellSaid unreachable / no key"},
                       503: {"model": ErrorResponse, "description": "Engine not available"}},
            description="Voices available for rendering (`speaker_id` goes into POST /v1/jobs JSON "
                        "and POST /v1/jobs/{id}/fix). Cached server-side for six hours.")
def voices(refresh: bool = Query(False, description="Bypass the cache")):
    return engine.voices(force=refresh)


@router.get("/lexicon", response_model=LexiconResponse, tags=["lexicon"], summary="Heteronym lexicon",
            description="Every word Retake checks, with each sense (description, IPA) and the "
                        "WellSaid-format respelling candidates used to fix it.")
def lexicon():
    entries = engine.lexicon_entries()
    return {"count": len(entries), "words": entries}
