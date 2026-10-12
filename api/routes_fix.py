"""Fixes: re-render flagged occurrences through WellSaid and verify them."""
import logging
import os
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from . import engine, settings
from .auth import require_key
from .jobs import STATUS_DONE, rewrite_paths
from .schemas import ErrorResponse, FixCreated, FixRequest, FixResponse

log = logging.getLogger("retake.fix")
router = APIRouter(tags=["fix"], dependencies=[Depends(require_key)])


def _err(status, code, message, details=None):
    d = {"code": code, "message": message}
    if details is not None:
        d["details"] = details
    return HTTPException(status_code=status, detail=d)


def _public_fix(fix, job, store):
    # type: (Dict[str, Any], Dict[str, Any], Any) -> Dict[str, Any]
    jdir = store.job_dir(job["id"])
    params = fix.get("params") or {}
    summary = {"requested": 0, "done": 0, "verified": 0, "unverified": 0, "errors": 0}
    summary.update(fix.get("summary") or {})
    return {"fix_id": fix["id"], "job_id": job["id"], "status": fix["status"], "stage": fix.get("stage"),
            "progress": fix.get("progress") or 0.0, "created": fix["created"], "updated": fix.get("updated"),
            "speaker_id": params.get("speaker_id"), "model": params.get("model"),
            "occurrence_ids": list(params.get("occurrence_ids") or []),
            "results": rewrite_paths(fix.get("results") or [], job["id"], jdir, job.get("token")),
            "summary": summary, "error": fix.get("error")}


_EXAMPLES = {"all flagged, caruso": {"value": {"speaker_id": 26, "model": "caruso"}},
             "two occurrences on preview": {"value": {"occurrence_ids": ["o3", "o5"], "speaker_id": 26,
                                                      "model": "preview"}}}


@router.post("/jobs/{job_id}/fix", status_code=202, response_model=FixCreated,
             summary="Re-render flagged words through WellSaid and verify",
             responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse},
                        409: {"model": ErrorResponse, "description": "Job not finished"},
                        503: {"model": ErrorResponse}},
             description="Queues a fix for the given occurrences (default: all flagged). `caruso` uses an "
                         "inline `<respell>` tag; `preview` uses spelling-as-pronunciation variants. Each try "
                         "renders the full sentence, then the word is re-checked with the phoneme scorer; "
                         "the first verified take wins. **Spends WellSaid credits.** Poll the returned URL.")
def create_fix(job_id: str, request: Request, body: FixRequest = Body(..., openapi_examples=_EXAMPLES)):
    store, worker = request.app.state.store, request.app.state.worker
    job = store.get_job(job_id)
    if not job:
        raise _err(404, "not_found", "Job %s does not exist." % job_id)
    if job["status"] != STATUS_DONE or not job.get("result"):
        raise _err(409, "job_not_done", "Job %s is %s; fixes need a finished check." % (job_id, job["status"]))
    if not engine.wellsaid_configured():
        raise _err(503, "wellsaid_unavailable", "This server has no WellSaid API key; fixes cannot render.")
    occs = [o for o in job["result"].get("occurrences", []) if isinstance(o, dict)]
    by_id = {o.get("id"): o for o in occs}
    if body.occurrence_ids:
        unknown = [i for i in body.occurrence_ids if i not in by_id]
        if unknown:
            raise _err(400, "unknown_occurrence", "Unknown occurrence id(s): %s" % ", ".join(unknown),
                       {"known": list(by_id.keys())})
        ids = list(dict.fromkeys(body.occurrence_ids))
    else:
        ids = [o["id"] for o in occs if o.get("flag")]
        if not ids:
            raise _err(400, "nothing_to_fix", "No flagged occurrences in job %s." % job_id)
    fix = store.create_fix(job_id, {"occurrence_ids": ids, "speaker_id": body.speaker_id,
                                    "model": body.model, "max_tries": body.max_tries})
    worker.enqueue_fix(job_id, fix["id"])
    log.info("fix queued", extra={"event": "fix_queued", "job_id": job_id, "fix_id": fix["id"], "n": len(ids)})
    return FixCreated(fix_id=fix["id"], job_id=job_id, status="queued",
                      status_url="%s/jobs/%s/fixes/%s" % (settings.API_PREFIX, job_id, fix["id"]))


@router.get("/jobs/{job_id}/fixes/{fix_id}", response_model=FixResponse, summary="Fix status + results",
            responses={404: {"model": ErrorResponse}})
def get_fix(job_id: str, fix_id: str, request: Request):
    store = request.app.state.store
    job = store.get_job(job_id)
    if not job:
        raise _err(404, "not_found", "Job %s does not exist." % job_id)
    fix = store.get_fix(job_id, fix_id)
    if not fix:
        raise _err(404, "not_found", "Fix %s does not exist on job %s." % (fix_id, job_id))
    return _public_fix(fix, job, store)
