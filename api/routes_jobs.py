"""Jobs: create (multipart upload or JSON script), list, get, files, delete."""
import json
import logging
import os
import re
import shutil
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import ValidationError

from . import engine, settings
from .auth import file_access_allowed, require_key
from .jobs import STATUS_DONE, file_url, rewrite_paths
from .schemas import (DeleteResponse, ErrorResponse, JobCreated, JobListResponse, JobResponse,
                      JobTextRequest)

log = logging.getLogger("retake.jobs")
router = APIRouter(tags=["jobs"])

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_ENGINES = ("claude", "spacy", "both")


def _err(status, code, message, details=None):
    # type: (int, str, str, Any) -> HTTPException
    d = {"code": code, "message": message}
    if details is not None:
        d["details"] = details
    return HTTPException(status_code=status, detail=d)


def _store(request):
    return request.app.state.store


def _worker(request):
    return request.app.state.worker


def _safe_filename(name):
    # type: (Optional[str]) -> str
    base = os.path.basename(name or "").strip() or "media"
    base = _SAFE_NAME.sub("_", base)[:120].strip("._") or "media"
    root, ext = os.path.splitext(base)
    return "media" + ext.lower() if ext else "media"


def _public_job(job, store, full=True):
    # type: (Dict[str, Any], Any, bool) -> Dict[str, Any]
    jdir = store.job_dir(job["id"])
    token = job.get("token")
    params = dict(job.get("params") or {})
    out = {k: job.get(k) for k in ("id", "status", "stage", "progress", "created", "updated", "mode",
                                   "filename", "duration", "occurrences", "flags", "error")}
    if not full:
        return out
    out["params"] = {"engine": params.get("engine"), "speaker_id": params.get("speaker_id"),
                     "model": params.get("model"), "has_script": bool(params.get("script") or params.get("text"))}
    files = {}
    media_name = job.get("filename") if job.get("mode") == "script" else params.get("media_name")
    if media_name and os.path.isfile(os.path.join(jdir, media_name)):
        files["media"] = file_url(job["id"], media_name, token)
    for key, name in (("report_json", "report.json"), ("report_html", "report.html")):
        if os.path.isfile(os.path.join(jdir, name)):
            files[key] = file_url(job["id"], name, token)
    out["files"] = files
    out["result"] = rewrite_paths(job["result"], job["id"], jdir, token) if job.get("result") else None
    out["fixes"] = store.list_fix_ids(job["id"])
    return out


_MULTIPART_DOC = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {"type": "object", "required": ["media"], "properties": {
                    "media": {"type": "string", "format": "binary",
                              "description": "Audio or video file (any ffmpeg-readable container), up to 500 MB"},
                    "script": {"type": "string", "description": "Optional script text; aligned to the transcript"},
                    "engine": {"type": "string", "enum": list(_ENGINES), "default": "both",
                               "description": "Intent engine"}}}},
            "application/json": {
                "schema": {"$ref": "#/components/schemas/JobTextRequest"},
                "examples": {"render and check": {"value": {
                    "text": "Please record the record live, live from the studio.",
                    "speaker_id": 26, "model": "caruso"}}}}}}}


@router.post("/jobs", status_code=202, response_model=JobCreated, dependencies=[Depends(require_key)],
             summary="Submit media (multipart) or a script (JSON) for checking", openapi_extra=_MULTIPART_DOC,
             responses={202: {"description": "Accepted; poll `status_url`"},
                        400: {"model": ErrorResponse}, 413: {"model": ErrorResponse, "description": "Upload too large"},
                        415: {"model": ErrorResponse, "description": "Not a readable audio/video file"},
                        422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
             description="**multipart/form-data**: `media` file (+ optional `script`, `engine`). The file is "
                         "probed with ffmpeg; anything without an audio stream is rejected (415).\n\n"
                         "**application/json**: `{text, speaker_id, model}` renders the text through WellSaid "
                         "first (spends credits), then checks the result against the script.\n\n"
                         "Returns **202** with `job_id` and `status_url`; poll it every ~1.5 s until "
                         "`status` is `done` or `error`.")
async def create_job(request: Request):
    ctype = (request.headers.get("content-type") or "").lower()
    store, worker = _store(request), _worker(request)
    if ctype.startswith("multipart/form-data"):
        job = await _create_upload_job(request, store)
    elif ctype.startswith("application/json") or not ctype:
        job = await _create_text_job(request, store)
    else:
        raise _err(415, "unsupported_content_type",
                   "Send multipart/form-data (media upload) or application/json ({text, speaker_id, model}).")
    worker.enqueue_check(job["id"])
    log.info("job queued", extra={"event": "job_queued", "job_id": job["id"], "mode": job["mode"]})
    return JobCreated(job_id=job["id"], status="queued",
                      status_url="%s/jobs/%s" % (settings.API_PREFIX, job["id"]))


async def _create_upload_job(request, store):
    limit = settings.max_upload_bytes()
    clen = request.headers.get("content-length")
    if clen and clen.isdigit() and int(clen) > limit + 1024 * 1024:
        raise _err(413, "payload_too_large", "Upload exceeds the %g MB limit." % (limit / (1024.0 * 1024)))
    try:
        form = await request.form(max_files=4, max_fields=16)
    except Exception as e:
        raise _err(400, "bad_multipart", "Could not parse the multipart body: %s" % engine._safe_err(e))
    upload = form.get("media")
    if upload is None or not hasattr(upload, "file"):
        raise _err(400, "missing_media", "Multipart field 'media' (a file) is required.")
    engine_name = (form.get("engine") or settings.default_engine()).strip().lower()
    if engine_name not in _ENGINES:
        raise _err(400, "bad_engine", "engine must be one of claude, spacy, both.")
    script = form.get("script")
    script = script.strip() if isinstance(script, str) and script.strip() else None
    filename = _safe_filename(getattr(upload, "filename", None))

    job = store.create_job("upload", os.path.basename(getattr(upload, "filename", None) or filename),
                           {"engine": engine_name, "script": script, "media_name": filename})
    jdir = store.job_dir(job["id"])
    dst = os.path.join(jdir, filename)
    size = 0
    try:
        with open(dst, "wb") as fh:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise _err(413, "payload_too_large",
                               "Upload exceeds the %g MB limit." % (limit / (1024.0 * 1024)))
                fh.write(chunk)
        if size == 0:
            raise _err(400, "empty_media", "The uploaded file is empty.")
        try:
            info = engine.probe_media(dst)
        except engine.NotMedia as e:
            raise _err(415, "unsupported_media", str(e))
        except engine.CoreUnavailable as e:
            raise _err(503, "ffmpeg_unavailable", str(e))
    except HTTPException:
        store.delete_job(job["id"])
        raise
    finally:
        try:
            await upload.close()
        except Exception:
            pass
    store.update_job(job["id"], duration=info.get("duration"),
                     params={"engine": engine_name, "script": script, "media_name": filename,
                             "size": size, "has_video": info.get("has_video")})
    return store.get_job(job["id"])


async def _create_text_job(request, store):
    try:
        raw = await request.body()
        body = JobTextRequest.model_validate(json.loads(raw.decode("utf-8") or "{}"))
    except ValidationError as e:
        raise _err(422, "validation_error", "Invalid JSON body for a script job.", e.errors())
    except (ValueError, UnicodeDecodeError) as e:
        raise _err(400, "bad_json", "Body is not valid JSON: %s" % e)
    text = body.text.strip()
    if len(text) > settings.max_text_chars():
        raise _err(400, "text_too_long", "Script is %d characters; the limit is %d."
                   % (len(text), settings.max_text_chars()))
    if not engine.wellsaid_configured():
        raise _err(503, "wellsaid_unavailable",
                   "This server has no WellSaid API key, so scripts cannot be rendered. Upload media instead.")
    return store.create_job("script", None, {"engine": body.engine or settings.default_engine(),
                                             "text": text, "speaker_id": body.speaker_id,
                                             "model": body.model})


@router.get("/jobs", response_model=JobListResponse, dependencies=[Depends(require_key)],
            summary="Recent jobs (newest first)")
def list_jobs(request: Request, limit: int = Query(50, ge=1, le=200)):
    store = _store(request)
    rows = [_public_job(j, store, full=False) for j in store.list_jobs(limit)]
    return {"count": len(rows), "jobs": rows}


@router.get("/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_key)],
            summary="Job status + result", responses={404: {"model": ErrorResponse}},
            description="`status`: queued → running → done | error. While running, `stage` and "
                        "`progress` (0–1) advance. When done, `result` is the full check with every "
                        "file path rewritten to a URL under `/v1/jobs/{id}/files/`.")
def get_job(job_id: str, request: Request):
    store = _store(request)
    job = store.get_job(job_id)
    if not job:
        raise _err(404, "not_found", "Job %s does not exist." % job_id)
    return _public_job(job, store)


@router.get("/jobs/{job_id}/files/{name:path}", summary="Download a job file", response_class=FileResponse,
            responses={200: {"content": {"audio/*": {}, "video/*": {}, "application/json": {}, "text/html": {}}},
                       401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
            description="Serves the media, word/sentence clips, fix takes, `report.json` and `report.html`. "
                        "Accepts the API key header, or the per-job token (`?t=`) embedded in the URLs the "
                        "job response returns, so browser players can load them directly.")
def get_file(job_id: str, name: str, request: Request):
    store = _store(request)
    job = store.get_job(job_id)
    if not job:
        raise _err(404, "not_found", "Job %s does not exist." % job_id)
    if not file_access_allowed(request, job.get("token")):
        raise HTTPException(status_code=401, detail={"code": "unauthorized",
                                                     "message": "Missing or invalid API key / file token."},
                            headers={"WWW-Authenticate": "ApiKey"})
    jdir = os.path.abspath(store.job_dir(job_id))
    path = os.path.abspath(os.path.join(jdir, name))
    if not path.startswith(jdir + os.sep) or not os.path.isfile(path):
        raise _err(404, "not_found", "No such file in job %s." % job_id)
    media_type = None
    low = path.lower()
    if low.endswith(".json"):
        media_type = "application/json"
    elif low.endswith(".html"):
        media_type = "text/html; charset=utf-8"
    elif low.endswith(".wav"):
        media_type = "audio/wav"
    elif low.endswith(".mp3"):
        media_type = "audio/mpeg"
    resp = FileResponse(path, media_type=media_type, filename=os.path.basename(path)
                        if request.query_params.get("download") else None)
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@router.delete("/jobs/{job_id}", response_model=DeleteResponse, dependencies=[Depends(require_key)],
               summary="Delete a job and its files", responses={404: {"model": ErrorResponse},
                                                               409: {"model": ErrorResponse}})
def delete_job(job_id: str, request: Request):
    store, worker = _store(request), _worker(request)
    job = store.get_job(job_id)
    if not job:
        raise _err(404, "not_found", "Job %s does not exist." % job_id)
    cur = worker.current
    if cur and cur[1] == job_id:
        raise _err(409, "job_running", "Job %s is being processed right now; try again when it finishes." % job_id)
    store.delete_job(job_id)
    log.info("job deleted", extra={"event": "job_deleted", "job_id": job_id})
    return DeleteResponse(deleted=job_id, ok=True)
