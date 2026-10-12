"""X-Api-Key authentication.

Rules (SPEC "Environment facts"):
  * `RETAKE_API_KEYS` env = comma-separated list of accepted keys.
  * Unset/empty  -> OPEN mode: every request is accepted; /v1/health reports
    `auth: "open"` so nobody mistakes a dev box for a locked-down deployment.
  * Set          -> KEY mode: requests must carry `X-Api-Key: <key>` (or a
    `Authorization: Bearer <key>` header, accepted as a convenience).
    Missing/invalid -> 401 {"error": {"code": "unauthorized", ...}}.

Keys are compared with constant-time `hmac.compare_digest` and are never
logged, echoed, or included in error messages.
"""
import hmac
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import APIKeyHeader

from . import settings

# Declared so Swagger shows the "Authorize" button and sends the header.
api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False,
                              description="Your Retake API key (omit in open/dev mode).")


def _bearer(authorization):
    # type: (Optional[str]) -> Optional[str]
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or None
    return None


def key_is_valid(candidate):
    # type: (Optional[str]) -> bool
    """True when `candidate` matches one configured key (constant time)."""
    if not candidate:
        return False
    for k in settings.api_keys():
        if hmac.compare_digest(candidate.encode("utf-8"), k.encode("utf-8")):
            return True
    return False


def _unauthorized(message):
    # type: (str) -> HTTPException
    return HTTPException(status_code=401,
                         detail={"code": "unauthorized", "message": message},
                         headers={"WWW-Authenticate": "ApiKey"})


async def require_key(request: Request,
                      x_api_key: Optional[str] = Depends(api_key_header),
                      authorization: Optional[str] = Header(default=None)):
    """FastAPI dependency: enforce the key in key mode, no-op in open mode."""
    if settings.auth_mode() == "open":
        request.state.auth = "open"
        return "open"
    candidate = x_api_key or _bearer(authorization)
    if candidate is None:
        raise _unauthorized("Missing X-Api-Key header.")
    if not key_is_valid(candidate):
        raise _unauthorized("Invalid API key.")
    request.state.auth = "key"
    return "key"


def file_access_allowed(request, job_token):
    # type: (Request, Optional[str]) -> bool
    """Auth rule for GET /v1/jobs/{id}/files/*.

    Browsers cannot attach custom headers to <audio src> / <a download>, so a
    job's files may also be fetched with the per-job capability token that the
    API embeds in the rewritten file URLs (`?t=...`). The token grants access
    to that one job's files only; it is not an API key.
    """
    if settings.auth_mode() == "open":
        return True
    candidate = request.headers.get("x-api-key") or _bearer(request.headers.get("authorization"))
    if key_is_valid(candidate):
        return True
    t = request.query_params.get("t")
    if t and job_token and hmac.compare_digest(t.encode("utf-8"), job_token.encode("utf-8")):
        return True
    return False
