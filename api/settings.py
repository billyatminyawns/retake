"""Environment-driven settings for the Retake API.

Everything is read from the process environment at import time (and re-read
where a value may legitimately change under test, e.g. the API key list).
Secrets are never logged: `summary()` only reports whether they are present.
"""
import os
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent          # project root
LEXICON_PATH = ROOT / "data" / "heteronyms.json"
UI_DIR = ROOT / "ui"

VERSION = "1.0.0"
API_PREFIX = "/v1"


def _env(name, default=None):
    # type: (str, Optional[str]) -> Optional[str]
    val = os.environ.get(name)
    if val is None or not str(val).strip():
        return default
    return str(val).strip()


def data_dir():
    # type: () -> Path
    """RETAKE_DATA (default <root>/var). Holds jobs.db + jobs/<id>/."""
    return Path(_env("RETAKE_DATA", str(ROOT / "var"))).expanduser()


def jobs_dir():
    # type: () -> Path
    return data_dir() / "jobs"


def db_path():
    # type: () -> Path
    return data_dir() / "jobs.db"


def api_keys():
    # type: () -> List[str]
    """RETAKE_API_KEYS as a list (comma separated). Empty list == open mode."""
    raw = _env("RETAKE_API_KEYS", "") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


def auth_mode():
    # type: () -> str
    return "key" if api_keys() else "open"


def cors_origins():
    # type: () -> List[str]
    raw = _env("RETAKE_CORS_ORIGINS", "*") or "*"
    return [o.strip() for o in raw.split(",") if o.strip()] or ["*"]


def max_upload_bytes():
    # type: () -> int
    mb = _env("RETAKE_MAX_UPLOAD_MB", "500")
    try:
        return int(float(mb) * 1024 * 1024)
    except ValueError:
        return 500 * 1024 * 1024


def max_text_chars():
    # type: () -> int
    """Upper bound for script text we will render through WellSaid."""
    try:
        return int(_env("RETAKE_MAX_TEXT_CHARS", "6000") or "6000")
    except ValueError:
        return 6000


def default_engine():
    # type: () -> str
    return _env("RETAKE_INTENT_ENGINE", "both") or "both"


def core_package():
    # type: () -> str
    """Import name of the engine package. Tests may point this at a stub."""
    return _env("RETAKE_CORE_PACKAGE", "core") or "core"


def log_level():
    # type: () -> str
    return (_env("RETAKE_LOG_LEVEL", "INFO") or "INFO").upper()


def anthropic_configured():
    # type: () -> bool
    return bool(_env("ANTHROPIC_API_KEY"))


def summary():
    # type: () -> dict
    """Loggable, secret-free view of the configuration."""
    return {
        "data_dir": str(data_dir()),
        "auth": auth_mode(),
        "cors_origins": cors_origins(),
        "max_upload_mb": round(max_upload_bytes() / (1024 * 1024)),
        "core_package": core_package(),
        "ui_dir": str(UI_DIR) if UI_DIR.is_dir() else None,
        "anthropic": anthropic_configured(),
    }
