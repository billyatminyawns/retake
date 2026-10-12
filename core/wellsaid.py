# -*- coding: utf-8 -*-
"""WellSaid REST client (port of heteronym-poc wellsaid_client.py, + models).

- key resolution: explicit > env WELLSAID_API_KEY > ~/Downloads/wellsaid-connector/.env
- rate limiter (>= 0.5 s between request starts), 429 backoff (2 s, 3 tries)
- voices() via GET /tts/avatars (normalized, cached, sorted by name)
- render(text, speaker_id, model, fmt, sample_rate) -> audio bytes via POST /tts/stream
  body: {speaker_id, text, model?, audio_configs: {file_format, sample_rate?}, library_ids?}
  (shape verified against the WellSaid connector, 10/2026; models: legacy|caruso|preview)
- render_timed(...) -> {audio_bytes, timing:[{word,start,end}], timing_raw, ext}
  via POST /tts/word-timing (zip: audio member + Google-STT-style timing JSON
  with "0.560s" offset strings)
- respell_suggestions(word) -> list[str] (defensive, never raises)
- normalize_timing(timing_raw) -> [{"word","start","end"}]

The API key is NEVER printed or logged, and never included in error messages.
Renders SPEND CREDITS: core.selftest never calls them.
"""
import io
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional

BASE_URL = os.environ.get("WELLSAID_API_BASE", "https://api.wellsaidlabs.com/v1").rstrip("/")
CONNECTOR_ENV = os.path.expanduser("~/Downloads/wellsaid-connector/.env")

MODELS = ("legacy", "caruso", "preview")
MAX_TEXT_CHARS = 1000          # WellSaid limit, INCLUDING markup
MIN_REQUEST_INTERVAL = 0.5     # seconds between request starts (client-side)
BACKOFF_429 = 2.0              # seconds to wait after a 429
MAX_TRIES = 3                  # total attempts per request
RENDER_TIMEOUT = 120           # seconds (renders can be slow)
META_TIMEOUT = 30              # seconds for metadata GETs


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------

def _read_key_from_env_file(path):
    # type: (str) -> Optional[str]
    """Parse WELLSAID_API_KEY=... from a dotenv-style file. None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() != "WELLSAID_API_KEY":
                    continue
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    except OSError:
        pass
    return None


def resolve_key(key=None):
    # type: (Optional[str]) -> Optional[str]
    """Explicit key > env WELLSAID_API_KEY > connector .env. None if absent."""
    if key:
        return key
    env_key = os.environ.get("WELLSAID_API_KEY", "").strip()
    if env_key:
        return env_key
    return _read_key_from_env_file(CONNECTOR_ENV)


def available():
    # type: () -> bool
    """True when a WellSaid key resolves (for the health endpoint)."""
    return bool(resolve_key())


# ---------------------------------------------------------------------------
# Timing normalization (pure function, exported)
# ---------------------------------------------------------------------------

_WRAPPER_KEYS = ("words", "word_timings", "wordTimings", "timings", "timing",
                 "word_timing", "segments", "items", "results", "data")
_WORD_KEYS = ("word", "text", "token", "label")
_START_KEYS = ("start", "start_time", "startTime", "start_ms", "startMs", "begin", "from")
_END_KEYS = ("end", "end_time", "endTime", "end_ms", "endMs", "stop", "to")
_DUR_KEYS = ("duration", "dur", "length")


def _as_float(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _first_numeric(item, keys):
    for key in keys:
        if key in item:
            val = _as_float(item[key])
            if val is not None:
                return val, key
    return None, None


def _unwrap_timing(timing_raw):
    obj = timing_raw
    if isinstance(obj, (bytes, str)):
        try:
            obj = json.loads(obj)
        except (ValueError, UnicodeDecodeError):
            return None
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in _WRAPPER_KEYS:
            if isinstance(obj.get(key), list):
                return obj[key]
        lists = [v for v in obj.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    return None


def normalize_timing(timing_raw):
    # type: (Any) -> List[Dict[str, Any]]
    """Normalize a word-timing payload to [{"word","start","end"}] in file order.

    WellSaid's real /tts/word-timing shape (verified 10/9/26) is Google-STT
    style: {"results":[{"alternatives":[{"words":[{"startOffset":"0.560s",
    "endOffset":"0.880s","word":"resume"}, ...]}]}]} — string offsets with a
    trailing "s"; the very first word may omit startOffset (meaning 0.0).
    That shape is handled first; the tolerant generic path (key variants,
    ms vs s, wrappers, triples) follows. Unrecognized payloads yield [].
    """
    if isinstance(timing_raw, dict) and isinstance(timing_raw.get("results"), list):
        def _sec(v, default=None):
            if v is None:
                return default
            if isinstance(v, str):
                v = v.rstrip("sS").strip()
            try:
                return float(v)
            except (TypeError, ValueError):
                return default
        out = []
        for res in timing_raw["results"]:
            alts = res.get("alternatives") if isinstance(res, dict) else None
            if not isinstance(alts, list) or not alts:
                continue
            for w in alts[0].get("words") or []:
                if not isinstance(w, dict) or not w.get("word"):
                    continue
                start = _sec(w.get("startOffset"), default=(out[-1]["end"] if out else 0.0))
                end = _sec(w.get("endOffset"))
                if start is None or end is None:
                    continue
                out.append({"word": str(w["word"]), "start": start, "end": end})
        if out:
            return out
    items = _unwrap_timing(timing_raw)
    if not items:
        return []

    entries = []
    saw_ms_key = False
    max_val = 0.0
    for item in items:
        word = None
        start = end = None
        if isinstance(item, dict):
            for key in _WORD_KEYS:
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    word = val
                    break
            start, skey = _first_numeric(item, _START_KEYS)
            end, ekey = _first_numeric(item, _END_KEYS)
            if end is None and start is not None:
                dur, dkey = _first_numeric(item, _DUR_KEYS)
                if dur is not None:
                    end = start + dur
                    ekey = dkey
            for key in (skey, ekey):
                if key and key.lower().endswith("ms"):
                    saw_ms_key = True
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            if isinstance(item[0], str) and item[0].strip():
                word = item[0]
            start = _as_float(item[1])
            end = _as_float(item[2])
        if word is None or start is None or end is None:
            continue
        max_val = max(max_val, abs(start), abs(end))
        entries.append((word, start, end))

    scale = 1000.0 if (saw_ms_key or max_val > 1000.0) else 1.0
    out = []
    for word, start, end in entries:
        start, end = start / scale, end / scale
        if end < start:
            start, end = end, start
        out.append({"word": word, "start": start, "end": end})
    return out


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class WellSaidError(RuntimeError):
    """API-level failure (non-2xx after retries, transport error, bad payload)."""


def _xml_escape(s):
    # type: (str) -> str
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def respell_tag(word, respell):
    # type: (str, str) -> str
    """Inline WellSaid respelling markup (caruso honors it; preview does not)."""
    return '<respell value="%s">%s</respell>' % (_xml_escape(respell), _xml_escape(word))


class WellSaid(object):
    """Thread-safe WellSaid REST client. `.ok` is True when a key resolved."""

    def __init__(self, key=None):
        self._key = resolve_key(key)
        self.ok = bool(self._key)
        self._voices_cache = None
        self._voices_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._last_request_start = 0.0

    def __repr__(self):
        return "<WellSaid ok=%s>" % self.ok  # never the key

    # -- plumbing -----------------------------------------------------------

    def _throttle(self):
        with self._rate_lock:
            now = time.monotonic()
            wait = self._last_request_start + MIN_REQUEST_INTERVAL - now
            if wait > 0:
                time.sleep(wait)
            self._last_request_start = time.monotonic()

    def _request(self, method, path, body=None, accept="application/json",
                 timeout=META_TIMEOUT):
        # type: (str, str, Optional[Dict[str, Any]], str, int) -> bytes
        if not self.ok:
            raise WellSaidError(
                "No WellSaid API key found (set WELLSAID_API_KEY or add it to "
                "~/Downloads/wellsaid-connector/.env)")
        url = BASE_URL + path
        data = None
        headers = {"X-Api-Key": self._key, "Accept": accept}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        last_err = "unknown error"
        for attempt in range(1, MAX_TRIES + 1):
            self._throttle()
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read(500).decode("utf-8", "replace").strip()
                except Exception:
                    pass
                last_err = "HTTP %s from %s %s: %s" % (e.code, method, path, detail or e.reason)
                if e.code == 429 and attempt < MAX_TRIES:
                    time.sleep(BACKOFF_429)
                    continue
                raise WellSaidError(last_err)
            except (urllib.error.URLError, OSError) as e:
                last_err = "network error on %s %s: %s" % (method, path, e)
                if attempt < MAX_TRIES:
                    time.sleep(1.0)
                    continue
        raise WellSaidError(last_err)

    @staticmethod
    def _check_text(text):
        if not text or not str(text).strip():
            raise ValueError("cannot render empty text")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(
                "text is %d chars, over the %d-char WellSaid limit (markup "
                "included) — split before rendering" % (len(text), MAX_TEXT_CHARS))

    @staticmethod
    def _body(text, speaker_id, model, fmt, sample_rate, library_ids):
        body = {"speaker_id": int(speaker_id), "text": text}  # type: Dict[str, Any]
        if model:
            if model not in MODELS:
                raise ValueError("unknown WellSaid model %r (use one of %s)"
                                 % (model, ", ".join(MODELS)))
            body["model"] = model
        cfg = {}  # type: Dict[str, Any]
        if fmt:
            cfg["file_format"] = fmt
        if sample_rate:
            cfg["sample_rate"] = int(sample_rate)
        if cfg:
            body["audio_configs"] = cfg
        if library_ids:
            body["library_ids"] = list(library_ids)
        return body

    # -- endpoints ----------------------------------------------------------

    def voices(self, force=False):
        # type: (bool) -> List[Dict[str, Any]]
        """GET /tts/avatars -> [{speaker_id:int, name, style, accent, gender}],
        sorted by name, cached for the process lifetime."""
        with self._voices_lock:
            if self._voices_cache is not None and not force:
                return self._voices_cache
            raw = self._request("GET", "/tts/avatars")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise WellSaidError("GET /tts/avatars returned non-JSON payload")
            if isinstance(payload, dict):
                items = payload.get("avatars") or payload.get("voices") or []
            elif isinstance(payload, list):
                items = payload
            else:
                items = []
            voices = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                sid = item.get("speaker_id", item.get("id"))
                try:
                    sid = int(sid)
                except (TypeError, ValueError):
                    continue
                voices.append({
                    "speaker_id": sid,
                    "name": item.get("name"),
                    "style": item.get("style"),
                    "accent": item.get("accent", item.get("accent_type")),
                    "gender": item.get("gender"),
                })
            voices.sort(key=lambda v: (v["name"] is None, v["name"] or ""))
            self._voices_cache = voices
            return voices

    def render(self, text, speaker_id, model="caruso", fmt="mp3", sample_rate=None,
               library_ids=None):
        # type: (str, int, Optional[str], str, Optional[int], Optional[List[str]]) -> bytes
        """POST /tts/stream -> audio bytes (mp3 by default). SPENDS CREDITS."""
        self._check_text(text)
        body = self._body(text, speaker_id, model, fmt, sample_rate, library_ids)
        accept = "audio/wav" if (fmt or "").lower() == "wav" else "audio/mpeg"
        audio = self._request("POST", "/tts/stream", body=body, accept=accept,
                              timeout=RENDER_TIMEOUT)
        if not audio:
            raise WellSaidError("POST /tts/stream returned an empty body")
        return audio

    def render_timed(self, text, speaker_id, model="caruso", fmt="mp3", sample_rate=None,
                     library_ids=None):
        # type: (str, int, Optional[str], str, Optional[int], Optional[List[str]]) -> Dict[str, Any]
        """POST /tts/word-timing -> {"audio_bytes": bytes, "timing": [{word,start,end}],
        "timing_raw": <json>, "ext": "mp3"|"wav"}. SPENDS CREDITS.

        The endpoint returns a zip containing the audio plus a word-timing
        JSON (and srt/vtt we ignore). First .mp3 (else .wav) is the audio,
        first .json is the timing.
        """
        self._check_text(text)
        body = self._body(text, speaker_id, model, fmt, sample_rate, library_ids)
        raw = self._request("POST", "/tts/word-timing", body=body,
                            accept="application/zip", timeout=RENDER_TIMEOUT)
        if not raw:
            raise WellSaidError("POST /tts/word-timing returned an empty body")
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
        except (zipfile.BadZipFile, OSError):
            head = raw[:200].decode("utf-8", "replace")
            raise WellSaidError("POST /tts/word-timing did not return a readable zip "
                                "(%d bytes; starts: %r)" % (len(raw), head))
        names = [n for n in zf.namelist() if not n.endswith("/") and "__MACOSX" not in n]
        audio_name = next((n for n in names if n.lower().endswith(".mp3")), None)
        if audio_name is None:
            audio_name = next((n for n in names if n.lower().endswith(".wav")), None)
        json_name = next((n for n in names if n.lower().endswith(".json")), None)
        if not audio_name or not json_name:
            raise WellSaidError("word-timing zip is missing the audio or timing member; "
                                "members: %s" % (", ".join(names) if names else "(empty zip)"))
        audio = zf.read(audio_name)
        if not audio:
            raise WellSaidError("word-timing zip audio member %s is empty" % audio_name)
        try:
            timing_raw = json.loads(zf.read(json_name).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise WellSaidError("word-timing zip member %s is not valid JSON" % json_name)
        return {"audio_bytes": audio,
                "timing": normalize_timing(timing_raw),
                "timing_raw": timing_raw,
                "ext": audio_name.rsplit(".", 1)[-1].lower()}

    def respell_suggestions(self, word):
        # type: (str) -> List[str]
        """GET /tts/respelling_suggestions?word=<w> -> list[str]. Never raises."""
        if not word or not str(word).strip():
            return []
        try:
            raw = self._request("GET", "/tts/respelling_suggestions?word="
                                + urllib.parse.quote(str(word).strip()))
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return []
        if isinstance(payload, dict):
            for key in ("suggestions", "respellings", "results", "data"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                payload = [payload]
        if not isinstance(payload, list):
            return []
        out = []
        for item in payload:
            value = None
            if isinstance(item, str):
                value = item
            elif isinstance(item, dict):
                for key in ("phonetic_respelling", "respelling", "respell", "suggestion", "value"):
                    if isinstance(item.get(key), str) and item[key].strip():
                        value = item[key]
                        break
            if value:
                value = value.strip()
                if value and value not in out:
                    out.append(value)
        return out
