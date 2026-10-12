"""Lazy bridge between the API and the `core` engine package.

Every heavy or optional import happens inside a function so the API process
starts (and /v1/health, /docs, the UI keep working) even while `core` is
still being written or its models are not yet installed. A missing module
raises `CoreUnavailable`, which the routers turn into a clean 503
`{"error": {"code": "core_unavailable", ...}}` instead of a crash.

The package name is read from `RETAKE_CORE_PACKAGE` (default "core") so a
self-test can point the API at a stub engine without touching the real one.

Also here: ffmpeg discovery + media probing (reject non-media uploads), the
WellSaid key presence check (the key itself is never read into a log), a
script-only heteronym pre-flight (`check_text`) that prefers core's own
implementation and otherwise ports the proven check.py/analyze.py logic
(compounds re-/un-/pre-/mis- + plural tails, spaCy POS fallback, Claude).
"""
import importlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import settings

log = logging.getLogger("retake.engine")


class CoreUnavailable(Exception):
    """The engine (or one of its optional dependencies) cannot be imported."""


class UpstreamError(Exception):
    """A dependency (WellSaid, Anthropic) failed; maps to 502."""


# ---------------------------------------------------------------------------
# core imports
# ---------------------------------------------------------------------------

_IMPORT_LOCK = threading.Lock()


def core_module(name):
    # type: (str) -> Any
    """Import `<core_package>.<name>`; raise CoreUnavailable with a clear message."""
    pkg = settings.core_package()
    full = "%s.%s" % (pkg, name)
    with _IMPORT_LOCK:
        try:
            return importlib.import_module(full)
        except ImportError as e:
            raise CoreUnavailable(
                "The analysis engine module '%s' is not available (%s). "
                "Install the engine dependencies and restart the API." % (full, e))
        except Exception as e:  # a broken module must not take the API down
            raise CoreUnavailable("The analysis engine module '%s' failed to import: %s: %s"
                                  % (full, type(e).__name__, e))


def core_importable():
    # type: () -> bool
    try:
        importlib.import_module(settings.core_package())
        return True
    except Exception:
        return False


def _attr(mod, name):
    fn = getattr(mod, name, None)
    if fn is None:
        raise CoreUnavailable("The analysis engine lacks '%s.%s'." % (mod.__name__, name))
    return fn


# ---------------------------------------------------------------------------
# model state (for /v1/health)
# ---------------------------------------------------------------------------

_MODEL_SEEN = {"whisper": False, "phonemes": False}   # set once a job used them here


def note_models_used():
    _MODEL_SEEN["whisper"] = True
    _MODEL_SEEN["phonemes"] = True


def _probe_loaded(modname, attr_candidates):
    # type: (str, List[str]) -> Optional[bool]
    """Best-effort: ask core whether a model is resident. None = unknown."""
    try:
        mod = importlib.import_module("%s.%s" % (settings.core_package(), modname))
    except Exception:
        return None
    for attr in attr_candidates:
        obj = getattr(mod, attr, None)
        if obj is None:
            continue
        if callable(obj) and not isinstance(obj, type):
            try:
                val = obj()
                if isinstance(val, bool):
                    return val
            except Exception:
                continue
        elif isinstance(obj, bool):
            return obj
        else:
            return True   # a populated module-level model object
    return None


def model_state():
    # type: () -> Dict[str, str]
    out = {}
    for key, modname, attrs in (
            ("whisper", "transcribe", ["is_loaded", "loaded", "_MODEL", "_WHISPER", "_model"]),
            ("phonemes", "phonemes", ["is_loaded", "loaded", "_SCORER", "_MODEL", "_model"])):
        probed = _probe_loaded(modname, attrs)
        if probed is None:
            probed = _MODEL_SEEN[key]
        out[key] = "loaded" if probed else "cold"
    # PhonemeScorer is a lazy singleton: look at its instance when exposed.
    if out["phonemes"] == "cold":
        try:
            mod = importlib.import_module("%s.phonemes" % settings.core_package())
            cls = getattr(mod, "PhonemeScorer", None)
            inst = getattr(cls, "_instance", None) if cls else None
            if inst is not None and getattr(inst, "loaded", getattr(inst, "_loaded", False)):
                out["phonemes"] = "loaded"
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# keys / clients (presence only — values are never logged)
# ---------------------------------------------------------------------------

CONNECTOR_ENV = os.path.expanduser("~/Downloads/wellsaid-connector/.env")


def _dotenv_has_key(path, name):
    # type: (str, str) -> bool
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:].lstrip()
                if line.startswith(name + "="):
                    return bool(line.partition("=")[2].strip().strip('"').strip("'"))
    except OSError:
        pass
    return False


def wellsaid_configured():
    # type: () -> bool
    if os.environ.get("WELLSAID_API_KEY", "").strip():
        return True
    return _dotenv_has_key(CONNECTOR_ENV, "WELLSAID_API_KEY")


_WS_CLIENT = None
_WS_LOCK = threading.Lock()


def wellsaid_client():
    """core.wellsaid.WellSaid() singleton. Raises CoreUnavailable / UpstreamError."""
    global _WS_CLIENT
    with _WS_LOCK:
        if _WS_CLIENT is None:
            mod = core_module("wellsaid")
            cls = _attr(mod, "WellSaid")
            _WS_CLIENT = cls()
        client = _WS_CLIENT
    if getattr(client, "ok", True) is False:
        raise UpstreamError("No WellSaid API key is configured on the server "
                            "(set WELLSAID_API_KEY).")
    return client


_VOICES_CACHE = {"at": 0.0, "voices": None}
_VOICES_TTL = 6 * 3600


def voices(force=False):
    # type: (bool) -> List[Dict[str, Any]]
    now = time.time()
    if (not force and _VOICES_CACHE["voices"] is not None
            and now - _VOICES_CACHE["at"] < _VOICES_TTL):
        return _VOICES_CACHE["voices"]
    client = wellsaid_client()
    try:
        raw = client.voices()
    except Exception as e:
        raise UpstreamError("WellSaid voice list failed: %s" % _safe_err(e))
    out = []
    for v in raw or []:
        if not isinstance(v, dict):
            continue
        try:
            sid = int(v.get("speaker_id", v.get("id")))
        except (TypeError, ValueError):
            continue
        out.append({"speaker_id": sid, "name": v.get("name"), "style": v.get("style"),
                    "accent": v.get("accent"), "gender": v.get("gender")})
    out.sort(key=lambda v: (v["name"] is None, v["name"] or ""))
    _VOICES_CACHE["voices"] = out
    _VOICES_CACHE["at"] = now
    return out


def _safe_err(e):
    # type: (Exception) -> str
    """Error text with anything that looks like a secret scrubbed."""
    s = "%s" % e
    s = re.sub(r"(?i)(x-api-key|api[_-]?key|authorization)\s*[:=]\s*\S+", r"\1: [redacted]", s)
    s = re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}", "[redacted]", s)
    return s[:600]


def render_script(text, speaker_id, model, out_mp3, progress=None):
    # type: (str, int, str, str, Any) -> str
    """Render `text` through WellSaid into `out_mp3` (SPENDS CREDITS).

    Texts over WellSaid's 1000-char cap are split at sentence boundaries into
    <=900-char chunks, rendered one by one, and concatenated with ffmpeg.
    """
    client = wellsaid_client()
    chunks = split_text(text, 900)
    parts = []
    tmpdir = os.path.join(os.path.dirname(out_mp3), "render")
    os.makedirs(tmpdir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        if progress:
            progress("rendering", float(i) / max(1, len(chunks)))
        try:
            try:
                audio = client.render(chunk, speaker_id=int(speaker_id), model=model)
            except TypeError:
                audio = client.render(chunk, int(speaker_id), model)
        except Exception as e:
            raise UpstreamError("WellSaid render failed: %s" % _safe_err(e))
        if not audio:
            raise UpstreamError("WellSaid returned empty audio.")
        p = os.path.join(tmpdir, "part%03d.mp3" % i)
        with open(p, "wb") as fh:
            fh.write(audio)
        parts.append(p)
    if len(parts) == 1:
        shutil.move(parts[0], out_mp3)
    else:
        concat_audio(parts, out_mp3)
    shutil.rmtree(tmpdir, ignore_errors=True)
    if progress:
        progress("rendering", 1.0)
    return out_mp3


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_text(text, limit):
    # type: (str, int) -> List[str]
    """Split at sentence boundaries into chunks of <= limit chars (hard-wrap
    any single sentence longer than the limit at a space)."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ""
    for sent in _SENT_SPLIT.split(text):
        while len(sent) > limit:
            cut = sent.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            piece, sent = sent[:cut].strip(), sent[cut:].strip()
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(piece)
        if not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= limit:
            cur = cur + " " + sent
        else:
            chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------

_FFMPEG = {"path": None}


def ffmpeg_path():
    # type: () -> str
    """Resolve ffmpeg: RETAKE_FFMPEG > core.media > PATH > imageio_ffmpeg
    (in-process, then via /usr/bin/python3 like heteronym-poc/common.py)."""
    if _FFMPEG["path"]:
        return _FFMPEG["path"]
    cands = []  # type: List[Optional[str]]
    cands.append(os.environ.get("RETAKE_FFMPEG"))
    try:
        media = importlib.import_module("%s.media" % settings.core_package())
        fn = getattr(media, "ffmpeg_path", None)
        if callable(fn):
            cands.append(fn())
    except Exception:
        pass
    cands.append(shutil.which("ffmpeg"))
    try:
        import imageio_ffmpeg  # type: ignore
        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    if os.path.exists("/usr/bin/python3"):
        try:
            out = subprocess.check_output(
                ["/usr/bin/python3", "-c", "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"],
                stderr=subprocess.DEVNULL, timeout=60)
            cands.append(out.decode("utf-8", "replace").strip())
        except Exception:
            pass
    for c in cands:
        if c and os.path.exists(c):
            _FFMPEG["path"] = c
            return c
    raise CoreUnavailable("ffmpeg was not found (set RETAKE_FFMPEG, install ffmpeg, "
                          "or pip install imageio-ffmpeg).")


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


class NotMedia(Exception):
    """The uploaded file is not decodable audio/video."""


def probe_media(path):
    # type: (str) -> Dict[str, Any]
    """Decode the first 0.2 s with ffmpeg; raise NotMedia when the file has no
    audio stream or does not decode. Returns {duration, has_audio, has_video}."""
    ff = ffmpeg_path()
    cmd = [ff, "-hide_banner", "-nostdin", "-i", path, "-t", "0.2", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except subprocess.TimeoutExpired:
        raise NotMedia("ffmpeg timed out probing the upload.")
    err = proc.stderr.decode("utf-8", "replace")
    has_audio = bool(re.search(r"Stream #\d+:\d+.*?: Audio:", err))
    has_video = bool(re.search(r"Stream #\d+:\d+.*?: Video:", err))
    if proc.returncode != 0 and not has_audio:
        tail = [l for l in err.strip().splitlines() if l.strip()][-1:] or ["no decodable streams"]
        raise NotMedia("Not a readable audio/video file (%s)." % tail[0].strip()[:160])
    if not has_audio:
        raise NotMedia("The file has no audio stream.")
    duration = None
    m = _DUR_RE.search(err)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return {"duration": duration, "has_audio": has_audio, "has_video": has_video}


def concat_audio(parts, out_path):
    # type: (List[str], str) -> str
    ff = ffmpeg_path()
    listfile = out_path + ".concat.txt"
    with open(listfile, "w", encoding="utf-8") as fh:
        for p in parts:
            fh.write("file '%s'\n" % p.replace("'", "'\\''"))
    cmd = [ff, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
           "-i", listfile, "-c", "copy", out_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    if proc.returncode != 0:   # fall back to a re-encode (mixed frame params)
        cmd = [ff, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
               "-i", listfile, "-c:a", "libmp3lame", "-q:a", "2", out_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if proc.returncode != 0:
            raise UpstreamError("ffmpeg could not join the rendered chunks: %s"
                                % proc.stderr.decode("utf-8", "replace")[-300:])
    try:
        os.remove(listfile)
    except OSError:
        pass
    return out_path


# ---------------------------------------------------------------------------
# engine calls used by the worker
# ---------------------------------------------------------------------------

def check_media(media_path, script, engine, work_dir, progress):
    # type: (str, Optional[str], str, str, Any) -> Dict[str, Any]
    fn = _attr(core_module("checker"), "check_media")
    res = fn(media_path, script=script, engine=engine, work_dir=work_dir, progress=progress)
    note_models_used()
    return res


def fix_occurrence(occ, speaker_id, model, work_dir, max_tries, progress):
    # type: (Dict[str, Any], int, str, str, int, Any) -> Dict[str, Any]
    fn = _attr(core_module("fixer"), "fix_occurrence")
    return fn(occ, speaker_id=int(speaker_id), model=model, work_dir=work_dir,
              max_tries=int(max_tries), progress=progress)


# ---------------------------------------------------------------------------
# lexicon
# ---------------------------------------------------------------------------

_LEX = {"data": None, "mtime": None}
_LEX_LOCK = threading.Lock()


def lexicon():
    # type: () -> Dict[str, Any]
    """Heteronym lexicon dict {word: {senses:[...], respell_candidates:[...]}}.
    Prefers core.lexicon.load(); falls back to data/heteronyms.json."""
    with _LEX_LOCK:
        path = str(settings.LEXICON_PATH)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
        if _LEX["data"] is not None and _LEX["mtime"] == mtime:
            return _LEX["data"]
        data = None
        try:
            mod = importlib.import_module("%s.lexicon" % settings.core_package())
            fn = getattr(mod, "load", None)
            if callable(fn):
                data = fn()
        except Exception:
            data = None
        if not isinstance(data, dict) or not data:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError) as e:
                raise CoreUnavailable("The heteronym lexicon could not be loaded: %s" % e)
        _LEX["data"] = data
        _LEX["mtime"] = mtime
        return data


def respell_of(sense):
    # type: (Dict[str, Any]) -> Optional[str]
    """Readable pronunciation hint = the trailing (…) in the sense desc."""
    m = re.search(r"\(([^()]*)\)\s*$", sense.get("desc", "") or "")
    if m:
        return m.group(1).strip()
    r = sense.get("respell")
    return r if r and r != "TODO" else None


def lexicon_entries():
    # type: () -> List[Dict[str, Any]]
    out = []
    for word, entry in sorted(lexicon().items()):
        senses = []
        for s in entry.get("senses", []):
            toks = list(s.get("ipa", []))
            senses.append({"id": s.get("id"), "pos": s.get("pos"), "desc": s.get("desc", ""),
                           "ipa": "".join(toks), "ipa_tokens": toks, "respell": respell_of(s)})
        out.append({"word": word, "senses": senses,
                    "respell_candidates": list(entry.get("respell_candidates", []) or [])})
    return out


# ---------------------------------------------------------------------------
# script pre-flight: find heteronyms in text + decide intent (no audio)
# ---------------------------------------------------------------------------

def check_text(text, engine=None):
    # type: (str, Optional[str]) -> Dict[str, Any]
    """Prefer `core.checker.check_text` when the engine provides it; otherwise
    run the ported detection + intent logic below (same rules as check.py)."""
    engine = engine or settings.default_engine()
    try:
        mod = core_module("checker")
        fn = getattr(mod, "check_text", None)
    except CoreUnavailable:
        fn = None
    if callable(fn):
        res = fn(text, engine=engine)
        if isinstance(res, dict) and "occurrences" in res:
            return _finish_text_result(text, res["occurrences"], res.get("engine") or engine, res)
        if isinstance(res, list):
            return _finish_text_result(text, res, engine, None)
    return _local_check_text(text, engine)


def _finish_text_result(text, occs, engine, extra):
    occs = [o for o in occs if isinstance(o, dict)]
    for i, o in enumerate(occs, 1):          # make core payloads schema-safe
        o.setdefault("id", "o%d" % i)
        o.setdefault("base", o.get("word") or "")
        o.setdefault("sentence", text)
        if not isinstance(o.get("char_start"), int) or not isinstance(o.get("char_end"), int):
            w = str(o.get("word") or "")
            cs = o["sentence"].lower().find(w.lower()) if w else -1
            o["char_start"], o["char_end"] = (cs, cs + len(w)) if cs >= 0 else (0, 0)
        if not o.get("intent_engine") and o.get("intended"):
            o["intent_engine"] = engine if engine in ("claude", "spacy") else None
    out = {"text": text, "count": len(occs), "occurrences": occs, "engine": engine,
           "summary": (extra or {}).get("summary") or [_summary_line(o) for o in occs]}
    return out


def _summary_line(o):
    # type: (Dict[str, Any]) -> str
    conf = o.get("intent_confidence")
    conf_s = (" %.2f" % conf) if isinstance(conf, (int, float)) else ""
    return '"%s" -> %s (%s) [%s%s]' % (o.get("word"), o.get("intended"),
                                       (o.get("intended_desc") or "").split(" (")[0],
                                       o.get("intent_engine") or "?", conf_s)


# --- ported from heteronym-poc/src/check.py (tokens, sentences, compounds) ---

PREFIXES = {"re": ["ɹ", "iː"], "un": ["ʌ", "n"], "pre": ["p", "ɹ", "iː"], "mis": ["m", "ɪ", "s"]}
PREFIX_RESPELL = {"re": "ree", "un": "un", "pre": "pree", "mis": "mis"}
VOICELESS = {"p", "t", "k", "f", "θ"}
SIBILANT = {"s", "z", "ʃ", "ʒ", "tʃ", "dʒ"}
_CORE_RE = re.compile(r"[0-9A-Za-zÀ-ɏ'’-]+")
_SENT_END = re.compile(r"[.!?]+[\"'”’)\]]*$")


def _core_span(raw):
    # type: (str) -> Tuple[int, int]
    ms = list(_CORE_RE.finditer(raw))
    if not ms:
        return (0, len(raw))
    return (ms[0].start(), ms[-1].end())


def _tokens_from_text(text):
    toks = []
    for m in re.finditer(r"\S+", text):
        a, b = _core_span(m.group(0))
        toks.append({"raw": m.group(0), "abs": m.start(), "norm": m.group(0)[a:b].lower()})
    return toks


def _build_sentences(tokens):
    """[{text, spans:[(tok_idx, cs, ce)], tok_abs:{tok_idx: abs_offset_of_raw}}]"""
    sentences, cur, parts, pos = [], [], [], 0
    cur_abs = {}

    def flush():
        if parts:
            sentences.append({"text": " ".join(parts), "spans": list(cur), "tok_abs": dict(cur_abs)})

    for i, t in enumerate(tokens):
        raw = t["raw"]
        if parts:
            pos += 1
        a, b = _core_span(raw)
        cur.append((i, pos + a, pos + b))
        cur_abs[i] = t["abs"] + a
        parts.append(raw)
        pos += len(raw)
        if _SENT_END.search(raw):
            flush()
            cur, parts, pos, cur_abs = [], [], 0, {}
    flush()
    return sentences


def _tail_phonemes(base_ipa, tail):
    last = base_ipa[-1] if base_ipa else ""
    if last in SIBILANT or tail == "es":
        return ["ᵻ", "z"]
    if last in VOICELESS:
        return ["s"]
    return ["z"]


def _tail_split(key, lex):
    if key.endswith("es") and key[:-2] in lex:
        return (key[:-2], "es")
    if key.endswith("s") and key[:-1] in lex:
        return (key[:-1], "s")
    return None


def _decompose(key, lex):
    if not key or key in lex:
        return None
    for prefix in PREFIXES:
        if key.startswith(prefix) and len(key) > len(prefix):
            rest = key[len(prefix):]
            if rest in lex:
                return (rest, prefix, None)
            t = _tail_split(rest, lex)
            if t:
                return (t[0], prefix, t[1])
    t = _tail_split(key, lex)
    if t:
        return (t[0], None, t[1])
    return None


def _build_candidates(base, prefix, tail, lex):
    cands = []
    for sense in lex[base].get("senses", []):
        ipa = list(sense.get("ipa", []))
        respell = respell_of(sense) or sense.get("id", "?")
        if prefix:
            ipa = PREFIXES[prefix] + ipa
            respell = PREFIX_RESPELL[prefix] + "-" + respell
        if tail:
            tp = _tail_phonemes(list(sense.get("ipa", [])), tail)
            ipa = ipa + tp
            respell += "-iz" if tp == ["ᵻ", "z"] else tp[0].upper()
        cands.append({"id": sense["id"], "pos": sense.get("pos"), "desc": sense.get("desc", ""),
                      "ipa": ipa, "respell": respell})
    return cands


def _build_pattern(forms):
    forms = sorted({f.lower() for f in forms if f}, key=len, reverse=True)
    return re.compile(r"\b(?:%s)\b" % "|".join(re.escape(f) for f in forms), re.IGNORECASE)


def _detect_direct(sentence_text, lex, pattern):
    out = []
    for m in pattern.finditer(sentence_text):
        w = m.group(0).lower()
        if w in lex:
            out.append((w, m.start(), m.end()))
    return out


def _collect_occurrences(text, lex):
    tokens = _tokens_from_text(text)
    pattern = _build_pattern(lex.keys())
    occs = []
    for sent in _build_sentences(tokens):
        direct = {}
        for (w, cs0, ce0) in _detect_direct(sent["text"], lex, pattern):
            for (ti, cs, ce) in sent["spans"]:
                if cs <= cs0 and ce0 <= ce and tokens[ti]["norm"] == w:
                    direct[ti] = w
        for (ti, cs, ce) in sent["spans"]:
            tok = tokens[ti]
            if ti in direct:
                base, prefix, tail = direct[ti], None, None
            else:
                dec = _decompose(re.sub(r"[-'’]", "", tok["norm"]), lex)
                if not dec:
                    continue
                base, prefix, tail = dec
            cands = _build_candidates(base, prefix, tail, lex)
            if len(cands) < 2:
                continue
            abs_start = sent["tok_abs"][ti]
            occs.append({"word": sent["text"][cs:ce], "base": base, "prefix": prefix, "tail": tail,
                         "sentence": sent["text"], "char_start": cs, "char_end": ce,
                         "text_start": abs_start, "text_end": abs_start + (ce - cs),
                         "candidates": cands})
    return occs


# --- intent engines: core.intent if it exposes analyze.py's names, else local ---

_NLP = {"nlp": None}
_NLP_LOCK = threading.Lock()


def _spacy_nlp():
    with _NLP_LOCK:
        if _NLP["nlp"] is None:
            try:
                import spacy  # heavy
                _NLP["nlp"] = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
            except ImportError:
                raise CoreUnavailable("spaCy is not installed; the intent engine is unavailable.")
            except OSError:
                raise CoreUnavailable("spaCy model en_core_web_sm is not installed.")
        return _NLP["nlp"]


def _spacy_pick(doc, cs, ce, senses):
    span = doc.char_span(cs, ce, alignment_mode="expand")
    tok = span[0] if span is not None and len(span) > 0 else None
    if tok is None:
        for t in doc:
            if t.idx <= cs < t.idx + len(t.text):
                tok = t
                break
    pos = tok.pos_ if tok is not None else None
    matches = [s for s in senses if s.get("pos") == pos]
    if len(matches) == 1:
        return {"sense_id": matches[0].get("id"), "confidence": 0.75}
    return {"sense_id": senses[0].get("id"), "confidence": 0.5}


def _intent_module():
    try:
        return importlib.import_module("%s.intent" % settings.core_package())
    except Exception:
        return None


def _run_spacy(items):
    mod = _intent_module()
    fn = getattr(mod, "run_spacy_engine", None) if mod else None
    if callable(fn):
        fn(items)
        if all("spacy" in it for it in items):
            return
    nlp = _spacy_nlp()
    cache = {}
    for it in items:
        doc = cache.get(it["text"])
        if doc is None:
            doc = nlp(it["text"])
            cache[it["text"]] = doc
        it["spacy"] = _spacy_pick(doc, it["char_start"], it["char_end"], it["senses"])


def _run_claude(items, model):
    # type: (List[Dict[str, Any]], str) -> bool
    """Fill item["claude"]; return True when Claude ran. Uses core.intent's
    engine when present (same names as analyze.py); otherwise a faithful port."""
    if not settings.anthropic_configured():
        return False
    mod = _intent_module()
    resolve = getattr(mod, "_resolve_claude_client", None) if mod else None
    run = getattr(mod, "run_claude_engine", None) if mod else None
    if callable(resolve) and callable(run):
        client = resolve(required=False)
        if client is None:
            return False
        run(client, model, items)
        return True
    try:
        import anthropic
    except ImportError:
        return False
    try:
        client = anthropic.Anthropic()
    except Exception:
        return False
    for b in range(0, len(items), 25):
        batch = items[b:b + 25]
        try:
            res = _claude_batch(client, model, batch)
        except Exception as e:
            log.warning("claude batch failed; spaCy fallback for %d items: %s", len(batch), _safe_err(e))
            res = {}
        for j, it in enumerate(batch):
            it["claude"] = res.get(j)
    return True


_CLAUDE_SYSTEM = ("You are a careful English pronunciation editor. Decide which sense of a "
                  "heteronym is intended from its sentence context.")


def _claude_batch(client, model, batch):
    lines = ["For each numbered item, pick the intended sense of the word marked «like this».",
             "Respond with ONLY a JSON array, one object per item, in this exact shape:",
             '[{"i": 1, "sense_id": "<one of the listed ids>", "confidence": 0.9}, ...]',
             '"confidence" is your certainty from 0.0 to 1.0. No other keys, no commentary.', ""]
    for n, it in enumerate(batch, 1):
        t, cs, ce = it["text"], it["char_start"], it["char_end"]
        lines.append("%d. %s" % (n, t[:cs] + "«" + t[cs:ce] + "»" + t[ce:]))
        for s in it["senses"]:
            lines.append("   - %s: %s" % (s.get("id"), s.get("desc", "")))
    messages = [{"role": "user", "content": "\n".join(lines)}]

    def call(msgs):
        resp = client.messages.create(model=model, max_tokens=2000, system=_CLAUDE_SYSTEM, messages=msgs)
        return "".join(getattr(b, "text", "") or "" for b in (getattr(resp, "content", None) or []))

    def parse(raw):
        s = (raw or "").strip()
        s = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        a, z = s.find("["), s.rfind("]")
        if a == -1 or z <= a:
            raise ValueError("no JSON array in reply")
        parsed = json.loads(s[a:z + 1])
        out = {}
        for el in parsed if isinstance(parsed, list) else []:
            if not isinstance(el, dict):
                continue
            try:
                i = int(el.get("i"))
            except (TypeError, ValueError):
                continue
            if not 1 <= i <= len(batch):
                continue
            sid = el.get("sense_id")
            if sid not in [s.get("id") for s in batch[i - 1]["senses"]]:
                continue
            try:
                conf = max(0.0, min(1.0, float(el.get("confidence", 0.5))))
            except (TypeError, ValueError):
                conf = 0.5
            out[i - 1] = {"sense_id": sid, "confidence": conf}
        if not out:
            raise ValueError("no usable items in reply")
        return out

    raw = call(messages)
    try:
        return parse(raw)
    except ValueError:
        raw = call(messages + [{"role": "assistant", "content": (raw or "(empty)")[:4000]},
                               {"role": "user", "content": "That was not the required strict JSON. "
                                                           "Reply again with ONLY the JSON array."}])
        return parse(raw)


CLAUDE_MODEL = "claude-opus-5-5"


def _local_check_text(text, engine):
    # type: (str, str) -> Dict[str, Any]
    lex = lexicon()
    occs = _collect_occurrences(text, lex)
    used = "none"
    if occs:
        items = [{"text": o["sentence"], "word": o["base"], "char_start": o["char_start"],
                  "char_end": o["char_end"], "senses": lex[o["base"]]["senses"]} for o in occs]
        _run_spacy(items)
        used = "spacy"
        claude_ran = False
        if engine in ("claude", "both"):
            claude_ran = _run_claude(items, CLAUDE_MODEL)
            if engine == "claude" and not claude_ran:
                raise UpstreamError("The Claude intent engine is not available on this server "
                                    "(ANTHROPIC_API_KEY unset); use engine=spacy or both.")
        if claude_ran and any(it.get("claude") for it in items):
            used = "claude"
        for o, it in zip(occs, items):
            chosen = it.get("claude") or it["spacy"]
            o["intended"] = chosen["sense_id"]
            o["intent_confidence"] = chosen["confidence"]
            o["intent_engine"] = "claude" if it.get("claude") else "spacy"
    out = []
    for i, o in enumerate(occs, 1):
        cand = {c["id"]: c for c in o["candidates"]}
        intended = cand.get(o.get("intended"))
        out.append({
            "id": "o%d" % i, "word": o["word"], "base": o["base"], "sentence": o["sentence"],
            "char_start": o["char_start"], "char_end": o["char_end"],
            "text_start": o["text_start"], "text_end": o["text_end"],
            "prefix": o.get("prefix"), "tail": o.get("tail"),
            "intended": o.get("intended"),
            "intended_desc": intended["desc"] if intended else None,
            "intended_ipa": "".join(intended["ipa"]) if intended else None,
            "intended_respell": intended["respell"] if intended else None,
            "respell_candidates": list(lex[o["base"]].get("respell_candidates", []) or []),
            "intent_engine": o.get("intent_engine"),
            "intent_confidence": o.get("intent_confidence"),
            "candidates": {c["id"]: "".join(c["ipa"]) for c in o["candidates"]},
            "senses": [{"id": c["id"], "pos": c.get("pos"), "desc": c.get("desc", ""),
                        "ipa": "".join(c["ipa"]), "ipa_tokens": list(c["ipa"]),
                        "respell": c.get("respell")} for c in o["candidates"]],
        })
    return _finish_text_result(text, out, used, None)
