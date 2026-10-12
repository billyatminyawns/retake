# -*- coding: utf-8 -*-
"""Intended-sense engines (port of heteronym-poc analyze.py).

Finds heteronym occurrences in text (case-insensitive, word-boundary regex
over the lexicon's surface forms) and decides the INTENDED sense of each
occurrence from context:

  * spacy  — en_core_web_sm POS of the occurrence token -> the sense whose
             `pos` matches (exactly one match: confidence 0.75; tie/none:
             first sense, 0.5). Always runs; it is the floor.
  * claude — Anthropic Messages API (stable SDK surface only, no
             `temperature`), batches of 25, strict-JSON reply, one parse
             retry. Takes precedence when it returns a verdict.

Library behaviour: no sys.exit, no prints. Errors raise IntentError; a
missing Anthropic key degrades `engine="both"` to spaCy (and is reported in
each item's `final.engine`). `engine="claude"` without a key raises.

Item shape consumed by `decide()`:
    {"text": sentence, "word": lexicon key, "char_start", "char_end",
     "senses": [sense dicts]}
and filled in place with "spacy", "claude" (or None) and
"final": {"sense_id", "confidence", "engine"}.
"""
import json
import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Pattern

DEFAULT_MODEL = "claude-opus-5-5"
CLAUDE_BATCH_SIZE = 25
ZSHENV = os.path.expanduser("~/.zshenv")

_CLAUDE_SYSTEM = (
    "You are a terse expert linguist specializing in English heteronyms. "
    "Given sentences where one word is marked with « guillemets », you decide "
    "which of the listed senses is intended in context. You respond with strict JSON "
    "only: a single JSON array, no prose, no markdown, no code fences."
)


class IntentError(RuntimeError):
    """Engine unavailable or misconfigured."""


# ---------------------------------------------------------------------------
# occurrence detection
# ---------------------------------------------------------------------------

def build_pattern(surface_forms):
    # type: (Iterable[str]) -> Pattern
    """One case-insensitive word-boundary alternation over all surface forms
    (longest first, so overlapping forms prefer the longer match)."""
    forms = sorted({f.lower() for f in surface_forms if f}, key=len, reverse=True)
    if not forms:
        raise ValueError("no surface forms to build a pattern from")
    joined = "|".join(re.escape(f) for f in forms)
    return re.compile(r"\b(?:%s)\b" % joined, re.IGNORECASE)


def detect_occurrences(text, lexicon, pattern=None):
    # type: (str, Dict[str, Any], Optional[Pattern]) -> List[Dict[str, Any]]
    """Scan `text` for lexicon surface forms (case-insensitive, word-boundary).

    Returns [{"word", "char_start", "char_end"}, ...] ordered by char_start.
    `word` is the lowercase lexicon key; offsets are 0-based [start, end).
    """
    if pattern is None:
        pattern = build_pattern(lexicon.keys())
    out = []
    for m in pattern.finditer(text or ""):
        word = m.group(0).lower()
        if word in lexicon:
            out.append({"word": word, "char_start": m.start(), "char_end": m.end()})
    return out


# ---------------------------------------------------------------------------
# spaCy engine
# ---------------------------------------------------------------------------

_NLP = None
_NLP_LOCK = threading.Lock()


def _load_spacy():
    global _NLP
    if _NLP is not None:
        return _NLP
    with _NLP_LOCK:
        if _NLP is not None:
            return _NLP
        try:
            import spacy  # heavy import, kept out of module import time
        except ImportError:
            raise IntentError("spaCy is not installed in this environment")
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        except OSError:
            raise IntentError("spaCy model 'en_core_web_sm' is not installed "
                              "(python -m spacy download en_core_web_sm)")
        _NLP = nlp
    return _NLP


def spacy_status():
    # type: () -> str
    return "loaded" if _NLP is not None else "cold"


def _token_at(doc, char_start, char_end):
    span = doc.char_span(char_start, char_end, alignment_mode="expand")
    if span is not None and len(span) > 0:
        return span[0]
    for tok in doc:
        if tok.idx <= char_start < tok.idx + len(tok.text):
            return tok
    return None


def spacy_pick(doc, char_start, char_end, senses):
    # type: (Any, int, int, List[Dict[str, Any]]) -> Dict[str, Any]
    """The sense whose `pos` matches the occurrence token's coarse POS.
    Exactly one match -> confidence 0.75; tie or none -> first sense, 0.5."""
    tok = _token_at(doc, char_start, char_end)
    pos = tok.pos_ if tok is not None else None
    matches = [s for s in senses if s.get("pos") == pos]
    if len(matches) == 1:
        return {"sense_id": matches[0].get("id"), "confidence": 0.75, "pos": pos}
    return {"sense_id": senses[0].get("id"), "confidence": 0.5, "pos": pos}


def run_spacy(items):
    # type: (List[Dict[str, Any]]) -> None
    nlp = _load_spacy()
    doc_cache = {}  # type: Dict[str, Any]
    with _NLP_LOCK:
        for item in items:
            doc = doc_cache.get(item["text"])
            if doc is None:
                doc = nlp(item["text"])
                doc_cache[item["text"]] = doc
            item["spacy"] = spacy_pick(doc, item["char_start"], item["char_end"],
                                       item["senses"])


# ---------------------------------------------------------------------------
# Claude engine
# ---------------------------------------------------------------------------

def _read_env_file(path, name):
    # type: (str, str) -> Optional[str]
    """`export NAME=value` / `NAME=value` from a shell/dotenv file. None on failure."""
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
                k, _, v = line.partition("=")
                if k.strip() != name:
                    continue
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    except OSError:
        pass
    return None


def anthropic_key():
    # type: () -> Optional[str]
    """ANTHROPIC_API_KEY from the environment, else from ~/.zshenv (this Mac).
    Never log the result."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    return _read_env_file(ZSHENV, "ANTHROPIC_API_KEY")


def anthropic_available():
    # type: () -> bool
    if not anthropic_key():
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def anthropic_client(required=False):
    """An Anthropic client, or None when the engine must degrade.
    required=True raises IntentError with a one-line reason."""
    msg = None
    key = anthropic_key()
    if not key:
        msg = "ANTHROPIC_API_KEY is not set"
    else:
        try:
            import anthropic
        except ImportError:
            msg = "the 'anthropic' package is not installed"
        else:
            try:
                return anthropic.Anthropic(api_key=key)
            except Exception as e:
                msg = "could not create the Anthropic client (%s)" % e
    if required:
        raise IntentError("%s; cannot run the claude engine" % msg)
    return None


def _build_claude_prompt(batch):
    # type: (List[Dict[str, Any]]) -> str
    lines = [
        "For each numbered item, pick the intended sense of the word marked "
        "«like this».",
        "Respond with ONLY a JSON array, one object per item, in this exact shape:",
        '[{"i": 1, "sense_id": "<one of the listed ids>", "confidence": 0.9}, ...]',
        '"confidence" is your certainty from 0.0 to 1.0. No other keys, no commentary.',
        "",
    ]
    for n, item in enumerate(batch, 1):
        text = item["text"]
        cs, ce = item["char_start"], item["char_end"]
        marked = text[:cs] + "«" + text[cs:ce] + "»" + text[ce:]
        lines.append("%d. %s" % (n, marked))
        for sense in item["senses"]:
            lines.append("   - %s: %s" % (sense.get("id"), sense.get("desc", "")))
    return "\n".join(lines)


def _response_text(resp):
    # type: (Any) -> str
    parts = []
    for block in getattr(resp, "content", None) or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


def _claude_call(client, model, messages):
    # type: (Any, str, List[Dict[str, Any]]) -> str
    """One messages.create call with a single retry on transient API errors."""
    last = None
    for attempt in range(2):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2000,
                system=_CLAUDE_SYSTEM,
                messages=messages,
            )
            return _response_text(resp)
        except Exception as e:  # old SDKs raise varied exception types
            last = e
            if attempt == 0:
                time.sleep(2.0)
    raise IntentError("anthropic request failed: %s" % last)


def _extract_json_array(raw):
    # type: (str) -> Any
    """Tolerant parse: strip code fences, grab first '[' .. last ']'."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON array found in model reply")
    return json.loads(s[start:end + 1])


def _validate_batch_reply(parsed, batch):
    # type: (Any, List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]
    if not isinstance(parsed, list):
        raise ValueError("model reply is not a JSON array")
    out = {}  # type: Dict[int, Dict[str, Any]]
    for el in parsed:
        if not isinstance(el, dict):
            continue
        try:
            i = int(el.get("i"))
        except (TypeError, ValueError):
            continue
        if not 1 <= i <= len(batch):
            continue
        item = batch[i - 1]
        sense_id = el.get("sense_id")
        if sense_id not in [s.get("id") for s in item["senses"]]:
            continue
        try:
            conf = float(el.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        out[i - 1] = {"sense_id": sense_id, "confidence": max(0.0, min(1.0, conf))}
    if not out:
        raise ValueError("no usable items in model reply")
    return out


def claude_classify_batch(client, model, batch):
    # type: (Any, str, List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]
    """Classify one batch (<=25 occurrences); one retry on parse failure."""
    messages = [{"role": "user", "content": _build_claude_prompt(batch)}]
    raw = _claude_call(client, model, messages)
    try:
        return _validate_batch_reply(_extract_json_array(raw), batch)
    except ValueError:
        retry_messages = messages + [
            {"role": "assistant", "content": (raw or "(empty)")[:4000]},
            {"role": "user", "content": "That was not the required strict JSON. "
                                        "Reply again with ONLY the JSON array, "
                                        "nothing else."},
        ]
        raw = _claude_call(client, model, retry_messages)
        return _validate_batch_reply(_extract_json_array(raw), batch)


def run_claude(client, model, items, log=None):
    # type: (Any, str, List[Dict[str, Any]], Any) -> bool
    """Fill item["claude"] for every item (None where no verdict).
    Returns True when every batch succeeded."""
    ok = True
    total = (len(items) + CLAUDE_BATCH_SIZE - 1) // CLAUDE_BATCH_SIZE
    for b in range(total):
        batch = items[b * CLAUDE_BATCH_SIZE:(b + 1) * CLAUDE_BATCH_SIZE]
        try:
            res = claude_classify_batch(client, model, batch)
        except Exception as e:
            ok = False
            res = {}
            if log:
                log("claude batch %d/%d failed (%s); spaCy verdicts kept"
                    % (b + 1, total, e))
        for j, item in enumerate(batch):
            item["claude"] = res.get(j)
    return ok


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------

def decide(items, engine="both", model=DEFAULT_MODEL, log=None):
    # type: (List[Dict[str, Any]], str, str, Any) -> Dict[str, Any]
    """Run the engines over `items` in place. Returns a small status dict:
    {"engine_used": "claude+spacy"|"spacy", "claude_ok": bool, "note": str|None}."""
    if engine not in ("claude", "spacy", "both"):
        raise ValueError("engine must be claude, spacy or both")
    if not items:
        return {"engine_used": "none", "claude_ok": True, "note": None}
    client = None
    note = None
    use_claude = engine in ("claude", "both")
    if use_claude:
        try:
            client = anthropic_client(required=(engine == "claude"))
        except IntentError:
            raise
        if client is None:
            use_claude = False
            note = "no Anthropic key: intent decided by spaCy POS only"
            if log:
                log(note)
    run_spacy(items)
    claude_ok = True
    if use_claude:
        claude_ok = run_claude(client, model, items, log=log)
    else:
        for it in items:
            it["claude"] = None
    for it in items:
        chosen = it["claude"] if it.get("claude") is not None else it["spacy"]
        it["final"] = {"sense_id": chosen["sense_id"],
                       "confidence": float(chosen["confidence"]),
                       "engine": "claude" if it.get("claude") is not None else "spacy"}
    return {"engine_used": "claude+spacy" if use_claude else "spacy",
            "claude_ok": claude_ok, "note": note}
