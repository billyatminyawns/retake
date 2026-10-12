# -*- coding: utf-8 -*-
"""End-to-end media check (port of heteronym-poc check.py as a library).

    check_media(media_path, *, script=None, engine="both", pad=0.1,
                work_dir, progress=None) -> dict

Pipeline (nothing is rendered; no WellSaid calls):
  1. ffmpeg extracts a 16 kHz mono wav into work_dir/audio16k.wav.
  2. core.transcribe gives word timestamps (cached in work_dir/transcript.json).
  3. Heteronym occurrences are found in the transcript (or in `script`,
     aligned to the transcript) via intent.detect_occurrences, plus
     prefix/suffix compounds such as "re-records" and "reads".
  4. The intended sense comes from core.intent (Claude when a key is
     present, spaCy fallback) on the sentence containing the word.
  5. Each word window is CTC-scored against every candidate pronunciation
     and free-decoded (greedy argmax) as IPA on a wider window. Flag when
     the best candidate is not the intended one, or when the free decode
     does not contain the intended pronunciation (fuzzy containment).
  6. Word + sentence clips are cut for the UI, a confidence (high/med/low)
     is derived from the margin and rule agreement, and a summary is built.

`progress(stage, frac)` is called with the UI's stage names:
transcribing -> finding heteronyms -> deciding intent -> listening -> done.

CLI (dev): python -m core.checker MEDIA [--script FILE] [--engine spacy|claude|both]
           [--work-dir DIR] [--json]
"""
import argparse
import difflib
import json
import math
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

from core import intent, lexicon, media

DEFAULT_PAD = 0.1
FREE_DECODE_EXTRA_PAD = 0.3   # seconds added on top of pad for the free decode window
SENTENCE_PAD = 0.15           # seconds around the sentence clip
PREFIXES = {
    "re": ["ɹ", "iː"],
    "un": ["ʌ", "n"],
    "pre": ["p", "ɹ", "iː"],
    "mis": ["m", "ɪ", "s"],
}
PREFIX_RESPELL = {"re": "ree", "un": "un", "pre": "pree", "mis": "mis"}
VOICELESS = {"p", "t", "k", "f", "θ"}
SIBILANT = {"s", "z", "ʃ", "ʒ", "tʃ", "dʒ"}
POS_LABEL = lexicon.POS_LABEL
IPA_VARIANTS = {"ᵻ": "ɪ", "ə": "ɪ", "ɔːɹ": "ɔɹ", "oːɹ": "ɔɹ"}
STAGES = ("transcribing", "finding heteronyms", "deciding intent", "listening", "done")

CHECK_LOCK = threading.RLock()  # one check at a time per process (models are not re-entrant)


class CheckError(RuntimeError):
    """Bad input or a pipeline stage failed."""


# ---------------------------------------------------------------------------
# Tokens + sentences (shared by transcript and script text)
# ---------------------------------------------------------------------------

_CORE_RE = re.compile(r"[^\W_]", re.UNICODE)
_SENT_END = re.compile(r"[.!?]+[\"'”’)\]]*$")


def _core_span(raw):
    # type: (str) -> Tuple[int, int]
    """[start, end) of the alnum core of a raw token ("content." -> content)."""
    ms = list(_CORE_RE.finditer(raw))
    if not ms:
        return (0, len(raw))
    return (ms[0].start(), ms[-1].end())


def _norm(raw):
    # type: (str) -> str
    a, b = _core_span(raw)
    return raw[a:b].lower()


def tokens_from_whisper(words):
    # type: (List[Dict[str, Any]]) -> List[Dict[str, Any]]
    """Whisper word pieces -> tokens. Pieces that continue the previous word
    (no leading space and a hyphen join, e.g. " re" + "-records") are merged."""
    toks = []  # type: List[Dict[str, Any]]
    for w in words:
        raw = w.get("w") or ""
        stripped = raw.strip()
        if not stripped:
            continue
        joins = (toks and not raw[:1].isspace()
                 and (stripped.startswith("-") or toks[-1]["raw"].endswith("-")))
        if joins:
            prev = toks[-1]
            prev["raw"] = prev["raw"] + stripped
            prev["e"] = max(prev["e"], float(w["e"]))
            prev["p"] = min(prev["p"], float(w.get("p", 1.0)))
        else:
            toks.append({"raw": stripped, "s": float(w["s"]), "e": float(w["e"]),
                         "p": float(w.get("p", 1.0))})
    for t in toks:
        t["norm"] = _norm(t["raw"])
    return toks


def tokens_from_text(text):
    # type: (str) -> List[Dict[str, Any]]
    """Whitespace tokens of a script; `pos` is the token's offset in `text`."""
    toks = []
    for m in re.finditer(r"\S+", text or ""):
        toks.append({"raw": m.group(0), "s": None, "e": None, "p": None,
                     "norm": _norm(m.group(0)), "pos": m.start()})
    return toks


def build_sentences(tokens):
    # type: (List[Dict[str, Any]]) -> List[Dict[str, Any]]
    """Group tokens into sentences. Each sentence: {text, spans:[(tok_idx, cs, ce)]}
    where (cs, ce) is the token's alnum core span inside `text`."""
    sentences = []
    cur = []  # type: List[Tuple[int, int, int]]
    parts = []  # type: List[str]
    pos = 0

    def flush():
        if parts:
            sentences.append({"text": " ".join(parts), "spans": list(cur)})

    for i, t in enumerate(tokens):
        raw = t["raw"]
        if parts:
            pos += 1  # the joining space
        a, b = _core_span(raw)
        cur.append((i, pos + a, pos + b))
        parts.append(raw)
        pos += len(raw)
        if _SENT_END.search(raw):
            flush()
            cur, parts, pos = [], [], 0
    flush()
    return sentences


# ---------------------------------------------------------------------------
# Heteronym occurrences incl. prefix/suffix compounds
# ---------------------------------------------------------------------------

def _tail_phonemes(base_ipa, tail):
    # type: (List[str], str) -> List[str]
    last = base_ipa[-1] if base_ipa else ""
    if last in SIBILANT or tail == "es":
        return ["ᵻ", "z"]
    if last in VOICELESS:
        return ["s"]
    return ["z"]


def _tail_split(key, lex):
    # type: (str, Dict[str, Any]) -> Optional[Tuple[str, str]]
    if key.endswith("es") and key[:-2] in lex:
        return (key[:-2], "es")
    if key.endswith("s") and key[:-1] in lex:
        return (key[:-1], "s")
    return None


def decompose(key, lex):
    # type: (str, Dict[str, Any]) -> Optional[Tuple[str, Optional[str], Optional[str]]]
    """Split a non-lexicon token into (base, prefix, tail) when its tail is a
    lexicon form: rerecords -> (record, re, s); reads -> (read, None, s)."""
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


def _with_tail_respell(respell, tp):
    # type: (str, List[str]) -> str
    if tp == ["ᵻ", "z"]:
        return respell + "-iz"
    last = re.split(r"[-\s]+", respell)[-1] if respell else ""
    letter = tp[0]
    return respell + (letter.upper() if last.isupper() else letter.lower())


def build_candidates(base, prefix, tail, lex):
    # type: (str, Optional[str], Optional[str], Dict[str, Any]) -> List[Dict[str, Any]]
    """Candidate pronunciations for an occurrence: one per sense of `base`,
    with the prefix/tail phonemes applied. `respell` is the WellSaid-format
    respelling for that sense (lexicon respell_candidates), compounded."""
    cands = []
    for sense in lex[base].get("senses", []):
        ipa = list(sense.get("ipa", []))
        respell = lexicon.respell_for_sense(base, sense.get("id"), lex) or lexicon.desc_respell(sense)
        if prefix:
            ipa = PREFIXES[prefix] + ipa
            respell = PREFIX_RESPELL[prefix] + "-" + respell
        if tail:
            tp = _tail_phonemes(list(sense.get("ipa", [])), tail)
            ipa = ipa + tp
            respell = _with_tail_respell(respell, tp)
        cands.append({"id": sense["id"], "pos": sense.get("pos"),
                      "desc": sense.get("desc", ""), "ipa": ipa, "respell": respell})
    return cands


def find_occurrences(sentence, tokens, lex, pattern):
    # type: (Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Any) -> List[Dict[str, Any]]
    """Occurrences in one sentence: direct lexicon forms via
    intent.detect_occurrences (accepted only when the match is a whole token),
    then compound/suffix decomposition for the remaining tokens."""
    direct = {}  # tok_idx -> lexicon word
    for d in intent.detect_occurrences(sentence["text"], lex, pattern):
        for (ti, cs, ce) in sentence["spans"]:
            if cs <= d["char_start"] and d["char_end"] <= ce and tokens[ti]["norm"] == d["word"]:
                direct[ti] = d["word"]
    occs = []
    for (ti, cs, ce) in sentence["spans"]:
        tok = tokens[ti]
        if ti in direct:
            base, prefix, tail = direct[ti], None, None
        else:
            key = re.sub(r"[-'’]", "", tok["norm"])
            dec = decompose(key, lex)
            if not dec:
                continue
            base, prefix, tail = dec
        cands = build_candidates(base, prefix, tail, lex)
        if len(cands) < 2:
            continue
        occ = {
            "tok_idx": ti,
            "word": sentence["text"][cs:ce],
            "base": base,
            "prefix": prefix,
            "tail": tail,
            "sentence": sentence["text"],
            "sentence_tok_idx": [s[0] for s in sentence["spans"]],
            "char_start": cs,
            "char_end": ce,
            "candidates": cands,
            "t_start": tok["s"],
            "t_end": tok["e"],
        }
        if tok.get("pos") is not None:  # script text: absolute offsets too
            a, _b = _core_span(tok["raw"])
            occ["text_start"] = tok["pos"] + a
            occ["text_end"] = tok["pos"] + a + (ce - cs)
        occs.append(occ)
    return occs


def collect_occurrences(tokens, lex):
    # type: (List[Dict[str, Any]], Dict[str, Any]) -> List[Dict[str, Any]]
    pattern = intent.build_pattern(lex.keys())
    occs = []
    for sent in build_sentences(tokens):
        occs.extend(find_occurrences(sent, tokens, lex, pattern))
    return occs


def align_script(script_tokens, transcript_tokens):
    # type: (List[Dict[str, Any]], List[Dict[str, Any]]) -> Dict[int, int]
    """Map script token index -> transcript token index by normalized-token
    matching (equal blocks, plus same-length replace blocks positionally)."""
    a = [t["norm"] for t in script_tokens]
    b = [t["norm"] for t in transcript_tokens]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    mapping = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and (i2 - i1) == (j2 - j1)):
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
    return mapping


# ---------------------------------------------------------------------------
# Intended sense
# ---------------------------------------------------------------------------

def decide_intent(occs, lex, engine="both", model=None, log=None):
    # type: (List[Dict[str, Any]], Dict[str, Any], str, Optional[str], Any) -> Dict[str, Any]
    items = []
    for o in occs:
        items.append({"text": o["sentence"], "word": o["base"],
                      "char_start": o["char_start"], "char_end": o["char_end"],
                      "senses": lex[o["base"]]["senses"]})
    if not items:
        return {"engine_used": "none", "claude_ok": True, "note": None}
    model = model or os.environ.get("RETAKE_CLAUDE_MODEL", "").strip() or intent.DEFAULT_MODEL
    info = intent.decide(items, engine=engine, model=model, log=log)
    for o, it in zip(occs, items):
        o["intended"] = it["final"]["sense_id"]
        o["intent_confidence"] = it["final"]["confidence"]
        o["intent_engine"] = it["final"]["engine"]
        o["spacy"] = it.get("spacy")
        o["claude"] = it.get("claude")
    return info


# ---------------------------------------------------------------------------
# CTC scoring + free decode + flag rules
# ---------------------------------------------------------------------------

def norm_ipa_tokens(tokens):
    # type: (List[str]) -> List[str]
    """Collapse the usual espeak variants so containment checks are fair."""
    return [IPA_VARIANTS.get(t, t) for t in tokens]


def substring_distance(needle, hay):
    # type: (List[str], List[str]) -> int
    """Minimum Levenshtein distance between `needle` and ANY contiguous
    sub-sequence of `hay` (semi-global alignment: free start/end in hay)."""
    n, m = len(needle), len(hay)
    if n == 0:
        return 0
    if m == 0:
        return n
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if needle[i - 1] == hay[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return min(prev)


def free_decode_slack(n_phonemes):
    # type: (int) -> int
    """Edits tolerated before the free decode counts as NOT containing the
    intended pronunciation: one per four intended phonemes, at least one."""
    return max(1, n_phonemes // 4)


def score_window(wav16k, t_start, t_end, candidates, pad, clip_path, scorer=None):
    # type: (str, float, float, List[Dict[str, Any]], float, str, Any) -> Dict[str, Any]
    """CTC-score every candidate on the tight window [t_start-pad, t_end+pad]
    (widened once by 0.15 s when too short) and free-decode a wider window.

    candidates: [{"id", "ipa": [tokens]}]. Returns {scores, detected, margin,
    undecidable, free_tokens, free_decode, clip, wide_clip, pad_used}.
    """
    from core.phonemes import PhonemeScorer
    sc = scorer or PhonemeScorer()
    sc.load()
    cand_ids = {}
    for c in candidates:
        cand_ids[c["id"]] = sc.ids(c["ipa"], c["id"])
    max_ext = max(2 * len(ids) + 1 for ids in cand_ids.values())

    eff_pad = float(pad)
    logp = None
    for _attempt in range(2):
        media.cut_clip(wav16k, t_start, t_end, eff_pad, clip_path)
        logp = sc.logprobs(media.load_wav(clip_path))
        if logp.shape[0] >= max_ext:
            break
        eff_pad += 0.15

    # Free decode on a WIDER window: wav2vec2 needs acoustic context on both
    # sides, and whisper word boundaries routinely clip final consonants.
    wide_path = clip_path[:-4] + "_wide.wav"
    media.cut_clip(wav16k, t_start, t_end, eff_pad + FREE_DECODE_EXTRA_PAD, wide_path)
    free_tokens = sc.greedy_tokens(sc.logprobs(media.load_wav(wide_path)))
    out = {"free_tokens": free_tokens, "free_decode": "".join(free_tokens),
           "clip": clip_path, "wide_clip": wide_path, "pad_used": round(eff_pad, 3),
           "free_decode_pad": round(eff_pad + FREE_DECODE_EXTRA_PAD, 3)}

    if logp.shape[0] < max_ext:
        out.update({"scores": None, "detected": None, "margin": None, "undecidable": True})
        return out
    scores = {}
    for cid, ids in cand_ids.items():
        from core.phonemes import ctc_forced_loglik
        ll = ctc_forced_loglik(logp, ids, sc.blank_id)
        scores[cid] = ll / len(ids)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not math.isfinite(ranked[0][1]):
        out.update({"scores": {k: None for k in scores}, "detected": None,
                    "margin": None, "undecidable": True})
        return out
    margin = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 0.0
    out.update({
        "scores": {k: (round(v, 4) if math.isfinite(v) else None) for k, v in scores.items()},
        "detected": ranked[0][0],
        "margin": round(margin, 4) if math.isfinite(margin) else None,
        "undecidable": False,
    })
    return out


def judge(occ):
    # type: (Dict[str, Any]) -> None
    """Apply the flag rules in place: sets flag, reason, free_decode_distance,
    free_decode_slack, rule_ctc, rule_free."""
    cand = {c["id"]: c for c in occ["candidates"]}
    intended = occ.get("intended")
    reasons = []
    occ["rule_ctc"] = False
    occ["rule_free"] = False
    if occ.get("undecidable"):
        occ["flag"] = False
        occ["reason"] = "undecidable (clip too short for CTC scoring)"
        return
    detected = occ.get("detected")
    if detected != intended:
        occ["rule_ctc"] = True
        reasons.append("best candidate is %s (%s), not intended %s (%s)"
                       % (detected, cand[detected]["respell"], intended,
                          cand[intended]["respell"]))
    want = norm_ipa_tokens(cand[intended]["ipa"])
    got = norm_ipa_tokens(occ.get("free_tokens") or [])
    dist = substring_distance(want, got)
    slack = free_decode_slack(len(want))
    occ["free_decode_distance"] = dist
    occ["free_decode_slack"] = slack
    if dist > slack:
        occ["rule_free"] = True
        reasons.append("free decode /%s/ is %d edit(s) from intended /%s/ (allowed %d)"
                       % (occ.get("free_decode") or "", dist,
                          "".join(cand[intended]["ipa"]), slack))
    occ["flag"] = bool(reasons)
    occ["reason"] = "; ".join(reasons) if reasons else "ok"


def confidence(occ):
    # type: (Dict[str, Any]) -> Tuple[str, str]
    """(level, why) for the verdict, from the CTC margin, agreement between the
    two rules, and how sure the intent engine was.

      high: both rules agree and |margin| >= 0.75 (clear acoustic separation)
      med:  both rules agree with |margin| >= 0.25, or they disagree with a
            large margin (>= 1.0)
      low:  undecidable, tiny margin, or rules disagree on a small margin
    An uncertain intent (confidence < 0.6, i.e. a spaCy POS tie) lowers the
    level one step: the verdict can only be as good as the intended sense.
    """
    if occ.get("undecidable") or occ.get("margin") is None:
        return "low", "clip too short to score"
    m = abs(float(occ["margin"]))
    agree = bool(occ.get("rule_ctc")) == bool(occ.get("rule_free"))
    if agree and m >= 0.75:
        level, why = "high", "CTC and free decode agree, margin %.2f" % m
    elif agree and m >= 0.25:
        level, why = "med", "CTC and free decode agree, margin %.2f" % m
    elif (not agree) and m >= 1.0:
        level, why = "med", "rules disagree but CTC margin is large (%.2f)" % m
    elif agree:
        level, why = "low", "CTC and free decode agree but margin is small (%.2f)" % m
    else:
        level, why = "low", "CTC and free decode disagree (margin %.2f)" % m
    ic = occ.get("intent_confidence")
    if ic is not None and float(ic) < 0.6:
        level = {"high": "med", "med": "low", "low": "low"}[level]
        why += "; intended sense uncertain (%s %.2f)" % (occ.get("intent_engine") or "intent", float(ic))
    return level, why


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------

def _sentence_bounds(occ, ttoks, mapping=None):
    # type: (Dict[str, Any], List[Dict[str, Any]], Optional[Dict[int, int]]) -> Tuple[float, float]
    idxs = occ.get("sentence_tok_idx") or []
    times = []
    for i in idxs:
        j = mapping.get(i) if mapping is not None else i
        if j is None or j >= len(ttoks):
            continue
        t = ttoks[j]
        if t.get("s") is not None:
            times.append((float(t["s"]), float(t["e"])))
    if times:
        return min(a for a, _ in times), max(b for _, b in times)
    return max(0.0, float(occ["t_start"]) - 1.5), float(occ["t_end"]) + 1.5


def _occurrence_record(o, idx):
    # type: (Dict[str, Any], int) -> Dict[str, Any]
    cand = {c["id"]: c for c in o["candidates"]}
    intended = o["intended"]
    detected = o.get("detected")
    rec = {
        "id": "o%d" % idx,
        "t_start": round(float(o["t_start"]), 3),
        "t_end": round(float(o["t_end"]), 3),
        "word": o["word"],
        "sentence": o["sentence"],
        "char_start": o["char_start"],
        "char_end": o["char_end"],
        "base": o["base"],
        "prefix": o.get("prefix"),
        "tail": o.get("tail"),
        "intended": intended,
        "intended_desc": cand[intended]["desc"],
        "intended_pos": POS_LABEL.get(cand[intended].get("pos"), "?"),
        "intended_ipa": "".join(cand[intended]["ipa"]),
        "intended_respell": cand[intended]["respell"],
        "detected": detected,
        "detected_desc": cand[detected]["desc"] if detected else None,
        "detected_pos": POS_LABEL.get(cand[detected].get("pos"), "?") if detected else None,
        "detected_ipa": "".join(cand[detected]["ipa"]) if detected else None,
        "detected_respell": cand[detected]["respell"] if detected else None,
        "scores": o.get("scores"),
        "margin": o.get("margin"),
        "free_decode": o.get("free_decode"),
        "free_decode_distance": o.get("free_decode_distance"),
        "free_decode_slack": o.get("free_decode_slack"),
        "undecidable": bool(o.get("undecidable")),
        "flag": bool(o["flag"]),
        "reason": o["reason"],
        "confidence": o.get("confidence"),
        "confidence_reason": o.get("confidence_reason"),
        "clip_wav": o.get("clip_wav"),
        "score_clip_wav": o.get("score_clip_wav"),
        "sentence_wav": o.get("sentence_wav"),
        "sentence_t_start": o.get("sentence_t_start"),
        "sentence_t_end": o.get("sentence_t_end"),
        "candidates": {c["id"]: "".join(c["ipa"]) for c in o["candidates"]},
        "candidate_tokens": {c["id"]: list(c["ipa"]) for c in o["candidates"]},
        "candidate_respells": {c["id"]: c["respell"] for c in o["candidates"]},
        "candidate_descs": {c["id"]: c["desc"] for c in o["candidates"]},
        "intent_engine": o.get("intent_engine"),
        "intent_confidence": o.get("intent_confidence"),
        "pad_used": o.get("pad_used"),
    }
    if o.get("heard_as") is not None:
        rec["heard_as"] = o["heard_as"]
    return rec


def summary_line(rec):
    # type: (Dict[str, Any]) -> str
    if rec.get("detected") and rec["detected"] != rec["intended"]:
        heard = "%s (%s)" % (rec.get("detected_respell") or rec["detected"],
                             rec.get("detected_pos") or "?")
    else:
        heard = "/%s/ (free decode)" % (rec.get("free_decode") or "?")
    margin = ("%.1f" % rec["margin"]) if rec.get("margin") is not None else "n/a"
    return "[%.1fs] %s — intended %s (%s), heard %s, margin %s, confidence %s" % (
        rec["t_start"], rec["word"], rec.get("intended_respell") or rec["intended"],
        rec.get("intended_pos") or "?", heard, margin, rec.get("confidence") or "?")


def build_summary(records, duration=None):
    # type: (List[Dict[str, Any]], Optional[float]) -> List[str]
    flags = [r for r in records if r.get("flag")]
    head = "%d heteronym occurrence%s, %d flagged" % (
        len(records), "" if len(records) == 1 else "s", len(flags))
    if duration:
        head += " in %.0f s of audio" % duration
    if not records:
        head += " — no heteronyms to check"
    elif not flags:
        head += " — every heteronym was pronounced as intended"
    lines = [head]
    for r in flags:
        lines.append(summary_line(r))
    return lines


def verdict(records):
    # type: (List[Dict[str, Any]]) -> str
    """One word for the header: clean | issues | empty."""
    if not records:
        return "empty"
    return "issues" if any(r.get("flag") for r in records) else "clean"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def check_text(text, engine="both", model=None):
    # type: (str, str, Optional[str]) -> Dict[str, Any]
    """Script pre-flight (no audio): every heteronym occurrence in `text` with
    its intended sense, respelling and candidates. Offsets are into `text`."""
    if not text or not text.strip():
        return {"text": text or "", "occurrences": [], "summary": ["no text"]}
    lex = lexicon.load()
    toks = tokens_from_text(text)
    occs = collect_occurrences(toks, lex)
    info = decide_intent(occs, lex, engine=engine, model=model)
    out = []
    for i, o in enumerate(occs, 1):
        cand = {c["id"]: c for c in o["candidates"]}
        it = o["intended"]
        out.append({
            "id": "o%d" % i,
            "word": o["word"],
            "base": o["base"],
            "prefix": o.get("prefix"),
            "tail": o.get("tail"),
            "sentence": o["sentence"],
            "char_start": o["char_start"],
            "char_end": o["char_end"],
            "text_start": o.get("text_start"),
            "text_end": o.get("text_end"),
            "intended": it,
            "intended_desc": cand[it]["desc"],
            "intended_pos": POS_LABEL.get(cand[it].get("pos"), "?"),
            "intended_ipa": "".join(cand[it]["ipa"]),
            "intended_respell": cand[it]["respell"],
            "candidates": {c["id"]: "".join(c["ipa"]) for c in o["candidates"]},
            "candidate_tokens": {c["id"]: list(c["ipa"]) for c in o["candidates"]},
            "candidate_respells": {c["id"]: c["respell"] for c in o["candidates"]},
            "candidate_descs": {c["id"]: c["desc"] for c in o["candidates"]},
            "intent_engine": o.get("intent_engine"),
            "intent_confidence": o.get("intent_confidence"),
        })
    n = len(out)
    summary = ["%d heteronym occurrence%s found" % (n, "" if n == 1 else "s")]
    if info.get("note"):
        summary.append(info["note"])
    return {"text": text, "occurrences": out, "summary": summary,
            "intent": info}


def check_media(media_path, *, script=None, engine="both", pad=DEFAULT_PAD, work_dir,
                progress=None, model=None, log=None):
    # type: (...) -> Dict[str, Any]
    """Run one media file through the full check. See the module docstring.

    Returns {"media", "duration", "transcript": {"text","words"}, "occurrences":
    [Occurrence...], "flags": int, "summary": [str...], "verdict", "wav16k",
    "intent", "engine"}. Every path is absolute and under `work_dir`.
    """
    if not media_path or not os.path.isfile(media_path):
        raise CheckError("media file not found: %s" % media_path)
    if engine not in ("claude", "spacy", "both"):
        raise CheckError("engine must be claude, spacy or both")
    try:
        pad = float(pad)
    except (TypeError, ValueError):
        raise CheckError("pad must be a number")
    if pad < 0:
        raise CheckError("pad must be >= 0")
    if not work_dir:
        raise CheckError("work_dir is required")
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    def _p(stage, frac):
        if progress:
            try:
                progress(stage, float(frac))
            except Exception:
                pass

    def _log(msg):
        if log:
            try:
                log(msg)
            except Exception:
                pass

    with CHECK_LOCK:
        return _check_media_locked(media_path, script, engine, pad, work_dir, _p, _log, model)


def _check_media_locked(media_path, script, engine, pad, work_dir, _p, _log, model):
    from core import transcribe
    lex = lexicon.load()

    # 1. audio
    _p("transcribing", 0.02)
    wav16k = os.path.join(work_dir, "audio16k.wav")
    try:
        media.extract_wav16k(media_path, wav16k)
    except media.MediaError as e:
        raise CheckError("could not extract audio: %s" % e)
    duration = media.wav_duration(wav16k)
    if duration <= 0.05:
        raise CheckError("the file has no audible audio")

    # 2. transcript
    _p("transcribing", 0.08)
    try:
        tdata = transcribe.transcribe_cached(wav16k, os.path.join(work_dir, "transcript.json"))
    except transcribe.TranscribeError as e:
        raise CheckError("transcription failed: %s" % e)
    _p("transcribing", 0.45)
    ttoks = tokens_from_whisper(tdata.get("words", []))
    transcript = {"text": tdata.get("text", ""), "words": tdata.get("words", []),
                  "backend": tdata.get("backend")}
    result = {"media": os.path.abspath(media_path), "duration": round(duration, 3),
              "transcript": transcript, "occurrences": [], "flags": 0, "summary": [],
              "verdict": "empty", "wav16k": wav16k, "engine": engine,
              "script_used": bool(script and script.strip()), "unaligned": 0,
              "intent": None}
    if not ttoks:
        result["summary"] = ["no speech found in the audio"]
        _p("done", 1.0)
        return result

    # 3. occurrences
    _p("finding heteronyms", 0.47)
    mapping = None
    unaligned = 0
    if script and script.strip():
        stoks = tokens_from_text(script)
        mapping = align_script(stoks, ttoks)
        occs = collect_occurrences(stoks, lex)
        kept = []
        for o in occs:
            ti = mapping.get(o["tok_idx"])
            if ti is None:
                unaligned += 1
                _log("script word %r in %r not aligned to the transcript; skipped"
                     % (o["word"], o["sentence"][:60]))
                continue
            o["t_start"], o["t_end"] = ttoks[ti]["s"], ttoks[ti]["e"]
            o["heard_as"] = ttoks[ti]["raw"]
            kept.append(o)
        occs = kept
        result["unaligned"] = unaligned
    else:
        occs = collect_occurrences(ttoks, lex)
    if not occs:
        result["summary"] = build_summary([], duration)
        _p("done", 1.0)
        return result

    # 4. intent
    _p("deciding intent", 0.5)
    try:
        result["intent"] = decide_intent(occs, lex, engine=engine, model=model, log=_log)
    except intent.IntentError as e:
        raise CheckError("intent engine failed: %s" % e)

    # 5. listen
    _p("listening", 0.55)
    from core.phonemes import PhonemeScorer, PhonemeError
    try:
        sc = PhonemeScorer().load()
    except PhonemeError as e:
        raise CheckError("phoneme model unavailable: %s" % e)
    clips_dir = os.path.join(work_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    n = len(occs)
    records = []
    for i, o in enumerate(occs, 1):
        safe = re.sub(r"\W+", "_", o["word"]).strip("_") or "word"
        clip_path = os.path.join(clips_dir, "o%d_%s.wav" % (i, safe))
        try:
            o.update(score_window(wav16k, o["t_start"], o["t_end"], o["candidates"], pad,
                                  clip_path, scorer=sc))
        except Exception as e:
            _log("scoring failed for %r at %.2fs: %s" % (o["word"], o["t_start"], e))
            o.update({"scores": None, "detected": None, "margin": None,
                      "free_decode": None, "free_tokens": [], "undecidable": True,
                      "clip": clip_path if os.path.exists(clip_path) else None,
                      "wide_clip": None})
        judge(o)
        o["confidence"], o["confidence_reason"] = confidence(o)
        # listening clips for the UI: the wide word clip + the whole sentence
        o["clip_wav"] = o.get("wide_clip") or o.get("clip")
        o["score_clip_wav"] = o.get("clip")
        ss, se = _sentence_bounds(o, ttoks, mapping)
        sent_path = os.path.join(clips_dir, "o%d_sentence.wav" % i)
        try:
            media.cut_clip(wav16k, ss, se, SENTENCE_PAD, sent_path)
            o["sentence_wav"] = sent_path
        except media.MediaError as e:
            _log("sentence clip failed for o%d: %s" % (i, e))
            o["sentence_wav"] = None
        o["sentence_t_start"] = round(max(0.0, ss - SENTENCE_PAD), 3)
        o["sentence_t_end"] = round(min(duration, se + SENTENCE_PAD), 3)
        rec = _occurrence_record(o, i)
        records.append(rec)
        _log("%s [%6.2fs] %-14s intended=%-14s detected=%-14s margin=%s free=/%s/"
             % ("FLAG" if rec["flag"] else "ok  ", rec["t_start"], rec["word"],
                rec["intended"], rec["detected"],
                ("%.2f" % rec["margin"]) if rec["margin"] is not None else "-",
                rec["free_decode"] or ""))
        _p("listening", 0.55 + 0.42 * (i / float(n)))

    result["occurrences"] = records
    result["flags"] = sum(1 for r in records if r["flag"])
    result["summary"] = build_summary(records, duration)
    if result["intent"] and result["intent"].get("note"):
        result["summary"].append(result["intent"]["note"])
    result["verdict"] = verdict(records)
    try:
        with open(os.path.join(work_dir, "check.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
    except OSError:
        pass
    _p("done", 1.0)
    return result


# ---------------------------------------------------------------------------
# CLI (dev)
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m core.checker",
                                 description="Run a media file through the Retake check.")
    ap.add_argument("media")
    ap.add_argument("--script", metavar="FILE", help="intended script text file")
    ap.add_argument("--engine", choices=["claude", "spacy", "both"], default="both")
    ap.add_argument("--pad", type=float, default=DEFAULT_PAD)
    ap.add_argument("--work-dir", default=None,
                    help="default: ./retake-work/<media basename>")
    ap.add_argument("--json", action="store_true", help="print the full result JSON")
    args = ap.parse_args(argv)
    script = None
    if args.script:
        if not os.path.isfile(args.script):
            print("error: script file not found: %s" % args.script, file=sys.stderr)
            return 2
        with open(args.script, "r", encoding="utf-8", errors="replace") as fh:
            script = fh.read()
    work_dir = args.work_dir or os.path.join(
        os.getcwd(), "retake-work", os.path.splitext(os.path.basename(args.media))[0])

    def prog(stage, frac):
        sys.stderr.write("  %-20s %3d%%\n" % (stage, int(frac * 100)))

    try:
        res = check_media(args.media, script=script, engine=args.engine, pad=args.pad,
                          work_dir=work_dir, progress=prog,
                          log=lambda m: sys.stderr.write("  " + m + "\n"))
    except CheckError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        print("")
        for line in res["summary"]:
            print(line)
        print("")
        for r in res["occurrences"]:
            print(("FLAG " if r["flag"] else "ok   ") + summary_line(r))
        print("\nwork dir: %s" % work_dir)
    return 3 if res["flags"] else 0


if __name__ == "__main__":
    sys.exit(main())
