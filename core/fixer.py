# -*- coding: utf-8 -*-
"""Fix a flagged occurrence: re-render the sentence through WellSaid with the
model-appropriate pronunciation mechanism, then verify the take with the
same phoneme scorer the checker used.

    fix_occurrence(occ, *, speaker_id, model, work_dir, max_tries=3, progress=None) -> dict

Strategy per model (tested WellSaid facts, 10/11/26):
  caruso  → inline <respell value="ruh-KORD">record</respell>; the respelling
            is the lexicon's WellSaid-format candidate for the intended sense
            (prefixed/pluralised for compounds), then GET /tts/respelling_
            suggestions alternatives, then the sense's own desc respelling.
  preview → spelling-as-pronunciation (see preview_spellings): the sense's
            field-verified spellings ("lyve"), the real word with a respelled
            prefix ("ree-recordz"), the WellSaid respelling lowercased
            ("ree-ruh-kordz"), the description hint, then simplified vowels.
            Preview garbles <respell> and phonetic libraries, honours plain
            spelling.
Each try renders the FULL sentence (occ["sentence"]), locates the word in the
take from WellSaid's word timing (whisper fallback), CTC-scores it against
every candidate pronunciation and greedy-decodes it; the loop stops at the
first verified take with a CTC margin >= STRONG_MARGIN, otherwise keeps
trying (max_tries) and returns the best verified take. Renders SPEND CREDITS and are never run by selftest.

Returns {"occurrence_id", "tries": [{"text","take_mp3","heard_as","detected",
"margin","verified", ...}], "best": take|None, "verified": bool, ...}.
"""
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from core import checker, lexicon, media
from core.wellsaid import WellSaid, WellSaidError, respell_tag, _xml_escape

FIX_MODELS = ("caruso", "preview")
FIX_LOCK = threading.RLock()
_SIMPLIFY = (("ee", "i"), ("ih", "i"), ("uh", "u"), ("eh", "e"))
# A verified take whose CTC margin (best candidate minus runner-up, per
# phoneme) is below this is kept but does not stop the loop: later tries may
# verify more convincingly, and the best verified margin wins. Field data
# (10/11/26, preview 13.3): "ree-recordz" verified at 0.27, "ree-ruh-kordz"
# at 0.88; "lyve" at 2.70.
STRONG_MARGIN = 0.8


class FixError(RuntimeError):
    """Bad input, unsupported model, or no WellSaid key."""


# ---------------------------------------------------------------------------
# Attempt planning (pure; selftest covers it)
# ---------------------------------------------------------------------------

# Preview spellings proven by phoneme decode (10/10-11/26). Keys are WellSaid
# respellings lowercased. Preview reads CAPITALS as spelled-out letters, so no
# variant may contain uppercase.
_OVERRIDES = {
    "lyv": ["lyve", "lahyv"], "lihv": ["liv", "lihv"],
    "lyvz": ["lyvz", "lahyvz"], "lihvz": ["livs", "lihvz"],
}


def spelling_variants(respell):
    # type: (str) -> List[str]
    """Preview-model spellings for a WellSaid respelling, in try order:
    proven overrides, the lowercased respelling, a silent-e form when the
    last syllable is a long y/i vowel closed by one consonant ("lyv" -> "lyve"),
    then a simplified-vowel form. Never the raw respelling: its capitals are
    read as letter names by the preview model."""
    r = (respell or "").strip()
    if not r:
        return []
    low = r.lower()
    out = []  # type: List[str]
    for v in _OVERRIDES.get(low, []):
        if v not in out:
            out.append(v)
    if low not in out:
        out.append(low)
    last = re.split(r"[-\s]+", low)[-1]
    if re.search(r"[yi][bdfgklmnprstvz]$", last) and not low.endswith("e"):
        se = low + "e"
        if se not in out:
            out.append(se)
    simp = low
    for a, b in _SIMPLIFY:
        simp = simp.replace(a, b)
    if simp not in out:
        out.append(simp)
    return out


_LETTER_NAMES = ("ɛl", "waɪ", "viː", "ɑːɹ", "ɛs", "tiː", "ɛm", "ɛn", "ɛf", "keɪ", "dʒeɪ",
                 "piː", "biː", "siː", "diː", "dʒiː", "eɪtʃ", "kjuː", "ɛks", "ziː", "zɛd")


def _norm_ipa(s):
    # type: (str) -> str
    s = (s or "").replace(" ", "")
    for a, b in (("ᵻ", "ɪ"), ("ə", "ɪ"), ("ɐ", "ʌ"), ("ɔːɹ", "ɔɹ"), ("oːɹ", "ɔɹ"), ("oʊ", "oʊ")):
        s = s.replace(a, b)
    return s


def heard_contains(intended_ipa, free_decode):
    # type: (str, str) -> bool
    """True when the intended phoneme string occurs contiguously in the free
    decode (variant-normalized, with and without length marks)."""
    want, got = _norm_ipa(intended_ipa), _norm_ipa(free_decode)
    if not want or not got:
        return False
    if want in got:
        return True
    return want.replace("ː", "") in got.replace("ː", "")


def spelled_out(free_decode, intended_ipa):
    # type: (str, str) -> bool
    """Preview read a spelling as letter names (e.g. "LYV" -> ɛl-waɪ-viː):
    two or more letter-name phoneme strings and no contiguous intended match."""
    d = (free_decode or "").replace(" ", "")
    hits = sum(1 for ln in _LETTER_NAMES if ln in d)
    return hits >= 2 and not heard_contains(intended_ipa, d)


def preview_spellings(occ, respell, sense=None, base_ipa=None):
    # type: (Dict[str, Any], str, Optional[Dict[str, Any]], Optional[List[str]]) -> List[str]
    """Preview spellings for an occurrence, in try order.

    Compounds (re-records): first keep the real base word and respell only the
    prefix + tail ("ree-recordz" — proven on 4 voices), then the lowercased
    full respelling ("ree-ruh-kordz"), then the lexicon desc respelling
    compounded ("ree-rih-kordz"). Plain words: spelling_variants(respell)
    (proven overrides first, never capitals).
    """
    out = []  # type: List[str]
    prefix, tail = occ.get("prefix"), occ.get("tail")
    base = (occ.get("base") or occ.get("word") or "").lower()
    if prefix or tail:
        ipa = list(base_ipa or (sense or {}).get("ipa") or [])
        tail_sp = ""
        if tail:
            tp = checker._tail_phonemes(ipa, tail) if ipa else ["z"]
            tail_sp = "ez" if "ᵻ" in tp or "ɪ" in tp else ("z" if "z" in tp else "s")
        keep = (checker.PREFIX_RESPELL.get(prefix, prefix) + "-" if prefix else "") + base + tail_sp
        out.append(keep.lower())
        variants = spelling_variants(respell)
        if variants and variants[0] not in out:
            out.append(variants[0])          # the lowercased WellSaid respelling
        if sense:
            d = lexicon.desc_respell(sense)
            if d:
                if prefix:
                    d = checker.PREFIX_RESPELL.get(prefix, prefix) + "-" + d
                if tail:
                    d = checker._with_tail_respell(d, checker._tail_phonemes(ipa, tail) if ipa else ["z"])
                d = d.lower()
                if d not in out:
                    out.append(d)               # the lexicon's own respelling, compounded
        for v in variants[1:]:                  # simplified vowels last: "ri-" was heard as "rih-" live
            if v not in out:
                out.append(v)
        return out
    return spelling_variants(respell)


def _stress_pattern(respell):
    # type: (str) -> str
    parts = re.split(r"[-\s]+", respell or "")
    return "".join("S" if p.isupper() else "u" for p in parts if p)


def _rank_suggestions(suggestions, intended_respell, other_respells):
    # type: (List[str], str, List[str]) -> List[str]
    """Order API respelling suggestions by closeness to the intended sense's
    respelling, dropping ones that read like another sense."""
    import difflib
    want = (intended_respell or "").lower()
    others = [o.lower() for o in other_respells if o]
    scored = []
    for s in suggestions:
        sl = s.lower()
        if sl == want:
            continue
        score = difflib.SequenceMatcher(None, want, sl).ratio()
        if _stress_pattern(s) == _stress_pattern(intended_respell):
            score += 0.15
        if any(difflib.SequenceMatcher(None, o, sl).ratio() > score for o in others):
            continue
        scored.append((score, s))
    scored.sort(key=lambda t: -t[0])
    return [s for _, s in scored]


def _candidates(occ):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    """Candidate list [{"id","ipa","respell","desc","pos"}] from the occurrence
    (candidate_tokens when present, else rebuilt from the lexicon)."""
    lex = lexicon.load()
    toks = occ.get("candidate_tokens")
    respells = occ.get("candidate_respells") or {}
    descs = occ.get("candidate_descs") or {}
    base = (occ.get("base") or occ.get("word") or "").lower()
    if isinstance(toks, dict) and toks:
        out = []
        for sid, ipa in toks.items():
            sense = lexicon.sense_by_id(base, sid, lex) or {}
            out.append({"id": sid, "ipa": list(ipa),
                        "respell": respells.get(sid) or lexicon.respell_for_sense(base, sid, lex),
                        "desc": descs.get(sid) or sense.get("desc", ""),
                        "pos": sense.get("pos")})
        return out
    if base not in lex:
        raise FixError("occurrence word %r is not in the lexicon" % base)
    return checker.build_candidates(base, occ.get("prefix"), occ.get("tail"), lex)


def _splice(sentence, cs, ce, replacement, escape_rest=True):
    # type: (str, int, int, str, bool) -> str
    before, after = sentence[:cs], sentence[ce:]
    if escape_rest:
        before, after = _xml_escape(before), _xml_escape(after)
    return before + replacement + after


def plan_attempts(occ, model, max_tries=3, client=None):
    # type: (Dict[str, Any], str, int, Optional[WellSaid]) -> List[Dict[str, Any]]
    """The ordered render texts for an occurrence: [{"strategy","respell","text"}].
    `client` (optional) is used for GET /tts/respelling_suggestions on caruso."""
    if model not in FIX_MODELS:
        raise FixError("model must be one of %s" % ", ".join(FIX_MODELS))
    sentence = occ.get("sentence") or ""
    word = occ.get("word") or ""
    cs, ce = occ.get("char_start"), occ.get("char_end")
    if not sentence or not word or cs is None or ce is None or sentence[cs:ce] != word:
        # tolerate a stale span: find the word in the sentence
        m = re.search(r"(?<![\w])" + re.escape(word) + r"(?![\w])", sentence, re.IGNORECASE)
        if not m:
            raise FixError("cannot locate %r in the sentence" % word)
        cs, ce = m.start(), m.end()
    cands = _candidates(occ)
    intended = occ.get("intended")
    cand = {c["id"]: c for c in cands}
    if intended not in cand:
        raise FixError("intended sense %r is not a candidate" % intended)
    intended_respell = occ.get("intended_respell") or cand[intended]["respell"]
    others = [c["respell"] for c in cands if c["id"] != intended]

    respells = [intended_respell]
    if model == "caruso":
        if client is not None and client.ok:
            try:
                sugg = client.respell_suggestions(word)
            except Exception:
                sugg = []
            for s in _rank_suggestions(sugg, intended_respell, others):
                if s not in respells:
                    respells.append(s)
        base = (occ.get("base") or word).lower()
        sense = lexicon.sense_by_id(base, intended)
        if sense:
            d = lexicon.desc_respell(sense)
            if d:
                if occ.get("prefix"):
                    d = checker.PREFIX_RESPELL[occ["prefix"]] + "-" + d
                if occ.get("tail"):
                    tp = checker._tail_phonemes(list(sense.get("ipa", [])), occ["tail"])
                    d = checker._with_tail_respell(d, tp)
                if d not in respells:
                    respells.append(d)
        attempts = [{"strategy": "respell", "respell": r,
                     "text": _splice(sentence, cs, ce, respell_tag(word, r))}
                    for r in respells]
    else:
        base = (occ.get("base") or word).lower()
        sense = lexicon.sense_by_id(base, intended)
        variants = preview_spellings(occ, intended_respell, sense=sense,
                                     base_ipa=(sense or {}).get("ipa"))
        attempts = [{"strategy": "spelling", "respell": v,
                     "text": _splice(sentence, cs, ce, _xml_escape(v))}
                    for v in variants]
    seen = set()
    out = []
    for a in attempts:
        if a["text"] in seen:
            continue
        seen.add(a["text"])
        a["char_start"], a["char_end"] = cs, ce
        out.append(a)
    return out[:max(1, int(max_tries))]


# ---------------------------------------------------------------------------
# Locating the word in a take
# ---------------------------------------------------------------------------

def _plain_tokens(text):
    # type: (str) -> List[Dict[str, Any]]
    """Whitespace tokens of a render text with markup stripped."""
    plain = re.sub(r"<[^>]+>", "", text)
    plain = plain.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return checker.tokens_from_text(plain)


def _target_index(plain_tokens, sentence, cs):
    # type: (List[Dict[str, Any]], str, int) -> int
    """Index of the token holding sentence[cs] (same index in the plain render
    text, since the replacement stays inside its token)."""
    sent_toks = checker.tokens_from_text(sentence)
    for i, t in enumerate(sent_toks):
        if t["pos"] <= cs < t["pos"] + len(t["raw"]):
            return min(i, len(plain_tokens) - 1)
    return 0


def locate_span(plain_tokens, k, timing):
    # type: (List[Dict[str, Any]], int, List[Dict[str, Any]]) -> Optional[Tuple[float, float, str]]
    """(start, end, how) of plain token k inside `timing` [{word,start,end}].
    Direct alignment first; else the gap between the neighbouring aligned
    words (covers respellings the API split on hyphens)."""
    if not timing:
        return None
    ttoks = [{"raw": str(t["word"]), "norm": checker._norm(str(t["word"])),
              "s": float(t["start"]), "e": float(t["end"])} for t in timing]
    mapping = checker.align_script(plain_tokens, ttoks)
    j = mapping.get(k)
    if j is not None:
        return ttoks[j]["s"], ttoks[j]["e"], "timing"
    prev_end = 0.0
    for i in range(k - 1, -1, -1):
        if i in mapping:
            prev_end = ttoks[mapping[i]]["e"]
            break
    next_start = ttoks[-1]["e"]
    for i in range(k + 1, len(plain_tokens)):
        if i in mapping:
            next_start = ttoks[mapping[i]]["s"]
            break
    if next_start > prev_end:
        return prev_end, next_start, "gap"
    return None


# ---------------------------------------------------------------------------
# Fix loop
# ---------------------------------------------------------------------------

def fix_occurrence(occ, *, speaker_id, model, work_dir, max_tries=3, progress=None,
                   client=None, log=None, pad=checker.DEFAULT_PAD):
    # type: (...) -> Dict[str, Any]
    """Re-render occ["sentence"] with the pronunciation fix and verify it.
    See the module docstring. Raises FixError for bad input / no key; a failed
    render is recorded on its try and the loop moves on."""
    if model not in FIX_MODELS:
        raise FixError("model must be one of %s" % ", ".join(FIX_MODELS))
    try:
        speaker_id = int(speaker_id)
    except (TypeError, ValueError):
        raise FixError("speaker_id must be an integer")
    if not work_dir:
        raise FixError("work_dir is required")
    occ_id = occ.get("id") or "occ"
    ws = client or WellSaid()
    if not ws.ok:
        raise FixError("no WellSaid API key (set WELLSAID_API_KEY)")

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

    with FIX_LOCK:
        attempts = plan_attempts(occ, model, max_tries=max_tries, client=ws)
        cands = _candidates(occ)
        cands_ipa = {c["id"]: c.get("ipa") for c in cands}
        intended = occ["intended"]
        fix_dir = os.path.join(os.path.abspath(work_dir), "fixes", re.sub(r"\W+", "_", str(occ_id)))
        os.makedirs(fix_dir, exist_ok=True)
        result = {"occurrence_id": occ_id, "word": occ.get("word"), "intended": intended,
                  "intended_respell": occ.get("intended_respell"), "model": model,
                  "speaker_id": speaker_id, "strategy": attempts[0]["strategy"] if attempts else None,
                  "tries": [], "best": None, "verified": False, "summary": ""}
        n = len(attempts)
        for i, att in enumerate(attempts, 1):
            _p("rendering", (i - 1) / float(n))
            t = {"n": i, "text": att["text"], "respell": att["respell"],
                 "strategy": att["strategy"], "take_mp3": None, "take_wav": None,
                 "word_wav": None, "heard_as": None, "detected": None, "margin": None,
                 "scores": None, "verified": False, "span": None, "located": None,
                 "reason": None, "error": None}
            result["tries"].append(t)
            try:
                take = _render(ws, att["text"], speaker_id, model)
                ext = take.get("ext") or "mp3"
                take_path = os.path.join(fix_dir, "try%d.%s" % (i, ext))
                media.write_bytes(take_path, take["audio_bytes"])
                t["take_mp3"] = take_path
                wav16k = os.path.join(fix_dir, "try%d_16k.wav" % i)
                media.extract_wav16k(take_path, wav16k)
                t["take_wav"] = wav16k
                _p("verifying", (i - 0.5) / float(n))
                plain = _plain_tokens(att["text"])
                k = _target_index(plain, occ["sentence"], att["char_start"])
                span = locate_span(plain, k, take.get("timing") or [])
                if span is None:
                    span = _locate_with_whisper(wav16k, plain, k)
                if span is None:
                    raise FixError("could not locate %r in the take" % occ.get("word"))
                s, e, how = span
                t["span"] = [round(s, 3), round(e, 3)]
                t["located"] = how
                clip = os.path.join(fix_dir, "try%d_word.wav" % i)
                sc = checker.score_window(wav16k, s, e, cands, pad, clip)
                probe = {"candidates": cands, "intended": intended}
                probe.update(sc)
                checker.judge(probe)
                t["word_wav"] = sc.get("wide_clip") or clip
                t["heard_as"] = sc.get("free_decode")
                t["detected"] = sc.get("detected")
                t["margin"] = sc.get("margin")
                t["scores"] = sc.get("scores")
                t["reason"] = probe.get("reason")
                intended_ipa = "".join(cands_ipa.get(intended, [])) if isinstance(cands_ipa.get(intended), list) else str(cands_ipa.get(intended) or "")
                heard_ok = heard_contains(intended_ipa, sc.get("free_decode"))
                letters = spelled_out(sc.get("free_decode"), intended_ipa)
                t["verified"] = bool(not probe.get("flag") and sc.get("detected") == intended
                                     and heard_ok and not letters)
                if letters:
                    t["reason"] = "the voice spelled the letters out (%s)" % (sc.get("free_decode") or "")
                elif not heard_ok and sc.get("detected") == intended:
                    t["reason"] = "intended phonemes not heard contiguously in /%s/" % (sc.get("free_decode") or "")
                _log("fix %s try %d (%s %r): detected=%s margin=%s heard=/%s/ %s"
                     % (occ_id, i, att["strategy"], att["respell"], t["detected"],
                        t["margin"], t["heard_as"], "VERIFIED" if t["verified"] else "not verified"))
            except (WellSaidError, media.MediaError, FixError, ValueError) as e:
                t["error"] = str(e)
                _log("fix %s try %d failed: %s" % (occ_id, i, e))
            if t["verified"]:
                result["verified"] = True
                if (t.get("margin") or 0.0) >= STRONG_MARGIN:
                    break  # convincing: stop spending renders
        verified = [t for t in result["tries"] if t.get("verified")]
        if verified:
            result["best"] = max(verified, key=lambda t: t.get("margin") or 0.0)
            result["strong"] = (result["best"].get("margin") or 0.0) >= STRONG_MARGIN
        else:
            result["strong"] = False
            # best effort: the try whose detected sense matched with the largest margin
            ok = [t for t in result["tries"] if t.get("detected") == intended]
            if ok:
                result["best"] = max(ok, key=lambda t: t.get("margin") or 0.0)
        _p("done", 1.0)
        result["summary"] = _summary(result)
        return result


def _render(ws, text, speaker_id, model):
    # type: (WellSaid, str, int, str) -> Dict[str, Any]
    try:
        return ws.render_timed(text, speaker_id, model=model)
    except WellSaidError as e:
        # word-timing may be unavailable for a model: fall back to a plain render
        audio = ws.render(text, speaker_id, model=model)
        return {"audio_bytes": audio, "timing": [], "ext": "mp3", "note": str(e)}


def _locate_with_whisper(wav16k, plain_tokens, k):
    # type: (str, List[Dict[str, Any]], int) -> Optional[Tuple[float, float, str]]
    from core import transcribe
    try:
        data = transcribe.transcribe(wav16k)
    except transcribe.TranscribeError:
        return None
    ttoks = checker.tokens_from_whisper(data.get("words", []))
    timing = [{"word": t["raw"], "start": t["s"], "end": t["e"]} for t in ttoks]
    span = locate_span(plain_tokens, k, timing)
    if span is None:
        return None
    return span[0], span[1], "whisper"


def _summary(res):
    # type: (Dict[str, Any]) -> str
    n = len(res["tries"])
    if res["verified"]:
        b = res["best"]
        return "%s fixed on try %d/%d (%s %r): heard /%s/, margin %s%s" % (
            res.get("word"), b["n"], n, b["strategy"], b["respell"], b.get("heard_as"),
            ("%.2f" % b["margin"]) if b.get("margin") is not None else "n/a",
            "" if res.get("strong", True) else " (weak — below %.1f)" % STRONG_MARGIN)
    errs = [t["error"] for t in res["tries"] if t.get("error")]
    if errs and len(errs) == n:
        return "%s: every render failed (%s)" % (res.get("word"), errs[-1])
    return "%s: not verified after %d tr%s" % (res.get("word"), n, "y" if n == 1 else "ies")


def fix_many(occs, *, speaker_id, model, work_dir, max_tries=3, progress=None, client=None,
             log=None):
    # type: (...) -> List[Dict[str, Any]]
    """fix_occurrence over several occurrences with one shared client and a
    combined progress (stage, frac). Errors become per-occurrence results."""
    ws = client or WellSaid()
    out = []
    n = max(1, len(occs))
    for i, occ in enumerate(occs):
        def _sub(stage, frac, _i=i):
            if progress:
                progress(stage, (_i + frac) / float(n))
        try:
            out.append(fix_occurrence(occ, speaker_id=speaker_id, model=model, work_dir=work_dir,
                                      max_tries=max_tries, progress=_sub, client=ws, log=log))
        except FixError as e:
            out.append({"occurrence_id": occ.get("id"), "word": occ.get("word"), "tries": [],
                        "best": None, "verified": False, "error": str(e),
                        "summary": "%s: %s" % (occ.get("word"), e)})
    if progress:
        progress("done", 1.0)
    return out
