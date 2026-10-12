# -*- coding: utf-8 -*-
"""Heteronym lexicon: load, validate, and look up senses / respellings.

Data lives in data/heteronyms.json:
    {surface_form: {"senses": [{id, pos, desc, ipa, respell}, ...],
                    "respell_candidates": ["REH-kurd", "ruh-KORD", ...]}}
`ipa` is a list of phoneme tokens drawn from the facebook/wav2vec2-lv-60-
espeak-cv-ft vocabulary (espeak IPA, no stress marks). Every sense `desc`
ends with the sense's respelling in parentheses, e.g. "(REH-kurd)";
`respell_candidates` are WellSaid-format respellings for the whole word
(from GET /tts/respelling_suggestions). `respell_for_sense` matches the two.

CLI:
    python -m core.lexicon validate [--with-model-vocab]
"""
import argparse
import difflib
import json
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Set

from core import LEXICON_PATH

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
STRESS_MARKS = ("ˈ", "ˌ")  # length mark ː is kept
VALID_POS = {"NOUN", "VERB", "ADJ", "ADV"}
POS_LABEL = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}

# Fallback token inventory (every token verified present in the model's
# vocab.json). Multi-character tokens first (longest-match wins).
DEFAULT_TOKENS = [
    "aɪɚ", "ɔːɹ", "ɑːɹ", "oːɹ",
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "iː", "uː", "ɑː", "ɔː", "ɜː",
    "ɪɹ", "ɛɹ", "ʊɹ", "tʃ", "dʒ",
    "əl", "ᵻ", "ɐ", "ɚ", "æ", "ɛ", "ɪ", "ʊ", "ʌ", "ə", "i", "u", "o",
    "ɹ", "ɾ", "ŋ", "θ", "ð", "ʃ", "ʒ", "ɡ",
    "b", "d", "f", "h", "j", "k", "l", "m", "n", "p", "s", "t", "v", "w", "z",
]

_CACHE = {}  # type: Dict[str, Dict[str, Any]]
_CACHE_LOCK = threading.Lock()
_PAREN_RE = re.compile(r"\(([^()]*)\)\s*$")


# ---------------------------------------------------------------------------
# load / validate
# ---------------------------------------------------------------------------

def load(path=None):
    # type: (Optional[str]) -> Dict[str, Any]
    """Load data/heteronyms.json (or an alternate path). Cached per path."""
    p = os.path.abspath(str(path) if path is not None else LEXICON_PATH)
    with _CACHE_LOCK:
        lex = _CACHE.get(p)
        if lex is None:
            with open(p, "r", encoding="utf-8") as f:
                lex = json.load(f)
            if not isinstance(lex, dict) or not lex:
                raise ValueError("lexicon %s is empty or not a JSON object" % p)
            _CACHE[p] = lex
        return lex


def validate(tokenizer_vocab=None, lex=None):
    # type: (Optional[Set[str]], Optional[Dict[str, Any]]) -> List[str]
    """Return a list of problem strings (empty list == lexicon is valid).

    Checks: lowercase surface forms, duplicate sense ids, entries with <2
    senses, identical IPA pairs within an entry, empty/missing tokens, stress
    marks inside tokens, bad POS, a respelling in every desc, non-empty
    respell_candidates, and (when `tokenizer_vocab` is given) membership of
    every token in it.
    """
    if lex is None:
        lex = load()
    problems = []  # type: List[str]
    seen_ids = set()  # type: Set[str]
    for word, entry in lex.items():
        senses = entry.get("senses", [])
        if word != word.lower():
            problems.append("%s: surface form is not lowercase" % word)
        if len(senses) < 2:
            problems.append("%s: fewer than 2 senses" % word)
        rc = entry.get("respell_candidates")
        if not isinstance(rc, list) or not rc or not all(isinstance(r, str) and r for r in rc):
            problems.append("%s: respell_candidates missing or empty" % word)
        ipa_seqs = []
        for sense in senses:
            sid = sense.get("id") or "%s:<missing id>" % word
            if sid in seen_ids:
                problems.append("duplicate sense id: %s" % sid)
            seen_ids.add(sid)
            if sense.get("pos") not in VALID_POS:
                problems.append("%s: bad pos %r" % (sid, sense.get("pos")))
            if not desc_respell(sense):
                problems.append("%s: desc has no trailing (respelling)" % sid)
            ipa = sense.get("ipa")
            if not isinstance(ipa, list) or not ipa:
                problems.append("%s: ipa missing or empty" % sid)
                continue
            for tok in ipa:
                if not isinstance(tok, str) or not tok:
                    problems.append("%s: empty ipa token" % sid)
                elif any(m in tok for m in STRESS_MARKS):
                    problems.append("%s: stress mark inside token %r" % (sid, tok))
                elif tokenizer_vocab is not None and tok not in tokenizer_vocab:
                    problems.append("%s: token %r not in model vocab" % (sid, tok))
            ipa_seqs.append(tuple(ipa))
        if len(set(ipa_seqs)) != len(ipa_seqs):
            problems.append("%s: two senses have identical IPA sequences" % word)
    return problems


def load_model_vocab():
    # type: () -> Set[str]
    """The wav2vec2 phoneme vocab from the repo's vocab.json (small file; the
    tokenizer class is bypassed because it fails to load on transformers 4.57)."""
    from huggingface_hub import hf_hub_download  # lazy
    path = hf_hub_download(MODEL_ID, "vocab.json")
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f).keys())


# ---------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------

def senses(word, lex=None):
    # type: (str, Optional[Dict[str, Any]]) -> List[Dict[str, Any]]
    """The sense list for a surface form ([] when unknown)."""
    lex = lex if lex is not None else load()
    entry = lex.get((word or "").lower())
    return list(entry.get("senses", [])) if entry else []


def sense_by_id(word, sense_id, lex=None):
    # type: (str, str, Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]
    for s in senses(word, lex):
        if s.get("id") == sense_id:
            return s
    return None


def desc_respell(sense):
    # type: (Dict[str, Any]) -> str
    """The respelling written at the end of a sense desc: "(REH-kurd)" -> "REH-kurd"."""
    m = _PAREN_RE.search(sense.get("desc", "") or "")
    return m.group(1).strip() if m else ""


def _respell_key(s):
    # type: (str) -> str
    return re.sub(r"[^a-z]", "", (s or "").lower())


def respell_for_sense(word, sense_id, lex=None):
    # type: (str, str, Optional[Dict[str, Any]]) -> str
    """The WellSaid-format respelling for one sense of a word.

    Picks the `respell_candidates` entry closest to the sense's own desc
    respelling (exact match ignoring case/hyphens first, then the best
    difflib ratio, provided it beats the other senses' respellings for that
    candidate). Falls back to the desc respelling itself, so the result is
    never empty for a valid lexicon entry.
    """
    lex = lex if lex is not None else load()
    entry = lex.get((word or "").lower()) or {}
    sense = sense_by_id(word, sense_id, lex)
    if sense is None:
        return ""
    want = desc_respell(sense) or sense.get("respell") or ""
    if want.upper() == "TODO":
        want = ""
    cands = [c for c in (entry.get("respell_candidates") or []) if isinstance(c, str) and c]
    if not cands:
        return want
    if not want:
        return cands[0]
    wk = _respell_key(want)
    for c in cands:
        if _respell_key(c) == wk:
            return c
    other = [_respell_key(desc_respell(s)) for s in entry.get("senses", [])
             if s.get("id") != sense_id]
    best, best_score = None, -1.0
    for c in cands:
        ck = _respell_key(c)
        raw = difflib.SequenceMatcher(None, wk, ck).ratio()
        score = raw
        # stress pattern agreement (upper-case syllables) is a strong hint
        if _stress_pattern(c) == _stress_pattern(want):
            score += 0.15
        # a candidate that reads MORE like another sense's respelling belongs
        # to that sense (LED vs LEED/LEHD: LEED is the other sense exactly)
        if any(difflib.SequenceMatcher(None, o, ck).ratio() > raw for o in other):
            score -= 0.5
        if score > best_score:
            best, best_score = c, score
    return best or want


def _stress_pattern(respell):
    # type: (str) -> str
    parts = re.split(r"[-\s]+", respell or "")
    return "".join("S" if p.isupper() else "u" for p in parts if p)


def public_view(lex=None):
    # type: (Optional[Dict[str, Any]]) -> List[Dict[str, Any]]
    """Lexicon as the API exposes it (GET /v1/lexicon): one row per word."""
    lex = lex if lex is not None else load()
    rows = []
    for word in sorted(lex):
        entry = lex[word]
        rows.append({
            "word": word,
            "respell_candidates": list(entry.get("respell_candidates") or []),
            "senses": [{
                "id": s.get("id"),
                "pos": s.get("pos"),
                "pos_label": POS_LABEL.get(s.get("pos"), s.get("pos")),
                "desc": s.get("desc", ""),
                "ipa": "".join(s.get("ipa", [])),
                "ipa_tokens": list(s.get("ipa", [])),
                "respell": respell_for_sense(word, s.get("id"), lex),
            } for s in entry.get("senses", [])],
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m core.lexicon",
                                     description="Load and validate the heteronym lexicon.")
    sub = parser.add_subparsers(dest="cmd")
    val = sub.add_parser("validate", help="validate data/heteronyms.json")
    val.add_argument("--with-model-vocab", action="store_true",
                     help="check every IPA token against the %s vocab.json "
                          "(small download, cached)" % MODEL_ID)
    args = parser.parse_args(argv)
    if args.cmd != "validate":
        parser.print_help()
        return 2
    try:
        lex = load()
    except Exception as exc:
        print("error: cannot load %s: %s" % (LEXICON_PATH, exc), file=sys.stderr)
        return 2
    n_senses = sum(len(e.get("senses", [])) for e in lex.values())
    print("lexicon: %d entries, %d senses (%s)" % (len(lex), n_senses, LEXICON_PATH))
    vocab = None
    if args.with_model_vocab:
        try:
            vocab = load_model_vocab()
            print("model vocab: %d tokens loaded" % len(vocab))
        except Exception as exc:
            print("error: could not load model vocab: %s" % exc, file=sys.stderr)
            return 2
    problems = validate(tokenizer_vocab=vocab, lex=lex)
    if problems:
        print("\n%d problem(s):" % len(problems))
        for p in problems:
            print("  - %s" % p)
        return 1
    print("OK: no problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
