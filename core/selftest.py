# -*- coding: utf-8 -*-
"""Retake core self-test: `python -m core.selftest`.

No network, no model weights, no renders. Covers:
  * CTC forward recursion vs analytic answers and brute-force path enumeration
  * lexicon validation + respelling-for-sense matching
  * occurrence detection, whisper token joining, sentences, compounds
    (re-/un-/pre-/mis- prefixes, plural tails), script alignment
  * flag rules (CTC + fuzzy free-decode containment) and confidence levels
  * fixer attempt planning for caruso and preview + take-word location
  * WellSaid timing normalization (Google-STT shape and generic shapes)
  * media.probe on a synthetic wav (ffmpeg must resolve)
Exit 0 on success, 1 on any failure.
"""
import json
import math
import os
import sys
import tempfile
import wave

import numpy as np

from core import checker, fixer, intent, lexicon, media, phonemes, wellsaid

_FAILS = []  # type: list
_N = [0]


def check(name, cond, detail=""):
    _N[0] += 1
    print("  [%s] %s%s" % ("ok" if cond else "FAIL", name, (" — " + str(detail)) if (detail and not cond) else ""))
    if not cond:
        _FAILS.append(name)


def close(a, b, tol=1e-9):
    if math.isinf(b):
        return math.isinf(a) and (a < 0) == (b < 0)
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------

def test_ctc():
    print("CTC forward (pure numpy)")
    p = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    lp = np.log(p)
    want = math.log(0.6 * 0.5 + 0.3 * 0.2 + 0.3 * 0.5)
    check("analytic T=2 V=3 target=[1]", close(phonemes.ctc_forced_loglik(lp, [1], 0), want))
    p3 = np.array([[0.5, 0.4, 0.1], [0.3, 0.3, 0.4], [0.2, 0.7, 0.1]])
    check("analytic T=3 target=[1,1] (blank-separated)",
          close(phonemes.ctc_forced_loglik(np.log(p3), [1, 1], 0), math.log(0.4 * 0.3 * 0.7)))
    check("infeasible T=2 target=[1,1] is -inf", close(phonemes.ctc_forced_loglik(lp, [1, 1], 0), -np.inf))
    check("brute force agrees it is -inf", close(phonemes.ctc_brute_force_loglik(lp, [1, 1], 0), -np.inf))
    rng = np.random.default_rng(20261009)
    T, V = 6, 8
    lp_rand = phonemes.log_softmax_np(rng.normal(size=(T, V)))
    for name, labels, blank in [
        ("random 6x8 target len2", [3, 5], 0),
        ("random 6x8 target len3", [2, 6, 1], 0),
        ("random 6x8 repeated label [4,4]", [4, 4], 0),
        ("random 6x8 repeat inside [1,3,3]", [1, 3, 3], 0),
        ("random 6x8 non-zero blank id (7)", [2, 0, 5], 7),
    ]:
        got = phonemes.ctc_forced_loglik(lp_rand, labels, blank)
        want = phonemes.ctc_brute_force_loglik(lp_rand, labels, blank)
        check(name, close(got, want), "got %r want %r" % (got, want))
    lp2 = phonemes.log_softmax_np(rng.normal(size=(T, V)))
    labels2 = [int(x) for x in rng.integers(1, V, size=3)]
    check("random 6x8 fresh draw target=%s" % labels2,
          close(phonemes.ctc_forced_loglik(lp2, labels2, 0), phonemes.ctc_brute_force_loglik(lp2, labels2, 0)))
    check("collapse merges repeats and drops blanks", phonemes.collapse([0, 1, 1, 0, 2, 2, 0, 2], 0) == [1, 2, 2])
    try:
        phonemes.ctc_forced_loglik(lp, [0], 0)
        check("blank in labels raises", False)
    except ValueError:
        check("blank in labels raises", True)


def test_lexicon():
    print("lexicon")
    lex = lexicon.load()
    check("100 entries loaded", len(lex) == 100, len(lex))
    problems = lexicon.validate(lex=lex)
    check("validate: no problems", not problems, problems[:3])
    bad = {"bad": {"senses": [{"id": "x", "pos": "NOUN", "desc": "a (A)", "ipa": ["a"], "respell": "TODO"}],
                   "respell_candidates": []}}
    probs = lexicon.validate(lex=bad)
    check("validate catches <2 senses + empty candidates",
          any("fewer than 2" in p for p in probs) and any("respell_candidates" in p for p in probs), probs)
    vocab = set(lexicon.DEFAULT_TOKENS)
    check("validate with a vocab flags unknown tokens",
          lexicon.validate(tokenizer_vocab=vocab - {"ɹ"}, lex=lex))
    picks = {
        ("record", "record_n"): "REH-kurd", ("record", "record_v"): "ruh-KORD",
        ("live", "live_v"): "LIHV", ("live", "live_adj"): "LYV",
        ("read", "read_present"): "REED", ("read", "read_past"): "REHD",
        ("lead", "lead_v"): "LEED", ("lead", "lead_n"): "LEHD",
        ("present", "present_v"): "pruh-ZEHNT", ("close", "close_adj"): "KLOHS",
    }
    for (w, sid), want in picks.items():
        got = lexicon.respell_for_sense(w, sid, lex)
        check("respell_for_sense %s -> %s" % (sid, want), got == want, got)
    check("respell_for_sense unknown sense -> ''", lexicon.respell_for_sense("record", "nope", lex) == "")
    rows = lexicon.public_view(lex)
    check("public_view rows carry senses with respell", rows and all(r["senses"] and all(s["respell"] for s in r["senses"]) for r in rows))


def test_detection():
    print("detection + compounds")
    lex = lexicon.load()
    occ = intent.detect_occurrences("Please Record the call; the record shows it.", lex)
    check("detect_occurrences finds both 'record's", [o["word"] for o in occ] == ["record", "record"], occ)
    check("offsets are exact", occ[0]["char_start"] == 7 and occ[0]["char_end"] == 13, occ)
    check("decompose rerecords", checker.decompose("rerecords", lex) == ("record", "re", "s"))
    check("decompose reads", checker.decompose("reads", lex) == ("read", None, "s"))
    check("decompose unread", checker.decompose("unread", lex) == ("read", "un", None))
    check("decompose prerecord", checker.decompose("prerecord", lex) == ("record", "pre", None))
    check("decompose misreads", checker.decompose("misreads", lex) == ("read", "mis", "s"))
    check("decompose lexicon word -> None", checker.decompose("record", lex) is None)
    check("decompose junk -> None", checker.decompose("zzz", lex) is None)
    cands = checker.build_candidates("record", "re", "s", lex)
    ipa = {c["id"]: "".join(c["ipa"]) for c in cands}
    check("re-records candidates carry prefix + plural", ipa == {"record_n": "ɹiːɹɛkɚdz", "record_v": "ɹiːɹᵻkɔːɹdz"}, ipa)
    rsp = {c["id"]: c["respell"] for c in cands}
    check("re-records respellings compounded", rsp == {"record_n": "ree-REH-kurdz", "record_v": "ree-ruh-KORDZ"}, rsp)
    uses = {c["id"]: "".join(c["ipa"]) for c in checker.build_candidates("use", None, "es", lex)}
    check("uses -> ᵻz tail after sibilant", uses == {"use_n": "juːsᵻz", "use_v": "juːzᵻz"}, uses)
    words = [{"w": " We", "s": 0.0, "e": 0.2, "p": 1}, {"w": " re", "s": 0.3, "e": 0.4, "p": 1},
             {"w": "-records", "s": 0.4, "e": 0.9, "p": 0.9}, {"w": " it", "s": 1.0, "e": 1.1, "p": 1},
             {"w": " live.", "s": 1.2, "e": 1.5, "p": 1}, {"w": " Done.", "s": 1.6, "e": 1.9, "p": 1}]
    toks = checker.tokens_from_whisper(words)
    check("whisper hyphen pieces joined", [t["raw"] for t in toks] == ["We", "re-records", "it", "live.", "Done."], toks)
    check("joined token keeps the end time", toks[1]["e"] == 0.9 and toks[1]["s"] == 0.3)
    sents = checker.build_sentences(toks)
    check("two sentences", [s["text"] for s in sents] == ["We re-records it live.", "Done."], sents)
    occs = checker.collect_occurrences(toks, lex)
    check("collect: re-records + live", [(o["word"], o["base"], o["prefix"], o["tail"]) for o in occs]
          == [("re-records", "record", "re", "s"), ("live", "live", None, None)], occs)
    check("occurrence timing from the token", occs[0]["t_start"] == 0.3 and occs[1]["t_end"] == 1.5)
    stoks = checker.tokens_from_text("We re-record it live. Done.")
    mapping = checker.align_script(stoks, toks)
    check("align_script maps equal + same-length replace blocks", mapping.get(0) == 0 and mapping.get(3) == 3 and mapping.get(1) == 1, mapping)
    ct = checker.tokens_from_text("The record. Then record it.")
    o2 = checker.collect_occurrences(ct, lex)
    check("script tokens carry absolute offsets", [(o["text_start"], o["text_end"]) for o in o2] == [(4, 10), (17, 23)], o2)


def test_rules():
    print("flag rules + confidence")
    lex = lexicon.load()
    cands = checker.build_candidates("record", None, None, lex)
    base = {"candidates": cands, "intended": "record_v", "intent_confidence": 0.75, "intent_engine": "spacy"}

    o = dict(base, undecidable=False, detected="record_v", margin=1.2,
             free_tokens=["ɹ", "ɪ", "k", "ɔːɹ", "d"])
    checker.judge(o)
    check("clean when CTC + free decode agree with intent", o["flag"] is False and o["reason"] == "ok", o["reason"])
    check("ᵻ/ɪ and ɔːɹ/ɔɹ variants are tolerated", o["free_decode_distance"] == 0)
    check("confidence high on a clear clean", checker.confidence(o)[0] == "high", checker.confidence(o))

    o = dict(base, undecidable=False, detected="record_n", margin=0.9,
             free_tokens=["ð", "ə", "ɹ", "ɛ", "k", "ɚ", "d", "z"])
    checker.judge(o)
    check("flag when the noun was said for an intended verb", o["flag"] and o["rule_ctc"] and o["rule_free"], o["reason"])
    check("confidence high when both rules agree, margin 0.9", checker.confidence(o)[0] == "high", checker.confidence(o))
    o["intent_confidence"] = 0.5
    check("uncertain intent lowers confidence one step", checker.confidence(o)[0] == "med", checker.confidence(o))

    o = dict(base, undecidable=False, detected="record_v", margin=0.3,
             free_tokens=["ɹ", "ɛ", "k", "ɚ", "d"])
    checker.judge(o)
    check("free-decode rule alone still flags", o["flag"] and not o["rule_ctc"] and o["rule_free"], o["reason"])
    check("rules disagree on a small margin -> low", checker.confidence(o)[0] == "low", checker.confidence(o))

    o = dict(base, undecidable=True)
    checker.judge(o)
    check("undecidable never flags", o["flag"] is False and "undecidable" in o["reason"])
    check("undecidable -> low", checker.confidence(o)[0] == "low")

    check("substring_distance semi-global", checker.substring_distance(["a", "b"], ["x", "a", "b", "y"]) == 0
          and checker.substring_distance(["a", "c"], ["x", "a", "b", "y"]) == 1)
    check("free_decode_slack", [checker.free_decode_slack(n) for n in (1, 3, 4, 8, 9)] == [1, 1, 1, 2, 2])
    recs = [{"id": "o1", "t_start": 1.0, "word": "record", "flag": True, "intended": "record_v",
             "intended_respell": "ruh-KORD", "intended_pos": "verb", "detected": "record_n",
             "detected_respell": "REH-kurd", "detected_pos": "noun", "margin": 0.9, "confidence": "high"}]
    s = checker.build_summary(recs, 12.0)
    check("summary lists the flag", len(s) == 2 and "REH-kurd" in s[1] and "1 flagged" in s[0], s)
    check("verdict", checker.verdict(recs) == "issues" and checker.verdict([]) == "empty")


def test_fixer():
    print("fixer planning (no renders)")
    occ = {"id": "o3", "word": "re-records", "base": "record", "prefix": "re", "tail": "s",
           "sentence": "Autopilot re-records the line & ships it.", "char_start": 10, "char_end": 20,
           "intended": "record_v", "intended_respell": "ree-ruh-KORDZ",
           "candidate_tokens": {"record_n": ["ɹ", "iː", "ɹ", "ɛ", "k", "ɚ", "d", "z"],
                                "record_v": ["ɹ", "iː", "ɹ", "ᵻ", "k", "ɔːɹ", "d", "z"]},
           "candidate_respells": {"record_n": "ree-REH-kurdz", "record_v": "ree-ruh-KORDZ"}}
    plan = fixer.plan_attempts(occ, "caruso", max_tries=3)
    check("caruso try 1 is an inline respell of the surface word",
          plan and plan[0]["text"] == 'Autopilot <respell value="ree-ruh-KORDZ">re-records</respell> the line &amp; ships it.', plan)
    check("caruso falls back to the desc respelling compounded",
          any(p["respell"] == "ree-rih-KORDZ" for p in plan), [p["respell"] for p in plan])
    plan = fixer.plan_attempts(occ, "preview", max_tries=3)
    check("preview spellings in verified order", [p["respell"] for p in plan] == ["ree-recordz", "ree-ruh-kordz", "ree-rih-kordz"], plan)
    check("preview try 1 keeps the real word, respells the prefix", plan[0]["text"] == "Autopilot ree-recordz the line &amp; ships it.", plan[0]["text"])
    occ_live = {"id": "o9", "word": "live", "base": "live", "sentence": "We go live today.",
                "char_start": 6, "char_end": 10, "intended": "live_adj"}
    plan_live = fixer.plan_attempts(occ_live, "preview", max_tries=3)
    check("preview uses the lexicon's verified spelling first", [p["respell"] for p in plan_live][:2] == ["lyve", "lahyv"], plan_live)
    check("STRONG_MARGIN is 0.8", abs(fixer.STRONG_MARGIN - 0.8) < 1e-9)
    check("spelling_variants('ree-KORDZ')", fixer.spelling_variants("ree-KORDZ") == ["ree-kordz", "ri-kordz"])
    check("max_tries caps the plan", len(fixer.plan_attempts(occ, "preview", max_tries=1)) == 1)
    try:
        fixer.plan_attempts(occ, "legacy")
        check("unsupported fix model raises", False)
    except fixer.FixError:
        check("unsupported fix model raises", True)
    occ2 = {"id": "o1", "word": "live", "base": "live", "sentence": "We go live today.",
            "char_start": 6, "char_end": 10, "intended": "live_adj"}
    plan = fixer.plan_attempts(occ2, "caruso")
    check("candidates rebuilt from the lexicon when tokens are absent", plan[0]["respell"] == "LYV", plan)
    plain = fixer._plain_tokens(plan[0]["text"])
    check("markup stripped for alignment", [t["raw"] for t in plain] == ["We", "go", "live", "today."], plain)
    k = fixer._target_index(plain, occ2["sentence"], 6)
    timing = [{"word": "We", "start": 0.0, "end": 0.2}, {"word": "go", "start": 0.2, "end": 0.4},
              {"word": "live", "start": 0.45, "end": 0.8}, {"word": "today.", "start": 0.85, "end": 1.3}]
    check("locate_span by timing", fixer.locate_span(plain, k, timing) == (0.45, 0.8, "timing"))
    timing2 = [timing[0], timing[1], {"word": "ly", "start": 0.45, "end": 0.6},
               {"word": "ve", "start": 0.6, "end": 0.8}, timing[3]]
    check("locate_span falls back to the gap between neighbours",
          fixer.locate_span(plain, k, timing2) == (0.4, 0.85, "gap"), fixer.locate_span(plain, k, timing2))
    check("locate_span with no timing -> None", fixer.locate_span(plain, k, []) is None)


def test_wellsaid():
    print("wellsaid (offline)")
    nt = wellsaid.normalize_timing
    got = nt({"results": [{"alternatives": [{"words": [
        {"endOffset": "0.5s", "word": "Hi"},
        {"startOffset": "0.56s", "endOffset": "0.88s", "word": "there"}]}]}]})
    check("Google-STT shape with 's' strings", got == [{"word": "Hi", "start": 0.0, "end": 0.5},
                                                      {"word": "there", "start": 0.56, "end": 0.88}], got)
    cases = [
        ("bare list", [{"word": "the", "start": 0.1, "end": 0.4}], [("the", 0.1, 0.4)]),
        ("wrapped ms", {"words": [{"word": "record", "start": 1200, "end": 1700}]}, [("record", 1.2, 1.7)]),
        ("key variants", {"word_timings": [{"text": "hi", "startTime": 0.0, "endTime": 0.5},
                                           {"token": "there", "start_time": "0.5", "end_time": "1.0"}]},
         [("hi", 0.0, 0.5), ("there", 0.5, 1.0)]),
        ("duration + triple", [{"word": "x", "start": 2.0, "duration": 0.5}, ["y", 2.5, 3.0]],
         [("x", 2.0, 2.5), ("y", 2.5, 3.0)]),
        ("garbage", {"nothing": "here"}, []),
    ]
    for label, raw, want in cases:
        g = nt(raw)
        ok = len(g) == len(want) and all(a["word"] == w and abs(a["start"] - s) < 1e-9 and abs(a["end"] - e) < 1e-9
                                         for a, (w, s, e) in zip(g, want))
        check("normalize_timing %s" % label, ok, g)
    check("respell_tag escapes", wellsaid.respell_tag("A&B", "AY-bee") == '<respell value="AY-bee">A&amp;B</respell>')
    ws = wellsaid.WellSaid(key="not-a-real-key")
    try:
        ws.render("x" * 1200, 3)
        check("1000-char guard", False)
    except ValueError:
        check("1000-char guard", True)
    try:
        ws.render("hi", 3, model="nope")
        check("unknown model guard", False)
    except ValueError:
        check("unknown model guard", True)
    body = ws._body("hi", 3, "preview", "wav", 48000, None)
    check("render body shape", body == {"speaker_id": 3, "text": "hi", "model": "preview",
                                        "audio_configs": {"file_format": "wav", "sample_rate": 48000}}, body)
    check("repr never shows the key", "not-a-real-key" not in repr(ws))
    no = wellsaid.WellSaid(key=None)
    check("client without key reports ok=False or resolves a local key", isinstance(no.ok, bool))


def test_media():
    print("media (ffmpeg)")
    try:
        ff = media.ffmpeg_path()
    except media.MediaError as e:
        check("ffmpeg resolves", False, e)
        return
    check("ffmpeg resolves", os.path.exists(ff))
    tmp = tempfile.mkdtemp(prefix="retake-selftest-")
    wav = os.path.join(tmp, "tone.wav")
    sr = 16000
    t = np.arange(int(sr * 1.5)) / float(sr)
    pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    check("wav_duration", abs(media.wav_duration(wav) - 1.5) < 1e-3)
    info = media.probe(wav)
    check("probe accepts a wav", info["ok"] and info["has_audio"] and not info["has_video"], info)
    bad = os.path.join(tmp, "not-media.txt")
    with open(bad, "w") as fh:
        fh.write("hello")
    check("probe rejects a text file", media.probe(bad)["ok"] is False)
    clip = os.path.join(tmp, "clip.wav")
    media.cut_clip(wav, 0.5, 0.7, 0.1, clip)
    check("cut_clip length", abs(media.wav_duration(clip) - 0.4) < 0.02, media.wav_duration(clip))
    out = os.path.join(tmp, "cat.wav")
    media.concat_audio([clip, clip], out)
    check("concat_audio length", abs(media.wav_duration(out) - 0.8) < 0.03, media.wav_duration(out))
    spl = os.path.join(tmp, "spliced.wav")
    media.replace_segment(wav, 0.5, 1.0, clip, spl)
    check("replace_segment length", abs(media.wav_duration(spl) - 1.4) < 0.03, media.wav_duration(spl))
    x = media.load_wav(clip)
    check("load_wav float32 mono", x.dtype == np.float32 and x.ndim == 1 and len(x) > 1000)


def main():
    print("Retake core self-test (no network, no models)")
    for fn in (test_ctc, test_lexicon, test_detection, test_rules, test_fixer, test_wellsaid, test_media):
        try:
            fn()
        except Exception as e:  # a crashed group is a failure, not a skip
            import traceback
            traceback.print_exc()
            check("%s crashed" % fn.__name__, False, e)
    if _FAILS:
        print("\nSELFTEST FAILED: %d of %d checks: %s" % (len(_FAILS), _N[0], ", ".join(_FAILS)), file=sys.stderr)
        return 1
    print("\nselftest OK (%d checks)" % _N[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
