#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure-core unit tests for Retake — no network, no WellSaid/Anthropic calls, no model weights.

Run from the project root:
    .venv/bin/python -m unittest tests.test_core -v
    .venv/bin/python tests/test_core.py
    pytest tests/test_core.py            (if pytest happens to be installed)

What is covered
  * `python -m core.selftest` exits 0 (builder A's own CTC brute-force / compound / lexicon checks)
  * lexicon: loads, validates clean, 100-ish entries with >= 2 senses of distinct IPA, every
    token in the model's inventory, WellSaid-format respelling candidates present, and
    validate() actually catches a bad lexicon
  * CTC forced log-likelihood against analytic answers and (when exposed) the brute-force path
  * prefix/suffix compound candidates: re-records -> ɹiː + each `record` sense + z, reads, uses
  * the SPEC signatures the API and UI are coded against (check_media, fix_occurrence,
    transcribe, PhonemeScorer, WellSaid)
Tests that depend on an optional core surface skip with a clear reason instead of failing.
Python 3.9 compatible.
"""
import copy
import inspect
import math
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Belt and braces: nothing in here may download a model.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _import(modname):
    """Import a core module or skip the test class with the reason."""
    try:
        return __import__(modname, fromlist=["_"])
    except ImportError as exc:  # pragma: no cover - depends on the install state
        raise unittest.SkipTest("%s not importable: %s" % (modname, exc))


def _first_attr(mod, names):
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    return None


# ---------------------------------------------------------------------------
# 1. core.selftest
# ---------------------------------------------------------------------------

class SelfTestCLI(unittest.TestCase):

    def test_core_selftest_passes(self):
        path = os.path.join(ROOT, "core", "selftest.py")
        self.assertTrue(os.path.exists(path),
                        "core/selftest.py is required by SPEC (`python -m core.selftest`)")
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run([sys.executable, "-m", "core.selftest"], cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=600)
        self.assertEqual(proc.returncode, 0,
                         "core.selftest failed (rc=%d):\n%s" % (proc.returncode, proc.stdout[-4000:]))


# ---------------------------------------------------------------------------
# 2. lexicon
# ---------------------------------------------------------------------------

class LexiconTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lexicon = _import("core.lexicon")
        cls.lex = cls.lexicon.load()

    def _validate(self, lex=None):
        validate = getattr(self.lexicon, "validate", None)
        if validate is None:
            self.skipTest("core.lexicon.validate not present")
        try:
            return list(validate(None, lex) if lex is not None else validate())
        except TypeError:
            if lex is not None:
                self.skipTest("core.lexicon.validate does not take a lexicon argument")
            raise

    def test_loads_business_lexicon(self):
        self.assertGreaterEqual(len(self.lex), 90)
        for w in ("record", "present", "project", "live", "read", "content", "produce"):
            self.assertIn(w, self.lex)

    def test_validate_reports_no_problems(self):
        self.assertEqual(self._validate(), [])

    def test_every_entry_has_two_distinct_senses(self):
        for word, entry in self.lex.items():
            senses = entry.get("senses", [])
            self.assertGreaterEqual(len(senses), 2, word)
            ipas = [tuple(s.get("ipa") or ()) for s in senses]
            self.assertTrue(all(ipas), "%s: empty IPA" % word)
            self.assertEqual(len(set(ipas)), len(ipas), "%s: senses share an IPA sequence" % word)
            ids = [s.get("id") for s in senses]
            self.assertEqual(len(set(ids)), len(ids), "%s: duplicate sense ids" % word)

    def test_tokens_in_model_inventory(self):
        inventory = getattr(self.lexicon, "DEFAULT_TOKENS", None)
        if not inventory:
            self.skipTest("core.lexicon.DEFAULT_TOKENS not present")
        inv = set(inventory)
        for word, entry in self.lex.items():
            for s in entry.get("senses", []):
                for tok in s.get("ipa", []):
                    self.assertIn(tok, inv, "%s/%s: token %r is not in the wav2vec2 inventory"
                                  % (word, s.get("id"), tok))

    def test_respell_candidates_present(self):
        missing = []
        for word, entry in self.lex.items():
            at_entry = bool(entry.get("respell_candidates"))
            at_sense = any(s.get("respell_candidates") for s in entry.get("senses", []))
            if not (at_entry or at_sense):
                missing.append(word)
        self.assertLessEqual(len(missing), max(0, len(self.lex) // 20),
                             "entries without WellSaid respell_candidates: %s" % missing[:10])

    def test_validate_catches_identical_ipa(self):
        bad = copy.deepcopy(self.lex)
        senses = bad["record"]["senses"]
        senses[1]["ipa"] = list(senses[0]["ipa"])
        problems = self._validate(bad)
        self.assertTrue(problems, "validate() accepted two senses with identical IPA")


# ---------------------------------------------------------------------------
# 3. CTC forced scoring (pure numpy)
# ---------------------------------------------------------------------------

class CTCTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise unittest.SkipTest("numpy missing: %s" % exc)
        cls.np = np
        cls.phonemes = _import("core.phonemes")
        ctc = _first_attr(cls.phonemes, ["ctc_forced_loglik", "ctc_loglik", "forced_loglik"])
        if ctc is None:
            raise unittest.SkipTest("core.phonemes exposes no ctc_forced_loglik")
        brute = _first_attr(cls.phonemes, ["ctc_brute_force_loglik", "ctc_brute_force"])
        # staticmethod: a plain function stored on the class would otherwise be bound as a method
        cls.ctc = staticmethod(ctc)
        cls.brute = staticmethod(brute) if brute else None

    def test_analytic_two_frames(self):
        # T=2, V=3, blank=0, target [1]: paths (b,1) (1,b) (1,1)
        p = self.np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
        want = math.log(0.6 * 0.5 + 0.3 * 0.2 + 0.3 * 0.5)
        got = self.ctc(self.np.log(p), [1], 0)
        self.assertAlmostEqual(float(got), want, places=9)

    def test_repeated_label_needs_a_blank(self):
        # T=3, target [1,1]: the only valid path is (1, b, 1)
        p = self.np.array([[0.5, 0.4, 0.1], [0.3, 0.3, 0.4], [0.2, 0.7, 0.1]])
        want = math.log(0.4 * 0.3 * 0.7)
        got = self.ctc(self.np.log(p), [1, 1], 0)
        self.assertAlmostEqual(float(got), want, places=9)

    def test_infeasible_target_is_minus_inf(self):
        p = self.np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
        got = self.ctc(self.np.log(p), [1, 1], 0)
        self.assertTrue(math.isinf(got) and got < 0)

    def test_matches_brute_force(self):
        if self.brute is None:
            self.skipTest("no brute-force reference exposed by core.phonemes")
        rng = self.np.random.default_rng(20261011)
        x = rng.normal(size=(5, 6))
        lp = x - self.np.log(self.np.exp(x).sum(axis=-1, keepdims=True))
        for labels, blank in (([2, 4], 0), ([3, 3], 0), ([1, 2, 5], 0), ([1, 0, 4], 5)):
            self.assertAlmostEqual(float(self.ctc(lp, labels, blank)),
                                   float(self.brute(lp, labels, blank)), places=9,
                                   msg="labels=%s blank=%d" % (labels, blank))

    def test_scorer_surface(self):
        scorer_cls = getattr(self.phonemes, "PhonemeScorer", None)
        self.assertIsNotNone(scorer_cls, "core.phonemes.PhonemeScorer missing")
        for name in ("load", "logprobs", "score", "greedy"):
            self.assertTrue(callable(getattr(scorer_cls, name, None)),
                            "PhonemeScorer.%s missing" % name)


# ---------------------------------------------------------------------------
# 4. compounds (re-records, reads, uses)
# ---------------------------------------------------------------------------

class CompoundTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.checker = _import("core.checker")
        cls.lex = _import("core.lexicon").load()
        decompose = _first_attr(cls.checker, ["decompose", "decompose_token", "split_compound"])
        build = _first_attr(cls.checker, ["build_candidates", "compound_candidates", "candidates_for"])
        if decompose is None or build is None:
            raise unittest.SkipTest("core.checker exposes no decompose/build_candidates pair "
                                    "(compounds are still covered by core.selftest)")
        cls.decompose = staticmethod(decompose)
        cls.build = staticmethod(build)

    def test_re_records_decomposes(self):
        self.assertEqual(tuple(self.decompose("rerecords", self.lex)), ("record", "re", "s"))

    def test_reads_decomposes(self):
        self.assertEqual(tuple(self.decompose("reads", self.lex)), ("read", None, "s"))

    def test_lexicon_word_itself_is_not_a_compound(self):
        self.assertIsNone(self.decompose("record", self.lex))
        self.assertIsNone(self.decompose("", self.lex))
        self.assertIsNone(self.decompose("banana", self.lex))

    def test_re_records_candidates(self):
        cands = self.build("record", "re", "s", self.lex)
        self.assertEqual(len(cands), 2)
        ids = {c["id"] for c in cands}
        self.assertEqual(ids, {"record_n", "record_v"})
        for c in cands:
            self.assertEqual(c["ipa"][:2], ["ɹ", "iː"], c)
            self.assertEqual(c["ipa"][-1], "z", c)
            if c.get("respell"):
                self.assertTrue(c["respell"].startswith("ree-"), c["respell"])

    def test_voiceless_tail_gets_s(self):
        # present ends in /t/ -> "presents" takes /s/
        cands = self.build("present", None, "s", self.lex)
        self.assertTrue(cands)
        for c in cands:
            self.assertEqual(c["ipa"][-1], "s", c)

    def test_sibilant_tail_gets_iz(self):
        if "use" not in self.lex:
            self.skipTest("'use' not in lexicon")
        cands = self.build("use", None, "s", self.lex)
        for c in cands:
            self.assertEqual(c["ipa"][-2:], ["ᵻ", "z"], c)


# ---------------------------------------------------------------------------
# 5. the SPEC contract the API/UI are coded against
# ---------------------------------------------------------------------------

class ContractSignatures(unittest.TestCase):

    def test_transcribe_signature(self):
        mod = _import("core.transcribe")
        sig = inspect.signature(mod.transcribe)
        params = list(sig.parameters)
        self.assertEqual(params[0], "wav16k_path")
        self.assertIn("backend", sig.parameters)
        self.assertEqual(sig.parameters["backend"].default, "auto")

    def test_check_media_signature(self):
        mod = _import("core.checker")
        sig = inspect.signature(mod.check_media)
        p = sig.parameters
        self.assertEqual(list(p)[0], "media_path")
        for name in ("script", "engine", "pad", "work_dir", "progress"):
            self.assertIn(name, p, name)
            self.assertEqual(p[name].kind, inspect.Parameter.KEYWORD_ONLY, name)
        self.assertEqual(p["engine"].default, "both")
        self.assertAlmostEqual(p["pad"].default, 0.1)

    def test_fix_occurrence_signature(self):
        mod = _import("core.fixer")
        sig = inspect.signature(mod.fix_occurrence)
        p = sig.parameters
        self.assertEqual(list(p)[0], "occ")
        for name in ("speaker_id", "model", "work_dir", "max_tries", "progress"):
            self.assertIn(name, p, name)
            self.assertEqual(p[name].kind, inspect.Parameter.KEYWORD_ONLY, name)
        self.assertEqual(p["max_tries"].default, 3)

    def test_wellsaid_client_surface(self):
        mod = _import("core.wellsaid")
        cls = getattr(mod, "WellSaid", None)
        self.assertIsNotNone(cls, "core.wellsaid.WellSaid missing")
        for name in ("render", "render_timed", "voices", "respell_suggestions"):
            self.assertTrue(callable(getattr(cls, name, None)), "WellSaid.%s missing" % name)

    def test_package_metadata(self):
        core = _import("core")
        self.assertTrue(getattr(core, "__version__", ""))
        self.assertTrue(os.path.exists(core.LEXICON_PATH))


if __name__ == "__main__":
    unittest.main(verbosity=2)
