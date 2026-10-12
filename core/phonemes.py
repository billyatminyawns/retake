# -*- coding: utf-8 -*-
"""Phoneme scoring with facebook/wav2vec2-lv-60-espeak-cv-ft (port of
heteronym-poc verify.py).

    scorer = PhonemeScorer()          # cheap; the model loads on first use
    lp = scorer.logprobs(wav_f32)     # np.ndarray [T, V] log-softmax
    scorer.score(lp, ["ɹ","ɛ","k","ɚ","d"])  # CTC forced log-lik / len
    scorer.greedy(lp)                 # "ɹɛkɚdz" (greedy CTC decode, IPA)
    scorer.vocab                      # {token: id}

The model bundle is shared at class level, so every PhonemeScorer() instance
is the same lazily-loaded singleton. Loading and inference are guarded by
locks (the model is not re-entrant). `ctc_forced_loglik` is a pure-numpy
log-space forward recursion; `ctc_brute_force_loglik` is its exponential
reference used by core.selftest.

NOTE: this checkpoint's Wav2Vec2PhonemeCTCTokenizer fails to load on
transformers 4.57 (from_pretrained returns False), so the tokenizer is
bypassed: the vocab comes from the repo's vocab.json and the CTC blank id
from model.config.pad_token_id (verified identical: both 0).
"""
import itertools
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from core import models_dir  # noqa: F401  (sets HF_HOME from RETAKE_MODELS)

MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16000
SPECIAL_TOKENS = {"<s>", "</s>", "<unk>", "<pad>", "|"}


class PhonemeError(RuntimeError):
    """Model unavailable or an IPA token is outside the model vocab."""


# ---------------------------------------------------------------------------
# CTC forced log-likelihood (pure numpy, log space)
# ---------------------------------------------------------------------------

def ctc_forced_loglik(log_probs, labels, blank_id):
    """Log-likelihood that the CTC output `log_probs` spells out `labels`.

    log_probs: np.ndarray [T, V] of per-frame log-softmax scores.
    labels:    sequence of vocab ids (no blanks), length >= 1.
    blank_id:  the CTC blank id.

    Standard forward (alpha) recursion over the blank-extended sequence
    l' = [b, l1, b, l2, ..., b]; returns logaddexp of the two terminal states.
    Returns -inf when no alignment exists (e.g. too few frames).
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T, V = log_probs.shape
    labels = list(labels)
    if not labels:
        raise ValueError("labels must be non-empty")
    for l in labels:
        if not (0 <= int(l) < V):
            raise ValueError("label id %r out of vocab range %d" % (l, V))
        if int(l) == blank_id:
            raise ValueError("labels must not contain the blank id")

    ext = [blank_id]
    for l in labels:
        ext.append(int(l))
        ext.append(blank_id)
    S = len(ext)  # 2L + 1

    NEG = -np.inf
    alpha = np.full(S, NEG, dtype=np.float64)
    alpha[0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[1] = log_probs[0, ext[1]]

    for t in range(1, T):
        prev = alpha
        alpha = np.full(S, NEG, dtype=np.float64)
        for s in range(S):
            v = prev[s]
            if s >= 1:
                v = np.logaddexp(v, prev[s - 1])
            # The skip transition is allowed only into a non-blank state whose
            # label differs from the label two states back.
            if s >= 2 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                v = np.logaddexp(v, prev[s - 2])
            if v != NEG:
                alpha[s] = v + log_probs[t, ext[s]]

    if S == 1:
        return float(alpha[0])
    return float(np.logaddexp(alpha[S - 1], alpha[S - 2]))


def collapse(path, blank_id):
    """CTC collapse: merge repeats, then drop blanks."""
    out = []
    prev = None
    for p in path:
        if p != prev:
            if p != blank_id:
                out.append(p)
        prev = p
    return out


def ctc_brute_force_loglik(log_probs, labels, blank_id):
    """Reference implementation: enumerate every length-T path over the vocab,
    keep those that collapse to `labels`, logsumexp their path scores.
    Exponential — only for tiny self-test cases."""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T, V = log_probs.shape
    labels = list(int(l) for l in labels)
    total = -np.inf
    for path in itertools.product(range(V), repeat=T):
        if collapse(path, blank_id) == labels:
            score = 0.0
            for t in range(T):
                score += log_probs[t, path[t]]
            total = np.logaddexp(total, score)
    return float(total)


def log_softmax_np(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    e = x - m
    return e - np.log(np.exp(e).sum(axis=-1, keepdims=True))


# ---------------------------------------------------------------------------
# Scorer (lazy singleton)
# ---------------------------------------------------------------------------

class PhonemeScorer(object):
    """Lazy singleton around the wav2vec2 phoneme model (see module doc)."""

    _bundle = None  # type: Optional[Dict[str, Any]]
    _load_lock = threading.Lock()
    run_lock = threading.RLock()  # serializes inference
    load_seconds = None  # type: Optional[float]

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def status(cls):
        # type: () -> str
        return "loaded" if cls._bundle is not None else "cold"

    @property
    def loaded(self):
        # type: () -> bool
        return PhonemeScorer._bundle is not None

    def load(self):
        """Load the model once (thread-safe). Returns self."""
        if PhonemeScorer._bundle is not None:
            return self
        with PhonemeScorer._load_lock:
            if PhonemeScorer._bundle is not None:
                return self
            try:
                import torch  # noqa: F401
                from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise PhonemeError("torch/transformers not importable: %s" % e)
            t0 = time.time()
            try:
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
                model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
                model.eval()
                with open(hf_hub_download(MODEL_NAME, "vocab.json"), encoding="utf-8") as f:
                    vocab = json.load(f)
            except Exception as e:
                raise PhonemeError("could not load %s: %s" % (MODEL_NAME, e))
            blank = model.config.pad_token_id
            if blank is None or vocab.get("<pad>") != blank:
                raise PhonemeError(
                    "cannot determine the CTC blank id safely (config pad_token_id=%r, "
                    "vocab['<pad>']=%r)" % (blank, vocab.get("<pad>")))
            PhonemeScorer.load_seconds = round(time.time() - t0, 1)
            PhonemeScorer._bundle = {
                "fe": feature_extractor, "model": model, "blank": int(blank),
                "vocab": vocab, "inv": {v: k for k, v in vocab.items()},
            }
        return self

    def _b(self):
        # type: () -> Dict[str, Any]
        if PhonemeScorer._bundle is None:
            self.load()
        return PhonemeScorer._bundle  # type: ignore

    # -- vocab --------------------------------------------------------------

    @property
    def vocab(self):
        # type: () -> Dict[str, int]
        return self._b()["vocab"]

    @property
    def inv_vocab(self):
        # type: () -> Dict[int, str]
        return self._b()["inv"]

    @property
    def blank_id(self):
        # type: () -> int
        return self._b()["blank"]

    def ids(self, ipa_tokens, label="sequence"):
        # type: (List[str], str) -> List[int]
        vocab = self.vocab
        missing = [t for t in ipa_tokens if t not in vocab]
        if missing:
            raise PhonemeError("%s: IPA tokens not in model vocab: %s"
                               % (label, ", ".join(missing)))
        return [vocab[t] for t in ipa_tokens]

    # -- inference ----------------------------------------------------------

    def logprobs(self, wav_float32):
        """Run a float32 mono 16 kHz clip through the model -> [T, V] log-softmax."""
        import torch
        b = self._b()
        wav = np.asarray(wav_float32, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        inputs = b["fe"](wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with PhonemeScorer.run_lock:
            with torch.no_grad():
                logits = b["model"](inputs.input_values).logits[0]
                logp = torch.log_softmax(logits, dim=-1)
        return logp.cpu().numpy().astype(np.float64)

    def score(self, logprobs, ipa_tokens, label="sequence"):
        # type: (np.ndarray, List[str], str) -> float
        """Length-normalised CTC forced log-likelihood (ll / len(ipa_tokens)).
        -inf when the clip has too few frames for the sequence."""
        ids = self.ids(list(ipa_tokens), label)
        ll = ctc_forced_loglik(logprobs, ids, self.blank_id)
        return ll / float(len(ids))

    def min_frames(self, ipa_tokens):
        # type: (List[str]) -> int
        """Frames needed to spell the sequence (2L+1, the extended length)."""
        return 2 * len(ipa_tokens) + 1

    def greedy_tokens(self, logprobs):
        # type: (np.ndarray) -> List[str]
        """Greedy CTC decode: per-frame argmax, collapse, drop blanks/specials."""
        path = [int(i) for i in np.argmax(logprobs, axis=-1)]
        ids = collapse(path, self.blank_id)
        inv = self.inv_vocab
        toks = [inv.get(i, "?") for i in ids]
        return [t for t in toks if t not in SPECIAL_TOKENS]

    def greedy(self, logprobs):
        # type: (np.ndarray) -> str
        return "".join(self.greedy_tokens(logprobs))


def scorer():
    # type: () -> PhonemeScorer
    """Convenience accessor for the shared scorer (does not load it)."""
    return PhonemeScorer()
