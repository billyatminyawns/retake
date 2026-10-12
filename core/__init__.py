# -*- coding: utf-8 -*-
"""Retake core — the pronunciation-QA engine (builder A).

Modules (see SPEC.md "core API"):
    lexicon     heteronym lexicon: load / validate / respelling lookup
    intent      occurrence detection + intended-sense engines (Claude, spaCy)
    transcribe  word-level transcription (faster-whisper default, mlx optional)
    phonemes    wav2vec2 phoneme scorer: forced CTC + greedy decode
    checker     media -> occurrences -> flags (the end-to-end check)
    fixer       re-render flagged lines through WellSaid and verify the take
    wellsaid    WellSaid REST client (render, timing, voices, respellings)
    media       ffmpeg helpers (probe / extract / cut / concat / splice)
    selftest    `python -m core.selftest` — no network, no models

Importing this package is cheap: every heavy dependency (torch, transformers,
faster-whisper, spaCy, anthropic) is imported lazily inside the module that
needs it, on first use, behind a lock.
"""
import os

__version__ = "1.0.0"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
LEXICON_PATH = os.path.join(DATA_DIR, "heteronyms.json")


def models_dir():
    # type: () -> str
    """Directory the ML models are cached in.

    `RETAKE_MODELS` when set (Docker sets it to /models), else ~/.cache.
    When `RETAKE_MODELS` is set and `HF_HOME` is not, `HF_HOME` is pointed at
    it so transformers, huggingface_hub and faster-whisper all share one
    cache — that is what makes the Docker pre-download and the runtime load
    resolve to the same files.
    """
    custom = os.environ.get("RETAKE_MODELS", "").strip()
    if custom:
        custom = os.path.abspath(os.path.expanduser(custom))
        if not os.environ.get("HF_HOME"):
            os.environ["HF_HOME"] = custom
        return custom
    return os.path.expanduser("~/.cache")


# Resolve once at import so any later `import transformers` sees HF_HOME.
models_dir()
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
