#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-download Retake's ML models (deploy/Dockerfile step 2; also handy on a fresh box).

The models are fetched THROUGH core's own loaders whenever `core` is importable, so the files
land exactly where the runtime will look for them (core.models_dir() -> $RETAKE_MODELS, which
also points HF_HOME there). If core is missing or its loader surface differs, the same model
ids are pulled directly from the Hugging Face Hub.

Models (same ids as core/phonemes.py and core/transcribe.py):
    facebook/wav2vec2-lv-60-espeak-cv-ft   phoneme CTC scorer (~1.2 GB safetensors)
    large-v3-turbo (faster-whisper)        -> mobiuslabsgmbh/faster-whisper-large-v3-turbo (~1.6 GB)

Usage:
    RETAKE_MODELS=/models python deploy/prefetch_models.py [--skip-whisper] [--skip-phonemes] [--direct]

No WellSaid or Anthropic calls. Safe to re-run: cached files are never downloaded twice.
Python 3.9+ compatible.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

WAV2VEC2_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
WHISPER_SIZE = "large-v3-turbo"  # faster-whisper alias -> mobiuslabsgmbh/faster-whisper-large-v3-turbo


def _log(msg):
    # type: (str) -> None
    sys.stdout.write("[prefetch] %s\n" % msg)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# wav2vec2 phoneme model
# ---------------------------------------------------------------------------

def _direct_wav2vec2():
    # type: () -> None
    """Mirror core/phonemes.py's load path: feature extractor + CTC model + vocab.json.
    (The checkpoint's tokenizer class is broken on transformers 4.57, so vocab.json is read
    directly — exactly what the runtime does.)"""
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_ID)
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_ID)
    with open(hf_hub_download(WAV2VEC2_ID, "vocab.json"), encoding="utf-8") as fh:
        vocab = json.load(fh)
    blank = model.config.pad_token_id
    if blank is None or vocab.get("<pad>") != blank:
        raise SystemExit("cannot determine the CTC blank id (config pad_token_id=%r, "
                         "vocab['<pad>']=%r)" % (blank, vocab.get("<pad>")))
    _log("phoneme model ready (direct): %s, vocab %d, blank id %d"
         % (WAV2VEC2_ID, len(vocab), blank))


def prefetch_phonemes(direct=False):
    # type: (bool) -> None
    if not direct:
        try:
            from core.phonemes import PhonemeScorer  # lazy singleton per SPEC
            scorer = PhonemeScorer()
            scorer.load()
            vocab = getattr(scorer, "vocab", None)
            _log("phoneme model ready via core.phonemes (vocab %s)"
                 % (len(vocab) if vocab else "?"))
            return
        except Exception as exc:  # noqa: BLE001 — fall back to the direct download
            _log("core.phonemes path unavailable (%s: %s); using the direct download"
                 % (type(exc).__name__, exc))
    _direct_wav2vec2()


# ---------------------------------------------------------------------------
# faster-whisper
# ---------------------------------------------------------------------------

def _silent_wav(path, seconds=1.0):
    # type: (str, float) -> str
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def _direct_whisper():
    # type: () -> None
    from faster_whisper.utils import download_model
    path = download_model(WHISPER_SIZE)
    _log("faster-whisper %s cached (direct) at %s" % (WHISPER_SIZE, path))


def prefetch_whisper(work_dir, direct=False):
    # type: (str, bool) -> None
    if not direct:
        try:
            from core.transcribe import transcribe
            wav = _silent_wav(os.path.join(work_dir, "silence16k.wav"))
            res = transcribe(wav, backend="faster")
            _log("whisper ready via core.transcribe (backend=%s, words on 1 s of silence=%d)"
                 % (res.get("backend"), len(res.get("words") or [])))
            return
        except Exception as exc:  # noqa: BLE001
            _log("core.transcribe path unavailable (%s: %s); using the direct download"
                 % (type(exc).__name__, exc))
    _direct_whisper()


# ---------------------------------------------------------------------------

def main(argv=None):
    # type: (list) -> int
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-whisper", action="store_true", help="only the phoneme model")
    ap.add_argument("--skip-phonemes", action="store_true", help="only faster-whisper")
    ap.add_argument("--direct", action="store_true",
                    help="bypass core's loaders and download from the Hub directly")
    args = ap.parse_args(argv)

    try:
        import core  # noqa: F401  (sets HF_HOME from RETAKE_MODELS at import)
        from core import models_dir
        where = models_dir()
    except Exception:  # noqa: BLE001
        where = os.environ.get("RETAKE_MODELS") or os.environ.get("HF_HOME") or "~/.cache (default)"
    _log("model cache: %s   (HF_HOME=%s)" % (where, os.environ.get("HF_HOME", "<unset>")))

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="retake-prefetch-") as tmp:
        if not args.skip_phonemes:
            prefetch_phonemes(direct=args.direct)
        if not args.skip_whisper:
            prefetch_whisper(tmp, direct=args.direct)
    _log("done in %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
