# -*- coding: utf-8 -*-
"""Word-level transcription with two interchangeable backends.

    transcribe(wav16k_path, backend="auto") -> {"text", "words": [{"w","s","e","p"}], "backend"}

  * "faster" (default, portable): faster-whisper `large-v3-turbo` on CPU,
    int8, word_timestamps=True. Loaded lazily once per process.
  * "mlx": mlx_whisper (Apple silicon, importable only by /usr/bin/python3)
    run as a subprocess exactly as heteronym-poc check.py does.
  * "auto": env RETAKE_WHISPER ("faster" | "mlx"), default "faster".

Both return the same shape: `w` keeps Whisper's leading space on word starts
(" re" + "-records"), which is what lets the checker rejoin hyphenated
compounds. Transcription is serialized by a module lock (the model is not
re-entrant).
"""
import json
import os
import subprocess
import threading
import time
from typing import Any, Dict, Optional

from core import models_dir  # noqa: F401  (sets HF_HOME from RETAKE_MODELS)

FASTER_MODEL = "large-v3-turbo"
MLX_REPO = "mlx-community/whisper-large-v3-turbo"
SYSTEM_PY = "/usr/bin/python3"
LANGUAGE = "en"

_MODEL = None
_LOAD_LOCK = threading.Lock()
RUN_LOCK = threading.RLock()  # serializes transcription calls
_LAST_INFO = {}  # type: Dict[str, Any]


class TranscribeError(RuntimeError):
    """Backend unavailable or the transcription failed."""


# Runs under /usr/bin/python3 (the only interpreter that can import mlx_whisper).
MLX_SCRIPT = r'''
import json, sys, wave
import numpy as np
import mlx_whisper
wav_path, repo = sys.argv[1], sys.argv[2]
with wave.open(wav_path, "rb") as wf:
    if wf.getnchannels() != 1 or wf.getframerate() != 16000 or wf.getsampwidth() != 2:
        raise SystemExit("expected 16 kHz mono pcm_s16le wav, got ch=%d sr=%d sw=%d"
                         % (wf.getnchannels(), wf.getframerate(), wf.getsampwidth()))
    pcm = wf.readframes(wf.getnframes())
audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
res = mlx_whisper.transcribe(audio, path_or_hf_repo=repo, word_timestamps=True,
                             language="en")
words = []
for seg in res.get("segments", []):
    for w in seg.get("words", []):
        words.append({"w": w["word"], "s": float(w["start"]), "e": float(w["end"]),
                      "p": float(w.get("probability", 0.0))})
sys.stdout.write(json.dumps({"text": res.get("text", ""), "words": words},
                            ensure_ascii=False))
'''


def resolve_backend(backend="auto"):
    # type: (str) -> str
    b = (backend or "auto").strip().lower()
    if b == "auto":
        b = os.environ.get("RETAKE_WHISPER", "faster").strip().lower() or "faster"
    if b in ("faster", "faster-whisper", "faster_whisper", "whisper"):
        return "faster"
    if b == "mlx":
        return "mlx"
    raise TranscribeError("unknown whisper backend %r (use faster or mlx)" % backend)


def status():
    # type: () -> str
    """"loaded" once the faster-whisper model is resident, else "cold"."""
    return "loaded" if _MODEL is not None else "cold"


def load():
    """Load the faster-whisper model once (thread-safe). Returns the model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise TranscribeError("faster-whisper is not installed: %s" % e)
        threads = int(os.environ.get("RETAKE_WHISPER_THREADS", "0") or 0)
        kwargs = {"device": "cpu", "compute_type": "int8"}
        if threads > 0:
            kwargs["cpu_threads"] = threads
        t0 = time.time()
        try:
            model = WhisperModel(FASTER_MODEL, **kwargs)
        except Exception as e:
            raise TranscribeError("could not load faster-whisper %s: %s" % (FASTER_MODEL, e))
        _LAST_INFO["load_seconds"] = round(time.time() - t0, 1)
        _MODEL = model
    return _MODEL


def _transcribe_faster(wav16k_path):
    # type: (str) -> Dict[str, Any]
    model = load()
    with RUN_LOCK:
        segments, info = model.transcribe(wav16k_path, language=LANGUAGE,
                                          word_timestamps=True, beam_size=5,
                                          condition_on_previous_text=False)
        words = []
        texts = []
        for seg in segments:  # generator: consumed under the lock
            texts.append(seg.text)
            for w in (seg.words or []):
                words.append({"w": w.word, "s": float(w.start), "e": float(w.end),
                              "p": float(w.probability)})
    return {"text": "".join(texts).strip(), "words": words, "backend": "faster",
            "duration": float(getattr(info, "duration", 0.0) or 0.0)}


def _transcribe_mlx(wav16k_path):
    # type: (str) -> Dict[str, Any]
    if not os.path.exists(SYSTEM_PY):
        raise TranscribeError("%s not found (needed to run mlx_whisper)" % SYSTEM_PY)
    env = dict(os.environ)
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    with RUN_LOCK:
        try:
            proc = subprocess.run([SYSTEM_PY, "-c", MLX_SCRIPT, wav16k_path, MLX_REPO],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  env=env, timeout=1800)
        except subprocess.TimeoutExpired:
            raise TranscribeError("mlx whisper subprocess timed out")
    if proc.returncode != 0:
        raise TranscribeError("mlx whisper subprocess failed: %s"
                              % proc.stderr.decode("utf-8", "replace").strip()[-600:])
    try:
        data = json.loads(proc.stdout.decode("utf-8"))
    except ValueError as e:
        raise TranscribeError("mlx whisper subprocess printed invalid JSON (%s)" % e)
    data["backend"] = "mlx"
    data["text"] = (data.get("text") or "").strip()
    return data


def transcribe(wav16k_path, backend="auto"):
    # type: (str, str) -> Dict[str, Any]
    """Transcribe a 16 kHz mono wav with word timestamps.

    Returns {"text": str, "words": [{"w","s","e","p"}], "backend": str}.
    """
    if not os.path.isfile(wav16k_path):
        raise TranscribeError("wav not found: %s" % wav16k_path)
    b = resolve_backend(backend)
    if b == "mlx":
        return _transcribe_mlx(wav16k_path)
    return _transcribe_faster(wav16k_path)


def transcribe_cached(wav16k_path, cache_json, backend="auto"):
    # type: (str, str, str) -> Dict[str, Any]
    """transcribe() with a JSON cache next to the wav (idempotent re-runs)."""
    if cache_json and os.path.exists(cache_json):
        try:
            with open(cache_json, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("words"), list):
                return data
        except (OSError, ValueError):
            pass
    data = transcribe(wav16k_path, backend=backend)
    if cache_json:
        d = os.path.dirname(cache_json)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = cache_json + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, cache_json)
    return data
