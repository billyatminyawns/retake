# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Retake API — hosted pronunciation QA for AI voiceover.
#
# Build from the PROJECT ROOT (the context must hold core/ api/ ui/ data/ requirements.txt):
#   docker build -f deploy/Dockerfile -t retake-api .
#   docker run --rm -p 8080:8080 --env-file deploy/.env -v retake_data:/data retake-api
#   open http://localhost:8080        (UI)   ·   http://localhost:8080/docs   (Swagger)
#
# The image bakes BOTH ML models (~2.7 GB) so a cold container answers its first job without
# downloading anything:
#   facebook/wav2vec2-lv-60-espeak-cv-ft        phoneme CTC scorer   (~1.2 GB)
#   faster-whisper large-v3-turbo (int8 on CPU) word timestamps      (~1.5 GB on disk)
# Build with --build-arg PREFETCH_MODELS=0 to skip the bake (models then download on first
# use into /models, which docker-compose mounts as a volume).
#
# Runtime contract: honors $PORT (Hugging Face Spaces uses 7860), runs as uid 1000,
# writes only under /data (jobs) and /models (model cache), HEALTHCHECK on GET /v1/health.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

ARG PREFETCH_MODELS=1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/retake \
    HF_HOME=/models \
    RETAKE_MODELS=/models \
    RETAKE_DATA=/data \
    RETAKE_WHISPER=faster \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8080

# ffmpeg: core/media.py resolves `which ffmpeg` here (imageio-ffmpeg is the Mac-only fallback).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Non-root user (uid 1000 is what Hugging Face Spaces expects) + the two writable dirs.
RUN useradd --create-home --uid 1000 --shell /bin/bash retake \
 && mkdir -p /app /models /data \
 && chown -R retake:retake /app /models /data

WORKDIR /app

# 1. Python deps. torch is installed from the CPU wheel index FIRST (pinned to whatever
#    requirements.txt pins) so the CUDA build — and ~3 GB of nvidia libraries — never gets
#    pulled in; everything else comes from requirements.txt (which pins the spaCy English
#    model wheel by URL — `spacy download` below is only the fallback if that pin is removed).
COPY requirements.txt ./
RUN set -eux; \
    TORCH_SPEC="$(grep -iE '^torch([=<>~!]|$)' requirements.txt | head -n1 | tr -d '[:space:]' || true)"; \
    pip install --index-url "$TORCH_INDEX_URL" "${TORCH_SPEC:-torch}"; \
    pip install --extra-index-url "$TORCH_INDEX_URL" -r requirements.txt; \
    python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null \
      || python -m spacy download en_core_web_sm; \
    python -c "import torch, transformers, faster_whisper, spacy, fastapi, uvicorn, anthropic; \
assert torch.version.cuda is None, 'CUDA torch slipped in; the image must use the CPU wheel'; \
spacy.load('en_core_web_sm'); print('deps ok: torch', torch.__version__, 'transformers', transformers.__version__)"

# 2. Engine + lexicon first (the model bake depends only on these), then the bake itself,
#    run AS the runtime user so /models never needs a second chown layer.
COPY --chown=retake:retake core/ core/
COPY --chown=retake:retake data/ data/
COPY --chown=retake:retake deploy/prefetch_models.py deploy/prefetch_models.py

USER retake

RUN if [ "$PREFETCH_MODELS" = "1" ]; then python deploy/prefetch_models.py; \
    else echo "PREFETCH_MODELS=0: models will download on first use into $RETAKE_MODELS"; fi

# 3. The service + UI (change most often → last layers).
COPY --chown=retake:retake api/ api/
COPY --chown=retake:retake ui/ ui/

EXPOSE 8080

# curl is not in python:slim; the health probe is stdlib Python. start-period covers a slow disk.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import os, sys, urllib.request; \
r = urllib.request.urlopen('http://127.0.0.1:%s/v1/health' % os.environ.get('PORT', '8080'), timeout=8); \
sys.exit(0 if r.status == 200 else 1)"

# $PORT is honored (Spaces: 7860). ONE worker on purpose: the models are not re-entrant and the
# job store is SQLite. --proxy-headers so status/file URLs carry the public scheme behind a proxy.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 75"]
