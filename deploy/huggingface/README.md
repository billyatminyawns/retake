---
title: Retake · pronunciation QA
emoji: 🎙️
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Heteronym pronunciation QA + fix loop for AI voiceover
---

# Retake — pronunciation QA for AI voiceover

This Space runs the **Retake API** (FastAPI, Docker). Upload any audio or video, or render a
script through WellSaid; Retake transcribes it, finds every heteronym (*REcord/reCORD,
live/live, read/read…*), decides from context which pronunciation the script meant, scores
what the voice actually said at the phoneme level, flags the mismatches with evidence, and
re-takes the flagged lines through WellSaid until a take verifies.

* Swagger: `https://<owner>-<space>.hf.space/docs` · health: `/v1/health`
* The UI is served at `/` (same build as https://billyatminyawns.github.io/retake).
* Source + full README: the `retake` repo this Space was pushed from (`deploy/hf-space.sh`).

## How this Space is built

`deploy/hf-space.sh` stages the project (core/, api/, ui/, data/, requirements.txt, the
Dockerfile and this README) into a throw-away git repo and pushes it to
`https://huggingface.co/spaces/<owner>/retake`. The Dockerfile is the same one used for
every other host (`deploy/Dockerfile`) with one line appended, `ENV PORT=7860`, because
Spaces route to `app_port` and do not inject `PORT`.

The build pre-downloads both models (≈2.8 GB: `facebook/wav2vec2-lv-60-espeak-cv-ft` and
`faster-whisper large-v3-turbo`) so the first request is served cold-but-ready. Expect a
10–15 minute first build.

## Settings → Variables and secrets

| Name | Kind | Purpose |
|---|---|---|
| `WELLSAID_API_KEY` | secret | render scripts, fix flagged lines (metadata GETs work without it) |
| `ANTHROPIC_API_KEY` | secret | Claude intent engine; unset = spaCy fallback |
| `RETAKE_API_KEYS` | secret | comma list of accepted `X-Api-Key` values; unset = open dev mode |
| `RETAKE_CORS_ORIGINS` | variable | `https://billyatminyawns.github.io` (or `*`) so the hosted UI can call this API |
| `RETAKE_INTENT_ENGINE` | variable | `both` (default) / `claude` / `spacy` |

`deploy/hf-space.sh --with-secrets` pushes the three secrets from your shell environment
through the Hub API (values are never printed). Secrets become available on the next
restart (Settings → Factory reboot is not needed; "Restart Space" is).

## Hardware

| Tier | Price | Notes |
|---|---|---|
| CPU basic (2 vCPU · 16 GB) | free | works; a 2-minute film takes several minutes; Space sleeps after 48 h idle (cold start ≈ 1 min) |
| CPU upgrade (8 vCPU · 32 GB) | ≈ $0.03/h | 3–4× faster transcription; recommended for a live demo |
| Persistent storage (small, 20 GB) | ≈ $5/mo | mounts at `/data` = `RETAKE_DATA`, so jobs survive restarts; without it, jobs live only until the next restart |

No GPU is needed: whisper-turbo runs int8 on CPU and the phoneme scorer is a single
wav2vec2 forward pass per word window.

## Limits worth knowing

* Uploads: the API accepts up to 500 MB (`RETAKE_MAX_UPLOAD_MB`); Spaces' proxy is fine with that.
* One worker thread: jobs run one at a time (the models are not re-entrant). Queue depth is unbounded.
* Everything in `/data` is lost on restart unless persistent storage is enabled.
* The Space is public by default. Set `RETAKE_API_KEYS` before sharing the URL.
