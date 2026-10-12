# Retake — pronunciation QA for AI voiceover

**Retake listens to every heteronym in a voiceover, tells you which pronunciation the script
meant, which one the voice actually said, and re-takes the lines that are wrong.**

Upload any audio or video (or paste a script and render it through WellSaid). Retake
transcribes it, finds every heteronym — *REcord / reCORD, LIVE / live, READ / read,
PREsent / preSENT* — decides from context which pronunciation was intended, scores what the
voice said at the phoneme level, flags the mismatches with evidence you can hear, and fixes
flagged lines by re-rendering them through WellSaid until a take verifies.

| | |
|---|---|
| Hosted UI | https://billyatminyawns.github.io/retake (GitHub Pages; `?mock=1` for a no-API tour). Styled as a module inside WellSaid Studio — same rail, palette and type as studio.wellsaidlabs.com, dark by default with a light toggle |
| API | one Docker image, runs anywhere: Hugging Face Space, Fly.io, a laptop behind a tunnel |
| Local | `./run.sh` → http://localhost:8080 (UI at `/`, Swagger at `/docs`) |

---

## The bug WER can't see

The standard way to QA synthetic speech is to run ASR over it and compare the transcript to
the script (word error rate). Heteronyms are invisible to that check: whichever pronunciation
the voice picked, the ASR writes down the same word. "record" is "record" — WER stays zero
while the clip audibly says *the REH-kurd shows* where the script meant *rih-KORD your screen*.
Customers hear it in the first ten seconds. Transcript metrics never will.

Retake closes that gap by checking **which pronunciation was spoken**, not which word.

## What one job does

```
media ──► ffmpeg ──► 16 kHz wav
                       │
                       ├─► faster-whisper large-v3-turbo ─────────► words + timestamps
                       │                                               │
   script (optional) ──┴─► find heteronyms (lexicon of 100 + compounds: re-records, reads, mis-reads…)
                                                                       │
                           "what did the SCRIPT mean?"  ◄── Claude (opus 5.5) or spaCy POS ─┤
                                                                       │
                           "what did the VOICE say?"    ◄── wav2vec2 phoneme CTC: forced score of
                                                            every candidate pronunciation on the word
                                                            window + a free decode as evidence
                                                                       │
                                        intended ≠ detected  ──►  FLAG (reason, confidence, clips)
                                                                       │
                                  fix: re-render the sentence through WellSaid with the model's own
                                  mechanism, re-score the word, keep the first take that verifies
```

Forced scoring is deliberate: instead of asking a phoneme recognizer "what was said?" (noisy),
Retake asks "which of these two known pronunciations explains the audio better?" — a far
more reliable question, with the free decode kept as human-readable evidence.

## Architecture

```
                 browser ── https://billyatminyawns.github.io/retake  (GitHub Pages, static)
                 ┌───────────────────────────────────────────────────────────────┐
                 │  ui/   plain HTML/JS/CSS · drop media or paste a script        │
                 │        Settings drawer: API base + X-Api-Key (localStorage)    │
                 └───────────────────────────────┬───────────────────────────────┘
                                                 │ https · CORS (RETAKE_CORS_ORIGINS) · X-Api-Key
                                                 ▼
   ┌──────────────────────── Retake API · FastAPI · one Docker image ─────────────────────────┐
   │  api/     POST /v1/jobs · GET /v1/jobs/{id} · POST /v1/jobs/{id}/fix · /v1/check-text    │
   │           /v1/voices · /v1/lexicon · /v1/health · /docs · static ui/ at /                 │
   │           X-Api-Key auth · SQLite job store · ONE worker thread · file serving            │
   │  core/                                                                                    │
   │    media.py       ffmpeg: probe · 16 kHz extract · cut clips · splice takes               │
   │    transcribe.py  faster-whisper large-v3-turbo (int8, CPU) · optional mlx on Apple Si    │
   │    intent.py      which sense the SCRIPT meant: Claude or spaCy POS                        │
   │    phonemes.py    wav2vec2-lv-60-espeak-cv-ft · CTC forced log-likelihood · greedy decode │
   │    checker.py     occurrences (+ re-/un-/pre-/mis- compounds, -s/-es tails) → flags        │
   │    fixer.py       caruso: <respell> tag · preview: spelling variants → re-render → verify  │
   │    wellsaid.py    /tts/stream · /tts/word-timing · /tts/avatars · /tts/respelling_suggestions│
   │    lexicon.py     data/heteronyms.json — 100 words × 2 senses: IPA + WellSaid respellings  │
   └──────────────┬──────────────────────────────────────────────────┬─────────────────────────┘
                  │                                                  │
     /models  wav2vec2 (1.2 GB) + whisper-turbo (1.5 GB)      WellSaid API · Anthropic API
     /data    jobs.db + jobs/<id>/ media · clips · takes · report.json · report.html
```

- `mcp/` — Claude tools over this API (`retake_*` MCP server: check a file, list flags, fix a line); see [mcp/README.md](mcp/README.md).

## Quickstart (local, this Mac)

```bash
cd ~/Downloads/retake-cloud
# .venv already holds faster-whisper, torch (CPU), transformers, spaCy + en_core_web_sm, fastapi…
export WELLSAID_API_KEY=…      # or leave unset: falls back to ~/Downloads/wellsaid-connector/.env
export ANTHROPIC_API_KEY=…     # optional; unset = spaCy intent engine
# export RETAKE_API_KEYS=rtk_…  # optional; unset = open dev mode (health says so)
./run.sh                       # uvicorn api.main:app on :8080
open http://localhost:8080     # UI   ·   http://localhost:8080/docs   # Swagger
```

First run on a fresh machine downloads the two models (~2.7 GB) into `RETAKE_MODELS`
(default `~/.cache`); this Mac already has both cached. On Apple silicon you can switch
transcription to MLX with `RETAKE_WHISPER=mlx` (uses `/usr/bin/python3`'s mlx-whisper).

### Try it in one command

```bash
tests/smoke.sh                       # health → upload the known film → poll → assert flags → summary
tests/smoke.sh http://localhost:8080 "" ~/some/other.mp4
.venv/bin/python -m unittest tests.test_core -v     # pure-core tests, no network, no models
.venv/bin/python -m core.selftest                   # the engine's own CTC/compound/lexicon checks
```

The bundled test film (`autopilot-updates-ld-leaders.mp4`, 66 s) flags **re-records** and
**live** — both real mispronunciations the production voice made — in about 25 s, with
either intent engine (`SMOKE_ENGINE=spacy` reproduces it without an Anthropic key).

## API (v1)

JSON everywhere, `X-Api-Key` header when `RETAKE_API_KEYS` is set, errors as
`{"error": {"code", "message"}}`. Interactive docs with examples at `/docs` and `/redoc`.

| Method | Path | What |
|---|---|---|
| GET | `/v1/health` | `{ok, version, auth: open\|key, models: {whisper, phonemes: loaded\|cold}, wellsaid, anthropic}` |
| GET | `/v1/voices` | WellSaid voices (cached) `[{speaker_id, name, style, accent, gender}]` |
| GET | `/v1/lexicon` | the 100 heteronyms with senses (desc, IPA, respell candidates) |
| POST | `/v1/check-text` | `{text}` → occurrences + intended sense, no audio — script pre-flight |
| POST | `/v1/jobs` | multipart `media` (≤ 500 MB) + optional `script`, `engine`; **or** JSON `{text, speaker_id, model}` to render through WellSaid first → `202 {job_id, status_url}` |
| GET | `/v1/jobs` | newest 50 jobs |
| GET | `/v1/jobs/{id}` | `{status: queued\|running\|done\|error, stage, progress, result, error}` — result paths are URLs under `…/files/` |
| GET | `/v1/jobs/{id}/files/{name}` | media, word/sentence clips, takes, `report.json`, `report.html` |
| POST | `/v1/jobs/{id}/fix` | `{occurrence_ids?, speaker_id, model: preview\|caruso}` → `202 {fix_id}` |
| GET | `/v1/jobs/{id}/fixes/{fix_id}` | `{status, results: [{occurrence_id, tries, best, verified}], summary}` |
| DELETE | `/v1/jobs/{id}` | remove the job and its files |

### curl

```bash
BASE=http://localhost:8080          # or https://<your-api>
KEY=rtk_…                           # drop the header entirely in open (dev) mode

curl -s $BASE/v1/health | python3 -m json.tool

# script pre-flight: which heteronyms are in the text and what each one should sound like
curl -s -X POST $BASE/v1/check-text -H "X-Api-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"text":"Record the session, then file the record. We go live at nine."}'

# 1a. check a finished file (audio or video); optional script text and intent engine
curl -s -X POST $BASE/v1/jobs -H "X-Api-Key: $KEY" -F media=@film.mp4 -F engine=both
#     → {"job_id":"j_…","status_url":"/v1/jobs/j_…"}

# 1b. …or render a script through WellSaid first, then check the render
curl -s -X POST $BASE/v1/jobs -H "X-Api-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"text":"Please record your screen. The record shows we went live.","speaker_id":26,"model":"preview"}'

# 2. poll until status is done (stage/progress update as it goes)
curl -s $BASE/v1/jobs/$JOB -H "X-Api-Key: $KEY"

# 3. evidence and reports
curl -s $BASE/v1/jobs/$JOB/files/report.html -H "X-Api-Key: $KEY" -o report.html

# 4. fix the flagged lines (renders through WellSaid — this is the only step that spends credits)
curl -s -X POST $BASE/v1/jobs/$JOB/fix -H "X-Api-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"speaker_id":26,"model":"preview"}'        # → {"fix_id":"f_…"}
curl -s $BASE/v1/jobs/$JOB/fixes/$FIX -H "X-Api-Key: $KEY"
```

### Python

```python
import time, requests
BASE, H = "http://localhost:8080", {"X-Api-Key": "rtk_…"}

job = requests.post(f"{BASE}/v1/jobs", headers=H, files={"media": open("film.mp4", "rb")}).json()
while True:
    j = requests.get(f"{BASE}/v1/jobs/{job['job_id']}", headers=H).json()
    if j["status"] in ("done", "error"):
        break
    time.sleep(1.5)
for o in j["result"]["occurrences"]:
    if o["flag"]:
        print(o["t_start"], o["word"], "meant", o["intended"], "said", o["detected"], o["reason"])
```

Each occurrence carries: `word`, `t_start/t_end`, `sentence`, `intended` + `intended_desc` +
`intended_respell`, `detected` + `detected_desc`, per-sense `scores` (CTC log-likelihood per
phoneme), `margin`, `free_decode` (what the phoneme model heard, as IPA), `flag`, `reason`,
`confidence` (high/med/low), `clip_wav` and `sentence_wav` URLs, `intent_engine`.

### How a fix works

* **caruso** model: the sentence is re-rendered with an inline
  `<respell value="rih-KORD">record</respell>` using WellSaid's own respelling format (from the
  lexicon's candidates, else `GET /tts/respelling_suggestions`).
* **preview** (13.3) model: it garbles `<respell>` tags but honors spelling-as-pronunciation, so
  the fixer tries spelling variants (`ree-kordz` → `ri-kordz` → the respelling as-is).
* Every try re-renders the **whole sentence** (so prosody stays natural), re-scores the word with
  the same phoneme model, and stops at the first take that verifies. Up to 3 tries per line.
  True heteronyms are fixed per occurrence, never account-wide.

## Deploy

The UI is static and lives on GitHub Pages. The API is one Docker image; pick a host.

### 1. UI → GitHub Pages (`billyatminyawns.github.io/retake`)

```bash
API_BASE=https://<your-api-origin> deploy/pages-publish.sh
```

Idempotent: copies `ui/` → `docs/`, writes `docs/config.js` with `window.RETAKE_API_BASE`,
commits, creates the public repo `billyatminyawns/retake` if missing (`gh`), pushes `main`,
enables Pages from `/docs` through the REST API and prints the URL. Re-run whenever the API
URL changes — or just paste the new URL into the UI's Settings drawer. Set
`RETAKE_CORS_ORIGINS=https://billyatminyawns.github.io` on the API (or `*`).

### 2. API → Docker (any host)

```bash
docker build -f deploy/Dockerfile -t retake-api .          # bakes both models, ~10 min
docker run --rm -p 8080:8080 --env-file deploy/.env -v retake_data:/data retake-api
# or
cp deploy/.env.example deploy/.env && docker compose -f deploy/docker-compose.yml up --build -d
```

`python:3.11-slim` + ffmpeg, CPU-only torch, non-root uid 1000, `HEALTHCHECK` on
`/v1/health`, honors `$PORT`. `--build-arg PREFETCH_MODELS=0` skips the model bake (they then
download on first use into the `/models` volume). Needs ~8 GB RAM headroom for 500 MB uploads.

### 3. API → Hugging Face Space (free CPU tier works)

```bash
HF_TOKEN=hf_… deploy/hf-space.sh [owner/retake] [--with-secrets]
```

Stages the project with the same Dockerfile (+ `ENV PORT=7860`) and the Space front-matter
in `deploy/huggingface/README.md`, pushes it, optionally sets the secrets from your env.
Free "CPU basic" runs it (slowly; sleeps after 48 h idle); "CPU upgrade" (~$0.03/h) is the
demo-grade option; persistent storage mounts at `/data` so jobs survive restarts.

### 4. API → Fly.io (always-on, autostop when idle)

```bash
fly auth login
WELLSAID_API_KEY=… ANTHROPIC_API_KEY=… RETAKE_API_KEYS=… deploy/fly.sh
```

`deploy/fly.toml`: one `performance-2x` / 8 GB machine, a 10 GB volume at `/data`, health
checks on `/v1/health`, `auto_stop_machines` so it costs nothing while idle. Deployed with
`--ha=false` on purpose (SQLite job store, non-re-entrant models).

### 5. Laptop behind a tunnel (fastest path to a live demo)

```bash
./run.sh                 # terminal 1
deploy/tunnel.sh         # terminal 2 → https://<random>.trycloudflare.com, saved to deploy/.tunnel-url
deploy/pages-publish.sh  # terminal 3: bakes that URL into the hosted UI (or paste it in Settings)
```

Prefers a **cloudflared quick tunnel** (`brew install cloudflared`; free, no account, no
interstitial, 100 MB request cap). Falls back to **localtunnel** via `npx`, whose reminder
page is bypassed when every request carries `Bypass-Tunnel-Reminder: true` — `tests/smoke.sh`
sends it, and the UI's fetch helper (`authHeaders()` in `ui/app.js`) is where it belongs.

### Configuration

Everything is an environment variable — see `deploy/.env.example`. The ones that matter:

| Variable | Purpose |
|---|---|
| `WELLSAID_API_KEY` | render + fix (metadata GETs work without it) |
| `ANTHROPIC_API_KEY` | Claude intent engine; unset = spaCy |
| `RETAKE_API_KEYS` | comma list of accepted `X-Api-Key`; unset = open dev mode |
| `RETAKE_CORS_ORIGINS` | browser origins allowed (default `*`) |
| `RETAKE_DATA` / `RETAKE_MODELS` | job store / model cache (`/data`, `/models` in Docker) |
| `RETAKE_WHISPER` | `faster` (default) or `mlx` (Apple silicon) |
| `PORT` | listening port (8080; Spaces 7860) |

Keys are read server-side only, never logged, never sent to the browser.

## Models, storage, speed

* `facebook/wav2vec2-lv-60-espeak-cv-ft` (1.2 GB) — phoneme CTC; the tokenizer class is
  broken on transformers 4.57, so the vocab is read from `vocab.json` directly.
* `faster-whisper large-v3-turbo` (~1.5 GB on disk, quantized to int8 at load) — word timestamps on CPU.
* Both load lazily once per process, behind locks; transcription and scoring are serialized
  (one worker thread). Jobs queue; `GET /v1/jobs/{id}` shows stage + progress.
* Measured: the bundled 66 s test film checks in ~25 s on an Apple-silicon Mac with cold
  models (≈ 20 s transcribing, 5 s listening). Expect roughly real-time ÷ 2–3 on a Mac,
  ~1–2× real-time on a 2-vCPU container, faster with more cores. Peak RAM ≈ 4 GB.
* Everything a job produces lives under `RETAKE_DATA/jobs/<id>/` (media, wav, clips, takes,
  reports) and is served back at `/v1/jobs/{id}/files/…`; `DELETE` removes it.

## Costs

| Item | Cost |
|---|---|
| GitHub Pages UI | free |
| Hugging Face Space, CPU basic | free (sleeps after 48 h idle); CPU upgrade ≈ $0.03/h; persistent storage ≈ $5/mo |
| Fly.io | performance-2x · 8 GB ≈ $0.10/h while awake; autostop → pennies when idle; 10 GB volume ≈ $1.50/mo |
| cloudflared quick tunnel | free |
| WellSaid | **only** render-mode jobs and fixes spend credits: one sentence per render, ≤ 3 renders per flagged line. Checking uploaded media costs nothing. |
| Anthropic | one Claude request per 25 occurrences (opus 5.5); cents per job. Unset the key for a free spaCy run. |
| Models | one-time ~2.7 GB download (baked into the Docker image) |

Deployment scripts never print a key. `tests/smoke.sh` only spends WellSaid credits when
you pass `SMOKE_FIX=1`.

## Limitations (honest ones)

* **English, lexicon-bound.** Retake checks the 100 heteronyms in `data/heteronyms.json`
  (plus `re-/un-/pre-/mis-` compounds and plural tails). A word that is not a heteronym is not
  judged at all — this is a pronunciation-sense checker, not a general mispronunciation detector.
* **Stress is invisible.** The phoneme model emits no stress marks, so senses are separated by
  segmental differences only (vowel reduction/quality, e.g. /ɛ/ vs /ᵻ/ in REcord / reCORD).
  Pairs that differ purely in stress are kept out of the lexicon.
* **espeak-IPA approximation.** Lexicon pronunciations are espeak-style IPA mapped onto the
  wav2vec2 vocabulary — a good but not perfect model of US speech.
* **Two-candidate forced scoring.** The verdict is "which lexicon sense fits better"; a third,
  genuinely different mispronunciation is still scored against those two (the free decode and
  its edit distance are the safety net, and the usual source of extra flags on music beds).
* **Timing comes from whisper.** Word boundaries can clip a final syllable; scoring uses a
  small pad, the free decode a wider window. Garbled or overlapping speech lowers confidence.
* **Intent ceiling.** Without an Anthropic key the intended sense is a spaCy POS heuristic —
  fine on clear sentences, weak on ambiguous ones. Flag quality can never beat intent quality.
  Supplying the script (`script` field) removes transcription errors from that step.
* **Preview-model fixes are best-effort.** Preview (13.3) ignores `<respell>`; spelling variants
  usually land but are not guaranteed, hence the verify loop and the explicit ✓/✗ per take.
* **One worker, one box.** Jobs run sequentially; the job store is local SQLite, so the API is
  a single instance (no horizontal scaling without moving the store).
* **Free-tier realities.** HF free Spaces sleep and lose `/data` without persistent storage;
  Cloudflare quick tunnels cap a request at 100 MB and change hostname every start.

## Layout

```
core/      engine: lexicon · intent · transcribe · phonemes · checker · fixer · wellsaid · media · selftest
api/       FastAPI: main · auth · jobs (SQLite + worker) · schemas · routes_* · engine bridge
ui/        static SPA: index.html · app.js · style.css · config.js · mock.json
data/      heteronyms.json (100 words) · test_sentences.jsonl
deploy/    Dockerfile · docker-compose.yml · .env.example · prefetch_models.py
           huggingface/README.md + hf-space.sh · fly.toml + fly.sh · pages-publish.sh · tunnel.sh
tests/     smoke.sh (curl flow against a running API) · test_core.py (pure-core unit tests)
run.sh     local uvicorn on :8080 · requirements.txt · SPEC.md (the build contract)
```

Built on the heteronym verifier proven in `~/Downloads/heteronym-poc` (CTC forced scoring on
wav2vec2 + whisper word windows), productized as a hosted API + UI for WellSaid Labs.
