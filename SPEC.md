# Retake — hosted pronunciation QA for AI voiceover (API + UI)

Product: upload any audio/video (or a script to render through WellSaid); Retake transcribes it,
finds every heteronym (REcord/reCORD, live/live…), decides from context which pronunciation was
intended, scores what the voice actually said at the phoneme level, flags mismatches with evidence,
and fixes flagged lines by re-rendering through WellSaid with the model-appropriate mechanism.
Brand: "Retake". Billy Sheng (WellSaid Labs) demos it to the product team; it must look and feel
like a product, not a notebook.

Project root: `/Users/william/Downloads/retake-cloud/`. Deploy shape: UI → GitHub Pages
(billyatminyawns.github.io/retake), API → Docker container (any host; Hugging Face Spaces + Fly
configs provided), also runnable locally (`.venv`, Python 3.9 on this Mac; Docker uses python:3.11).

## Proven engine to port (do NOT reinvent; copy + adapt)
`/Users/william/Downloads/heteronym-poc/src/` — `lexicon.py` (load/validate; IPA tokens in the
wav2vec2-lv-60-espeak-cv-ft vocab), `analyze.py` (occurrence detection `detect_occurrences`,
intent engines: Claude `claude-opus-5-5` via anthropic SDK stable surface, NO `temperature`
param; spaCy en_core_web_sm fallback), `verify.py` (model load: `Wav2Vec2FeatureExtractor` +
`Wav2Vec2ForCTC` + vocab via `hf_hub_download(MODEL,'vocab.json')` — the tokenizer class returns
False on transformers 4.57, never use it; `ctc_forced_loglik` log-space forward; `_clip_logprobs`),
`check.py` (end-to-end media check: compounds re-/un-/pre-/mis- + plural tails; whisper word
windows; CTC candidate scoring + fuzzy free-decode containment; flag rules), `wellsaid_client.py`
(REST: /tts/stream, /tts/word-timing zip → Google-STT-style timing JSON with "0.560s" strings,
/tts/avatars, /tts/respelling_suggestions; rate limit; key from env WELLSAID_API_KEY else
~/Downloads/wellsaid-connector/.env). `data/heteronyms.json` (100 words, each sense has `ipa`
list + `respell_candidates` in WellSaid format). Read these files first; their docstrings and the
SPEC at heteronym-poc/SPEC.md are the ground truth for behavior.

Tested WellSaid pronunciation facts (10/11/26; see memory wellsaid-pronunciation-mechanics):
caruso honors inline `<respell value="REH-kurdz">records</respell>`; preview (13.3) garbles
respellings but honors spelling-as-pronunciation ("ree-recordz", "lyve", "livs") and plain-TEXT
replacement libraries; hyphenated words never match libraries; true heteronyms are fixed PER
OCCURRENCE, never account-wide. Respelling strings must come from WellSaid's own format
(`respell_candidates`, or GET /tts/respelling_suggestions).

## Environment facts
- Local: `/usr/bin/python3` 3.9.6; project venv `.venv` already has faster-whisper 1.2.1 +
  ctranslate2 (CPU). Still to install into `.venv` (builder A does it, separately per package, in
  this order): torch, transformers, protobuf, sentencepiece, huggingface_hub, soundfile, numpy
  (whatever pip resolves for cp39), "spacy<3.8" + `python -m spacy download en_core_web_sm`,
  anthropic, fastapi, "uvicorn[standard]", python-multipart, pydantic. No other deps.
- ffmpeg: `imageio_ffmpeg.get_ffmpeg_exe()` from /usr/bin/python3 (`subprocess` lookup) locally;
  in Docker `apt-get install ffmpeg` and `which ffmpeg`. core must resolve either.
- Whisper backend: faster-whisper `WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")`
  with `word_timestamps=True` (portable, default). Optional backend `mlx` (mlx_whisper, only on
  Apple silicon with /usr/bin/python3 — run via subprocess exactly as check.py does) selected by env
  `RETAKE_WHISPER=mlx`. Both return the same {text, words:[{w,s,e,p}]} shape.
- Models download on first use into `RETAKE_MODELS` (default ~/.cache); Docker pre-downloads at
  build. wav2vec2 weights ~1.2 GB, whisper turbo int8 ~0.8 GB.
- Keys: `WELLSAID_API_KEY`, `ANTHROPIC_API_KEY` (server env; both exist on this Mac — Anthropic in
  ~/.zshenv, WellSaid in the connector .env fallback). Product auth: `RETAKE_API_KEYS` env, comma
  list; requests send `X-Api-Key`. If `RETAKE_API_KEYS` is unset, auth is open (dev mode) and the
  health endpoint says so. Never log keys.

## Layout
```
core/            engine package (builder A)         api/            FastAPI app (builder B)
  __init__.py                                        __init__.py
  lexicon.py     ← port                              main.py       app factory, routers, CORS, static
  intent.py      ← analyze.py engines                auth.py       X-Api-Key dependency
  transcribe.py  faster-whisper | mlx backends       jobs.py       SQLite store + worker thread
  phonemes.py    ← verify.py model/CTC/greedy         schemas.py    pydantic models (the API contract)
  checker.py     ← check.py media → occurrences      routes_*.py   jobs, fix, check_text, lexicon, health
  fixer.py       strategies + verify loop             openapi tags + examples
  wellsaid.py    ← wellsaid_client.py (+ render)
  media.py       ffmpeg helpers (extract/cut/stitch)
ui/              static SPA (builder C): index.html, app.js, style.css, config.js
deploy/          (builder D): Dockerfile, docker-compose.yml, huggingface/README.md (Space config),
                 fly.toml, pages-publish.sh, tunnel.sh, .env.example
tests/           (builder D): smoke.sh (curl flow against a running API), test_core.py (pure-core
                 unit tests: ctc selftest, compounds, lexicon validate)
README.md        (builder D)   run.sh (builder B: local uvicorn)   requirements.txt (builder A)
data/            heteronyms.json, test_sentences.jsonl (present)
```
Python 3.9-compatible everywhere (no match, no `X | Y`).

## core API (builder A) — exact signatures other builders code against
```python
# core/transcribe.py
def transcribe(wav16k_path: str, backend: str = "auto") -> dict  # {"text": str, "words": [{"w","s","e","p"}], "backend": str}
# core/phonemes.py
class PhonemeScorer:  # lazy singleton: .load(), .logprobs(wav_float32)->np[T,V], .score(logprobs, ipa_tokens)->float (ll/len), .greedy(logprobs)->str (IPA), .vocab
# core/checker.py
def check_media(media_path: str, *, script: str = None, engine: str = "both", pad: float = 0.1,
                work_dir: str, progress=None) -> dict
# returns {"media": path, "duration": float, "transcript": {"text","words"},
#          "occurrences": [Occurrence...], "flags": int, "summary": [str...]}
# Occurrence = {"id": "o3", "t_start","t_end","word","sentence","base","intended","intended_desc",
#   "intended_ipa","intended_respell","detected","detected_desc","scores":{sense:ll},"margin",
#   "free_decode","flag": bool,"reason","confidence": "high|med|low","clip_wav": path,
#   "sentence_wav": path,"sentence_t_start","sentence_t_end","candidates":{sense: ipa_str},
#   "intent_engine"}
# core/fixer.py
def fix_occurrence(occ: dict, *, speaker_id: int, model: str, work_dir: str, max_tries: int = 3,
                   progress=None) -> dict
# strategy: model=="caruso" → inline <respell value=intended_respell>word</respell> (respell from
#   lexicon respell_candidates matching the intended sense, else GET respelling_suggestions);
#   model=="preview" → spelling variants: [respell lowercased w/o stress caps, e.g. "ree-kordz"],
#   ["ri-kordz"], [intended_respell as-is]; render the FULL sentence (occ["sentence"]) each try,
#   re-check that word with PhonemeScorer (CTC + greedy), stop at first verified take.
# returns {"occurrence_id","tries":[{"text","take_mp3","heard_as","detected","margin","verified"}],
#          "best": take dict|None, "verified": bool}
# core/wellsaid.py: WellSaid(key=None).render(text, speaker_id, model, fmt, sample_rate)->bytes,
#   .render_timed(...)->{audio_bytes, timing:[{word,start,end}]}, .voices(), .respell_suggestions(w)
```
`check_media` and `fix_occurrence` call `progress(stage:str, frac:float)` if given. All heavy
models load lazily once per process (thread-safe locks); transcription and scoring are serialized
by a module lock. Add `core/selftest.py` (`python -m core.selftest`): CTC brute-force check, compound
candidates, lexicon validation — no network, no models.

## HTTP API (builder B) — v1, JSON, `X-Api-Key`
- `GET /v1/health` → {ok, version, auth: "open|key", models: {whisper: "loaded|cold", phonemes:…},
  wellsaid: bool, anthropic: bool}
- `GET /v1/voices` → WellSaid voices (cached) `[{speaker_id,name,style,accent,gender}]`
- `GET /v1/lexicon` → words + senses (desc, ipa, respell_candidates)
- `POST /v1/check-text` {text} → occurrences with intended sense (no audio) — script pre-flight
- `POST /v1/jobs` multipart (`media` file up to 500 MB, optional `script` text, `engine`) OR JSON
  {text, speaker_id, model} (render first via WellSaid, then check) → 202 {job_id, status_url}
- `GET /v1/jobs` (list, newest first, 50) ; `GET /v1/jobs/{id}` → {id, status: queued|running|done|
  error, stage, progress, created, mode, result: check_media result with every path rewritten to
  URLs under /v1/jobs/{id}/files/…, error}
- `GET /v1/jobs/{id}/files/{name}` (media, clips, takes, report.json, report.html) — FileResponse
- `POST /v1/jobs/{id}/fix` {occurrence_ids?: [..] (default all flagged), speaker_id, model:
  "preview"|"caruso"} → 202 {fix_id}; `GET /v1/jobs/{id}/fixes/{fix_id}` → {status, results:[fix
  results with take URLs], summary}
- `DELETE /v1/jobs/{id}`
- `/docs` (Swagger) and `/redoc` enabled with tags + examples; `/` serves `ui/` (static) when present.
Job store: SQLite `RETAKE_DATA/jobs.db` + `RETAKE_DATA/jobs/<id>/`. One worker thread consumes a
queue (models are not re-entrant); progress written to the DB. CORS: `RETAKE_CORS_ORIGINS` env
(default `*` in dev). Upload size limit + media type sniff via ffmpeg probe (reject non-media).
`run.sh`: `uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}` with `.venv`. Structured
logs, no keys. Errors → `{error: {code, message}}`.

## UI (builder C) — plain HTML/JS/CSS, no build, no CDN, served at `/` and from GitHub Pages
`ui/config.js` sets `window.RETAKE_API_BASE` (default: same origin; the Pages build points at the
API URL) and the API key is entered once in a Settings drawer and kept in localStorage. Dark
studio look consistent with ~/Downloads/retake/static (charcoal #101014, cards #1a1a21, coral
#ff5f56, teal #28c8b7, grade colors). Wordmark "RETAKE · pronunciation QA". Views:
1. Home: drop zone (audio/video) + "or paste a script" (render via WellSaid: voice picker + model
   preview/caruso), optional script textarea for uploads; Settings (API base, key); recent jobs list.
2. Job: progress (stages: uploading → transcribing → finding heteronyms → deciding intent → listening
   → done) polling every 1.5 s.
3. Results: header (file, duration, N heteronyms, N flagged, overall verdict), transcript with every
   heteronym highlighted (flagged = coral underline, clean = teal), cards per occurrence: word,
   timestamp, sentence, "You meant" (sense desc + respelling), "It said" (sense + free-decoded IPA),
   confidence chip, players for word clip + sentence clip; filter All / Flagged. "Fix flagged"
   panel: voice picker + model → run → per occurrence before/after players, heard-as, verified ✓/✗,
   download take; "Download report" (JSON + HTML). Copy-curl snippet for the same job (API page).
4. API page: health, your key, curl + Python examples for POST /v1/jobs, poll, fix.
Errors → toasts; works at phone width; `?mock=1` renders a bundled `ui/mock.json` job (complete
schema sample: 6 occurrences, 3 flagged incl. one re-records compound, a fix result with 2 tries).

## Deploy (builder D)
- `deploy/Dockerfile`: python:3.11-slim, ffmpeg, requirements, pre-download both models
  (`python -c` snippet) into /models (ENV HF_HOME, RETAKE_MODELS), non-root, `EXPOSE 8080`, CMD uvicorn.
  Also honor `PORT` env (HF Spaces uses 7860).
- `deploy/docker-compose.yml` (volumes: data, models; env_file .env).
- `deploy/huggingface/README.md` with the Space front-matter (`sdk: docker`, `app_port: 7860`) and
  `deploy/hf-space.sh` (creates/pushes the Space via git with HF token env) — documented, not run.
- `deploy/fly.toml` + `deploy/fly.sh` (documented, not run).
- `deploy/pages-publish.sh`: copies `ui/` into `docs/` with `config.js` pointing at `$API_BASE`,
  commits, pushes to GitHub repo `billyatminyawns/retake` main, enables Pages from /docs via
  `gh api` (idempotent). `deploy/tunnel.sh`: starts a cloudflared quick tunnel to the local API and
  prints the public URL (documents `brew install cloudflared`).
- `tests/smoke.sh API_BASE KEY MEDIA`: health → POST job → poll → assert ≥1 flag for the known file
  `/Users/william/Downloads/wellsaid-ci-whitelabel/demo-lms/films/out/orig/autopilot-updates-ld-leaders.mp4`
  (expects re-records + live flagged) → print summary. `tests/test_core.py` runs core.selftest.
- README.md: product story, architecture diagram (ascii), quickstart local, API reference summary,
  deploy (Pages + Docker + HF + Fly + tunnel), cost notes, limitations.

## Rules for all builders
No WellSaid renders and no Anthropic calls in self-tests unless the task explicitly says so (metadata
GETs fine). Don't download the 1.2 GB model in self-tests unless already cached (`~/.cache/huggingface`
has it; faster-whisper turbo may need a download — allowed once, it's cached). Python 3.9. Every CLI
and endpoint has graceful errors. Write only your own files. Report deviations.

## BRANDING OVERRIDE (10/11/26, Billy): "make sure you have the Studio WellSaid branding so it feels like the same product"
The UI must look like a module inside WellSaid Studio, not a separate dark app. Source of truth is
`ui/brand/` (copied from the WellSaid white-label skill): `colors.css` (Figma primitives — mint is
primary/CTA, lavender secondary/nav fills, sand neutrals, gray text ramp, red error, sunflower warning,
azure info), `skin.css` (the Studio skin), `shell.js.txt` (Studio left rail + top bar shell: Projects,
Studio, Voices, Libraries, API, Integrations linking to https://studio.wellsaidlabs.com/dashboard/…),
`wellsaid-logo.svg`, `wellsaid-mark.svg`, `favicon.png`, `fonts/` (DM Sans display, Assistant body —
bundle via @font-face, no Google Fonts). Default theme dark (Studio: cards #1e1e1e on #111, text
#fcfcfc, mint-300 #7fed9b accents, lavender-500 #63477e nav fills) with a light toggle (sand-50
surfaces, gray-1000 text, mint-500 CTA). Page title format "WellSaid – Retake". The Retake wordmark
sits in the rail as the active module ("Retake · Pronunciation QA"). Flag/verdict colors: flagged =
red-300/400, clean = mint-300/400, warning = sunflower-300. No coral (#ff5f56) anywhere.
