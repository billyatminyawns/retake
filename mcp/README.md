# Retake MCP server

Retake's pronunciation QA as Claude tools. This is a thin MCP layer over the
Retake HTTP API (`../api`): the server, wherever it runs (laptop, tunnel,
Hugging Face Space, Fly), does the transcription, phoneme scoring and WellSaid
renders; this process only calls it and formats the answers for a model.

```
Claude / any MCP client ──stdio or Streamable HTTP──► retake-mcp ──HTTPS + X-Api-Key──► Retake API ──► whisper · wav2vec2 · WellSaid
```

## Tools

| Tool | What it does | Spends WellSaid credits |
|---|---|---|
| `retake_health` | API version, auth mode, model warm state, configured keys | no |
| `retake_list_voices` | WellSaid voices (`query` filter) → `speaker_id` for fixes/renders | no |
| `retake_check_text` | Pre-flight a script: every heteronym + the pronunciation it should get | no |
| `retake_check_pronunciation` | Check `media_path` / `media_url` / `text`+`speaker_id`+`model`; waits and returns flagged words with evidence (meant vs heard, phonemes, CTC margin, clip + report URLs) | only the `text` route (one render) |
| `retake_get_job` · `retake_list_jobs` | Poll / browse check jobs | no |
| `retake_fix_pronunciation` | Re-render flagged lines through WellSaid with the intended pronunciation, verify every take by ear, return tries + best take URL | yes (≤ `max_tries` renders per line, default 3) |
| `retake_get_fix` | Poll a fix | no |

Every result is a short human summary followed by a JSON object with the same
facts (absolute URLs, ids to pass to the next tool). Long waits send MCP
progress notifications, so clients that reset their timeout on progress
(Claude Desktop, Claude Code) can wait through a multi-minute check; pass
`wait: false` to get an id back immediately and poll instead.

## Run

```bash
cd mcp && npm install
RETAKE_API_BASE=http://localhost:8080 RETAKE_API_KEY=rtk_… npm run doctor   # connectivity check
npm start            # stdio (what Claude Desktop / Claude Code spawn)
npm run start:http   # Streamable HTTP on :8788/mcp (PORT env) for a remote connector
npm test             # mock-API test (no models, no network) + stdio smoke
npm run inspect      # MCP Inspector UI
```

Configuration is two environment variables, read from the process env, then
`mcp/.env`, then `../.env` (the project root — so a local API and this server
share `RETAKE_API_KEYS`):

| Variable | Meaning |
|---|---|
| `RETAKE_API_BASE` | API origin (default `http://localhost:8080`) |
| `RETAKE_API_KEY` | sent as `X-Api-Key`; falls back to the first entry of `RETAKE_API_KEYS`; omit when the API reports `auth: open` |

Keys are never logged. In stdio mode nothing but JSON-RPC is written to stdout.

### Claude Desktop / Claude Code

Merge `claude_desktop_config.example.json` into your MCP config (fix the path,
API base and key), restart the client, and ask:

> Check `/path/to/film.mp4` for mispronounced heteronyms, then fix them with
> Jennie V. on the preview model.

Claude will call `retake_check_pronunciation`, read the flagged words, and call
`retake_fix_pronunciation` with `speaker_id` 169 and `model: "preview"`.

### Live run (spends credits on the fix step)

```bash
node test/live.smoke.js check ~/film.mp4 [script.txt]
node test/live.smoke.js fix <job_id> 169 preview [o3 o4]
node test/live.smoke.js tool retake_check_text '{"text":"Record the record."}'
```

## Layout

```
src/index.js    entry: stdio | --http | --doctor; loads .env files
src/server.js   McpServer + the retake_* tools (zod schemas, formatting)
src/retake.js   tiny Retake API client (fetch, multipart upload, polling)
test/           mock-api.test.js (in-memory transport vs a mock API) · stdio.smoke.js · live.smoke.js
```

Node ≥ 18.17 (global `fetch`/`FormData`/`Blob`). Dependencies: `@modelcontextprotocol/sdk`, `zod`.
