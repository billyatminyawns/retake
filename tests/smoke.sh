#!/usr/bin/env bash
# tests/smoke.sh — end-to-end smoke test against a RUNNING Retake API.
#
#   tests/smoke.sh [API_BASE] [KEY] [MEDIA]
#   tests/smoke.sh                                   # http://localhost:8080, open auth, the known film
#   tests/smoke.sh https://xxx.trycloudflare.com rtk_… ~/clip.mp4
#
# Flow: GET /v1/health → POST /v1/jobs (multipart media) → poll GET /v1/jobs/{id} until done
#       → assert ≥ 1 flagged occurrence → print the summary + report URL.
# With the default MEDIA (the known "autopilot-updates-ld-leaders.mp4" film) it also expects
# `re-records` and `live` among the flagged words: a WARN by default, fatal with STRICT=1.
#
# Env:  SMOKE_TIMEOUT  seconds to wait for the job (default 1800)      SMOKE_POLL  seconds (default 5)
#       SMOKE_ENGINE   both | claude | spacy (default: server default)
#       SMOKE_FIX=1    also POST /v1/jobs/{id}/fix for the flagged lines — RENDERS THROUGH WELLSAID
#                      (spends credits). SMOKE_SPEAKER (default 26 = Wade C.), SMOKE_MODEL (preview|caruso).
# Exit: 0 pass · 1 assertion / API failure · 2 usage.   Needs: curl, python3 (stdlib only).
set -euo pipefail

KNOWN_MEDIA="/Users/william/Downloads/wellsaid-ci-whitelabel/demo-lms/films/out/orig/autopilot-updates-ld-leaders.mp4"
API_BASE="${1:-${API_BASE:-http://localhost:8080}}"; API_BASE="${API_BASE%/}"
KEY="${2:-${RETAKE_API_KEY:-}}"
MEDIA="${3:-${MEDIA:-$KNOWN_MEDIA}}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-1800}"
SMOKE_POLL="${SMOKE_POLL:-5}"
STRICT="${STRICT:-0}"
PY="$(command -v python3 || true)"; [[ -x /usr/bin/python3 ]] && PY=/usr/bin/python3

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1;36m[%s]\033[0m %s\n' "$1" "$2"; }
pass() { printf '\033[1;32mPASS\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

[[ -n "$PY" ]] || { echo "python3 is required" >&2; exit 2; }
command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
[[ -f "$MEDIA" ]] || { echo "media file not found: $MEDIA" >&2; exit 2; }

HDR=(-H "Accept: application/json" -H "Bypass-Tunnel-Reminder: true")
[[ -n "$KEY" ]] && HDR+=(-H "X-Api-Key: $KEY")

# jp PATH  — print a dotted path from the JSON on stdin (strings raw, everything else as JSON, "" if missing)
jp() { "$PY" -c '
import json, sys
d = json.load(sys.stdin)
for part in sys.argv[1].split("."):
    if part == "": continue
    if isinstance(d, list):
        try: d = d[int(part)]
        except Exception: d = None
    elif isinstance(d, dict): d = d.get(part)
    else: d = None
    if d is None: break
if d is None: print("")
elif isinstance(d, str): print(d)
else: print(json.dumps(d))' "$1"; }

# call METHOD PATH [curl args…] → BODY, CODE
call() {
  local method="$1" path="$2" out; shift 2
  out="$(curl -sS -m "${CURL_TIMEOUT:-900}" "${HDR[@]}" -X "$method" "$API_BASE$path" "$@" -w $'\n%{http_code}')" \
    || { CODE=000; BODY=""; return 1; }
  CODE="${out##*$'\n'}"; BODY="${out%$'\n'*}"
}

bold "Retake smoke test → $API_BASE"
echo "media: $MEDIA ($(( $(stat -f %z "$MEDIA" 2>/dev/null || stat -c %s "$MEDIA") / 1024 / 1024 )) MB)"

# ── 1. health ─────────────────────────────────────────────────────────────────
step 1/4 "GET /v1/health"
call GET /v1/health || fail "no answer from $API_BASE/v1/health (is the API running? is the tunnel up?)"
[[ "$CODE" == "200" ]] || fail "health returned HTTP $CODE: $BODY"
echo "$BODY" | "$PY" -c '
import json, sys
h = json.load(sys.stdin)
print("  ok=%s version=%s auth=%s models=%s wellsaid=%s anthropic=%s" % (
    h.get("ok"), h.get("version"), h.get("auth"), json.dumps(h.get("models")), h.get("wellsaid"), h.get("anthropic")))'
[[ "$(echo "$BODY" | jp ok)" == "true" ]] || fail "health.ok is not true"
auth="$(echo "$BODY" | jp auth)"
if [[ "$auth" == "key" && -z "$KEY" ]]; then warn "server requires X-Api-Key and none was given — the job POST will be rejected"; fi
pass "health"

# ── 2. submit ─────────────────────────────────────────────────────────────────
step 2/4 "POST /v1/jobs (multipart media${SMOKE_ENGINE:+, engine=$SMOKE_ENGINE})"
extra=(); [[ -n "${SMOKE_ENGINE:-}" ]] && extra=(-F "engine=$SMOKE_ENGINE")
call POST /v1/jobs -F "media=@$MEDIA" ${extra[@]+"${extra[@]}"} || fail "upload failed (curl)"
[[ "$CODE" == "202" || "$CODE" == "200" ]] || fail "POST /v1/jobs returned HTTP $CODE: $BODY"
JOB="$(echo "$BODY" | jp job_id)"
[[ -n "$JOB" ]] || fail "no job_id in response: $BODY"
echo "  job_id=$JOB status_url=$(echo "$BODY" | jp status_url)"
pass "job accepted"

# ── 3. poll ───────────────────────────────────────────────────────────────────
step 3/4 "poll GET /v1/jobs/$JOB (timeout ${SMOKE_TIMEOUT}s)"
t0=$SECONDS; last=""
while :; do
  call GET "/v1/jobs/$JOB" || fail "poll failed (curl)"
  [[ "$CODE" == "200" ]] || fail "GET /v1/jobs/$JOB returned HTTP $CODE: $BODY"
  status="$(echo "$BODY" | jp status)"; stage="$(echo "$BODY" | jp stage)"; prog="$(echo "$BODY" | jp progress)"
  line="$status · ${stage:-—} · ${prog:-—}"
  if [[ "$line" != "$last" ]]; then printf '  %5ss  %s\n' "$((SECONDS - t0))" "$line"; last="$line"; fi
  case "$status" in
    done)  break ;;
    error) fail "job ended in error: $(echo "$BODY" | jp error)" ;;
  esac
  (( SECONDS - t0 < SMOKE_TIMEOUT )) || fail "timed out after ${SMOKE_TIMEOUT}s (status=$status stage=$stage)"
  sleep "$SMOKE_POLL"
done
pass "job done in $((SECONDS - t0))s"

# ── 4. assert + summary ───────────────────────────────────────────────────────
step 4/4 "result"
expect=""; [[ "$MEDIA" == "$KNOWN_MEDIA" ]] && expect="re-records,live"
echo "$BODY" | EXPECT_WORDS="$expect" STRICT="$STRICT" "$PY" -c '
import json, os, re, sys
job = json.load(sys.stdin)
res = job.get("result") or {}
occs = res.get("occurrences") or []
flagged = [o for o in occs if o.get("flag")]
nflags = res.get("flags")
if nflags is None: nflags = len(flagged)
print("  duration=%.1fs heteronyms=%d flagged=%d engine=%s" % (
    float(res.get("duration") or 0), len(occs), nflags,
    (occs[0].get("intent_engine") if occs else "—")))
for s in (res.get("summary") or [])[:12]:
    print("  · " + str(s))
for o in flagged:
    print("  FLAG %-14s %6.2fs  meant %-14s said %-14s conf=%-4s  %s" % (
        o.get("word"), float(o.get("t_start") or 0), o.get("intended"), o.get("detected"),
        o.get("confidence"), (o.get("reason") or "")[:70]))
rc = 0
if nflags < 1:
    print("FAIL expected at least one flagged occurrence, got 0", file=sys.stderr); rc = 1
norm = lambda w: re.sub(r"[^a-z0-9]", "", (w or "").lower())
expect = [w for w in os.environ.get("EXPECT_WORDS", "").split(",") if w]
if expect:
    got = {norm(o.get("word")) for o in flagged}
    missing = [w for w in expect if norm(w) not in got]
    if missing:
        msg = "known film: expected %s among the flagged words, missing %s (flagged: %s)" % (
            expect, missing, sorted(got))
        if os.environ.get("STRICT") == "1":
            print("FAIL " + msg, file=sys.stderr); rc = 1
        else:
            print("WARN " + msg, file=sys.stderr)
    else:
        print("  expected words flagged: %s" % ", ".join(expect))
sys.exit(rc)' || exit 1
pass "≥ 1 flag"
echo "  report: $API_BASE/v1/jobs/$JOB/files/report.html   (json: …/report.json)"

# ── optional: fix loop (spends WellSaid credits) ──────────────────────────────
if [[ "${SMOKE_FIX:-0}" == "1" ]]; then
  SPK="${SMOKE_SPEAKER:-26}"; MODEL="${SMOKE_MODEL:-preview}"
  step fix "POST /v1/jobs/$JOB/fix (speaker_id=$SPK model=$MODEL) — renders through WellSaid"
  call POST "/v1/jobs/$JOB/fix" -H 'Content-Type: application/json' \
       --data "{\"speaker_id\": $SPK, \"model\": \"$MODEL\"}" || fail "fix request failed (curl)"
  [[ "$CODE" == "202" || "$CODE" == "200" ]] || fail "fix returned HTTP $CODE: $BODY"
  FIX="$(echo "$BODY" | jp fix_id)"; [[ -n "$FIX" ]] || fail "no fix_id: $BODY"
  t1=$SECONDS; last=""
  while :; do
    call GET "/v1/jobs/$JOB/fixes/$FIX" || fail "fix poll failed"
    status="$(echo "$BODY" | jp status)"
    if [[ "$status" != "$last" ]]; then printf '  %5ss  %s\n' "$((SECONDS - t1))" "$status"; last="$status"; fi
    case "$status" in done) break ;; error) fail "fix ended in error: $(echo "$BODY" | jp error)" ;; esac
    (( SECONDS - t1 < SMOKE_TIMEOUT )) || fail "fix timed out"
    sleep "$SMOKE_POLL"
  done
  echo "$BODY" | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
for r in d.get("results") or []:
    best = r.get("best") or {}
    print("  %-6s tries=%d verified=%s heard_as=%s take=%s" % (
        r.get("occurrence_id"), len(r.get("tries") or []), r.get("verified"),
        best.get("heard_as"), best.get("take_mp3")))
print("  summary: %s" % d.get("summary"))'
  pass "fix loop finished"
fi

echo
bold "SMOKE OK  job=$JOB"
