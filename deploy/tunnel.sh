#!/usr/bin/env bash
# deploy/tunnel.sh — put the LOCAL Retake API on a public https URL so the hosted UI
# (https://billyatminyawns.github.io/retake) can talk to the API running on this Mac.
#
#   ./run.sh            # terminal 1: API on :8080
#   deploy/tunnel.sh    # terminal 2: prints https://<random>.trycloudflare.com and keeps running
#
# Preferred: cloudflared quick tunnel — free, no account, no interstitial:   brew install cloudflared
# Fallback:  localtunnel through npx (needs node):   npx localtunnel --port 8080
#            localtunnel answers browser requests with a "tunnel reminder" page unless the request
#            carries the header  Bypass-Tunnel-Reminder: true.  tests/smoke.sh sends it; every API
#            client must send it on every request, and the Retake UI's fetch helper
#            (authHeaders() in ui/app.js) is the one place to add it. <audio> players cannot send
#            headers, so for the clip players open the tunnel URL once in the same browser, enter
#            the tunnel password (`curl https://loca.lt/mytunnelpassword`) and continue; that sets
#            a cookie for the rest of the session.
#
# The public URL is also written to deploy/.tunnel-url, so the next step is just:
#   deploy/pages-publish.sh            (reads deploy/.tunnel-url when API_BASE is unset)
# …or paste the URL into the UI's Settings drawer (API base) — no republish needed.
#
# Usage: deploy/tunnel.sh [port]         default: $PORT or 8080
#        TUNNEL=cloudflared|localtunnel  force a provider
# Caveats: quick tunnels get a NEW hostname every start; Cloudflare caps a single request body
# at 100 MB (the API itself allows 500 MB) — use localtunnel or a real host for bigger files.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-${PORT:-8080}}"
URL_FILE="$ROOT/deploy/.tunnel-url"
LOG="$(mktemp "${TMPDIR:-/tmp}/retake-tunnel.XXXXXX")"
TUNNEL="${TUNNEL:-}"
PID=""

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
cleanup() { [[ -n "$PID" ]] && kill "$PID" 2>/dev/null || true; rm -f "$LOG"; }
trap cleanup EXIT INT TERM

if curl -fsS -m 5 "http://localhost:$PORT/v1/health" >/dev/null 2>&1; then
  say "Local API answers on http://localhost:$PORT/v1/health"
else
  echo "warning: nothing answered http://localhost:$PORT/v1/health — start the API first (./run.sh). Continuing." >&2
fi

if [[ -z "$TUNNEL" ]]; then
  if command -v cloudflared >/dev/null; then TUNNEL=cloudflared
  elif command -v npx >/dev/null; then TUNNEL=localtunnel
  else die "neither cloudflared nor npx found. Install one:  brew install cloudflared   (or node for npx localtunnel)"; fi
fi

URL=""
wait_for_url() {  # regex — polls the log for up to 90 s, sets URL to the first match
  local re="$1" i
  for i in $(seq 1 90); do
    URL="$(grep -oE "$re" "$LOG" | head -n1 || true)"
    if [[ -n "$URL" ]]; then return 0; fi
    if ! kill -0 "$PID" 2>/dev/null; then
      cat "$LOG" >&2
      die "$TUNNEL exited before printing a URL (log above)"
    fi
    sleep 1
  done
  cat "$LOG" >&2
  die "timed out waiting for the $TUNNEL URL"
}

case "$TUNNEL" in
  cloudflared)
    say "cloudflared quick tunnel → http://localhost:$PORT"
    cloudflared tunnel --url "http://localhost:$PORT" --no-autoupdate >"$LOG" 2>&1 &
    PID=$!
    wait_for_url 'https://[a-z0-9-]+\.trycloudflare\.com'
    ;;
  localtunnel)
    say "localtunnel (npx) → http://localhost:$PORT   (brew install cloudflared for the nicer option)"
    npx --yes localtunnel --port "$PORT" >"$LOG" 2>&1 &
    PID=$!
    wait_for_url 'https://[a-z0-9-]+\.loca\.lt'
    ;;
  *) die "TUNNEL must be cloudflared or localtunnel (got: $TUNNEL)" ;;
esac

[[ -n "$URL" ]] || die "no public URL captured"
printf '%s\n' "$URL" > "$URL_FILE"
cat <<EOF

  ┌──────────────────────────────────────────────────────────────────┐
    Public API:  $URL
    Health:      $URL/v1/health
    Saved to:    deploy/.tunnel-url
  └──────────────────────────────────────────────────────────────────┘
  Next:  deploy/pages-publish.sh          (bakes this URL into the hosted UI)
     or  paste it into the UI's Settings drawer → API base.
  CORS:  the local API must allow the Pages origin (RETAKE_CORS_ORIGINS=* is the dev default).
  Ctrl-C stops the tunnel.

EOF
if [[ "$TUNNEL" == "localtunnel" ]]; then
  echo "  localtunnel: every request must send 'Bypass-Tunnel-Reminder: true' (tests/smoke.sh does; the UI's"
  echo "  fetch helper must too) or browsers get a reminder page. One-time browser unlock password:"
  echo "  curl https://loca.lt/mytunnelpassword"
  echo
fi

wait "$PID" || true
