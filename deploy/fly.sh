#!/usr/bin/env bash
# deploy/fly.sh — deploy the Retake API to Fly.io. DOCUMENTED, NOT RUN BY THE BUILDER.
#
#   brew install flyctl && fly auth login
#   WELLSAID_API_KEY=… ANTHROPIC_API_KEY=… RETAKE_API_KEYS=… deploy/fly.sh [app-name]
#
# Env:  FLY_APP (default retake-api — must match `app` in deploy/fly.toml unless you pass a name),
#       FLY_ORG (default personal), FLY_VOLUME_GB (default 10), FLY_REGION (default: primary_region
#       from fly.toml — keep them equal or the volume lands in the wrong region).
#
# Steps (each idempotent):
#   1. fly apps create            if the app does not exist
#   2. fly volumes create         a 10 GB volume "retake_data" in the primary region if none exists
#   3. fly secrets set --stage    WELLSAID_API_KEY / ANTHROPIC_API_KEY / RETAKE_API_KEYS from your env
#                                 (only the ones that are set; values are never printed)
#   4. fly deploy --ha=false      builds deploy/Dockerfile remotely (≈10 min first time) and starts ONE machine
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="deploy/fly.toml"
VOLUME="retake_data"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

if command -v fly >/dev/null; then FLY=fly; elif command -v flyctl >/dev/null; then FLY=flyctl; else
  die "flyctl is not installed: brew install flyctl  (then: fly auth login)"; fi
$FLY auth whoami >/dev/null 2>&1 || die "not logged in: run 'fly auth login'"

cd "$ROOT"
[[ -f "$CONFIG" && -f deploy/Dockerfile ]] || die "run from a checkout that has $CONFIG and deploy/Dockerfile"

APP="${1:-${FLY_APP:-$(sed -nE 's/^app *= *"([^"]+)".*/\1/p' "$CONFIG" | head -n1)}}"
TOML_REGION="$(sed -nE 's/^primary_region *= *"([^"]+)".*/\1/p' "$CONFIG" | head -n1)"
REGION="${FLY_REGION:-${TOML_REGION:-sea}}"
[[ "$REGION" == "$TOML_REGION" ]] || echo "warning: FLY_REGION=$REGION differs from primary_region=$TOML_REGION in $CONFIG" >&2

# 1. App
if $FLY status -a "$APP" >/dev/null 2>&1; then
  say "App $APP exists"
else
  say "Creating app $APP (org ${FLY_ORG:-personal})"
  $FLY apps create "$APP" --org "${FLY_ORG:-personal}"
fi

# 2. Volume (one, in the primary region)
nvol="$($FLY volumes list -a "$APP" --json 2>/dev/null | python3 -c 'import json,sys
try:
    print(len(json.load(sys.stdin)))
except Exception:
    print(0)' || echo 0)"
if [[ "${nvol:-0}" -gt 0 ]]; then
  say "Volume present ($nvol)"
else
  say "Creating ${FLY_VOLUME_GB:-10} GB volume $VOLUME in $REGION"
  $FLY volumes create "$VOLUME" -a "$APP" --region "$REGION" --size "${FLY_VOLUME_GB:-10}" --yes
fi

# 3. Secrets (staged: applied by the deploy below, no extra restart)
names=()
kv=()
for k in WELLSAID_API_KEY ANTHROPIC_API_KEY RETAKE_API_KEYS; do
  if [[ -n "${!k:-}" ]]; then names+=("$k"); kv+=("$k=${!k}"); fi
done
if ((${#kv[@]})); then
  say "Staging secrets: ${names[*]}"
  $FLY secrets set -a "$APP" --stage "${kv[@]}" >/dev/null
else
  say "No WELLSAID_API_KEY / ANTHROPIC_API_KEY / RETAKE_API_KEYS in the environment — skipping secrets (auth will be OPEN)"
fi

# 4. Deploy — one machine, remote build (the models are baked in during the build).
say "Deploying $APP from deploy/Dockerfile (remote build; first time ≈ 10 min)"
$FLY deploy --config "$CONFIG" --dockerfile deploy/Dockerfile --app "$APP" --ha=false --remote-only

cat <<EOF

Done.
  API base:   https://$APP.fly.dev        → /v1/health, /docs, UI at /
  Logs:       $FLY logs -a $APP
  Scale:      $FLY scale memory 16384 -a $APP     (if 500 MB uploads + both models feel tight)
  Hosted UI:  API_BASE=https://$APP.fly.dev deploy/pages-publish.sh
  Cost:       performance-2x · 8 GB ≈ \$0.10/h while running; auto_stop_machines puts it to sleep
              when idle (storage ≈ \$0.15/GB-mo). See https://fly.io/docs/about/pricing/
EOF
