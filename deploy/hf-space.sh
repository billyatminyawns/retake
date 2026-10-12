#!/usr/bin/env bash
# deploy/hf-space.sh — create (if needed) and push the Retake API to a Hugging Face Space.
# DOCUMENTED, NOT RUN BY THE BUILDER.
#
#   HF_TOKEN=hf_xxx deploy/hf-space.sh [owner/space] [--with-secrets] [--private]
#
#   owner/space     default: <your HF username>/retake  (also HF_SPACE env)
#   --with-secrets  also push WELLSAID_API_KEY / ANTHROPIC_API_KEY / RETAKE_API_KEYS from your
#                   shell env as Space secrets, and RETAKE_CORS_ORIGINS as a variable
#   --private       create the Space private (default public)
#
# Needs: HF_TOKEN with "write" scope (https://huggingface.co/settings/tokens), git, curl, python3.
# The token is passed to git through GIT_ASKPASS — it never appears in a URL, a log line or
# .git/config. Nothing here prints secret values.
#
# What happens:
#   1. POST /api/repos/create (type=space, sdk=docker)      — 409 "already exists" is fine
#   2. stage core/ api/ ui/ data/ requirements.txt deploy/prefetch_models.py + Dockerfile (+ENV PORT=7860)
#      + deploy/huggingface/README.md (the Space front-matter) into a temp git repo
#   3. force-push that single commit to the Space's main branch (the Space is a mirror, not history)
#   4. optionally set secrets/variables, then print the Space URL and what to do next
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_API="https://huggingface.co/api"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

SPACE="${HF_SPACE:-}"
WITH_SECRETS=0
PRIVATE=false
for arg in "$@"; do
  case "$arg" in
    --with-secrets) WITH_SECRETS=1 ;;
    --private)      PRIVATE=true ;;
    -h|--help)      sed -n '2,24p' "$0"; exit 0 ;;
    */*)            SPACE="$arg" ;;
    *)              die "unexpected argument: $arg (expected owner/space)" ;;
  esac
done

[[ -n "${HF_TOKEN:-}" ]] || die "set HF_TOKEN (a write token from https://huggingface.co/settings/tokens)"
for t in git curl python3; do command -v "$t" >/dev/null || die "$t is required"; done
for p in core api ui data requirements.txt deploy/Dockerfile deploy/prefetch_models.py deploy/huggingface/README.md; do
  [[ -e "$ROOT/$p" ]] || die "missing $p — run from a complete checkout"
done

# Who am I (also the default owner)?
ME="$(curl -fsS -H "Authorization: Bearer $HF_TOKEN" "$HF_API/whoami-v2" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["name"])')" \
  || die "HF_TOKEN was rejected by $HF_API/whoami-v2"
[[ -n "$SPACE" ]] || SPACE="$ME/retake"
OWNER="${SPACE%%/*}"
NAME="${SPACE##*/}"
SPACE_URL="https://huggingface.co/spaces/$SPACE"

# 1. Create the Space (idempotent).
say "Ensuring Space $SPACE exists (sdk=docker, private=$PRIVATE)"
if [[ "$OWNER" == "$ME" ]]; then
  body="{\"type\":\"space\",\"name\":\"$NAME\",\"private\":$PRIVATE,\"sdk\":\"docker\"}"
else
  body="{\"type\":\"space\",\"name\":\"$NAME\",\"organization\":\"$OWNER\",\"private\":$PRIVATE,\"sdk\":\"docker\"}"
fi
resp_file="$(mktemp)"
code="$(curl -sS -o "$resp_file" -w '%{http_code}' -X POST "$HF_API/repos/create" \
        -H "Authorization: Bearer $HF_TOKEN" -H 'Content-Type: application/json' -d "$body")"
case "$code" in
  200|201) say "Created $SPACE_URL" ;;
  409)     say "$SPACE already exists — updating it" ;;
  *)       cat "$resp_file" >&2; echo >&2; rm -f "$resp_file"; die "repo create failed (HTTP $code)" ;;
esac
rm -f "$resp_file"

# 2. Stage the deployable subset.
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/retake-space.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
say "Staging into $STAGE"
mkdir -p "$STAGE/deploy"
cp -R "$ROOT/core" "$ROOT/api" "$ROOT/ui" "$ROOT/data" "$STAGE/"
cp "$ROOT/requirements.txt" "$STAGE/"
cp "$ROOT/deploy/prefetch_models.py" "$STAGE/deploy/"
cp "$ROOT/deploy/huggingface/README.md" "$STAGE/README.md"
{ cat "$ROOT/deploy/Dockerfile"; printf '\n# Appended by deploy/hf-space.sh: Spaces route to app_port 7860 and do not set PORT.\nENV PORT=7860\n'; } > "$STAGE/Dockerfile"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '.DS_Store' -delete 2>/dev/null || true
# The Hub rejects non-LFS files over 10 MB; nothing we ship should be that big.
big="$(find "$STAGE" -type f -size +10M | sed "s#^$STAGE/##" | head -n 5 || true)"
[[ -z "$big" ]] || die "files over 10 MB cannot be pushed without git-lfs: $big"

# 3. Push as one commit via GIT_ASKPASS (token never hits a URL or .git/config).
ASKPASS="$STAGE/.askpass.sh"
cat > "$ASKPASS" <<'EOS'
#!/bin/sh
case "$1" in
  Username*) printf '%s\n' "$HF_USER" ;;
  *)         printf '%s\n' "$HF_TOKEN" ;;
esac
EOS
chmod 700 "$ASKPASS"
say "Pushing to $SPACE_URL (main)"
(
  cd "$STAGE"
  git init -q
  git symbolic-ref HEAD refs/heads/main
  git -c user.name="retake-deploy" -c user.email="retake-deploy@users.noreply.huggingface.co" add -A
  git -c user.name="retake-deploy" -c user.email="retake-deploy@users.noreply.huggingface.co" \
      commit -q -m "Deploy Retake API ($(date -u +%Y-%m-%dT%H:%MZ))"
  export HF_USER="$ME" HF_TOKEN GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0
  git push -q --force "$SPACE_URL" main 2>&1 | sed "s#$HF_TOKEN#<hf-token>#g" || die "git push failed"
)

# 4. Optional secrets / variables.
if [[ "$WITH_SECRETS" == "1" ]]; then
  set_kv() {  # kind(secrets|variables) key value — the value travels via env + stdin, never argv
    local kind="$1" key="$2" value="$3" c
    c="$(RK_KEY="$key" RK_VALUE="$value" python3 -c 'import json, os; print(json.dumps({"key": os.environ["RK_KEY"], "value": os.environ["RK_VALUE"]}))' \
         | curl -sS -o /dev/null -w '%{http_code}' -X POST "$HF_API/spaces/$SPACE/$kind" \
                -H "Authorization: Bearer $HF_TOKEN" -H 'Content-Type: application/json' --data-binary @-)"
    case "$c" in 200|201) echo "    $kind: $key set" ;; *) echo "    $kind: $key FAILED (HTTP $c)" >&2 ;; esac
  }
  say "Setting Space secrets/variables from the environment (values not shown)"
  for k in WELLSAID_API_KEY ANTHROPIC_API_KEY RETAKE_API_KEYS; do
    if [[ -n "${!k:-}" ]]; then set_kv secrets "$k" "${!k}"; else echo "    secrets: $k not in env, skipped"; fi
  done
  set_kv variables RETAKE_CORS_ORIGINS "${RETAKE_CORS_ORIGINS:-https://billyatminyawns.github.io}"
fi

cat <<EOF

Done. Space: $SPACE_URL
  Build logs:  $SPACE_URL?logs=build      (first build ≈ 10–15 min: pip + 2.8 GB of models)
  API base:    https://${OWNER}-${NAME}.hf.space          → /v1/health, /docs, and the UI at /
  Secrets:     Settings → Variables and secrets (or re-run with --with-secrets)
  Hardware:    Settings → Hardware → "CPU upgrade" for a snappier demo (≈\$0.03/h)
  Hosted UI:   API_BASE=https://${OWNER}-${NAME}.hf.space deploy/pages-publish.sh
EOF
