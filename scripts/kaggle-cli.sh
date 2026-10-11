#!/usr/bin/env bash
# Thin wrapper that runs the Kaggle CLI inside a container.
#
# Why a container: the current Kaggle CLI (>= 1.8, needed for the modern
# KGAT_ access tokens) requires Python >= 3.11, and the server this was
# written for ships 3.10. Installing another Python system-wide on a shared
# box is not worth it for a deploy helper.
#
# The token is read from ~/.kaggle/access_token (mode 600) and passed as an
# environment variable. It is never written into the image or into this repo.
#
#   ./scripts/kaggle-cli.sh datasets list -m
#   ./scripts/kaggle-cli.sh kernels status shakaibkaggle/local-llm-gpu-worker
set -euo pipefail

IMAGE="${KAGGLE_CLI_IMAGE:-localllm-kaggle-cli:latest}"
TOKEN_FILE="${KAGGLE_TOKEN_FILE:-$HOME/.kaggle/access_token}"
DOCKER="${DOCKER:-docker}"

if ! $DOCKER image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "==> building $IMAGE (one time)" >&2
  $DOCKER build -q -t "$IMAGE" - >&2 <<'DOCKERFILE'
FROM python:3.12-slim
RUN pip install --no-cache-dir "kaggle>=1.8.0"
WORKDIR /work
ENTRYPOINT ["kaggle"]
DOCKERFILE
fi

# Prefer the modern access token; fall back to legacy kaggle.json if that is
# what the host has.
declare -a cred_args=()
if [[ -f "$TOKEN_FILE" ]]; then
  cred_args+=(-e "KAGGLE_API_TOKEN=$(cat "$TOKEN_FILE")")
elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  cred_args+=(-v "$HOME/.kaggle:/root/.kaggle:ro")
elif [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  cred_args+=(-e "KAGGLE_USERNAME=$KAGGLE_USERNAME" -e "KAGGLE_KEY=$KAGGLE_KEY")
else
  echo "error: no Kaggle credentials found." >&2
  echo "  expected $TOKEN_FILE, or ~/.kaggle/kaggle.json, or KAGGLE_USERNAME+KAGGLE_KEY" >&2
  exit 1
fi

exec $DOCKER run --rm -i \
  "${cred_args[@]}" \
  -v "$(pwd):/work" \
  "$IMAGE" "$@"
