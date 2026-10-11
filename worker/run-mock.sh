#!/usr/bin/env bash
# Launch the mock worker against the local deployment (verification only).
cd "$(dirname "$0")"
WT=$(grep "^WORKER_TOKEN=" ../.env | cut -d= -f2-)
export ORACLE_WS_URL="${ORACLE_WS_URL:-ws://127.0.0.1:8081/worker/ws}"
export WORKER_TOKEN="$WT"
export WORKER_NAME="${WORKER_NAME:-mock-oracle}"
export WORKER_ID="${WORKER_ID:-mock-oracle-1}"
export ENGINE=mock
export MODEL_ID="${MODEL_ID:-qwen3-0.6b}"
export LOG_LEVEL=INFO
exec ./.venv/bin/python worker.py
