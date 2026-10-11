# Inference worker

Dials **out** to the Oracle backend over WebSocket and serves inference jobs.
The outbound direction is required: Kaggle and Colab runtimes are behind NAT and
cannot be reached inbound.

## Engines

| Engine | Use | Notes |
| --- | --- | --- |
| `llamacpp` | Production (Kaggle 2×T4) | GGUF, splits across GPUs by layer |
| `mock` | Pipeline verification | No model, canned text, real timings |

## Running

```bash
pip install -r requirements.txt

export ORACLE_WS_URL="wss://chat.example.com/worker/ws"
export WORKER_TOKEN="…"          # must match the server
export WORKER_NAME="kaggle-t4x2"
export ENGINE="llamacpp"
export MODEL_ID="qwen3-30b-a3b"
export MODEL_PATH="/kaggle/input/qwen3-30b-a3b-gguf/Qwen3-30B-A3B-Q4_K_M.gguf"
export CONTEXT_LENGTH=16384

python worker.py
```

For a pipeline smoke test with no GPU and no model:

```bash
ENGINE=mock MODEL_ID=qwen3-0.6b WORKER_NAME=mock-worker python worker.py
```

## Never commit

`WORKER_TOKEN`, `kaggle.json`, or any `.gguf`. All are covered by `.gitignore`.
