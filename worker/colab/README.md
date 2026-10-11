# Running the worker on Google Colab

> **Status: written, not yet run against a live Colab session.** The Kaggle path
> is verified end-to-end; this mirrors it and reuses the same `worker.py` and
> engines, but the Colab-specific steps have not been executed yet.

## What Drive is actually for

Drive is the **persistence layer**, not a speed optimisation. Be clear-eyed
about that, because it's easy to expect the wrong benefit:

| | First run | Later runs |
| --- | --- | --- |
| Model download (~18.6 GB) | 15–40 min to Drive | **0** — already there |
| llama.cpp CUDA build | 20–30 min | **0** — binaries on Drive |
| Total cold start | ~45–70 min | **~2–4 min** |

A direct HuggingFace pull is *faster* than reading from Drive (~230 MB/s
measured on Kaggle vs tens of MB/s for Drive). What Drive buys is that the work
survives the session and is portable: any Colab account that mounts the same
Drive skips both steps entirely. That is exactly the workflow you described.

## Setup

1. **New Colab notebook** → *Runtime → Change runtime type → T4 GPU*.

2. **Mount Drive and fetch the worker code:**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   !mkdir -p /content/drive/MyDrive/local-llm/worker
   !git clone --depth 1 <your-repo-url> /content/repo
   !cp -r /content/repo/worker/worker.py /content/repo/worker/engines \
          /content/repo/worker/colab/colab_worker.py \
          /content/drive/MyDrive/local-llm/worker/
   ```

3. **Add the worker token.** Either the key icon in Colab's sidebar (name it
   exactly `WORKER_TOKEN`), or a file at
   `/content/drive/MyDrive/local-llm/worker-config.json`:

   ```json
   { "WORKER_TOKEN": "...", "ORACLE_WS_URL": "wss://chat.example.com/worker/ws" }
   ```

   The token is in the server's `/opt/local-llm/.env`. It never goes in the
   notebook source.

4. **Run:**

   ```python
   !python /content/drive/MyDrive/local-llm/worker/colab_worker.py
   ```

The first run populates Drive; every run after is minutes.

## Colab versus Kaggle — the honest comparison

| | Kaggle | Colab Free |
| --- | --- | --- |
| GPU | **2× T4 = 32 GB** | 1× T4 = 16 GB |
| Qwen3-30B-A3B | **Q4_K_M**, 16k context | Smaller quant or CPU offload |
| Background execution | Yes | **No** |
| Idle disconnect | — | **~90 min** |
| Session cap | 12 h | 12 h |
| Quota | 30 h/week, visible in the app | Undisclosed, throttled |
| Persistence | Kaggle Datasets | Google Drive |
| Start/stop from the app | **Yes** (API) | Manual |

**Colab free is the weaker host for this workload**, mainly because 16 GB
forces a worse quantisation of a 30B model. It earns its place as a *fallback*
when the Kaggle quota is spent, and as the path to a paid tier later.

## What will not work

**Keep-alive scripts to dodge the idle timeout.** Colab's FAQ says sessions end
when "the workload conflicts with Colab's priorities," and Google disallows
free Colab for long-running background compute; anti-idle automation gets
accounts throttled or banned. Free Colab also has no background execution, so
the session dies when the browser closes regardless of any script.

There is no free always-on GPU. The app's Start/Stop control plus idle
auto-shutdown is the answer that actually works within the free tiers.

## Larger models

For 70B-class models, the arithmetic is unforgiving:

| Model | Q4_K_M | Fits 16 GB? | Fits 32 GB? |
| --- | --- | --- | --- |
| Qwen3-30B-A3B (MoE, 3.3B active) | ~18.6 GB | No | **Yes** |
| 70B dense | ~40 GB | No | No |
| 70B at Q2_K | ~26 GB | No | Yes, with real quality loss |

A 70B dense model also activates *all* its parameters per token, unlike
Qwen3-30B-A3B's 3.3B — so even where it fits, expect single-digit tokens/sec
rather than the ~60–160 the MoE achieves. Running 70B properly means rented
hardware, not a free tier.
