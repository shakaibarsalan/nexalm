# Running the worker on Kaggle

> **Status: verified end-to-end on real hardware.** A 2x T4 session loads
> Qwen3-30B-A3B at Q4_K_M (~20 GB across both cards) and serves streaming
> inference to a public control plane. The full suite passed 34/34 through
> the public domain.
>
> Commands below use the author's Kaggle username, `shakaibkaggle`. Use yours
> (`KAGGLE_USERNAME` in `scripts/launch-kaggle-worker.sh`).

## Hardware reality

Kaggle's free tier is **2× Tesla T4** (16 GB each, Turing `sm_75`) — but not
reliably. The first real run of this kernel came back with **1× Tesla P100
(16 GB)**, because `enable_gpu: true` alone does not pick which accelerator you
get; that is a per-notebook setting (*Session options → Accelerator → GPU T4
×2*). Set it, or the kernel silently gets half the VRAM.

**A CLI push resets the accelerator.** `kaggle kernels push` puts the notebook
back on the default (P100) every time — `kernel-metadata.json` has no field for
accelerator type. After each push, re-set *Session options → Accelerator →
GPU T4 ×2* before running, or the kernel quietly runs on half the VRAM.

That unreliability drives every decision here:

| Constraint | Consequence |
| --- | --- |
| Turing, not Ampere | No bf16, no FlashAttention. Use fp16 paths. |
| No NVLink between the T4s | Tensor-parallel would be bottlenecked by PCIe. llama.cpp splits **by layer** instead, which needs no fast interconnect. |
| 32 GB total VRAM | Qwen3-30B-A3B at Q4_K_M is ~18.6 GB, leaving generous KV-cache room. |
| Allocation varies per session | The kernel reads `nvidia-smi` and picks the largest quantisation fitting measured headroom (VRAM minus a 2.5 GB reserve for CUDA context and KV cache), then derives `--n-gpu-layers` from model size vs that headroom. A shortfall becomes CPU offload, not a crash. |
| 12 h session cap, ~30 h/week quota | The worker deregisters gracefully on shutdown so the UI shows offline immediately, not after the lease expires. |

This is why the engine is llama.cpp rather than vLLM. vLLM is faster where it
fits, but on `sm_75` it loses Marlin kernels and bf16 and leans on newer
Qwen3-MoE paths. `engines/vllm.py` is the seam for an Ampere-or-newer GPU later.

## Setup

1. **The one secret.** In the notebook editor: *Add-ons → Secrets → Add secret*

   | Field | Value |
   | --- | --- |
   | Label | `WORKER_TOKEN` — the **name**, exactly this |
   | Value | the `WORKER_TOKEN` from the server's `/opt/local-llm/.env` |

   `secret()` resolves a secret **by name**. Putting the token value into the
   `secret(...)` call instead of adding a secret named `WORKER_TOKEN` fails
   silently *and* leaves the token in the notebook source.

   The Oracle WebSocket URL is not a secret and needs no entry.

2. **Kaggle API credentials** — download `kaggle.json` from your Kaggle account
   page and place it at `~/.kaggle/kaggle.json` (`chmod 600`), or export
   `KAGGLE_USERNAME` / `KAGGLE_KEY`. Never commit it.

3. **Model dataset (strongly recommended)** — upload
   `Qwen3-30B-A3B-Q4_K_M.gguf` once as a **private Kaggle Dataset**. It then
   mounts read-only at `/kaggle/input/…` instantly on every session.

   Without it, the first run downloads ~18.6 GB from Hugging Face and then
   throws it away when the session ends — repeated on every single run.

   > **On Google Drive:** the shared folder is not a practical source here. A
   > share link alone grants no programmatic authentication, so Kaggle would
   > need OAuth or an `rclone` service account, and it would still re-pull the
   > full file each session over a slower path. A Kaggle Dataset is the correct
   > cache; keep Drive as an archival copy if useful.

4. **llama.cpp binaries (optional)** — the first run builds llama.cpp with CUDA
   (~10–15 min). Save `build/bin` as another private dataset to skip that
   thereafter.

## Launching

```bash
./scripts/launch-kaggle-worker.sh
```

That uploads `worker/` as a private dataset (a script kernel is a single file,
so the package travels alongside it) and pushes the kernel with GPU and
internet enabled.

```bash
./scripts/kaggle-cli.sh kernels status shakaibkaggle/local-llm-gpu-worker
./scripts/kaggle-cli.sh kernels output shakaibkaggle/local-llm-gpu-worker -p ./kaggle-output
```

Once it registers, the app's header shows the worker and the metrics panel
lists both T4s.

## What the kernel picks, by allocation

| Allocation | Usable for weights | Quantisation | Context | Placement |
| --- | --- | --- | --- | --- |
| 2× T4 (32 GB) | 29.5 GB | Q4_K_M (18.6 GB) | 16384 | all 48 layers on GPU |
| 1× T4 / P100 (16 GB) | 13.5 GB | IQ3_XXS (11.9 GB) | 8192 | all 48 layers on GPU |
| 1× A100 (40 GB) | 37.5 GB | Q4_K_M (18.6 GB) | 16384 | all 48 layers on GPU |

Only the 2× T4 row runs the intended Q4_K_M quality, which is why the
accelerator setting matters.

## Performance, and a metric that lied

Qwen3-30B-A3B activates only ~3.3 B of its 30.5 B parameters per token, so it
behaves far more like a small model than its size suggests.

Beware one trap, since it produced wrong numbers here for a while. Qwen3 emits
a **reasoning trace** that llama.cpp reports as `reasoning_content`, not
`content`. Reading only `content` discards those tokens — the GPU still spends
time on them — and corrupts throughput: `completion_tokens` counts the thinking
while a `gen_ms` timed from the first *content* token does not. That inflated
tokens/sec to physically impossible values (7957 tok/s on a T4; a stored
average of 1527).

The worker now emits reasoning on its own `job.reasoning` frame, times TTFT
from the first token of *any* kind, and the UI shows the trace as a collapsed
"Thought process" block. Trust `inference_runs` only for rows written after
that fix; earlier rows are inflated.

A one-word answer legitimately costs ~100-400 tokens, because most of them are
thinking.
