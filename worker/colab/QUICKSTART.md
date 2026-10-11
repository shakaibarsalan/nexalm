# Colab quick start

Use **`NexaLM_Colab_Worker.ipynb`** in this folder. Upload it to Colab
(*File → Upload notebook*) and run the cells top to bottom.

Set *Runtime → Change runtime type → **T4 GPU*** first.

## Why the notebook embeds the worker

An earlier version cloned this repository at runtime. That cannot work: the
repo is **private**, and there is no way for `git` inside Colab to
authenticate, so it fails with

```
fatal: could not read Username for 'https://github.com': No such device or address
```

The notebook therefore carries the worker source inside itself as a base64
tarball. Nothing is cloned, no token is needed for GitHub, and it runs on any
Colab account that can mount your Drive.

Regenerate it after changing worker code:

```bash
python worker/colab/build_notebook.py
```

## Drive folder

The notebook **reuses a folder you already have** — anything in *My Drive*
named like `NexaLM`, `Local-LLM` or `local llm` (spacing and case are ignored).
It only creates `NexaLM` if none exists. To force a specific one, set
`DRIVE_ROOT` before running cell 2.

Inside it:

```
<your folder>/
  worker/            worker.py, engines/, colab_worker.py
  models/            the .gguf, downloaded once
  llama-bin/         compiled llama.cpp, built once
  worker-config.json the worker token, if not using Colab secrets
```

## The worker token

Read it from the server:

```bash
ssh you@your-server 'grep WORKER_TOKEN /opt/local-llm/.env'
```

Preferred: Colab's **key icon** in the sidebar, secret named exactly
`WORKER_TOKEN`, enabled for the notebook. Otherwise cell 3 prompts for it with
`getpass` — so it never lands in saved notebook output — and stores it in
`worker-config.json` on your Drive for next time.

## Timings

| Stage | First run | Later runs |
| --- | --- | --- |
| Model download to Drive | 15–40 min | skipped |
| llama.cpp CUDA build | 20–30 min | skipped |
| Model load onto the GPU | 1–2 min | 1–2 min |

Once it registers, NexaLM's **Model** panel goes *Offline → Loading model →
Ready* and the resource charts start plotting.

## Expect lower quality than Kaggle

Free Colab gives **one 16 GB T4**; Kaggle gives **two (32 GB)**. Qwen3-30B-A3B
at Q4_K_M is ~18.6 GB of weights, so a Colab session automatically drops to a
smaller quantisation or offloads layers to the CPU — slower, and noticeably
worse answers.

Colab is the fallback for when the Kaggle weekly quota is spent. It is also not
an always-on host: free Colab has no background execution and reclaims the VM
after roughly 90 minutes idle.
