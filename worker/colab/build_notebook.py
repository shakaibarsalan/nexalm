#!/usr/bin/env python3
"""Generate a self-contained Colab notebook for the NexaLM worker.

The worker source is embedded as a base64 tarball rather than cloned at
runtime: the repository is private, and `git clone` inside Colab has no way to
authenticate — it fails with "could not read Username for https://github.com".
Embedding removes the dependency entirely, so the notebook works on any Colab
account with nothing but a Drive mount.

    python worker/colab/build_notebook.py

Writes worker/colab/NexaLM_Colab_Worker.ipynb (~80 KB).
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER_DIR = HERE.parent
OUT = HERE / "NexaLM_Colab_Worker.ipynb"

# Everything the worker needs at runtime. colab_worker.py is the entry point.
PAYLOAD = ["worker.py", "models.py", "engines", "provision", "colab/colab_worker.py"]


def build_payload() -> str:
    """Tar the worker package and base64 it for embedding in a code cell."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in PAYLOAD:
            src = WORKER_DIR / rel
            # Flatten colab/colab_worker.py to the root so imports resolve the
            # same way they do on Kaggle.
            arcname = Path(rel).name if rel.endswith("colab_worker.py") else rel
            tar.add(src, arcname=arcname, filter=_strip)
    return base64.b64encode(buf.getvalue()).decode()


def _strip(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if "__pycache__" in info.name or info.name.endswith(".pyc"):
        return None
    # Deterministic output so regenerating produces no spurious diff.
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


def main() -> None:
    payload = build_payload()

    cells = [
        md(
            "# NexaLM — Colab GPU worker",
            "",
            "Runs Qwen3-30B-A3B on Colab's GPU and connects it to your NexaLM server.",
            "",
            "**Before running:** *Runtime → Change runtime type → **T4 GPU***.",
            "",
            "The worker code is embedded in this notebook, so nothing is cloned and",
            "no GitHub access is needed. Run the cells in order.",
            "",
            "| | First run | Later runs |",
            "| --- | --- | --- |",
            "| Model download to Drive | 15–40 min | skipped |",
            "| llama.cpp CUDA build | 20–30 min | skipped |",
            "| Model load onto GPU | 1–2 min | 1–2 min |",
            "",
            "> Free Colab gives **one 16 GB T4**; Kaggle gives **two (32 GB)**. A 30B model",
            "> at Q4_K_M needs ~18.6 GB, so Colab drops to a smaller quantisation or",
            "> offloads to CPU — slower and lower quality. Colab is the fallback for when",
            "> the Kaggle weekly quota runs out.",
        ),
        md("## 1. Mount Drive", "", "Google will ask you to authorise this — the one step only you can do."),
        code(
            "from google.colab import drive",
            "drive.mount('/content/drive')",
        ),
        md(
            "## 2. Unpack the worker onto Drive",
            "",
            "Reuses an existing NexaLM/Local-LLM folder if you already have one,",
            "rather than creating another.",
        ),
        code(
            "import base64, io, os, tarfile, glob",
            "",
            "MYDRIVE = '/content/drive/MyDrive'",
            "",
            "# Reuse a folder you already have instead of making a new one. Names are",
            "# matched loosely because Drive folders picked by hand vary in spacing/case.",
            "def pick_root():",
            "    override = os.environ.get('DRIVE_ROOT')",
            "    if override:",
            "        return override",
            "    for entry in sorted(glob.glob(os.path.join(MYDRIVE, '*'))):",
            "        if not os.path.isdir(entry):",
            "            continue",
            "        flat = os.path.basename(entry).lower().replace(' ', '').replace('_', '')",
            "        if flat in ('nexalm', 'locallm', 'localllm', 'local-llm'.replace('-', '')):",
            "            return entry",
            "    return os.path.join(MYDRIVE, 'NexaLM')",
            "",
            "DRIVE_ROOT = pick_root()",
            "WORKER_DIR = os.path.join(DRIVE_ROOT, 'worker')",
            "os.makedirs(WORKER_DIR, exist_ok=True)",
            "print('using Drive folder:', DRIVE_ROOT)",
            "",
            f'PAYLOAD_B64 = "{payload}"',
            "",
            "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD_B64)), mode='r:gz') as tar:",
            "    tar.extractall(WORKER_DIR)",
            "",
            "print('unpacked:')",
            "for p in sorted(glob.glob(os.path.join(WORKER_DIR, '**', '*.py'), recursive=True)):",
            "    print('  ', os.path.relpath(p, WORKER_DIR))",
        ),
        md(
            "## 3. Worker token",
            "",
            "Read it from your server:",
            "",
            "```bash",
            "ssh you@your-server 'grep WORKER_TOKEN /opt/local-llm/.env'",
            "```",
            "",
            "Preferred: add it under the **key icon** in Colab's sidebar, named exactly",
            "`WORKER_TOKEN`, and enable it for this notebook. Otherwise the cell below",
            "prompts for it and writes it to a private file on your Drive.",
        ),
        code(
            "import os, json, getpass",
            "",
            "CONFIG = os.path.join(DRIVE_ROOT, 'worker-config.json')",
            "ORACLE_WS_URL = 'wss://chat.example.com/worker/ws'  # change to your server",
            "",
            "token = ''",
            "try:",
            "    from google.colab import userdata",
            "    token = userdata.get('WORKER_TOKEN') or ''",
            "    if token:",
            "        print('using WORKER_TOKEN from Colab secrets')",
            "except Exception:",
            "    pass",
            "",
            "if not token and os.path.exists(CONFIG):",
            "    token = json.load(open(CONFIG)).get('WORKER_TOKEN', '')",
            "    if token:",
            "        print('using WORKER_TOKEN from', CONFIG)",
            "",
            "if not token:",
            "    # getpass keeps the token out of the notebook's saved output.",
            "    token = getpass.getpass('WORKER_TOKEN: ').strip()",
            "    json.dump({'WORKER_TOKEN': token, 'ORACLE_WS_URL': ORACLE_WS_URL}, open(CONFIG, 'w'))",
            "    os.chmod(CONFIG, 0o600)",
            "    print('saved to', CONFIG, '— later runs will not ask again')",
            "",
            "assert token, 'no token supplied'",
            "os.environ['WORKER_TOKEN'] = token",
            "os.environ['ORACLE_WS_URL'] = ORACLE_WS_URL",
            "os.environ['DRIVE_ROOT'] = DRIVE_ROOT",
            "print('ready')",
        ),
        md(
            "## 4. Run the worker",
            "",
            "Leave this cell running. It builds llama.cpp and downloads the model on the",
            "first run only — both are cached on Drive afterwards.",
            "",
            "Stop it from NexaLM's **Inference → GPU session → Stop**, or by interrupting",
            "this cell.",
        ),
        code(
            "!cd {WORKER_DIR} && python colab_worker.py",
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }

    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
