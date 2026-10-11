"""Colab entry point for the Local LLM GPU worker.

Paste-and-run cell for a Colab notebook. Mirrors the Kaggle bootstrap, with one
important difference: Colab has no dataset equivalent, so Google Drive is the
persistence layer. The model *and* the compiled llama.cpp binaries live there,
which turns a ~30 minute cold start into a couple of minutes on every later run
and on any Colab account that mounts the same Drive.

    from google.colab import drive; drive.mount('/content/drive')
    !python colab_worker.py

Honest limits, so nothing here is a surprise:
  * Free Colab gives ONE T4 (16 GB), not Kaggle's 2x T4 (32 GB). A 30B model at
    Q4_K_M needs ~18.6 GB, so a free session drops to a smaller quantisation or
    offloads layers to CPU. Quality and speed both suffer.
  * Free Colab has NO background execution and reclaims the VM after roughly
    90 minutes idle. It cannot be an always-on host, and keep-alive tricks
    violate Colab's terms.
  * Drive is slower than a direct HuggingFace pull (tens of MB/s vs ~230 MB/s
    measured). Its value is persistence and portability, not speed.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import shutil
import time
import sys
import urllib.error
import urllib.request

DRIVE_ROOT = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/local-llm")
MODEL_DIR = os.path.join(DRIVE_ROOT, "models")
BIN_DIR = os.path.join(DRIVE_ROOT, "llama-bin")
CODE_DIR = os.path.join(DRIVE_ROOT, "worker")
CONFIG_PATH = os.path.join(DRIVE_ROOT, "worker-config.json")

REPO = os.environ.get("HF_REPO", "unsloth/Qwen3-30B-A3B-GGUF")
# Binaries run from local disk: Drive cannot hold the exec bit, cannot create
# the hard links the archive uses, and loading shared objects across FUSE is
# needlessly slow.
LOCAL_BIN = "/content/llama-bin"


_SECRETS: set[str] = set()


def remember_secret(value: str) -> str:
    """Register a value that must never appear in printed output."""
    if value and len(value) >= 8:
        _SECRETS.add(value)
    return value


def redact(text: str) -> str:
    """Strip known secrets from anything echoed to the notebook.

    A .ipynb carries its cell output with it, so a token printed here follows
    the notebook wherever it is shared. Defence in depth: no caller should be
    putting a secret on a command line in the first place.
    """
    out = text
    for value in _SECRETS:
        out = out.replace(value, "<redacted>")
    return out


def sh(cmd: str, check: bool = True) -> int:
    print("$ " + redact(cmd), flush=True)
    return subprocess.run(cmd, shell=True, check=check, executable="/bin/bash").returncode


# ── 0. Drive ────────────────────────────────────────────────────────────────
if not os.path.isdir("/content/drive/MyDrive"):
    print("!! Google Drive is not mounted. Run this first, then re-run:")
    print("     from google.colab import drive; drive.mount('/content/drive')")
    sys.exit(1)

for d in (MODEL_DIR, BIN_DIR, CODE_DIR):
    os.makedirs(d, exist_ok=True)
print(f"-> Drive workspace: {DRIVE_ROOT}")

# ── 1. Hardware ─────────────────────────────────────────────────────────────
# Provisioning decisions live in worker/provision/common.py so that Kaggle and
# Colab cannot answer them differently. They used to, and the differences were
# silent: Colab picked its model alphabetically and targeted the wrong CUDA
# architecture on anything that was not a T4.
sys.path.insert(0, CODE_DIR)
from provision.common import (  # noqa: E402
    arch_mismatch_message,
    cuda_arch,
    install_failure_beacon,
    default_context,
    ensure_libcuda_link,
    loader_path,
    model_id_for,
    pick_gguf,
    plan_placement,
    prebuilt_runs_here,
    probe_gpus,
)

gpu_lines, gpu_count, total_vram_gb, headroom_gb = probe_gpus()
if gpu_count == 0:
    print()
    print("!! This Colab runtime has no GPU attached, so there is nothing to run the")
    print("   model on. Fix it in two clicks:")
    print()
    print("     Runtime -> Change runtime type -> Hardware accelerator: T4 GPU -> Save")
    print()
    print("   Colab then restarts the runtime. Re-run this cell afterwards; the model")
    print("   and binaries already on Drive are reused, so it picks up where it was.")
    sys.exit(1)

# ── 2. Credentials ──────────────────────────────────────────────────────────
# Colab Secrets first (the key icon in the sidebar), then a Drive config file,
# then the environment. The token is never written into this file.
def secret(name: str, default: str = "") -> str:
    """Resolve a secret, registering it so it can never be echoed.

    Every secret in this file flows through here, so registering at the single
    resolution point is what makes redact() actually complete.
    """

    def found(value: str) -> str:
        return remember_secret(value) if name.endswith("TOKEN") else value

    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        if value:
            return found(value)
    except Exception:  # noqa: BLE001 — secret absent or not on Colab
        pass
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as fh:
                cfg = json.load(fh)
            if cfg.get(name):
                return found(cfg[name])
        except Exception:  # noqa: BLE001 — malformed config must not be fatal
            pass
    return found(os.environ.get(name, default))


ORACLE_WS_URL = secret("ORACLE_WS_URL", "wss://chat.example.com/worker/ws")
WORKER_TOKEN = secret("WORKER_TOKEN")

# From here on an unhandled failure is reported to the server before the
# process dies. Provisioning failures were previously invisible: the session
# just ended, having spent quota, with nothing to look at.
install_failure_beacon("colab", ORACLE_WS_URL, WORKER_TOKEN)

if not WORKER_TOKEN:
    sys.exit(
        "Missing WORKER_TOKEN.\n"
        "Either add it under the key icon in Colab's sidebar (name it exactly\n"
        f"WORKER_TOKEN), or create {CONFIG_PATH} containing:\n"
        '  {"WORKER_TOKEN": "...", "ORACLE_WS_URL": "wss://.../worker/ws"}'
    )

# ── 3. Model, cached on Drive ───────────────────────────────────────────────
# Ask the repo what it actually has, then pick by real size. Constructing a
# filename from a quantisation name is what produced a 404 here: Unsloth's
# dynamic quants carry a "UD-" prefix, so Qwen3-30B-A3B-IQ3_XXS.gguf does not
# exist while Qwen3-30B-A3B-UD-IQ3_XXS.gguf does.
from engines.model_catalog import choose_gguf  # noqa: E402

# Preloading is off by default. It cost every session minutes before it could
# do anything, and the model it chose was whatever this file happened to
# default to -- so asking for a picture meant loading a language model first
# and immediately evicting it. A model already on Drive is still located,
# because that makes a later on-demand load instant. Set PRELOAD_MODEL=1 for
# the old behaviour.
PRELOAD = os.environ.get("PRELOAD_MODEL", "0").strip().lower() not in ("0", "false", "no", "")

existing = glob.glob(os.path.join(MODEL_DIR, "*.gguf"))
if not existing and not PRELOAD:
    model_path = ""
    size_gb = 0.0
    print(chr(10) + "-> nothing on Drive and preloading is off; starting with no model.")
    print("   The worker connects immediately; load what you want from the dashboard.")
elif existing:
    # Best quantisation present, not the alphabetically first: with both a
    # Q4_K_M and an IQ3_XXS on Drive, sorted()[0] silently loaded the worse one.
    model_path = pick_gguf(existing)
    size_gb = os.path.getsize(model_path) / 1e9
    print("\n-> model already on Drive: %s (%.1f GB)" % (model_path, size_gb))
else:
    filename, expected = choose_gguf(REPO, headroom_gb * 1e9)
    model_path = os.path.join(MODEL_DIR, filename)
    print("\n-> %.1f GB usable -> %s (%.1f GB)" % (headroom_gb, filename, expected / 1e9))
    print("   downloading to Drive (one time, ~15-40 min)")
    sh("pip install -q huggingface_hub")
    from huggingface_hub import hf_hub_download

    # Download to local disk first: writing 18 GB straight through the Drive
    # FUSE mount is markedly slower and more failure-prone than copying.
    tmp = hf_hub_download(
        repo_id=REPO, filename=filename, local_dir="/content/model-dl",
        token=secret("HF_TOKEN") or None,
    )
    sh('cp "%s" "%s"' % (tmp, model_path))
    size_gb = os.path.getsize(model_path) / 1e9
    print("   saved to Drive (%.1f GB) — later runs skip this" % size_gb)


HF_URL = "https://huggingface.co/{repo}/resolve/main/{name}"


def _read_speed(path: str, sample: int = 192 << 20) -> float:
    """MB/s reading the head of a file, in big sequential chunks."""
    started = time.time()
    read = 0
    with open(path, "rb", buffering=0) as fh:
        while read < sample:
            chunk = fh.read(8 << 20)
            if not chunk:
                break
            read += len(chunk)
    elapsed = max(time.time() - started, 1e-6)
    return read / elapsed / 1e6


def _hf_speed(repo: str, name: str, sample: int = 64 << 20) -> float:
    """MB/s pulling a byte range from the HF CDN, or 0.0 if it is unreachable."""
    url = HF_URL.format(repo=repo, name=name)
    headers = {"Range": "bytes=0-%d" % (sample - 1), "User-Agent": "nexalm-probe"}
    tok = secret("HF_TOKEN")
    if tok:
        headers["Authorization"] = "Bearer " + tok
    try:
        req = urllib.request.Request(url, headers=headers)
        started = time.time()
        read = 0
        with urllib.request.urlopen(req, timeout=60) as res:
            while read < sample:
                chunk = res.read(8 << 20)
                if not chunk:
                    break
                read += len(chunk)
        elapsed = max(time.time() - started, 1e-6)
        return read / elapsed / 1e6
    except Exception as exc:  # noqa: BLE001 — a probe failure just means "prefer Drive"
        print("   HF probe failed (%s); will use Drive" % exc.__class__.__name__)
        return 0.0


def obtain_local_model(drive_path: str, repo: str) -> str:
    """Return a local-disk path to the GGUF, fetching it by whichever route is faster.

    llama.cpp mmaps the model, and mmap over the Drive FUSE mount is
    pathological: every page fault is a round trip with no readahead to
    amortise it, so the load crawls while the GPU sits idle — which is also
    what trips Colab's idle-GPU watchdog. So the weights have to reach local
    disk one way or another.

    Which route is faster is *not* obvious, and guessing it wrong is what made
    an earlier version of this slower rather than faster: Drive FUSE reads and
    the HF CDN differ by several times in either direction depending on the
    runtime you are handed. So both are measured on a small sample and the
    winner moves the remaining ~12 GB. The probe costs a few seconds against a
    transfer of several minutes.

    The staged copy persists for the life of the session, so restarting the
    worker after the first run skips all of this.
    """
    name = os.path.basename(drive_path)
    dest = os.path.join("/content/models", name)
    need = os.path.getsize(drive_path)

    if os.path.exists(dest) and os.path.getsize(dest) == need:
        print("-> model already on local disk: %s" % dest)
        return dest

    free = shutil.disk_usage("/content").free
    if free < need * 1.15:
        print("   local disk has %.1f GB free, model needs %.1f GB — "
              "loading straight from Drive (slower, but it will work)"
              % (free / 1e9, need / 1e9))
        return drive_path

    os.makedirs("/content/models", exist_ok=True)
    drive_mbps = _read_speed(drive_path)
    hf_mbps = _hf_speed(repo, name)
    print("-> transfer probe: Drive %.0f MB/s vs HF CDN %.0f MB/s" % (drive_mbps, hf_mbps))

    started = time.time()
    # Only switch to HF on a clear win: it re-downloads bytes already paid for,
    # and a marginal difference is not worth the risk of a mid-transfer failure.
    if hf_mbps > drive_mbps * 1.25:
        print("   fetching from HF (~%.0f min at that rate)" % (need / (hf_mbps * 1e6) / 60))
        sh("pip install -q huggingface_hub")
        from huggingface_hub import hf_hub_download

        try:
            got = hf_hub_download(repo_id=repo, filename=name, local_dir="/content/models",
                                  token=secret("HF_TOKEN") or None)
            if os.path.abspath(got) != os.path.abspath(dest):
                dest = got
        except Exception as exc:  # noqa: BLE001
            print("   HF download failed (%s); falling back to Drive" % exc.__class__.__name__)
            sh('cp "%s" "%s"' % (drive_path, dest))
    else:
        print("   copying from Drive (~%.0f min at that rate)"
              % (need / (max(drive_mbps, 1) * 1e6) / 60))
        sh('cp "%s" "%s"' % (drive_path, dest))

    if not os.path.exists(dest) or os.path.getsize(dest) != need:
        print("   staged copy is the wrong size; using the Drive path instead")
        return drive_path
    print("   staged in %.0fs" % (time.time() - started))
    return dest


model_path = obtain_local_model(model_path, REPO)

# ── 4. llama.cpp, also cached on Drive ──────────────────────────────────────
llama_server = os.path.join(BIN_DIR, "llama-server")


def fetch_prebuilt() -> bool:
    """Pull the prebuilt sm_75 binaries from the NexaLM server.

    Compiling here is close to pointless: the output is identical every run,
    and Colab's GPU runtime gives ~2 vCPUs, so the build takes over an hour
    while the GPU sits idle — which is also what trips Colab's idle-GPU
    watchdog and can disconnect the session mid-build.

    sm_75 is Turing, the same architecture as Kaggle's T4, so one build serves
    both platforms.
    """
    base = ORACLE_WS_URL.split("://")[1].split("/")[0]
    url = "https://%s/api/artifacts/llamacpp-cuda-sm75.tar.gz" % base
    tarball = "/content/llamacpp-sm75.tar.gz"

    # urllib with an Authorization header rather than curl with ?token=:
    # sh() echoes every command it runs, so a token in the URL ends up in the
    # notebook's saved output. The server accepts either form
    # (server/src/routes/artifacts.ts:30-37); only one of them is safe here.
    print("-> fetching prebuilt llama.cpp from %s" % base)
    req = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer " + WORKER_TOKEN, "User-Agent": "nexalm-worker"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as res:
            with open(tarball, "wb") as out:
                shutil.copyfileobj(res, out, 8 << 20)
    except urllib.error.HTTPError as exc:
        # Say which failure it was. A silent fallback here costs half an hour
        # of compiling with the GPU idle, and the operator deserves the reason.
        print("   server returned HTTP %s (%s)" % (exc.code, exc.reason))
        return False
    except Exception as exc:  # noqa: BLE001 — any failure means "no prebuilt"
        print("   download failed: %s: %s" % (exc.__class__.__name__, exc))
        return False

    if os.path.getsize(tarball) < 1_000_000:
        print("   downloaded file is too small to be the archive")
        return False
    # Extracted to local disk rather than Drive. The archive stores repeated
    # files as hard links, and the Drive FUSE mount cannot create them at all --
    # "Operation not permitted", which is where this used to die on a fresh
    # account. Local disk handles them fine. At ~49 MB the download takes about
    # a second, so caching these on Drive was never worth a trip through FUSE.
    sh('rm -rf "%s" && mkdir -p "%s"' % (LOCAL_BIN, LOCAL_BIN))
    sh('tar -xzf "%s" -C "%s"' % (tarball, LOCAL_BIN))
    return os.path.exists(os.path.join(LOCAL_BIN, "llama-server"))


# The prebuilt binaries -- whether cached on Drive or fetched from the server --
# are sm_75 only. On anything else they abort inside a CUDA kernel.
_prebuilt_ok = prebuilt_runs_here(gpu_lines)
if not _prebuilt_ok:
    print()
    print("!! " + arch_mismatch_message(gpu_lines))
    print("   Ignoring the prebuilt binaries for this card.")

_bin_is_local = False
if _prebuilt_ok and fetch_prebuilt():
    print("-> fetched prebuilt llama.cpp to local disk (skipped the ~1h build)")
    _bin_is_local = True
elif _prebuilt_ok and os.path.exists(llama_server):
    print("-> server unreachable; falling back to the copy cached on Drive")
elif os.environ.get("ALLOW_SOURCE_BUILD") == "1":
    print("\n-> building llama.cpp with CUDA (slow here: Colab gives ~2 vCPUs)")
    sh("apt-get update -qq && apt-get install -y -qq cmake build-essential libcurl4-openssl-dev")
    if not os.path.isdir("/content/llama.cpp/.git"):
        sh("rm -rf /content/llama.cpp")
        sh("git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp")

    # Same CUDA::cuda_driver problem as Kaggle: the driver ships as
    # libcuda.so.1 and CMake needs an unversioned libcuda.so to link against.
    ensure_libcuda_link()

    arch = cuda_arch(gpu_lines)
    sh(
        f"cmake -S /content/llama.cpp -B /content/llama.cpp/build "
        f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} -DLLAMA_CURL=OFF "
        f"-DCMAKE_BUILD_TYPE=Release "
        f"'-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64;/usr/local/cuda/lib64/stubs' && "
        f"cmake --build /content/llama.cpp/build --target llama-server -j$(nproc)"
    )
    sh(f'mkdir -p {LOCAL_BIN} && cp -r /content/llama.cpp/build/bin/. {LOCAL_BIN}/')
    _bin_is_local = True
else:
    # Compiling takes 20-30 minutes on Colab's ~2 vCPUs with the GPU idle at
    # 0 GB, and that is also what trips the idle-GPU watchdog. Falling into it
    # silently because a download failed wastes half an hour without saying
    # why, so it now has to be asked for.
    print(
        "\n!! Could not fetch the prebuilt llama.cpp binaries (reason above).\n"
        "   Not compiling from source: that takes 20-30 minutes here with the\n"
        "   GPU idle. Check the server is reachable and WORKER_TOKEN is right,\n"
        "   then re-run. To compile anyway, set ALLOW_SOURCE_BUILD=1."
    )
    sys.exit(1)

# Run from local disk: the Drive FUSE mount cannot hold the exec bit, and
# loading .so files across it is needlessly slow. Skip the copy when a correct
# local copy already survives from earlier in this session -- the previous
# version printed "already on Drive" and then re-copied the whole directory off
# FUSE anyway, so the message and the actual cost disagreed.
_staged = os.path.join(LOCAL_BIN, "llama-server")
if not _bin_is_local:
    _source = os.path.join(BIN_DIR, "llama-server")
    if os.path.isfile(_staged) and os.path.getsize(_staged) == os.path.getsize(_source):
        print("-> llama.cpp already staged on local disk")
    else:
        sh(f'rm -rf {LOCAL_BIN} && cp -r "{BIN_DIR}" {LOCAL_BIN}')
sh(f"chmod +x {LOCAL_BIN}/* 2>/dev/null || true", check=False)
llama_server = _staged

# ── 5. Worker code, synced to Drive ─────────────────────────────────────────
if not os.path.exists(os.path.join(CODE_DIR, "worker.py")):
    sys.exit(
        f"Worker code not found in {CODE_DIR}.\n"
        "Copy worker.py and engines/ from the repository into that folder, or run:\n"
        f"  !git clone <repo> /content/repo && cp -r /content/repo/worker/* {CODE_DIR}/"
    )

sh("pip install -q 'websockets>=13,<15' 'httpx>=0.27,<0.29'")

# ── 6. Placement and launch ─────────────────────────────────────────────────
if model_path:
    size_gb = os.path.getsize(model_path) / 1e9
    n_gpu_layers, placement = plan_placement(size_gb, headroom_gb)
else:
    # Nothing to place yet; every later load sizes itself.
    size_gb, n_gpu_layers, placement = 0.0, 999, "decided per model, on demand"
context_length = int(os.environ.get("CONTEXT_LENGTH", str(default_context(headroom_gb))))

env = {
    **os.environ,
    "ORACLE_WS_URL": ORACLE_WS_URL,
    "WORKER_TOKEN": WORKER_TOKEN,
    "WORKER_NAME": os.environ.get("WORKER_NAME", f"colab-{gpu_count}x{gpu_lines[0].split(',')[0].strip().replace(' ', '-')}"),
    "WORKER_ID": os.environ.get("WORKER_ID", "colab-worker-1"),
    "ENGINE": "llamacpp",
    # Stated, not inferred: probing the filesystem for /kaggle or /content
    # mislabelled a Colab worker as Kaggle, and this file only ever runs on
    # one of them.
    "WORKER_PLATFORM": "colab",
    "PRELOAD_MODEL": "1" if PRELOAD else "0",
    # Weights land on Drive so a new session does not re-download them. Without
    # this, anything loaded on demand went to /content, which is wiped when the
    # session ends -- so FLUX cost a 10 GB download every single time.
    # Free Drive is 15 GB, so FLUX (10.0 GB) and a chat model will not both fit;
    # SD 1.5 at 1.6 GB leaves room for one.
    "MODEL_CACHE_DIR": MODEL_DIR,
    # The sd-server build stays on local disk: it is cheap to replace, it is
    # published to the control plane for other sessions anyway, and compiling
    # through the Drive FUSE mount is markedly slower.
    "WORKER_SCRATCH_DIR": "/content/nexalm-build",
    "MODEL_ID": model_id_for(model_path) if model_path else "",
    "MODEL_PATH": model_path,
    "LLAMA_SERVER_BIN": llama_server,
    "CONTEXT_LENGTH": str(context_length),
    "N_GPU_LAYERS": str(n_gpu_layers),
    "LD_LIBRARY_PATH": loader_path(llama_server),
    # Explicit rather than inherited. A forgotten Colab tab holding a GPU is
    # wasted capacity, and pinning this to 0 disabled the timer entirely.
    "IDLE_SHUTDOWN_MINUTES": os.environ.get("IDLE_SHUTDOWN_MINUTES", "30"),
    "LOG_LEVEL": "INFO",
    "PYTHONUNBUFFERED": "1",
}

print("\n" + "=" * 70)
print("Starting worker")
print(f"  GPUs         : {gpu_count} ({total_vram_gb:.1f} GB total)")
print(f"  placement    : {placement}")
print(f"  context      : {context_length}")
print(f"  model        : {model_path}")
print(f"  oracle       : {ORACLE_WS_URL.split('://')[0]}://…  (token withheld)")
print("=" * 70 + "\n")

os.chdir(CODE_DIR)
os.execve(sys.executable, [sys.executable, "worker.py"], env)
