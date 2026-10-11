"""Provisioning + launch logic for the Kaggle GPU worker.

Shipped INSIDE the worker-code dataset, not in the kernel. The kernel is a
thin loader that execs this file, so iterating on provisioning only needs a
dataset version — never a `kaggle kernels push`, which silently resets both the
notebook accelerator and its attached Secrets.

Runs as a Kaggle *script* kernel with GPU and internet enabled. It provisions
llama.cpp, locates the model, and hands control to the ordinary worker client —
the same code a Colab or dedicated-GPU host would run.

Expected attached inputs (see kernel-metadata.json):
  /kaggle/input/local-llm-worker-code/   worker.py + engines/
  /kaggle/input/<model-dataset>/         the .gguf, cached as a Kaggle Dataset

Required Kaggle Secret (Add-ons -> Secrets in the notebook editor):
  WORKER_TOKEN    must match the server's WORKER_TOKEN

Add it with the LABEL exactly "WORKER_TOKEN" and the token as its value.
`secret()` looks a secret up BY NAME — putting the value in the call is the
one mistake that silently produces "missing secret".

The Oracle URL is not a secret and is a plain default below. Nothing secret is
ever written to this file.
"""

import glob
import json
import os
import subprocess
import sys

# ── 0. What hardware did Kaggle actually give us? ───────────────────────────
# The free tier is not a fixed allocation: a session may return 2x T4 (32 GB),
# 1x P100 (16 GB) or 1x T4 (16 GB). Every decision below keys off what is
# actually present — this is measured, never assumed.
# Provisioning decisions live in worker/provision/common.py so that Kaggle and
# Colab cannot answer them differently. The module ships in the same private
# dataset as the rest of the worker code.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _candidate in glob.glob("/kaggle/input/**/provision", recursive=True):
    sys.path.insert(0, os.path.dirname(_candidate))
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

print("GPU allocation for this session")
gpu_lines, gpu_count, total_vram_gb, headroom_gb = probe_gpus()

if gpu_count == 0:
    print()
    print("!! This Kaggle session has no GPU attached.")
    print("   Session options -> Accelerator -> GPU T4 x2, then re-run.")
    sys.exit(1)

# /kaggle/working is the *output* directory, capped at ~19.5 GB and uploaded as
# kernel output when the run ends. An 18.6 GB model dropped there nearly fills
# the quota and adds a pointless upload. Scratch space is neither capped that
# way nor collected.
SCRATCH = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"

if gpu_count == 1 and total_vram_gb < 20:
    print(
        "\n!! Only one GPU with %.0f GB. Qwen3-30B-A3B at Q4_K_M is ~18.6 GB of\n"
        "   weights alone. Setting the notebook accelerator to 'GPU T4 x2' gives\n"
        "   32 GB and runs the intended quantisation. Continuing with a smaller\n"
        "   quantisation and/or CPU offload, which will be slower.\n" % total_vram_gb
    )

# ── 1. Secrets ──────────────────────────────────────────────────────────────
def worker_dir_hint() -> str:
    """Directory of the attached worker-code dataset, for importing engines."""
    hits = sorted(glob.glob("/kaggle/input/**/worker.py", recursive=True))
    return os.path.dirname(hits[0]) if hits else "."


def _dataset_config() -> dict:
    """Config shipped inside the private worker-code dataset.

    Needed because a session started through the Kaggle API cannot read the
    notebook's attached Secrets — pushing a version drops them. The dataset is
    private to the account; Secrets remain preferred and are checked first.
    """
    for path in glob.glob("/kaggle/input/**/worker-config.json", recursive=True):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001 — malformed config must not be fatal
            continue
    return {}


_CONFIG = _dataset_config()


def secret(name: str, default: str = "") -> str:
    """Kaggle Secret, then private-dataset config, then environment."""
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(name)
        if value:
            return value
    except Exception:  # noqa: BLE001 — secret absent or not running on Kaggle
        pass
    return _CONFIG.get(name) or os.environ.get(name, default)


# Public endpoint — not a secret, so it needs no Secrets entry.
ORACLE_WS_URL = secret("ORACLE_WS_URL", "wss://chat.example.com/worker/ws")

# The one real secret. Must match WORKER_TOKEN in the server's .env.
WORKER_TOKEN = secret("WORKER_TOKEN")

# From here on an unhandled failure is reported to the server before the
# process dies. Provisioning failures were previously invisible: the session
# just ended, having spent quota, with nothing to look at.
install_failure_beacon("kaggle", ORACLE_WS_URL, WORKER_TOKEN)

if not WORKER_TOKEN:
    sys.exit(
        "Missing WORKER_TOKEN.\n"
        "\n"
        "In the notebook editor: Add-ons -> Secrets -> Add secret\n"
        "  Label: WORKER_TOKEN      <- the NAME, exactly this\n"
        "  Value: <the token from the server's /opt/local-llm/.env>\n"
        "\n"
        "secret() looks a secret up BY NAME. Passing the token value into the\n"
        "call instead of the label always fails here, and leaves the token\n"
        "sitting in the notebook source."
    )

# ── 2. Model: prefer an attached dataset over re-downloading ────────────────
# A ~18.6 GB pull on every session is the single biggest waste in this design.
# Uploaded once as a private Kaggle Dataset it mounts read-only, instantly.
# One ranked choice rather than an ordered ladder of globs: the ladder only
# knew three quantisation names, so anything else -- Unsloth's UD- dynamic
# quants included -- fell through to "any .gguf, alphabetically".
# Preloading is off by default now. It cost every session minutes before it
# could do anything, and the model it chose was whatever this file happened
# to default to -- so asking for a picture meant loading an 18.6 GB language
# model first and immediately evicting it. Set PRELOAD_MODEL=1 for the old
# behaviour. An attached dataset is still located either way, because that
# makes a later on-demand load instant rather than a download.
PRELOAD = os.environ.get("PRELOAD_MODEL", "0").strip().lower() not in ("0", "false", "no", "")

model_path = pick_gguf(glob.glob("/kaggle/input/**/*.gguf", recursive=True))

if model_path:
    size_gb = os.path.getsize(model_path) / 1e9
    print("\n-> model from attached dataset: %s (%.1f GB)" % (model_path, size_gb))
    if not PRELOAD:
        print("   not preloading it; the dashboard decides what to load.")
elif not PRELOAD:
    size_gb = 0.0
    print(chr(10) + "-> no .gguf attached and preloading is off; starting with no model.")
    print("   The worker connects immediately; load what you want from the dashboard.")
else:
    repo = os.environ.get("HF_REPO", "unsloth/Qwen3-30B-A3B-GGUF")
    # List the repo rather than building a filename from a quantisation name.
    # The constructed form is wrong for Unsloth's dynamic quants, which carry a
    # "UD-" prefix, and the hardcoded size table drifts when a repo is
    # requantised. Real names, real sizes.
    sys.path.insert(0, worker_dir_hint())
    from engines.model_catalog import choose_gguf

    filename, expected = choose_gguf(repo, headroom_gb * 1e9)
    print(
        "\n!! No .gguf attached. %.1f GB usable for weights -> %s (%.1f GB)."
        % (headroom_gb, filename, expected / 1e9)
    )
    print("   Upload it as a private Kaggle Dataset afterwards to skip this next time.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=os.path.join(SCRATCH, "model"),
        token=secret("HF_TOKEN") or None,
    )
    size_gb = os.path.getsize(model_path) / 1e9
    print("   downloaded to %s (%.1f GB)" % (model_path, size_gb))

# ── 3. llama.cpp ────────────────────────────────────────────────────────────
# Prefer prebuilt binaries from an attached dataset; otherwise build for the
# detected architecture. T4 is sm_75 (Turing); P100 is sm_60 (Pascal).
# An attached llamacpp-cuda-sm75 dataset removes the ~25 minute compile.
# sm_75 is Turing, which is the T4 on both Kaggle and Colab, so one build
# serves either platform.
prebuilt = sorted(glob.glob("/kaggle/input/**/llama-server", recursive=True))
if prebuilt and not prebuilt_runs_here(gpu_lines):
    # Refuse rather than abort later inside a CUDA kernel with a stack trace of
    # hex offsets, which is what running sm_75 code on a Pascal card looks like.
    print()
    print("!! " + arch_mismatch_message(gpu_lines))
    print("   Building from source for this card instead.")
    prebuilt = []
if prebuilt:
    # /kaggle/input is read-only and the binary needs +x, so copy the whole bin
    # directory out: llama-server links against libllama-server-impl.so and the
    # ggml backends, which must sit beside it.
    src_bin = os.path.dirname(prebuilt[0])
    dst_bin = os.path.join(SCRATCH, "llama-bin")
    subprocess.run("rm -rf %s && cp -r %s %s" % (dst_bin, src_bin, dst_bin), shell=True, check=True)
    subprocess.run("chmod +x %s/* 2>/dev/null || true" % dst_bin, shell=True, check=False)
    llama_server = os.path.join(dst_bin, "llama-server")
    print("\n-> llama-server from attached dataset: %s" % prebuilt[0])
    print("   copied to %s (skipping the ~25 min CUDA build)" % dst_bin)
else:
    arch = cuda_arch(gpu_lines)
    src = "/kaggle/working/llama.cpp"
    llama_server = src + "/build/bin/llama-server"

    # Re-running a notebook cell is normal, and /kaggle/working survives it.
    # Every step here is therefore idempotent: a half-finished previous attempt
    # must resume, not abort with "destination path already exists".
    if os.path.exists(llama_server):
        print("\n-> llama-server already built at %s" % llama_server)
    else:
        print("\n-> building llama.cpp for CUDA arch sm_%s (several minutes)" % arch)
        subprocess.run(
            "apt-get update -qq && apt-get install -y -qq cmake build-essential libcurl4-openssl-dev",
            shell=True, check=True,
        )
        if os.path.isdir(os.path.join(src, ".git")):
            print("   reusing existing checkout")
        else:
            # Any partial or non-git leftover is unusable — clear it first.
            subprocess.run("rm -rf %s" % src, shell=True, check=True)
            subprocess.run(
                "git clone --depth 1 https://github.com/ggml-org/llama.cpp %s" % src,
                shell=True, check=True,
            )
        # CMake creates the CUDA::cuda_driver target only when
        # find_library(cuda) succeeds, i.e. when an unversioned `libcuda.so`
        # exists. A GPU node has the driver as `libcuda.so.1` and the toolkit's
        # stubs/ directory is empty, so the link name is missing and configure
        # dies with "Target ggml-cuda links to CUDA::cuda_driver but the target
        # was not found". Provide the link name ourselves.
        ensure_libcuda_link()

        # CMAKE_LIBRARY_PATH is a ';'-separated CMake list. It MUST be quoted:
        # unquoted, the shell treats ';' as a command separator and tries to
        # execute the second path.
        cmake_cmd = (
            "cmake -S {src} -B {src}/build "
            "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} -DLLAMA_CURL=OFF "
            "-DCMAKE_BUILD_TYPE=Release "
            "'-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64;/usr/local/cuda/lib64/stubs' "
            "&& cmake --build {src}/build --target llama-server -j$(nproc)"
        ).format(src=src, arch=arch)
        print("   $ " + cmake_cmd)
        subprocess.run(cmake_cmd, shell=True, check=True, executable="/bin/bash")
        print("   built. Save build/bin as a Kaggle Dataset to skip this next time.")

    if not os.path.exists(llama_server):
        sys.exit("llama.cpp build finished but %s is missing." % llama_server)

# ── 4. Worker code ──────────────────────────────────────────────────────────
# Recursive: a dataset may mount its contents nested rather than at the root,
# and a single-level glob silently misses it.
code_dirs = sorted(glob.glob("/kaggle/input/**/worker.py", recursive=True))
if not code_dirs:
    print("\n!! worker.py not found. Contents of /kaggle/input:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count("/") - 2
        if depth > 3:
            dirs[:] = []
            continue
        print("   %s%s/" % ("  " * depth, os.path.basename(root) or "/kaggle/input"))
        for f in sorted(files)[:12]:
            print("   %s  %s" % ("  " * depth, f))
    sys.exit(
        "Worker code not attached.\n"
        "Run scripts/launch-kaggle-worker.sh, which uploads worker/ as a private dataset,\n"
        "and confirm the dataset appears under Input in the notebook editor."
    )
worker_dir = os.path.dirname(code_dirs[0])
print("\n-> worker code from attached dataset: %s" % worker_dir)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "websockets>=13,<15", "httpx>=0.27,<0.29"],
    check=True,
)

# ── 5. Placement and launch ─────────────────────────────────────────────────
# Blindly passing --n-gpu-layers 999 is what turns a too-small card into an OOM
# instead of a slower-but-working run. llama.cpp keeps the remainder on CPU.
if model_path:
    n_gpu_layers, placement = plan_placement(size_gb, headroom_gb)
    model_id = model_id_for(model_path)
else:
    # Nothing to place yet. Every later load sizes itself, and 999 is already
    # what the manager passes for "as many layers as fit".
    n_gpu_layers, placement = 999, "decided per model, on demand"
    model_id = ""
context_length = int(os.environ.get("CONTEXT_LENGTH", str(default_context(headroom_gb))))

gpu_label = gpu_lines[0].split(",")[0].strip().replace(" ", "-")
env = {
    **os.environ,
    "ORACLE_WS_URL": ORACLE_WS_URL,
    "WORKER_TOKEN": WORKER_TOKEN,
    "WORKER_NAME": os.environ.get("WORKER_NAME", "kaggle-%dx%s" % (gpu_count, gpu_label)),
    "WORKER_ID": os.environ.get("WORKER_ID", "kaggle-worker-1"),
    "ENGINE": "llamacpp",
    "PRELOAD_MODEL": "1" if PRELOAD else "0",
    # Stated, not inferred: probing the filesystem for /kaggle or /content
    # mislabelled a Colab worker as Kaggle, and this file only ever runs on
    # one of them.
    "WORKER_PLATFORM": "kaggle",
    "MODEL_ID": model_id,
    "MODEL_PATH": model_path,
    "LLAMA_SERVER_BIN": llama_server,
    "CONTEXT_LENGTH": str(context_length),
    "N_GPU_LAYERS": str(n_gpu_layers),
    "LD_LIBRARY_PATH": loader_path(llama_server),
    # Explicit rather than inherited: Kaggle bills wall-clock, so a worker left
    # running with nothing to do burns the week's budget.
    "IDLE_SHUTDOWN_MINUTES": os.environ.get("IDLE_SHUTDOWN_MINUTES", "30"),
    "LOG_LEVEL": "INFO",
    "PYTHONUNBUFFERED": "1",
}

print("\n" + "=" * 70)
print("Starting worker")
print("  GPUs         : %d (%.1f GB total)" % (gpu_count, total_vram_gb))
print("  model        : %s" % model_id)
print("  placement    : %s" % placement)
print("  context      : %d" % context_length)
print("  llama-server : %s" % llama_server)
print("  oracle       : %s://…  (token withheld)" % ORACLE_WS_URL.split("://")[0])
print("=" * 70 + "\n")

# exec so signals reach the worker directly and its graceful deregister runs.
os.chdir(worker_dir)
os.execve(sys.executable, [sys.executable, "worker.py"], env)
