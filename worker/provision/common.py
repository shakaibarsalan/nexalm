"""Provisioning logic shared by the Kaggle and Colab bootstraps.

The two bootstraps had grown nine independent copies of the same decisions --
GPU probing, VRAM headroom, layer placement, context sizing, model naming, the
libcuda link fixup, the CUDA architecture choice. Copies drift, and they drifted
in ways that broke one platform while leaving the other fine: Colab picked its
model alphabetically so IQ3 beat Q4, and its CUDA fallback targeted sm_80 on a
card that can only run sm_60.

Everything platform-specific -- where the weights come from, how secrets are
resolved, how the binaries arrive -- deliberately stays in the bootstraps. What
lives here is the part that should never have had two answers.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Reserve for the CUDA context, activations and KV cache; the rest is what the
# weights may occupy on-device.
VRAM_RESERVE_GB = 2.5

N_LAYERS = 48  # Qwen3-30B-A3B

# Quantisation tags in descending quality, matching engines/model_catalog.py.
# Duplicated as a literal rather than imported because this module is used
# before the engines package is guaranteed to be importable on either platform.
QUANT_TAGS = [
    "Q8_0", "Q6_K_XL", "Q6_K", "Q5_K_M", "Q5_K_S",
    "Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "IQ4_XS",
    "Q3_K_M", "UD-Q3_K_XL", "Q3_K_S", "UD-IQ3_XXS",
    "UD-Q2_K_XL", "Q2_K_L", "Q2_K", "UD-IQ2_M", "UD-IQ2_XXS",
]

# Anything at Q4 or better is the full-quality model as far as the server's
# model table is concerned; below that is the reduced single-GPU fallback.
_FULL_QUALITY_CUTOFF = QUANT_TAGS.index("IQ4_XS")


def detect_quant(filename: str) -> str:
    """The quantisation tag in a GGUF filename, longest match first.

    Longest-first matters: "Q2_K" is a substring of "Q2_K_L", and matching the
    short tag first would label the larger file as the smaller one.
    """
    for tag in sorted(QUANT_TAGS, key=len, reverse=True):
        if tag in filename:
            return tag
    return "unknown"


def quant_rank(filename: str) -> int:
    """Position in QUANT_TAGS; unknown quantisations sort last."""
    tag = detect_quant(filename)
    return QUANT_TAGS.index(tag) if tag in QUANT_TAGS else len(QUANT_TAGS)


def pick_gguf(paths: list[str]) -> str:
    """Best-quality GGUF from a list of candidates.

    Colab used to take the alphabetically first file, so with both a Q4_K_M and
    an IQ3_XXS on Drive it silently loaded the worse one and then announced it
    under the better one's name.
    """
    if not paths:
        return ""
    return sorted(paths, key=lambda p: (quant_rank(os.path.basename(p)), os.path.basename(p)))[0]


def model_id_for(model_path: str) -> str:
    """The server-side model id this file should register as.

    The server only knows two GPU model ids -- the full one and a reduced
    fallback -- so this classifies by quantisation family. It used to test for
    the literal string "Q4_K_M", which meant UD-Q4_K_XL, a Q4 by every measure,
    was announced to the server as a Q3.
    """
    return (
        "qwen3-30b-a3b"
        if quant_rank(os.path.basename(model_path)) <= _FULL_QUALITY_CUTOFF
        else "qwen3-30b-a3b-q3"
    )


def probe_gpus(verbose: bool = True) -> tuple[list[str], int, float, float]:
    """(gpu_lines, count, total VRAM GB, usable-for-weights GB).

    Free tiers are not a fixed allocation -- a session may hand back 2x T4,
    1x P100 or 1x T4 -- so every downstream decision keys off what is actually
    present. Measured, never assumed.
    """
    if verbose:
        print("=" * 70)
        try:
            subprocess.run(["nvidia-smi"], check=False)
        except FileNotFoundError:
            # check=False only suppresses a non-zero exit. A missing binary
            # raises from Popen before the process ever starts, which turned
            # "you are on a CPU runtime" into a twenty-line traceback.
            print("nvidia-smi is not installed here, so this runtime has no GPU.")
        print("=" * 70)

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        gpu_lines = [line for line in out.splitlines() if line.strip()]
    except FileNotFoundError:
        print("!! No GPU on this runtime.")
        gpu_lines = []
    except Exception as exc:  # noqa: BLE001 — no GPU is a caller decision
        print("!! nvidia-smi failed (%s); treating this as no GPU." % exc)
        gpu_lines = []

    count = len(gpu_lines)
    total = sum(float(line.split(",")[1]) for line in gpu_lines) / 1024 if gpu_lines else 0.0
    headroom = max(0.0, total - VRAM_RESERVE_GB)
    if verbose:
        print("\n-> %d GPU(s), %.1f GB total VRAM (%.1f GB usable for weights)"
              % (count, total, headroom))
    return gpu_lines, count, total, headroom


def cuda_arch(gpu_lines: list[str]) -> str:
    """CUDA architecture to compile for, from the card actually present.

    Colab defaulted non-T4 cards to sm_80, which on the P100 it sometimes hands
    out builds kernels the card cannot run. Both free tiers only ever offer
    Turing (T4, sm_75) or Pascal (P100, sm_60).
    """
    joined = " ".join(gpu_lines)
    if "T4" in joined:
        return "75"
    if "P100" in joined:
        return "60"
    if "V100" in joined:
        return "70"
    if any(tag in joined for tag in ("A100", "L4", "A10")):
        return "80"
    return "75"


def plan_placement(size_gb: float, headroom_gb: float) -> tuple[int, str]:
    """(n_gpu_layers, human description).

    Blindly passing --n-gpu-layers 999 is what turns a too-small card into an
    OOM instead of a slower-but-working run; llama.cpp keeps the remainder on
    CPU quite happily.
    """
    if size_gb <= headroom_gb:
        return 999, "all layers on GPU (%.1f GB of %.1f GB usable)" % (size_gb, headroom_gb)
    layers = max(0, int(N_LAYERS * (headroom_gb / size_gb)))
    return layers, (
        "partial offload: ~%d/%d layers on GPU (%.1f GB model vs %.1f GB usable) "
        "— expect lower tokens/sec" % (layers, N_LAYERS, size_gb, headroom_gb)
    )


def default_context(headroom_gb: float) -> int:
    """Context window sized to the card: the KV cache competes with weights."""
    return 16384 if headroom_gb >= 20 else 8192 if headroom_gb >= 12 else 4096


# CMake needs an unversioned libcuda.so to link CUDA::cuda_driver against, but a
# GPU node ships the driver as libcuda.so.1 and leaves the toolkit's stubs/
# directory empty. Without the link name, configure dies with "Target ggml-cuda
# links to CUDA::cuda_driver but the target was not found".
_ENSURE_LIBCUDA = r"""
set -e
target_dir=/usr/local/cuda/lib64
mkdir -p "$target_dir"
if [ -e "$target_dir/libcuda.so" ]; then
  echo "   libcuda.so already present"
else
  cand=""
  for p in $(ldconfig -p 2>/dev/null | grep -oE '/[^ ]*libcuda\.so(\.[0-9]+)*' | sort -u) \
           /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
           /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
           /usr/lib64/libcuda.so.1; do
    if [ -e "$p" ]; then cand="$p"; break; fi
  done
  if [ -z "$cand" ]; then
    cand=$(find / -xdev -name 'libcuda.so*' -not -path '*/proc/*' 2>/dev/null | head -1)
  fi
  if [ -n "$cand" ]; then
    ln -sf "$cand" "$target_dir/libcuda.so"
    echo "   linked $target_dir/libcuda.so -> $cand"
  else
    echo "   WARNING: no libcuda.so* found on this machine"
  fi
fi
echo "   --- libcuda visibility ---"
ls -l "$target_dir/libcuda.so" 2>/dev/null || echo "   (still missing)"
ldconfig -p 2>/dev/null | grep -i libcuda || echo "   (nothing in ldconfig)"
"""


def ensure_libcuda_link() -> None:
    """Create the unversioned libcuda.so link name a CUDA build needs.

    Colab had a one-line ldconfig-only version of this and Kaggle had the
    hardened search; the hardened one is now what both get.
    """
    subprocess.run(_ENSURE_LIBCUDA, shell=True, check=False, executable="/bin/bash")


def install_deps() -> None:
    """Runtime dependencies for worker.py."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "websockets>=13,<15", "httpx>=0.27,<0.29"],
        check=True,
    )


def loader_path(llama_server: str) -> str:
    """LD_LIBRARY_PATH value that lets llama-server find its own libraries.

    llama-server links against libllama-server-impl.so and the ggml backend
    shared objects sitting beside it, so its directory has to be on the loader
    path wherever it came from.
    """
    return os.path.dirname(llama_server) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


def install_failure_beacon(platform: str, oracle_ws_url: str, worker_token: str) -> None:
    """Report an unhandled provisioning failure to the server before dying.

    A GPU session that dies during provisioning is otherwise a black box. A
    Kaggle session started through the API is interactive, so it saves no
    kernel output, and `kernels output` returns the last committed run instead
    -- so a failed session can burn several minutes of a finite weekly quota
    and leave nothing to diagnose.

    Installed as an excepthook rather than wrapping the bootstraps in a giant
    try block: the bootstraps are top-to-bottom scripts, and an excepthook
    catches every failure in them without restructuring either one.
    """
    import json
    import traceback
    import urllib.request

    if not (oracle_ws_url and worker_token):
        return

    host = oracle_ws_url.split("://")[-1].split("/")[0]
    url = "https://%s/api/workers/report" % host

    def report(stage: str, message: str, detail: str) -> None:
        payload = json.dumps(
            {"platform": platform, "stage": stage, "message": message, "detail": detail}
        ).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + worker_token,
                "User-Agent": "nexalm-worker",
            },
        )
        try:
            urllib.request.urlopen(req, timeout=20).read()
            print("-> reported the failure to %s" % host)
        except Exception as exc:  # noqa: BLE001 — reporting must never mask the real error
            print("-> could not report the failure (%s)" % exc.__class__.__name__)

    previous = sys.excepthook

    def hook(exc_type, exc, tb):  # noqa: ANN001
        # SystemExit carrying a message is how these scripts refuse to start
        # (no GPU, no worker code); it is a real failure worth reporting, while
        # a plain exit(0) is not.
        if issubclass(exc_type, SystemExit):
            if exc.code not in (0, None):
                report("exit", str(exc.code), "")
        else:
            report(
                "exception",
                "%s: %s" % (exc_type.__name__, exc),
                "".join(traceback.format_exception(exc_type, exc, tb)),
            )
        previous(exc_type, exc, tb)

    sys.excepthook = hook


# The prebuilt binaries shipped by the server and by the Kaggle dataset are
# compiled for sm_75 only.
PREBUILT_ARCH = "75"


def prebuilt_runs_here(gpu_lines: list[str]) -> bool:
    """Whether the prebuilt sm_75 binaries can execute on this GPU.

    Running them on a Pascal card does not fail politely: llama-server aborts
    with SIGABRT somewhere inside a fused CUDA kernel, and the only clue is a
    stack trace full of hex offsets. Checking the architecture first turns that
    into a sentence.
    """
    return cuda_arch(gpu_lines) == PREBUILT_ARCH


def arch_mismatch_message(gpu_lines: list[str]) -> str:
    name = gpu_lines[0].split(",")[0].strip() if gpu_lines else "unknown GPU"
    return (
        "The prebuilt llama.cpp binaries are compiled for sm_%s (Turing, T4) but this "
        "session was given a %s (sm_%s). Running them anyway aborts inside a CUDA "
        "kernel with no useful message." % (PREBUILT_ARCH, name, cuda_arch(gpu_lines))
    )
