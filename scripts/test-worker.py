#!/usr/bin/env python3
"""Offline tests for the worker's decision-making.

Everything here runs without a GPU, without a network and without a server,
because these are the decisions that are painful to debug once they are running
on a rented T4 twenty minutes into a session: which GGUF to load, whether it
fits, what to evict, which CUDA architecture to build for, and whether a token
can leak into a notebook's saved output.

    python3 scripts/test-worker.py

Each check names the real failure it exists to prevent.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import shutil
import subprocess
import sys
import base64
import platform
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))

passed = failed = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {label}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f" — {detail}" if detail else ""))


# ── provisioning decisions ────────────────────────────────────────────────
print("Provisioning")
from provision import common as c  # noqa: E402

# Colab used to take sorted(glob(...))[0], so with both files cached on Drive it
# loaded the worse one and announced it under the better one's name.
files = [
    "/d/Qwen3-30B-A3B-IQ3_XXS.gguf",
    "/d/Qwen3-30B-A3B-Q4_K_M.gguf",
    "/d/Qwen3-30B-A3B-UD-Q2_K_XL.gguf",
]
check(
    "best quantisation wins over the alphabetically first",
    c.pick_gguf(files).endswith("Q4_K_M.gguf"),
    os.path.basename(c.pick_gguf(files)),
)

# The id was derived by testing for the literal "Q4_K_M", so UD-Q4_K_XL -- a Q4
# by every measure -- was announced to the server as a Q3.
check(
    "UD-Q4_K_XL is announced as the full model, not the fallback",
    c.model_id_for("Qwen3-30B-A3B-UD-Q4_K_XL.gguf") == "qwen3-30b-a3b",
    c.model_id_for("Qwen3-30B-A3B-UD-Q4_K_XL.gguf"),
)
check(
    "a Q2 build is announced as the reduced model",
    c.model_id_for("Qwen3-30B-A3B-UD-Q2_K_XL.gguf") == "qwen3-30b-a3b-q3",
)
check("longest quantisation tag wins", c.detect_quant("m-Q2_K_L.gguf") == "Q2_K_L")

# Colab defaulted any non-T4 to sm_80; a P100 cannot execute those kernels.
check("T4 builds for Turing", c.cuda_arch(["Tesla T4, 15360"]) == "75")
check("P100 builds for Pascal, not Ampere", c.cuda_arch(["Tesla P100-PCIE-16GB, 16280"]) == "60")

# sm_75 binaries on a Pascal card abort inside a CUDA kernel with nothing but
# hex offsets to show for it.
check("prebuilt sm_75 binaries accepted on a T4", c.prebuilt_runs_here(["Tesla T4, 15360"]))
check("prebuilt sm_75 binaries refused on a P100", not c.prebuilt_runs_here(["Tesla P100, 16280"]))

layers, text = c.plan_placement(18.6, 12.5)
check(
    "an oversized model offloads rather than OOMs",
    0 < layers < c.N_LAYERS and "partial offload" in text,
    f"{layers}/{c.N_LAYERS} layers",
)
check("a fitting model keeps every layer on the GPU", c.plan_placement(11.8, 12.5)[0] == 999)
# A CPU runtime has no nvidia-smi at all. subprocess check=False only suppresses
# a non-zero exit; a missing binary raises from Popen before the process starts,
# which turned "you are on a CPU runtime" into a twenty-line traceback.
_real_run = c.subprocess.run
c.subprocess.run = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi"))
try:
    lines, count, total, headroom = c.probe_gpus(verbose=True)
    check("a runtime with no GPU reports instead of crashing", count == 0 and total == 0.0)
except FileNotFoundError:
    check("a runtime with no GPU reports instead of crashing", False, "raised FileNotFoundError")
finally:
    c.subprocess.run = _real_run

check(
    "context is sized to the card",
    (c.default_context(29.5), c.default_context(12.5), c.default_context(6.0)) == (16384, 8192, 4096),
)


# ── the resident model set ────────────────────────────────────────────────
print()
print("Model manager")


class FakeEngine:
    """Stands in for llama-server: the manager's job here is bookkeeping."""

    stopped: list[str] = []

    def __init__(self, path, **kw):
        self.path, self.port = path, kw.get("port")

    async def start(self):
        pass

    async def stop(self):
        FakeEngine.stopped.append(self.path)

    def cancel(self, job_id):
        pass


sys.modules["engines.llamacpp"] = types.ModuleType("engines.llamacpp")
sys.modules["engines.llamacpp"].LlamaCppEngine = FakeEngine
from models import ModelManager, ModelLoadError  # noqa: E402

tmp = tempfile.mkdtemp()

# Sizes are reported, never allocated. An earlier version of this file created
# the fixtures with truncate(), which NTFS honours literally rather than
# sparsely -- forty gigabytes per run, and a full disk after a few of them.
SIZES: dict[str, float] = {}


def fake_model(model_id: str, gb: float) -> None:
    d = os.path.join(tmp, model_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{model_id}-Q4_K_M.gguf")
    open(path, "wb").close()
    SIZES[os.path.abspath(path)] = gb * 1e9


for mid, gb in (("gemma-3-12b", 7.3), ("qwen3-14b", 9.0), ("qwen3-8b", 5.0), ("qwen3-32b", 19.8)):
    fake_model(mid, gb)

import models as _models  # noqa: E402

_real_getsize = _models.os.path.getsize
_models.os.path.getsize = lambda p: SIZES.get(os.path.abspath(p), _real_getsize(p))


async def manager_checks() -> None:
    m = ModelManager(model_dir=tmp, total_vram_gb=29.5, context_length=8192)
    await m.load("gemma-3-12b")
    await m.load("qwen3-14b")
    await m.load("qwen3-8b")
    check("three models fit a 2x T4 at once", len(m.resident) == 3, ", ".join(sorted(m.resident)))

    m.resident["qwen3-8b"].last_used = 0
    await m.load("qwen3-32b")
    check(
        "the least-recently-used model is evicted to make room",
        "qwen3-32b" in m.resident and "qwen3-8b" not in m.resident,
        ", ".join(sorted(m.resident)),
    )
    check("each resident model has its own port",
          len({r.port for r in m.resident.values()}) == len(m.resident))

    # Evicting under an in-flight job would kill a generation mid-stream.
    busy = ModelManager(model_dir=tmp, total_vram_gb=14.0, context_length=8192)
    await busy.load("qwen3-8b")
    busy.note_job("qwen3-8b", "job-1")
    msg = ""
    try:
        await busy.load("qwen3-14b")
    except ModelLoadError as e:
        msg = str(e)
    check("a model mid-generation is never evicted", "mid-generation" in msg, msg[:60])

    busy.note_job("qwen3-8b", "job-1", finished=True)
    await busy.load("qwen3-14b")
    check("it becomes evictable once the job finishes", "qwen3-14b" in busy.resident)

    small = ModelManager(model_dir=tmp, total_vram_gb=12.5, context_length=8192)
    msg = ""
    try:
        await small.load("qwen3-32b")
    except ModelLoadError as e:
        msg = str(e)
    check("a model larger than the card is refused with both numbers",
          "in total" in msg and "GB" in msg, msg[:70])


asyncio.run(manager_checks())
_models.os.path.getsize = _real_getsize
shutil.rmtree(tmp, ignore_errors=True)


# ── the engine's own footguns ─────────────────────────────────────────────
print()
print("Engine")

llamacpp_src = (ROOT / "worker" / "engines" / "llamacpp.py").read_text(encoding="utf-8")

# The libraries declare SONAME=libllama.so.0 while the files on disk are
# libllama.so.0.4.0; symlinks survive neither a Kaggle download nor a Drive
# mount, and the loader then finds nothing. llama-server exits 127 in silence.
aliases = []
for name in ["libllama.so.0.4.0", "libggml-base.so.0.23.0", "libmtmd.so.0.4.0"]:
    match = re.match(r"^(?P<stem>.+\.so)\.(?P<major>\d+)(?:\.\d+)+$", name)
    aliases += [f"{match.group('stem')}.{match.group('major')}", match.group("stem")]
check(
    "the SONAME pattern derives the aliases the loader needs",
    aliases[:2] == ["libllama.so.0", "libllama.so"],
    ", ".join(aliases[:3]),
)
check("the engine recreates them before launching", "ensure_soname_links(" in llamacpp_src)
check("vision models can be given a projector", "--mmproj" in llamacpp_src)
check(
    "loading progress is read by carriage return, not readline",
    "os.read(" in llamacpp_src and 'b"\\r"' in llamacpp_src,
)


# ── secrets must not reach a notebook's saved output ──────────────────────
print()
print("Secret handling")

colab_src = (ROOT / "worker" / "colab" / "colab_worker.py").read_text(encoding="utf-8")
code_only = "\n".join(l for l in colab_src.splitlines() if not l.lstrip().startswith("#"))
check("no token is ever put in a URL", "?token=" not in code_only)
check("the artifact fetch uses an Authorization header", "Bearer " in code_only)

ns: dict = {"os": os, "subprocess": subprocess}
block = colab_src[colab_src.index("_SECRETS: set[str] = set()"):colab_src.index("# \u2500\u2500 0. Drive")]
exec(compile(block, "excerpt", "exec"), ns)
tok = "a1b2c3d4" * 8
ns["remember_secret"](tok)
check(
    "a registered token is redacted from any echoed command",
    tok not in ns["redact"](f'curl -H "Authorization: Bearer {tok}" https://x'),
)
ns["remember_secret"]("abc")
check("short strings are not mistaken for secrets", ns["redact"]("abcdef") == "abcdef")
check(
    "a failed download never starts a 30-minute build unasked",
    "ALLOW_SOURCE_BUILD" in colab_src and "Not compiling from source" in colab_src,
)

kaggle_src = (ROOT / "worker" / "kaggle" / "bootstrap.py").read_text(encoding="utf-8")
check('Colab declares its platform outright', 'WORKER_PLATFORM": "colab"' in colab_src)
check('Kaggle declares its platform outright', 'WORKER_PLATFORM": "kaggle"' in kaggle_src)


# ── what actually ships to each platform ──────────────────────────────────
# ── image generation ──────────────────────────────────────────────────────
print()
print("Image generation")

from engines.stablediffusion import (  # noqa: E402
    FAMILY_DEFAULTS,
    WEIGHT_SETS,
    StableDiffusionEngine,
    family_of,
    weights_vram_gb,
)

check("a FLUX id resolves to the flux weight set", family_of("flux-schnell", None) == "flux")
check("an SD id resolves to the sd weight set", family_of("sd-1-5", None) == "sd")
check(
    "an explicit family wins over the id",
    family_of("anything", "flux") == "flux",
)

# Every FLUX file must come from ONE repository. Splitting them across the
# obvious three puts the VAE in black-forest-labs/FLUX.1-schnell, which is
# `gated: auto` -- an anonymous worker fails there *after* downloading 6.7 GB
# of diffusion model, which is the worst possible place to find out.
flux_repos = {repo for repo, _ in WEIGHT_SETS["flux"].values()}
check(
    "every FLUX file comes from a single repository",
    len(flux_repos) == 1,
    next(iter(flux_repos)),
)
check(
    "FLUX pulls all four parts it needs",
    set(WEIGHT_SETS["flux"]) == {"diffusion", "vae", "clip_l", "t5xxl"},
)
check(
    "the T5 encoder is taken quantised",
    "Q4" in WEIGHT_SETS["flux"]["t5xxl"][1],
    WEIGHT_SETS["flux"]["t5xxl"][1],
)
check("SD 1.5 is a single file", set(WEIGHT_SETS["sd"]) == {"model"})

# schnell is the four-step distilled model and wants cfg 1.0; running it with
# SD's 20 steps and cfg 7 produces a burnt image very slowly.
check("FLUX defaults to four steps", FAMILY_DEFAULTS["flux"]["steps"] == 4)
check("FLUX defaults to cfg 1.0", FAMILY_DEFAULTS["flux"]["cfg_scale"] == 1.0)
check("SD 1.5 asks for real guidance", FAMILY_DEFAULTS["sd"]["cfg_scale"] > 1.0)

flux_engine = StableDiffusionEngine(
    {
        "diffusion": "/w/flux.gguf",
        "vae": "/w/ae.safetensors",
        "clip_l": "/w/clip_l.safetensors",
        "t5xxl": "/w/t5.gguf",
    },
    family="flux",
    port=8100,
    server_binary="sd-server",
)
cmd = flux_engine._build_command()
for flag in ("--diffusion-model", "--vae", "--clip_l", "--t5xxl", "--clip-on-cpu"):
    check(f"the FLUX command passes {flag}", flag in cmd)
check(
    "the server is told where to listen",
    "--listen-port" in cmd and "8100" in cmd,
)
check("a four-file model never uses -m", "-m" not in cmd)

# Where each part runs. The text encoders are 3.2 GB that ran on two vCPUs
# while a whole 15 GB card sat at 0%. They are a separate module handing over a
# small conditioning tensor, not half of a split matmul, so the idle card can
# take them without needing the fast interconnect a pair of T4s does not have.
import engines.gpu as _gpu  # noqa: E402


class _Card:
    def __init__(self, i):
        self.index, self.name, self.total_mb = i, "Tesla T4", 15360


_real_detect = _gpu.detect_gpus
_gpu.detect_gpus = lambda: [_Card(0), _Card(1)]
two_card = flux_engine._build_command()
_gpu.detect_gpus = lambda: [_Card(0)]
one_card = flux_engine._build_command()
os.environ["SD_BACKEND"] = "te=cpu"
overridden = flux_engine._build_command()
del os.environ["SD_BACKEND"]
_gpu.detect_gpus = _real_detect

check(
    "a second card takes the text encoders",
    "--backend" in two_card and "te=cuda1,diffusion=cuda0,vae=cuda0" in two_card,
)
check("a second card is not asked to run them on the CPU", "--clip-on-cpu" not in two_card)
check(
    "one card still keeps the encoders off it",
    "--clip-on-cpu" in one_card and "--backend" not in one_card,
)
check(
    "the placement can be overridden for measurement",
    "te=cpu" in overridden,
)

sd_engine = StableDiffusionEngine({"model": "/w/sd15.gguf"}, family="sd", port=8101)
check("a single-file model uses -m", "-m" in sd_engine._build_command())

# The text encoders are deliberately kept on CPU so they do not compete with
# the diffusion model for a T4. Charging them against VRAM would refuse a model
# that fits comfortably.
_real_getsize = os.path.getsize
os.path.getsize = lambda p: {"/w/flux.gguf": 6_690_000_000, "/w/ae.safetensors": 340_000_000,
                             "/w/clip_l.safetensors": 250_000_000, "/w/t5.gguf": 2_750_000_000}[p]
vram = weights_vram_gb(flux_engine.weights)
os.path.getsize = _real_getsize
check(
    "VRAM counts the diffusion model and VAE, not the text encoders",
    6.9 < vram < 7.2,
    f"{vram:.2f} GB",
)

# Routing picks a model by family prefix when a conversation names none. An
# image model in the same resident set is exactly what that path would hand a
# conversation to, so the engine refuses rather than answering with noise.
async def _refuses_chat() -> bool:
    try:
        async for _ in flux_engine.generate("job", [], None):
            return False
    except RuntimeError as exc:
        return "image model" in str(exc)
    return False


check("an image engine refuses a chat turn outright", asyncio.run(_refuses_chat()))

body = flux_engine._request_body("a cat", {})
check("the request carries the prompt", body["prompt"] == "a cat")
check("the request uses the family's step count", body["sample_params"]["sample_steps"] == 4)
check(
    "an explicit step count overrides the default",
    flux_engine._request_body("x", {"steps": 8})["sample_params"]["sample_steps"] == 8,
)

# The dispatch this whole feature turns on: models.py hardcoded LlamaCppEngine,
# which is why loading FLUX produced "unknown model architecture: 'flux'" and
# evicted a working chat model on the way.
models_src = (ROOT / "worker" / "models.py").read_text(encoding="utf-8")
check(
    "the loader dispatches on the catalogue's engine",
    "stablediffusion" in models_src and "_load_image" in models_src,
)
# Scoped to the image path: the chat path evicts too, and comparing against
# the whole file would compare the wrong two lines.
image_body = models_src[models_src.index("async def _load_image") :]
image_body = image_body[: image_body.index("async def _unload")]
check(
    "nothing is evicted before the weights and binary are in hand",
    image_body.index("ensure_sd_server") < image_body.index("await self._evict_for"),
)

# A query token ends up in a saved notebook's output for anyone with the link,
# which happened once already on this project. Comments are stripped first so
# the rule is checked against code rather than against prose describing it.
def _code_only(text: str) -> str:
    return chr(10).join(line.split("#")[0] for line in text.splitlines())


worker_src = (ROOT / "worker" / "worker.py").read_text(encoding="utf-8")
check(
    "the image upload uses an Authorization header, not a query token",
    "Authorization" in worker_src and "?token=" not in _code_only(worker_src),
)
# The status vocabulary. sd-server answers 'generating', which was not in the
# list of in-flight names I guessed, so a generation that was working perfectly
# died on a word. Only the three documented endings are terminal now; anything
# else means "still going", which is the reading that cannot invent a failure.
class _StubResponse:
    def __init__(self, code, payload):
        self.status_code = code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _StubClient:
    """Answers one img_gen submission, then the given statuses in order."""

    def __init__(self, states):
        self.states = list(states)

    async def post(self, url, **kwargs):
        return _StubResponse(202, {"id": "job_1", "queue_position": 0})

    async def get(self, url, **kwargs):
        return _StubResponse(200, self.states.pop(0))


_png = base64.b64encode(bytes([0x89]) + b"PNG" + bytes(16)).decode()


async def _drive(states):
    engine = StableDiffusionEngine({"model": "/w/x.gguf"}, family="sd", port=8102)
    engine._client = _StubClient(states)
    seen = []
    async for frame in engine.generate_image("job", "a cat", {}):
        seen.append(frame)
    return seen


_frames = asyncio.run(
    _drive([
        {"status": "generating", "queue_position": 0},
        {"status": "completed", "result": {"images": [{"b64_json": _png}]}},
    ])
)
check(
    "an unrecognised status is progress, not a failure",
    any(f["type"] == "progress" and f["status"] == "generating" for f in _frames),
)
check(
    "the image still arrives after one",
    any(f["type"] == "image" for f in _frames),
)


def _terminal(states):
    try:
        asyncio.run(_drive(states))
        return None
    except BaseException as exc:  # noqa: BLE001 — CancelledError is not an Exception
        return type(exc).__name__


check(
    "a failed job still raises",
    _terminal([{"status": "failed", "error": {"message": "boom"}}]) == "RuntimeError",
)
check(
    "a cancelled job still cancels",
    _terminal([{"status": "cancelled"}]) == "CancelledError",
)


# A binary for the wrong CPU fails with "cannot execute binary file", which says
# nothing about why. The machine that serves an artifact need not share an
# architecture with the machines that run it -- this project's control plane is
# an Ampere A1 (aarch64) while both GPU hosts are x86_64, and a build produced
# on the server ran nowhere.
from engines.stablediffusion import _elf_machine, _runs_here  # noqa: E402

_elf_dir = tempfile.mkdtemp()
_x86 = os.path.join(_elf_dir, "x86")
_arm = os.path.join(_elf_dir, "arm")
# Minimal ELF headers: e_machine sits at offset 18, little-endian.
_hdr = lambda m: bytes([0x7F]) + b"ELF" + bytes(14) + bytes([m, 0]) + bytes(2)
io.open(_x86, "wb").write(_hdr(0x3E))
io.open(_arm, "wb").write(_hdr(0xB7))
check("an x86_64 binary is recognised", _elf_machine(_x86) == "x86_64")
check("an aarch64 binary is recognised", _elf_machine(_arm) == "aarch64")
check(
    "a binary for the wrong CPU is refused rather than run",
    not _runs_here(_arm if platform.machine() in ("x86_64", "AMD64") else _x86),
)
shutil.rmtree(_elf_dir, ignore_errors=True)

sd_src = (ROOT / "worker" / "engines" / "stablediffusion.py").read_text(encoding="utf-8")
check(
    "the binary download uses an Authorization header too",
    "Authorization" in sd_src and "?token=" not in _code_only(sd_src),
)


print()
print("Packaging")

nb = (ROOT / "worker" / "colab" / "NexaLM_Colab_Worker.ipynb").read_text(encoding="utf-8")
import base64, json, tarfile  # noqa: E402

payload = re.search(r'PAYLOAD_B64\s*=\s*"([A-Za-z0-9+/=\\n\s]+?)"', "".join(
    "".join(cell["source"]) for cell in json.loads(nb)["cells"]
))
names = tarfile.open(
    fileobj=io.BytesIO(base64.b64decode(re.sub(r"[^A-Za-z0-9+/=]", "", payload.group(1))))
).getnames()
for want in ("worker.py", "models.py", "provision/common.py", "engines/llamacpp.py",
             "engines/stablediffusion.py"):
    check(f"the notebook carries {want}", any(n.endswith(want) for n in names))

launch = (ROOT / "scripts" / "launch-kaggle-worker.sh").read_text(encoding="utf-8")
for want in ("worker.py", "models.py", "worker/provision", "publish_model_dataset.py"):
    check(f"the Kaggle dataset ships {want}", want in launch)


print()
print(f"=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
