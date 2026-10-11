"""stable-diffusion.cpp engine — the one that answers with a picture.

llama.cpp cannot generate images at all, so this is a genuinely separate
runtime rather than another model behind the existing one. It shares the shape
of :class:`~engines.llamacpp.LlamaCppEngine` on purpose — a child process
serving localhost HTTP, drained logs, a readiness poll — because everything the
worker already knows how to do (evict, cancel, report) then applies unchanged.

Three things differ from llama.cpp and each one drove a design decision here:

**Generation is a job, not a stream.** ``sd-server`` accepts a request, returns
a job id, and is polled. There is no first token and no per-step counter in the
job payload, so progress is reported as elapsed time against the step count we
asked for. That is coarser than a token stream and is described as such rather
than dressed up as a percentage the server never gave us.

**A model is several files.** FLUX is a diffusion model plus two text encoders
plus a VAE, passed as four separate flags. The chat path assumes one GGUF, so
weights for this engine are resolved by :func:`fetch_weights` instead.

**It must never answer a chat turn.** :meth:`generate` raises rather than
returning text, so a mis-routed conversation fails loudly at the engine instead
of quietly handing a diffusion model a prompt it will answer with noise.
"""

from __future__ import annotations

import asyncio
import base64
import glob
import logging
import os
import shutil
import subprocess
import time
from typing import AsyncIterator, Iterable

import httpx

from .base import GenerationParams, InferenceEngine

log = logging.getLogger("worker.stablediffusion")

# Pinned rather than tracking master. The flag names here have visibly moved
# (the server example's own README already says --llm where FLUX still wants
# --t5xxl), and a rename landing upstream would otherwise turn into an
# unexplained startup failure on a machine we cannot attach a debugger to.
SD_COMMIT = "d8fb10c02977c8ca999f3fb4e02df9ecf10f7ba6"
SD_REPO = "https://github.com/leejet/stable-diffusion.cpp"

# Upstream publishes Linux binaries for CPU, ROCm and Vulkan, but not CUDA, so
# there is nothing to download for an NVIDIA card and this has to be built —
# the same situation as llama.cpp, handled the same way.
ARTIFACT_NAME = "sdcpp-cuda-sm75.tar.gz"


class SdServerMissing(RuntimeError):
    """No usable sd-server binary, with the ways of getting one already tried."""


# ── weights ─────────────────────────────────────────────────
#
# Every file comes from one ungated repository. The obvious sources are split
# across three (city96 for the GGUF, comfyanonymous for the encoders,
# black-forest-labs for the VAE) and the last of those is `gated: auto`, which
# an anonymous worker cannot download from — it fails after the first 6.7 GB,
# which is the worst possible place to discover it.
#
# The T5 encoder is taken quantised: Q4_K is 2.75 GB against 9.79 GB for fp16,
# on a text encoder whose output is a conditioning vector. That is the
# difference between a 10 GB download and a 17 GB one.
WEIGHT_SETS: dict[str, dict[str, tuple[str, str]]] = {
    "flux": {
        "diffusion": ("second-state/FLUX.1-schnell-GGUF", "flux1-schnell-Q4_0.gguf"),
        "t5xxl": ("second-state/FLUX.1-schnell-GGUF", "t5xxl-Q4_K.gguf"),
        "clip_l": ("second-state/FLUX.1-schnell-GGUF", "clip_l.safetensors"),
        "vae": ("second-state/FLUX.1-schnell-GGUF", "ae.safetensors"),
    },
    "sd": {
        "model": (
            "second-state/stable-diffusion-v1-5-GGUF",
            "stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf",
        ),
    },
}

# What each family wants asked of it by default. schnell is the four-step
# distilled model and is run at cfg 1.0 with euler, per upstream's own FLUX
# notes; SD 1.5 is an ordinary guided model and needs real CFG and more steps.
FAMILY_DEFAULTS: dict[str, dict] = {
    "flux": {"steps": 4, "cfg_scale": 1.0, "sample_method": "euler", "size": 768},
    "sd": {"steps": 20, "cfg_scale": 7.0, "sample_method": "euler_a", "size": 512},
}


def family_of(model_id: str, family: str | None) -> str:
    """Which weight set a catalogue row means."""
    if family and family in WEIGHT_SETS:
        return family
    lowered = model_id.lower()
    if "flux" in lowered:
        return "flux"
    return "sd"


def _find_local(root_dirs: Iterable[str], model_id: str, filename: str) -> str | None:
    """An already-present copy of one weight file, if any host has it."""
    for root in root_dirs:
        if not root or not os.path.isdir(root):
            continue
        hits = sorted(glob.glob(os.path.join(root, "**", model_id, filename), recursive=True))
        hits += sorted(glob.glob(os.path.join(root, model_id, filename)))
        # A Kaggle Dataset may flatten the directory away, so accept the file
        # on its own name too — these filenames are specific enough that a
        # collision between two models is not a real risk.
        hits += sorted(glob.glob(os.path.join(root, "**", filename), recursive=True))
        for hit in hits:
            if os.path.isfile(hit):
                return hit
    return None


def fetch_weights(
    model_id: str,
    family: str,
    *,
    model_dir: str,
    read_only_roots: Iterable[str] = (),
    hf_token: str | None = None,
) -> dict[str, str]:
    """Resolve every file this model needs, downloading only what is absent.

    Returns a mapping of role (``diffusion``/``vae``/``clip_l``/``t5xxl``, or
    ``model`` for a single-file family) to an absolute path.
    """
    spec = WEIGHT_SETS.get(family)
    if not spec:
        raise SdServerMissing(f"no weight set is defined for the {family!r} family")

    roots = [model_dir, *read_only_roots]
    resolved: dict[str, str] = {}
    missing: list[tuple[str, str, str]] = []

    for role, (repo, filename) in spec.items():
        found = _find_local(roots, model_id, filename)
        if found:
            resolved[role] = found
        else:
            missing.append((role, repo, filename))

    if missing:
        from huggingface_hub import hf_hub_download

        total = len(missing)
        for index, (role, repo, filename) in enumerate(missing, start=1):
            log.info("downloading %s (%d/%d) — %s", filename, index, total, role)
            resolved[role] = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=os.path.join(model_dir, model_id),
                token=hf_token or None,
            )

    on_disk = sum(os.path.getsize(p) for p in resolved.values()) / 1e9
    log.info("%s weights ready: %d file(s), %.1f GB", model_id, len(resolved), on_disk)
    return resolved


def weights_vram_gb(weights: dict[str, str]) -> float:
    """VRAM the resident model will actually hold.

    Not the same as bytes on disk: the text encoders run on CPU (``--clip-on-cpu``)
    precisely so they do not compete with the diffusion model for a 15 GB card,
    so counting them against VRAM would refuse a model that fits comfortably.
    """
    gpu_roles = ("diffusion", "model", "vae")
    return sum(os.path.getsize(p) for role, p in weights.items() if role in gpu_roles) / 1e9


# ── the binary ──────────────────────────────────────────────
def _http_base() -> str:
    """The control-plane's HTTP origin, derived from the socket URL we hold."""
    ws = os.environ.get("ORACLE_WS_URL", "")
    if not ws:
        return ""
    base = ws.split("://", 1)[-1].split("/", 1)[0]
    scheme = "https" if ws.startswith("wss") else "http"
    return f"{scheme}://{base}"


def _download_prebuilt(dest_dir: str) -> str | None:
    """Fetch a previously published CUDA build, if the server holds one."""
    base, token = _http_base(), os.environ.get("WORKER_TOKEN", "")
    if not base or not token:
        return None

    url = f"{base}/api/artifacts/{ARTIFACT_NAME}"
    tarball = os.path.join(dest_dir, ARTIFACT_NAME)
    os.makedirs(dest_dir, exist_ok=True)
    try:
        # Authorization header rather than ?token=, so the URL is safe to log:
        # a query token ends up in a saved notebook's output for anyone with
        # the link. The same mistake was already made once on this project.
        with httpx.stream(
            "GET", url, headers={"Authorization": f"Bearer {token}"}, timeout=600.0,
            follow_redirects=True,
        ) as res:
            if res.status_code != 200:
                log.info("no prebuilt sd-server on the server (HTTP %d)", res.status_code)
                return None
            with open(tarball, "wb") as handle:
                for chunk in res.iter_bytes(1 << 20):
                    handle.write(chunk)
    except Exception as exc:  # noqa: BLE001 — building is the fallback
        log.info("could not fetch a prebuilt sd-server: %s", exc)
        return None

    subprocess.run(["tar", "xzf", tarball, "-C", dest_dir], check=True)
    os.remove(tarball)
    binary = _first_binary([dest_dir])
    if binary and not _runs_here(binary):
        log.warning(
            "the published sd-server is built for %s but this host is %s -- ignoring it "
            "and building from source instead.",
            _elf_machine(binary), __import__("platform").machine(),
        )
        return None
    if binary:
        subprocess.run(f"chmod +x {os.path.dirname(binary)}/* 2>/dev/null || true",
                       shell=True, check=False)
        log.info("sd-server from the published build: %s", binary)
    return binary


# ELF e_machine values, read straight from the header. A binary for the wrong
# CPU fails with "cannot execute binary file", which says nothing about why --
# and the mistake is easy to make, because the machine that serves the artifact
# need not share an architecture with the machines that run it. This project's
# own control plane is an Ampere A1 (aarch64) while both GPU hosts are x86_64,
# and a build produced on the server ran nowhere.
_ELF_MACHINES = {0x3E: "x86_64", 0xB7: "aarch64"}


def _elf_machine(path: str) -> str | None:
    """The CPU architecture a binary was built for, or None if unreadable."""
    try:
        with open(path, "rb") as handle:
            header = handle.read(20)
    except OSError:
        return None
    if len(header) < 20 or header[:4] != b"ELF":
        return None
    return _ELF_MACHINES.get(header[18] | (header[19] << 8))


def _runs_here(path: str) -> bool:
    """Whether this binary matches the CPU we are on."""
    import platform

    built_for = _elf_machine(path)
    if built_for is None:
        return True  # Unreadable header: let the loader have its say.
    here = platform.machine()
    here = "x86_64" if here in ("x86_64", "AMD64") else here
    return built_for == here


def _first_binary(roots: Iterable[str]) -> str | None:
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        hits = sorted(glob.glob(os.path.join(root, "**", "sd-server"), recursive=True))
        for hit in hits:
            if os.path.isfile(hit):
                return hit
    return None


def _build(src: str, arch: str) -> str:
    """Compile sd-server with CUDA for this card. Twenty-odd minutes."""
    binary = os.path.join(src, "build", "bin", "sd-server")
    if os.path.exists(binary):
        log.info("sd-server already built at %s", binary)
        return binary

    log.info("building stable-diffusion.cpp for sm_%s — this takes ~25 minutes", arch)
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq cmake build-essential",
        shell=True, check=True,
    )

    if os.path.isdir(os.path.join(src, ".git")):
        log.info("reusing existing checkout")
    else:
        subprocess.run(f"rm -rf {src}", shell=True, check=True)
        # Submodules carry ggml; a plain clone builds against nothing.
        subprocess.run(
            f"git clone --recursive {SD_REPO} {src}", shell=True, check=True,
        )
        subprocess.run(
            f"cd {src} && git checkout {SD_COMMIT} && git submodule update --init --recursive",
            shell=True, check=True, executable="/bin/bash",
        )

    # Same CMake trap as llama.cpp: CUDA::cuda_driver only exists when an
    # unversioned libcuda.so is findable, and a GPU node ships libcuda.so.1
    # with an empty stubs directory.
    try:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from provision.common import ensure_libcuda_link

        ensure_libcuda_link()
    except Exception as exc:  # noqa: BLE001 — configure will report it properly
        log.debug("could not pre-create the libcuda link: %s", exc)

    cmd = (
        f"cmake -S {src} -B {src}/build "
        f"-DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} -DCMAKE_BUILD_TYPE=Release "
        f"'-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64;/usr/local/cuda/lib64/stubs' "
        f"&& cmake --build {src}/build --target sd-server -j$(nproc)"
    )
    log.info("$ %s", cmd)
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")

    if not os.path.exists(binary):
        raise SdServerMissing(f"the build finished but {binary} is missing")
    log.info("built %s in %s", binary, src)
    _publish_build(binary)
    return binary


def _publish_build(binary: str) -> None:
    """Send this build back to the server so the next session skips the compile.

    Best effort by design: a worker that cannot publish has still built itself a
    working binary, and failing the load over an optimisation would be absurd.

    The control plane cannot produce these itself -- it is an Ampere A1
    (aarch64) while every GPU host is x86_64 -- so a worker is the only machine
    that can, and this is the only way the artifact ever gets there.
    """
    base, token = _http_base(), os.environ.get("WORKER_TOKEN", "")
    if not base or not token:
        return

    bin_dir = os.path.dirname(binary)
    tarball = os.path.join(bin_dir, ARTIFACT_NAME)
    try:
        subprocess.run(
            ["tar", "czf", tarball, "-C", bin_dir, os.path.basename(binary)], check=True
        )
        size_mb = os.path.getsize(tarball) / 1e6
        log.info("publishing the build (%.0f MB) so other sessions skip it", size_mb)
        with open(tarball, "rb") as handle:
            res = httpx.post(
                f"{base}/api/artifacts/{ARTIFACT_NAME}",
                content=handle.read(),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/octet-stream",
                },
                timeout=900.0,
            )
        if res.status_code == 200:
            log.info("published %s", ARTIFACT_NAME)
        else:
            log.warning("could not publish the build (HTTP %d)", res.status_code)
    except Exception as exc:  # noqa: BLE001 — never fail a load over this
        log.warning("could not publish the build: %s", exc)
    finally:
        if os.path.exists(tarball):
            os.remove(tarball)


def ensure_sd_server(
    *, scratch_dir: str, read_only_roots: Iterable[str] = (), arch: str = "75"
) -> str:
    """Find an sd-server binary, or produce one. Cheapest route first."""
    explicit = os.environ.get("SD_SERVER_BIN", "")
    if explicit and os.path.exists(explicit):
        return explicit
    if explicit and shutil.which(explicit):
        return explicit

    attached = _first_binary(read_only_roots)
    if attached and not _runs_here(attached):
        log.warning(
            "the attached sd-server is built for %s, not this host -- ignoring it.",
            _elf_machine(attached),
        )
        attached = None
    if attached:
        # A read-only mount cannot be made executable, and the binary links
        # against the ggml shared objects beside it, so the directory is copied
        # out whole — the same reason llama.cpp's bootstrap copies rather than
        # runs in place.
        dest = os.path.join(scratch_dir, "sd-bin")
        subprocess.run(
            f"rm -rf {dest} && cp -r {os.path.dirname(attached)} {dest}", shell=True, check=True
        )
        subprocess.run(f"chmod +x {dest}/* 2>/dev/null || true", shell=True, check=False)
        binary = os.path.join(dest, "sd-server")
        log.info("sd-server from an attached dataset: %s", binary)
        return binary

    downloaded = _download_prebuilt(os.path.join(scratch_dir, "sd-bin"))
    if downloaded:
        return downloaded

    return _build(os.path.join(scratch_dir, "stable-diffusion.cpp"), arch)


class StableDiffusionEngine(InferenceEngine):
    name = "stablediffusion"

    def __init__(
        self,
        weights: dict[str, str],
        *,
        family: str = "sd",
        host: str = "127.0.0.1",
        port: int = 8100,
        server_binary: str | None = None,
        clip_on_cpu: bool = True,
        startup_timeout_s: int = 900,
        extra_args: list[str] | None = None,
    ) -> None:
        self.weights = weights
        self.family = family
        self.host = host
        self.port = port
        self.server_binary = server_binary or os.environ.get("SD_SERVER_BIN", "sd-server")
        # FLUX's T5 encoder is 2.75 GB that only runs once per prompt. Keeping
        # it off the card leaves the diffusion model the room it needs on a T4,
        # at a cost of a second or two per image.
        self.clip_on_cpu = clip_on_cpu
        self.startup_timeout_s = startup_timeout_s
        self.extra_args = extra_args or []

        self._proc: subprocess.Popen | None = None
        self._ready = False
        self._client: httpx.AsyncClient | None = None
        self._cancelled: set[str] = set()
        self._remote_jobs: dict[str, str] = {}

    @property
    def max_context(self) -> int:
        # Not a token window; diffusion prompts are conditioning, not history.
        # Reported so callers that ask do not divide by zero.
        return 512

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def defaults(self) -> dict:
        return FAMILY_DEFAULTS.get(self.family, FAMILY_DEFAULTS["sd"])

    # ── lifecycle ───────────────────────────────────────────
    def _placement(self) -> list[str]:
        """Where each part of the model runs.

        The text encoders are the interesting decision. They are 3.2 GB that
        run once per prompt, and on one card there is no room for them beside
        the diffusion model -- hence ``--clip-on-cpu``, which puts them on the
        two vCPUs a hosted runtime gives you.

        With a second card they belong on it. This is not splitting one model
        across devices: the encoders are a separate module that hands over a
        small conditioning tensor, so it does not depend on a fast interconnect
        the way a split would. Upstream is explicit that row-splitting a single
        module is "usually the faster option when the devices have fast
        interconnect" -- and a pair of T4s is PCIe with no NVLink, which is not
        that case. So the idle card is given whole work rather than half a
        matmul.

        SD_BACKEND overrides all of it, so an alternative placement can be
        measured without changing code.
        """
        override = os.environ.get("SD_BACKEND", "").strip()
        if override:
            return ["--backend", override]

        try:
            from .gpu import detect_gpus

            gpus = detect_gpus()
        except Exception:  # noqa: BLE001 — fall back to the single-card layout
            gpus = []

        if len(gpus) >= 2:
            return ["--backend", "te=cuda1,diffusion=cuda0,vae=cuda0"]
        return ["--clip-on-cpu"] if self.clip_on_cpu else []

    def _build_command(self) -> list[str]:
        cmd = [
            self.server_binary,
            "--listen-ip", self.host,
            "--listen-port", str(self.port),
        ]
        if "model" in self.weights:
            cmd += ["-m", self.weights["model"]]
        else:
            cmd += ["--diffusion-model", self.weights["diffusion"]]
            for flag, role in (("--vae", "vae"), ("--clip_l", "clip_l"), ("--t5xxl", "t5xxl")):
                if role in self.weights:
                    cmd += [flag, self.weights[role]]
        cmd += self._placement()
        return cmd + self.extra_args

    def _check_flags(self, cmd: list[str]) -> None:
        """Refuse to launch with a flag this build does not have.

        Upstream tags roll forward daily and flags have already been renamed
        once. Without this, a rename shows up as an immediate non-zero exit
        with the reason buried in output nobody is watching; with it, the error
        names the flag.
        """
        try:
            help_text = subprocess.run(
                [self.server_binary, "-h"], capture_output=True, text=True, timeout=60
            )
        except Exception as exc:  # noqa: BLE001 — a missing help screen is not fatal
            log.debug("could not read sd-server --help: %s", exc)
            return

        text = (help_text.stdout or "") + (help_text.stderr or "")
        if len(text) < 40:
            return
        unknown = [a for a in cmd[1:] if a.startswith("-") and a not in text]
        if unknown:
            raise SdServerMissing(
                "this sd-server build does not accept " + ", ".join(unknown)
                + f". It was pinned at {SD_COMMIT[:7]}; a newer build may have renamed them."
            )

    async def start(self) -> None:
        for role, path in self.weights.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{role} weights not found: {path}")
        if shutil.which(self.server_binary) is None and not os.path.exists(self.server_binary):
            raise SdServerMissing(
                f"sd-server binary not found: {self.server_binary}. "
                "Set SD_SERVER_BIN, attach a build as a Dataset, or let the worker build it."
            )

        cmd = self._build_command()
        self._check_flags(cmd)
        log.info("starting sd-server: %s", " ".join(cmd))
        self._ready = False
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        asyncio.create_task(self._drain_logs())

        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(None))
        await self._await_ready()
        log.info("sd-server ready at %s", self.base_url)

    async def _drain_logs(self) -> None:
        proc = self._proc
        if not proc or not proc.stdout:
            return
        loop = asyncio.get_running_loop()
        fd = proc.stdout.fileno()
        pending = b""
        while True:
            block = await loop.run_in_executor(None, lambda: os.read(fd, 65536))
            if not block:
                break
            pending += block
            *lines, pending = pending.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
            level = logging.DEBUG if self._ready else logging.INFO
            for raw in lines:
                text = raw.decode(errors="replace").rstrip()
                if text:
                    log.log(level, "[sd-server] %s", text)

    async def _await_ready(self) -> None:
        assert self._client is not None
        deadline = time.monotonic() + self.startup_timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise SdServerMissing(
                    f"sd-server exited during startup (code {self._proc.returncode})"
                )
            try:
                res = await self._client.get("/sdcpp/v1/capabilities", timeout=10)
                if res.status_code == 200:
                    self._ready = True
                    return
            except Exception as exc:  # noqa: BLE001 — retry until the deadline
                last_err = exc
            await asyncio.sleep(3)
        raise TimeoutError(f"sd-server not ready within {self.startup_timeout_s}s: {last_err}")

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # ── generation ──────────────────────────────────────────
    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    async def generate(
        self,
        job_id: str,
        messages: list[dict],
        params: GenerationParams,
        system_prompt: str | None = None,
    ) -> AsyncIterator[tuple[str, str, None]]:
        """Refuse a chat turn outright.

        Routing picks a model by family prefix when a conversation names none,
        and an image model sitting in the same resident set is exactly the kind
        of thing that path would hand a conversation to. Failing here makes
        that a visible error instead of a diffusion model being asked to hold a
        conversation.
        """
        raise RuntimeError(
            "This is an image model and cannot answer a chat turn. "
            "Pick a text model for this conversation."
        )
        yield  # pragma: no cover — keeps this an async generator

    def _request_body(self, prompt: str, params: dict) -> dict:
        d = self.defaults
        steps = int(params.get("steps") or d["steps"])
        size = int(params.get("width") or params.get("size") or d["size"])
        height = int(params.get("height") or size)
        return {
            "prompt": prompt,
            "negative_prompt": params.get("negativePrompt") or "",
            "width": size,
            "height": height,
            "seed": int(params.get("seed", -1)),
            "batch_count": 1,
            "output_format": "png",
            "sample_params": {
                "sample_method": params.get("sampleMethod") or d["sample_method"],
                "sample_steps": steps,
                "guidance": {
                    "txt_cfg": float(params.get("cfgScale") or d["cfg_scale"]),
                },
            },
        }

    async def generate_image(
        self, job_id: str, prompt: str, params: dict | None = None
    ) -> AsyncIterator[dict]:
        """Submit one image job and follow it to completion.

        Yields ``{"type": "progress", ...}`` while it runs and one
        ``{"type": "image", ...}`` at the end. Progress carries elapsed time and
        the server's queue position, which is everything sd-server actually
        reports — it exposes no per-step counter, so a percentage here would be
        invented.
        """
        assert self._client is not None, "engine not started"
        body = self._request_body(prompt, params or {})
        started = time.monotonic()

        submit = await self._client.post("/sdcpp/v1/img_gen", json=body, timeout=60)
        if submit.status_code not in (200, 202):
            raise RuntimeError(f"sd-server refused the request ({submit.status_code}): {submit.text[:300]}")
        job = submit.json()
        remote_id = job.get("id")
        if not remote_id:
            raise RuntimeError("sd-server accepted the request but returned no job id")
        self._remote_jobs[job_id] = remote_id

        steps = body["sample_params"]["sample_steps"]
        yield {
            "type": "progress", "status": "queued", "elapsedMs": 0,
            "steps": steps, "queuePosition": job.get("queue_position"),
        }

        try:
            while True:
                if job_id in self._cancelled:
                    await self._cancel_remote(remote_id)
                    raise asyncio.CancelledError()

                await asyncio.sleep(1.0)
                poll = await self._client.get(f"/sdcpp/v1/jobs/{remote_id}", timeout=30)
                if poll.status_code == 404:
                    raise RuntimeError("sd-server lost the job")
                state = poll.json()
                status = state.get("status")
                elapsed = int((time.monotonic() - started) * 1000)

                # Anything that is not a terminal state is progress. Listing the
                # in-flight names instead was wrong the first time it ran: the
                # server says 'generating', which is not a name I guessed, and a
                # generation that was working fine died on the vocabulary. The
                # three endings below are the ones the API documents, so those
                # are what gets enumerated -- a status nobody predicted now
                # means "still going", which is the safe reading.
                if status not in ("completed", "failed", "cancelled"):
                    yield {
                        "type": "progress", "status": status, "elapsedMs": elapsed,
                        "steps": steps, "queuePosition": state.get("queue_position"),
                    }
                    continue

                if status == "cancelled":
                    raise asyncio.CancelledError()
                if status == "failed":
                    err = (state.get("error") or {}).get("message") or "generation failed"
                    raise RuntimeError(err)
                if status == "completed":
                    images = ((state.get("result") or {}).get("images") or [])
                    if not images:
                        raise RuntimeError("sd-server reported success but returned no image")
                    b64 = images[0].get("b64_json") or ""
                    raw = base64.b64decode(b64)
                    yield {
                        "type": "image",
                        "b64": b64,
                        "bytes": len(raw),
                        "mime": "image/png",
                        "width": body["width"],
                        "height": body["height"],
                        "steps": steps,
                        "seed": state.get("seed", body["seed"]),
                        "elapsedMs": elapsed,
                    }
                    return
                raise RuntimeError(f"sd-server returned an unexpected status: {status!r}")  # noqa: E501
        finally:
            self._remote_jobs.pop(job_id, None)
            self._cancelled.discard(job_id)

    async def _cancel_remote(self, remote_id: str) -> None:
        if not self._client:
            return
        try:
            await self._client.post(f"/sdcpp/v1/jobs/{remote_id}/cancel", timeout=20)
        except Exception as exc:  # noqa: BLE001 — the job ends either way
            log.debug("could not cancel %s upstream: %s", remote_id, exc)
