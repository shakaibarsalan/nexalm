"""Several models resident at once, loaded and unloaded on request.

The worker used to load one model, fixed by an environment variable at process
start, so changing model meant restarting the whole session -- minutes of wall
clock, and on Kaggle it spends a metered quota. This keeps a set of models in
VRAM instead, each behind its own llama-server, and lets the control plane move
models in and out while jobs keep running against the others.

Two properties matter more than anything else here:

* **Nothing is evicted while it is working.** A model with an in-flight job is
  never chosen for eviction, because doing so would kill a generation the user
  is watching stream.
* **The server is told what is true, not what was asked for.** Every change
  publishes the resident set, so a load that failed half way cannot leave the
  pool advertising a model it does not have.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import time

log = logging.getLogger("worker.models")

# Reserve for the CUDA context, activations and KV cache of each resident model.
# The same figure the provisioning code uses when it sizes a single model.
PER_MODEL_OVERHEAD_GB = 2.5


class ModelLoadError(RuntimeError):
    """A model could not be brought into VRAM, with a reason worth reporting."""


class Resident:
    """One model held in VRAM, with the engine serving it."""

    def __init__(self, model_id: str, path: str, engine, size_gb: float, port: int) -> None:
        self.model_id = model_id
        self.path = path
        self.engine = engine
        self.size_gb = size_gb
        self.port = port
        self.active_jobs: set[str] = set()
        self.last_used = time.monotonic()

    @property
    def busy(self) -> bool:
        return bool(self.active_jobs)


class ModelManager:
    """The set of models this worker currently holds."""

    def __init__(
        self,
        *,
        model_dir: str,
        total_vram_gb: float,
        context_length: int,
        n_gpu_layers: int = 999,
        base_port: int = 8090,
        hf_token: str | None = None,
        read_only_roots: list[str] | None = None,
        scratch_dir: str | None = None,
    ) -> None:
        self.model_dir = model_dir
        # Weights want to persist; a compiled binary wants to be fast to write
        # and is cheap to replace. On Colab those are different filesystems --
        # weights belong on Drive so a session does not re-download 10 GB, but
        # building through the Drive FUSE mount is slow and failure-prone.
        self.scratch_dir = scratch_dir or model_dir
        # Read-only mounts searched before downloading anything: a Kaggle
        # Dataset lands under /kaggle/input and can be mmapped in place, which
        # is the difference between a session that starts in a minute and one
        # that pulls eighteen gigabytes first.
        self.read_only_roots = read_only_roots or []
        self.total_vram_gb = total_vram_gb
        self.context_length = context_length
        self.n_gpu_layers = n_gpu_layers
        self.base_port = base_port
        self.hf_token = hf_token
        self.resident: dict[str, Resident] = {}
        self._lock = asyncio.Lock()
        os.makedirs(model_dir, exist_ok=True)

    # ── accounting ──────────────────────────────────────────
    @property
    def used_gb(self) -> float:
        return sum(r.size_gb + PER_MODEL_OVERHEAD_GB for r in self.resident.values())

    @property
    def free_gb(self) -> float:
        return max(0.0, self.total_vram_gb - self.used_gb)

    def state(self) -> dict:
        return {
            "loaded": sorted(self.resident.keys()),
            "freeVramGb": round(self.free_gb, 1),
        }

    def engine_for(self, model_id: str):
        entry = self.resident.get(model_id)
        if entry is None:
            return None
        entry.last_used = time.monotonic()
        return entry.engine

    def note_job(self, model_id: str, job_id: str, finished: bool = False) -> None:
        """Track in-flight work so a busy model is never evicted under it."""
        entry = self.resident.get(model_id)
        if entry is None:
            return
        if finished:
            entry.active_jobs.discard(job_id)
        else:
            entry.active_jobs.add(job_id)
        entry.last_used = time.monotonic()

    # ── weights on disk ─────────────────────────────────────
    def _local_gguf(self, model_id: str, quantization: str | None) -> str | None:
        """A already-downloaded GGUF for this model, if there is one.

        Files are kept under a per-model directory so two models cannot be
        confused for one another by a glob, which is how the Colab bootstrap
        once ended up loading the wrong weights.
        """
        found = sorted(glob.glob(os.path.join(self.model_dir, model_id, "*.gguf")))
        for root in self.read_only_roots:
            # Matched on the model id as a path segment so two models sharing a
            # prefix cannot be confused -- the mistake that once had Colab
            # loading an IQ3 build while announcing a Q4.
            found += sorted(glob.glob(os.path.join(root, "**", model_id, "*.gguf"), recursive=True))
            found += sorted(glob.glob(os.path.join(root, model_id, "*.gguf")))
        if not found:
            return None
        if quantization:
            preferred = [f for f in found if quantization in os.path.basename(f)]
            if preferred:
                return preferred[0]
        return found[0]

    def _local_mmproj(self, model_id: str) -> str | None:
        """The vision projector for this model, if one has been fetched."""
        for root in [self.model_dir, *self.read_only_roots]:
            hits = sorted(glob.glob(os.path.join(root, "**", model_id, "*mmproj*.gguf"), recursive=True))
            hits += sorted(glob.glob(os.path.join(root, model_id, "*mmproj*.gguf")))
            if hits:
                return hits[0]
        return None

    def _fetch_mmproj(self, model_id: str, repo: str) -> str | None:
        """Download the vision projector, which is small next to the weights."""
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from engines.model_catalog import list_gguf

        try:
            candidates = [f for f, _ in list_gguf(repo) if "mmproj" in f.lower()]
        except Exception as exc:  # noqa: BLE001 — no projector is not fatal
            log.warning("could not list %s for a projector: %s", repo, exc)
            return None
        if not candidates:
            log.warning("%s advertises vision but has no mmproj file", model_id)
            return None

        from huggingface_hub import hf_hub_download

        log.info("downloading vision projector %s", candidates[0])
        return hf_hub_download(
            repo_id=repo, filename=candidates[0],
            local_dir=os.path.join(self.model_dir, model_id), token=self.hf_token or None,
        )

    def _fetch_gguf(self, model_id: str, repo: str, quantization: str | None) -> str:
        """Download the best-fitting GGUF for this model, once."""
        import sys

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from engines.model_catalog import choose_gguf

        budget = self.total_vram_gb - PER_MODEL_OVERHEAD_GB
        filename, expected = choose_gguf(repo, budget * 1e9)
        log.info(
            "downloading %s from %s (%s, %.1f GB) — first use only",
            model_id, repo, filename, expected / 1e9,
        )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:  # noqa: PERF203 — installed lazily; workers are slim
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True
            )
            from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=os.path.join(self.model_dir, model_id),
            token=self.hf_token or None,
        )

    # ── lifecycle ───────────────────────────────────────────
    def _free_port(self) -> int:
        taken = {r.port for r in self.resident.values()}
        port = self.base_port
        while port in taken:
            port += 1
        return port

    async def _evict_for(self, need_gb: float) -> list[str]:
        """Unload least-recently-used idle models until `need_gb` fits."""
        evicted: list[str] = []
        while self.free_gb < need_gb:
            idle = [r for r in self.resident.values() if not r.busy]
            if not idle:
                raise ModelLoadError(
                    "Not enough VRAM, and every resident model is mid-generation. "
                    "Try again once they finish."
                )
            victim = min(idle, key=lambda r: r.last_used)
            log.info("evicting %s to make room (%.1f GB)", victim.model_id, victim.size_gb)
            await self._unload(victim.model_id)
            evicted.append(victim.model_id)
        return evicted

    async def load(
        self,
        model_id: str,
        *,
        repo: str | None = None,
        quantization: str | None = None,
        context_length: int | None = None,
        vision: bool = False,
        engine: str | None = None,
        family: str | None = None,
        kind: str = "chat",
    ) -> dict:
        """Bring a model into VRAM, evicting idle ones if it needs the room."""
        async with self._lock:
            if model_id in self.resident:
                return {"alreadyLoaded": True, **self.state()}

            # The catalogue has carried an `engine` column since the beginning
            # and nothing read it: every load built a LlamaCppEngine. That was
            # invisible while every model was a GGUF language model, and became
            # actively harmful the moment the catalogue listed a diffusion
            # model -- llama.cpp was handed FLUX weights, failed with
            # "unknown model architecture", and the eviction it had already
            # performed to make room stood, taking a working chat model with it.
            if (engine or "").lower() in ("stablediffusion", "sdcpp") or kind == "image":
                return await self._load_image(model_id, family=family)

            path = self._local_gguf(model_id, quantization)
            if path is None:
                if not repo:
                    raise ModelLoadError(
                        f"{model_id} is not on disk and no repository was given to fetch it from."
                    )
                path = self._fetch_gguf(model_id, repo, quantization)

            size_gb = os.path.getsize(path) / 1e9
            need = size_gb + PER_MODEL_OVERHEAD_GB
            if need > self.total_vram_gb:
                raise ModelLoadError(
                    f"{model_id} needs about {need:.1f} GB but this worker has "
                    f"{self.total_vram_gb:.1f} GB in total."
                )
            evicted = await self._evict_for(need)

            from engines.llamacpp import LlamaCppEngine

            mmproj = None
            if vision:
                mmproj = self._local_mmproj(model_id)
                if mmproj is None and repo:
                    mmproj = self._fetch_mmproj(model_id, repo)

            port = self._free_port()
            engine = LlamaCppEngine(
                path,
                context_length=context_length or self.context_length,
                port=port,
                n_gpu_layers=self.n_gpu_layers,
                mmproj_path=mmproj,
            )
            log.info("loading %s on port %d (%.1f GB)", model_id, port, size_gb)
            started = time.monotonic()
            try:
                await engine.start()
            except Exception as exc:
                # Leave no half-started server behind holding a port and VRAM.
                try:
                    await engine.stop()
                except Exception:  # noqa: BLE001
                    pass
                raise ModelLoadError(f"{model_id} failed to start: {exc}") from exc

            self.resident[model_id] = Resident(model_id, path, engine, size_gb, port)
            log.info("%s ready in %.1fs", model_id, time.monotonic() - started)
            return {"evicted": evicted, **self.state()}

    async def _load_image(self, model_id: str, *, family: str | None) -> dict:
        """Bring up a diffusion model behind its own sd-server.

        Everything that can fail -- resolving weights, finding or building the
        binary -- happens *before* anything is evicted. The first attempt at
        this evicted a working chat model and then failed, which is the worst
        possible order: the user lost what they had and gained nothing.

        Both steps run on a thread. Either can take tens of minutes (a 10 GB
        download, or a CUDA build), and blocking the event loop that long would
        stop the heartbeat and have the server sweep this worker out of the
        pool mid-load.
        """
        from engines.gpu import detect_gpus
        from engines.stablediffusion import (
            StableDiffusionEngine,
            ensure_sd_server,
            family_of,
            fetch_weights,
            weights_vram_gb,
        )

        fam = family_of(model_id, family)
        weights = await asyncio.to_thread(
            fetch_weights,
            model_id,
            fam,
            model_dir=self.model_dir,
            read_only_roots=self.read_only_roots,
            hf_token=self.hf_token,
        )

        gpus = detect_gpus()
        arch = "75"
        try:
            import sys

            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from provision.common import cuda_arch

            arch = cuda_arch([g.name for g in gpus])
        except Exception as exc:  # noqa: BLE001 — sm_75 covers both free tiers
            log.debug("could not detect the CUDA architecture (%s); assuming sm_75", exc)

        binary = await asyncio.to_thread(
            ensure_sd_server,
            scratch_dir=self.scratch_dir,
            read_only_roots=self.read_only_roots,
            arch=arch,
        )

        # Only the weights that stay on the card count: the text encoders are
        # deliberately kept on CPU, so charging them against VRAM would refuse
        # a model that fits.
        size_gb = weights_vram_gb(weights)
        need = size_gb + PER_MODEL_OVERHEAD_GB
        if need > self.total_vram_gb:
            raise ModelLoadError(
                f"{model_id} needs about {need:.1f} GB but this worker has "
                f"{self.total_vram_gb:.1f} GB in total."
            )
        evicted = await self._evict_for(need)

        port = self._free_port()
        engine = StableDiffusionEngine(weights, family=fam, port=port, server_binary=binary)
        log.info("loading %s on port %d (%.1f GB on the card)", model_id, port, size_gb)
        started = time.monotonic()
        try:
            await engine.start()
        except Exception as exc:
            try:
                await engine.stop()
            except Exception:  # noqa: BLE001
                pass
            raise ModelLoadError(f"{model_id} failed to start: {exc}") from exc

        primary = weights.get("diffusion") or weights.get("model") or ""
        self.resident[model_id] = Resident(model_id, primary, engine, size_gb, port)
        log.info("%s ready in %.1fs", model_id, time.monotonic() - started)
        return {"evicted": evicted, **self.state()}

    async def _unload(self, model_id: str) -> None:
        entry = self.resident.pop(model_id, None)
        if entry is None:
            return
        try:
            await entry.engine.stop()
        except Exception as exc:  # noqa: BLE001 — a stuck stop must not wedge the worker
            log.warning("error stopping %s: %s", model_id, exc)

    async def unload(self, model_id: str) -> dict:
        async with self._lock:
            entry = self.resident.get(model_id)
            if entry and entry.busy:
                raise ModelLoadError(
                    f"{model_id} is mid-generation; it will stay loaded until that finishes."
                )
            await self._unload(model_id)
            return self.state()

    async def stop_all(self) -> None:
        for model_id in list(self.resident):
            await self._unload(model_id)
