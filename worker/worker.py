#!/usr/bin/env python3
"""Local LLM inference worker.

Dials OUT to the Oracle backend over WebSocket and serves inference jobs.
The outbound direction is not an implementation detail: Kaggle and Colab
runtimes sit behind NAT and can never be reached inbound, so the worker must
initiate and hold the connection.

Configuration is environment-only — no token ever reaches source control:

    ORACLE_WS_URL   wss://chat.example.com/worker/ws
    WORKER_TOKEN    shared bearer token (must match the server)
    WORKER_NAME     human label, e.g. kaggle-t4x2
    WORKER_ID       stable id across reconnects (default: derived from name)
    ENGINE          llamacpp | vllm | mock
    MODEL_ID        model id advertised to the backend
    MODEL_PATH      path to the .gguf (llamacpp only)
    CONTEXT_LENGTH  context window (default 16384)
    IDLE_SHUTDOWN_MINUTES  exit after this long with no work (0 disables)

The idle timer matters on metered runtimes: Kaggle bills wall-clock against a
30 h/week quota whether or not the GPU is doing anything, so a worker left
running overnight silently burns the week's budget.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
import uuid

import websockets
from websockets.exceptions import ConnectionClosed

from engines.base import GenerationParams, InferenceEngine
from engines.gpu import describe, detect_gpus, free_vram_mb, total_vram_mb

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("worker")


def _default_cache_dir() -> str:
    """Where downloaded weights live when nothing says otherwise.

    Kaggle's /kaggle/working is the *output* directory: it is size-capped and
    uploaded when the run ends, so a model dropped there nearly fills the quota
    and adds a pointless upload. Scratch is neither.
    """
    if os.path.isdir("/kaggle/temp"):
        return "/kaggle/temp/nexalm-models"
    if os.path.isdir("/content"):
        return "/content/nexalm-models"
    return "/tmp/nexalm-models"


def _infer_platform() -> str:
    """Which host this worker is running on.

    Read from the filesystem rather than a name, because the directories are a
    fact about the machine while a WORKER_NAME is only a convention someone
    might change. WORKER_PLATFORM overrides it for anywhere else.
    """
    if os.path.isdir("/kaggle/input") or os.path.isdir("/kaggle/working"):
        return "kaggle"
    if os.path.isdir("/content/drive") or os.path.isdir("/content"):
        return "colab"
    return "other"

VERSION = "1.0.0"


def _has_image_engine() -> bool:
    """Whether this build carries the diffusion engine at all."""
    try:
        import importlib.util

        return importlib.util.find_spec("engines.stablediffusion") is not None
    except Exception:  # noqa: BLE001 — absence is the answer
        return False


def env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        log.error("missing required environment variable: %s", name)
        sys.exit(2)
    return value or ""


class Worker:
    def __init__(self) -> None:
        self.ws_url = env("ORACLE_WS_URL", required=True)
        self.token = env("WORKER_TOKEN", required=True)
        self.name = env("WORKER_NAME", f"worker-{socket.gethostname()}")
        self.worker_id = env("WORKER_ID", "") or f"{self.name}-{uuid.uuid4().hex[:8]}"
        self.model_id = env("MODEL_ID", "qwen3-30b-a3b")
        self.context_length = int(env("CONTEXT_LENGTH", "16384"))
        self.idle_shutdown_s = float(env("IDLE_SHUTDOWN_MINUTES", "30")) * 60
        # Defaults on, so a worker running an older bootstrap behaves exactly
        # as it did. The Kaggle and Colab bootstraps turn it off.
        self.preload = env("PRELOAD_MODEL", "1").strip().lower() not in ("0", "false", "no")
        self.last_activity = time.monotonic()

        self.engine: InferenceEngine | None = None
        # The set of models held in VRAM. `engine` above remains the one loaded
        # at startup, so a deployment where the control plane never sends a load
        # frame behaves exactly as it did before.
        self.models: "ModelManager | None" = None
        self.platform = env("WORKER_PLATFORM", "") or _infer_platform()
        self.gpus = detect_gpus()
        self.heartbeat_interval = 10.0
        self.running = True
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._tasks: dict[str, asyncio.Task] = {}
        self._shutdown = asyncio.Event()

    # ── engine ──────────────────────────────────────────────
    def build_engine(self) -> InferenceEngine:
        kind = env("ENGINE", "llamacpp").lower()

        if kind == "mock":
            from engines.mock import MockEngine

            log.warning("using the MOCK engine — responses are canned, not inference")
            return MockEngine(context_length=self.context_length)

        if kind == "vllm":
            from engines.vllm import VllmEngine

            return VllmEngine(
                env("MODEL_REPO", required=True),
                context_length=self.context_length,
                quantization=env("QUANTIZATION", "awq") or None,
                extra_args=env("EXTRA_ARGS", "").split() if env("EXTRA_ARGS", "") else [],
            )

        if kind == "llamacpp":
            from engines.llamacpp import LlamaCppEngine

            model_path = env("MODEL_PATH", required=True)
            extra = env("EXTRA_ARGS", "").split() if env("EXTRA_ARGS", "") else []
            return LlamaCppEngine(
                model_path,
                context_length=self.context_length,
                n_gpu_layers=int(env("N_GPU_LAYERS", "999")),
                extra_args=extra,
            )

        log.error("unknown ENGINE=%s (expected llamacpp, vllm or mock)", kind)
        sys.exit(2)

    async def start_engine(self) -> None:
        """Come up ready to work, loading a model only if asked to.

        Preloading used to be unconditional, and it cost a session minutes
        before it could do anything -- then cost them again, because the model
        it chose was whatever the notebook happened to default to. Asking for a
        picture meant loading an 18.6 GB language model first and immediately
        evicting it. A worker with nothing resident is useful the moment it
        connects, and the control plane already loads on demand.
        """
        from models import ModelManager, Resident

        log.info("GPUs: %s", describe(self.gpus))
        total_vram_gb = total_vram_mb(self.gpus) / 1024
        self.models = ModelManager(
            model_dir=env("MODEL_CACHE_DIR", _default_cache_dir()),
            read_only_roots=[r for r in ("/kaggle/input",) if os.path.isdir(r)],
            total_vram_gb=total_vram_gb,
            context_length=self.context_length,
            n_gpu_layers=int(env("N_GPU_LAYERS", "999")),
            base_port=int(env("LLAMA_BASE_PORT", "8090")) + 1,
            hf_token=env("HF_TOKEN", ""),
            scratch_dir=env("WORKER_SCRATCH_DIR", "") or _default_cache_dir(),
        )
        model_path = env("MODEL_PATH", "")
        if not self.preload:
            log.info(
                "no model preloaded — %.1f GB of VRAM free, loading on request",
                total_vram_gb,
            )
            return

        self.engine = self.build_engine()
        log.info("loading model %s via %s…", self.model_id, self.engine.name)
        started = time.monotonic()
        await self.engine.start()
        log.info("model ready in %.1fs", time.monotonic() - started)

        # Adopt the startup model into the resident set so it is managed like
        # any other: evictable, countable against VRAM, and reported to the
        # server alongside anything loaded later.
        size_gb = os.path.getsize(model_path) / 1e9 if model_path and os.path.exists(model_path) else 0.0
        self.models.resident[self.model_id] = Resident(
            self.model_id, model_path, self.engine, size_gb, int(env("LLAMA_PORT", "8090"))
        )
        log.info(
            "%s resident, %.1f GB of %.1f GB VRAM free for more models",
            self.model_id, self.models.free_gb, total_vram_gb,
        )

    # ── protocol ────────────────────────────────────────────
    async def send(self, frame_type: str, payload: dict | None = None) -> None:
        if not self._ws:
            return
        try:
            await self._ws.send(json.dumps({"type": frame_type, "payload": payload or {}}))
        except (ConnectionClosed, OSError):
            log.warning("send failed — connection is down")

    async def register(self) -> None:
        await self.send(
            "register",
            {
                "workerId": self.worker_id,
                "name": self.name,
                # The configured runtime, not "unknown": with preloading off
                # there is no engine object yet, but the worker is perfectly
                # capable of building one the moment a load arrives.
                "engine": self.engine.name if self.engine else env("ENGINE", "llamacpp"),
                "model": self.model_id,
                "gpus": [g.as_dict() for g in self.gpus],
                "vramTotalMb": total_vram_mb(self.gpus),
                "maxContext": self.engine.max_context if self.engine else self.context_length,
                "version": VERSION,
                "capabilities": {
                    "streaming": True,
                    "cancel": True,
                    "modelSwitching": True,
                    # Declared rather than assumed. A session started before
                    # this engine shipped is still connected and still serving
                    # chat perfectly well, and the server has no other way to
                    # tell -- handing it an image model would put diffusion
                    # weights into llama.cpp, which fails *after* evicting a
                    # working model to make room.
                    "imageGeneration": _has_image_engine(),
                },
                "platform": self.platform,
                "loaded": sorted(self.models.resident) if self.models else [self.model_id],
                "freeVramGb": round(self.models.free_gb, 1) if self.models else None,
            },
        )

    async def heartbeat_loop(self) -> None:
        while self.running and self._ws:
            await asyncio.sleep(self.heartbeat_interval)
            # Re-read the cards each beat so the UI charts real utilisation and
            # memory pressure over time, not a snapshot from registration.
            self.gpus = detect_gpus() or self.gpus
            if self._tasks:
                self.last_activity = time.monotonic()

            await self.send(
                "heartbeat",
                {
                    "status": "WORKER_BUSY" if self._tasks else "WORKER_READY",
                    "vramFreeMb": free_vram_mb(self.gpus),
                    "activeJobs": len(self._tasks),
                    "gpus": [g.as_dict() for g in self.gpus],
                    "idleSeconds": int(time.monotonic() - self.last_activity),
                },
            )

            idle = time.monotonic() - self.last_activity
            if self.idle_shutdown_s > 0 and not self._tasks and idle > self.idle_shutdown_s:
                log.info(
                    "idle for %.0f min (limit %.0f) — shutting down to stop burning quota",
                    idle / 60,
                    self.idle_shutdown_s / 60,
                )
                await self.shutdown("idle timeout")
                return

    async def publish_models(self) -> None:
        """Tell the server what is actually resident, whatever just happened."""
        if self.models:
            await self.send("model.state", self.models.state())

    async def handle_load(self, payload: dict) -> None:
        model_id = payload.get("modelId", "")
        if not self.models or not model_id:
            return
        try:
            result = await self.models.load(
                model_id,
                repo=payload.get("repo"),
                quantization=payload.get("quantization"),
                context_length=payload.get("contextLength"),
                vision=bool(payload.get("vision")),
                engine=payload.get("engine"),
                family=payload.get("family"),
                kind=payload.get("kind") or "chat",
            )
            if result.get("evicted"):
                log.info("evicted %s to make room for %s", result["evicted"], model_id)
            log.info("loaded %s; resident: %s", model_id, ", ".join(result["loaded"]))
        except Exception as exc:  # noqa: BLE001 — every failure is reportable
            log.error("could not load %s: %s", model_id, exc)
            await self.send("model.error", {"modelId": model_id, "message": str(exc)})
        finally:
            # Always publish, including after a failure: the server marks the
            # worker busy when it sends the load, and only this clears it.
            await self.publish_models()

    async def handle_unload(self, payload: dict) -> None:
        model_id = payload.get("modelId", "")
        if not self.models or not model_id:
            return
        try:
            await self.models.unload(model_id)
            log.info("unloaded %s", model_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not unload %s: %s", model_id, exc)
            await self.send("model.error", {"modelId": model_id, "message": str(exc)})
        finally:
            await self.publish_models()

    @staticmethod
    def image_prompt(payload: dict) -> str:
        """The text to draw, which is the last thing the user actually asked for.

        Earlier turns are deliberately ignored: a diffusion model conditions on
        one prompt rather than a conversation, so folding history in would make
        every image in a thread a blend of everything said before it.
        """
        for message in reversed(payload.get("messages") or []):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            # A vision-shaped message is a list of parts; take the text ones.
            if isinstance(content, list):
                parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                joined = " ".join(p for p in parts if p).strip()
                if joined:
                    return joined
        return ""

    def http_base(self) -> str:
        """The control plane's HTTP origin, derived from the socket URL."""
        base = self.ws_url.split("://", 1)[-1].split("/", 1)[0]
        return ("https" if self.ws_url.startswith("wss") else "http") + "://" + base

    async def upload_image(self, job_id: str, model_id: str, image: dict) -> str:
        """Hand the finished PNG to the server over HTTP and get back its id.

        Not over the job socket: a PNG is one to two megabytes against an 8 MB
        frame ceiling, and every other frame on that link is small text.
        """
        import httpx

        body = {
            "jobId": job_id,
            "model": model_id,
            "b64": image["b64"],
            "mime": image.get("mime", "image/png"),
            "width": image.get("width"),
            "height": image.get("height"),
            "seed": image.get("seed"),
            "steps": image.get("steps"),
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(
                self.http_base() + "/api/images",
                json=body,
                # A header rather than ?token=: a query token is echoed into
                # logs and into a saved notebook's output.
                headers={"Authorization": "Bearer " + self.token},
            )
        if res.status_code != 200:
            raise RuntimeError(
                "the server refused the image (HTTP %d): %s" % (res.status_code, res.text[:200])
            )
        return res.json()["id"]

    async def run_image_job(self, job_id: str, model_id: str, engine, payload: dict) -> None:
        """Draw one picture and report it, in place of a token stream."""
        prompt = self.image_prompt(payload)
        if not prompt:
            raise RuntimeError("there is nothing to draw -- the prompt was empty.")

        image: dict | None = None
        async for frame in engine.generate_image(job_id, prompt, payload.get("params") or {}):
            if frame.get("type") == "image":
                image = frame
                continue
            await self.send(
                "job.progress",
                {
                    "jobId": job_id,
                    "status": frame.get("status", "running"),
                    "elapsedMs": frame.get("elapsedMs", 0),
                    "steps": frame.get("steps"),
                    "queuePosition": frame.get("queuePosition"),
                },
            )

        if image is None:
            raise RuntimeError("the image engine finished without producing an image")

        image_id = await self.upload_image(job_id, model_id, image)
        await self.send(
            "job.done",
            {
                "jobId": job_id,
                "genMs": image.get("elapsedMs"),
                "stopReason": "complete",
                "imageId": image_id,
            },
        )
        log.info(
            "job %s drew a %sx%s image in %.1fs",
            job_id[:8], image.get("width"), image.get("height"),
            (image.get("elapsedMs") or 0) / 1000,
        )

    async def run_job(self, payload: dict) -> None:
        job_id = payload["jobId"]
        params = payload.get("params", {})
        gen = GenerationParams(
            temperature=params.get("temperature", 0.7),
            top_p=params.get("topP", 0.9),
            top_k=params.get("topK"),
            max_tokens=params.get("maxTokens", 2048),
            repeat_penalty=params.get("repeatPenalty"),
            stop=params.get("stop") or [],
        )

        await self.send("job.accepted", {"jobId": job_id})

        # Route to the engine holding the model this job asked for. The server
        # only dispatches to a worker that reports the model resident, so a miss
        # here means the two views drifted -- worth saying out loud rather than
        # quietly answering from whatever happens to be loaded.
        wanted = payload.get("model") or self.model_id
        engine = self.models.engine_for(wanted) if self.models else self.engine
        if engine is None:
            await self.send(
                "job.error",
                {"jobId": job_id, "message": f"{wanted} is not loaded on this worker."},
            )
            return
        if self.models:
            self.models.note_job(wanted, job_id)

        started = time.monotonic()
        first_sent = False
        # Coalesce tokens into small time slices: one WS frame per token would
        # swamp the link without making the UI look any smoother.
        buffer: list[str] = []
        buffer_kind = "content"
        last_flush = time.monotonic()

        async def flush() -> None:
            nonlocal buffer, last_flush
            if buffer:
                frame = "job.reasoning" if buffer_kind == "reasoning" else "job.delta"
                await self.send(frame, {"jobId": job_id, "text": "".join(buffer)})
                buffer = []
                last_flush = time.monotonic()

        try:
            # A diffusion model produces one picture, not a token stream, and
            # its engine refuses generate() outright rather than answering a
            # chat turn with noise. `return` here still runs the finally block
            # below, which is what releases the model for eviction.
            if getattr(engine, "name", "") == "stablediffusion":
                await self.run_image_job(job_id, wanted, engine, payload)
                return

            async for kind, text, result in engine.generate(
                job_id,
                payload.get("messages", []),
                gen,
                payload.get("systemPrompt"),
            ):
                if result is not None:
                    await flush()
                    await self.send(
                        "job.done",
                        {
                            "jobId": job_id,
                            "promptTokens": result.prompt_tokens,
                            "completionTokens": result.completion_tokens,
                            "genMs": result.gen_ms,
                            "tokensPerSec": result.tokens_per_sec,
                            "stopReason": result.stop_reason,
                        },
                    )
                    log.info(
                        "job %s done: %s tokens, %s tok/s",
                        job_id[:8],
                        result.completion_tokens,
                        result.tokens_per_sec,
                    )
                    break

                if not first_sent:
                    first_sent = True
                    # First token of ANY kind. A thinking model can spend
                    # hundreds of tokens reasoning before its first content
                    # token, so timing from content alone would report a
                    # latency the user never actually experienced.
                    await self.send(
                        "job.first_token",
                        {"jobId": job_id, "ttftMs": int((time.monotonic() - started) * 1000)},
                    )

                # Never mix reasoning and answer text in one frame.
                if kind != buffer_kind:
                    await flush()
                    buffer_kind = kind

                buffer.append(text)
                if time.monotonic() - last_flush >= 0.05 or len("".join(buffer)) > 256:
                    await flush()

        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — any engine failure must reach the user
            log.exception("job %s failed", job_id[:8])
            await self.send("job.error", {"jobId": job_id, "message": str(exc), "retryable": True})
        finally:
            self._tasks.pop(job_id, None)
            self.last_activity = time.monotonic()
            # Release the model so it becomes evictable again. Without this a
            # model that once served a job could never be unloaded to make room.
            if self.models:
                self.models.note_job(wanted, job_id, finished=True)

    async def handle(self, raw: str) -> None:
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("dropped malformed frame")
            return

        kind = frame.get("type")
        payload = frame.get("payload") or {}

        if kind == "registered":
            self.heartbeat_interval = payload.get("heartbeatIntervalMs", 10000) / 1000
            # The server owns this: it protects a metered quota, and changing it
            # should not mean editing a notebook and re-running it. The env var
            # remains the floor for a host with no server-side opinion.
            server_idle = payload.get("idleShutdownMinutes")
            if server_idle is not None:
                self.idle_shutdown_s = float(server_idle) * 60
            log.info(
                "registered with Oracle (heartbeat every %.0fs, idle shutdown %s)",
                self.heartbeat_interval,
                "off" if self.idle_shutdown_s <= 0 else "%.0f min" % (self.idle_shutdown_s / 60),
            )

        elif kind == "job.assign":
            job_id = payload["jobId"]
            log.info("job %s assigned", job_id[:8])
            self.last_activity = time.monotonic()
            self._tasks[job_id] = asyncio.create_task(self.run_job(payload))

        elif kind == "model.load":
            # Loading can take minutes, so it runs as its own task: blocking the
            # frame loop would stop heartbeats and the server would sweep this
            # worker out of the pool mid-download.
            asyncio.create_task(self.handle_load(payload))

        elif kind == "model.unload":
            asyncio.create_task(self.handle_unload(payload))

        elif kind == "shutdown":
            # Operator pressed Stop in the UI. Ending the process ends the
            # hosted session, which is what actually stops the quota meter.
            log.info("shutdown requested by server: %s", payload.get("reason", "no reason given"))
            await self.shutdown(payload.get("reason") or "stopped from the dashboard")
            return

        elif kind == "job.cancel":
            job_id = payload["jobId"]
            log.info("job %s cancelled by server", job_id[:8])
            for entry in (self.models.resident.values() if self.models else []):
                entry.engine.cancel(job_id)
            if self.engine:
                self.engine.cancel(job_id)

        elif kind == "ping":
            await self.send("pong")

        elif kind == "error":
            log.error("server error: %s", payload.get("message"))

    # ── connection loop ─────────────────────────────────────
    async def connect_forever(self) -> None:
        backoff = 1.0
        while self.running:
            try:
                log.info("connecting to %s…", self.ws_url)
                async with websockets.connect(
                    self.ws_url,
                    additional_headers={"Authorization": f"Bearer {self.token}"},
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                    max_size=8 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    backoff = 1.0
                    await self.register()
                    hb = asyncio.create_task(self.heartbeat_loop())
                    try:
                        async for message in ws:
                            await self.handle(message)
                    finally:
                        hb.cancel()

            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — reconnect on anything
                log.warning("connection lost (%s: %s)", type(exc).__name__, exc)
            finally:
                self._ws = None

            if not self.running:
                break
            log.info("reconnecting in %.0fs", backoff)
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=backoff)
                break  # shutdown requested during the wait
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, 30.0)

    async def shutdown(self, reason: str = "worker shutting down") -> None:
        """Deregister cleanly so the UI shows offline immediately, not after the lease expires."""
        if not self.running:
            return
        log.info("shutting down: %s", reason)
        self.running = False

        for task in list(self._tasks.values()):
            task.cancel()

        if self._ws:
            await self.send("deregister", {"reason": reason})
            await asyncio.sleep(0.3)  # let the frame flush before the socket closes
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass

        if self.models:
            await self.models.stop_all()
            self.engine = None
        if self.engine:
            await self.engine.stop()

        self._shutdown.set()


async def main() -> None:
    worker = Worker()
    loop = asyncio.get_running_loop()

    def request_stop(signame: str) -> None:
        loop.create_task(worker.shutdown(f"received {signame}"))

    for signame in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signame):
            try:
                loop.add_signal_handler(getattr(signal, signame), request_stop, signame)
            except NotImplementedError:
                # Windows/notebook loops do not support signal handlers.
                pass

    try:
        await worker.start_engine()
    except Exception as exc:  # noqa: BLE001
        log.error("engine failed to start: %s", exc)
        sys.exit(1)

    try:
        await worker.connect_forever()
    finally:
        await worker.shutdown("event loop exiting")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
