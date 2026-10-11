"""llama.cpp engine.

Chosen for the Kaggle target because the free tier is Turing (sm_75): no bf16,
no FlashAttention, and no NVLink between the two T4s. llama.cpp splits a model
across cards by layer, needs no fast interconnect, and ships an
OpenAI-compatible server we can stream from.

The engine talks to a ``llama-server`` child process over localhost HTTP rather
than binding the library in-process, which keeps a model crash from taking the
worker (and its Oracle connection) down with it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from typing import AsyncIterator

import httpx

from .base import GenerationParams, GenerationResult, InferenceEngine
from .gpu import detect_gpus, tensor_split

log = logging.getLogger("worker.llamacpp")


def ensure_soname_links(bin_dir: str) -> list[str]:
    """Recreate the SONAME symlinks a normal build directory would have.

    The shared objects declare SONAME=libllama.so.0 while the files on disk are
    libllama.so.0.4.0. CMake normally leaves a libllama.so.0 -> libllama.so.0.4.0
    symlink beside them, but symlinks do not survive being shipped through a
    Kaggle dataset or a per-file download, so the loader finds nothing and
    llama-server exits 127 with no useful message.

    The major version is taken from the filename rather than by shelling out to
    readelf, which is not guaranteed to be installed on these runtimes.
    """
    created: list[str] = []
    if not os.path.isdir(bin_dir):
        return created

    for name in os.listdir(bin_dir):
        match = re.match(r"^(?P<stem>.+\.so)\.(?P<major>\d+)(?:\.\d+)+$", name)
        if not match:
            continue
        stem, major = match.group("stem"), match.group("major")
        for alias in (f"{stem}.{major}", stem):
            target = os.path.join(bin_dir, alias)
            if os.path.exists(target) or os.path.islink(target):
                continue
            try:
                os.symlink(name, target)
                created.append(alias)
            except OSError as exc:  # noqa: PERF203 — read-only mount is expected
                log.debug("could not link %s: %s", alias, exc)
    if created:
        log.info("created %d SONAME link(s) in %s", len(created), bin_dir)
    return created


class LlamaCppEngine(InferenceEngine):
    name = "llamacpp"

    def __init__(
        self,
        model_path: str,
        *,
        context_length: int = 16384,
        host: str = "127.0.0.1",
        port: int = 8090,
        n_gpu_layers: int = 999,
        server_binary: str | None = None,
        extra_args: list[str] | None = None,
        startup_timeout_s: int = 900,
        mmproj_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.context_length = context_length
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.server_binary = server_binary or os.environ.get("LLAMA_SERVER_BIN", "llama-server")
        self.extra_args = extra_args or []
        # llama.cpp reads images through a separate projector file rather than
        # baking vision into the weights, so a multimodal model is an ordinary
        # GGUF plus this one extra argument.
        self.mmproj_path = mmproj_path
        self.startup_timeout_s = startup_timeout_s

        self._proc: subprocess.Popen | None = None
        self._ready = False
        self._client: httpx.AsyncClient | None = None
        self._cancelled: set[str] = set()

    @property
    def max_context(self) -> int:
        return self.context_length

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # ── lifecycle ───────────────────────────────────────────
    def _build_command(self) -> list[str]:
        gpus = detect_gpus()
        cmd = [
            self.server_binary,
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.context_length),
            "--n-gpu-layers", str(self.n_gpu_layers if gpus else 0),
            # One slot: a T4 pair holding a 30B MoE has no room to batch, and
            # serialising keeps tokens/sec honest per request.
            "--parallel", "1",
        ]

        split = tensor_split(gpus)
        if split:
            cmd += ["--tensor-split", split]
            log.info("multi-GPU detected (%d cards); tensor-split=%s", len(gpus), split)
        elif gpus:
            log.info("single GPU detected: %s", gpus[0].name)
        else:
            log.warning("no GPU detected — llama.cpp will run on CPU")

        # NOTE: --flash-attn is deliberately not passed. Its argument form has
        # changed across llama.cpp releases (bare flag vs on/off/auto), and
        # Turing (T4/P100) has no FlashAttention kernels anyway, so the default
        # is already correct here. Pass it via EXTRA_ARGS on Ampere or newer.
        if gpus and all(_is_turing_or_older(g.name) for g in gpus):
            log.info("Turing-class GPUs — leaving FlashAttention at its default (off)")

        if self.mmproj_path:
            cmd += ["--mmproj", self.mmproj_path]
            log.info("vision enabled via projector %s", os.path.basename(self.mmproj_path))

        return cmd + self.extra_args

    async def start(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model file not found: {self.model_path}")
        if shutil.which(self.server_binary) is None and not os.path.exists(self.server_binary):
            raise FileNotFoundError(
                f"llama-server binary not found: {self.server_binary}. "
                "Set LLAMA_SERVER_BIN or install llama.cpp."
            )

        # Must run before launching: without the SONAME links the dynamic
        # loader cannot resolve the ggml/llama libraries and the process dies
        # with exit 127 before printing anything.
        ensure_soname_links(os.path.dirname(os.path.abspath(self.server_binary)))

        cmd = self._build_command()
        log.info("starting llama-server: %s", " ".join(cmd))
        self._ready = False
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        asyncio.create_task(self._drain_logs())

        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(None))
        await self._await_ready()
        log.info("llama-server ready at %s", self.base_url)

    async def _drain_logs(self) -> None:
        """Surface child output; a silent model load looks like a hang.

        Startup lines go out at INFO because that is exactly when the operator
        is staring at a wall of 503s wondering whether anything is happening —
        llama.cpp prints its tensor-loading progress here. Once the server is
        serving, the same stream is per-request noise, so it drops to DEBUG.
        """
        proc = self._proc
        if not proc or not proc.stdout:
            return
        loop = asyncio.get_running_loop()
        fd = proc.stdout.fileno()
        pending = b""
        while True:
            # os.read rather than readline: llama.cpp draws its tensor-loading
            # progress with carriage returns and no newline, so readline blocks
            # for the whole load and the operator sees an unexplained silence
            # at exactly the moment they most want output.
            block = await loop.run_in_executor(None, lambda: os.read(fd, 65536))
            if not block:
                break
            pending += block
            pending = pending.replace(b"\r\n", b"\n")
            *lines, pending = pending.replace(b"\r", b"\n").split(b"\n")
            level = logging.DEBUG if self._ready else logging.INFO
            for raw in lines:
                text = raw.decode(errors="replace").rstrip()
                if text:
                    log.log(level, "[llama-server] %s", text)

    async def _await_ready(self) -> None:
        """Poll /health until the model finishes loading (18 GB takes a while)."""
        assert self._client is not None
        deadline = time.monotonic() + self.startup_timeout_s
        last_err: Exception | None = None

        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup (code {self._proc.returncode})")
            try:
                res = await self._client.get("/health", timeout=10)
                if res.status_code == 200:
                    self._ready = True
                    return
            except Exception as exc:  # noqa: BLE001 — retry until the deadline
                last_err = exc
            await asyncio.sleep(3)

        raise TimeoutError(f"llama-server not ready within {self.startup_timeout_s}s: {last_err}")

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

    # ── inference ───────────────────────────────────────────
    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    async def generate(
        self,
        job_id: str,
        messages: list[dict],
        params: GenerationParams,
        system_prompt: str | None = None,
    ) -> AsyncIterator[tuple[str, str, GenerationResult | None]]:
        assert self._client is not None, "engine not started"
        self._cancelled.discard(job_id)

        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages += [{"role": m["role"], "content": m["content"]} for m in messages]

        payload = {
            "messages": payload_messages,
            "stream": True,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "stream_options": {"include_usage": True},
        }
        if params.top_k is not None:
            payload["top_k"] = params.top_k
        if params.repeat_penalty is not None:
            payload["repeat_penalty"] = params.repeat_penalty
        if params.stop:
            payload["stop"] = params.stop

        started = time.monotonic()
        first_token_at: float | None = None
        completion_tokens = 0
        prompt_tokens: int | None = None
        reported_completion: int | None = None
        stop_reason: str | None = None

        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as res:
            if res.status_code != 200:
                body = (await res.aread()).decode("utf-8", "replace")[:500]
                raise RuntimeError(f"llama-server returned {res.status_code}: {body}")

            async for line in res.aiter_lines():
                if job_id in self._cancelled:
                    stop_reason = "cancelled"
                    break
                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    reported_completion = usage.get("completion_tokens", reported_completion)

                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        stop_reason = choice["finish_reason"]

                    # Qwen3 and other thinking models emit their trace here.
                    # llama.cpp keeps it out of `content`, so reading only
                    # `content` silently discards tokens the GPU already spent
                    # time on.
                    reasoning = delta.get("reasoning_content") or ""
                    text = delta.get("content") or ""
                    if not text and not reasoning:
                        continue
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    completion_tokens += 1
                    if reasoning:
                        yield "reasoning", reasoning, None
                    if text:
                        yield "content", text, None

        total = reported_completion if reported_completion is not None else completion_tokens
        gen_ms = int((time.monotonic() - (first_token_at or started)) * 1000)
        tps = round(total / (gen_ms / 1000), 2) if total and gen_ms > 0 else None

        yield "", "", GenerationResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=total,
            gen_ms=gen_ms,
            tokens_per_sec=tps,
            stop_reason=stop_reason or "stop",
        )


def _is_turing_or_older(gpu_name: str) -> bool:
    """T4/P100/V100 and the Turing GeForce cards Kaggle and Colab hand out."""
    lowered = gpu_name.lower()
    return any(tag in lowered for tag in ("t4", "p100", "p4", "v100", "k80", "gtx 16", "rtx 20"))
