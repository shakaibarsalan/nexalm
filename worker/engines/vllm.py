"""vLLM engine — for Ampere-or-newer GPUs.

Not the default. On Kaggle's free tier (2x T4, Turing sm_75) vLLM loses its
Marlin quantisation kernels and bf16, and Qwen3-MoE support leans on newer
paths, which is why llama.cpp is the shipped default there. On an A100/L4/H100
this engine is the faster choice — especially under concurrency, where
continuous batching matters and llama.cpp's single-slot serialisation does not.

Talks to `vllm serve`'s OpenAI-compatible endpoint over localhost, mirroring
:class:`~engines.llamacpp.LlamaCppEngine`, so switching engines is a config
change rather than a code change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from typing import AsyncIterator

import httpx

from .base import GenerationParams, GenerationResult, InferenceEngine
from .gpu import detect_gpus

log = logging.getLogger("worker.vllm")


def _supports_bf16(gpu_name: str) -> bool:
    """Ampere and newer. Turing/Pascal must run fp16."""
    lowered = gpu_name.lower()
    turing_or_older = ("t4", "p100", "p4", "v100", "k80", "gtx 16", "rtx 20")
    return not any(tag in lowered for tag in turing_or_older)


class VllmEngine(InferenceEngine):
    name = "vllm"

    def __init__(
        self,
        model: str,
        *,
        context_length: int = 16384,
        host: str = "127.0.0.1",
        port: int = 8091,
        quantization: str | None = "awq",
        gpu_memory_utilization: float = 0.90,
        extra_args: list[str] | None = None,
        startup_timeout_s: int = 1800,
    ) -> None:
        self.model = model
        self.context_length = context_length
        self.host = host
        self.port = port
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.extra_args = extra_args or []
        self.startup_timeout_s = startup_timeout_s

        self._proc: subprocess.Popen | None = None
        self._client: httpx.AsyncClient | None = None
        self._cancelled: set[str] = set()

    @property
    def max_context(self) -> int:
        return self.context_length

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _build_command(self) -> list[str]:
        gpus = detect_gpus()
        cmd = [
            "vllm", "serve", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--max-model-len", str(self.context_length),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]

        # Unlike llama.cpp's layer split, vLLM shards tensors across GPUs, so
        # this wants a fast interconnect to pay off.
        if len(gpus) > 1:
            cmd += ["--tensor-parallel-size", str(len(gpus))]
            log.info("tensor-parallel across %d GPUs", len(gpus))

        if self.quantization:
            cmd += ["--quantization", self.quantization]

        dtype = "bfloat16" if gpus and all(_supports_bf16(g.name) for g in gpus) else "float16"
        cmd += ["--dtype", dtype]
        if dtype == "float16":
            log.warning("pre-Ampere GPU detected — forcing float16 (bf16 unsupported)")

        return cmd + self.extra_args

    async def start(self) -> None:
        if shutil.which("vllm") is None:
            raise FileNotFoundError("vllm not installed (pip install vllm)")

        cmd = self._build_command()
        log.info("starting vLLM: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, "VLLM_LOGGING_LEVEL": "INFO"},
        )
        asyncio.create_task(self._drain_logs())

        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(None))
        await self._await_ready()
        log.info("vLLM ready at %s", self.base_url)

    async def _drain_logs(self) -> None:
        proc = self._proc
        if not proc or not proc.stdout:
            return
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if not line:
                break
            log.debug("[vllm] %s", line.rstrip())

    async def _await_ready(self) -> None:
        assert self._client is not None
        deadline = time.monotonic() + self.startup_timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup (code {self._proc.returncode})")
            try:
                if (await self._client.get("/health", timeout=10)).status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001 — retry until the deadline
                last_err = exc
            await asyncio.sleep(5)
        raise TimeoutError(f"vLLM not ready within {self.startup_timeout_s}s: {last_err}")

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

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

        payload: dict = {
            "model": self.model,
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
            payload["repetition_penalty"] = params.repeat_penalty
        if params.stop:
            payload["stop"] = params.stop

        started = time.monotonic()
        first_token_at: float | None = None
        counted = 0
        prompt_tokens: int | None = None
        reported: int | None = None
        stop_reason: str | None = None

        async with self._client.stream("POST", "/v1/chat/completions", json=payload) as res:
            if res.status_code != 200:
                body = (await res.aread()).decode("utf-8", "replace")[:500]
                raise RuntimeError(f"vLLM returned {res.status_code}: {body}")

            async for line in res.aiter_lines():
                if job_id in self._cancelled:
                    stop_reason = "cancelled"
                    break
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if usage := chunk.get("usage"):
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    reported = usage.get("completion_tokens", reported)

                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        stop_reason = choice["finish_reason"]
                    reasoning = delta.get("reasoning_content") or ""
                    text = delta.get("content") or ""
                    if not text and not reasoning:
                        continue
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    counted += 1
                    if reasoning:
                        yield "reasoning", reasoning, None
                    if text:
                        yield "content", text, None

        total = reported if reported is not None else counted
        gen_ms = int((time.monotonic() - (first_token_at or started)) * 1000)
        yield "", "", GenerationResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=total,
            gen_ms=gen_ms,
            tokens_per_sec=round(total / (gen_ms / 1000), 2) if total and gen_ms > 0 else None,
            stop_reason=stop_reason or "stop",
        )
