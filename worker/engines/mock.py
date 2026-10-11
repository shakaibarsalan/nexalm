"""Synthetic engine used to verify the pipeline without a GPU.

This exists so streaming, cancellation, metrics persistence, heartbeat/lease
handling and the UI can all be exercised end-to-end before a Kaggle session is
available — and so the Oracle deployment can be smoke-tested at any time
without burning GPU quota. It performs no inference and must never be presented
to a user as a model response.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import AsyncIterator

from .base import GenerationParams, GenerationResult, InferenceEngine

_SAMPLE = """Here is a short answer to **{prompt}**.

This response comes from the mock engine, so the text is canned — but the
transport, token timing and metrics around it are real.

```python
def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

A few things worth noting:

1. Tokens arrive incrementally over SSE.
2. Time-to-first-token and tokens/sec are measured, not simulated.
3. Cancellation stops the stream mid-flight.

| Metric | Source |
| --- | --- |
| TTFT | measured at the first delta |
| tok/s | completion tokens over generation time |
"""


class MockEngine(InferenceEngine):
    name = "mock"

    def __init__(self, context_length: int = 8192, tokens_per_sec: float = 25.0) -> None:
        self.context_length = context_length
        self.delay = 1.0 / max(tokens_per_sec, 1.0)
        self._cancelled: set[str] = set()

    @property
    def max_context(self) -> int:
        return self.context_length

    async def start(self) -> None:
        # Stand in for model load time so the READY transition is observable.
        await asyncio.sleep(0.5)

    async def stop(self) -> None:
        return None

    def cancel(self, job_id: str) -> None:
        self._cancelled.add(job_id)

    async def generate(
        self,
        job_id: str,
        messages: list[dict],
        params: GenerationParams,
        system_prompt: str | None = None,
    ) -> AsyncIterator[tuple[str, str, GenerationResult | None]]:
        self._cancelled.discard(job_id)

        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "your question")
        body = _SAMPLE.format(prompt=last_user.strip()[:80])

        # Split on whitespace boundaries so the markdown stays parseable mid-stream.
        pieces = [p + " " for p in body.split(" ")]
        prompt_tokens = sum(len(m["content"]) for m in messages) // 4 + 8

        await asyncio.sleep(random.uniform(0.2, 0.6))  # plausible prefill latency

        started = time.monotonic()
        first_token_at: float | None = None
        emitted = 0
        stop_reason = "stop"

        for piece in pieces:
            if job_id in self._cancelled:
                stop_reason = "cancelled"
                break
            if emitted >= params.max_tokens:
                stop_reason = "length"
                break
            if first_token_at is None:
                first_token_at = time.monotonic()
            emitted += 1
            yield "content", piece, None
            await asyncio.sleep(self.delay)

        gen_ms = int((time.monotonic() - (first_token_at or started)) * 1000)
        yield "", "", GenerationResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=emitted,
            gen_ms=gen_ms,
            tokens_per_sec=round(emitted / (gen_ms / 1000), 2) if gen_ms > 0 else None,
            stop_reason=stop_reason,
        )
