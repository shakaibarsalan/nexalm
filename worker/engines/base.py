"""Engine abstraction.

The worker protocol is engine-agnostic on purpose: swapping llama.cpp for vLLM,
or a Kaggle T4 pair for a dedicated A100, must not require touching the Oracle
backend or the transport layer.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class GenerationParams:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int | None = None
    max_tokens: int = 2048
    repeat_penalty: float | None = None
    stop: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Terminal stats for one generation."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    gen_ms: int | None = None
    tokens_per_sec: float | None = None
    stop_reason: str | None = None


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int | None = None
    util_pct: int | None = None
    mem_util_pct: int | None = None
    temp_c: int | None = None
    power_w: float | None = None

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "totalMb": self.total_mb,
            "freeMb": self.free_mb,
            "utilPct": self.util_pct,
            "memUtilPct": self.mem_util_pct,
            "tempC": self.temp_c,
            "powerW": self.power_w,
        }


class InferenceEngine(abc.ABC):
    """A local model runtime the worker drives."""

    name: str = "base"

    @abc.abstractmethod
    async def start(self) -> None:
        """Load the model and become ready to serve. Raises on failure."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Release the model and any child process."""

    @abc.abstractmethod
    async def generate(
        self,
        job_id: str,
        messages: list[dict],
        params: GenerationParams,
        system_prompt: str | None = None,
    ) -> AsyncIterator[tuple[str, str, "GenerationResult | None"]]:
        """Yield ``(kind, delta_text, None)`` per token, then ``("", "", result)``.

        ``kind`` is ``"content"`` for answer text or ``"reasoning"`` for a
        thinking trace. Reasoning is surfaced rather than dropped: models like
        Qwen3 can spend hundreds of tokens there, and discarding them makes the
        user wait for output they never see *and* corrupts tokens/sec, because
        the token count would include them while the elapsed time would not.

        Implementations must stop promptly when :meth:`cancel` is called for the
        same ``job_id``.
        """
        raise NotImplementedError
        yield  # pragma: no cover — makes this an async generator for type checkers

    @abc.abstractmethod
    def cancel(self, job_id: str) -> None:
        """Request that an in-flight generation stop."""

    @property
    @abc.abstractmethod
    def max_context(self) -> int:
        """Context window the engine was configured with."""
