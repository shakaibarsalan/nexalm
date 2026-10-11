"""GPU / accelerator discovery.

Kaggle's free tier is not a fixed allocation: a session may come back with
2x T4 (16 GB each, Turing sm_75), 1x P100, or occasionally 1x T4. The worker
must therefore *measure* the runtime rather than assume it, and report what it
found so the backend and UI show the truth.
"""

from __future__ import annotations

import shutil
import subprocess

from .base import GpuInfo


def detect_gpus() -> list[GpuInfo]:
    """Query nvidia-smi. Returns an empty list on a CPU-only host.

    Also collects utilisation and temperature so the UI can chart how hard the
    cards are actually working, not just how full they are.
    """
    if shutil.which("nvidia-smi") is None:
        return []

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,"
                "utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []

    def _num(parts: list[str], i: int) -> float | None:
        """nvidia-smi prints [N/A] for unsupported counters on some cards."""
        if i >= len(parts):
            return None
        try:
            return float(parts[i])
        except ValueError:
            return None

    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mb=int(float(parts[2])),
                    free_mb=int(v) if (v := _num(parts, 3)) is not None else None,
                    util_pct=int(v) if (v := _num(parts, 4)) is not None else None,
                    mem_util_pct=int(v) if (v := _num(parts, 5)) is not None else None,
                    temp_c=int(v) if (v := _num(parts, 6)) is not None else None,
                    power_w=round(v, 1) if (v := _num(parts, 7)) is not None else None,
                )
            )
        except ValueError:
            continue
    return gpus


def total_vram_mb(gpus: list[GpuInfo]) -> int:
    return sum(g.total_mb for g in gpus)


def free_vram_mb(gpus: list[GpuInfo]) -> int | None:
    values = [g.free_mb for g in gpus if g.free_mb is not None]
    return sum(values) if values else None


def tensor_split(gpus: list[GpuInfo]) -> str | None:
    """Proportional llama.cpp ``--tensor-split`` for a multi-GPU host.

    Splitting by layer (rather than tensor-parallel) is what makes 2x T4 viable:
    the cards have no NVLink, so per-token cross-GPU traffic would dominate.
    Weighted by capacity so mismatched cards still balance.
    """
    if len(gpus) < 2:
        return None
    largest = max(g.total_mb for g in gpus) or 1
    return ",".join(f"{g.total_mb / largest:.3f}" for g in sorted(gpus, key=lambda g: g.index))


def describe(gpus: list[GpuInfo]) -> str:
    if not gpus:
        return "CPU only"
    return ", ".join(f"{g.name} ({g.total_mb / 1024:.0f} GB)" for g in gpus)
