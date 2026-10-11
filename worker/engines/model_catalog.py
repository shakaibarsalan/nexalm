"""Pick a GGUF that actually exists, sized to the GPU you actually got.

Filenames were previously constructed from a quantisation name, e.g.
``Qwen3-30B-A3B-{quant}.gguf``. That is wrong in both directions: Unsloth
prefixes its dynamic quants with ``UD-`` (so ``Qwen3-30B-A3B-IQ3_XXS.gguf``
404s while ``Qwen3-30B-A3B-UD-IQ3_XXS.gguf`` exists), and the hardcoded size
table was an estimate that drifts whenever a repo is requantised.

So the repository is listed at runtime and the choice is made from real
filenames and real byte sizes.
"""

from __future__ import annotations

import json
import logging
import urllib.request

log = logging.getLogger("worker.catalog")

HF_TREE = "https://huggingface.co/api/models/{repo}/tree/main?recursive=true"


def list_gguf(repo: str, timeout: int = 60) -> list[tuple[str, int]]:
    """[(filename, bytes)] for single-file GGUFs, smallest first.

    Sharded weights (``…-00001-of-00002.gguf``, or anything in a subdirectory)
    are skipped: llama.cpp can load them, but only given every shard, and the
    download/caching path here handles one file.
    """
    url = HF_TREE.format(repo=repo)
    req = urllib.request.Request(url, headers={"User-Agent": "nexalm-worker"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        rows = json.loads(res.read().decode())

    found: list[tuple[str, int]] = []
    for row in rows:
        path = row.get("path", "")
        if not path.endswith(".gguf") or "/" in path:
            continue
        if "-of-" in path:
            continue
        size = (row.get("lfs") or {}).get("size") or row.get("size") or 0
        if size:
            found.append((path, int(size)))

    found.sort(key=lambda p: p[1])
    return found


# Best quality first. Deliberately NOT "the largest that fits": on 2x T4,
# picking purely by size lands on Q6_K_XL (26 GB of a 29.5 GB budget), which
# starves the KV cache and is slower on memory bandwidth for a quality gain
# nobody can see. Q4_K_M is the knee of the curve here, and is what measured
# 52-68 tok/s on that hardware.
PREFERENCE = [
    "Q4_K_M",
    "UD-Q4_K_XL",
    "Q4_K_S",
    "IQ4_XS",
    "Q3_K_M",
    "UD-Q3_K_XL",
    "Q3_K_S",
    "UD-IQ3_XXS",
    "UD-Q2_K_XL",
    "Q2_K_L",
    "Q2_K",
    "UD-IQ2_M",
    "UD-IQ2_XXS",
]


def _rank(filename: str) -> int:
    """Position in PREFERENCE, matching the longest tag first.

    Longest-first matters because "Q4_K_M" is a substring of nothing here but
    "Q2_K" is a substring of "Q2_K_L"; matching short tags first would
    mislabel the larger file.
    """
    for i, tag in enumerate(sorted(PREFERENCE, key=len, reverse=True)):
        if tag in filename:
            return PREFERENCE.index(tag)
    return len(PREFERENCE)


def choose_gguf(repo: str, headroom_bytes: float) -> tuple[str, int]:
    """Best quantisation that fits the GPU, else the smallest available.

    Falling back to the smallest rather than failing is deliberate: the worker
    can still run it with layers offloaded to CPU, which is slow but useful,
    whereas refusing to start leaves the user with nothing.
    """
    files = list_gguf(repo)
    if not files:
        raise RuntimeError(f"no single-file GGUF found in {repo}")

    fits = [f for f in files if f[1] <= headroom_bytes]
    if not fits:
        chosen = files[0]
        log.warning(
            "nothing fits %.1f GB of headroom; taking the smallest (%s, %.1f GB) "
            "and offloading the remainder to CPU",
            headroom_bytes / 1e9,
            chosen[0],
            chosen[1] / 1e9,
        )
        return chosen

    # Rank first, then prefer the larger file among equals.
    fits.sort(key=lambda f: (_rank(f[0]), -f[1]))
    return fits[0]
