"""Turn a downloaded GGUF into a private Kaggle Dataset, from inside a session.

Why this runs on Kaggle rather than on the server: the weights are already on
the Kaggle machine, and Kaggle's link to Hugging Face is fast, while pushing
eighteen gigabytes out through the Oracle box would be slow and pointless.

Once a model is published this way and attached to the kernel, it mounts
read-only under /kaggle/input and is mmapped in place -- a session start then
moves zero bytes instead of downloading the model again every time.

    python publish_model_dataset.py qwen3-14b
    python publish_model_dataset.py --all

The dataset is named `<user>/nexalm-<model-id>` and holds a single directory
named after the model id, which is exactly the layout ModelManager looks for.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

CACHE_ROOTS = ["/kaggle/temp/nexalm-models", "/content/nexalm-models", "/tmp/nexalm-models"]
STAGE = "/kaggle/temp/nexalm-dataset-stage"


def cached_models() -> dict[str, str]:
    """Model id -> GGUF path, for everything already downloaded here."""
    found: dict[str, str] = {}
    for root in CACHE_ROOTS:
        if not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, "*", "*.gguf"))):
            found.setdefault(os.path.basename(os.path.dirname(path)), path)
    return found


def already_attached(model_id: str) -> str | None:
    hits = glob.glob(f"/kaggle/input/**/{model_id}/*.gguf", recursive=True)
    return hits[0] if hits else None


def kaggle_user() -> str:
    for path in ("/kaggle/input/nexalm-kaggle/kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json")):
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)["username"]
    user = os.environ.get("KAGGLE_USERNAME", "")
    if not user:
        sys.exit("Set KAGGLE_USERNAME, or place kaggle.json at ~/.kaggle/kaggle.json.")
    return user


def publish(model_id: str, gguf: str, user: str) -> None:
    slug = f"nexalm-{model_id}"
    stage = os.path.join(STAGE, slug, model_id)
    os.makedirs(stage, exist_ok=True)

    target = os.path.join(stage, os.path.basename(gguf))
    if not os.path.exists(target):
        # Hard-linked rather than copied: the file is many gigabytes and both
        # paths are on the same filesystem, so a copy would double the space
        # for no benefit. Falls back to a copy across devices.
        try:
            os.link(gguf, target)
        except OSError:
            shutil.copy2(gguf, target)

    meta = os.path.join(STAGE, slug, "dataset-metadata.json")
    with open(meta, "w") as fh:
        json.dump(
            {"title": f"NexaLM {model_id}", "id": f"{user}/{slug}", "licenses": [{"name": "unknown"}]},
            fh,
            indent=2,
        )

    exists = subprocess.run(
        ["kaggle", "datasets", "status", f"{user}/{slug}"],
        capture_output=True, text=True,
    ).returncode == 0

    cmd = (
        ["kaggle", "datasets", "version", "-p", os.path.join(STAGE, slug), "-m", "weights", "-r", "skip"]
        if exists
        else ["kaggle", "datasets", "create", "-p", os.path.join(STAGE, slug), "-r", "skip"]
    )
    print(f"-> {'updating' if exists else 'creating'} {user}/{slug} ({os.path.getsize(gguf) / 1e9:.1f} GB)")
    subprocess.run(cmd, check=True)
    print(f"   done. Attach {user}/{slug} to the kernel; it then mounts read-only.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_id", nargs="?", help="model id to publish")
    ap.add_argument("--all", action="store_true", help="publish every cached model")
    ap.add_argument("--list", action="store_true", help="show what is cached here")
    args = ap.parse_args()

    cached = cached_models()
    if args.list or not (args.model_id or args.all):
        if not cached:
            print("Nothing cached yet. Load a model from the app first, then run this.")
        for mid, path in cached.items():
            mark = "already a dataset" if already_attached(mid) else "not published"
            print(f"  {mid:<22} {os.path.getsize(path) / 1e9:>6.1f} GB   {mark}")
        return

    user = kaggle_user()
    targets = list(cached) if args.all else [args.model_id]
    for mid in targets:
        if mid not in cached:
            print(f"!! {mid} is not cached here; load it from the app first.")
            continue
        if already_attached(mid):
            print(f"-> {mid} already mounts from /kaggle/input; nothing to do.")
            continue
        publish(mid, cached[mid], user)


if __name__ == "__main__":
    main()
