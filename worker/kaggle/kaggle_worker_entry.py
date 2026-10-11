"""Kaggle kernel entry point — a thin, deliberately stable loader.

All real logic lives in `bootstrap.py`, shipped inside the attached worker-code
dataset. This file should almost never change.

Why the indirection: `kaggle kernels push` silently resets the notebook's
accelerator back to the default (P100 rather than T4 x2) *and* drops its
attached Secrets, so every code change pushed this way costs a broken run and a
manual re-configuration. Updating a dataset does neither. Keeping the kernel
thin means iteration happens through `kaggle datasets version` alone.

Attached inputs (see kernel-metadata.json):
  /kaggle/input/local-llm-worker-code/   bootstrap.py, worker.py, engines/
  /kaggle/input/<model-dataset>/         the .gguf (optional but recommended)
  /kaggle/input/<llamacpp-bin>/          prebuilt llama-server (optional)

Required Kaggle Secret (Add-ons -> Secrets):
  WORKER_TOKEN   — the LABEL must be exactly this; the value is the token
                   from the server's /opt/local-llm/.env
"""

import glob
import os
import runpy
import sys

CANDIDATES = sorted(glob.glob("/kaggle/input/**/bootstrap.py", recursive=True))

if not CANDIDATES:
    print("!! bootstrap.py not found. Contents of /kaggle/input:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count("/") - 2
        if depth > 3:
            dirs[:] = []
            continue
        print("   %s%s/" % ("  " * depth, os.path.basename(root) or "/kaggle/input"))
        for f in sorted(files)[:12]:
            print("   %s  %s" % ("  " * depth, f))
    sys.exit(
        "Worker code dataset is not attached.\n"
        "Run scripts/launch-kaggle-worker.sh to publish it, then attach\n"
        "'Local LLM worker code' under Input in the notebook editor."
    )

BOOTSTRAP = CANDIDATES[0]
print("-> loading %s" % BOOTSTRAP)

# Make the dataset directory importable so `import engines...` resolves.
sys.path.insert(0, os.path.dirname(BOOTSTRAP))

runpy.run_path(BOOTSTRAP, run_name="__main__")
