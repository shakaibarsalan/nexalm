#!/usr/bin/env bash
# Push and run the GPU worker as a Kaggle kernel.
#
# Requires the Kaggle API to be authenticated. Provide credentials by ONE of:
#   * ~/.kaggle/kaggle.json  (chmod 600)
#   * KAGGLE_USERNAME + KAGGLE_KEY in the environment
# Never put either into this repository — .gitignore covers kaggle.json.
#
# By default this updates ONLY the worker-code dataset, which is where all the
# provisioning logic lives (bootstrap.py). That is deliberate:
#
#   `kaggle kernels push` resets the notebook's accelerator to the default
#   (P100 instead of T4 x2) AND drops its attached Secrets. Every push
#   therefore costs a failed run plus manual reconfiguration in the UI.
#
# Pass --push-kernel only when the thin loader itself changed.
#
# WORKER_TOKEN is never passed here; it is read from Kaggle Secrets at runtime,
# so it never enters the kernel source, the dataset, or this repo.
set -euo pipefail

PUSH_KERNEL=0
for arg in "$@"; do
  case "$arg" in
    --push-kernel) PUSH_KERNEL=1 ;;
    -h|--help)
      echo "usage: $0 [--push-kernel]"
      echo "  (default) update the worker-code dataset only"
      echo "  --push-kernel  also push the kernel; resets accelerator + Secrets"
      exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_USER="${KAGGLE_USERNAME:-shakaibkaggle}"
CODE_DATASET="${CODE_DATASET:-local-llm-worker-code}"
# Stage inside the repo: the containerised CLI mounts the working directory,
# so a /tmp staging dir would be invisible to it.
STAGE="$REPO_ROOT/.kaggle-stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
cd "$REPO_ROOT"

# The CLI runs in a container: kaggle >= 1.8 needs Python >= 3.11 and the
# Oracle host ships 3.10. See scripts/kaggle-cli.sh.
KAGGLE="$REPO_ROOT/scripts/kaggle-cli.sh"
[[ -x "$KAGGLE" ]] || { echo "error: scripts/kaggle-cli.sh missing or not executable" >&2; exit 1; }

echo "==> verifying Kaggle authentication"
"$KAGGLE" config view >/dev/null 2>&1 || {
  echo "error: Kaggle API is not authenticated." >&2
  echo "  Put kaggle.json at ~/.kaggle/kaggle.json (chmod 600)," >&2
  echo "  or export KAGGLE_USERNAME and KAGGLE_KEY." >&2
  exit 1
}

# ── 1. Ship the worker source as a private dataset ──────────────────────────
# A Kaggle script kernel is a single file, so the worker package travels as an
# attached dataset. This also versions it: each push is a new dataset version.
echo "==> staging worker source"
mkdir -p .kaggle-stage/code
cp "$REPO_ROOT/worker/worker.py" "$STAGE/code/"
# Holds the resident model set: loading and unloading without a restart.
cp "$REPO_ROOT/worker/models.py" "$STAGE/code/"
# bootstrap.py carries the provisioning logic, so it ships with the code.
cp "$REPO_ROOT/worker/kaggle/bootstrap.py" "$STAGE/code/"
# Turns a downloaded model into a private Dataset so the next session
# mounts it read-only instead of downloading it again.
cp "$REPO_ROOT/worker/kaggle/publish_model_dataset.py" "$STAGE/code/"
cp -r "$REPO_ROOT/worker/engines" "$STAGE/code/"
# Provisioning decisions shared with the Colab bootstrap: GPU probing, layer
# placement, model naming, the libcuda fixup. Kept in one place so the two
# platforms cannot drift apart again.
cp -r "$REPO_ROOT/worker/provision" "$STAGE/code/"
find "$STAGE/code" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# A session started through the Kaggle API cannot read the notebook's attached
# Secrets, so the worker token also travels in the private dataset. Sourced
# from .env at upload time, never from the repository.
if [[ -f "$REPO_ROOT/.env" ]]; then
  WT=$(grep "^WORKER_TOKEN=" "$REPO_ROOT/.env" | cut -d= -f2-)
  ORACLE=$(grep "^PUBLIC_ORIGIN=" "$REPO_ROOT/.env" | cut -d= -f2- | sed "s|^https|wss|; s|^http|ws|")
  if [[ -n "$WT" ]]; then
    cat > "$STAGE/code/worker-config.json" <<CFGEOF
{
  "WORKER_TOKEN": "$WT",
  "ORACLE_WS_URL": "${ORACLE:-wss://chat.example.com}/worker/ws"
}
CFGEOF
    chmod 600 "$STAGE/code/worker-config.json"
    echo "    included worker-config.json (private dataset only)"
  fi
fi

cat > "$STAGE/code/dataset-metadata.json" <<EOF
{
  "title": "Local LLM worker code",
  "id": "$KAGGLE_USER/$CODE_DATASET",
  "licenses": [{ "name": "unknown" }]
}
EOF

if "$KAGGLE" datasets status "$KAGGLE_USER/$CODE_DATASET" >/dev/null 2>&1; then
  echo "==> updating dataset $KAGGLE_USER/$CODE_DATASET"
  "$KAGGLE" datasets version -p .kaggle-stage/code -m "worker $(git -C "$REPO_ROOT" rev-parse --short HEAD)" --dir-mode zip
else
  echo "==> creating private dataset $KAGGLE_USER/$CODE_DATASET"
  "$KAGGLE" datasets create -p .kaggle-stage/code --dir-mode zip
fi

# ── 2. Push the kernel (only when explicitly asked) ─────────────────────────
# A push resets the notebook's accelerator to the default P100 and drops its
# attached Secrets, so it is never done as a side effect of shipping code.
if [[ $PUSH_KERNEL -eq 1 ]]; then
  echo "==> pushing kernel (this RESETS the accelerator and Secrets)"
  mkdir -p .kaggle-stage/kernel
  cp "$REPO_ROOT/worker/kaggle/kaggle_worker_entry.py" "$STAGE/kernel/"
  sed "s|\"shakaibkaggle/|\"$KAGGLE_USER/|g" \
      "$REPO_ROOT/worker/kaggle/kernel-metadata.json" > "$STAGE/kernel/kernel-metadata.json"

  "$KAGGLE" kernels push -p .kaggle-stage/kernel

  echo
  echo "  !! Before the next run, in the notebook editor:"
  echo "     1. Session options -> Accelerator -> GPU T4 x2"
  echo "     2. Add-ons -> Secrets -> confirm WORKER_TOKEN is present and ticked"
else
  echo "==> dataset updated; kernel untouched (accelerator + Secrets preserved)"
  echo "    Re-run the notebook from the UI to pick up the new worker code."
fi

echo
echo "==> Follow the run with:"
echo "     ./scripts/kaggle-cli.sh kernels status $KAGGLE_USER/local-llm-gpu-worker"
echo "     ./scripts/kaggle-cli.sh kernels output $KAGGLE_USER/local-llm-gpu-worker -p ./kaggle-output"
echo
echo "Reminders:"
echo "  * WORKER_TOKEN must exist as a Kaggle Secret (Add-ons -> Secrets)."
echo "  * Attach the model dataset holding the .gguf, or the first run downloads it."
echo "  * Sessions cap at 12 h and GPU quota at ~30 h/week; the worker deregisters"
echo "    cleanly on shutdown, so the UI shows offline rather than hanging."
