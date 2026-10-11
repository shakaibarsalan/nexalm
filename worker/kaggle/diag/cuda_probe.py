"""CPU-only probe: work out how to make llama.cpp's CUDA build configure here.

Runs with accelerator = None so it costs no GPU quota. The cmake CONFIGURE step
is what has been failing ("Target ggml-cuda links to CUDA::cuda_driver but the
target was not found"), and configuring needs only the CUDA toolkit, not a GPU.
"""

import glob
import os
import subprocess

def sh(cmd, check=False):
    print("\n$ " + cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-3000:])
    return r.returncode, out

print("=" * 70); print("1. Where is libcuda?"); print("=" * 70)
sh("ldconfig -p | grep -i 'libcuda' || echo '(none in ldconfig)'")
for d in ["/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs",
          "/usr/local/cuda/targets/x86_64-linux/lib",
          "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
          "/usr/lib/x86_64-linux-gnu"]:
    hits = sorted(glob.glob(os.path.join(d, "libcuda*")))
    print(f"  {d}: {hits if hits else '(no libcuda*)'}")

sh("nvcc --version | tail -2; cmake --version | head -1")

print("\n" + "=" * 70); print("2. Fetch llama.cpp"); print("=" * 70)
sh("apt-get update -qq && apt-get install -y -qq cmake build-essential", check=True)
if not os.path.isdir("/kaggle/working/llama.cpp/.git"):
    sh("rm -rf /kaggle/working/llama.cpp && git clone --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp")

BASE = ("cmake -S /kaggle/working/llama.cpp -B /kaggle/working/probe-%s "
        "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF "
        "-DCMAKE_BUILD_TYPE=Release")

print("\n" + "=" * 70); print("3. Candidate fixes"); print("=" * 70)

results = {}

# A: baseline, to confirm the failure reproduces without a GPU present.
rc, _ = sh(BASE % "a")
results["A baseline"] = rc

# B: quote CMAKE_LIBRARY_PATH properly (the ';' broke the shell last time).
rc, _ = sh((BASE % "b") + " '-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs'")
results["B CMAKE_LIBRARY_PATH=stubs"] = rc

# C: symlink the stub to a standard name so find_library(cuda) succeeds.
sh("for d in /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib; do "
   "  if [ -e $d/stubs/libcuda.so ] && [ ! -e $d/libcuda.so ]; then "
   "    ln -sf $d/stubs/libcuda.so $d/libcuda.so && echo linked $d/libcuda.so; fi; done; "
   "ls -l /usr/local/cuda/lib64/libcuda.so /usr/local/cuda/targets/x86_64-linux/lib/libcuda.so 2>/dev/null")
rc, _ = sh(BASE % "c")
results["C stub symlink"] = rc

# D: point CUDAToolkit_ROOT explicitly as well.
rc, _ = sh((BASE % "d") + " -DCUDAToolkit_ROOT=/usr/local/cuda")
results["D + CUDAToolkit_ROOT"] = rc

print("\n" + "=" * 70); print("RESULTS (0 = configure succeeded)"); print("=" * 70)
for k, v in results.items():
    print(f"  {'PASS' if v == 0 else 'FAIL'}  {k}  (rc={v})")

winner = next((k for k, v in results.items() if v == 0), None)
print("\n-> first working option:", winner or "NONE")
