#!/usr/bin/env bash
# Verify the complete foundation and LIL NCCL environment on one GPU.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path=${1:?Pass the installed venv path}
device=${2:-0}
python="${venv_path}/bin/python"
site_packages=$("${python}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cuda_root="${site_packages}/nvidia/cu13"
nccl_library=$("${venv_path}/bin/local-inference-nccl-path")
temporary=$(mktemp -d)
trap 'rm -rf "${temporary}"' EXIT

"${cuda_root}/bin/nvcc" \
  -std=c++17 \
  -arch=sm_120 \
  --cudart=static \
  "${script_dir}/tests/smoke_cuda.cu" \
  -L"${cuda_root}/lib" \
  -o "${temporary}/smoke_cuda"
CUDA_VISIBLE_DEVICES="${device}" "${temporary}/smoke_cuda"

LD_PRELOAD="${nccl_library}" \
CUDA_VISIBLE_DEVICES="${device}" \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29634 \
"${python}" - <<'PY'
from pathlib import Path

import torch
import torch.distributed as dist

assert torch.__version__ == "2.14.0a0+4fdf77b940.nv26.08"
assert torch.version.cuda == "13.4"
assert torch._C._GLIBCXX_USE_CXX11_ABI
value = torch.ones(1, device="cuda")
dist.init_process_group("nccl", rank=0, world_size=1)
dist.all_reduce(value)
dist.destroy_process_group()

library_names = (
    "libcudart",
    "libcublas",
    "libcudnn",
    "libcusparse",
    "libnccl",
    "libnvrtc",
    "libnvJitLink",
)
loaded = sorted(
    {
        line.rsplit(maxsplit=1)[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/" in line and any(name in line for name in library_names)
    }
)
assert not any("/usr/local/cuda" in path for path in loaded), loaded
assert any("local_inference_nccl" in path for path in loaded), loaded
assert any("site-packages/nvidia/cu13" in path for path in loaded), loaded
print(f"clean_uv_gpu_smoke=PASS device={torch.cuda.get_device_name()}")
PY
