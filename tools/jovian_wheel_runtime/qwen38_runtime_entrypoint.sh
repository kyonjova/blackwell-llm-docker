#!/usr/bin/env bash
# Expose the wheel-packaged CUDA toolkit and NCCL library to native JIT loaders.
set -euo pipefail

launcher_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -x ${launcher_dir}/python ]]; then
  runtime_python=${launcher_dir}/python
else
  runtime_python=$(command -v python)
fi
runtime_site=$(
  "${runtime_python}" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)
cuda_root=${runtime_site}/nvidia/cu13
test -f "${cuda_root}/include/cuda.h"
test -x "${cuda_root}/bin/nvcc"

export CUDA_HOME="${cuda_root}"
export CUDA_PATH="${cuda_root}"
export PATH="${cuda_root}/bin:${launcher_dir}:${PATH}"
export CPATH="${cuda_root}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${cuda_root}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${cuda_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

nccl_so=$(local-inference-nccl-path)
export LD_PRELOAD="${nccl_so}${LD_PRELOAD:+:${LD_PRELOAD}}"
export VLLM_NCCL_SO_PATH="${nccl_so}"
exec "$@"
