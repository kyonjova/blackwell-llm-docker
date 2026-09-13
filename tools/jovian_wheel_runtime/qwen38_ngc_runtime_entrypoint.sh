#!/usr/bin/env bash
# Select the source-locked CUDA toolkit and LIL NCCL build before Python starts.
set -euo pipefail

cuda_root=/usr/local/cuda
test -f "${cuda_root}/include/cuda.h"
test -x "${cuda_root}/bin/nvcc"

export CUDA_HOME="${cuda_root}"
export CUDA_PATH="${cuda_root}"
export PATH="${cuda_root}/bin:/opt/venv/bin:${PATH}"
export CPATH="${cuda_root}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${cuda_root}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${cuda_root}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

nccl_so=$(local-inference-nccl-path)
export LD_PRELOAD="${nccl_so}${LD_PRELOAD:+:${LD_PRELOAD}}"
export VLLM_NCCL_SO_PATH="${nccl_so}"
exec "$@"
