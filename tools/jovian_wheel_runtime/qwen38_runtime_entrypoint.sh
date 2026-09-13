#!/usr/bin/env bash
# Preload the packaged NCCL build before Python imports PyTorch or vLLM.
set -euo pipefail

nccl_so=$(local-inference-nccl-path)
export LD_PRELOAD="${nccl_so}${LD_PRELOAD:+:${LD_PRELOAD}}"
export VLLM_NCCL_SO_PATH="${nccl_so}"
exec "$@"
