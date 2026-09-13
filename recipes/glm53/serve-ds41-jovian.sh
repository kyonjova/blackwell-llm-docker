#!/usr/bin/env bash
set -euo pipefail

# The common image supplies GLM tuning. Remove only those inherited values;
# preserve logging, tracing, networking, cache paths and different-value
# overrides. Docker cannot distinguish an explicit value equal to its default.
while read -r name inherited; do
    if [[ ${!name-} == "$inherited" ]]; then unset "$name"; fi
done <<'GLM_IMAGE_DEFAULTS'
VLLM_GLM53_L2_PREFETCH 1
VLLM_GLM53_L2_PREFETCH_PERSIST_MB 0
VLLM_GLM53_DFLASH_ATTN 1
VLLM_CAUSAL_CONV1D_UPDATE_HOIST 1
VLLM_GLM53_KDA_GATE_SIDE_STREAM 1
VLLM_GLM53_MTP_DRAFT_HEAD nvfp4
VLLM_MXFP8_LM_HEAD 0
VLLM_MTP_NVFP4_LM_HEAD 0
VLLM_LM_HEAD_A16 1
VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE 768KB
VLLM_GLM53_ONLINE_DENSE_MXFP8 0
VLLM_DISABLE_SHARED_EXPERTS_STREAM 0
VLLM_DISABLED_KERNELS MarlinFP8ScaledMMLinearKernel
VLLM_CPP_AR_1STAGE_NCCL_CUTOFF 56KB
VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS 0
VLLM_SOURCE_OVERLAY_ACTIVE 1
B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE 1
B12X_DYNAMIC_DIRECT_EXPERT_SCALES 1
B12X_DYNAMIC_SPLIT_LOW_SMEM 1
B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET 1
B12X_DYNAMIC_SPLIT_FAST_PREPARE 1
B12X_DYNAMIC_WORK_SOURCE persistent_grid
B12X_DYNAMIC_SPLIT_COMPUTE_MAC 224
B12X_PCIE_ONESHOT_THREADS 512
B12X_PCIE_ONESHOT_BLOCK_LIMIT 4
B12X_PCIE_ONESHOT_PDL 1
B12X_MHC_PDL 1
GLM_IMAGE_DEFAULTS
unset NCCL_GRAPH_FILE
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}

export PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
export MODEL_PATH=${MODEL_PATH:-${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}}
export HOST=${HOST:-0.0.0.0} PORT=${PORT:-8000}
export TP_SIZE=${TP_SIZE:-4}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
export LOAD_FORMAT=${LOAD_FORMAT:-instanttensor}
export ENGRAM_TABLE_MEMORY=${ENGRAM_TABLE_MEMORY:-disk}
export VLLM_HOST_IP=127.0.0.1 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1} NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-16}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-16}
export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-2097152}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
case "$ENGRAM_TABLE_MEMORY" in
    ram|disk) ;;
    *) echo 'ENGRAM_TABLE_MEMORY must be ram or disk.' >&2; exit 2 ;;
esac

generation_args=(--override-generation-config '{"temperature":1.0,"top_p":0.95}')
for argument in "$@"; do
    case "${argument//_/-}" in
        --generation-config|--generation-config=*|--override-generation-config|\
        --override-generation-config=*|--override-generation-config.*|--config|--config=*)
            generation_args=() ;;
    esac
done
exec "${VLLM_SOURCE_DIR:-/opt/glm53-flash/vllm}/serve-ds41-flash.sh" \
    --decode-context-parallel-size "${DCP_SIZE:-1}" \
    --kv-cache-dtype fp8 --enable-chunked-prefill \
    --attention-backend B12X \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    "${generation_args[@]}" "$@"
