#!/usr/bin/env bash
# probe-image.sh -- prove a built serving image contains what its profile
# says, before spending a two-node boot on it.
#
# Usage:  ./probe-image.sh IMAGE [build.env]
#
# With a profile, VLLM_PIN / B12X_PIN are compared against what the image
# reports. Checks, in order: source identity, the RoCEnante composition
# (vLLM PR #597 + b12x PR #295), then the generic image contract.
set -euo pipefail
IMAGE="${1:?usage: probe-image.sh IMAGE [build.env]}"
ENV_FILE="${2:-}"
py() { docker run --rm --entrypoint /opt/venv/bin/python "$IMAGE" - "$@"; }
sh() { docker run --rm --entrypoint bash "$IMAGE" -c "$1"; }
pass=0; fail=0
ok()   { echo "  PASS  $*"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail+1)); }
want_vllm=""; want_b12x=""
if [[ -n "$ENV_FILE" ]]; then
  want_vllm="$(grep -E '^VLLM_PIN=' "$ENV_FILE" | cut -d= -f2 || true)"
  want_b12x="$(grep -E '^B12X_PIN=' "$ENV_FILE" | cut -d= -f2 || true)"
fi

echo "== image: $IMAGE"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image not present"; exit 1; }
echo "  digest: $(docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || echo '(local only, none)')"
echo "  labels: $(docker inspect --format '{{json .Config.Labels}}' "$IMAGE" | head -c 400)"

echo "== source identity (image labels vs profile pins)"
label() { docker inspect --format "{{index .Config.Labels \"$1\"}}" "$IMAGE" 2>/dev/null; }
img_vllm="$(label local-inference.vllm.commit)"; img_b12x="$(label local-inference.b12x.commit)"
echo "  vllm commit label: ${img_vllm:-(none)}    b12x commit label: ${img_b12x:-(none)}"
if [[ -n "$want_vllm" ]]; then
  [[ "$img_vllm" == "$want_vllm" ]] && ok "vllm commit matches VLLM_PIN ${want_vllm:0:9}" || bad "vllm commit label '${img_vllm:-none}' != VLLM_PIN ${want_vllm:0:9}"
fi
if [[ -n "$want_b12x" ]]; then
  [[ "$img_b12x" == "$want_b12x" ]] && ok "b12x commit matches B12X_PIN ${want_b12x:0:9}" || bad "b12x commit label '${img_b12x:-none}' != B12X_PIN ${want_b12x:0:9}"
fi
py <<'PY' 2>/dev/null || echo "  NOTE  could not import vllm for version (GPU-less import limitation)"
import vllm; print("  vllm version:", vllm.__version__)
PY

echo "== RoCEnante composition (vLLM #597 + b12x #295)"
# #597 adds vllm/distributed/device_communicators/b12x_roce_all_reduce.py,
# hooks in cuda_communicator.py, and three env vars.
if py <<'PY' 2>/dev/null; then ok "vllm: b12x_roce_all_reduce module present"; else bad "vllm: b12x_roce_all_reduce module MISSING (PR #597 not in image)"; fi
import importlib
importlib.import_module("vllm.distributed.device_communicators.b12x_roce_all_reduce")
PY
if py <<'PY' 2>/dev/null; then ok "vllm: VLLM_ENABLE_ROCE_ALLREDUCE / VLLM_ROCE_*_MAX_SIZE env vars registered"; else bad "vllm: RoCE env vars not registered"; fi
import vllm.envs as e
for k in ("VLLM_ENABLE_ROCE_ALLREDUCE", "VLLM_ROCE_ALLREDUCE_MAX_SIZE", "VLLM_ROCE_ALLGATHER_MAX_SIZE"):
    assert k in e.environment_variables, k
PY
if py <<'PY' 2>/dev/null; then ok "vllm: cuda_communicator carries the B12X_ROCENANTE hook"; else bad "vllm: cuda_communicator has no B12X_ROCENANTE hook"; fi
import pathlib, importlib.util
spec = importlib.util.find_spec("vllm.distributed.device_communicators.cuda_communicator")
src = pathlib.Path(spec.origin).read_text()
assert "B12X_ROCENANTE" in src and "b12x_roce_all_reduce" in src
PY
if py <<'PY' 2>/dev/null; then ok "b12x: b12x.comm.roce importable (PR #295)"; else bad "b12x: b12x.comm.roce MISSING (PR #295 not in image)"; fi
import importlib
m = importlib.import_module("b12x.comm.roce")
# lazy API from OpMeta: these names are installed on import (PR #295)
for n in ("AllReduce", "is_supported", "discover_hcas", "default_gid_index"):
    assert hasattr(m, n), n
print("b12x.comm.roce ok; is_supported() =", m.is_supported() if callable(m.is_supported) else "?")
PY

# #295 builds its RDMA proxy (_roce_proxy.c) at runtime with the host C
# compiler, into B12X_ROCE_CACHE_DIR. A runtime image without cc cannot use it.
if sh 'command -v cc >/dev/null || command -v gcc >/dev/null' 2>/dev/null; then ok "b12x: C compiler present for the RoCE proxy build"; else bad "b12x: no cc/gcc in image -- RoCEnante proxy cannot compile at runtime"; fi
if py <<'PY' 2>/dev/null; then ok "b12x: _roce_proxy.c shipped"; else bad "b12x: _roce_proxy.c missing from installed package"; fi
import pathlib, importlib.util
spec = importlib.util.find_spec("b12x.comm.roce")
assert (pathlib.Path(spec.origin).parent / "_roce_proxy.c").is_file()
PY

echo "== KDA prefill backends"
if py <<'PY' 2>/dev/null; then ok "vllm: FlashKDA extension (vllm._flashkda_C) built"; else echo "  WARN  vllm: _flashkda_C absent -- optional extension skipped at build; --kda-prefill-backend flashkda falls back to triton"; fi
import importlib; importlib.import_module("vllm._flashkda_C")
PY
if py <<'PY' 2>/dev/null; then ok "b12x: b12x.sequence.kda_prefill importable"; else echo "  NOTE  b12x: kda_prefill module absent (pre-20260903 b12x)"; fi
import importlib; importlib.import_module("b12x.sequence.kda_prefill")
PY

# Flavor detection: cu133 images (NGC foundation) carry NCCL under
# /opt/local-inference/nccl/lib and no cuBLAS overlay; cu132 images carry
# /opt/libnccl-local-inference.so.2.30.4 and the overlay symlink.
if sh 'test -d /opt/local-inference/nccl/lib' 2>/dev/null; then FLAVOR=cu133; EXPECTED_CUDA=13.3; else FLAVOR=cu132; EXPECTED_CUDA=13.2; fi
echo "== generic image contract (flavor: $FLAVOR, expect CUDA $EXPECTED_CUDA)"
docker run --rm --entrypoint /opt/venv/bin/python -e EXPECTED_CUDA="$EXPECTED_CUDA" "$IMAGE" - <<'PY' && ok "aarch64, torch 2.13.0, CUDA $EXPECTED_CUDA, b12x + humming importable" || bad "generic contract (arch/torch/CUDA/kernels)"
import platform, torch, importlib.util, os
assert platform.machine() == "aarch64", platform.machine()
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.version.cuda.startswith(os.environ["EXPECTED_CUDA"]), torch.version.cuda
assert importlib.util.find_spec("b12x") and (importlib.util.find_spec("humming_kernels") or importlib.util.find_spec("humming"))
PY
# vLLM custom-op extension: this lineage ships the ops in _C_stable_libtorch;
# vllm._C may legitimately be absent (the upstream build asserts only the
# stable-libtorch extension and cumem_allocator). Report, do not fail.
if py <<'PY' 2>/dev/null; then ok "vllm: _C_stable_libtorch + cumem_allocator extensions importable"; else bad "vllm: _C_stable_libtorch / cumem_allocator missing"; fi
import importlib
importlib.import_module("vllm._C_stable_libtorch"); importlib.import_module("vllm.cumem_allocator")
PY
if py <<'PY' 2>/dev/null; then echo "  INFO  vllm._C present (legacy ops extension)"; else echo "  INFO  vllm._C absent (expected on this lineage; ops live in _C_stable_libtorch)"; fi
import importlib; importlib.import_module("vllm._C")
PY
sh 'test -f /usr/local/bin/serve-glm53-flash-nvfp4.sh' && ok "GLM-5.3 launcher present" || bad "serve-glm53-flash-nvfp4.sh missing (wrong vLLM branch?)"
if [[ "$FLAVOR" == cu132 ]]; then
  sh 'test -f /opt/libnccl-local-inference.so.2.30.4 && test -f /usr/local/cuda/compat/libcuda.so.1' && ok "patched NCCL 2.30.4 + CUDA compat shim present" || bad "NCCL/compat shim missing"
  sh 'readlink -f /usr/local/cuda/targets/sbsa-linux/lib/libcublas.so.13 | grep -q /usr/lib/aarch64-linux-gnu/' && ok "cuBLAS overlay resolves into aarch64 multiarch" || bad "cuBLAS overlay not applied"
else
  nccl_so="$(sh 'ls /opt/local-inference/nccl/lib/libnccl.so.2.* 2>/dev/null | head -1')"
  [[ -n "$nccl_so" ]] && ok "patched NCCL present: $nccl_so" || bad "patched NCCL missing under /opt/local-inference/nccl/lib"
  compat="$(sh 'ls /usr/local/cuda/compat/lib.real/libcuda.so.1 /usr/local/cuda/compat/libcuda.so.1 2>/dev/null | head -1')"
  [[ -n "$compat" ]] && ok "CUDA forward-compat libcuda present: $compat (use this path in LD_PRELOAD if the host driver predates CUDA $EXPECTED_CUDA)" || echo "  NOTE  no forward-compat libcuda found under /usr/local/cuda/compat; the host driver must satisfy CUDA $EXPECTED_CUDA natively"
  echo "  NOTE  cu133 lineage: no cuBLAS overlay (NGC base ships its own cuBLAS); rank env needs the cu133 loader block"
fi
sh 'env | grep -q "^VLLM_PCIE_ALLREDUCE_BACKEND=b12x$"' && ok "baked VLLM_PCIE_ALLREDUCE_BACKEND=b12x" || echo "  NOTE  baked PCIe backend env not b12x (launcher overrides with -e anyway)"

echo "== result: $pass passed, $fail failed"
[[ "$fail" == 0 ]]
