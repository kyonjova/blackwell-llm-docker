#!/usr/bin/env bash
# verify-image-cu132.sh -- prove a built serving image contains what its
# profile says, before spending a two-node boot on it. This is THE image
# verification step: build-spark-cu132.sh builds only and defers to this.
#
# Usage:  ./verify-image-cu132.sh IMAGE [build.env]
#
# With a profile, VLLM_PIN / B12X_PIN are compared against what the image
# reports, and TORCH_VERSION / VLLM_REQUIRED_LAUNCHERS set the expectations
# for the generic contract. Checks, in order: source identity, the RoCEnante
# composition (vLLM PR #597 + b12x PR #295), KDA, DeepGEMM, LMCache, SparkCache, io_uring + the
# baked seccomp profile (image checks, then HOST notes), then the generic
# image contract.
set -euo pipefail
IMAGE="${1:?usage: verify-image-cu132.sh IMAGE [build.env]}"
ENV_FILE="${2:-}"
# -i is REQUIRED: without it docker does not attach the heredoc to the
# container's stdin, `python -` reads EOF, runs nothing and exits 0 -- every
# py check would PASS without executing a line.
py() { docker run -i --rm --entrypoint /opt/venv/bin/python "$IMAGE" - "$@"; }
sh() { docker run --rm --entrypoint bash "$IMAGE" -c "$1"; }
pass=0; fail=0
ok()   { echo "  PASS  $*"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail+1)); }
prof() { [[ -n "$ENV_FILE" ]] && grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- || true; }
want_vllm="$(prof VLLM_PIN)"; want_b12x="$(prof B12X_PIN)"
# Contract expectations: from the profile when given, else the cu132 line.
torch_full="$(prof TORCH_VERSION)"; torch_full="${torch_full:-2.13.0+cu132}"
want_torch="${torch_full%%+*}"; _cu="${torch_full##*+cu}"; want_cuda="${_cu:0:2}.${_cu:2}"
want_launchers="$(prof VLLM_REQUIRED_LAUNCHERS)"
want_launchers="${want_launchers:-serve-glm53-flash-nvfp4.sh}"

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

echo "== DeepGEMM (DS4 path)"
# deep_gemm is only exercised on the DS4 path, so a missing shared library
# surfaces at serve time, not at build time. KK's vllm-project/DeepGEMM links
# libdw.so.1 (elfutils); the final stage is FROM system-base and gets it only
# via the PATCH_DEEPGEMM_LIBDW runtime install. ldd runs with torch/lib on
# LD_LIBRARY_PATH: torch extensions have no RPATH to it and resolve
# libc10/libtorch_* only because `import torch` loaded them first, so without
# this every torch lib is a false "not found".
py <<'PY' && ok "deep_gemm imports; every native extension resolves (libdw.so.1 present)" || bad "deep_gemm import or shared-library resolution failed (see output above)"
import glob, os, subprocess, torch, deep_gemm
dg_dir = os.path.dirname(deep_gemm.__file__)
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
env = dict(os.environ, LD_LIBRARY_PATH=":".join(
    p for p in (torch_lib, os.environ.get("LD_LIBRARY_PATH", "")) if p))
sos = glob.glob(os.path.join(dg_dir, "**", "*.so"), recursive=True)
assert sos, f"no native extension under {dg_dir}"
for so in sos:
    out = subprocess.run(["ldd", so], capture_output=True, text=True, env=env).stdout
    missing = [l.strip() for l in out.splitlines() if "not found" in l]
    assert not missing, f"{so}: unresolved {missing}"
PY

echo "== LMCache (pin + KK connector imports)"
want_lmc="$(prof LMCACHE_BUILD_VERSION)"
img_lmc="$(label local-inference.lmcache.commit)"
echo "  lmcache commit label: ${img_lmc:-(none)}    profile LMCACHE_COMMIT: $(prof LMCACHE_COMMIT)"
WANT_LMC="$want_lmc" docker run -i --rm -e WANT_LMC --entrypoint /opt/venv/bin/python "$IMAGE" - <<'PY' \
  && ok "lmcache ${want_lmc:-(unpinned)} installed; MP server + LMCacheMPConnector + LMCacheRecurrentCheckpointConnector import against this vLLM" \
  || bad "LMCache version mismatch or connector import failure (see output above)"
import importlib.metadata as md, os
want = os.environ.get("WANT_LMC", "")
have = md.version("lmcache")
print("  lmcache", have)
assert not want or have == want, f"installed {have} != profile {want}"
import lmcache.v1.multiprocess.server  # noqa: F401
import lmcache.integration.vllm.lmcache_mp_connector  # noqa: F401
import lmcache.integration.vllm.recurrent_checkpoint_connector  # noqa: F401
PY

echo "== SparkCache (PATCH_SPARKCACHE)"
img_sc="$(label org.local-inference.sparkcache.commit)"
want_sc="$(prof SPARKCACHE_COMMIT)"
want_sc_patches="$(prof SPARKCACHE_VLLM_PATCHES)"
if [[ -z "$img_sc" ]]; then
  if [[ -n "$want_sc" && "$(prof PATCH_SPARKCACHE)" != off ]]; then
    bad "profile pins SparkCache ${want_sc:0:12} but the image has no sparkcache label (built without PATCH_SPARKCACHE?)"
  else
    echo "  NOTE  SparkCache not in this image (no org.local-inference.sparkcache.commit label)"
  fi
else
  sc_patches="$(label org.local-inference.sparkcache.vllm-patches)"
  echo "  sparkcache commit: ${img_sc:0:12}"
  echo "  vLLM patches (image label): ${sc_patches:-(none)}"
  [[ -z "$want_sc" || "$img_sc" == "$want_sc" ]] && ok "sparkcache commit matches the profile" \
    || bad "sparkcache commit ${img_sc:0:12} != profile ${want_sc:0:12}"
  [[ -z "$want_sc_patches" || "$sc_patches" == "$want_sc_patches" ]] && ok "applied vLLM patch list matches SPARKCACHE_VLLM_PATCHES" \
    || bad "image applied [$sc_patches] but the profile lists [$want_sc_patches]"
  SC_PATCHES="$sc_patches" docker run -i --rm -e SC_PATCHES --entrypoint /opt/venv/bin/python "$IMAGE" - <<'PY' \
    && ok "SparkContextCacheConnector imports; installed vLLM code matches the labelled patches" \
    || bad "SparkCache connector import or vLLM patch state wrong (see output above)"
import inspect, os
from sparkcache.spark_context_cache_connector import SparkContextCacheConnector  # noqa: F401
from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
labelled = os.environ.get("SC_PATCHES", "")
has_vmm = "SparkContextCacheConnector" in inspect.getsource(VllmConfig._verify_kv_transfer_compat)
has_040 = hasattr(KVCacheManager, "attach_shared_prefix_lease")
has_041 = hasattr(Scheduler, "_finalize_shared_prefix_leases")
print(f"  vmm-exemption={has_vmm}  lease(040)={has_040}  attach(041)={has_041}")
assert has_vmm == ("vmm-exemption" in labelled), "vmm-exemption state does not match the image label"
# 041 without 040 crashes the scheduler on its first step: they must agree.
assert has_040 == has_041 == ("shared-prefix" in labelled), "040/041 must match the label, both or neither"
PY
fi

echo "== io_uring loader path (PATCH_IO_URING)"
# The b12x loader's Spark read path compiles against liburing at first use;
# without it: "io_uring bounce support is unavailable".
sh 'pkg-config --exists liburing' && ok "liburing $(sh 'pkg-config --modversion liburing' 2>/dev/null) visible to pkg-config" \
  || bad "liburing-dev missing: LOAD_FORMAT=b12x dies at load (rebuild with PATCH_IO_URING=auto|on)"
seccomp_path="$(label org.local-inference.seccomp-profile)"; seccomp_sha="$(label org.local-inference.seccomp-profile.sha256)"
if [[ -z "$seccomp_path" ]]; then
  echo "  NOTE  no seccomp profile baked (org.local-inference.seccomp-profile label absent)"
else
  echo "  seccomp profile: $seccomp_path  sha256 ${seccomp_sha:0:12}  from $(label org.local-inference.seccomp-profile.source)"
  sh "echo '$seccomp_sha  $seccomp_path' | sha256sum -c - >/dev/null" && ok "baked seccomp profile matches its label sha256" \
    || bad "baked seccomp profile missing or does not match its label"
fi

# HOST checks -- informational: they describe the machine running this probe,
# not the image. An image cannot apply seccomp to itself; the runtime installs
# the filter before the entrypoint. Serving needs io_uring allowed by the
# daemon default (no --security-opt in any launcher).
host_prof="$(docker info --format '{{range .SecurityOptions}}{{println .}}{{end}}' 2>/dev/null | sed -n 's/^name=seccomp,profile=//p')"
echo "  HOST  docker default seccomp: ${host_prof:-(none reported)}"
if [[ -n "$seccomp_sha" && -f "$host_prof" ]]; then
  if [[ "$(sha256sum "$host_prof" | cut -d' ' -f1)" == "$seccomp_sha" ]]; then
    echo "  HOST  daemon default profile == the image's baked profile"
  else
    echo "  HOST  daemon default profile DIFFERS from the image's baked profile (re-extract after a pin bump if the upstream profile changed)"
  fi
fi
# io_uring_setup is syscall 425 on aarch64 and x86_64. NULL params: an
# allowed kernel answers EFAULT; seccomp (or io_uring_disabled) answers EPERM.
uring="$(docker run --rm --entrypoint /opt/venv/bin/python "$IMAGE" -c '
import ctypes, errno
libc = ctypes.CDLL(None, use_errno=True); libc.syscall(425, 0, 0)
print("BLOCKED" if ctypes.get_errno() == errno.EPERM else "OK")' 2>/dev/null || echo ERROR)"
case "$uring" in
  OK) echo "  HOST  io_uring ALLOWED in a default container -- launchers need no --security-opt on this host" ;;
  *BLOCKED*)
    echo "  HOST  io_uring BLOCKED in a default container (EPERM). One-time fix per node (restarts dockerd):"
    if [[ -n "$seccomp_path" ]]; then
      echo "          sudo mkdir -p /etc/docker/seccomp"
      echo "          docker run --rm --entrypoint cat $IMAGE $seccomp_path | sudo tee /etc/docker/seccomp/spark-io-uring.json >/dev/null"
    fi
      cat <<'EOT'
          sudo jq --arg p /etc/docker/seccomp/spark-io-uring.json '. + {"seccomp-profile": $p}' \
            /etc/docker/daemon.json > /tmp/daemon.json && sudo mv /tmp/daemon.json /etc/docker/daemon.json
          sudo systemctl restart docker
        and confirm: sysctl kernel.io_uring_disabled  (must be 0)
EOT
      ;;
  *) echo "  HOST  io_uring probe inconclusive: $uring" ;;
esac

echo "== generic image contract"
EXPECT_TORCH="$want_torch" EXPECT_CUDA="$want_cuda" \
  docker run -i --rm -e EXPECT_TORCH -e EXPECT_CUDA --entrypoint /opt/venv/bin/python "$IMAGE" - <<'PY' \
  && ok "aarch64, torch ${want_torch}, CUDA ${want_cuda}, b12x + humming importable" || bad "generic contract (expected torch ${want_torch}, CUDA ${want_cuda})"
import os, platform, torch, importlib.util
assert platform.machine() == "aarch64", platform.machine()
assert torch.__version__.startswith(os.environ["EXPECT_TORCH"]), torch.__version__
assert torch.version.cuda == os.environ["EXPECT_CUDA"], torch.version.cuda
assert importlib.util.find_spec("b12x") and (importlib.util.find_spec("humming_kernels") or importlib.util.find_spec("humming"))
PY
for l in $want_launchers; do
  sh "test -x /usr/local/bin/$l" && ok "launcher present: $l" || bad "launcher missing: $l (wrong vLLM branch?)"
done
sh 'test -f /opt/libnccl-local-inference.so.2.30.4 && test -f /usr/local/cuda/compat/libcuda.so.1' && ok "patched NCCL + CUDA compat shim present" || bad "NCCL/compat shim missing"
sh 'readlink -f /usr/local/cuda/targets/sbsa-linux/lib/libcublas.so.13 | grep -q /usr/lib/aarch64-linux-gnu/' && ok "cuBLAS overlay resolves into aarch64 multiarch" || bad "cuBLAS overlay not applied"
sh 'env | grep -q "^VLLM_PCIE_ALLREDUCE_BACKEND=b12x$"' && ok "baked VLLM_PCIE_ALLREDUCE_BACKEND=b12x" || echo "  NOTE  baked PCIe backend env not b12x (launcher overrides with -e anyway)"

echo "== result: $pass passed, $fail failed"
[[ "$fail" == 0 ]]
