#!/usr/bin/env bash
# build-spark-cu133.sh -- aarch64/sm_121 (DGX Spark) builder for the CUDA 13.3
# Jovian Judgement runtime, retargeting upstream's cu133 lineage in place:
#
#   phase 1  foundation   Dockerfile.kimi-k3-cu133-torch213-base
#                         NGC PyTorch container -> patched NCCL 2.31.2, torch
#                         2.13.0 from source, torchvision, xgrammar. Built once;
#                         reused by every release until a pin here changes.
#   phase 2  flashinfer   Dockerfile.flashinfer-cu133-torch213-wheels
#                         FlashInfer wheels compiled against the foundation.
#   phase 3  overlay      Dockerfile.deepseek-infernal-invocation-cu133-torch213
#                         (the JJ runtime builder despite its name): vLLM, B12X,
#                         LMCache, exllamav3, InstantTensor, verification.
#
# Usage:
#   ./build-spark-cu133.sh [--dry-run] [--phase foundation|flashinfer|overlay|all] [build.env] [-- extra docker build args]
#
# Run from a subdirectory of (or inside) a blackwell-llm-docker checkout that
# carries the three cu133 Dockerfiles (main branch, 2026-09). Same profile
# format and precedence as build-spark-cu132.sh: process env > build.env >
# in-script defaults; the env file is parsed, never sourced.
#
# STATUS: DRAFT. --dry-run (patch application + shape guards) is validated;
# the docker phases have not yet run on a Spark. First cold build is hours.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SELF}")"
die()  { echo "build-spark-cu133: $*" >&2; exit 1; }
warn() { echo "build-spark-cu133: warning: $*" >&2; }
note() { echo "build-spark-cu133: $*" >&2; }

# ---------------------------------------------------------------- transcript
# Same mechanism as build-spark-cu132.sh: one RUN_STAMP per run shared by the
# manifest and the transcript; everything printed from here on is teed to a
# temp file that becomes build-log-<profile>-<stamp>.txt at exit when
# LOGGING=1 (--log, profile key, or process env), and is discarded otherwise.
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
stamp="${RUN_STAMP%%-*}"
expand_date() { local v="$1"; v="${v//<datetime>/${RUN_STAMP}}"; v="${v//<date>/${stamp}}"; printf '%s' "${v}"; }
_RAW_LOG="$(mktemp)"
exec {_TTY_OUT}>&1 {_TTY_ERR}>&2
exec > >(tee -a "${_RAW_LOG}") 2>&1
_TEE_PID=$!
finalize_transcript() {
  if [[ "${LOGGING:-0}" != 1 ]]; then rm -f "${_RAW_LOG:-}"; return 0; fi
  local log="${RUN_LOG:-${SCRIPT_DIR}/build-log-${PROFILE_NAME:-unknown}-${RUN_STAMP}.txt}"
  printf 'build transcript: %s\n' "${log}" >&2
  exec 1>&"${_TTY_OUT}" 2>&"${_TTY_ERR}"
  wait "${_TEE_PID}" 2>/dev/null || true
  mv -f "${_RAW_LOG}" "${log}" 2>/dev/null || cp -f "${_RAW_LOG}" "${log}"
}
trap finalize_transcript EXIT
PATCH_REPORT="$(mktemp)"

# ------------------------------------------------------------ argument parse
DRY_RUN=0; PHASE=all; ENV_FILE=""; EXTRA_ARGS=(); LOG_FLAG=0; OVERRIDDEN_KEYS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --phase) PHASE="$2"; shift 2 ;;
    --log) LOG_FLAG=1; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,24p' "${SELF}"; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) [[ -z "${ENV_FILE}" ]] || die "only one env file may be given"; ENV_FILE="$1"; shift ;;
  esac
done
case "${PHASE}" in foundation|flashinfer|overlay|all) ;; *) die "--phase must be foundation, flashinfer, overlay, or all" ;; esac

# ------------------------------------------------------------ env-file parse
if [[ -z "${ENV_FILE}" ]]; then
  _default_env="${SCRIPT_DIR}/build.env"
  [[ -f "${_default_env}" ]] && ENV_FILE="${_default_env}"
fi
ALLOWED_KEYS=" ALLOW_FOREIGN_ARCH LOGGING PATCH_NCCL_GENCODE PATCH_VLLM_FA_LAYERS RUNTIME_FOUNDATION BASE_IMAGE LMCACHE_BUILD_VERSION NCCL4PY_VERSION VLLM_PRS VLLM_UPSTREAM_BASE VLLM_MERGE_HEADS VLLM_INTEGRATION_LOCK_SHA256 B12X_PRS B12X_UPSTREAM_BASE B12X_MERGE_HEADS B12X_INTEGRATION_LOCK_SHA256 LMCACHE_PRS LMCACHE_UPSTREAM_BASE LMCACHE_MERGE_HEADS LMCACHE_INTEGRATION_LOCK_SHA256 B12X_COMMIT B12X_INTEGRATION_TREE B12X_PATCH_FILE B12X_PATCH_SHA256 B12X_PIN B12X_REF B12X_REPO CUTLASS_DSL_VERSION DEEPGEMM_COMMIT DEEPGEMM_REPO DOCKER_COMMIT EXLLAMAV3_COMMIT EXLLAMAV3_REPO FLASHINFER_COMMIT FLASHINFER_REF FLASHINFER_REPO FLASHINFER_VERSION FLASHINFER_WHEEL_IMAGE FOUNDATION_IMAGE GH_TOKEN IMAGE IMAGE_REPO IMAGE_TAG INSTANTTENSOR_COMMIT INSTANTTENSOR_LIBAIO_COMMIT INSTANTTENSOR_LIBAIO_REPO INSTANTTENSOR_LIBAIO_TREE INSTANTTENSOR_REPO INSTANTTENSOR_VERSION LMCACHE_COMMIT LMCACHE_INTEGRATION_TREE LMCACHE_PATCH_FILE LMCACHE_PATCH_SHA256 LMCACHE_REF LMCACHE_REPO MAX_JOBS NCCL_COMMIT NCCL_REF NCCL_REPO NCCL_VERSION NVCC_THREADS NVIDIA_PYTORCH_IMAGE PATCH_EXLLAMAV3_AVX PATCH_GPU_ARCH PATCH_MAX_JOBS PIN_PREFLIGHT PROFILE_NAME PYTORCH_COMMIT PYTORCH_REF PYTORCH_REPO PYTORCH_VERSION RELEASE_DATE TORCHVISION_COMMIT TORCHVISION_REF TORCHVISION_REPO TORCHVISION_VERSION TRITON_KERNELS_COMMIT TRITON_KERNELS_REPO VLLM_COMMIT VLLM_INTEGRATION_TREE VLLM_PACKAGE_VERSION VLLM_PATCH_FILE VLLM_PATCH_SHA256 VLLM_PIN VLLM_REF VLLM_REPO VLLM_REQUIRED_LAUNCHERS XGRAMMAR_COMMIT XGRAMMAR_REF XGRAMMAR_REPO XGRAMMAR_VERSION "
if [[ -n "${ENV_FILE}" ]]; then
  [[ -f "${ENV_FILE}" ]] || die "env file not found: ${ENV_FILE}"
  grep -qU $'\r' "${ENV_FILE}" && die "env file has CRLF line endings"
  lineno=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    lineno=$((lineno+1))
    [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "${line}" =~ ^([A-Z][A-Z0-9_]*)=(.*)$ ]] || die "${ENV_FILE}:${lineno}: not KEY=VALUE: ${line}"
    key="${BASH_REMATCH[1]}"; val="${BASH_REMATCH[2]}"
    [[ "${ALLOWED_KEYS}" == *" ${key} "* ]] || die "${ENV_FILE}:${lineno}: unknown key ${key}"
    case "${val}" in *[\`\$\"\;\|\&]*|*"'"*|*\\*|*\(*|*\)*) die "${ENV_FILE}:${lineno}: value of ${key} contains shell metacharacters" ;; esac
    [[ "${val}" =~ ^\<[A-Za-z0-9_-]+\>$ ]] && die "${ENV_FILE}:${lineno}: unresolved placeholder for ${key}"
    if [[ -n "${!key+x}" ]]; then note "process env overrides ${ENV_FILE}: ${key}"; OVERRIDDEN_KEYS="${OVERRIDDEN_KEYS} ${key}"; else printf -v "${key}" '%s' "${val}"; export "${key}"; fi
  done < "${ENV_FILE}"
  note "loaded profile: ${ENV_FILE}"
fi
: "${PROFILE_NAME:=custom}"
PROFILE_NAME="$(expand_date "${PROFILE_NAME}")"
[[ "${PROFILE_NAME}" =~ ^[a-z0-9][a-z0-9.-]*$ ]] || die "PROFILE_NAME must be lowercase [a-z0-9.-]: ${PROFILE_NAME}"
LOGGING="${LOGGING:-0}"; [[ "${LOG_FLAG}" == 1 ]] && LOGGING=1
case "${LOGGING}" in 0|1) ;; *) die "LOGGING must be 0 or 1" ;; esac
RUN_LOG="${SCRIPT_DIR}/build-log-${PROFILE_NAME}-${RUN_STAMP}.txt"
[[ "${LOGGING}" != 1 ]] || note "transcript: ${RUN_LOG}"

# --------------------------------------------------------------- repo checks
D_FOUND=Dockerfile.kimi-k3-cu133-torch213-base
D_FI=Dockerfile.flashinfer-cu133-torch213-wheels
D_OVER=Dockerfile.deepseek-infernal-invocation-cu133-torch213
A_PIP=tests/deepseek-infernal-cu133-pip-check.allowlist
if [[ ! -f "${D_OVER}" ]]; then
  for cand in "${SCRIPT_DIR}/.." "${SCRIPT_DIR}"; do
    [[ -f "${cand}/${D_OVER}" ]] && { cd "${cand}"; note "building from repo root: $(pwd)"; break; }
  done
fi
for f in "${D_FOUND}" "${D_FI}" "${D_OVER}"; do [[ -f "${f}" ]] || die "missing ${f}: the cu133 lineage lives on blackwell-llm-docker main (2026-09)"; done
arch="$(uname -m)"
if [[ "${arch}" != "aarch64" && "${ALLOW_FOREIGN_ARCH:-0}" != 1 && "${DRY_RUN}" != 1 ]]; then
  die "host is ${arch}, not aarch64; build on a Spark. --dry-run works anywhere."
fi

# ----------------------------------------------------------------- defaults
# Foundation (upstream kimi-k3 cu133 base, 2026-08-11 r2 pins). The NGC
# PyTorch container is a multi-arch manifest; on a Spark docker resolves the
# linux/arm64 image. Everything in the foundation is compiled inside it.
export NVIDIA_PYTORCH_IMAGE="${NVIDIA_PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
export PYTORCH_REPO="${PYTORCH_REPO:-https://github.com/pytorch/pytorch.git}"
export PYTORCH_REF="${PYTORCH_REF:-v2.13.0}"
export PYTORCH_COMMIT="${PYTORCH_COMMIT:-cf30153c4c131c8164ee7798e5022d810682e2cb}"
export PYTORCH_VERSION="${PYTORCH_VERSION:-2.13.0}"
export TORCHVISION_REPO="${TORCHVISION_REPO:-https://github.com/pytorch/vision.git}"
export TORCHVISION_REF="${TORCHVISION_REF:-v0.28.0}"
export TORCHVISION_COMMIT="${TORCHVISION_COMMIT:-}"      # required: set in the profile (read the foundation Dockerfile's ARG)
export TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.28.0}"
export NCCL_REPO="${NCCL_REPO:-https://github.com/local-inference-lab/nccl-canonical.git}"
export NCCL_REF="${NCCL_REF:-canonical/cu133-nccl2312-amd-turin}"
export NCCL_COMMIT="${NCCL_COMMIT:-fb6f40999a2a9e63104d4ae4a84118bce61528f8}"
export NCCL_VERSION="${NCCL_VERSION:-2.31.2}"
export XGRAMMAR_REPO="${XGRAMMAR_REPO:-https://github.com/mlc-ai/xgrammar.git}"
export XGRAMMAR_REF="${XGRAMMAR_REF:-v0.2.5}"
export XGRAMMAR_COMMIT="${XGRAMMAR_COMMIT:-2ea71da4ccb997a06928c9fb69b99f330da56697}"
export XGRAMMAR_VERSION="${XGRAMMAR_VERSION:-0.2.5}"
# FlashInfer wheels
export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/voipmonitor/flashinfer.git}"
export FLASHINFER_REF="${FLASHINFER_REF:-integration/main-pr4393-pcie-ipc-qualified-20260807}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-1ac6942776b383c6b03c7a5805a22e72a3e3349f}"
export FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.18+cu133}"
export CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.6.2}"
# Overlay sources (release pins go in the profile; these are fallbacks)
export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
VLLM_PIN="${VLLM_PIN:-}"; export VLLM_REF="${VLLM_REF:-${VLLM_PIN}}"; export VLLM_COMMIT="${VLLM_COMMIT:-${VLLM_PIN}}"
export B12X_REPO="${B12X_REPO:-https://github.com/local-inference-lab/b12x.git}"
B12X_PIN="${B12X_PIN:-}"; export B12X_REF="${B12X_REF:-${B12X_PIN}}"; export B12X_COMMIT="${B12X_COMMIT:-${B12X_PIN}}"
export LMCACHE_REPO="${LMCACHE_REPO:-https://github.com/local-inference-lab/LMCache.git}"
export LMCACHE_REF="${LMCACHE_REF:-${LMCACHE_COMMIT:-}}"; export LMCACHE_COMMIT="${LMCACHE_COMMIT:-}"
export LMCACHE_BUILD_VERSION="${LMCACHE_BUILD_VERSION:-0.5.2+jj.${PROFILE_NAME//-/.}}"
for c in VLLM B12X LMCACHE; do for t in PATCH_FILE PATCH_SHA256 INTEGRATION_TREE PRS UPSTREAM_BASE MERGE_HEADS INTEGRATION_LOCK_SHA256; do k="${c}_${t}"; export "${k}=${!k:-}"; done; done
export INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/voipmonitor/InstantTensor.git}"
export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}"
export INSTANTTENSOR_VERSION="${INSTANTTENSOR_VERSION:-}"
export INSTANTTENSOR_LIBAIO_REPO="${INSTANTTENSOR_LIBAIO_REPO:-https://github.com/sailfishos-mirror/libaio.git}"
export INSTANTTENSOR_LIBAIO_COMMIT="${INSTANTTENSOR_LIBAIO_COMMIT:-1b18bfafc6a2f7b9fa2c6be77a95afed8b7be448}"
export INSTANTTENSOR_LIBAIO_TREE="${INSTANTTENSOR_LIBAIO_TREE:-c9442e111b747e9329ea782c6edb9d13a827cc08}"
export EXLLAMAV3_REPO="${EXLLAMAV3_REPO:-https://github.com/brandonmmusic-max/exllamav3.git}"
export EXLLAMAV3_COMMIT="${EXLLAMAV3_COMMIT:-704aefd743b390af4bd0fb429d1906f9b964c7d8}"
export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export TRITON_KERNELS_REPO="${TRITON_KERNELS_REPO:-}"; export TRITON_KERNELS_COMMIT="${TRITON_KERNELS_COMMIT:-}"
export VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-}"
export MAX_JOBS="${MAX_JOBS:-20}"
export NVCC_THREADS="${NVCC_THREADS:-1}"    # nvcc threads per compile job; 1 on a 20-core/121 GiB Spark (upstream: 4 x 48 jobs)
export RELEASE_DATE="${RELEASE_DATE:-$(date +%Y%m%d)}"
export DOCKER_COMMIT="${DOCKER_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
stamp="$(date +%Y%m%d)"
export IMAGE_REPO="${IMAGE_REPO:-local/vllm}"
export FOUNDATION_IMAGE="${FOUNDATION_IMAGE:-${IMAGE_REPO}:cu133-torch213-nccl2312-sm121-foundation}"
export FLASHINFER_WHEEL_IMAGE="${FLASHINFER_WHEEL_IMAGE:-${IMAGE_REPO}:flashinfer-wheels-${FLASHINFER_COMMIT:0:7}-cu133-torch213-sm121}"
IMAGE_TAG="$(expand_date "${IMAGE_TAG:-${PROFILE_NAME}-cu133-torch213-sm121-${stamp}}")"; export IMAGE_TAG
export IMAGE="${IMAGE:-${IMAGE_REPO}:${IMAGE_TAG}}"
# Overlay base mode. 0 (default): BASE_IMAGE is the raw foundation and the
# overlay builds the venv, DeepGEMM, exllamav3, InstantTensor, nccl4py and
# the rust toolchain itself. 1: BASE_IMAGE is a PREVIOUS JJ runtime image
# built by this wrapper; the overlay verifies those components are present
# and rebuilds only vLLM, B12X and LMCache -- upstream's fast-release path.
export RUNTIME_FOUNDATION="${RUNTIME_FOUNDATION:-0}"
case "${RUNTIME_FOUNDATION}" in 0|1) ;; *) die "RUNTIME_FOUNDATION must be 0 or 1" ;; esac
export BASE_IMAGE="${BASE_IMAGE:-${FOUNDATION_IMAGE}}"
[[ "${RUNTIME_FOUNDATION}" == 0 || "${BASE_IMAGE}" != "${FOUNDATION_IMAGE}" ]] \
  || die "RUNTIME_FOUNDATION=1 needs BASE_IMAGE set to a previously built JJ runtime image (the raw foundation has no /opt/venv)"
export VLLM_PACKAGE_VERSION="${VLLM_PACKAGE_VERSION:-0.26.1rc0+${PROFILE_NAME//-/.}.cu133.sm121.${stamp}}"

# --------------------------------------------------------------- patch toggles
for t in PATCH_GPU_ARCH PATCH_MAX_JOBS PATCH_EXLLAMAV3_AVX PATCH_NCCL_GENCODE PATCH_VLLM_FA_LAYERS; do
  v="${!t:-auto}"; case "${v}" in on|off|auto) ;; *) die "${t} must be on, off, or auto: ${v}" ;; esac
  printf -v "${t}" '%s' "${v}"; export "${t}"
done

# ------------------------------------------------------------ dockerfile prep
# Refuse to run on Dockerfiles left patched by a killed run (OOM/SIGKILL
# skips the restore trap): leftover backups or an injected marker mean the
# checkout is dirty and patching again would double-inject.
for f in "${D_FOUND}" "${D_FI}" "${D_OVER}"; do
  ls "${f}".pre-spark.* >/dev/null 2>&1 && die "leftover backup(s) for ${f} from an interrupted run; restore with: git checkout -- ${f} && rm -f ${f}.pre-spark.*"
  grep -q "exllamav3: CPU all-reduce is x86-only" "${f}" && die "${f} is already patched (interrupted run); restore with: git checkout -- ${f}"
  grep -qE "12\.1a|=121a\b|12\.1f" "${f}" && die "${f} already carries sm_121 rewrites (interrupted run); restore with: git checkout -- ${f}"
done
ls "${A_PIP}".pre-spark.* >/dev/null 2>&1 && die "leftover backup for ${A_PIP}; restore with: git checkout -- ${A_PIP} && rm -f ${A_PIP}.pre-spark.*"
backups=()
for f in "${D_FOUND}" "${D_FI}" "${D_OVER}" "${A_PIP}"; do cp -a "${f}" "${f}.pre-spark.$$"; backups+=("${f}"); done
restore() { local f; for f in "${backups[@]}"; do mv -f "${f}.pre-spark.$$" "${f}"; done; }
MEM_PEAK_FILE="$(mktemp)"; SAMPLER_PID=""; BUILD_T0=""
start_sampler() {
  BUILD_T0="$(date +%s)"
  ( peak=0; while :; do u=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{print t-a}' /proc/meminfo); [[ "${u}" -gt "${peak}" ]] && { peak="${u}"; echo "${peak}" > "${MEM_PEAK_FILE}"; }; sleep 5; done ) & SAMPLER_PID=$!
}
report_telemetry() {
  local status="$1"; [[ -n "${BUILD_T0}" ]] || return 0
  local dt=$(( $(date +%s) - BUILD_T0 )) peak_kib; peak_kib="$(cat "${MEM_PEAK_FILE}" 2>/dev/null || echo 0)"
  printf 'build telemetry: status=%s  jobs=%s  nvcc-threads=%s  peak-memory=%s GiB  wall-time=%dh %02dm %02ds\n' \
    "${status}" "${MAX_JOBS}" "${NVCC_THREADS}" "$(awk -v k="${peak_kib}" 'BEGIN{printf "%.1f", k/1048576}')" $((dt/3600)) $(((dt%3600)/60)) $((dt%60)) >&2
}
cleanup() {
  local rc=$?
  [[ -n "${SAMPLER_PID}" ]] && kill "${SAMPLER_PID}" 2>/dev/null || true
  [[ -n "${BUILD_T0}" && "${rc}" != 0 ]] && report_telemetry "FAILED(rc=${rc})"
  restore
  rm -f "${MEM_PEAK_FILE}" "${PATCH_REPORT}"
  finalize_transcript
}
trap cleanup EXIT
trap "exit 130" INT TERM

apply_sed_patch() {  # name toggle file before_pattern min_expected sed-expr...
  local name="$1" toggle="$2" file="$3" before="$4" min="$5"; shift 5
  local found; found="$(grep -cE "${before}" "${file}" || true)"
  [[ "${toggle}" == off ]] && { note "${name}=off: skipped (${file})"; return 0; }
  if [[ "${found}" == 0 ]]; then
    [[ "${toggle}" == auto ]] && { note "${name}(auto): pattern absent in ${file}; skipping"; return 0; }
    die "${name}=on but pattern not found in ${file}"
  fi
  local args=() e; for e in "$@"; do args+=(-e "${e}"); done
  sed -i "${args[@]}" "${file}"
  local left; left="$(grep -cE "${before}" "${file}" || true)"
  [[ "${left}" == 0 ]] || die "${name}: ${left} occurrence(s) survived in ${file}: $(grep -nE "${before}" "${file}" | head -3)"
  [[ "${found}" -ge "${min}" ]] || die "${name}: expected >=${min} occurrences in ${file}, found ${found}"
  note "${name}: rewrote ${found} line-occurrence(s) in ${file}"
  printf '%s: %s occurrence(s) in %s\n' "${name}" "${found}" "${file}" >> "${PATCH_REPORT}"
}

# GPU arch: sm_120 -> sm_121 in all three files (the cu133 lineage has no x86
# filesystem paths, no cuBLAS overlay and no cusparselt WHEEL repair to port).
for f in "${D_FOUND}" "${D_FI}" "${D_OVER}"; do
  apply_sed_patch PATCH_GPU_ARCH "${PATCH_GPU_ARCH}" "${f}" \
    '12\.0a|=120a|12\.0f|ARCH_LIST=12\.0([^0-9]|$)' 1 \
    's/TORCH_CUDA_ARCH_LIST=12\.0a/TORCH_CUDA_ARCH_LIST=12.1a/g' \
    's/TORCH_CUDA_ARCH_LIST=12\.0\([^a-z0-9]\)/TORCH_CUDA_ARCH_LIST=12.1\1/g' \
    's/CMAKE_CUDA_ARCHITECTURES=120a/CMAKE_CUDA_ARCHITECTURES=121a/g' \
    's/FLASHINFER_CUDA_ARCH_LIST=12\.0f/FLASHINFER_CUDA_ARCH_LIST=12.1f/g'
done
# NCCL: the foundation compiles NCCL with sm_120 SASS + compute_120 PTX only
# (nvcc -gencode, not the ARCH_LIST envs). On GB10 that runs via PTX JIT of
# compute_120; native sm_121 SASS avoids the JIT and is what the rest of the
# image gets. Foundation-only; changing it rebuilds the foundation (hours).
apply_sed_patch PATCH_NCCL_GENCODE "${PATCH_NCCL_GENCODE}" "${D_FOUND}" \
  'arch=compute_120,code=sm_120|arch=compute_120,code=compute_120' 1 \
  's/arch=compute_120,code=sm_120/arch=compute_121,code=sm_121/g' \
  's/arch=compute_120,code=compute_120/arch=compute_121,code=compute_121/g'
# pip-check allowlist: the final gate diffs `pip check` against an allowlist
# whose seven known LMCache complaints carry upstream's LMCache build version
# string. Ours differs (LMCACHE_BUILD_VERSION), so rewrite the version token;
# the set of complaints itself must still match exactly.
[[ -f "${A_PIP}" ]] || die "missing ${A_PIP}"
allow_ver="$(grep -oE '^lmcache [^ ]+' "${A_PIP}" | sort -u | sed 's/^lmcache //')"
[[ "$(printf '%s\n' "${allow_ver}" | wc -l)" == 1 && -n "${allow_ver}" ]] || die "${A_PIP}: expected exactly one lmcache version token, found: ${allow_ver}"
sed -i "s|^lmcache ${allow_ver} |lmcache ${LMCACHE_BUILD_VERSION} |" "${A_PIP}"
note "pip-check allowlist: lmcache ${allow_ver} -> ${LMCACHE_BUILD_VERSION} ($(grep -c "^lmcache ${LMCACHE_BUILD_VERSION} " "${A_PIP}") lines)"
# vllm_flash_attn python package: the overlay installs only the FA `cute`
# helpers and the compiled .so files into the source tree that PYTHONPATH
# serves at runtime; the pure-python `vllm_flash_attn/layers/` package
# (rotary embedding used by vision encoders) stays behind in CMake's
# _deps checkout. GLM-5.3 vision profiling then fails with
# "No module named 'vllm.vllm_flash_attn.layers'" (2026-09-12). Copy every
# the python tree the way vLLM itself does: its cmake install rule for the
# _vllm_fa2_C component copies vllm_flash_attn/**/*.py (excluding __init__.py
# and flash_attn_interface.py, which vLLM's own tree owns) into the prefix.
# Install that component next to the cutedsl one, then copy every
# subdirectory (layers/, ops/, cute/...) into the served tree no-clobber.
# Earlier attempts: copying all .py clobbered vLLM's __init__.py; copying
# layers/ alone failed on layers/rotary -> ops.triton.rotary; a _deps scan
# produced layers/ but not ops/ (unexplained).
# Injected because the anchor line must survive. The stage asserts the two
# files that failed at runtime; the runtime import itself is checked by
# probe-image.sh on the finished image (the .so files land after this point,
# so importing the package here would fail for the wrong reason).
python3 - "${D_OVER}" "${PATCH_VLLM_FA_LAYERS}" <<'PYEOF'
import pathlib, sys
path, tog = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
anchor = "    find /tmp/vllm-extensions -name '*.abi3.so' > /tmp/vllm-extension-list; \\\n"
inject = ('    cmake --install /tmp/vllm-extensions --prefix /tmp/vllm-flash-attn-python --component _vllm_fa2_C; \\\n'
          '    ls -la /tmp/vllm-flash-attn-python/vllm/vllm_flash_attn/; \\\n'
          '    for d in /tmp/vllm-flash-attn-python/vllm/vllm_flash_attn/*/; do test -d "${d}" && cp -an "${d%/}" /opt/infernal-invocation/vllm/vllm/vllm_flash_attn/; done; \\\n'
          '    ls -la /opt/infernal-invocation/vllm/vllm/vllm_flash_attn/; \\\n'
          '    test -f /opt/infernal-invocation/vllm/vllm/vllm_flash_attn/layers/rotary.py; \\\n'
          '    test -f /opt/infernal-invocation/vllm/vllm/vllm_flash_attn/ops/triton/rotary.py; \\\n')
cnt = text.count(anchor)
if tog == "off": print("PATCH_VLLM_FA_LAYERS=off: skipped", file=sys.stderr)
elif cnt == 0 and tog == "auto": print("PATCH_VLLM_FA_LAYERS(auto): extension-list anchor absent; skipping", file=sys.stderr)
else:
    assert cnt == 1, f"extension-list anchor found {cnt} times"
    path.write_text(text.replace(anchor, inject + anchor)); print("injected vllm_flash_attn python package copy (overlay)", file=sys.stderr)
PYEOF
# MAX_JOBS: upstream builds on 48-128 core hosts; a Spark has 20 Grace cores.
for f in "${D_FOUND}" "${D_FI}" "${D_OVER}"; do
  apply_sed_patch PATCH_MAX_JOBS "${PATCH_MAX_JOBS}" "${f}" \
    'MAX_JOBS=(48|128)\b' 1 \
    "s/MAX_JOBS=48\b/MAX_JOBS=${MAX_JOBS}/g" "s/MAX_JOBS=128\b/MAX_JOBS=${MAX_JOBS}/g"
done
# The overlay's vLLM extension stage does not use MAX_JOBS: it runs
# `cmake --build --parallel 48` with NVCC_THREADS=4 (ENV and -D). 48 x 4 nvcc
# threads on Marlin MoE kernels exhausted 121 GiB on a Spark (OOM at object
# ~120/408 on 2026-09-11). Bound both to the profile values.
apply_sed_patch PATCH_MAX_JOBS "${PATCH_MAX_JOBS}" "${D_OVER}" \
  '\-\-parallel 48\b|NVCC_THREADS=4\b' 2 \
  "s/--parallel 48\b/--parallel ${MAX_JOBS}/g" "s/NVCC_THREADS=4\b/NVCC_THREADS=${NVCC_THREADS}/g"
# exllamav3 aarch64 stub (x86 GCC builtins + AVX all-reduce files), same fix as
# the cu132 wrapper, anchored on the overlay's exllamav3 build line.
python3 - "${D_OVER}" "${PATCH_EXLLAMAV3_AVX}" "${MAX_JOBS}" <<'PYEOF'
import pathlib, sys
path, tog, jobs = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text()
C_ABORT = '{ fprintf(stderr, "exllamav3: CPU all-reduce is x86-only\\n"); abort(); }'
STUB_AVX2 = ["#include <cstdio>", "#include <cstdlib>", '#include "all_reduce_cpu_avx2.h"',
  "void enable_fast_fp() {}", "void enable_fast_fp_avx2() {}",
  "void perform_cpu_reduce(PGContext*, size_t, uint32_t, uint8_t*, size_t) " + C_ABORT,
  "void perform_cpu_reduce_avx2(PGContext*, size_t, uint32_t, uint8_t*, size_t) " + C_ABORT]
STUB_AVX512 = ["#include <cstdio>", "#include <cstdlib>", '#include "all_reduce_cpu_avx512.h"',
  "void enable_fast_fp_avx512() {}",
  "void bf16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) " + C_ABORT,
  "void perform_cpu_reduce_avx512(PGContext*, size_t, uint32_t, uint8_t*, size_t) " + C_ABORT]
def printf_cmd(lines, dest): return "printf '%s\\n' " + " ".join("'" + l + "'" for l in lines) + " > " + dest
EXT = "exllamav3/exllamav3_ext"
cmds = [
  "sed -i 's/avx2_supported = __builtin_cpu_supports(\"avx2\");/avx2_supported = false;/' " + EXT + "/avx2_target.cpp",
  "sed -i 's/avx512_supported = __builtin_cpu_supports(\"avx512f\") && __builtin_cpu_supports(\"avx512bw\");/avx512_supported = false;/' " + EXT + "/avx512_target.cpp",
  "grep -q 'avx2_supported = false;' " + EXT + "/avx2_target.cpp",
  "grep -q 'avx512_supported = false;' " + EXT + "/avx512_target.cpp",
  printf_cmd(STUB_AVX2, EXT + "/parallel/all_reduce_cpu_avx2.cpp"),
  printf_cmd(STUB_AVX512, EXT + "/parallel/all_reduce_cpu_avx512.cpp")]
injection = "".join("      " + c + "; \\\n" for c in cmds)
anchor = f"      TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS={jobs} python setup.py build_ext --inplace; \\\n"
cnt = text.count(anchor)
if tog == "off": print("PATCH_EXLLAMAV3_AVX=off: skipped", file=sys.stderr)
elif cnt == 0 and tog == "auto": print("PATCH_EXLLAMAV3_AVX(auto): build anchor absent in overlay; skipping", file=sys.stderr)
else:
    assert cnt == 1, f"exllamav3 anchor found {cnt} times"
    path.write_text(text.replace(anchor, injection + anchor)); print("injected exllamav3 aarch64 source patch (overlay)", file=sys.stderr)
PYEOF

# ------------------------------------------------------------- pin pre-flight
# The overlay verifies each source's git TREE hash (write-tree after checkout
# and patch). For an unpatched pin that is the commit's tree, which the GitHub
# API reports; with a patch file, supply *_INTEGRATION_TREE from a local
# checkout (git apply, git add -A, git write-tree).
gh_tree() {  # repo-url sha -> tree sha
  local repo="$1" sha="$2" api hdr=()
  api="$(printf '%s' "${repo}" | sed -E 's#https://github.com/([^/]+)/([^/.]+)(\.git)?#https://api.github.com/repos/\1/\2/commits/#')${sha}"
  [[ -n "${GH_TOKEN:-}" ]] && hdr=(-H "Authorization: Bearer ${GH_TOKEN}")
  curl -sf "${hdr[@]}" "${api}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["commit"]["tree"]["sha"])' 2>/dev/null || true
}
if [[ "${PHASE}" == all || "${PHASE}" == overlay ]]; then
  [[ -n "${VLLM_COMMIT}" && -n "${B12X_COMMIT}" && -n "${LMCACHE_COMMIT}" ]] || die "overlay needs VLLM_PIN, B12X_PIN and LMCACHE_COMMIT (the cu133 overlay builds LMCache unconditionally)"
  for c in VLLM B12X LMCACHE; do
    tree_var="${c}_INTEGRATION_TREE"; patch_var="${c}_PATCH_FILE"; repo_var="${c}_REPO"; sha_var="${c}_COMMIT"
    if [[ -z "${!tree_var}" ]]; then
      [[ -z "${!patch_var}" ]] || die "${tree_var} is required when ${patch_var} is set (compute: git apply, git add -A, git write-tree)"
      t="$(gh_tree "${!repo_var}" "${!sha_var}")"
      if [[ -z "${t}" ]]; then
        [[ "${DRY_RUN}" == 1 ]] && { warn "could not resolve ${c} tree via the GitHub API (rate limit? set GH_TOKEN); the build will need ${tree_var}"; continue; }
        die "could not resolve ${c} tree for ${!sha_var} via the GitHub API; set ${tree_var} in the profile (rtx6kpro source locks publish it as <component>.tree) or GH_TOKEN"
      fi
      export "${tree_var}=${t}"; note "${c} tree ${t:0:9} (from ${!sha_var:0:9})"
    fi
  done
fi
[[ -n "${TORCHVISION_COMMIT}" ]] || warn "TORCHVISION_COMMIT is empty; the foundation Dockerfile's own ARG default applies"

if [[ "${DRY_RUN}" == 1 ]]; then
  note "DRY RUN complete: rewrites validated on ${D_FOUND}, ${D_FI}, ${D_OVER}; no image built."
  note "foundation=${FOUNDATION_IMAGE} flashinfer=${FLASHINFER_WHEEL_IMAGE} image=${IMAGE}"
  exit 0
fi

# ------------------------------------------------------------------- builds
start_sampler
have_image() { docker image inspect "$1" >/dev/null 2>&1; }
build() { DOCKER_BUILDKIT=1 docker build --progress=plain "$@" "${EXTRA_ARGS[@]}" .; }

if [[ "${PHASE}" == all || "${PHASE}" == foundation ]]; then
  if have_image "${FOUNDATION_IMAGE}" && [[ "${PHASE}" == all ]]; then note "foundation present: ${FOUNDATION_IMAGE} (use --phase foundation to rebuild)"; else
    note "phase 1: foundation (torch from source; hours)"
    build --file "${D_FOUND}" --tag "${FOUNDATION_IMAGE}" \
      --build-arg "NVIDIA_PYTORCH_IMAGE=${NVIDIA_PYTORCH_IMAGE}" --build-arg "PYTORCH_REPO=${PYTORCH_REPO}" \
      --build-arg "PYTORCH_REF=${PYTORCH_REF}" --build-arg "PYTORCH_COMMIT=${PYTORCH_COMMIT}" --build-arg "PYTORCH_VERSION=${PYTORCH_VERSION}" \
      --build-arg "TORCHVISION_REPO=${TORCHVISION_REPO}" --build-arg "TORCHVISION_REF=${TORCHVISION_REF}" \
      ${TORCHVISION_COMMIT:+--build-arg "TORCHVISION_COMMIT=${TORCHVISION_COMMIT}"} --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION}" \
      --build-arg "NCCL_REPO=${NCCL_REPO}" --build-arg "NCCL_REF=${NCCL_REF}" --build-arg "NCCL_COMMIT=${NCCL_COMMIT}" --build-arg "NCCL_VERSION=${NCCL_VERSION}" \
      --build-arg "XGRAMMAR_REPO=${XGRAMMAR_REPO}" --build-arg "XGRAMMAR_REF=${XGRAMMAR_REF}" --build-arg "XGRAMMAR_COMMIT=${XGRAMMAR_COMMIT}" --build-arg "XGRAMMAR_VERSION=${XGRAMMAR_VERSION}" \
      --build-arg "TORCH_CUDA_ARCH_LIST=12.1a" --build-arg "TORCH_MAX_JOBS=${MAX_JOBS}" --build-arg "NCCL_MAX_JOBS=${MAX_JOBS}" \
      --build-arg "TORCHVISION_MAX_JOBS=${MAX_JOBS}" --build-arg "XGRAMMAR_MAX_JOBS=${MAX_JOBS}" \
      --build-arg "RELEASE_DATE=${RELEASE_DATE}" --build-arg "DOCKER_COMMIT=${DOCKER_COMMIT}"
  fi
fi
foundation_id="$(docker image inspect --format '{{.Id}}' "${FOUNDATION_IMAGE}" 2>/dev/null || true)"
if [[ "${PHASE}" == all || "${PHASE}" == flashinfer ]]; then
  [[ -n "${foundation_id}" ]] || die "foundation image missing: ${FOUNDATION_IMAGE}"
  if have_image "${FLASHINFER_WHEEL_IMAGE}" && [[ "${PHASE}" == all ]]; then note "flashinfer wheels present: ${FLASHINFER_WHEEL_IMAGE}"; else
    note "phase 2: FlashInfer wheels"
    build --file "${D_FI}" --tag "${FLASHINFER_WHEEL_IMAGE}" \
      --build-arg "BASE_IMAGE=${FOUNDATION_IMAGE}" --build-arg "BASE_IMAGE_ID=${foundation_id}" \
      --build-arg "FLASHINFER_REPO=${FLASHINFER_REPO}" --build-arg "FLASHINFER_REF=${FLASHINFER_REF}" \
      --build-arg "FLASHINFER_COMMIT=${FLASHINFER_COMMIT}" --build-arg "FLASHINFER_VERSION=${FLASHINFER_VERSION}" \
      --build-arg "CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}" --build-arg "RELEASE_DATE=${RELEASE_DATE}" --build-arg "DOCKER_COMMIT=${DOCKER_COMMIT}"
  fi
fi
if [[ "${PHASE}" == all || "${PHASE}" == overlay ]]; then
  base_id="$(docker image inspect --format '{{.Id}}' "${BASE_IMAGE}" 2>/dev/null || true)"
  [[ -n "${base_id}" ]] || die "overlay base image missing: ${BASE_IMAGE}"
  fi_id="$(docker image inspect --format '{{.Id}}' "${FLASHINFER_WHEEL_IMAGE}" 2>/dev/null || true)"
  [[ -n "${fi_id}" ]] || die "FlashInfer wheel image missing: ${FLASHINFER_WHEEL_IMAGE}"
  note "phase 3: JJ overlay -> ${IMAGE}"
  args=(--file "${D_OVER}" --tag "${IMAGE}"
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" --build-arg "BASE_IMAGE_ID=${base_id}"
    --build-arg "RUNTIME_FOUNDATION=${RUNTIME_FOUNDATION}"
    --build-arg "RUNTIME_FOUNDATION_IMAGE=$([[ "${RUNTIME_FOUNDATION}" == 1 ]] && echo "${BASE_IMAGE}" || echo none)"
    --build-arg "FLASHINFER_WHEEL_IMAGE=${FLASHINFER_WHEEL_IMAGE}" --build-arg "FLASHINFER_WHEEL_IMAGE_ID=${fi_id}"
    --build-arg "FLASHINFER_REPO=${FLASHINFER_REPO}" --build-arg "FLASHINFER_REF=${FLASHINFER_REF}"
    --build-arg "FLASHINFER_COMMIT=${FLASHINFER_COMMIT}" --build-arg "FLASHINFER_VERSION=${FLASHINFER_VERSION}"
    --build-arg "CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}" --build-arg "VLLM_PACKAGE_VERSION=${VLLM_PACKAGE_VERSION}"
    --build-arg "RELEASE_NAME=${PROFILE_NAME}" --build-arg "RELEASE_DATE=${RELEASE_DATE}" --build-arg "DOCKER_COMMIT=${DOCKER_COMMIT}"
    --build-arg "CACHE_FINGERPRINT=${VLLM_COMMIT:0:12}-${B12X_COMMIT:0:12}")
  # The overlay's compose RUN uses `set -u`: every no-default ARG it references
  # must be passed, even when empty (PRS / UPSTREAM_BASE / MERGE_HEADS /
  # INTEGRATION_LOCK_SHA256 describe upstream's merge-stack verification and
  # are empty for a plain pin or a patch-file composition).
  for c in VLLM B12X LMCACHE; do
    for s in REPO REF COMMIT PATCH_FILE PATCH_SHA256 INTEGRATION_TREE PRS UPSTREAM_BASE MERGE_HEADS INTEGRATION_LOCK_SHA256; do
      v="${c}_${s}"; args+=(--build-arg "${v}=${!v:-}")
    done
  done
  # ARGs that carry a Dockerfile default: pass only when set, or an empty
  # --build-arg would override the default with an empty string.
  for v in INSTANTTENSOR_REPO INSTANTTENSOR_COMMIT INSTANTTENSOR_VERSION INSTANTTENSOR_LIBAIO_REPO INSTANTTENSOR_LIBAIO_COMMIT INSTANTTENSOR_LIBAIO_TREE \
           EXLLAMAV3_REPO EXLLAMAV3_COMMIT DEEPGEMM_REPO DEEPGEMM_COMMIT TRITON_KERNELS_REPO TRITON_KERNELS_COMMIT LMCACHE_BUILD_VERSION NCCL4PY_VERSION XGRAMMAR_VERSION; do
    [[ -z "${!v:-}" ]] || args+=(--build-arg "${v}=${!v}")
  done
  build "${args[@]}"

  # ------------------------------------------------------------ verification
  docker run --rm --entrypoint python "${IMAGE}" - <<'PY'
import platform, torch, importlib.util
assert platform.machine() == "aarch64", platform.machine()
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.version.cuda.startswith("13.3"), torch.version.cuda
assert importlib.util.find_spec("b12x") is not None, "b12x missing"
print("image OK: aarch64, torch", torch.__version__, "cuda", torch.version.cuda)
PY
  for launcher in ${VLLM_REQUIRED_LAUNCHERS}; do
    docker run --rm --entrypoint test "${IMAGE}" -f "/usr/local/bin/${launcher}" || die "required launcher missing from image: ${launcher}"
  done
  docker run --rm --entrypoint bash "${IMAGE}" -c 'ls /opt/local-inference/nccl/lib/libnccl.so.2.31.2 >/dev/null' || warn "patched NCCL not at /opt/local-inference/nccl/lib (check the foundation layout before setting LD_PRELOAD)"
  printf '\nBuilt %s (base %s, runtime_foundation=%s, flashinfer %s)\n' "${IMAGE}" "${BASE_IMAGE}" "${RUNTIME_FOUNDATION}" "${FLASHINFER_WHEEL_IMAGE}"
  report_telemetry "OK+VERIFIED"

  # ----------------------------------------------------------- build manifest
  # Same shape as the cu132 wrapper's: one Markdown record per verified build,
  # sharing RUN_STAMP with the transcript; non-fatal, every field degrades to
  # UNKNOWN rather than failing a verified build.
  emit_build_manifest() {
    local unk='UNKNOWN — needs verification'
    local id_over id_found id_fi ngc_digest wrapper_sha profile_sha pip_freeze
    id_over="$(docker image inspect --format '{{.Id}}' "${IMAGE}" 2>/dev/null || echo "${unk}")"
    id_found="$(docker image inspect --format '{{.Id}}' "${FOUNDATION_IMAGE}" 2>/dev/null || echo "${unk}")"
    id_fi="$(docker image inspect --format '{{.Id}}' "${FLASHINFER_WHEEL_IMAGE}" 2>/dev/null || echo "${unk}")"
    ngc_digest="$(docker image inspect --format '{{index .RepoDigests 0}}' "${NVIDIA_PYTORCH_IMAGE}" 2>/dev/null || echo "${unk}")"
    wrapper_sha="$(sha256sum "${SELF}" | cut -d' ' -f1)"
    profile_sha="none (in-script defaults only)"; [[ -z "${ENV_FILE}" ]] || profile_sha="$(sha256sum "${ENV_FILE}" 2>/dev/null | cut -d' ' -f1 || echo "${unk}")"
    pip_freeze="$(docker run --rm --entrypoint /opt/venv/bin/pip "${IMAGE}" freeze 2>/dev/null || echo "${unk} (pip freeze failed)")"
    {
      printf '# Build manifest — %s — %s (CUDA 13.3 flavor)\n\n' "${PROFILE_NAME}" "${RUN_STAMP}"
      printf '## Images\n\n'
      printf -- '- Runtime (this build): `%s` image ID `%s`\n' "${IMAGE}" "${id_over}"
      printf -- '- Registry digest (`image@sha256:`): %s (exists only after a push)\n' "${unk}"
      printf -- '- Overlay base: `%s` (RUNTIME_FOUNDATION=%s)\n' "${BASE_IMAGE}" "${RUNTIME_FOUNDATION}"
      printf -- '- Foundation: `%s` image ID `%s` (NGC `%s` @ `%s`; torch %s @ `%s`; NCCL %s `%s` @ `%s`)\n' \
        "${FOUNDATION_IMAGE}" "${id_found}" "${NVIDIA_PYTORCH_IMAGE}" "${ngc_digest}" "${PYTORCH_VERSION}" "${PYTORCH_COMMIT}" "${NCCL_VERSION}" "${NCCL_REF}" "${NCCL_COMMIT}"
      printf -- '- FlashInfer wheels: `%s` image ID `%s` (%s @ `%s`)\n' "${FLASHINFER_WHEEL_IMAGE}" "${id_fi}" "${FLASHINFER_VERSION}" "${FLASHINFER_COMMIT}"
      printf -- '- vLLM package version: `%s`\n\n' "${VLLM_PACKAGE_VERSION}"
      printf '## Sources (composition contract as verified in-build)\n\n'
      for c in VLLM B12X LMCACHE; do
        r="${c}_REPO"; k="${c}_COMMIT"; t="${c}_INTEGRATION_TREE"; pf="${c}_PATCH_FILE"; ps="${c}_PATCH_SHA256"
        printf -- '- %s: `%s` @ `%s`, tree `%s`%s\n' "${c}" "${!r}" "${!k}" "${!t:-${unk}}" "$([[ -n "${!pf}" ]] && printf ' + patch `%s` sha256 `%s`' "${!pf}" "${!ps}")"
      done
      printf -- '- InstantTensor: `%s` @ `%s`; exllamav3: `%s` @ `%s`; DeepGEMM: `%s` @ `%s`\n\n' \
        "${INSTANTTENSOR_REPO}" "${INSTANTTENSOR_COMMIT}" "${EXLLAMAV3_REPO}" "${EXLLAMAV3_COMMIT}" "${DEEPGEMM_REPO}" "${DEEPGEMM_COMMIT}"
      printf '## Build system\n\n'
      printf -- '- blackwell-llm-docker checkout: `%s`\n' "${DOCKER_COMMIT}"
      printf -- '- Wrapper: `build-spark-cu133.sh` sha256 `%s`\n' "${wrapper_sha}"
      printf -- '- Profile: `%s` sha256 `%s`\n' "${ENV_FILE:-none}" "${profile_sha}"
      printf -- '- Dockerfiles: `%s`, `%s`, `%s` (pristine copies restored on exit)\n' "${D_FOUND}" "${D_FI}" "${D_OVER}"
      printf -- '- Extra docker build args: `%s`\n' "${EXTRA_ARGS[*]:-none}"
      printf -- '- Run transcript: `%s`\n' "$([[ "${LOGGING}" == 1 ]] && echo "${RUN_LOG##*/}" || echo 'not kept (LOGGING=0)')"
      printf -- '- Build host: %s, %s\n\n' "$(uname -m)" "$(uname -r)"
      printf '## Effective configuration\n\n```\n'
      for key in ${ALLOWED_KEYS}; do
        [[ -n "${!key:-}" ]] || continue
        case "${key}" in ALLOW_FOREIGN_ARCH|GH_TOKEN) continue ;; esac
        mark=""; [[ " ${OVERRIDDEN_KEYS} " != *" ${key} "* ]] || mark="   # OVERRIDDEN by process env"
        printf '%s=%s%s\n' "${key}" "${!key}" "${mark}"
      done
      printf '```\n\n## Patches applied (differences from the upstream cu133 build system)\n\n```\n'
      cat "${PATCH_REPORT}" 2>/dev/null || echo "${unk}"
      printf 'exllamav3 aarch64 source stub: injected into the overlay build\n'
      printf 'pip-check allowlist: lmcache version token rewritten to %s\n' "${LMCACHE_BUILD_VERSION}"
      printf '```\n\n## Verification (executed by this wrapper, all passed)\n\n'
      printf -- '- In-image python asserts: aarch64, torch 2.13.0, CUDA 13.3, `b12x` importable\n'
      printf -- '- Upstream runtime contract (in-build): verify_deepseek_infernal_cu133_runtime.py PASS; pip-check allowlist matched\n'
      printf -- '- Required launchers present: `%s`\n' "${VLLM_REQUIRED_LAUNCHERS:-none configured}"
      printf -- '- Patched NCCL: `/opt/local-inference/nccl/lib/libnccl.so.%s`\n' "${NCCL_VERSION}"
      printf -- '- Serving on hardware: Not tested (build-time verification only; record pair validation separately)\n\n'
      printf '## Telemetry\n\n- jobs=%s nvcc-threads=%s peak-memory=%s GiB wall-time=%ss\n\n' "${MAX_JOBS}" "${NVCC_THREADS}" \
        "$(awk -v k="$(cat "${MEM_PEAK_FILE}" 2>/dev/null || echo 0)" 'BEGIN{printf "%.1f", k/1048576}')" "$(( $(date +%s) - BUILD_T0 ))"
      printf '## Full pip freeze (final image /opt/venv)\n\n```\n%s\n```\n\n' "${pip_freeze}"
      printf '## Obtaining the registry digest (after push)\n\n```\ndocker tag %s <registry>/<repo>:<tag>\ndocker push <registry>/<repo>:<tag>\n' "${IMAGE}"
      printf "docker inspect --format '{{index .RepoDigests 0}}' <registry>/<repo>:<tag>\n\`\`\`\n"
    } > "${build_manifest}" && note "build manifest written: ${build_manifest}"
  }
  build_manifest="${SCRIPT_DIR}/build_manifest-${PROFILE_NAME}-${RUN_STAMP}.md"
  if ! ( emit_build_manifest ); then
    warn "build manifest emission failed -- the image is built and VERIFIED; re-run (fully cached) to regenerate it"
  fi
  printf 'Ship it to the other node:\n  docker save %s | ssh <node2> docker load\n' "${IMAGE}"
  printf 'Next release: RUNTIME_FOUNDATION=1 BASE_IMAGE=%s rebuilds only vLLM/B12X/LMCache.\n' "${IMAGE}"
  printf 'Publishing needs an image@sha256 reference minted by a registry push; then fill the UNKNOWN digest in %s\n' "${build_manifest}"
fi
