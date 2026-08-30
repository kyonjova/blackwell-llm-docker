#!/usr/bin/env bash
# build-spark-cu132.sh -- generic aarch64/sm_121 (DGX Spark) image builder for
# the local-inference-lab vLLM + B12X/SparkInfer serving stack.
#
# Usage:
#   ./build-spark-cu132.sh [--dry-run] [build.env] [-- extra docker build args]
#
# Run from inside a checkout of local-inference-lab/blackwell-llm-docker.
# Precedence: process environment > build.env file > in-script defaults.
# See BUILD-README.md for components, patches, safeguards, and attribution.
set -euo pipefail

# Absolute path to this script, captured before any cd (used by the frozen
# hash self-check and --help after we move to the repo root).
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SELF}")"

die()  { echo "build-spark-cu132: $*" >&2; exit 1; }
warn() { echo "build-spark-cu132: warning: $*" >&2; }
note() { echo "build-spark-cu132: $*" >&2; }

# ------------------------------------------------------------ argument parse
DRY_RUN=0
ENV_FILE=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,10p' "${SELF}"; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) [[ -z "${ENV_FILE}" ]] || die "only one env file may be given"; ENV_FILE="$1"; shift ;;
  esac
done

# ------------------------------------------------------------ env-file parse
# Parsed, never sourced: shell sourcing strips quotes and executes content.
# Only whitelisted KEY=VALUE lines are accepted; values may not contain shell
# metacharacters that indicate quoting or injection mistakes.
# Default profile: a build.env sitting next to this script.
if [[ -z "${ENV_FILE}" ]]; then
  _default_env="${SCRIPT_DIR}/build.env"
  [[ -f "${_default_env}" ]] && { ENV_FILE="${_default_env}"; }
fi

ALLOWED_KEYS=" ALLOW_FOREIGN_ARCH B12X_COMMIT B12X_PIN B12X_REF B12X_REPO BUILD_BASE_IMAGE_TAG CUTLASS_COMMIT CUTLASS_DSL_VERSION CUTLASS_REF DEEPGEMM_COMMIT DEEPGEMM_REF FASTSAFETENSORS_SPEC FLASHINFER_BUILD_CUBIN FLASHINFER_COMMIT FLASHINFER_REF FLASHINFER_REPO FROZEN_ACK HUMMING_KERNELS_SPEC IMAGE INSTANTTENSOR_COMMIT INSTANTTENSOR_REF INSTANTTENSOR_REPO LAUNCHER_COMMIT LAUNCHER_REF LAUNCHER_REPO MAX_JOBS NCCL_COMMIT NCCL_REF NCCL_REPO NVCC_THREADS PATCH_EXLLAMAV3_AVX PATCH_GPU_ARCH PATCH_HOST_ARCH PATCH_PCIE_ENV PATCH_PIPCHECK_WHEELTAG PATCH_VLLM_REQ_MARKERS PIN_SOURCE_COMMITS PROFILE_NAME QUACK_KERNELS_SPEC SPARKINFER_COMMIT SPARKINFER_REF SPARKINFER_REPO SYSTEM_BASE_IMAGE TILELANG_VERSION TOKENSPEED_MLA_VERSION TORCHVISION_VERSION TORCH_BUNDLED_NCCL_VERSION TORCH_VERSION TVM_FFI_VERSION VLLM_BUILD_VERSION VLLM_COMMIT VLLM_MAX_JOBS VLLM_NVCC_THREADS VLLM_PIN VLLM_REF VLLM_REPO VLLM_REQUIRED_LAUNCHERS VLLM_RUNTIME_EXTRA_PACKAGES XGRAMMAR_COMMIT XGRAMMAR_REF XGRAMMAR_TRANSFORMERS5_COMPAT XGRAMMAR_VERSION "
if [[ -n "${ENV_FILE}" ]]; then
  [[ -f "${ENV_FILE}" ]] || die "env file not found: ${ENV_FILE}"
  if grep -qU $'\r' "${ENV_FILE}"; then die "env file has CRLF line endings: sed -i 's/\r$//' ${ENV_FILE}"; fi
  lineno=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    lineno=$((lineno+1))
    [[ "${line}" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "${line}" =~ ^([A-Z][A-Z0-9_]*)=(.*)$ ]] \
      || die "${ENV_FILE}:${lineno}: not KEY=VALUE: ${line}"
    key="${BASH_REMATCH[1]}"; val="${BASH_REMATCH[2]}"
    [[ "${ALLOWED_KEYS}" == *" ${key} "* ]] \
      || die "${ENV_FILE}:${lineno}: unknown key ${key} (typo? see ALLOWED_KEYS in this script)"
    case "${val}" in
      *[\`\$\"\;\|\&]*|*"'"*|*\\*|*\(*|*\)*)
        die "${ENV_FILE}:${lineno}: value of ${key} contains shell metacharacters; this file is parsed, not sourced -- write values unquoted" ;;
    esac
    if [[ "${val}" =~ ^\<[A-Za-z0-9_-]+\>$ ]]; then die "${ENV_FILE}:${lineno}: unresolved placeholder for ${key}"; fi
    if [[ -n "${!key+x}" ]]; then
      note "process env overrides ${ENV_FILE}: ${key}"
    else
      printf -v "${key}" '%s' "${val}"
      export "${key}"
    fi
  done < "${ENV_FILE}"
  note "loaded profile: ${ENV_FILE}"
fi

: "${PROFILE_NAME:=custom}"
[[ "${PROFILE_NAME}" =~ ^[a-z0-9][a-z0-9.-]*$ ]] || die "PROFILE_NAME must be lowercase [a-z0-9.-]: ${PROFILE_NAME}"

# -------------------------------------------------------------- repo checks
# This wrapper lives in a subdirectory (e.g. dgx-spark-builder/) of a
# blackwell-llm-docker checkout, or is copied into the repo root. Locate the
# repo root automatically so it can be invoked from anywhere.
if [[ ! -f Dockerfile.vllm-b12x-cu132 ]]; then
  for cand in "${SCRIPT_DIR}/.." "${SCRIPT_DIR}"; do
    if [[ -f "${cand}/Dockerfile.vllm-b12x-cu132" ]]; then
      cd "${cand}"; note "building from repo root: $(pwd)"; break
    fi
  done
fi
[[ -f Dockerfile.vllm-b12x-cu132 && -x ./build-vllm-b12x-cu132.sh ]] || {
  echo "cannot find the blackwell-llm-docker build system; place this script" >&2
  echo "in a subdirectory of (or copy it into) a checkout of" >&2
  echo "  https://github.com/local-inference-lab/blackwell-llm-docker" >&2
  exit 1
}
arch="$(uname -m)"
if [[ "${arch}" != "aarch64" && "${ALLOW_FOREIGN_ARCH:-0}" != 1 && "${DRY_RUN}" != 1 ]]; then
  die "host is ${arch}, not aarch64; build on a Spark or set ALLOW_FOREIGN_ARCH=1 (qemu, slow). --dry-run works anywhere."
fi

# ---------------------------------------------------------------- sources
# In-script values are FALLBACK DEFAULTS (last combination qualified by this
# wrapper's maintainers); the profile (build.env) is the source of truth and
# should set every pin explicitly -- see the full manifest it ships with.
# Toolchain pins follow the infernal-invocation r2 (20260812) qualified
# combination the sm121-era images were cut from.

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
# dev/jovian-judgement moves several times a day; the vllm-build stage does
# `git checkout "${VLLM_REF}"` and THEN verifies the commit, so a branch name
# here races against upstream pushes mid-build (the branch advancing between
# ref resolution and the clone fails the build ~40 minutes in). Pin the sha
# as BOTH the ref and the commit: checkout of a sha is immune to branch
# movement, and the verify step becomes a tautology. To move the pin:
#   git ls-remote https://github.com/local-inference-lab/vllm.git dev/jovian-judgement
# then override VLLM_PIN (or VLLM_REF/VLLM_COMMIT individually).
# Fallback pin; profiles override (provenance notes belong in the profile).
VLLM_PIN="${VLLM_PIN:-da4d7be6c97434f6942292ed8abbf4b32dc44355}"
export VLLM_REF="${VLLM_REF:-${VLLM_PIN}}"
export VLLM_COMMIT="${VLLM_COMMIT:-${VLLM_PIN}}"
export LAUNCHER_REPO="${LAUNCHER_REPO:-${VLLM_REPO}}"
export LAUNCHER_REF="${LAUNCHER_REF:-${VLLM_REF}}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-${VLLM_COMMIT}}"
# Space-separated serve scripts that MUST land in the image (a cheap "is
# this the right branch for my model" guard). Set per profile.
export VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-}"

# B12X was renamed SparkInfer; the Docker build args keep the legacy name.
# Pinned the same way as vLLM (the b12x stage also checks out the ref before
# verifying the commit, so a moving branch races mid-build).
export B12X_REPO="${SPARKINFER_REPO:-${B12X_REPO:-https://github.com/local-inference-lab/sparkinfer.git}}"
B12X_PIN="${B12X_PIN:-2fcf23a0ce269be27b2e03fece73d46e90e6aeea}"
export B12X_REF="${SPARKINFER_REF:-${B12X_REF:-${B12X_PIN}}}"
export B12X_COMMIT="${SPARKINFER_COMMIT:-${B12X_COMMIT:-${B12X_PIN}}}"

export NCCL_REPO="${NCCL_REPO:-https://github.com/local-inference-lab/nccl-canonical.git}"
export NCCL_REF="${NCCL_REF:-canonical/cu132-nccl2304-amd-noxml}"
export NCCL_COMMIT="${NCCL_COMMIT:-dfab7c1ace32da250ba97757879429c341b7bcf9}"

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/voipmonitor/flashinfer.git}"
export FLASHINFER_REF="${FLASHINFER_REF:-integration/main-pr4393-pcie-ipc-qualified-20260807}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-1ac6942776b383c6b03c7a5805a22e72a3e3349f}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

# ---------------------------------------------------------------- toolchain
# The vLLM branch pins torch==2.13.0 in requirements/cuda.txt.
export TORCH_VERSION="${TORCH_VERSION:-2.13.0+cu132}"
export TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.28.0+cu132}"
export TORCH_BUNDLED_NCCL_VERSION="${TORCH_BUNDLED_NCCL_VERSION:-2.29.7}"
export CUTLASS_REF="${CUTLASS_REF:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.6.2}"
export TILELANG_VERSION="${TILELANG_VERSION:-0.1.12}"
export TOKENSPEED_MLA_VERSION="${TOKENSPEED_MLA_VERSION:-0.1.8}"
export TVM_FFI_VERSION="${TVM_FFI_VERSION:-0.1.11}"
export QUACK_KERNELS_SPEC="${QUACK_KERNELS_SPEC:-quack-kernels==0.6.4}"
export FASTSAFETENSORS_SPEC="${FASTSAFETENSORS_SPEC:-fastsafetensors>=0.3.3}"
# 0.1.12 matches the branch's requirements/cuda.txt and ships a
# manylinux_2_28_aarch64 wheel (verified on PyPI).
export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.12}"
export XGRAMMAR_REF="${XGRAMMAR_REF:-v0.2.5}"
export XGRAMMAR_COMMIT="${XGRAMMAR_COMMIT:-2ea71da4ccb997a06928c9fb69b99f330da56697}"
export XGRAMMAR_VERSION="${XGRAMMAR_VERSION:-0.2.5}"
export XGRAMMAR_TRANSFORMERS5_COMPAT="${XGRAMMAR_TRANSFORMERS5_COMPAT:-1}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/voipmonitor/InstantTensor.git}"
export INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}"
export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}"
export VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-nvtx==0.2.15 nccl4py==0.3.1}"

# ---------------------------------------------------------------- resources
# A Spark has 20 Grace cores and 128 GB unified memory shared with everything
# else. The upstream default MAX_JOBS=64 will thrash it.
export MAX_JOBS="${MAX_JOBS:-20}"
export VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-20}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
export PIN_SOURCE_COMMITS="${PIN_SOURCE_COMMITS:-1}"

stamp="$(date +%Y%m%d)"
export IMAGE="${IMAGE:-local/vllm:${PROFILE_NAME}-jj-b12x-cu132-sm121-${stamp}}"
export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-local/vllm:cu132-sm121-system-base-${stamp}}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-local/vllm:cu132-sm121-build-base-${stamp}}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.26.1rc0+jj.${PROFILE_NAME//-/.}.sm121.cu132.${stamp}}"


[[ -n "${VLLM_REQUIRED_LAUNCHERS}" ]] \
  || warn "VLLM_REQUIRED_LAUNCHERS is empty: no launcher guard -- the build cannot verify the ref matches your model. Set it in the profile."

# --------------------------------------------------------- frozen-block hash
# Everything between FROZEN-BEGIN and FROZEN-END feeds Docker layers that are
# ANCESTORS of the hours-long FlashInfer/vLLM compiles (system-base,
# build-base, base). Editing those bytes silently invalidates that cache on
# every builder. To change them intentionally, re-run with FROZEN_ACK set to
# the new hash this check prints, then update EXPECTED_FROZEN_SHA.
EXPECTED_FROZEN_SHA="8a685d5341c9552e0cf1924d75b5489ba4937d2ea4fa4e9a2ce1b3211d486542"
actual_frozen_sha="$(sed -n '/^# FROZEN-BEGIN/,/^# FROZEN-END/p' "${SELF}" | sha256sum | cut -d' ' -f1)"
if [[ "${actual_frozen_sha}" != "${EXPECTED_FROZEN_SHA}" && "${FROZEN_ACK:-}" != "${actual_frozen_sha}" ]]; then
  die "cache-critical frozen block modified (sha ${actual_frozen_sha}); this invalidates hours of compile cache. If intentional: FROZEN_ACK=${actual_frozen_sha}"
fi

# ------------------------------------------------------------- patch toggles
for t in PATCH_GPU_ARCH PATCH_HOST_ARCH PATCH_PCIE_ENV \
         PATCH_PIPCHECK_WHEELTAG PATCH_VLLM_REQ_MARKERS PATCH_EXLLAMAV3_AVX; do
  v="${!t:-auto}"
  case "${v}" in on|off|auto) ;; *) die "${t} must be on, off, or auto: ${v}" ;; esac
  printf -v "${t}" '%s' "${v}"; export "${t}"
done
if [[ "${PATCH_GPU_ARCH}" == off || "${PATCH_HOST_ARCH}" == off ]]; then
  warn "disabling arch patches produces an x86/sm120 image; that is upstream's stock build, not a Spark image"
fi

# ------------------------------------------------------------ dockerfile prep
dockerfile=Dockerfile.vllm-b12x-cu132
backup="${dockerfile}.pre-spark.$$"
# A build killed with SIGKILL (e.g. an OOM kill) or lost to a host crash
# never reaches the EXIT trap: the Dockerfile stays patched and a
# .pre-spark.* backup remains. Re-running on that state silently skips every
# `auto` patch (the anchors are gone) and re-injects the python rewrites, so
# refuse with recovery instructions instead. (Also refuses a second
# concurrent run in the same checkout, which is the intended one-run-per-
# checkout design.)
shopt -s nullglob
stale=( "${dockerfile}".pre-spark.* )
shopt -u nullglob
[[ ${#stale[@]} == 0 ]] || die "previous run left '${stale[0]}': the Dockerfile may still be patched. Restore it (git checkout -- ${dockerfile}) and delete the backup, then re-run."
cp -a "${dockerfile}" "${backup}"
# Backup may not exist if cleanup ever runs before `cp -a` (defensive: the
# trap is installed after this section today, but this keeps restore safe if
# that ordering ever changes). if-form: a missing backup must not turn into
# a nonzero return from the EXIT trap.
restore_dockerfile() { if [[ -f "${backup}" ]]; then mv -f "${backup}" "${dockerfile}"; fi; }

# ------------------------------------------------------- build telemetry
# Samples system memory use (MemTotal - MemAvailable) every 5s during the
# build and reports peak + wall time at the end -- including on failure, so
# an OOM-killed build still tells you how close it was.
MEM_PEAK_FILE="$(mktemp)"
SAMPLER_PID=""
BUILD_T0=""
start_sampler() {
  BUILD_T0="$(date +%s)"
  (
    peak=0
    while :; do
      used_kib=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{print t-a}' /proc/meminfo)
      [[ "${used_kib}" -gt "${peak}" ]] && { peak="${used_kib}"; echo "${peak}" > "${MEM_PEAK_FILE}"; }
      sleep 5
    done
  ) & SAMPLER_PID=$!
}
report_telemetry() {
  local status="$1"
  [[ -n "${BUILD_T0}" ]] || return 0
  local dt=$(( $(date +%s) - BUILD_T0 ))
  local peak_kib; peak_kib="$(cat "${MEM_PEAK_FILE}" 2>/dev/null || echo 0)"
  printf 'build telemetry: status=%s  jobs=%s  peak-memory=%s GiB  wall-time=%dh %02dm %02ds\n' \
    "${status}" "${MAX_JOBS}" \
    "$(awk -v k="${peak_kib}" 'BEGIN{printf "%.1f", k/1048576}')" \
    $((dt/3600)) $(((dt%3600)/60)) $((dt%60)) >&2
}
cleanup() {
  local rc=$?
  [[ -n "${SAMPLER_PID}" ]] && kill "${SAMPLER_PID}" 2>/dev/null || true
  # Read the peak BEFORE removing the file: the FAILED report needs it.
  [[ -n "${BUILD_T0}" && "${rc}" != 0 ]] && report_telemetry "FAILED(rc=${rc})"
  rm -f "${MEM_PEAK_FILE}"
  restore_dockerfile
}
trap cleanup EXIT

apply_sed_patch() {  # name toggle before_pattern min_expected sed-expr...
  local name="$1" toggle="$2" before="$3" min="$4"; shift 4
  local found; found="$(grep -cE "${before}" "${dockerfile}" || true)"
  if [[ "${toggle}" == off ]]; then note "${name}=off: skipped"; return 0; fi
  if [[ "${found}" == 0 ]]; then
    if [[ "${toggle}" == auto ]]; then note "${name}(auto): pattern absent; skipping (may be fixed upstream)"; return 0; fi
    die "${name}=on but pattern not found in ${dockerfile}"
  fi
  local args=() e; for e in "$@"; do args+=(-e "${e}"); done
  sed -i "${args[@]}" "${dockerfile}"
  local left; left="$(grep -cE "${before}" "${dockerfile}" || true)"
  [[ "${left}" == 0 ]] || die "${name}: ${left} occurrence(s) survived -- upstream changed shape: $(grep -nE "${before}" "${dockerfile}" | head -3)"
  [[ "${found}" -ge "${min}" ]] || die "${name}: expected >=${min} occurrences, found ${found} -- upstream changed shape"
  note "${name}: rewrote ${found} line-occurrence(s)"
}

# FROZEN-BEGIN -- cache-critical rewrite text: do not edit (see hash check)
apply_sed_patch PATCH_GPU_ARCH "${PATCH_GPU_ARCH}" \
  '12\.0a|=120a|12\.0f|ARCH_LIST=12\.0([^0-9]|$)' 6 \
  's/TORCH_CUDA_ARCH_LIST=12\.0a/TORCH_CUDA_ARCH_LIST=12.1a/g' \
  's/TORCH_CUDA_ARCH_LIST=12\.0\([^a-z0-9]\)/TORCH_CUDA_ARCH_LIST=12.1\1/g' \
  's/CMAKE_CUDA_ARCHITECTURES=120a/CMAKE_CUDA_ARCHITECTURES=121a/g' \
  's/FLASHINFER_CUDA_ARCH_LIST=12\.0f/FLASHINFER_CUDA_ARCH_LIST=12.1f/g'
apply_sed_patch PATCH_HOST_ARCH "${PATCH_HOST_ARCH}" \
  'x86_64-linux-gnu|targets/x86_64-linux' 8 \
  's|/usr/lib/x86_64-linux-gnu|/usr/lib/aarch64-linux-gnu|g' \
  's|targets/x86_64-linux|targets/sbsa-linux|g'
# FROZEN-END
apply_sed_patch PATCH_PCIE_ENV "${PATCH_PCIE_ENV}" \
  'VLLM_PCIE_ALLREDUCE_BACKEND=cpp' 1 \
  's/VLLM_PCIE_ALLREDUCE_BACKEND=cpp/VLLM_PCIE_ALLREDUCE_BACKEND=b12x/'

python3 - "${dockerfile}" <<'PYEOF'
import pathlib, re, sys
import os

def _tog(name, default="auto"):
    v = os.environ.get(name, default).strip().lower() or default
    if v not in ("on", "off", "auto"):
        raise SystemExit(f"{name} must be on, off, or auto (got: {v})")
    return v

TOG_PIP = _tog("PATCH_PIPCHECK_WHEELTAG")
TOG_MARK = _tog("PATCH_VLLM_REQ_MARKERS")
TOG_EXL = _tog("PATCH_EXLLAMAV3_AVX")

path = pathlib.Path(sys.argv[1])
text = path.read_text()

# Two variants of the repair, chosen per gate:
#
# FIX_WHEEL (build-base and base stage gates): WHEEL tag repair only — the
# cusparselt "sbsa" bug above. These stages are ANCESTORS of the hours-long
# vLLM compile; this text must stay byte-identical across wrapper revisions
# or their layers change and the whole downstream cache (FlashInfer, vLLM)
# invalidates. Do not touch it.
#
# FIX_FULL (final-stage /opt/venv gate only): the WHEEL repair plus vLLM
# metadata markers. The branch pins `instanttensor >= 0.1.9` (PyPI ships
# x86_64-only wheels; the aarch64 copy is built from source in a LATER
# final-stage step) and `PyNvVideoCodec==2.0.4` (NVIDIA only added aarch64
# wheels in 2.1.0; video-decode multimodal that text serving never
# imports). Append `platform_machine == 'x86_64'` markers to those two
# Requires-Dist lines in the INSTALLED vllm dist-info so the strict final
# pip check passes. Done here, not in requirements/cuda.txt before the wheel
# build, precisely so the vllm-build stage cache survives; the shipped
# image's metadata is identical either way. The final stage follows
# vllm-build in the graph, so changing its layers costs only the cheap
# final-stage steps.
FIX_WHEEL = ("{py} -c \"import sysconfig,pathlib; "
             "[p.write_text(p.read_text().replace('_sbsa','_aarch64')) "
             "for d in {{sysconfig.get_paths()['purelib'],sysconfig.get_paths()['platlib']}} "
             "for p in pathlib.Path(d).glob('*.dist-info/WHEEL')]\"")
FIX_FULL = ("{py} -c \"import sysconfig,pathlib,re; "
            "dirs={{sysconfig.get_paths()['purelib'],sysconfig.get_paths()['platlib']}}; "
            "[p.write_text(p.read_text().replace('_sbsa','_aarch64')) "
            "for d in dirs "
            "for p in pathlib.Path(d).glob('*.dist-info/WHEEL')]; "
            "[p.write_text(re.sub(r'(?m)^(Requires-Dist: (?:PyNvVideoCodec|instanttensor)\\b[^;\\n]*?)\\s*$', "
            "lambda m: m.group(1) + '; platform_machine == \\'x86_64\\'', p.read_text())) "
            "for d in dirs "
            "for p in pathlib.Path(d).glob('vllm-*.dist-info/METADATA')]\"")

venv_gates = 0

def inject(match):
    global venv_gates
    prefix, py = match.group(1), match.group(2)
    fix = FIX_WHEEL
    if "/opt/venv/" in py:
        if TOG_MARK != "off":
            fix = FIX_FULL
        venv_gates += 1
    return f"{prefix}{fix.format(py=py)} \\\n && {py} -m pip check"

# Match "&& <python> -m pip check" but not the tolerant "(... || true)" form.
pattern = re.compile(r"(&& )((?:/[\w./-]+/)?python[\w.]*) -m pip check(?! *\|\|)(?! *\))")
if TOG_PIP == "off":
    print("PATCH_PIPCHECK_WHEELTAG=off: pip-check repair injection skipped", file=sys.stderr)
else:
    text, n = pattern.subn(inject, text)
    if n == 0 and TOG_PIP == "auto":
        print("PATCH_PIPCHECK_WHEELTAG(auto): no strict pip-check gates found; "
              "skipping (upstream may have removed or relaxed them)", file=sys.stderr)
    else:
        assert n >= 3, f"expected >=3 strict pip check gates, found {n}"
        assert venv_gates == 1, f"expected exactly 1 /opt/venv pip check gate, found {venv_gates}"
        path.write_text(text)
        print(f"injected sbsa wheel-tag repair before {n} pip check gate(s) "
              f"(vllm metadata markers: {TOG_MARK != 'off'})", file=sys.stderr)

# exllamav3 aarch64 patch: the extension's CPU-dispatch helpers call
# __builtin_cpu_supports("avx2"/"avx512*") -- an x86-only GCC builtin -- and
# its CPU all-reduce files are written in AVX intrinsics. On Grace the AVX
# code paths can never run (dispatch is behind is_avx2_supported(), which we
# make return false, and vLLM only touches the extension for EXL3 quant
# anyway), so: force the detection functions to false and replace the two
# intrinsics files with link-compatible stubs. Injected right after the
# commit check inside the exllamav3 build RUN.
C_ABORT = ('{ fprintf(stderr, "exllamav3: CPU all-reduce is x86-only\\n"); '
           'abort(); }')
STUB_AVX2 = [
    "#include <cstdio>", "#include <cstdlib>",
    '#include "all_reduce_cpu_avx2.h"',
    "void enable_fast_fp() {}",
    "void enable_fast_fp_avx2() {}",
    "void perform_cpu_reduce(PGContext*, size_t, uint32_t, uint8_t*, size_t) "
    + C_ABORT,
    "void perform_cpu_reduce_avx2(PGContext*, size_t, uint32_t, uint8_t*, size_t) "
    + C_ABORT,
]
STUB_AVX512 = [
    "#include <cstdio>", "#include <cstdlib>",
    '#include "all_reduce_cpu_avx512.h"',
    "void enable_fast_fp_avx512() {}",
    "void bf16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) "
    + C_ABORT,
    "void perform_cpu_reduce_avx512(PGContext*, size_t, uint32_t, uint8_t*, size_t) "
    + C_ABORT,
]

def printf_cmd(lines, dest):
    quoted = " ".join("'" + ln + "'" for ln in lines)
    return f"printf '%s\\n' {quoted} > {dest}"

EXT = "exllamav3/exllamav3_ext"
patch_cmds = [
    ("sed -i 's/avx2_supported = __builtin_cpu_supports(\"avx2\");"
     f"/avx2_supported = false;/' {EXT}/avx2_target.cpp"),
    ("sed -i 's/avx512_supported = __builtin_cpu_supports(\"avx512f\") "
     '&& __builtin_cpu_supports("avx512bw");'
     f"/avx512_supported = false;/' {EXT}/avx512_target.cpp"),
    f"grep -q 'avx2_supported = false;' {EXT}/avx2_target.cpp",
    f"grep -q 'avx512_supported = false;' {EXT}/avx512_target.cpp",
    printf_cmd(STUB_AVX2, f"{EXT}/parallel/all_reduce_cpu_avx2.cpp"),
    printf_cmd(STUB_AVX512, f"{EXT}/parallel/all_reduce_cpu_avx512.cpp"),
]
injection = "".join(f" && {cmd} \\\n" for cmd in patch_cmds)

text = path.read_text()
anchor = ' && TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS="${MAX_JOBS}" \\\n      python setup.py build_ext --inplace'
cnt = text.count(anchor)
if TOG_EXL == "off":
    print("PATCH_EXLLAMAV3_AVX=off: exllamav3 source patch skipped", file=sys.stderr)
elif cnt == 0 and TOG_EXL == "auto":
    print("PATCH_EXLLAMAV3_AVX(auto): build anchor absent; skipping "
          "(upstream may have gained native aarch64 support)", file=sys.stderr)
else:
    assert cnt == 1, f"exllamav3 build anchor found {cnt} times"
    text = text.replace(anchor, injection + anchor)
    path.write_text(text)
    print("injected exllamav3 aarch64 source patch", file=sys.stderr)
PYEOF

if [[ "${DRY_RUN}" == 1 ]]; then
  echo
  note "DRY RUN complete: all rewrites validated against ${dockerfile}; no image built."
  note "profile=${PROFILE_NAME} image-would-be=${IMAGE} vllm=${VLLM_COMMIT:0:9} b12x=${B12X_COMMIT:0:9}"
  exit 0
fi

start_sampler
./build-vllm-b12x-cu132.sh "${EXTRA_ARGS[@]}"
report_telemetry OK

# ------------------------------------------------------------- verification
docker run --rm --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
import platform, torch
assert platform.machine() == "aarch64", platform.machine()
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.version.cuda == "13.2", torch.version.cuda
import importlib.util
assert importlib.util.find_spec("b12x") is not None, "b12x/sparkinfer missing"
assert importlib.util.find_spec("humming_kernels") is not None or \
       importlib.util.find_spec("humming") is not None, "humming kernels missing"
print("image OK: aarch64, torch", torch.__version__, "cuda", torch.version.cuda)
PY
for launcher in ${VLLM_REQUIRED_LAUNCHERS}; do
  docker run --rm --entrypoint test "${IMAGE}" -f "/usr/local/bin/${launcher}" \
    || die "required launcher missing from image: ${launcher}"
done
docker run --rm --entrypoint test "${IMAGE}" -f /opt/libnccl-local-inference.so.2.30.4
docker run --rm --entrypoint test "${IMAGE}" -f /usr/local/cuda/compat/libcuda.so.1
docker run --rm --entrypoint bash "${IMAGE}" -c '
  set -euo pipefail
  lib=/usr/local/cuda/targets/sbsa-linux/lib/libcublas.so.13
  test -L "${lib}"
  target="$(readlink -f "${lib}")"
  case "${target}" in
    /usr/lib/aarch64-linux-gnu/*) echo "cublas overlay OK: ${target}" ;;
    *) echo "ERROR: cublas overlay not applied: ${lib} -> ${target}" >&2; exit 1 ;;
  esac
'
printf '\nBuilt %s\n' "${IMAGE}"
report_telemetry "OK+VERIFIED"
printf 'Ship it to the other node:\n  docker save %s | ssh <node2> docker load\n' "${IMAGE}"
