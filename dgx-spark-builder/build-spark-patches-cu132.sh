#!/usr/bin/env bash
# build-spark-cu132.sh -- generic aarch64/sm_121 (DGX Spark) image builder for
# the local-inference-lab vLLM + B12X/SparkInfer serving stack.
#
# Usage:
#   ./build-spark-cu132.sh [--dry-run] [--log] [build.env] [-- extra build args]
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

# ---------------------------------------------------------------- transcript
# One timestamp per run, shared by the manifest and the transcript, so the two
# files pair unambiguously. Everything this script prints is teed to a temp
# file from here on -- deliberately BEFORE argument and env parsing, so a
# profile syntax error lands in the transcript too. The temp file is renamed
# to build-log-<profile>-<stamp>.txt at exit when LOGGING=1 (--log flag,
# profile key, or process env) and discarded otherwise. Side effect: stdout is
# now a pipe, so BuildKit emits plain non-TTY progress -- which is what you
# want in a file.
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
stamp="${RUN_STAMP%%-*}"
# The only date this script ever inserts, and only where a value asks for it by
# name. No token in a value means no date in that value.
expand_date() {
  local v="$1"
  v="${v//<datetime>/${RUN_STAMP}}"
  v="${v//<date>/${stamp}}"
  printf '%s' "${v}"
}
_RAW_LOG="$(mktemp)"
exec {_TTY_OUT}>&1 {_TTY_ERR}>&2
exec > >(tee -a "${_RAW_LOG}") 2>&1
_TEE_PID=$!
finalize_transcript() {
  if [[ "${LOGGING:-0}" != 1 ]]; then rm -f "${_RAW_LOG:-}"; return 0; fi
  # RUN_LOG is unset if we died before PROFILE_NAME resolved; keep the
  # transcript anyway -- that run is exactly the one worth reading.
  local log="${RUN_LOG:-${SCRIPT_DIR}/build-log-${PROFILE_NAME:-unknown}-${RUN_STAMP}.txt}"
  printf 'build transcript: %s\n' "${log}" >&2
  # Restore the real fds so tee sees EOF, then WAIT for it to flush before the
  # move; without this the last lines written are lost.
  exec 1>&"${_TTY_OUT}" 2>&"${_TTY_ERR}"
  wait "${_TEE_PID}" 2>/dev/null || true
  mv -f "${_RAW_LOG}" "${log}" 2>/dev/null || cp -f "${_RAW_LOG}" "${log}"
}
# Provisional trap so an early die still finalizes the transcript; replaced by
# the full cleanup trap once the Dockerfile backup exists.
trap finalize_transcript EXIT

# ------------------------------------------------------------ argument parse
DRY_RUN=0
LOG_FLAG=0
ENV_FILE=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --log) LOG_FLAG=1; shift ;;
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
  # Resolve by what is actually beside the script, not by the name build.env:
  # a lone profile called anything else used to fall through to the in-script
  # fallback pins SILENTLY, which is the worst possible default. More than one
  # profile is ambiguous and is refused rather than guessed.
  shopt -s nullglob; _profiles=( "${SCRIPT_DIR}"/*.env ); shopt -u nullglob
  case ${#_profiles[@]} in
    0) ;;
    1) ENV_FILE="${_profiles[0]}" ;;
    *) die "multiple profiles in ${SCRIPT_DIR}: ${_profiles[*]##*/} -- name the one to build" ;;
  esac
fi
[[ -n "${ENV_FILE}" ]] \
  || warn "no profile found beside this script: building from IN-SCRIPT FALLBACK PINS, which are not the source of truth and are probably stale"

ALLOWED_KEYS=" ALLOW_FOREIGN_ARCH B12X_COMMIT B12X_PATCH_FILE B12X_PATCH_SHA256 B12X_PIN B12X_REF B12X_REPO BUILD_BASE_IMAGE_TAG CUTLASS_COMMIT CUTLASS_DSL_VERSION CUTLASS_REF DEEPGEMM_COMMIT DEEPGEMM_REF FASTSAFETENSORS_SPEC FLASHINFER_BUILD_CUBIN FLASHINFER_COMMIT FLASHINFER_REF FLASHINFER_REPO FROZEN_ACK HUMMING_KERNELS_SPEC IMAGE IMAGE_REPO IMAGE_TAG INSTANTTENSOR_COMMIT INSTANTTENSOR_REF INSTANTTENSOR_REPO LAUNCHER_COMMIT LAUNCHER_REF LAUNCHER_REPO LOGGING MAX_JOBS NCCL_COMMIT NCCL_REF NCCL_REPO NVCC_THREADS PATCH_EXLLAMAV3_AVX PATCH_GPU_ARCH PATCH_HOST_ARCH PATCH_PCIE_ENV PATCH_PIPCHECK_WHEELTAG PATCH_VLLM_REQ_MARKERS PIN_PREFLIGHT PIN_SOURCE_COMMITS PROFILE_NAME QUACK_KERNELS_SPEC SPARKINFER_COMMIT SPARKINFER_REF SPARKINFER_REPO SYSTEM_BASE_IMAGE TILELANG_VERSION TOKENSPEED_MLA_VERSION TORCHVISION_VERSION TORCH_BUNDLED_NCCL_VERSION TORCH_VERSION TVM_FFI_VERSION VLLM_BUILD_VERSION VLLM_COMMIT VLLM_MAX_JOBS VLLM_NVCC_THREADS VLLM_PATCH_FILE VLLM_PATCH_SHA256 VLLM_PATCH_URL VLLM_PIN VLLM_REF VLLM_REPO VLLM_REQUIRED_LAUNCHERS VLLM_RUNTIME_EXTRA_PACKAGES XGRAMMAR_COMMIT XGRAMMAR_REF XGRAMMAR_TRANSFORMERS5_COMPAT XGRAMMAR_VERSION "
if [[ -n "${ENV_FILE}" ]]; then
  [[ -f "${ENV_FILE}" ]] || die "env file not found: ${ENV_FILE}"
  # Canonicalize now: the repo-root cd below would break a relative path for
  # every later reference (the build-log emitter reads this file post-cd).
  ENV_FILE="$(cd "$(dirname "${ENV_FILE}")" && pwd)/$(basename "${ENV_FILE}")"
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
    # <date>/<datetime> are substitution tokens, not unresolved placeholders.
    case "${val}" in "<date>"|"<datetime>") ;; *)
      if [[ "${val}" =~ ^\<[A-Za-z0-9_-]+\>$ ]]; then die "${ENV_FILE}:${lineno}: unresolved placeholder for ${key}"; fi ;;
    esac
    if [[ -n "${!key+x}" ]]; then
      note "process env overrides ${ENV_FILE}: ${key}"
      OVERRIDDEN_KEYS="${OVERRIDDEN_KEYS:-} ${key}"
    else
      printf -v "${key}" '%s' "${val}"
      export "${key}"
    fi
  done < "${ENV_FILE}"
  note "loaded profile: ${ENV_FILE}"
fi

# PROFILE_NAME names the manifest and the transcript, and supplies the default
# image tag. If a profile gave only IMAGE_TAG, name the paperwork after that
# rather than after nothing -- sanitized, since tags allow uppercase and
# underscores and profile names do not.
if [[ -n "${PROFILE_NAME+x}" ]]; then
  _profile_named=1
elif [[ -n "${IMAGE_TAG+x}" ]]; then
  _profile_named=1
  PROFILE_NAME="$(expand_date "${IMAGE_TAG}")"
  PROFILE_NAME="${PROFILE_NAME,,}"
  PROFILE_NAME="${PROFILE_NAME//[^a-z0-9.-]/-}"
  while [[ -n "${PROFILE_NAME}" && "${PROFILE_NAME}" != [a-z0-9]* ]]; do
    PROFILE_NAME="${PROFILE_NAME#?}"
  done
  [[ -n "${PROFILE_NAME}" ]] || PROFILE_NAME=custom
else
  _profile_named=0
fi
: "${PROFILE_NAME:=custom}"
PROFILE_NAME="$(expand_date "${PROFILE_NAME}")"
[[ "${PROFILE_NAME}" =~ ^[a-z0-9][a-z0-9.-]*$ ]] \
  || die "PROFILE_NAME must be lowercase [a-z0-9.-] after <date> expansion: ${PROFILE_NAME}"
export PROFILE_NAME

# --log wins over the profile; the profile wins over nothing. A separate flag
# variable avoids the parser reporting a bogus "process env overrides" note
# for a key the operator set on the command line.
: "${LOGGING:=0}"
[[ "${LOG_FLAG}" != 1 ]] || LOGGING=1
case "${LOGGING}" in 0|1) ;; *) die "LOGGING must be 0 or 1: ${LOGGING}" ;; esac
RUN_LOG="${SCRIPT_DIR}/build-log-${PROFILE_NAME}-${RUN_STAMP}.txt"
[[ "${LOGGING}" != 1 ]] || note "transcript: ${RUN_LOG}"

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
# REF and COMMIT default to the same pin, but either can arrive independently
# from the process environment and silently beat the profile. The build stage
# checks out the ref and THEN verifies the commit, so a disagreement is the
# branch-vs-sha race this design exists to prevent.
[[ "${VLLM_REF}" == "${VLLM_COMMIT}" ]] \
  || warn "VLLM_REF (${VLLM_REF}) != VLLM_COMMIT (${VLLM_COMMIT}); the vllm-build stage checks out the ref then verifies the commit -- this races unless the ref is immutable"
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
[[ "${B12X_REF}" == "${B12X_COMMIT}" ]] \
  || warn "B12X_REF (${B12X_REF}) != B12X_COMMIT (${B12X_COMMIT}); same checkout-then-verify race as vLLM"

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

# ------------------------------------------------------------- image naming
# A docker reference is REPOSITORY:TAG -- local/vllm:glm53-nvfp4. The wrapper
# invents no decoration of its own: the repository defaults to local/vllm, the
# tag defaults to PROFILE_NAME verbatim, and the profile owns both. Set IMAGE
# to take over the whole reference, or IMAGE_REPO / IMAGE_TAG to change one
# half of it.
#
# A date appears ONLY where a value asks for it: <date> expands to YYYYmmdd and
# <datetime> to YYYYmmdd-HHMMSS, wherever they occur in PROFILE_NAME, IMAGE,
# IMAGE_REPO, IMAGE_TAG, either base-stage tag, or VLLM_BUILD_VERSION. With no
# token there is no date, so rebuilding a profile overwrites its own tag
# instead of leaving one full image set per build day on a 4 TB node.
: "${IMAGE_REPO:=local/vllm}"
if [[ "${_profile_named}" == 1 ]]; then
  : "${IMAGE_TAG:=${PROFILE_NAME}}"
else
  # Nothing named this build, so there is nothing to name the tag after.
  : "${IMAGE_TAG:=<date>}"
fi
IMAGE_REPO="$(expand_date "${IMAGE_REPO}")"
IMAGE_TAG="$(expand_date "${IMAGE_TAG}")"
IMAGE="${IMAGE:-${IMAGE_REPO}:${IMAGE_TAG}}"
IMAGE="$(expand_date "${IMAGE}")"

# Shape checks, before anything derives from this. An untagged reference
# silently becomes :latest and an unexpanded token would be baked into a
# published tag; both are cheap here and expensive after an eight-hour build.
case "${IMAGE}" in
  *"<"*|*">"*) die "unexpanded token in IMAGE: ${IMAGE} (only <date> and <datetime> are substituted)" ;;
esac
_ref_last="${IMAGE##*/}"                       # drop registry host and namespace
case "${_ref_last}" in
  *:*) IMAGE_TAG="${_ref_last##*:}" ;;
  *)   die "IMAGE has no tag: ${IMAGE} -- docker would resolve this to :latest" ;;
esac
[[ "${IMAGE_TAG}" =~ ^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$ ]] \
  || die "IMAGE tag is not a valid docker tag: ${IMAGE_TAG}"
# Re-derived FROM the resolved reference (not the other way round), so that a
# profile setting IMAGE alone governs every name below it and the manifest
# reports the halves of the image that was actually built.
IMAGE_REPO="${IMAGE%":${IMAGE_TAG}"}"
export IMAGE_REPO IMAGE_TAG IMAGE

# The base-stage tags are build-cache handles, not artifacts. Derived per
# profile so two profiles with different toolchain pins can never silently
# overwrite each other's bases; point several profiles at one pair of names to
# share the layers (and the disk) when their toolchains match.
SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-${IMAGE_REPO}:${IMAGE_TAG}-system-base}"
BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-${IMAGE_REPO}:${IMAGE_TAG}-build-base}"
SYSTEM_BASE_IMAGE="$(expand_date "${SYSTEM_BASE_IMAGE}")"
BUILD_BASE_IMAGE_TAG="$(expand_date "${BUILD_BASE_IMAGE_TAG}")"
export SYSTEM_BASE_IMAGE BUILD_BASE_IMAGE_TAG

# PEP 440 wheel version. The local segment is the tag with every
# non-alphanumeric run collapsed to a single dot, because a version is not
# free-form: hyphens and doubled dots are rejected by packaging tools.
_local_version="${IMAGE_TAG//[^A-Za-z0-9]/.}"
while [[ "${_local_version}" == *..* ]]; do _local_version="${_local_version//../.}"; done
_local_version="${_local_version#.}"; _local_version="${_local_version%.}"
VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.26.1rc0+${_local_version}}"
VLLM_BUILD_VERSION="$(expand_date "${VLLM_BUILD_VERSION}")"
export VLLM_BUILD_VERSION


[[ -n "${VLLM_REQUIRED_LAUNCHERS}" ]] \
  || warn "VLLM_REQUIRED_LAUNCHERS is empty: no launcher guard -- the build cannot verify the ref matches your model. Set it in the profile."

# ------------------------------------------------------------ pin pre-flight
# One HTTPS round-trip per repo proves the pinned sha is reachable BEFORE
# spending the build. Catches typos, deleted qualification branches, and
# force-pushed fork heads -- all of which otherwise fail ~40 minutes in, after
# the clone. Set PIN_PREFLIGHT=0 to skip (air-gapped builders).
check_pin() {  # repo-url sha label
  case "$1" in *github.com/*) ;; *) return 0 ;; esac
  local nwo="${1#*github.com/}"; nwo="${nwo%.git}"
  local code auth=()
  # Only 404/422 mean "this sha is not there". Everything else -- 403 for the
  # anonymous rate limit, 5xx, or 000 for no network -- is INCONCLUSIVE and
  # must not fail a build. A token makes the check reliable on shared egress.
  [[ -z "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]] \
    || auth=(-H "Authorization: Bearer ${GH_TOKEN:-${GITHUB_TOKEN}}")
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "${auth[@]}" \
    "https://api.github.com/repos/${nwo}/commits/$2" 2>/dev/null || echo 000)"
  case "${code}" in
    200) ;;
    404|422) die "$3 pin $2 is not on ${nwo} (HTTP ${code}) -- deleted branch, force-push, or typo. Verify: git ls-remote https://github.com/${nwo}.git" ;;
    *) warn "$3 pin pre-flight inconclusive (HTTP ${code} from api.github.com: rate limit, auth, or no network); pin unverified"
       return 1 ;;
  esac
}
if [[ "${PIN_PREFLIGHT:-1}" == 1 ]] && command -v curl >/dev/null 2>&1; then
  _pins_ok=1
  check_pin "${VLLM_REPO}" "${VLLM_COMMIT}" vLLM || _pins_ok=0
  check_pin "${B12X_REPO}" "${B12X_COMMIT}" B12X || _pins_ok=0
  if [[ "${_pins_ok}" == 1 ]]; then
    note "pin pre-flight OK: vllm=${VLLM_COMMIT:0:9} b12x=${B12X_COMMIT:0:9} both reachable"
  else
    warn "pin pre-flight INCOMPLETE -- at least one pin is unverified; a bad sha will not surface until the clone (~40 min in). Set GH_TOKEN to make this check reliable."
  fi
fi

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
# Record the checkout's state BEFORE this wrapper touches it. Measured after
# the patch pass it would always read DIRTY -- the patched Dockerfile and the
# untracked .pre-spark backup are our own doing, not the operator's.
UPSTREAM_DIRTY="clean"
[[ -z "$(git status --porcelain 2>/dev/null)" ]] || UPSTREAM_DIRTY="DIRTY (uncommitted changes present)"
cp -a "${dockerfile}" "${backup}"
PRISTINE_DOCKERFILE_SHA="$(sha256sum "${dockerfile}" | cut -d' ' -f1)"
# Patch outcomes are appended here as they happen; the build log embeds it.
PATCH_REPORT="$(mktemp)"
# Backup may not exist if cleanup ever runs before `cp -a` (defensive: the
# trap is installed after this section today, but this keeps restore safe if
# that ordering ever changes). if-form: a missing backup must not turn into
# a nonzero return from the EXIT trap.
restore_dockerfile() { if [[ -f "${backup}" ]]; then mv -f "${backup}" "${dockerfile}"; fi; }
# Armed here, immediately after the backup exists. cleanup() and the
# telemetry helpers below tolerate their state not being set yet.
trap cleanup EXIT

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
  [[ -n "${SAMPLER_PID:-}" ]] && kill "${SAMPLER_PID}" 2>/dev/null || true
  # Read the peak BEFORE removing the file: the FAILED report needs it.
  [[ -n "${BUILD_T0:-}" && "${rc}" != 0 ]] && report_telemetry "FAILED(rc=${rc})"
  rm -f "${MEM_PEAK_FILE:-}" "${PATCH_REPORT:-}"
  restore_dockerfile
  # Last, so the transcript contains everything above it.
  finalize_transcript
}

plog() { printf '%s\n' "$*" >> "${PATCH_REPORT}"; }

apply_sed_patch() {  # name toggle before_pattern min_expected sed-expr...
  local name="$1" toggle="$2" before="$3" min="$4"; shift 4
  local found; found="$(grep -cE "${before}" "${dockerfile}" || true)"
  if [[ "${toggle}" == off ]]; then note "${name}=off: skipped"; plog "${name}=off: skipped"; return 0; fi
  if [[ "${found}" == 0 ]]; then
    if [[ "${toggle}" == auto ]]; then note "${name}(auto): pattern absent; skipping (may be fixed upstream)"; plog "${name}(auto): pattern absent; skipped"; return 0; fi
    die "${name}=on but pattern not found in ${dockerfile}"
  fi
  # Shape check BEFORE the rewrite: a changed upstream should be rejected
  # without having been half-edited first.
  [[ "${found}" -ge "${min}" ]] || die "${name}: expected >=${min} occurrences, found ${found} -- upstream changed shape"
  local args=() e; for e in "$@"; do args+=(-e "${e}"); done
  sed -i "${args[@]}" "${dockerfile}"
  local left; left="$(grep -cE "${before}" "${dockerfile}" || true)"
  [[ "${left}" == 0 ]] || die "${name}: ${left} occurrence(s) survived -- upstream changed shape: $(grep -nE "${before}" "${dockerfile}" | head -3)"
  note "${name}: rewrote ${found} line-occurrence(s)"
  plog "${name}: rewrote ${found} line-occurrence(s)"
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

python3 - "${dockerfile}" 2> >(tee -a "${PATCH_REPORT}" >&2) <<'PYEOF'
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
# Arch-agnostic on purpose. The literal "12.1a" only exists because
# PATCH_GPU_ARCH just rewrote it, so a literal anchor silently couples patch 6
# to patch 1: PATCH_GPU_ARCH=off -- or its documented retirement, upstream
# parametrizing the arch as build ARGs -- would make this patch auto-skip and
# the build would die hours later on x86-only GCC builtins.
ANCHOR_RE = re.compile(
    r' && TORCH_CUDA_ARCH_LIST=\S+ MAX_JOBS="\$\{MAX_JOBS\}" \\\n'
    r'      python setup\.py build_ext --inplace')
hits = ANCHOR_RE.findall(text)
cnt = len(hits)
if TOG_EXL == "off":
    print("PATCH_EXLLAMAV3_AVX=off: exllamav3 source patch skipped", file=sys.stderr)
elif cnt == 0 and TOG_EXL == "auto":
    # Auto-skip is only legitimate when exllamav3 is no longer built at all.
    # If the stage is still there and the anchor moved, dying now costs
    # seconds; skipping costs the whole build.
    assert "exllamav3" not in text, (
        "PATCH_EXLLAMAV3_AVX(auto): exllamav3 is still built but the build "
        "anchor did not match -- upstream changed shape; inspect the "
        "exllamav3 RUN and update ANCHOR_RE")
    print("PATCH_EXLLAMAV3_AVX(auto): exllamav3 no longer built; skipping "
          "(upstream may have dropped it or gained native aarch64 support)",
          file=sys.stderr)
else:
    assert cnt == 1, f"exllamav3 build anchor found {cnt} times"
    text = text.replace(hits[0], injection + hits[0])
    path.write_text(text)
    print("injected exllamav3 aarch64 source patch", file=sys.stderr)
PYEOF

# The exact Dockerfile the build consumed (all patches applied) -- the
# build-log analog of a published integration-patch sha.
PATCHED_DOCKERFILE_SHA="$(sha256sum "${dockerfile}" | cut -d' ' -f1)"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo
  note "DRY RUN complete: all rewrites validated against ${dockerfile}; no image built."
  note "profile=${PROFILE_NAME} vllm=${VLLM_COMMIT:0:9} b12x=${B12X_COMMIT:0:9}"
  note "image:         ${IMAGE}"
  note "system-base:   ${SYSTEM_BASE_IMAGE}"
  note "build-base:    ${BUILD_BASE_IMAGE_TAG}"
  note "wheel version: ${VLLM_BUILD_VERSION}"
  exit 0
fi

start_sampler
./build-vllm-b12x-cu132.sh "${EXTRA_ARGS[@]}"
report_telemetry OK

# ------------------------------------------------------------- verification
# Expectations come from the profile, not from literals: a profile that bumps
# TORCH_VERSION must not fail its own correct build at the last step.
_cu="${TORCH_VERSION##*+cu}"
docker run --rm \
  -e EXPECT_TORCH="${TORCH_VERSION%%+*}" \
  -e EXPECT_CUDA="${_cu:0:2}.${_cu:2}" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
import os, platform, torch
assert platform.machine() == "aarch64", platform.machine()
assert torch.__version__.startswith(os.environ["EXPECT_TORCH"]), torch.__version__
assert torch.version.cuda == os.environ["EXPECT_CUDA"], torch.version.cuda
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

# ----------------------------------------------------------- build manifest
# One self-contained record per completed build: everything the lab
# publishing checklist asks for that exists at build time, with the fields
# that only exist after a registry push marked UNKNOWN. Markdown so the
# announcement sections lift straight out. Shares RUN_STAMP with the optional
# transcript (LOGGING=1), so manifest and log pair by filename.
# Called non-fatally: a verified build must never be failed by its own
# paperwork, so every substitution degrades to UNKNOWN instead of dying.
emit_build_manifest() {
image_id="$(docker inspect --format '{{.Id}}' "${IMAGE}" 2>/dev/null || echo 'UNKNOWN — needs verification')"

# Resolve the external base (nvidia/cuda tag) the Dockerfile declares. Read
# from the patched file -- no patch touches these ARGs, so the value is the
# pristine one -- honoring any --build-arg override passed through EXTRA_ARGS,
# then read its registry digest (present locally after the pull the build
# just did).
cuda_ver="$(sed -n 's/^ARG CUDA_VERSION=\(.*\)$/\1/p' "${dockerfile}" | head -1)"
ubuntu_ver="$(sed -n 's/^ARG UBUNTU_VERSION=\(.*\)$/\1/p' "${dockerfile}" | head -1)"
for ea in "${EXTRA_ARGS[@]:-}"; do
  case "${ea}" in
    CUDA_VERSION=*)   cuda_ver="${ea#CUDA_VERSION=}" ;;
    UBUNTU_VERSION=*) ubuntu_ver="${ea#UBUNTU_VERSION=}" ;;
  esac
done
base_tag="nvidia/cuda:${cuda_ver}-cudnn-devel-ubuntu${ubuntu_ver}"
base_digest="$(docker image inspect --format '{{index .RepoDigests 0}}' "${base_tag}" 2>/dev/null \
  || echo 'UNKNOWN — needs verification')"

upstream_head="$(git rev-parse HEAD 2>/dev/null || echo 'UNKNOWN — needs verification (not a git checkout)')"
upstream_dirty="${UPSTREAM_DIRTY:-UNKNOWN — needs verification}"
wrapper_sha="$(sha256sum "${SELF}" | cut -d' ' -f1)"
profile_sha="none (in-script defaults only)"
[[ -z "${ENV_FILE}" ]] || profile_sha="$(sha256sum "${ENV_FILE}" 2>/dev/null | cut -d' ' -f1 || true)"
[[ -n "${profile_sha}" ]] || profile_sha='UNKNOWN — needs verification'

pip_freeze="$(docker run --rm --entrypoint /opt/venv/bin/pip "${IMAGE}" freeze 2>/dev/null \
  || echo 'UNKNOWN — needs verification (pip freeze failed)')"

{
  printf '# Build manifest — %s — %s\n\n' "${PROFILE_NAME}" "${RUN_STAMP}"
  printf '## Image\n\n'
  printf -- '- Local tag: `%s`\n' "${IMAGE}"
  printf -- '- Image ID (local config digest, survives `docker save`/`load`): `%s`\n' "${image_id}"
  printf -- '- Registry digest (`image@sha256:` form): UNKNOWN — needs verification (exists only after a push; see commands at the end of this manifest)\n'
  printf -- '- Base image: `%s` @ `%s`\n' "${base_tag}" "${base_digest}"
  printf -- '- vLLM build version: `%s`\n\n' "${VLLM_BUILD_VERSION}"
  printf '## Build system\n\n'
  printf -- '- blackwell-llm-docker checkout: `%s` (%s)\n' "${upstream_head}" "${upstream_dirty}"
  printf -- '- Wrapper: `build-spark-cu132.sh` sha256 `%s`\n' "${wrapper_sha}"
  printf -- '- Profile: `%s` sha256 `%s`\n' "${ENV_FILE:-none}" "${profile_sha}"
  printf -- '- Dockerfile sha256: pristine `%s`, as-built (post-patch) `%s`\n' \
    "${PRISTINE_DOCKERFILE_SHA}" "${PATCHED_DOCKERFILE_SHA:-UNKNOWN — needs verification}"
  printf -- '- Extra docker build args: `%s`\n' "${EXTRA_ARGS[*]:-none}"
  printf -- '- Run transcript: `%s`\n' "$([[ "${LOGGING}" == 1 ]] && echo "${RUN_LOG##*/}" || echo 'not kept (LOGGING=0)')"
  printf -- '- Build host: %s, %s\n\n' "$(uname -m)" "$(uname -r)"
  printf '## Effective configuration\n\n'
  printf 'Precedence applied: process env > profile > script defaults. Keys the\n'
  printf 'process environment overrode are marked; everything else came from the\n'
  printf 'profile or the script fallback.\n\n```\n'
  for key in ${ALLOWED_KEYS}; do
    [[ -n "${!key:-}" ]] || continue
    case "${key}" in FROZEN_ACK|ALLOW_FOREIGN_ARCH) continue ;; esac
    mark=""
    [[ " ${OVERRIDDEN_KEYS:-} " != *" ${key} "* ]] || mark="   # OVERRIDDEN by process env"
    printf '%s=%s%s\n' "${key}" "${!key}" "${mark}"
  done
  printf '```\n\n'
  printf '## Patches applied (differences from the upstream build system)\n\n```\n'
  cat "${PATCH_REPORT}" 2>/dev/null || echo "UNKNOWN — needs verification"
  printf '```\n\n'
  printf '## Verification (commands executed by this wrapper, all passed)\n\n'
  printf -- '- In-image python asserts: aarch64, torch 2.13.0+cu132, cuda 13.2, `b12x` and humming kernels importable\n'
  printf -- '- Required launchers present: `%s`\n' "${VLLM_REQUIRED_LAUNCHERS:-none configured}"
  printf -- '- Patched NCCL present: `/opt/libnccl-local-inference.so.2.30.4`\n'
  printf -- '- CUDA compat shim present: `/usr/local/cuda/compat/libcuda.so.1`\n'
  printf -- '- cuBLAS overlay symlink resolves into `/usr/lib/aarch64-linux-gnu/`\n'
  printf -- '- Serving on hardware: Not tested (build-time verification only; record pair validation separately)\n\n'
  printf '## Resolved floating specs\n\n'
  printf -- '- `FASTSAFETENSORS_SPEC=%s` resolved to: `%s`\n\n' \
    "${FASTSAFETENSORS_SPEC}" \
    "$(printf '%s\n' "${pip_freeze}" | grep -i '^fastsafetensors==' || echo 'UNKNOWN — needs verification')"
  printf '## Telemetry\n\n'
  printf -- '- jobs=%s peak-memory=%s GiB wall-time=%ss\n\n' \
    "${MAX_JOBS}" \
    "$(awk -v k="$(cat "${MEM_PEAK_FILE}" 2>/dev/null || echo 0)" 'BEGIN{printf "%.1f", k/1048576}')" \
    "$(( $(date +%s) - BUILD_T0 ))"
  printf '## Full pip freeze (final image /opt/venv)\n\n```\n%s\n```\n\n' "${pip_freeze}"
  printf '## Obtaining the registry digest (after push)\n\n```\n'
  printf 'docker tag %s <registry>/<repo>:<tag>\n' "${IMAGE}"
  printf 'docker push <registry>/<repo>:<tag>   # last line prints: digest: sha256:...\n'
  printf "docker inspect --format '{{index .RepoDigests 0}}' <registry>/<repo>:<tag>\n"
  printf '```\n'
} > "${build_manifest}" && note "build manifest written: ${build_manifest}"
}

# Path hoisted so the closing instructions below still resolve if the emitter
# dies, and run in a SUBSHELL: `if ! func` catches a nonzero return but NOT a
# `set -u` abort, which would kill the script after a verified build. A
# subshell contains both.
build_manifest="${SCRIPT_DIR}/build_manifest-${PROFILE_NAME}-${RUN_STAMP}.md"
if ! ( emit_build_manifest ); then
  warn "build manifest emission failed -- the image itself is built and VERIFIED; re-run the wrapper (fully cached) to regenerate it, and please report this"
fi

printf 'Ship it to the other node:\n  docker save %s | ssh <node2> docker load\n' "${IMAGE}"
printf '\nPublishing needs an immutable image@sha256 reference. That digest is minted\n'
printf 'by a registry push (a local image has only an image ID, and docker refuses\n'
printf 'tagging by digest). When ready:\n'
printf '  docker tag %s <registry>/<repo>:<tag>\n' "${IMAGE}"
printf '  docker push <registry>/<repo>:<tag>\n'
printf "  docker inspect --format '{{index .RepoDigests 0}}' <registry>/<repo>:<tag>\n"
printf 'then fill the UNKNOWN digest field in %s\n' "${build_manifest}"
