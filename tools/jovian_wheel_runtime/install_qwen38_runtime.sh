#!/usr/bin/env bash
# Install the complete Qwen3.8 CUDA 13.4 runtime into an isolated venv.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path=${1:?Pass a destination venv path that does not exist}
uv_binary=${UV_BIN:-uv}
native_verify=${QWEN38_NATIVE_VERIFY:-required}
export UV_NO_CONFIG=1

case "${native_verify}" in
  required|deferred) ;;
  *)
    printf 'QWEN38_NATIVE_VERIFY must be required or deferred.\n' >&2
    exit 2
    ;;
esac

test ! -e "${venv_path}"
uv_path=$(command -v "${uv_binary}")
expected_uv_version=$(awk -F= '$1 == "uv.version" {print $2}' "${bundle_dir}/foundation.lock")
expected_uv_sha256=$(awk -F= '$1 == "uv.sha256" {print $2}' "${bundle_dir}/foundation.lock")
expected_container_uv_sha256=$(awk -F= \
  '$1 == "uv.container-sha256" {print $2}' "${bundle_dir}/foundation.lock")
test "$("${uv_path}" --version | awk '{print $2}')" = "${expected_uv_version}"
actual_uv_sha256=$(sha256sum "${uv_path}" | awk '{print $1}')
test "${actual_uv_sha256}" = "${expected_uv_sha256}" \
  || test "${actual_uv_sha256}" = "${expected_container_uv_sha256}"
(cd "${bundle_dir}" && sha256sum --check SHA256SUMS)

"${uv_path}" venv --python 3.12 "${venv_path}"
"${uv_path}" pip install --python "${venv_path}/bin/python" \
  --require-hashes --no-deps -r "${bundle_dir}/foundation-runtime.lock"
"${uv_path}" pip install --python "${venv_path}/bin/python" \
  --require-hashes --no-deps -r "${bundle_dir}/qwen38-runtime.lock"
"${uv_path}" pip install --python "${venv_path}/bin/python" \
  --no-index --find-links "${bundle_dir}/wheels" --no-deps --require-hashes \
  -r "${bundle_dir}/requirements-local.txt"
"${uv_path}" pip check --python "${venv_path}/bin/python"
install -D -m 0644 "${bundle_dir}/manifest.json" \
  "${venv_path}/share/lil-runtime/manifest.json"
install -m 0755 "${bundle_dir}/qwen38_runtime_entrypoint.sh" \
  "${venv_path}/bin/qwen38-runtime"

if [[ ${native_verify} == required ]]; then
  "${venv_path}/bin/qwen38-runtime" \
    "${venv_path}/bin/python" "${bundle_dir}/verify_qwen38_runtime.py"
fi
printf 'qwen38_runtime=%s status=installed native_verification=%s gpu_qualification=required\n' \
  "$(realpath "${venv_path}")" "${native_verify}"
