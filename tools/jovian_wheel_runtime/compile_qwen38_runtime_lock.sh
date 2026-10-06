#!/usr/bin/env bash
# Compile the Qwen3.8 serving dependency locks for CPython 3.12 on x86-64 and aarch64 Linux.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
uv_binary=${UV_BIN:-uv}
export UV_NO_CONFIG=1

if ! uv_path=$(command -v "${uv_binary}"); then
  printf 'uv is required; set UV_BIN to its absolute path.\n' >&2
  exit 1
fi

expected_uv_version=$(awk -F= \
  '$1 == "uv.version" {print $2; found=1} END {exit !found}' \
  "${script_dir}/foundation.lock")
expected_uv_sha256=$(awk -F= \
  '$1 == "uv.sha256" {print $2; found=1} END {exit !found}' \
  "${script_dir}/foundation.lock")
test "$("${uv_path}" --version | awk '{print $2}')" = "${expected_uv_version}"
test "$(sha256sum "${uv_path}" | awk '{print $1}')" = \
  "${expected_uv_sha256}"

# linux/amd64 keeps its lock here; linux/arm64 (DGX Spark) keeps the same
# versions resolved for aarch64 wheels in linux-arm64/ (runtime_platform.py).
# Repository-relative paths keep the generated lock headers stable.
cd "${script_dir}/../.."
tools=tools/jovian_wheel_runtime
for target in x86_64:"${tools}" aarch64:"${tools}/linux-arm64"; do
  python_platform="${target%%:*}-manylinux_2_28"
  lock_dir="${target#*:}"
  "${uv_path}" pip compile \
    --quiet \
    --python-version 3.12 \
    --python-platform "${python_platform}" \
    --generate-hashes \
    --no-deps \
    --no-strip-markers \
    --output-file "${lock_dir}/qwen38-runtime.lock" \
    "${tools}/qwen38-runtime.in"

  resolved_lock=$(mktemp)
  "${uv_path}" pip compile \
    --quiet \
    --python-version 3.12 \
    --python-platform "${python_platform}" \
    --no-strip-markers \
    --output-file "${resolved_lock}" \
    "${tools}/qwen38-runtime.in"

  python3 "${tools}/verify_runtime_lock_closure.py" \
    --foundation-lock "${tools}/foundation-runtime.lock" \
    --runtime-lock "${lock_dir}/qwen38-runtime.lock" \
    --resolved-lock "${resolved_lock}"
  rm -f "${resolved_lock}"
done
