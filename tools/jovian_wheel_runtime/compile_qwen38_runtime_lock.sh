#!/usr/bin/env bash
# Compile the Qwen3.8 serving dependency lock for CPython 3.12 on x86-64 Linux.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
uv_binary=${UV_BIN:-uv}

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

exec "${uv_path}" pip compile \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 \
  --generate-hashes \
  --no-deps \
  --no-strip-markers \
  --output-file "${script_dir}/qwen38-runtime.lock" \
  "${script_dir}/qwen38-runtime.in"
