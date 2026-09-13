#!/usr/bin/env bash
# Install the locked Python foundation into an isolated Python 3.12 environment.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path=${1:-.venv-jovian}
uv_binary=${UV_BIN:-uv}

if ! uv_path=$(command -v "${uv_binary}"); then
  printf 'uv is required; set UV_BIN to its absolute path.\n' >&2
  exit 1
fi
expected_uv_version=$(awk -F= \
  '$1 == "uv.version" {print $2; found=1} END {exit !found}' \
  "${bundle_dir}/foundation.lock")
expected_uv_sha256=$(awk -F= \
  '$1 == "uv.sha256" {print $2; found=1} END {exit !found}' \
  "${bundle_dir}/foundation.lock")
test "$("${uv_path}" --version | awk '{print $2}')" = "${expected_uv_version}"
test "$(sha256sum "${uv_path}" | awk '{print $1}')" = "${expected_uv_sha256}"

(cd "${bundle_dir}" && sha256sum --check SHA256SUMS)
"${uv_path}" venv --python 3.12 "${venv_path}"
"${uv_path}" pip install \
  --python "${venv_path}/bin/python" \
  --require-hashes \
  -r "${bundle_dir}/cuda-runtime.lock"
"${uv_path}" pip install \
  --python "${venv_path}/bin/python" \
  --require-hashes \
  --no-index \
  --find-links "${bundle_dir}/wheels" \
  --no-deps \
  -r "${bundle_dir}/requirements-foundation.txt"

"${venv_path}/bin/python" - <<'PY'
import ctypes
import importlib.metadata
import torch

assert importlib.metadata.version("torch") == (
    "2.14.0a0+4fdf77b940.nv26.8.63802676"
)
assert torch.__version__ == "2.14.0a0+4fdf77b940.nv26.08"
assert torch.version.cuda == "13.4"
assert torch._C._GLIBCXX_USE_CXX11_ABI
ctypes.CDLL("libmpi.so.40")
for package in (
    "local-inference-torch-native-support",
    "nvidia-cuda-runtime",
    "nvidia-cuda-nvcc",
    "nvidia-cublas",
    "nvidia-cudnn-cu13",
):
    importlib.metadata.version(package)
print("CUDA 13.4 PyTorch foundation import: PASS")
PY

printf 'foundation_venv=%s status=installed native_runtime=self-contained driver=R580+ full_cuda134=R615+\n' \
  "$(realpath "${venv_path}")"
