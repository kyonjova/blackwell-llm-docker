#!/usr/bin/env bash
# Build and optionally qualify the Qwen3.8 runtime over the immutable NGC base.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${tool_dir}/../.." && pwd)"
bundle=${1:?Pass the assembled Qwen3.8 runtime bundle directory}
image=${2:?Pass the output container image name and tag}
builder=${BUILDX_BUILDER:-lil-wheel-cu134-sm120}
source_image=$(
  awk -F= '$1 == "source.image" {sub(/^[^=]*=/, ""); print}' \
    "${tool_dir}/foundation.lock"
)

bundle=$(realpath "${bundle}")
test -f "${bundle}/manifest.json"
test -f "${bundle}/SHA256SUMS"
(
  cd "${bundle}"
  sha256sum --check SHA256SUMS
)
source_commit=$(git -C "${repo_root}" rev-parse HEAD)
test -z "$(git -C "${repo_root}" status --porcelain)"

docker buildx build \
  --builder "${builder}" \
  --file "${tool_dir}/Dockerfile.qwen38-ngc-runtime" \
  --build-context "qwen-runtime-bundle=${bundle}" \
  --build-arg "SOURCE_IMAGE=${source_image}" \
  --build-arg "RUNTIME_SOURCE_COMMIT=${source_commit}" \
  --tag "${image}" \
  --load \
  "${repo_root}"

docker image inspect "${image}" \
  --format 'image={{.Id}} layers={{len .RootFS.Layers}} size={{.Size}}'

if [[ -n ${RUNTIME_GPU:-} ]]; then
  docker run --rm --device "nvidia.com/gpu=${RUNTIME_GPU}" \
    "${image}" python /opt/venv/libexec/verify_qwen38_runtime.py \
      --foundation ngc --require-gpu
fi
