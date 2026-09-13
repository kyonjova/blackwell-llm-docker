#!/usr/bin/env bash
# Build a container from the same immutable wheel bundle used by direct hosts.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${tool_dir}/../.." && pwd)"
bundle=${1:?Pass the assembled Qwen3.8 runtime bundle directory}
image=${2:?Pass the output container image name and tag}
builder=${BUILDX_BUILDER:-lil-wheel-cu134-sm120}

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
  --file "${tool_dir}/Dockerfile.qwen38-runtime" \
  --build-context "qwen-runtime-bundle=${bundle}" \
  --build-arg "RUNTIME_SOURCE_COMMIT=${source_commit}" \
  --tag "${image}" \
  --load \
  "${repo_root}"

docker image inspect "${image}" \
  --format 'image={{.Id}} layers={{len .RootFS.Layers}} size={{.Size}}'
