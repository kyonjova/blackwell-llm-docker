#!/usr/bin/env bash
# Provide the BuildKit worker that builds linux/arm64 (DGX Spark) wheels and
# images under QEMU on the amd64 wheel builder.
set -euo pipefail

builder=${1:-lil-wheel-cu134-arm64}
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The interpreter must be registered by the host (qemu-user-static) with the
# fix-binary flag so that it also runs inside containers.
binfmt=/proc/sys/fs/binfmt_misc/qemu-aarch64
if ! grep -q '^enabled' "${binfmt}" 2>/dev/null || ! grep -q '^flags:.*F' "${binfmt}"; then
  printf 'linux/arm64 emulation is not registered with the F flag (%s)\n' "${binfmt}" >&2
  exit 1
fi
if ! docker buildx inspect "${builder}" >/dev/null 2>&1; then
  docker buildx create --name "${builder}" --driver docker-container \
    --platform linux/arm64 --config "${tool_dir}/buildkitd.toml" >/dev/null
fi
docker buildx inspect --bootstrap "${builder}" | grep -q 'linux/arm64'
printf 'builder=%s platform=linux/arm64\n' "${builder}"
