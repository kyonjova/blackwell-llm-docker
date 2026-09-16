#!/usr/bin/env bash
# Serialize compilation jobs that share one BuildKit worker and cache namespace.
set -euo pipefail

lock_path=${LIL_WHEEL_BUILD_LOCK:-/var/lib/github-flashinfer/locks/cu134-sm120-build.lock}
mkdir -p "$(dirname "${lock_path}")"
exec flock --exclusive --wait 21600 "${lock_path}" "$@"
