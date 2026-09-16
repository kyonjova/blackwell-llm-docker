#!/usr/bin/env bash
# Install inside a model-neutral image that already contains its serving packages.
set -euo pipefail
if (($# < 2 || $# > 3)); then
    echo 'Usage: install.sh FOUNDATION_INSPECT_JSON RUNTIME_LOCK_SHA256 [BOOTSTRAP_EXECUTABLE]' >&2
    exit 2
fi
source_root=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$source_root"
bootstrap=()
if (($# == 3)); then bootstrap=(--bootstrap "$3"); fi
# Reject inherited policy and existing destinations before installing anything.
exec /opt/venv/bin/python -m runtime.image_install \
    --image-inspect "$1" --runtime-lock-sha256 "$2" "${bootstrap[@]}" \
    >/dev/null
