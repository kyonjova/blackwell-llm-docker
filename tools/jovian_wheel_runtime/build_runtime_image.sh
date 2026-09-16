#!/usr/bin/env bash
# Build the shared serving image from verified, reusable component wheels.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/build_runtime_image.py" "$@"
