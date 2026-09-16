#!/usr/bin/env bash
# Compatibility path; the canonical recipe serves all registered model profiles.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/build_runtime_image.sh" "$@"
