#!/usr/bin/env bash
# test-spark-builder-stale-guard.sh -- the dgx-spark-builder wrapper must
# refuse to run when a previous (killed) build left a .pre-spark.* backup,
# and must dry-run cleanly on a pristine tree without leaving residue.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wrapper="${repo_root}/dgx-spark-builder/build-spark-cu132.sh"
[[ -f "${wrapper}" ]] || { echo "wrapper missing: ${wrapper}" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

make_stub() { # $1 = stub repo dir
  mkdir -p "${1}"
  printf '%s\n' 'FROM scratch' > "${1}/Dockerfile.vllm-b12x-cu132"
  printf '%s\n' '#!/bin/sh' 'exit 0' > "${1}/build-vllm-b12x-cu132.sh"
  chmod +x "${1}/build-vllm-b12x-cu132.sh"
}

# case 1: a leftover backup must abort BEFORE any mutation
make_stub "${tmp}/dirty"
touch "${tmp}/dirty/Dockerfile.vllm-b12x-cu132.pre-spark.999"
if (cd "${tmp}/dirty" && bash "${wrapper}" --dry-run >out.log 2>err.log); then
  echo "FAIL: stale backup did not abort the run" >&2
  cat "${tmp}/dirty/err.log" >&2
  exit 1
fi
grep -q "pre-spark" "${tmp}/dirty/err.log" \
  || { echo "FAIL: no stale-backup diagnostic" >&2; cat "${tmp}/dirty/err.log" >&2; exit 1; }
[[ "$(cat "${tmp}/dirty/Dockerfile.vllm-b12x-cu132")" == 'FROM scratch' ]] \
  || { echo "FAIL: Dockerfile mutated on abort" >&2; exit 1; }
[[ -z "$(find "${tmp}/dirty" -name '*.pre-spark.*' ! -name '*.pre-spark.999' -print)" ]] \
  || { echo "FAIL: unexpected backup created on abort" >&2; exit 1; }
echo "case 1 OK: stale backup aborts before mutation"

# case 2: a pristine tree dry-runs to completion and leaves no residue
make_stub "${tmp}/clean"
(cd "${tmp}/clean" && bash "${wrapper}" --dry-run >out.log 2>err.log) \
  || { echo "FAIL: dry-run exited nonzero" >&2; cat "${tmp}/clean/out.log" "${tmp}/clean/err.log" >&2; exit 1; }
grep -q "DRY RUN complete" "${tmp}/clean/err.log" \
  || { echo "FAIL: dry-run did not complete" >&2; cat "${tmp}/clean/out.log" "${tmp}/clean/err.log" >&2; exit 1; }
# Residue of interest is only the wrapper's backup file; out.log/err.log are
# the test's own files and MEM_PEAK_FILE is a mktemp outside this tree.
[[ -z "$(find "${tmp}/clean" -name '*.pre-spark.*' -print)" ]] \
  || { echo "FAIL: residue left by clean run" >&2; exit 1; }
[[ "$(cat "${tmp}/clean/Dockerfile.vllm-b12x-cu132")" == 'FROM scratch' ]] \
  || { echo "FAIL: Dockerfile not restored byte-identical after dry-run" >&2; exit 1; }
echo "case 2 OK: pristine tree dry-runs and leaves no residue"

echo "stale-guard test passed"
