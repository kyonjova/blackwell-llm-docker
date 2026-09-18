#!/usr/bin/env bash
# Disk-backed Engram/PLE compiles the B12X io_uring reader at runtime.
set -euo pipefail

readonly liburing_package_version=2.5-1build1
readonly liburing_api_version=2.5

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
  "liburing2=${liburing_package_version}" \
  "liburing-dev=${liburing_package_version}"

command -v cc >/dev/null
test "$(pkg-config --modversion liburing)" = "${liburing_api_version}"
test "$(dpkg-query -W -f='${Version}' liburing2)" = "${liburing_package_version}"
test "$(dpkg-query -W -f='${Version}' liburing-dev)" = "${liburing_package_version}"
