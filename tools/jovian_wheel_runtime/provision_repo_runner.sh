#!/usr/bin/env bash
# Register one repository listener against the shared CUDA wheel build worker.
set -euo pipefail

repository=${1:?Pass an owner/repository name}
runner_id=${2:?Pass a lowercase service-safe runner identifier}
registration_token=${GITHUB_RUNNER_TOKEN:?Set GITHUB_RUNNER_TOKEN}
runner_user=github-flashinfer
runner_home=/var/lib/github-flashinfer
runner_version=2.337.0
runner_archive_sha256=70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613
runner_archive_url="https://github.com/actions/runner/releases/download/v${runner_version}/actions-runner-linux-x64-${runner_version}.tar.gz"
runner_dir="/opt/actions-runner-${runner_id}"
runner_name="frank2-${runner_id}-cu134-sm120-wheel-builder"
work_dir="${runner_home}/work-${runner_id}"
unit_source="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/systemd/lil-wheel-actions-runner@.service"
unit_target=/etc/systemd/system/lil-wheel-actions-runner@.service

if [[ ${EUID} -ne 0 ]]; then
  printf 'Run this provisioning command as root.\n' >&2
  exit 1
fi
if [[ $(hostname -s) != frank2 ]]; then
  printf 'This runner resource profile is qualified only for frank2.\n' >&2
  exit 1
fi
if [[ ! ${runner_id} =~ ^[a-z0-9-]+$ ]]; then
  printf 'Runner identifier must contain only lowercase letters, digits, and hyphens.\n' >&2
  exit 1
fi
id "${runner_user}" >/dev/null
getent group "${runner_user}" >/dev/null
test -S /run/lil-flashinfer-docker/docker.sock

cache_dir=/var/cache/lil-wheel-runner
archive="${cache_dir}/actions-runner-linux-x64-${runner_version}.tar.gz"
install -d -m 0755 "${cache_dir}"
if [[ ! -f ${archive} ]] || ! printf '%s  %s\n' \
    "${runner_archive_sha256}" "${archive}" | sha256sum --check --status; then
  temporary=$(mktemp --tmpdir="${cache_dir}" actions-runner.XXXXXX.tar.gz)
  trap 'rm -f "${temporary}"' EXIT
  curl --fail --location --retry 3 "${runner_archive_url}" --output "${temporary}"
  printf '%s  %s\n' "${runner_archive_sha256}" "${temporary}" \
    | sha256sum --check -
  install -m 0644 "${temporary}" "${archive}"
fi

if [[ -e ${runner_dir}/.runner ]]; then
  configured_url=$(jq -r .gitHubUrl "${runner_dir}/.runner")
  expected_url="https://github.com/${repository}"
  if [[ ${configured_url} != "${expected_url}" ]]; then
    printf 'Runner %s is registered to %s, not %s.\n' \
      "${runner_id}" "${configured_url}" "${expected_url}" >&2
    exit 1
  fi
else
  install -d -o "${runner_user}" -g "${runner_user}" "${runner_dir}" "${work_dir}"
  tar -xzf "${archive}" -C "${runner_dir}"
  chown -R "${runner_user}:${runner_user}" "${runner_dir}" "${work_dir}"
  runuser -u "${runner_user}" -- env HOME="${runner_home}" \
    "${runner_dir}/config.sh" \
      --unattended \
      --url "https://github.com/${repository}" \
      --token "${registration_token}" \
      --name "${runner_name}" \
      --labels lil-wheel-builder \
      --work "${work_dir}" \
      --disableupdate \
      --replace
fi

test "$(runuser -u "${runner_user}" -- \
  "${runner_dir}/bin/Runner.Listener" --version)" = "${runner_version}"
install -m 0644 "${unit_source}" "${unit_target}"
systemctl daemon-reload
systemctl enable --now "lil-wheel-actions-runner@${runner_id}.service"
systemctl --no-pager --full status "lil-wheel-actions-runner@${runner_id}.service"
