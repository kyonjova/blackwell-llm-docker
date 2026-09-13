#!/usr/bin/env bash
# Build and verify the immutable CUDA 13.4 Python foundation artifact.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${tool_dir}/../.." && pwd)"
lock_path="${tool_dir}/foundation.lock"
output_dir=${1:-"${tool_dir}/dist/foundation"}

value() {
  local key=$1
  awk -F= -v key="${key}" \
    '$1 == key {sub(/^[^=]*=/, ""); print; found=1} END {exit !found}' \
    "${lock_path}"
}

source_image=$(value source.image)
source_commit=$(git -C "${repo_root}" rev-parse HEAD)
source_tree=$(git -C "${repo_root}" rev-parse 'HEAD^{tree}')
source_date_epoch=$(value source.date-epoch)
test -z "$(git -C "${repo_root}" status --porcelain)"
repository=${GITHUB_REPOSITORY:-local-inference-lab/blackwell-llm-docker}
release_tag=${FOUNDATION_RELEASE_TAG:-"jovian-cu134-foundation-beta-${source_commit}"}
if ! docker image inspect "${source_image}" >/dev/null 2>&1; then
  docker image pull "${source_image}"
fi

mkdir -p "$(dirname "${output_dir}")"
if ! mkdir "${output_dir}"; then
  printf 'Output path already exists or is being built: %s\n' \
    "${output_dir}" >&2
  exit 1
fi
mkdir -p "${output_dir}/bundle/wheels"

builder=${BUILDX_BUILDER:-$(value buildx.builder)}
docker buildx build \
  --builder "${builder}" \
  --file "${tool_dir}/Dockerfile" \
  --build-arg "SOURCE_IMAGE=${source_image}" \
  --build-arg "SOURCE_DATE_EPOCH=${source_date_epoch}" \
  --target artifacts \
  --output "type=local,dest=${output_dir}/repacked" \
  "${repo_root}"
cp -a "${output_dir}/repacked/wheels/." "${output_dir}/bundle/wheels/"
cp "${output_dir}/repacked/repack-provenance.json" \
  "${output_dir}/bundle/repack-provenance.json"
cp "${output_dir}/repacked/torch-native-support-provenance.json" \
  "${output_dir}/bundle/torch-native-support-provenance.json"

for name in torch torchvision triton triton-kernels flash-attn; do
  actual=$(jq -r --arg name "${name}" \
    '.packages[] | select((.name | ascii_downcase | gsub("_"; "-")) == $name) | .source_record_sha256' \
    "${output_dir}/bundle/repack-provenance.json")
  test "${actual}" = "$(value "wheel.${name}.installed-record.sha256")"
done

wheel_metadata() {
  local wheel=$1 field=$2
  unzip -p "${wheel}" '*.dist-info/METADATA' \
    | awk -F': ' -v field="${field}" '$1 == field && !seen {print $2; seen=1}'
}

requirements="${output_dir}/bundle/requirements-foundation.txt"
github_requirements="${output_dir}/bundle/requirements-github.txt"
: > "${requirements}"
: > "${github_requirements}"
packages='[]'
while IFS= read -r wheel; do
  package=$(wheel_metadata "${wheel}" Name)
  version=$(wheel_metadata "${wheel}" Version)
  digest=$(sha256sum "${wheel}" | awk '{print $1}')
  printf '%s==%s --hash=sha256:%s\n' \
    "${package}" "${version}" "${digest}" >> "${requirements}"
  url="https://github.com/${repository}/releases/download/${release_tag}/$(basename "${wheel}")"
  printf '%s @ %s --hash=sha256:%s\n' \
    "${package}" "${url}" "${digest}" >> "${github_requirements}"
  normalized=$(tr '[:upper:]_' '[:lower:]-' <<<"${package}" | tr -d '\n')
  case "${normalized}" in
    torch|torchvision|triton|triton-kernels|flash-attn)
      provenance=hash-verified-installed-distribution-repack
      origin_sha=$(value "wheel.${normalized}.installed-record.sha256")
      ;;
    local-inference-torch-native-support)
      provenance=hash-locked-native-loader-closure
      origin_sha=$(sha256sum "${tool_dir}/torch_native_support.json" | awk '{print $1}')
      ;;
    *)
      printf 'Unexpected foundation package: %s\n' "${package}" >&2
      exit 1
      ;;
  esac
  packages=$(jq \
    --arg name "${package}" \
    --arg version "${version}" \
    --arg file "$(basename "${wheel}")" \
    --arg sha256 "${digest}" \
    --arg url "${url}" \
    --arg provenance "${provenance}" \
    --arg origin_sha256 "${origin_sha}" \
    '. + [{name: $name, version: $version, file: $file, sha256: $sha256,
      url: $url,
      provenance: $provenance, origin_sha256: $origin_sha256}]' \
    <<<"${packages}")
done < <(find "${output_dir}/bundle/wheels" -maxdepth 1 -type f \
  -name '*.whl' | sort)
test "$(jq length <<<"${packages}")" -eq 6
test "$(sort -u "${requirements}" | wc -l)" -eq 6

source_image_id=$(docker image inspect "${source_image}" --format '{{.Id}}')
jq -n \
  --arg status research-only \
  --arg source_image "${source_image}" \
  --arg source_image_id "${source_image_id}" \
  --arg publisher_repository "https://github.com/${repository}.git" \
  --arg publisher_commit "${source_commit}" \
  --arg publisher_tree "${source_tree}" \
  --arg release_tag "${release_tag}" \
  --arg python_version "$(value python.version)" \
  --arg cuda_version "$(value cuda.version)" \
  --arg pytorch_commit "$(value pytorch.commit)" \
  --arg nccl_commit "$(value nccl.commit)" \
  --arg cudnn_version "$(value host.cudnn.version)" \
  --arg cusparselt_version "$(value host.cusparselt.version)" \
  --arg cuda13_driver_minimum "$(value host.driver.cuda13.minimum)" \
  --arg cuda134_feature_driver "$(value host.driver.cuda134.features)" \
  --argjson packages "${packages}" \
  '{
    schema: "local-inference-jovian-foundation-bundle/v1",
    status: $status,
    scope: "Python ABI foundation wheels used by the CUDA 13.4 serving runtime",
    source: {
      image: $source_image,
      image_id: $source_image_id,
      publisher: {
        repository: $publisher_repository,
        commit: $publisher_commit,
        tree: $publisher_tree
      },
      pytorch_commit: $pytorch_commit,
      nccl_commit: $nccl_commit
    },
    release_tag: $release_tag,
    runtime: {
      python: $python_version,
      cuda: $cuda_version,
      cudnn: $cudnn_version,
      cusparselt: $cusparselt_version
    },
    packages: $packages,
    native: {
      nccl: {
        repository: "https://github.com/local-inference-lab/nccl-canonical.git",
        commit: $nccl_commit,
        version: "2.31.2",
        delivery: "local-inference-nccl-cu134 release artifact"
      },
      host_requirements: [
        ("NVIDIA R" + $cuda13_driver_minimum +
          " or newer driver for CUDA 13.x minor-version compatibility"),
        ("NVIDIA R" + $cuda134_feature_driver +
          " or newer driver for CUDA 13.4-specific features")
      ]
    }
  }' > "${output_dir}/bundle/manifest.json"

cp "${lock_path}" "${tool_dir}/foundation-runtime.lock" \
  "${tool_dir}/install_foundation.sh" \
  "${tool_dir}/verify_foundation_gpu.sh" \
  "${output_dir}/bundle/"
mkdir -p "${output_dir}/bundle/tests"
cp "${tool_dir}/tests/smoke_cuda.cu" "${output_dir}/bundle/tests/"
chmod 0755 "${output_dir}/bundle/install_foundation.sh"
chmod 0755 "${output_dir}/bundle/verify_foundation_gpu.sh"
(
  cd "${output_dir}/bundle"
  find wheels -type f -print0 | sort -z | xargs -0 sha256sum
  sha256sum foundation-runtime.lock foundation.lock install_foundation.sh \
    manifest.json repack-provenance.json requirements-foundation.txt \
    requirements-github.txt tests/smoke_cuda.cu \
    torch-native-support-provenance.json verify_foundation_gpu.sh
) > "${output_dir}/bundle/SHA256SUMS"

archive="${output_dir}/jovian-cu134-foundation.tar.zst"
tar --sort=name --mtime="@${source_date_epoch}" \
  --owner=0 --group=0 --numeric-owner --zstd \
  -C "${output_dir}/bundle" -cf "${archive}" .
(
  cd "${output_dir}"
  sha256sum "$(basename "${archive}")"
) > "${archive}.sha256"
printf '%s\n' "${archive}"
