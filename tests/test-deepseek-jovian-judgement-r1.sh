#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
builder="${repo_root}/build-deepseek-jovian-judgement-cu133-torch213.sh"
target_compose="${repo_root}/examples/docker-compose-ds4-jovian-judgement-r1.yml"
dspark_compose="${repo_root}/examples/docker-compose-ds4-dspark-jovian-judgement-r1.yml"
composition_root="${repo_root}/patches/releases/jovian-judgement-ds4-r1"

for component in vllm b12x lmcache; do
  lock="${composition_root}/${component}/integration.lock.json"
  patch="${composition_root}/${component}/integration.patch"

  test -f "${lock}"
  test -f "${patch}"
  jq -e '
    .schema_version == 1 and
    (.composition_strategy == "merge" or
      .composition_strategy == "cherry_pick") and
    (.base.repository | type == "string") and
    (.base.ref | type == "string") and
    (.base.commit | test("^[0-9a-f]{40}$")) and
    (.result.tree | test("^[0-9a-f]{40}$")) and
    (.result.patch_sha256 | test("^[0-9a-f]{64}$")) and
    ((.research_changes // []) | length) == 0 and
    (.source_patches | length) == 0
  ' "${lock}" >/dev/null
  echo "$(jq -er '.result.patch_sha256' "${lock}")  ${patch}" |
    sha256sum -c - >/dev/null
done

jq -e '
  .composition_strategy == "cherry_pick" and
  .base.ref == "refs/heads/dev/jovian-judgement" and
  .base.commit == "c085b910ebd4a8c89c2c4085cbf17ccaf15a384c" and
  .result.tree == "28bc825a1321bb480fc3294179fd34afeb468389" and
  [.pull_requests[].number] == [628, 630] and
  .pull_requests[0].head ==
    "cbb66bdff1763c174ebc794a7f968930e956580f" and
  .pull_requests[1].head ==
    "5b6fb80f5c868b62da2c01c3f52861b34c84d8ac"
' "${composition_root}/vllm/integration.lock.json" >/dev/null

jq -e '
  .base.ref == "refs/heads/master" and
  .base.commit == "aa90a277a61f9ded46c0f504e37a955b7706659b" and
  .composition_strategy == "cherry_pick" and
  .result.tree == "8a5b9bfbf59ad61d87efdc8017b91a269d5a319c" and
  [.pull_requests[].number] == [246, 302] and
  .pull_requests[0].head ==
    "2a2baaf3979c1f47f8fd139aab5c6eeea35003e4" and
  .pull_requests[1].head ==
    "f8a7b9c4070754ed4399c71fc4306d8970711e1c"
' "${composition_root}/b12x/integration.lock.json" >/dev/null

jq -e '
  .base.ref == "refs/heads/release/v0.5.2-glm52-dcp-base" and
  .base.commit == "a128b2e286ebb3556cb43124149e600ff99fe481" and
  .composition_strategy == "merge" and
  .result.tree == "eb4c227f68a4e1c45d6b8edf6b4934e18f6d1f8b" and
  (.pull_requests | length) == 14 and
  .pull_requests[-1].number == 44 and
  .pull_requests[-1].head ==
    "97ede799d6605ca1bd5285582df4e74a3d3c7b0d"
' "${composition_root}/lmcache/integration.lock.json" >/dev/null

output="$(
  RELEASE_DATE=20260903 \
  REVISION=r1 \
  COMPOSITION_ROOT=patches/releases/jovian-judgement-ds4-r1 \
  FLASHINFER_WHEEL_IMAGE=voipmonitor/vllm:flashinfer-wheels-fi1ac6942-cu133-torch213-20260820-r1@sha256:477a3b55b973df48b08a6dfae4a2a1e64c975a990dda22f65e31acd5217b86bb \
  FLASHINFER_REF=integration/main-pr4393-pcie-ipc-qualified-20260807 \
  FLASHINFER_COMMIT=1ac6942776b383c6b03c7a5805a22e72a3e3349f \
  PRINT_RELEASE_CONFIG=1 \
    "${builder}"
)"
grep -Fxq 'release=jovian-judgement-deepseek-v4-flash-cu133-torch213' <<<"${output}"
grep -Fxq 'revision=r1' <<<"${output}"
grep -Fxq 'vllm_ref=dev/jovian-judgement' <<<"${output}"
grep -Fxq 'vllm_tree=28bc825a1321bb480fc3294179fd34afeb468389' <<<"${output}"
grep -Fxq 'b12x_tree=8a5b9bfbf59ad61d87efdc8017b91a269d5a319c' <<<"${output}"
grep -Fq '20260903-r1' <<<"${output}"
grep -Fq 'releases/jovian-judgement-ds4-r1/vllm/integration.patch' <<<"${output}"
grep -Fq 'releases/jovian-judgement-ds4-r1/b12x/integration.patch' <<<"${output}"
grep -Fq 'base=voipmonitor/vllm@sha256:03b67e53dda73c3fa317d4cb529ad38a220c51c7365ee8d54c16e5063fcc54e2' <<<"${output}"
grep -Fxq 'runtime_foundation=1' <<<"${output}"

target_config="$(docker compose -f "${target_compose}" config)"
grep -Fq 'MODE: dspark-mtp0' <<<"${target_config}"
grep -Fq 'BACKEND: b12x-a8-dglin' <<<"${target_config}"
grep -Fq 'TP_SIZE: "2"' <<<"${target_config}"
grep -Fq 'MAX_NUM_SEQS: "32"' <<<"${target_config}"
grep -Fq 'MAX_NUM_BATCHED_TOKENS: "4096"' <<<"${target_config}"
grep -Fq 'ALLREDUCE_MODE: auto' <<<"${target_config}"
grep -Fq 'jovian-judgement-vllm28bc825-b12x8a5b9bf' <<<"${target_config}"
grep -Fq 'LMCACHE_MODE: "off"' <<<"${target_config}"
! grep -Fq 'KV_OFFLOADING_SIZE:' <<<"${target_config}"
! grep -Fq 'NATIVE_L2_' <<<"${target_config}"

dspark_config="$(docker compose -f "${dspark_compose}" config)"
grep -Fq 'MODE: dspark' <<<"${dspark_config}"
grep -Fq 'DSPARK_TOKENS: "5"' <<<"${dspark_config}"
grep -Fq 'MAX_NUM_SEQS: "8"' <<<"${dspark_config}"
grep -Fq 'GRAPH: auto' <<<"${dspark_config}"
grep -Fq 'MAX_MODEL_LEN: "1048576"' <<<"${dspark_config}"
grep -Fq 'GPU_MEMORY_UTILIZATION: "0.975"' <<<"${dspark_config}"
grep -Fq 'jovian-judgement-vllm28bc825-b12x8a5b9bf' <<<"${dspark_config}"
grep -Fq 'LMCACHE_MODE: "off"' <<<"${dspark_config}"
! grep -Fq 'KV_OFFLOADING_SIZE:' <<<"${dspark_config}"
! grep -Fq 'NATIVE_L2_' <<<"${dspark_config}"

for component in VLLM B12X LMCACHE; do
  grep -Fq -- "--build-arg \"${component}_UPSTREAM_BASE=\${${component}_UPSTREAM_BASE}\"" "${builder}"
  grep -Fq -- "--build-arg \"${component}_MERGE_HEADS=\${${component}_MERGE_HEADS}\"" "${builder}"
done

grep -Fq -- '--build-arg "RUNTIME_FOUNDATION=${runtime_foundation}"' "${builder}"
grep -Fq -- '--build-arg "RUNTIME_FOUNDATION_IMAGE=${runtime_foundation_image}"' "${builder}"
