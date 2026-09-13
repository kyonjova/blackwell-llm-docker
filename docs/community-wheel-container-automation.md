# Community containers assembled from published wheels

Status: **implemented**. Scheduled operation requires the workflow on `main`.
Successful publication proves the checks recorded in the release receipt; it
does not establish full-model accuracy or performance parity.

## Source and release contract

`tools/jovian_wheel_runtime/community-channel.json` declares the `jovian-judgement`
channel and the source branch for each of its six components. The LMCache branch
is `integration/local-inference-lab`, independent of the serving channel name.
Git cannot represent both `dev` and `dev/local-inference-lab` in the same
repository. The existing `dev` branch is preserved.

LMCache wheels must include the attributed filesystem and checkpoint changes
reviewed in #49, #50, #51, #55, #56, #61, #62, #65 and #66. PR #49 includes #67.
The publisher verifies ancestry, required modules and byte equality of every
tracked Python module with the wheel. Native extensions are rebuilt from the
same complete source. B12X remains a separate package from its `master` source;
FlashInfer must not supply its files or plugins.

The configured bootstrap pairs identify previously published, source-audited
CI artifacts and their exact runtime branch heads. A bootstrap is eligible only
while that branch still points to the declared commit. It is not permission to
retain an outdated wheel after a runtime source change. Once publishers are
merged and branch builds exist, their ordinary releases replace these inputs.

## Triggering and batching

`.github/workflows/community-container-release.yml` checks component releases
every five minutes and on relevant recipe pushes. GitHub may delay scheduled
runs; five minutes is a requested interval, not a delivery guarantee.
`repository_dispatch` with type `component-wheel-published` and manual dispatch
also request a scan. Dispatch payloads are not used as build instructions.
Polling needs no cross-repository write credential. An optional GitHub App can
send immediate dispatches; its absence does not prevent periodic operation.

Each scan selects complete wheel releases from the declared source branches.
A relevant source change without a wheel leaves the assembly pending. Uploads
must include the manifest, source archive and checksum; partial releases cannot
enter a container build. ABI incompatibility or missing LMCache source proof
fails the scan. Recipe and component identities form one SHA-256 assembly ID.
An ID with a complete publication receipt does not trigger another build.
A release with missing assets is retried; uploads are completed before a draft
release becomes public. The receipt binds the runtime manifest checksum and
registry digest to the assembly ID.

Changes observed during one interval are assembled together. GitHub concurrency
serializes channel jobs, and the frank2 worker lock serializes native compilation.
The rootless builder retains its bounded memory/CPU allocation and persistent
download, object and Docker layer caches. Other component wheels are downloaded,
not recompiled by the container job.
`LIL_COMPONENT_ARCHIVE_CACHE` names the persistent release-archive directory.
Archives are keyed by SHA-256 and verified on every reuse. Unchanged component
archives require no network transfer; only missing content is downloaded.

The Docker repository runner has a 12 GiB memory throttle threshold and 16 GiB
hard limit, declared in
`tools/jovian_wheel_runtime/systemd/lil-wheel-actions-runner@blackwell-llm-docker.service.d/memory.conf`.
Buildx buffers OCI layer transfers in the client process, outside the 256 GiB
native compiler allocation. Other repository runners retain their 3/4 GiB limits.

## Names and publication

The immutable image tag is
`ghcr.io/local-inference-lab/vllm:jovian-judgement-beta-YYYYMMDD-ASSEMBLY_ID`,
where `ASSEMBLY_ID` is the first 16 hex characters of the full assembly hash.
The exact digest, full ID, component release URLs, source commits and wheel
hashes are retained in GitHub Releases and the installed runtime manifest at
`/opt/venv/share/lil-runtime/manifest.json`.

Publication runs only from `main`. The alias `jovian-judgement-beta` moves only
when source branches have not advanced during the build. The explicitly named
CI branch performs build and qualification with a read-only GitHub token; it
cannot publish an image or move the alias. Repository administrators must protect
write access to `main` and the trusted self-hosted qualification branch.
No running serving container is restarted or updated by publication.

The publisher verifies 67 preserved NGC foundation layers plus one application
layer, runs native GPU checks, and exercises LMCache checkpoint storage, disk
restart and vLLM allocator contract tests. `LIL_RUNTIME_QUALIFICATION_GPU` must
name a reserved GPU UUID. A busy GPU fails the check; no workload is stopped.
Tests use the installed packages and mount only their source-matched tests.
Full GLM cache serving and matched throughput qualification are separate gates.
Stable promotion is not automated by this beta workflow.
