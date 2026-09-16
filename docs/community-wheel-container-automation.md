# Community containers assembled from published wheels

Status: **implemented**. Scheduled operation requires the workflow on `main`.
Successful publication proves the checks recorded in the release receipt; it
does not establish full-model accuracy or performance parity.

## Source and release contract

`tools/jovian_wheel_runtime/community-channel.json` uses schema
`local-inference-container-channel/v2` to declare shared component dependencies
and two source channels:

| Source channel | vLLM branch | B12X branch | Mutable Docker tag |
|---|---|---|---|
| `main` | `dev/jovian-judgement` | `master` | `jovian-judgement` |
| `beta` | `integration/beta` | `integration/beta` | `jovian-judgement-beta` |

Both channels use the same Dockerfile, publisher, toolchain locks, model launcher
and qualification tests. Channel entries can override component branches and
the image tag, not the build recipe. A recipe change applies to both channels.
Integration branches contain attributed Git merges of selected source PRs;
merging into integration does not merge those PRs into JJ or B12X master.

FlashInfer, LMCache, InstantTensor and NCCL are shared. The LMCache branch
is `integration/local-inference-lab`, independent of the serving channel name.
Git cannot represent both `dev` and `dev/local-inference-lab` in the same
repository. The existing `dev` branch is preserved.

LMCache wheels must include the attributed filesystem and checkpoint changes
reviewed in #49, #50, #51, #55, #56, #61, #62, #65 and #66. PR #49 includes #67.
The publisher verifies ancestry, required modules and byte equality of every
tracked Python module with the wheel. Native extensions are rebuilt from the
same complete source. B12X remains a separate package from its selected source;
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

The reusable workflow `.github/workflows/component-container-dispatch.yml`
requests the resolver directly after a component publisher succeeds. Callers pin
the reusable workflow to a reviewed commit and depend on the publication job,
including successful verification of an already published release. Failed builds
and stable promotions do not request a container rebuild. The notification executes on
a GitHub-hosted runner without checking out component code. Its repository,
workflow, and `main` ref are fixed; it neither accepts source overrides nor
publishes an image itself. It retries a failed request up to three times and
reports an error rather than silently relying on the schedule.

Immediate notification requires the Actions secret
`LIL_CONTAINER_DISPATCH_TOKEN` in vLLM, B12X, FlashInfer, LMCache, InstantTensor,
and nccl-canonical. Use a fine-grained token restricted to
`local-inference-lab/blackwell-llm-docker` with **Actions: read and write**, or an
equivalently restricted GitHub App token. No Contents or Packages write access is
needed. A repository's ordinary `GITHUB_TOKEN` cannot dispatch into another
repository. An organization secret may be shared only with these six publishers.
Rotate the token before expiry; never commit or print its value. Without the
secret, immediate notification is unsupported; scheduled scans remain available.
The vLLM and B12X notification jobs are enabled with the repository variable
`LIL_CONTAINER_DISPATCH_ENABLED=true` after the secret is configured. A missing
or expired token with that variable enabled fails the notification visibly.

Every push to either configured vLLM or B12X source branch triggers that
repository's single wheel workflow. Build scripts, native cache keys and release
prefixes are identical across branches. Wheels are identified by source commit,
not by channel; an identical source release can be consumed by both channels.
The container resolver runs after wheel publication through notification or a
scheduled scan. No component changes are automatically merged between branches.

Each scan selects complete wheel releases from the declared source branches.
A relevant source change without a wheel leaves the assembly pending. Uploads
must include the manifest, source archive and checksum; partial releases cannot
enter a container build. ABI incompatibility or missing LMCache source proof
fails the scan. A channel waiting for wheels does not block a complete channel.
Recipe, channel and component identities form one SHA-256 assembly ID.
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
`ghcr.io/local-inference-lab/vllm:CHANNEL_TAG-YYYYMMDD-ASSEMBLY_ID`,
where `CHANNEL_TAG` is `jovian-judgement` or `jovian-judgement-beta`, and
where `ASSEMBLY_ID` is the first 16 hex characters of the full assembly hash.
The exact digest, full ID, component release URLs, source commits and wheel
hashes are retained in GitHub Releases and the installed runtime manifest at
`/opt/venv/share/lil-runtime/manifest.json`.

Publication runs only from the Docker repository's `main`. Each channel alias
moves only when its source branches and the recipe have not advanced during the
build. An obsolete build retains an immutable image but cannot move either
alias. The explicitly named
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
The non-beta tag identifies the JJ/master source channel, not a claim of broad
model qualification. GitHub assembly receipts remain prereleases; byte-identical
stable wheel promotion is a separate component workflow.
