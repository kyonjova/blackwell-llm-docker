# dgx-spark-builder — generic vLLM/B12X image builder for DGX Spark

Builds aarch64/sm_121 (Grace + GB10) serving images of the
local-inference-lab vLLM stack from source, by retargeting upstream's
x86/sm_120 build systems in place and patching the handful of things that
break on arm64. Model and release choice is a *profile* (`build-*.env`),
not a fork of a script.

Two flavors, two wrappers, one profile format:

| flavor | wrapper | upstream build system it retargets | CUDA / torch / NCCL | when |
|---|---|---|---|---|
| **cu132** | `build-spark-cu132.sh` | `Dockerfile.vllm-b12x-cu132` + `build-vllm-b12x-cu132.sh` (single multi-stage build) | 13.2 / 2.13.0+cu132 wheels / 2.30.4 | every release through R35; the measured baseline |
| **cu133** | `build-spark-cu133.sh` | the community image lineage: `Dockerfile.kimi-k3-cu133-torch213-base` → `Dockerfile.flashinfer-cu133-torch213-wheels` → `Dockerfile.deepseek-infernal-invocation-cu133-torch213` | 13.3 / 2.13.0 from source / 2.31.2 | upstream's qualification baseline since ~R20; first aarch64 build 2026-09-11 (R35) |

The images differ in toolchain only: same source pins, same launchers, same
serving env except the loader block (NCCL/compat paths). Keep both flavors
buildable; the cu132 one is what every benchmark in the deployment README
was measured on until the cu133 A/B lands.

**Currently set to build: local-inference-lab/vllm Branch: dev/jovian-judgement
(canonical head profiles) or the voipmonitor `integration/*` release branches
(`build-jj-r##.env`).**

## Layout and usage

This directory lives inside a checkout of
[blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker):

```
blackwell-llm-docker/
├── Dockerfile.vllm-b12x-cu132                          # cu132 upstream build system (untouched)
├── build-vllm-b12x-cu132.sh                            # cu132 upstream driver (untouched)
├── Dockerfile.kimi-k3-cu133-torch213-base              # cu133 foundation (untouched)
├── Dockerfile.flashinfer-cu133-torch213-wheels         # cu133 FlashInfer wheels (untouched)
├── Dockerfile.deepseek-infernal-invocation-cu133-torch213   # cu133 JJ overlay (untouched; name is a fossil)
├── tests/deepseek-infernal-cu133-pip-check.allowlist   # cu133 final gate (untouched)
├── patches/                                            # VLLM_PATCH_FILE / B12X_PATCH_FILE inputs
└── dgx-spark-builder/
    ├── build-spark-cu132.sh        # cu132 wrapper
    ├── build-spark-cu133.sh        # cu133 wrapper (three phases)
    ├── probe-image.sh              # post-build image probe (flavor-aware)
    ├── build-example.env           # annotated cu132 profile template
    ├── build-jj-r##.env            # cu132 release profiles (voipmonitor integration branches)
    ├── build-jj-r##-cu133.env      # cu133 release profiles
    └── README.md
```

```
cd blackwell-llm-docker/dgx-spark-builder
# cu132
./build-spark-cu132.sh --dry-run build-jj-r35.env     # validate rewrites, no build, any host
./build-spark-cu132.sh --log build-jj-r35.env         # build (hours cold), keep a transcript
./build-spark-cu132.sh build-jj-r35.env -- --build-arg CUBLAS_CUDA13_VERSION=x
# cu133
./build-spark-cu133.sh --dry-run build-jj-r35-cu133.env
./build-spark-cu133.sh --log build-jj-r35-cu133.env   # all three phases; skips phases whose image exists
./build-spark-cu133.sh --phase foundation build-jj-r35-cu133.env   # one phase
# after either
./probe-image.sh <image> build-jj-r35[-cu133].env
```

The script auto-locates the repo root (its parent directory). With exactly
one `*.env` beside it, that file is the profile whatever it is named; with
more than one it refuses rather than guess; with none it warns loudly that it
is falling back to in-script pins.

**Why build inside blackwell-llm-docker at all?** Because that repo *is* the
build system: it owns the multi-stage Dockerfile, the docker build context
(files COPY'd into the image), the env-to-build-arg plumbing, and the stage
caching design. This wrapper deliberately does not fork any of it — it
retargets the Dockerfile in place (with automatic restore) and delegates to
the upstream driver, so upstream improvements flow in with a `git pull` and
the diff we maintain stays six small patches instead of a divergent build
system.

## The cu133 flavor: three phases

Upstream's community images are two-layer overlays on a *runtime
foundation*; the cu133 wrapper reproduces that on aarch64 with three docker
builds, each tagged, each skipped when its image already exists:

| phase | Dockerfile | builds | output image | cold time on a Spark |
|---|---|---|---|---|
| 1 `foundation` | `Dockerfile.kimi-k3-cu133-torch213-base` | `FROM nvcr.io/nvidia/pytorch:26.07-py3` (multi-arch NGC container): patched NCCL 2.31.2 (`nccl-canonical` `canonical/cu133-nccl2312-amd-turin`), **torch 2.13.0 from source** at `TORCH_CUDA_ARCH_LIST=12.1a`, torchvision, xgrammar; runs upstream's `verify_kimi_k3_cu133_base.py` contract | `local/vllm:cu133-torch213-nccl2312-sm121-foundation` (~26 GB) | hours; **built once** and reused across releases until a foundation pin changes |
| 2 `flashinfer` | `Dockerfile.flashinfer-cu133-torch213-wheels` | FlashInfer python + JIT-cache wheels compiled against the foundation (`flashinfer_jit_cache-0.6.18+cu133-…-manylinux_2_28_aarch64.whl`) | `local/vllm:flashinfer-wheels-<sha7>-cu133-torch213-sm121` (~270 MB) | ~30 min; rebuilt only when `FLASHINFER_*` pins change |
| 3 `overlay` | `Dockerfile.deepseek-infernal-invocation-cu133-torch213` | vLLM, B12X, LMCache, exllamav3, InstantTensor, nccl4py, launchers; upstream's runtime contract + pip-check allowlist gate | `local/vllm:<IMAGE_TAG>` (~31 GB) | ~2-3 h cold; minutes when only the final gate reruns |

**Expected outcome per phase.** Phase 1 ends with
`Kimi-K3 CUDA base contract: PASS torch=2.13.0 torchvision=0.28.0 cuda=13.3
nccl=23102 …`. Phase 2 ends with `Successfully installed
flashinfer-jit-cache-0.6.18+cu133 flashinfer-python-0.6.18+cu133`. Phase 3
ends with `DeepSeek Infernal Invocation CUDA 13.3 runtime contract: PASS …
vllm=<VLLM_PACKAGE_VERSION> b12x=… flashinfer=… lmcache=…`, a matched
pip-check allowlist, the wrapper's own in-image asserts (aarch64, torch
2.13.0, CUDA 13.3, `b12x` importable, required launchers), telemetry, and a
build manifest. `PS1: unbound variable` from NGC's `bash.bashrc` in phase 1
is noise.

**The overlay's source contract.** For each of vLLM, B12X and LMCache the
overlay clones the repo, checks out the pin, optionally applies a patch
file, then compares `git write-tree` against `*_INTEGRATION_TREE`. For a
plain pin that is the commit's tree hash, which the wrapper resolves from
the GitHub API (`GH_TOKEN` avoids rate limits) unless the profile supplies
it — rtx6kpro's source locks publish it as `<component>.tree`, and the
`build-jj-r##-cu133.env` profiles carry those values. The overlay builds
LMCache unconditionally, so `LMCACHE_COMMIT` is required even though
LMCache is ignored at serve time on unified memory. The checkout paths are
`/opt/infernal-invocation/{vllm,b12x,lmcache}` — "Infernal Invocation" was
the previous release codename and the Dockerfile kept its paths; the sources
are whatever the profile pins.

**`RUNTIME_FOUNDATION`.** `0` (default) means `BASE_IMAGE` is the raw
foundation and the overlay builds the venv, DeepGEMM, exllamav3,
InstantTensor and nccl4py itself. `1` means `BASE_IMAGE` is a *previous JJ
runtime image built by this wrapper*: the overlay verifies those components
are present and rebuilds only vLLM, B12X and LMCache. That is upstream's
fast-release path; after the first full cu133 build, the next release is
`RUNTIME_FOUNDATION=1 BASE_IMAGE=<previous image>` and a fraction of the
time. The wrapper prints the exact invocation after every successful build.

**Patches the cu133 wrapper applies** (all `on | off | auto`, shape-guarded,
restored on exit like the cu132 ones):

1. `PATCH_GPU_ARCH` — `12.0a`/`120a`/`12.0f`/`12.0` → sm_121 in all three
   Dockerfiles.
2. `PATCH_NCCL_GENCODE` — the foundation compiles NCCL with explicit
   `-gencode arch=compute_120,code=sm_120` (nvcc flags, not the `ARCH_LIST`
   envs); rewritten to sm_121. A foundation built before this patch existed
   (the 2026-09-11 one) runs NCCL via PTX JIT on GB10 — works, one-time JIT
   at first init; rebuild the foundation for native SASS when convenient.
3. `PATCH_MAX_JOBS` — `MAX_JOBS=48/128` literals, **and** the overlay's vLLM
   extension stage which uses `cmake --build --parallel 48` with
   `NVCC_THREADS=4`: 48 × 4 nvcc threads on the Marlin MoE kernels exhausted
   121 GiB on a Spark. Bound to `MAX_JOBS` (20) and `NVCC_THREADS` (1).
4. `PATCH_EXLLAMAV3_AVX` — the same x86 GCC-builtin/AVX stub as cu132,
   anchored on the overlay's exllamav3 build line (`exllamav3/exllamav3_ext/`).
5. pip-check allowlist — the final gate diffs `pip check` against
   `tests/deepseek-infernal-cu133-pip-check.allowlist`, whose seven known
   LMCache complaints carry upstream's LMCache version string; the wrapper
   rewrites that token to `LMCACHE_BUILD_VERSION` so only a *new* complaint
   can fail the gate.

Not needed on cu133, unlike cu132: no x86 filesystem paths, no cuBLAS
overlay, no cusparselt wheel-tag repair, no torch wheel index (torch is
compiled). No `PATCH_HOST_ARCH`, `PATCH_PCIE_ENV`, `PATCH_PIPCHECK_WHEELTAG`
or `PATCH_VLLM_REQ_MARKERS` equivalents exist for this flavor.

**Interrupted runs.** An OOM kill or SIGKILL skips the restore trap and
leaves the Dockerfiles and allowlist patched, with `.pre-spark.<pid>`
backups beside them. The wrapper refuses to start on that state and prints
the `git checkout -- …` to run; Ctrl-C restores normally.

**What the cu133 image changes at serve time.** NCCL is
`/opt/local-inference/nccl/lib/libnccl.so.2.31.2`; the forward-compat
`libcuda` is `/usr/local/cuda/compat/lib.real/libcuda.so.1`; there is no
cuBLAS overlay. The rank env template carries a cu133 loader block for
`LD_PRELOAD` / `VLLM_NCCL_SO_PATH` / `NCCL_*_PATH`; `probe-image.sh` prints
the exact paths it finds. `vllm._C` is absent on **both** flavors (this
lineage ships the ops in `vllm._C_stable_libtorch`; the upstream build
asserts only that and `cumem_allocator`).

## The manifest (`build.env`)

`build.env` is the **complete, explicit build manifest**: every source
repo, ref, commit sha, and version pin is stated in it, uncommented. One
env file + one image digest = a reproducible build (and the
publishing-checklist announcement writes itself from it). The script
carries the same values as *fallback defaults* purely so a bare or partial
invocation still works and so removing a line from a profile has a defined
meaning — but the profile is the source of truth, and a published image
should always ship its profile.

Precedence: **process environment > `build.env` > in-script defaults.** The
env file is **parsed, never sourced** — plain `KEY=VALUE`, no quotes, no
shell syntax (sourcing strips quotes and executes content; we learned that
the hard way).

## Image naming

A docker reference is `REPOSITORY:TAG` — in `local/vllm:glm53-nvfp4` the
repository is `local/vllm` (an optional registry host, then namespace `local`,
then name `vllm`; with no host it stays on this machine) and the tag is
`glm53-nvfp4`. The profile owns both halves. **The wrapper contributes no
decoration of its own** — no `-jj-b12x-cu132-sm121`, no automatic date.

| Key | Default | Meaning |
| --- | --- | --- |
| `IMAGE_REPO` | `local/vllm` | the repository half |
| `IMAGE_TAG` | `PROFILE_NAME`, verbatim | the tag half |
| `PROFILE_NAME` | `IMAGE_TAG`, sanitized | names the manifest and transcript; optional when `IMAGE_TAG` is set |
| `IMAGE` | `IMAGE_REPO:IMAGE_TAG` | the whole reference; setting it governs everything below |
| `SYSTEM_BASE_IMAGE` | `<image>-system-base` | build-cache handle, not an artifact |
| `BUILD_BASE_IMAGE_TAG` | `<image>-build-base` | build-cache handle, not an artifact |
| `VLLM_BUILD_VERSION` | `0.26.1rc0+<tag>` | PEP 440 wheel version; non-alphanumeric runs collapse to dots |

With no profile and no `IMAGE`, nothing has named the build, so the tag falls
back to the date alone: `local/vllm:20260901`.

**A date appears only where a value asks for it.** `<date>` expands to
`YYYYmmdd` and `<datetime>` to `YYYYmmdd-HHMMSS`, wherever they occur in
`PROFILE_NAME`, `IMAGE`, `IMAGE_REPO`, `IMAGE_TAG`, either base-stage tag, or
`VLLM_BUILD_VERSION`. Both come from one timestamp taken at startup, so a
build that crosses midnight cannot date its image one day and its manifest the
next. Omit the token and there is no date: rebuilding a profile overwrites its
own tag instead of leaving a full image set per build day on the node. Add it
when you deliberately want side-by-side builds.

Because the base-stage tags derive from the image, two profiles get two sets
of base stages and can never silently overwrite each other's — which matters
when their toolchain pins differ. Profiles that share a toolchain can share
the layers, and the disk, by pinning both to one common pair of names.

Resolved names are printed by `--dry-run` and recorded in the build manifest.
An untagged `IMAGE` (which docker would silently resolve to `:latest`), an
invalid tag, or an unexpanded `<token>` all fail at startup rather than after
the build.

## Components and why each is here

- **[local-inference-lab/vllm](https://github.com/local-inference-lab/vllm)**
  (`dev/jovian-judgement`) — the inference engine itself; this fork carries
  the GLM-5.3/Qwen-3.8 model families, the DFlash/MTP speculators, and the
  in-tree B12X backends that upstream vLLM lacks.
- **[local-inference-lab/sparkinfer](https://github.com/local-inference-lab/sparkinfer)**
  (B12X) — the SM120/SM121 kernel package: attention, MoE, quantized linear
  (NVFP4/FP8/MXFP8) GEMMs and the PCIe allreduce; the compute heart of the
  image.
- **[local-inference-lab/blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker)**
  — the upstream multi-stage build system this script wraps; it owns the
  Dockerfile, stage graph, and qualified pin plumbing. All credit for the
  build architecture goes there — this wrapper only retargets it.
- **[voipmonitor/flashinfer](https://github.com/voipmonitor/flashinfer)**
  (qualified integration branch) — attention/sampling kernels vLLM requires;
  this fork carries the PCIe-IPC integration the stack was qualified against.
- **[local-inference-lab/nccl-canonical](https://github.com/local-inference-lab/nccl-canonical)**
  — patched NCCL 2.30.4 that makes cross-node RoCE on Spark pairs work
  (loaded via `LD_PRELOAD`/`VLLM_NCCL_SO_PATH` at serve time).
- **[brandonmmusic-max/exllamav3](https://github.com/brandonmmusic-max/exllamav3)**
  (`a1-retile-sm120`) — EXL3 quantization support; the final image
  import-validates its extension, so it must compile even for models that
  never use EXL3 (hence the aarch64 patch below).
- **[scitix/InstantTensor](https://github.com/scitix/InstantTensor)** (via
  the voipmonitor fork) — the high-throughput safetensors loader
  (`--load-format instanttensor`, ~2.5 GB/s on Spark NVMe); built from
  source because PyPI wheels are x86-only.
- **[deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)**,
  **[mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)**,
  **[LMCache](https://github.com/LMCache/LMCache)** — FP8 GEMMs, guided
  decoding, and KV cache offload, all built in-tree by the upstream
  Dockerfile at qualified pins.
- **PyPI kernel wheels** — `humming-kernels` (MTP MoE), `quack-kernels`,
  `tilelang`, `tokenspeed-mla`, `nvidia-cutlass-dsl`: every pin verified to
  ship aarch64 wheels before being adopted.

## Patches this script applies (and when each can retire)

Each patch is a toggle: `on` (must apply or die), `off` (skip), `auto`
(default: apply if the anchor exists, skip with a notice once upstream
absorbs the fix — the retirement path).

1. **`PATCH_GPU_ARCH`** — rewrites hardcoded `sm_120a/120a/12.0f` ENV lines
   to `sm_121a/121a/12.1f` for GB10. Retires when blackwell-llm-docker
   parametrizes the arch as build ARGs (the obvious upstream PR).
2. **`PATCH_HOST_ARCH`** — rewrites x86_64 filesystem paths (cuBLAS overlay,
   CUDA `targets/x86_64-linux`) to `aarch64-linux-gnu`/`targets/sbsa-linux`.
   Same retirement as above.
3. **`PATCH_PCIE_ENV`** — fixes the baked `VLLM_PCIE_ALLREDUCE_BACKEND=cpp`
   (pre-rename value) to `b12x`; the current vLLM validates the var eagerly
   at worker start and crashes on the stale name. Retires when upstream
   updates the ENV line.
4. **`PATCH_PIPCHECK_WHEELTAG`** — `nvidia-cusparselt-cu13 0.8.1` ships an
   aarch64 wheel whose *recorded* tag says the invalid `sbsa`, failing pip
   25.x's strict `pip check` gates; a metadata repair is injected before
   each gate. Retires when torch pins cusparselt >= 0.9.1 (NVIDIA fixed it).
5. **`PATCH_VLLM_REQ_MARKERS`** — vLLM pins `instanttensor` (x86-only on
   PyPI; source-built later in the image) and `PyNvVideoCodec==2.0.4`
   (aarch64 wheels only exist from 2.1.0) unconditionally; markers are added
   to the *installed* dist-info at the final gate — deliberately not to the
   requirements before the wheel build, which would invalidate the vLLM
   compile cache. Retires when vLLM's requirements gain platform markers.
6. **`PATCH_EXLLAMAV3_AVX`** — exllamav3's CPU dispatch uses x86-only GCC
   builtins and AVX intrinsics; detection is forced false and the two AVX
   all-reduce files become link-compatible stubs (the paths sit behind
   runtime dispatch that cannot fire on Grace). Retires when exllamav3
   gains native aarch64 guards.

## Safeguards

- **Parsed env files** — whitelist of known keys (typos die with the line
  number), rejection of shell metacharacters, CRLF, and `<placeholder>`
  values; process-env overrides are announced.
- **Frozen-block hash** — the rewrite text feeding Docker layers that are
  *ancestors* of the hours-long FlashInfer/vLLM compiles is fenced between
  `FROZEN-BEGIN/END` markers and hash-checked at startup. Editing those
  bytes — even whitespace — silently invalidates that cache on every
  builder; the check makes the cost explicit and requires `FROZEN_ACK` to
  proceed.
- **`--dry-run`** — applies every rewrite and injection to the Dockerfile,
  reports per-patch match counts, and restores it without building. Works
  on any host arch; run it after every upstream pull before spending a
  build.
- **Shape guards** — every patch verifies its expected match count and that
  zero pre-images survive; upstream refactors fail loudly with the
  offending lines, never silently half-apply.
- **Pin-as-ref** — `VLLM_PIN`/`B12X_PIN` are exported as *both* the git ref
  and the verify commit, because the build stages check out the ref before
  verifying: a branch name races against upstream pushes mid-build (fails
  ~40 minutes in); a sha cannot.
- **Pin pre-flight** — before anything expensive, each pinned sha is checked
  for reachability with one call to the GitHub commits API. A 404/422 kills
  the run immediately; a 403 (anonymous rate limit), 5xx, or no network is
  reported as *inconclusive* and never fails a build. This catches typos,
  deleted qualification branches, and force-pushed fork heads in seconds
  instead of ~40 minutes into the clone. `PIN_PREFLIGHT=0` disables it;
  `GH_TOKEN` makes it reliable on a shared egress IP.
- **Pristine restore** — the Dockerfile is backed up and restored on any
  exit, so the checkout never carries local modifications. A build killed
  with SIGKILL/OOM or a host crash skips the restore, so startup refuses
  with recovery instructions when a `*.pre-spark.*` backup is left behind
  (a second wrapper run in the same checkout is refused the same way).
- **Post-build verification** — the finished image must prove: aarch64,
  torch 2.13.0+cu132, b12x and humming kernels importable, every
  `VLLM_REQUIRED_LAUNCHERS` entry present (the "right branch for my model"
  guard), patched NCCL and the CUDA compat shim on disk, and the cuBLAS
  overlay symlink resolving into the arm64 multiarch dir (its Dockerfile
  loop is `if [[ -d ]]`-guarded and would skip silently on a wrong path).

## Build manifest and transcript (publishing provenance)

Every completed build writes `build_manifest-<profile>-<timestamp>.md` next to
this script: local image ID, the resolved `nvidia/cuda` base-image digest,
the effective configuration (process-env overrides annotated), the upstream
checkout commit and dirty state, pristine and as-built Dockerfile hashes,
the per-patch application report, the wrapper's verification results,
resolved versions of any floating pip specs, telemetry, and the final
image's full `pip freeze`. This is the raw material for the lab publishing
checklist — fields that only exist after a registry push (the immutable
`image@sha256:` reference) are pre-filled as `UNKNOWN — needs verification`,
and the wrapper's final output prints the push + `docker inspect
--format '{{index .RepoDigests 0}}'` commands that mint and read the digest.
A local image has only an image ID; the registry digest does not exist
until a push, and Docker refuses tagging by digest — plan the publish
around that. Build-time verification is not serving validation: the log
records pair serving as `Not tested` until you validate the pin on
hardware and say so explicitly.

With `LOGGING=1` — the `--log` flag, a profile key, or the process
environment — the wrapper also keeps `build-log-<profile>-<timestamp>.txt`
beside the manifest: the complete stdout+stderr transcript of the run,
including the upstream driver's output. Capture starts before argument and
env parsing, so a profile syntax error lands in it too, and it is written on
failure as well — a build that dies four hours in leaves the transcript that
explains why. It shares the manifest's timestamp, so the two files pair by
name. Two consequences worth knowing: stdout becomes a pipe, so BuildKit
emits plain non-TTY progress (better in a file, different on your terminal),
and the transcript captures everything the driver prints, so skim one before
attaching it to an announcement.

Neither file belongs in the upstream checkout's history, and left untracked
they make every later `git status` read dirty. Both are covered by
`dgx-spark-builder/.gitignore`.

## `probe-image.sh`: prove the image before spending a boot

```
./probe-image.sh <image> [build-profile.env]
```

Runs a series of `docker run --rm` checks against a built image and exits
non-zero on any FAIL. What it checks, in order:

- **Source identity** — the image's `local-inference.vllm.commit` /
  `local-inference.b12x.commit` labels against `VLLM_PIN` / `B12X_PIN` from
  the profile (both flavors' build systems stamp these labels).
- **RoCEnante composition** — `vllm.distributed.device_communicators.
  b12x_roce_all_reduce` importable, the three `VLLM_ROCE_*` envs registered,
  the `B12X_ROCENANTE` hook in `cuda_communicator.py`, `b12x.comm.roce`
  importable with its lazy API, a C compiler present (the RoCE proxy is
  compiled at first boot), `_roce_proxy.c` shipped.
- **KDA prefill backends** — the optional in-tree FlashKDA extension
  (`vllm._flashkda_C`) actually compiled for sm_121 (the build silently skips
  it on failure, and `--kda-prefill-backend flashkda` would then fall back to
  Triton), and `b12x.sequence.kda_prefill` importable.
- **Generic contract, flavor-aware** — the probe detects cu133 by
  `/opt/local-inference/nccl/lib` and asserts the matching CUDA version
  (13.2 or 13.3), aarch64, torch 2.13.0, `b12x` + humming importable, the
  vLLM `_C_stable_libtorch` + `cumem_allocator` extensions, the GLM-5.3
  launcher, and the flavor's NCCL and compat-shim paths (cu132: the
  `/opt/libnccl-local-inference.so.2.30.4` + `/usr/local/cuda/compat/
  libcuda.so.1` pair and the cuBLAS overlay symlink; cu133: the NCCL 2.31.2
  path and the NGC forward-compat `libcuda`, printed so the rank env can be
  filled in).

It cannot exercise a GPU kernel or a two-node collective; it proves the
image is the one the profile describes and that nothing the launcher's
`LD_PRELOAD` / `--verify` depends on is missing. Run it on the build node,
then again on the second node after `docker save | docker load` (same image
ID ⇒ identical output).

## Docker storage: what fills up, and how to reclaim it

Three separate stores, different reclaim commands, very different costs.
Look before you cut:

```
docker system df                # Images / Containers / Volumes / Build Cache
docker system df -v             # per-image and per-cache-record detail
sudo du -sh /var/lib/docker/*   # what the daemon actually holds on disk
```

- **Build cache (BuildKit records)** — this is what the frozen-block hash
  exists to protect. The system-base/build-base/base layers are *ancestors*
  of the FlashInfer and vLLM compiles; discarding them is the multi-hour
  rebuild.
- **Tagged images** — the serving image plus its `-system-base` and
  `-build-base` stages. A profile whose tag carries a `<date>` or `<datetime>`
  token mints a new set every build, which is usually the real disk consumer;
  the fix there is targeted `docker rmi`, not pruning. A profile with no date
  token overwrites its own tags and never accumulates.
- **Dangling layers** — mostly irrelevant under BuildKit, since intermediate
  stages live in the build cache rather than as untagged images.

Surgical to nuclear:

```
# 1. See which dated tags piled up, oldest first.
docker images 'local/vllm' \
  --format '{{.CreatedAt}}\t{{.Size}}\t{{.Repository}}:{{.Tag}}' | sort

# 2. Drop old intermediate TAGS only. This reclaims image storage without
#    touching BuildKit cache records, so the next build still hits cache.
#    build-base is the fat one (full CUDA devel toolchain).
docker rmi local/vllm:glm53-nvfp4-20260830-build-base \
           local/vllm:glm53-nvfp4-20260830-system-base

# 3. Age-filtered build cache prune; `until` is a duration, not a timestamp.
docker builder prune --filter until=168h        # keep the last week

# 4. Size-capped prune, keeping the most recently used records.
docker builder prune --keep-storage 150GB       # engine builder
docker buildx prune  --keep-storage 150GB       # if a buildx builder is used
#    Flag names move between engine versions: check `--help` first.

# 5. Nuclear. Know the bill before running either.
docker builder prune -a     # ALL build cache -> next build recompiles
                            # FlashInfer + vLLM from scratch (hours)
docker system prune -a      # ALSO removes every image not used by a RUNNING
                            # container: the serving image and both base tags
```

`docker system prune -a` is the one that hurts most: "unused" means "no
running container", which between serving sessions describes the entire
image set. Because node 2 gets its copy by `docker save | ssh node2 docker
load`, running it there means re-shipping tens of GB over the wire, not just
a rebuild.

If a buildx *container* driver is in play the cache lives in that builder's
own volume, and the commands above may target the wrong store:

```
docker buildx ls            # driver column: docker | docker-container
docker buildx du            # per-builder cache usage
docker buildx rm <name>     # destroys that builder's cache entirely
```

Two rules of thumb. **To force a clean rebuild, do not prune — use
`--no-cache`:** pruning discards cache shared with every other profile, while
`./build-spark-cu132.sh <profile> -- --no-cache` rebuilds this image and
leaves the rest intact. And **cap the cache rather than doing this by hand**,
in `/etc/docker/daemon.json`:

```json
{
  "builder": {
    "gc": {
      "enabled": true,
      "defaultKeepStorage": "150GB",
      "policy": [
        { "keepStorage": "50GB",  "filter": ["unused-for=168h"] },
        { "keepStorage": "150GB", "all": true }
      ]
    }
  }
}
```

Then `sudo systemctl restart docker`. **Restarting the daemon kills running
containers** — do it with both ranks down, never mid-serve. Size the cap so
it comfortably holds one full build's ancestor layers, or the GC will evict
exactly what the frozen-block guard is trying to preserve and you pay the
compile cost anyway.

## Profiles

`build-example.env` is an annotated cu132 template: copy it, set the naming
block and the two source pins, and delete the rest of the guidance.
`build-jj-r##.env` is a cu132 release profile pinning a voipmonitor
`integration/*` branch pair (the reviewed release composition);
`build-jj-r##-cu133.env` is the same release for the cu133 wrapper and adds
the foundation pins, `FLASHINFER_VERSION`, `LMCACHE_*`, the three
`*_INTEGRATION_TREE` hashes from rtx6kpro's source lock, and `RUNTIME_FOUNDATION`.
Head profiles (`build-glm53-head-<date>.env`) pin canonical
`dev/jovian-judgement` and `b12x` master directly.

Images are named for the shared runtime (`jovian-judgement-r##[-cu133]`),
not for a model: every JJ image carries the GLM, Qwen and DeepSeek launchers.

For another model on the same branch, copy it, change `PROFILE_NAME`,
`VLLM_REQUIRED_LAUNCHERS`, and (if needed) the pins — nothing in the script
itself is model-specific. Commit the profile next to the published image
digest and the build is reproducible from two files (see the publishing
checklist in the deployment README).
