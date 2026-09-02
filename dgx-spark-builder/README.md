# dgx-spark-builder — generic vLLM/B12X image builder for DGX Spark

Builds aarch64/sm_121 (Grace + GB10) serving images of the
local-inference-lab vLLM stack from source, by retargeting the upstream
x86/sm_120 build system and patching the handful of things that break on
arm64. Model choice is a *profile* (`build.env`), not a fork of the script.

**Currently set to build: local-inference-lab/vllm Branch: dev/jovian-judgement**

## Layout and usage

This directory lives inside a checkout of
[blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker):

```
blackwell-llm-docker/
├── Dockerfile.vllm-b12x-cu132      # upstream build system (untouched)
├── build-vllm-b12x-cu132.sh        # upstream build driver (untouched)
└── dgx-spark-builder/
    ├── build-spark-cu132.sh        # this wrapper
    ├── build.env                   # full build manifest (the profile)
    └── BUILD-README.md
```

```
cd blackwell-llm-docker/dgx-spark-builder
./build-spark-cu132.sh --dry-run            # validate rewrites, no build, any host
./build-spark-cu132.sh                      # build with ./build.env (hours first run)
./build-spark-cu132.sh other-model.env      # build a different profile
./build-spark-cu132.sh build.env -- --build-arg CUBLAS_CUDA13_VERSION=x
./build-spark-cu132.sh --log build-<image-name>.env   # keep a transcript
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

`build-example.env` is an annotated template: copy it, set the naming block
and the two source pins, and delete the rest of the guidance.
`build-glm53-r##.env` in this directory is the qualified **GLM-5.3-Flash** manifest.
`build-glm53-r##.env` pins `dev/jovian-judgement` and `b12x`
master HEAD from the canonical repositories.

For another model on the same branch, copy it, change `PROFILE_NAME`,
`VLLM_REQUIRED_LAUNCHERS`, and (if needed) the pins — nothing in the script
itself is model-specific. Commit the profile next to the published image
digest and the build is reproducible from two files (see the publishing
checklist in the deployment README).
