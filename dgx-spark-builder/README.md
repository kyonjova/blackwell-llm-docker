# dgx-spark-builder — generic vLLM/B12X image builder for DGX Spark

Builds aarch64/sm_121 (Grace + GB10) serving images of the
local-inference-lab vLLM stack from source, by retargeting the upstream
x86/sm_120 build system and patching the handful of things that break on
arm64. Model choice is a *profile* (`build.env`), not a fork of the script.

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
```

The script auto-locates the repo root (its parent directory) and defaults to
the `build.env` beside it, so no arguments are needed for the standard case.

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

## Profiles

`build.env` in this directory is the qualified **GLM-5.3-Flash** manifest.
For another model on the same branch, copy it, change `PROFILE_NAME`,
`VLLM_REQUIRED_LAUNCHERS`, and (if needed) the pins — nothing in the script
itself is model-specific. Commit the profile next to the published image
digest and the build is reproducible from two files (see the publishing
checklist in the deployment README).
