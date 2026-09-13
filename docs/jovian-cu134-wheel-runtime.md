# Jovian CUDA 13.4 wheel runtime

Status: **research-only**

The Jovian CUDA 13.4 wheel runtime defines one source-locked Python environment
for container and direct-host vLLM serving. It uses Python 3.12, CUDA 13.4.1,
the NVIDIA PyTorch 26.08 build, the C++11 ABI, and NVIDIA compute capability
12.0. Docker and direct-host installations consume the same wheel files and
SHA-256-locked dependency manifests.

The runtime has three independently versioned parts:

- the Python and CUDA foundation published by `blackwell-llm-docker`;
- the NCCL 2.31.2 build published by `nccl-canonical`;
- application wheels for FlashInfer, B12X, vLLM, LMCache, XGrammar, and
  InstantTensor.

## Python and CUDA foundation

The immutable source image is NVIDIA PyTorch 26.08 for Linux amd64 at manifest
digest `sha256:33ef5fc15e8937602d64022209cdb2777b32dadf742f41332023d946041b3c14`.
It contains PyTorch
`2.14.0a0+4fdf77b940.nv26.8.63802676`, CUDA 13.4.1, and the exact NVIDIA Triton
build used to compile the application wheels.

NVIDIA does not publish that PyTorch build through a public Python package
index. The foundation builder verifies every hashed installed file against the
package `RECORD` and produces deterministic wheels for PyTorch, TorchVision,
Triton, Triton Kernels, and FlashAttention. The NGC image changes
`torch/lib/libtorch_global_deps.so` after installation. The builder verifies
that installed mutation by hash, then replaces host-specific Torch library
runpaths with paths relative to the venv. Those paths resolve CUDA, cuDNN,
NVSHMEM, and the LIL NCCL package from `site-packages`. The provenance manifest
records every original and packaged runpath and packaged file hash.
Wheel timestamps use the immutable source image creation epoch
`2026-08-21T06:45:47Z`; unchanged package payloads therefore retain identical
wheel bytes when the publisher implementation changes.

The NVIDIA PyTorch binary links to HPC-X and MKL libraries installed outside
Python's package tree. `local-inference-torch-native-support` places the
hash-locked loader closure in `torch/lib`. Its supported purpose is importing
PyTorch and running NCCL-based inference. MPI and UCC collective execution is
unsupported.

Python dependencies declared by the repacked wheels and CUDA userspace and
developer components come from the packages listed in
`tools/jovian_wheel_runtime/foundation-runtime.lock`. Versions of the Python
packages match the immutable NGC source image. The lock includes CUDA Runtime,
NVCC, NVRTC, nvJitLink, cuBLAS, cuSPARSE, cuSOLVER,
cuFFT, cuRAND, cuFile, CUPTI, NVTX, cuDNN 9.26, cuSPARSELt, and NVSHMEM. The
host provides only an NVIDIA driver. CUDA 13.x minor-version compatibility
requires R580 or newer. CUDA 13.4-specific features require R615 or newer.

## NCCL selection

The `local-inference-nccl-cu134` package contains LIL NCCL 2.31.2. The serving
launcher resolves its shared-library path and sets both variables before
importing PyTorch:

```bash
NCCL_SO=$(local-inference-nccl-path)
export LD_PRELOAD="${NCCL_SO}"
export VLLM_NCCL_SO_PATH="${NCCL_SO}"
exec /path/to/venv/bin/python -m vllm.entrypoints.cli.main serve MODEL
```

Setting either variable after importing PyTorch can leave a different NCCL
library resident in the process and is unsupported.

## Installation

GitHub Releases provides immutable direct-download URLs rather than a PEP 503
package index. Each release publishes hashes and a requirements file. The
foundation installer creates an isolated environment without
`--system-site-packages`:

```bash
./install_foundation.sh /opt/local-inference/venvs/jovian-cu134
```

The installer obtains public CUDA wheels from PyPI, installs foundation wheels
from the release bundle, and verifies package versions and native payloads.
Importing PyTorch requires the separately versioned LIL NCCL package. The
complete runtime verifier imports PyTorch with that package preloaded, checks
the CUDA build version and C++ ABI, and rejects manifests whose Python, CUDA,
PyTorch, or C++ ABI fields differ.

After installing the LIL NCCL wheel, qualify the complete foundation on one
physical GPU with:

```bash
./verify_foundation_gpu.sh /opt/local-inference/venvs/jovian-cu134 0
```

The verifier compiles an SM120 CUDA program with the venv NVCC, executes an
NCCL collective through PyTorch, and rejects any CUDA userspace library loaded
from `/usr/local/cuda` instead of the venv. The standalone compiler smoke uses
the static CUDA runtime because NVIDIA's pip runtime wheel does not install an
unversioned `libcudart.so` linker name.

## Build isolation and caching

The `lil-wheel-builder` self-hosted runner on frank2 uses a dedicated rootless
Docker daemon and persistent BuildKit worker. The BuildKit container has a
256 GiB hard memory limit and inherits CPU affinity for logical CPUs 64–127.
Package downloads, compiler objects, and BuildKit layers persist across builds
without sharing state with the production Docker daemon. The BuildKit garbage
collector reserves 100 GB for reusable build state, permits at most 1 TB, and
starts reclaiming space when the host has less than 500 GB free.

One organization-scoped runner can serve the `flashinfer`, `vllm`, `b12x`,
`LMCache`, `nccl-canonical`, and `blackwell-llm-docker` repositories. The
build-controller GitHub credential lacks the
organization runner-administration permission required to create that runner.
Repository-scoped listeners can use the same resource-limited BuildKit worker
until an organization runner token is available.
