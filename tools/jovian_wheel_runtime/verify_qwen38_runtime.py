#!/usr/bin/env python3
"""Verify package, ABI, NCCL, and optional GPU contracts of the Qwen runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys


EXPECTED_VERSIONS = {
    "nvidia-modelopt": "0.46.1",
    "torch": "2.14.0a0+4fdf77b940.nv26.8.63802676",
    "xgrammar": "0.2.6",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="also execute one BF16 matrix multiplication on the visible GPU",
    )
    args = parser.parse_args()

    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package} version {actual!r} does not match {expected!r}")

    import b12x  # noqa: F401
    import flashinfer  # noqa: F401
    import flashinfer_jit_cache  # noqa: F401
    import lmcache  # noqa: F401
    import lmcache.cuda_ops  # noqa: F401
    import modelopt  # noqa: F401
    import torch
    import vllm  # noqa: F401
    # CUDA operators use vLLM's stable LibTorch ABI extensions.  The legacy
    # vllm._C module is a ROCm/CPU compatibility target and is intentionally
    # absent from the CUDA wheel.
    import vllm._C_stable_libtorch  # noqa: F401
    import vllm._moe_C_stable_libtorch  # noqa: F401
    import xgrammar  # noqa: F401
    from instanttensor import safe_open  # noqa: F401

    if torch.version.cuda != "13.4":
        raise RuntimeError(f"PyTorch reports CUDA {torch.version.cuda!r}, expected '13.4'")
    if not torch.compiled_with_cxx11_abi():
        raise RuntimeError("PyTorch and application wheels require the C++11 ABI")
    nccl_path = os.environ.get("VLLM_NCCL_SO_PATH")
    if not nccl_path or not os.path.isfile(nccl_path):
        raise RuntimeError("VLLM_NCCL_SO_PATH must identify the packaged NCCL library")
    if nccl_path not in os.environ.get("LD_PRELOAD", "").split(":"):
        raise RuntimeError("the packaged NCCL library must be preloaded before importing PyTorch")

    if args.require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("a visible CUDA GPU is required")
        left = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
        right = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
        product = left @ right
        torch.cuda.synchronize()
        if not torch.isfinite(product).all().item():
            raise RuntimeError("GPU matrix multiplication produced non-finite values")
    print("qwen38_runtime=PASS" + (" gpu=PASS" if args.require_gpu else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
