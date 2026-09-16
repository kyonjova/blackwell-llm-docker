#!/usr/bin/env python3
"""Verify package, ABI, dependency corrections and optional GPU runtime contracts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path


EXPECTED_VERSIONS = {
    "nvidia-modelopt": "0.46.1",
    "quack-kernels": "0.6.4",
    "torch-c-dlpack-ext": "0.1.5",
    "torch": "2.14.0a0+4fdf77b940.nv26.8.63802676",
    "xgrammar": "0.2.6",
}

NGC_PYTHON_HASHES = {
    "torch/_library/utils.py": (
        "7e694fb5c280d4cb671415d6b14a6975f9ed42c8fb3c9907e93c1a3d57493e64"
    ),
    "torch/_library/custom_ops.py": (
        "e86666f219071ce8fe966d8de7783cc0068040061c9c59a1279c617dafd8f710"
    ),
    "nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/jit_executor.py": (
        "a36eea9adcb75546aa5a373809581772234d13392df74a9f4854291e0e9e4221"
    ),
}
NGC_VENV_CUTLASS = "nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/jit_executor.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_b12x_package() -> None:
    """Verify the imported package against bytes from the selected B12X wheel."""
    manifest = json.loads(
        (Path(sys.prefix) / "share/lil-runtime/manifest.json").read_text()
    )
    contract = manifest["b12x_package"]
    runtime_site = Path(sys.prefix) / "lib/python3.12/site-packages"
    import b12x

    expected_init = (runtime_site / "b12x/__init__.py").resolve()
    if Path(b12x.__file__).resolve() != expected_init:
        raise RuntimeError(f"B12X import was redirected to {b12x.__file__}")
    if "b12x/__init__.py" not in contract["files"]:
        raise RuntimeError("B12X payload contract is incomplete")
    for relative, expected in contract["files"].items():
        target = (runtime_site / relative).resolve()
        if not target.is_relative_to(runtime_site / "b12x"):
            raise RuntimeError(f"B12X payload escapes its package: {relative}")
        if not target.is_file() or sha256(target) != expected:
            raise RuntimeError(f"B12X wheel payload mismatch: {relative}")
    flashinfer = importlib.metadata.distribution("flashinfer-python")
    for path in flashinfer.files or ():
        if str(path).startswith(("b12x/", "flashinfer/b12x/")):
            raise RuntimeError("FlashInfer distribution claims B12X package files")
    for entry in flashinfer.entry_points:
        if entry.value.startswith(("b12x.", "flashinfer.b12x.")):
            raise RuntimeError(
                f"FlashInfer distribution claims B12X plugin {entry.name}"
            )
    print(f"b12x_source={contract['source']['commit']} payload=PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="also execute one BF16 matrix multiplication on the visible GPU",
    )
    parser.add_argument(
        "--foundation",
        choices=("wheel", "ngc"),
        default="wheel",
        help="location of the source-locked PyTorch and CUDA foundation",
    )
    parser.add_argument(
        "--defer-native-imports",
        action="store_true",
        help="verify native extension presence without loading the absent build-time driver",
    )
    args = parser.parse_args()

    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(
                f"{package} version {actual!r} does not match {expected!r}"
            )

    verify_b12x_package()
    import flashinfer  # noqa: F401
    import flashinfer_jit_cache  # noqa: F401
    import lmcache  # noqa: F401
    import modelopt  # noqa: F401
    import quack  # noqa: F401
    import torch
    import torch_c_dlpack_ext  # noqa: F401
    import uvloop  # noqa: F401
    import vllm  # noqa: F401

    native_modules = (
        "lmcache.cuda_ops",
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
        "vllm._flashkda_C",
    )
    if args.defer_native_imports:
        for module in native_modules:
            specification = importlib.util.find_spec(module)
            if specification is None or not specification.origin:
                raise RuntimeError(f"native module is absent: {module}")
    else:
        for module in native_modules:
            __import__(module)
    import xgrammar  # noqa: F401
    from instanttensor import safe_open  # noqa: F401

    if torch.version.cuda != "13.4":
        raise RuntimeError(
            f"PyTorch reports CUDA {torch.version.cuda!r}, expected '13.4'"
        )
    if not torch.compiled_with_cxx11_abi():
        raise RuntimeError("PyTorch and application wheels require the C++11 ABI")
    cuda_home = Path(os.environ.get("CUDA_HOME", "")).resolve()
    if not (cuda_home / "include" / "cuda.h").is_file():
        raise RuntimeError("CUDA_HOME must expose the selected foundation's cuda.h")
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError("CUDA_HOME must expose the selected foundation's nvcc")
    if args.foundation == "ngc":
        expected_cuda = Path("/usr/local/cuda").resolve()
        if cuda_home != expected_cuda:
            raise RuntimeError(
                f"NGC runtime selected CUDA_HOME {cuda_home}, expected {expected_cuda}"
            )
        system_site = Path("/usr/local/lib/python3.12/dist-packages")
        torch_path = Path(torch.__file__).resolve()
        if not torch_path.is_relative_to(system_site):
            raise RuntimeError(
                f"NGC runtime imported PyTorch outside {system_site}: {torch_path}"
            )
        for relative, expected in NGC_PYTHON_HASHES.items():
            target = system_site / relative
            if not target.is_file() or sha256(target) != expected:
                raise RuntimeError(f"NGC dependency contract mismatch: {target}")
        venv_cutlass = (
            Path(sys.prefix) / "lib/python3.12/site-packages" / NGC_VENV_CUTLASS
        )
        expected_cutlass = NGC_PYTHON_HASHES[NGC_VENV_CUTLASS]
        if not venv_cutlass.is_file() or sha256(venv_cutlass) != expected_cutlass:
            raise RuntimeError(f"NGC venv dependency contract mismatch: {venv_cutlass}")
        from verify_torch_operator_contract import verify as verify_torch_operators

        print(
            "torch_operator_contract="
            + json.dumps(verify_torch_operators(), sort_keys=True)
        )
        manifest = json.loads(
            (Path(sys.prefix) / "share/lil-runtime/manifest.json").read_text()
        )
        receipt = json.loads(
            (Path(sys.prefix) / "share/lil-runtime/lmcache-cumem.json").read_text()
        )
        if receipt["commit"] != manifest["components"]["lmcache"]["source"]["commit"]:
            raise RuntimeError(
                "cuMem helper source differs from the installed LMCache wheel"
            )
        library = Path("/opt/lmcache/lib/liblmcache_cumem_shareable.so")
        if (
            receipt["library"] != str(library)
            or not library.is_file()
            or sha256(library) != receipt["sha256"]
        ):
            raise RuntimeError("LMCache cuMem helper hash contract mismatch")
    else:
        runtime_site = Path(sys.prefix) / "lib/python3.12/site-packages"
        expected_cuda = (runtime_site / "nvidia/cu13").resolve()
        if cuda_home != expected_cuda:
            raise RuntimeError(
                f"wheel runtime selected CUDA_HOME {cuda_home}, expected {expected_cuda}"
            )
    nccl_path = os.environ.get("VLLM_NCCL_SO_PATH")
    if not nccl_path or not os.path.isfile(nccl_path):
        raise RuntimeError("VLLM_NCCL_SO_PATH must identify the packaged NCCL library")
    if nccl_path not in os.environ.get("LD_PRELOAD", "").split(":"):
        raise RuntimeError(
            "the packaged NCCL library must be preloaded before importing PyTorch"
        )

    if args.require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("a visible CUDA GPU is required")
        left = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
        right = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
        product = left @ right
        torch.cuda.synchronize()
        if not torch.isfinite(product).all().item():
            raise RuntimeError("GPU matrix multiplication produced non-finite values")
    status = "qwen38_runtime=PASS"
    if args.defer_native_imports:
        status += " native_imports=deferred"
    if args.require_gpu:
        status += " gpu=PASS"
    print(status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
