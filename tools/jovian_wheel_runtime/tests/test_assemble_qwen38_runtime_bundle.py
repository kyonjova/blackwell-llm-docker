from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


TOOL_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "assemble_qwen38_runtime_bundle",
    TOOL_DIR / "assemble_qwen38_runtime_bundle.py",
)
assert SPEC is not None and SPEC.loader is not None
ASSEMBLER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSEMBLER)


def write_foundation_lock(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "source.image=nvcr.io/nvidia/pytorch@sha256:abc",
                "python.version=3.12",
                "cuda.version=13.4.1",
                "pytorch.version=2.14.0a0+nv26.8",
                "torchvision.version=0.29.0a0+nv26.8",
                "triton.version=3.8.0+nv26.8",
                "triton-kernels.version=1.0.0+nv26.8",
                "flash-attn.version=2.7.4.post1",
            )
        )
        + "\n"
    )


def test_ngc_foundation_manifest_describes_installed_packages(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "foundation.lock"
    write_foundation_lock(lock)

    manifest = ASSEMBLER.ngc_foundation_manifest(lock)

    assert manifest["source"] == {
        "image": "nvcr.io/nvidia/pytorch@sha256:abc"
    }
    assert manifest["runtime"] == {"python": "3.12", "cuda": "13.4.1"}
    packages = {
        package["name"]: package["version"] for package in manifest["packages"]
    }
    assert packages["torch"] == "2.14.0a0+nv26.8"
    assert packages["triton-kernels"] == "1.0.0+nv26.8"


def test_ngc_foundation_manifest_rejects_incomplete_lock(tmp_path: Path) -> None:
    lock = tmp_path / "foundation.lock"
    write_foundation_lock(lock)
    lock.write_text(lock.read_text().replace("flash-attn.version=2.7.4.post1\n", ""))

    with pytest.raises(ValueError, match="flash-attn.version"):
        ASSEMBLER.ngc_foundation_manifest(lock)


def test_read_lock_rejects_duplicate_keys(tmp_path: Path) -> None:
    lock = tmp_path / "foundation.lock"
    lock.write_text("cuda.version=13.4.1\ncuda.version=13.4.1\n")

    with pytest.raises(ValueError, match="duplicate foundation lock key"):
        ASSEMBLER.read_lock(lock)


def test_application_requirements_omit_ngc_foundation() -> None:
    packages = [
        {"component": "foundation", "name": "torch", "version": "2.14"},
        {
            "component": "vllm",
            "name": "vllm",
            "version": "0.1.dev1",
            "sha256": "1" * 64,
        },
    ]

    assert ASSEMBLER.application_requirements(packages) == (
        f"vllm==0.1.dev1 --hash=sha256:{'1' * 64}\n"
    )
