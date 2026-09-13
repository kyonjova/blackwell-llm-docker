from __future__ import annotations

import importlib.util
import hashlib
import zipfile
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


def make_wheel(path: Path, payload: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in payload.items():
            archive.writestr(name, content)
    return path


def test_standalone_b12x_payload_is_bound_to_exact_wheel_bytes(tmp_path: Path) -> None:
    wheel = make_wheel(tmp_path / "b12x.whl", {"b12x/__init__.py": b"native package"})
    hashes = ASSEMBLER.register_wheel_payload(wheel, "b12x", {}, {})
    assert hashes == {"b12x/__init__.py": hashlib.sha256(b"native package").hexdigest()}


@pytest.mark.parametrize("path", ["b12x/__init__.py", "flashinfer/b12x/__init__.py"])
def test_flashinfer_cannot_supply_a_b12x_snapshot(tmp_path: Path, path: str) -> None:
    wheel = make_wheel(tmp_path / "flashinfer.whl", {path: b"redirected package"})
    with pytest.raises(ValueError, match="B12X"):
        ASSEMBLER.register_wheel_payload(wheel, "flashinfer", {}, {})


def test_data_scheme_cannot_overwrite_another_wheel(tmp_path: Path) -> None:
    native = make_wheel(tmp_path / "native.whl", {"shared/module.py": b"a"})
    alias = make_wheel(
        tmp_path / "alias.whl", {"alias.data/purelib/shared/module.py": b"b"}
    )
    owners = {}
    ASSEMBLER.register_wheel_payload(native, "native", owners, {})
    with pytest.raises(ValueError, match="ownership collision"):
        ASSEMBLER.register_wheel_payload(alias, "alias", owners, {})


def test_flashinfer_cannot_register_a_b12x_plugin(tmp_path: Path) -> None:
    wheel = make_wheel(tmp_path / "flashinfer.whl", {
        "flashinfer.dist-info/entry_points.txt":
            b"[vllm.general_plugins]\nb12x_loader = flashinfer.b12x.loader:register\n"
    })
    with pytest.raises(ValueError, match="B12X entry point"):
        ASSEMBLER.register_wheel_payload(wheel, "flashinfer", {}, {})
