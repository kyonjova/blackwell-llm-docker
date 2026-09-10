"""Verify native reuse rejects source drift and produces an installable wheel."""

import base64
import csv
from email.parser import BytesParser
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

HELPER = Path(__file__).resolve().parents[1] / "pack_lmcache_native_reuse.py"
SPEC = importlib.util.spec_from_file_location("lmcache_native_reuse", HELPER)
packer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packer)
TAG = "cp312-cp312-linux_x86_64"


@pytest.fixture
def wheel_members():
    return {
        "lmcache/__init__.py": b'"""Python package fixture."""\n',
        "lmcache-0.1.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
        ),
        "lmcache-0.1.dist-info/METADATA": b"Name: lmcache\nVersion: 0.1\n",
        "lmcache-0.1.dist-info/RECORD": b"obsolete record\n",
    }


def test_platform_metadata_and_complete_record_preserve_payloads(wheel_members):
    native = {"lmcache/cuda_ops.so": b"CUDA fixture", "lmcache/nested/rust.so": b"Rust"}
    packed = packer.platform_wheel(wheel_members, native, TAG)
    for name, data in native.items():
        assert packed[name] == data
    assert packed["lmcache/__init__.py"] == wheel_members["lmcache/__init__.py"]
    metadata = packed["lmcache-0.1.dist-info/WHEEL"].decode()
    assert "Root-Is-Purelib: false" in metadata
    assert metadata.count("Tag:") == 1
    assert f"Tag: {TAG}" in metadata
    headers = BytesParser().parsebytes(packed["lmcache-0.1.dist-info/WHEEL"])
    assert headers.get_all("Tag") == [TAG]
    assert headers["Root-Is-Purelib"] == "false"
    rows = list(
        csv.reader(io.StringIO(packed["lmcache-0.1.dist-info/RECORD"].decode()))
    )
    assert {row[0] for row in rows} == set(packed)
    for name, digest, size in rows:
        if name.endswith("/RECORD"):
            assert (digest, size) == ("", "")
        else:
            expected = base64.urlsafe_b64encode(hashlib.sha256(packed[name]).digest())
            assert digest == "sha256=" + expected.rstrip(b"=").decode()
            assert size == str(len(packed[name]))


@pytest.mark.parametrize(
    "native,tag",
    [
        ({}, TAG),
        ({"outside.so": b"bad"}, TAG),
        ({"lmcache/a.so": b"a"}, "py3-none-any"),
    ],
)
def test_incompatible_native_payload_is_rejected(wheel_members, native, tag):
    with pytest.raises(ValueError):
        packer.platform_wheel(wheel_members, native, tag)


def test_cuda_reuse_preserves_rebuilt_cpu_libraries(wheel_members):
    rebuilt = {
        f"lmcache/{name}.so": b"rebuilt " + name.encode()
        for name in packer.CPU_EXTENSIONS
    }
    reference = {f"lmcache/{name}.so": b"reference" for name in packer.CPU_EXTENSIONS}
    reference["lmcache/cuda_ops.so"] = b"qualified CUDA"
    files = {**wheel_members, **rebuilt}
    selected = packer.select_native(files, reference, "cpu-rebuild-cuda-reuse")
    result = packer.platform_wheel(files, selected, TAG)
    assert result["lmcache/cuda_ops.so"] == b"qualified CUDA"
    for path, data in rebuilt.items():
        assert result[path] == data
    del files["lmcache/lmcache_fs.so"]
    with pytest.raises(ValueError, match="all three rebuilt CPU"):
        packer.select_native(files, reference, "cpu-rebuild-cuda-reuse")


@pytest.mark.parametrize("mode", ["reuse-all", "cpu-rebuild-cuda-reuse"])
@pytest.mark.parametrize(
    "changed",
    [
        None,
        "csrc/storage_backends/fs/connector.cpp",
        "csrc/cuda/kernel.cu",
        "csrc/shared.h",
        "setup.py",
    ],
)
def test_cli_reuses_only_native_inputs_with_identical_git_identity(
    tmp_path, wheel_members, mode, changed
):
    reference = tmp_path / "reference"
    reference.mkdir()
    paths = [name for name in packer.NATIVE_INPUTS if name != "csrc"]
    paths += [
        "csrc/storage_backends/fs/connector.cpp",
        "csrc/cuda/kernel.cu",
        "csrc/shared.h",
    ]
    for name in paths:
        (reference / name).parent.mkdir(parents=True, exist_ok=True)
        (reference / name).write_text(f"native input: {name}\n")
    subprocess.run(["git", "init", "-q", str(reference)], check=True)
    subprocess.run(["git", "-C", str(reference), "add", "."], check=True)
    git_commit = [
        "git",
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "Declare native inputs",
    ]
    subprocess.run(git_commit, cwd=reference, check=True)
    source = tmp_path / "source"
    subprocess.run(["git", "clone", "-q", str(reference), str(source)], check=True)
    (source / "python.py").write_text("# Python-only source change\n")
    if changed:
        (source / changed).write_text("changed native source\n")
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(git_commit, cwd=source, check=True)
    package = tmp_path / "site-packages" / "lmcache"
    package.mkdir(parents=True)
    (package / "cuda_ops.so").write_bytes(b"compiled CUDA fixture")
    for name in packer.CPU_EXTENSIONS:
        (package / f"{name}.so").write_bytes(b"reference CPU fixture")
    dist_info = package.parent / "lmcache-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "WHEEL").write_text(f"Tag: {TAG}\n")
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    suffix = "py3-none-any" if mode == "reuse-all" else TAG
    pure = wheel_dir / f"lmcache-0.1-{suffix}.whl"
    if mode == "cpu-rebuild-cuda-reuse":
        wheel_members.update(
            {
                f"lmcache/{name}.so": b"rebuilt CPU fixture"
                for name in packer.CPU_EXTENSIONS
            }
        )
    with zipfile.ZipFile(pure, "w") as archive:
        for name, data in wheel_members.items():
            archive.writestr(name, data)
    result = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--source",
            str(source),
            "--reference-source",
            str(reference),
            "--reference-package",
            str(package),
            "--wheel-dir",
            str(wheel_dir),
            "--mode",
            mode,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    platform = wheel_dir / f"lmcache-0.1-{TAG}.whl"
    compatible = changed is None or (
        mode == "cpu-rebuild-cuda-reuse"
        and changed == "csrc/storage_backends/fs/connector.cpp"
    )
    if not compatible:
        assert result.returncode != 0
        assert "native build inputs differ" in result.stderr
        assert pure.exists()
        if pure != platform:
            assert not platform.exists()
    else:
        assert result.returncode == 0, result.stderr
        if pure != platform:
            assert not pure.exists()
        with zipfile.ZipFile(platform) as archive:
            assert archive.read("lmcache/cuda_ops.so") == b"compiled CUDA fixture"
            if mode == "cpu-rebuild-cuda-reuse":
                assert archive.read("lmcache/lmcache_fs.so") == b"rebuilt CPU fixture"
