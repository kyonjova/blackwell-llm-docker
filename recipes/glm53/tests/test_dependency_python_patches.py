"""Fail closed on dependency drift and publish only fully validated sources."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "dependency_patches", RECIPE / "install_dependency_python_patches.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def fixture(tmp_path, *, bad_output=False):
    before, after = b"value = 1\n", b"value = 2\n"
    target = tmp_path / "package.py"
    target.write_bytes(before)
    patch = tmp_path / "correction.patch"
    patch.write_text(
        "--- a/package.py\n+++ b/package.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "local-inference-python-patches/v1",
                "patches": [
                    {
                        "path": "package.py",
                        "patch": patch.name,
                        "before_sha256": hashlib.sha256(before).hexdigest(),
                        "after_sha256": "0" * 64
                        if bad_output
                        else hashlib.sha256(after).hexdigest(),
                    }
                ],
            }
        )
    )
    return manifest, target, before, after


def test_installs_exact_pinned_source_and_removes_stale_bytecode(tmp_path):
    manifest, target, _, after = fixture(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    stale = cache / "package.cpython-312.pyc"
    stale.write_bytes(b"stale")
    installer.install(manifest, tmp_path)
    assert target.read_bytes() == after
    assert not stale.exists()


def test_unknown_input_is_not_patched(tmp_path):
    manifest, target, _, _ = fixture(tmp_path)
    target.write_bytes(b"different version\n")
    with pytest.raises(ValueError, match="Dependency source mismatch"):
        installer.install(manifest, tmp_path)
    assert target.read_bytes() == b"different version\n"


def test_invalid_patch_result_does_not_modify_installed_package(tmp_path):
    manifest, target, before, _ = fixture(tmp_path, bad_output=True)
    with pytest.raises(ValueError, match="checksum mismatch"):
        installer.install(manifest, tmp_path)
    assert target.read_bytes() == before


def test_all_inputs_are_checked_before_any_write(tmp_path):
    manifest, target, before, _ = fixture(tmp_path)
    data = json.loads(manifest.read_text())
    data["patches"].append(dict(data["patches"][0], before_sha256="0" * 64))
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Dependency source mismatch"):
        installer.install(manifest, tmp_path)
    assert target.read_bytes() == before
