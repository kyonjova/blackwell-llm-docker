"""The runtime file manifest that lil-bench checks containers against."""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import write_runtime_files as manifest

TOOLS = Path(__file__).resolve().parents[1]


def test_manifest_hashes_code_and_skips_bytecode(tmp_path):
    site = tmp_path / "site-packages"
    (site / "vllm/__pycache__").mkdir(parents=True)
    (site / "vllm/model.py").write_text("x = 1\n")
    (site / "vllm/__pycache__/model.cpython-312.pyc").write_bytes(b"cache")
    (site / "vllm/link.py").symlink_to(site / "vllm/model.py")
    launcher = tmp_path / "lil-serve"
    launcher.write_text("#!/bin/sh\n")
    document = manifest.build((str(site), str(launcher), str(tmp_path / "missing")))
    assert document["roots"] == [str(site), str(launcher)]
    assert set(document["files"]) == {str(site / "vllm/model.py"), str(launcher)}
    assert document["files"][str(launcher)] == manifest.file_hash(str(launcher))


def test_archive_is_deterministic(tmp_path):
    document = {"schema": "lil-runtime-files/1", "files": {"/a": "0" * 64}}
    manifest.write(document, tmp_path / "a.json.gz")
    manifest.write(document, tmp_path / "b.json.gz")
    assert (tmp_path / "a.json.gz").read_bytes() == (
        tmp_path / "b.json.gz"
    ).read_bytes()
    assert (
        json.loads(gzip.decompress((tmp_path / "a.json.gz").read_bytes())) == document
    )


def test_manifest_is_written_last_in_the_image_build():
    recipe = (TOOLS / "Dockerfile.runtime").read_text()
    assert recipe.index("runtime.image_install") < recipe.index(
        "write_runtime_files.py"
    )
    assert recipe.index("install_lil_bench.py") < recipe.index("write_runtime_files.py")


def test_release_publishes_the_manifest_and_its_hash():
    publisher = (TOOLS / "publish_container_channel.py").read_text()
    assert '"runtime_files_sha256": digest(runtime_files)' in publisher
    assert "/opt/venv/share/lil-runtime/runtime-files.json.gz" in publisher
    assert "            runtime_files,\n" in publisher
