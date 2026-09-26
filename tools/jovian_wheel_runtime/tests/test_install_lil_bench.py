"""lil-bench is installed from a locked archive and built for the image's GPUs."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import install_lil_bench as installer


def archive(files: dict) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(f"llm-inference-bench-abc/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_lock_pins_a_commit_and_archive():
    lock = json.loads(installer.LOCK.read_text())
    assert lock["repository"] == "local-inference-lab/llm-inference-bench"
    assert len(lock["commit"]) == 40 and int(lock["commit"], 16) >= 0
    assert len(lock["archive_sha256"]) == 64
    assert installer.archive_url(lock).endswith("/tar.gz/" + lock["commit"])


def test_extract_verifies_checksum_and_strips_top_directory(tmp_path):
    data = archive({"lil_bench/__init__.py": "VERSION = '1'\n", "llm_decode_bench.py": "x\n"})
    lock = {"archive_sha256": hashlib.sha256(data).hexdigest()}
    prefix = tmp_path / "opt/lil/bench"
    prefix.mkdir(parents=True)
    (prefix / "stale.py").write_text("old")
    installer.extract(data, lock, prefix)
    assert (prefix / "lil_bench/__init__.py").read_text() == "VERSION = '1'\n"
    assert not (prefix / "stale.py").exists()
    with pytest.raises(ValueError, match="checksum"):
        installer.extract(data, {"archive_sha256": "0" * 64}, prefix)


def test_launcher_runs_the_package_with_the_venv_python(tmp_path):
    launcher = tmp_path / "lil-bench"
    installer.install_launcher(Path("/opt/lil/bench"), launcher)
    text = launcher.read_text()
    assert 'PYTHONPATH="/opt/lil/bench' in text and "exec /opt/venv/bin/python -m lil_bench" in text
    assert launcher.stat().st_mode & 0o111


def test_runtime_image_installs_lil_bench_after_auxiliary():
    recipe = (Path(__file__).resolve().parents[1] / "Dockerfile.runtime").read_text()
    assert recipe.index("install_runtime_auxiliary.py") < recipe.index("install_lil_bench.py") < recipe.index(
        "runtime.image_install")


def test_p2pmark_targets_blackwell():
    assert "arch=compute_120,code=sm_120" in installer.GENCODE
    assert "arch=compute_121,code=sm_121" in installer.GENCODE
