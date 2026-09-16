from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools.jovian_wheel_runtime.verify_release_assets import verify_release


COMMIT = "2" * 40
BETA_TAG = f"jovian-cu134-foundation-beta-{COMMIT}"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_release(directory: Path, promotion: bool = False) -> None:
    wheels = {f"package-{index}.whl": f"wheel {index}".encode() for index in range(6)}
    for name, payload in wheels.items():
        (directory / name).write_bytes(payload)
    manifest = {
        "source": {"publisher": {"commit": COMMIT}},
        "release_tag": BETA_TAG,
        "packages": [
            {"file": name, "sha256": digest(payload)}
            for name, payload in wheels.items()
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    fixed = {
        "foundation-runtime.lock",
        "foundation.lock",
        "install_foundation.sh",
        "repack-provenance.json",
        "requirements-foundation.txt",
        "requirements-github.txt",
        "smoke_cuda.cu",
        "torch-native-support-provenance.json",
        "verify_foundation_gpu.sh",
    }
    for name in fixed:
        (directory / name).write_text(name)
    checksummed = list(wheels) + sorted(fixed | {"manifest.json"})
    (directory / "SHA256SUMS").write_text(
        "".join(
            f"{digest((directory / name).read_bytes())}  {name}\n"
            for name in checksummed
        )
    )
    archive = "jovian-cu134-foundation.tar.zst"
    (directory / archive).write_bytes(b"archive")
    (directory / f"{archive}.sha256").write_text(f"{digest(b'archive')}  {archive}\n")
    if promotion:
        record = {
            "schema": "local-inference-foundation-promotion/v1",
            "status": "qualified",
            "source_release": BETA_TAG,
            "source_commit": COMMIT,
            "source_manifest_sha256": digest(
                (directory / "manifest.json").read_bytes()
            ),
            "invariant": (
                "Wheel files and wheel SHA-256 digests are unchanged from the "
                "source beta release."
            ),
        }
        (directory / "stable-promotion.json").write_text(json.dumps(record))


def test_accepts_complete_beta_release(tmp_path):
    write_release(tmp_path)

    verify_release(tmp_path, COMMIT, BETA_TAG, promotion=False)


def test_rejects_changed_wheel(tmp_path):
    write_release(tmp_path)
    (tmp_path / "package-2.whl").write_text("changed")

    with pytest.raises(ValueError, match="package digest mismatch"):
        verify_release(tmp_path, COMMIT, BETA_TAG, promotion=False)


def test_accepts_complete_stable_promotion(tmp_path):
    write_release(tmp_path, promotion=True)

    verify_release(tmp_path, COMMIT, BETA_TAG, promotion=True)


def test_rejects_self_consistent_substitution_against_independent_bytes(tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    candidate.mkdir()
    write_release(candidate)
    shutil.copytree(candidate, reference)
    wheel = candidate / "package-2.whl"
    wheel.write_bytes(b"different wheel")
    manifest = candidate / "manifest.json"
    data = json.loads(manifest.read_text())
    data["packages"][2]["sha256"] = digest(wheel.read_bytes())
    manifest.write_text(json.dumps(data))
    checksums = candidate / "SHA256SUMS"
    names = [line.split(maxsplit=1)[1] for line in checksums.read_text().splitlines()]
    checksums.write_text("".join(
        f"{digest((candidate / name).read_bytes())}  {name}\n" for name in names
    ))
    verify_release(candidate, COMMIT, BETA_TAG, promotion=False)
    with pytest.raises(ValueError, match="independent reference mismatch"):
        verify_release(candidate, COMMIT, BETA_TAG, False, reference)


def test_stable_matches_independent_beta(tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    write_release(candidate, promotion=True)
    write_release(reference)
    verify_release(candidate, COMMIT, BETA_TAG, True, reference)
