"""Exercise the installed payload, neutral dispatcher and inherited-file guards."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from runtime import ConfigError
from runtime.entrypoint import command
from runtime.image_install import install
from runtime.packaging import audit_image_metadata

METADATA = {"Id": "sha256:" + "1" * 64, "Config": {"Env": ["PATH=/usr/bin"]}}


def test_raw_command_retains_bootstrap_and_ignores_model_defaults():
    assert command(
        ["python", "-c", "pass"],
        {"PROFILE": "ds41-flash"},
        {"bootstrap": ["/bootstrap"]},
    ) == ["/bootstrap", "python", "-c", "pass"]


def test_profile_environment_and_legacy_path():
    assert command([], {"PROFILE": "ds41-flash"}, {})[-1] == "runtime.launcher"
    assert command([], {}, {})[-1] == "--help"
    assert command(["serve-ds41-jovian.sh", "/model", "--port", "8001"], {}, {})[
        3:
    ] == ["--profile", "ds41-flash", "--model", "/model", "--port", "8001"]


def test_env_only_interface_inspects_without_runtime_or_gpu():
    result = subprocess.run(
        [sys.executable, "-m", "runtime.launcher", "--print-config"],
        env={
            "PATH": os.environ["PATH"],
            "PROFILE": "ds41-flash",
            "TP": "4",
            "OMP_NUM_THREADS": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(result.stdout)
    assert config["profile"] == "ds41-flash"
    assert config["environment"]["OMP_NUM_THREADS"]["value"] == "1"


def test_atomic_profile_install_and_existing_destination(tmp_path):
    destination, bindir, site = (
        tmp_path / name for name in ("lib/runtime", "bin", "site")
    )
    kwargs = {"destination": destination, "bin_directory": bindir, "python_site": site}
    receipt = install(METADATA, "2" * 64, "/bootstrap", **kwargs)
    for relative, expected in receipt["files"].items():
        assert (
            hashlib.sha256((destination / relative).read_bytes()).hexdigest()
            == expected
        )
    assert receipt["bootstrap"] == ["/bootstrap"]
    assert (site / "lil-model-runtime.pth").read_text().strip() == str(
        destination.parent
    )
    assert (bindir / "lil-entrypoint").stat().st_mode & 0o111
    stale = destination / "stale.py"
    stale.write_text("preserve operator-owned files")
    with pytest.raises(ConfigError, match="already exists"):
        install(METADATA, "2" * 64, "/bootstrap", **kwargs)
    assert stale.read_text() == "preserve operator-owned files"


@pytest.mark.parametrize(
    "image_id", ["sha256:invalid", "sha256:" + "0" * 63, "sha256:" + "g" * 64]
)
def test_rejects_incomplete_foundation_identity(image_id):
    with pytest.raises(ConfigError, match="immutable image ID"):
        audit_image_metadata({**METADATA, "Id": image_id})


def test_installer_rejects_excess_arguments_before_writing():
    result = subprocess.run(
        ["bash", "runtime/install.sh", "a", "b", "c", "d"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "Usage" in result.stderr
