from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_platform

TOOLS = Path(__file__).resolve().parents[1]
ROOT = TOOLS.parents[1]


def test_amd64_is_the_default_and_keeps_its_locks(monkeypatch):
    monkeypatch.delenv("LIL_RUNTIME_PLATFORM", raising=False)
    assert runtime_platform.selected() == "linux/amd64"
    assert runtime_platform.lock_dir(TOOLS, "linux/amd64") == TOOLS
    assert (
        runtime_platform.environment_policy(ROOT, "linux/amd64")
        == ROOT / "runtime/platform-environment.json"
    )


def test_unknown_platforms_are_refused(monkeypatch):
    monkeypatch.setenv("LIL_RUNTIME_PLATFORM", "linux/riscv64")
    with pytest.raises(ValueError, match="Unsupported runtime platform"):
        runtime_platform.selected()


@pytest.mark.parametrize("platform", ["linux/amd64", "linux/arm64"])
def test_each_platform_pins_one_foundation_and_its_policy(platform):
    locks = runtime_platform.lock_dir(TOOLS, platform)
    lock = runtime_platform.read_lock(locks / "foundation.lock")
    policy = json.loads(runtime_platform.environment_policy(ROOT, platform).read_text())
    assert lock["source.image.platform"] == platform
    assert policy["source_image"] == lock["source.image"]
    for name in ("qwen38-runtime.lock", "ngc-runtime-overlay.lock"):
        assert (locks / name).is_file()


def test_arm64_shares_the_amd64_foundation_versions():
    """The arm64 twin of NVIDIA PyTorch 26.08 ships the same packages."""
    amd64 = runtime_platform.read_lock(TOOLS / "foundation.lock")
    arm64 = runtime_platform.read_lock(TOOLS / "linux-arm64" / "foundation.lock")
    differing = {key for key in amd64.keys() | arm64.keys() if amd64.get(key) != arm64.get(key)}
    assert differing == {
        "source.image",
        "source.image.platform",
        "buildx.builder",
        "uv.container-sha256",
        "uv.container-image",
    }


def test_arm64_runtime_lock_pins_the_amd64_versions():
    def pins(path: Path) -> set[str]:
        return {
            line.split()[0]
            for line in path.read_text().splitlines()
            if "==" in line and not line.lstrip().startswith(("#", "--"))
        }

    for name in ("qwen38-runtime.lock", "ngc-runtime-overlay.lock"):
        assert pins(TOOLS / "linux-arm64" / name) == pins(TOOLS / name)
