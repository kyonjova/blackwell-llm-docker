"""Keep Python foundation publication independent of component notifications."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/jovian-wheel-runtime-release.yml"


def foundation_events():
    # BaseLoader preserves the YAML key `on` as a string instead of a YAML 1.1 bool.
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)["on"]


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/jovian-wheel-runtime-release.yml",
        "tools/jovian_wheel_runtime/Dockerfile",
        "tools/jovian_wheel_runtime/build_foundation_bundle.sh",
        "tools/jovian_wheel_runtime/build_torch_native_support.py",
        "tools/jovian_wheel_runtime/foundation-runtime.in",
        "tools/jovian_wheel_runtime/foundation-runtime.lock",
        "tools/jovian_wheel_runtime/foundation.lock",
        "tools/jovian_wheel_runtime/install_foundation.sh",
        "tools/jovian_wheel_runtime/repack_installed_distribution.py",
        "tools/jovian_wheel_runtime/run_serialized_build.sh",
        "tools/jovian_wheel_runtime/torch_native_support.json",
        "tools/jovian_wheel_runtime/verify_foundation_gpu.sh",
        "tools/jovian_wheel_runtime/verify_release_assets.py",
        "tools/jovian_wheel_runtime/tests/smoke_cuda.cu",
        "tools/jovian_wheel_runtime/tests/test_repack_installed_distribution.py",
        "tools/jovian_wheel_runtime/tests/test_verify_release_assets.py",
    ],
)
def test_foundation_inputs_trigger_publication(path):
    assert (ROOT / path).is_file()
    assert path in foundation_events()["push"]["paths"]


def test_notifications_and_container_tools_do_not_publish_foundation():
    events = foundation_events()
    paths = events["push"]["paths"]
    # Exact paths make this test independent of a partial GitHub glob emulator.
    assert all(not any(char in path for char in "*?[!") for path in paths)
    assert ".github/workflows/component-container-dispatch.yml" not in paths
    assert "tools/jovian_wheel_runtime/tests/test_component_dispatch.py" not in paths
    assert "tools/jovian_wheel_runtime/community-channel.json" not in paths
    assert "tools/jovian_wheel_runtime/publish_container_channel.py" not in paths
    assert "tools/jovian_wheel_runtime/Dockerfile.qwen38-ngc-runtime" not in paths
    assert "tools/jovian_wheel_runtime/Dockerfile.runtime" not in paths
    assert events["push"]["branches"] == ["main"]
    assert events["push"]["tags"] == ["jovian-cu134-foundation-stable-*"]
    assert "workflow_dispatch" in events
