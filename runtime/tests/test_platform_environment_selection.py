"""linux/arm64 images apply their own NGC foundation policy."""

import json

from runtime import launcher


def test_amd64_hosts_read_the_default_policy(monkeypatch):
    monkeypatch.setattr(launcher.host_platform, "machine", lambda: "x86_64")
    assert launcher.platform_environment_path().name == "platform-environment.json"


def test_arm64_hosts_read_the_arm64_policy(monkeypatch):
    monkeypatch.setattr(launcher.host_platform, "machine", lambda: "aarch64")
    path = launcher.platform_environment_path()
    assert path.name == "platform-environment.linux-arm64.json"
    policy = json.loads(path.read_text())
    # The arm64 NGC image lists 11.0 and omits 7.5.
    assert policy["environment"]["TORCH_CUDA_ARCH_LIST"] == "8.0 8.6 9.0 10.0 11.0 12.0+PTX"
    assert launcher.platform_environment() == policy["environment"]
