"""Source channels share one build recipe without sharing mutable image tags."""

import base64
import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import container_channel as channel
import publish_container_channel as publisher

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "tools/jovian_wheel_runtime/community-channel.json"


def test_config_separates_only_vllm_and_flashinfer_branches():
    """B12X ships inside FlashInfer; the beta channel overrides vLLM and the
    FlashInfer branch that carries the beta-only B12X changes."""
    config = json.loads(CONFIG.read_text())
    main = channel.channel_config(config, "karmic-kraken")
    beta = channel.channel_config(config, "karmic-kraken-beta")
    assert main["image_tag"] == "karmic-kraken"
    assert beta["image_tag"] == "karmic-kraken-beta"
    assert "b12x" not in main["components"] and "b12x" not in beta["components"]
    assert main["components"]["vllm"]["branch"] == "dev/karmic-kraken"
    assert (
        main["components"]["flashinfer"]["branch"]
        == "community/karmic-kraken-cu134-sm120"
    )
    for role in ("vllm", "flashinfer"):
        assert beta["components"][role]["branch"] == "integration/karmic-kraken-beta"
        without_branch = copy.deepcopy(beta["components"][role])
        without_branch["branch"] = main["components"][role]["branch"]
        assert without_branch == main["components"][role]
    for role in ("lmcache", "instanttensor", "nccl"):
        assert main["components"][role] == beta["components"][role]
    assert config == json.loads(CONFIG.read_text())


def test_only_karmic_containers_are_published_with_shared_foundation():
    config = json.loads(CONFIG.read_text())
    assert set(config["channels"]) == {"karmic-kraken", "karmic-kraken-beta"}
    kk = channel.channel_config(config, "karmic-kraken")
    beta = channel.channel_config(config, "karmic-kraken-beta")
    assert kk["channel"] == beta["channel"] == "karmic-kraken"
    assert kk["components"]["vllm"]["branch"] == "dev/karmic-kraken"
    for role in ("vllm", "flashinfer"):
        assert beta["components"][role]["branch"] == "integration/karmic-kraken-beta"
    for role in ("lmcache", "instanttensor", "nccl"):
        assert kk["components"][role] == beta["components"][role]
    assert (ROOT / ".github/workflows/jovian-wheel-runtime-release.yml").exists()
    assert not (
        ROOT / ".github/workflows/jovian-qwen38-ngc-runtime-release.yml"
    ).exists()
    assert config == json.loads(CONFIG.read_text())


@pytest.mark.parametrize(
    "name", ["main", "beta", "jovian-judgement", "jovian-judgement-beta"]
)
def test_retired_jovian_channels_cannot_resolve(name):
    with pytest.raises(ValueError, match="unknown release channel"):
        channel.channel_config(json.loads(CONFIG.read_text()), name)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "duplicate_tag",
        "unsafe_tag",
        "unknown_role",
        "unsafe_family",
        "unknown_changelog_component",
        "unknown_changelog_policy",
    ],
)
def test_invalid_channel_configuration_fails_closed(fault):
    config = json.loads(CONFIG.read_text())
    beta = config["channels"]["karmic-kraken-beta"]
    if fault == "duplicate_tag":
        beta["image_tag"] = config["channels"]["karmic-kraken"]["image_tag"]
    elif fault == "unsafe_tag":
        beta["image_tag"] = "../untrusted"
    elif fault == "unknown_role":
        beta["branches"]["untrusted"] = "main"
    elif fault == "unsafe_family":
        beta["channel"] = "../untrusted"
    elif fault == "unknown_changelog_component":
        beta["changelog"]["required_components"] = ["untrusted"]
    elif fault == "unknown_changelog_policy":
        beta["changelog"]["untrusted"] = True
    with pytest.raises(ValueError):
        channel.channel_config(
            config, "missing" if fault == "unknown" else "karmic-kraken-beta"
        )


@pytest.mark.parametrize(
    "name,prefix",
    [
        ("karmic-kraken", "karmic-kraken"),
        ("karmic-kraken-beta", "karmic-kraken-beta"),
    ],
)
def test_resolved_image_uses_selected_channel(monkeypatch, tmp_path, name, prefix):
    monkeypatch.setattr(
        channel,
        "select_component",
        lambda role, config: (
            {"source_commit": "a" * 40, "branch": config["branch"]},
            {},
        ),
    )
    monkeypatch.setattr(channel, "ngc_foundation_manifest", lambda path: {})
    monkeypatch.setattr(channel, "validate_compatibility", lambda manifests: None)
    monkeypatch.setattr(channel, "run", lambda args: b"b" * 40)
    assembly = channel.resolve(CONFIG, tmp_path / "assembly.json", name)
    assert assembly["release_channel"] == name
    assert assembly["alias"] == "ghcr.io/local-inference-lab/vllm:" + prefix
    assert assembly["image"].startswith(assembly["alias"] + "-")
    assert assembly["release_tag"] == prefix + "-" + assembly["assembly_sha256"]


def _unpaused_config(tmp_path):
    config = json.loads(CONFIG.read_text())
    for entry in config["channels"].values():
        entry.pop("paused", None)
    path = tmp_path / "community-channel.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("pending", ["karmic-kraken", "karmic-kraken-beta", None])
def test_one_pending_channel_does_not_block_the_other(monkeypatch, tmp_path, pending):
    def resolve(config, output, name):
        if name == pending:
            raise channel.PendingBuild("wheel upload is incomplete")
        return {"release_channel": name, "image": name}

    monkeypatch.setattr(channel, "resolve", resolve)
    monkeypatch.setattr(channel, "completed_publication", lambda repo, assembly: False)
    matrix = channel.resolve_matrix(
        _unpaused_config(tmp_path),
        tmp_path / "assemblies",
        "local-inference-lab/blackwell-llm-docker",
    )
    expected = {"karmic-kraken", "karmic-kraken-beta"} - {pending}
    assert {row["channel"] for row in matrix["include"]} == expected
    for row in matrix["include"]:
        assert (
            json.loads(base64.b64decode(row["assembly"]))["release_channel"]
            == row["channel"]
        )


def test_a_paused_channel_is_skipped(monkeypatch, tmp_path):
    """A paused channel is validated and logged but not resolved, while the
    other channel keeps publishing."""
    config = json.loads(CONFIG.read_text())
    config["channels"]["karmic-kraken"]["paused"] = "waiting for a component"
    path = tmp_path / "community-channel.json"
    path.write_text(json.dumps(config))
    resolved = []

    def resolve(config, output, name):
        resolved.append(name)
        return {"release_channel": name, "image": name}

    monkeypatch.setattr(channel, "resolve", resolve)
    monkeypatch.setattr(channel, "completed_publication", lambda repo, assembly: False)
    matrix = channel.resolve_matrix(path, tmp_path / "assemblies", "repo")
    assert resolved == ["karmic-kraken-beta"]
    assert [row["channel"] for row in matrix["include"]] == ["karmic-kraken-beta"]


@pytest.mark.parametrize("reason", ["", "  ", None, 1, ["reason"]])
def test_a_paused_channel_must_state_why(reason):
    config = json.loads(CONFIG.read_text())
    config["channels"]["karmic-kraken-beta"]["paused"] = reason
    with pytest.raises(ValueError, match="paused"):
        channel.channel_config(config, "karmic-kraken")


def test_completed_channels_do_not_rebuild(monkeypatch, tmp_path):
    monkeypatch.setattr(channel, "resolve", lambda *args: {"image": "complete"})
    monkeypatch.setattr(channel, "completed_publication", lambda *args: True)
    assert channel.resolve_matrix(CONFIG, tmp_path, "repo") == {"include": []}


@pytest.mark.parametrize("changed", [None, "recipe", "vllm", "ref"])
def test_alias_requires_matching_recipe_and_component_heads(monkeypatch, changed):
    assembly = {
        "recipe_commit": "a" * 40,
        "components": {
            "vllm": {
                "repository": "local-inference-lab/vllm",
                "branch": "integration/beta",
                "observed_branch_commit": "b" * 40,
            }
        },
    }

    def api(endpoint):
        recipe = "blackwell-llm-docker" in endpoint
        return {
            "sha": "c" * 40
            if changed == ("recipe" if recipe else "vllm")
            else ("a" if recipe else "b") * 40
        }

    monkeypatch.setattr(publisher, "api", api)
    assert publisher.alias_is_current(
        assembly,
        "local-inference-lab/blackwell-llm-docker",
        "refs/heads/untrusted" if changed == "ref" else "refs/heads/main",
    ) is (changed is None)


def test_workflow_uses_one_channel_matrix_and_shared_publisher():
    workflow = yaml.load(
        (ROOT / ".github/workflows/community-container-release.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    jobs = workflow["jobs"]
    assert "resolve-matrix" in jobs["resolve"]["steps"][-1]["run"]
    for name in ("qualify", "publish"):
        job = jobs[name]
        assert (
            job["strategy"]["matrix"] == "${{ fromJSON(needs.resolve.outputs.matrix) }}"
        )
        assert job["strategy"]["fail-fast"] == "false"
        step = job["steps"][-1]
        assert step["env"]["ASSEMBLY_BASE64"] == "${{ matrix.assembly }}"
        assert "publish_container_channel.py" in step["run"]
        assert "matrix.channel" in step["run"]
