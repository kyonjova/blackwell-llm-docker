"""Metadata isolation preserves layer identity and cached builder inputs."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_runtime_foundation import (  # noqa: E402
    MANIFEST,
    image_manifest,
    neutralize,
    prepare,
    read_blob,
    validate_layout,
    write_blob,
)


def fixture_layout(root):
    layout = root / "fixture"
    (layout / "blobs/sha256").mkdir(parents=True)
    layer = b"opaque layer bytes must never be rewritten"
    layer_digest = hashlib.sha256(layer).hexdigest()
    (layout / "blobs/sha256" / layer_digest).write_bytes(layer)
    config = {
        "architecture": "amd64",
        "os": "linux",
        "config": {"Env": ["PATH=/usr/bin", "NCCL_NET_PLUGIN=spcx"]},
        "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "a" * 64]},
        "history": [{"created_by": "fixture"}],
    }
    config_ref = write_blob(
        layout, config, {"mediaType": "application/vnd.oci.image.config.v1+json"}
    )
    manifest = {
        "schemaVersion": 2,
        "mediaType": MANIFEST,
        "config": config_ref,
        "layers": [
            {
                "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                "digest": "sha256:" + layer_digest,
                "size": len(layer),
            }
        ],
    }
    ref = write_blob(layout, manifest, {"mediaType": MANIFEST})
    (layout / "index.json").write_text(
        json.dumps({"schemaVersion": 2, "manifests": [ref]})
    )
    (layout / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    metadata = {
        "Id": config_ref["digest"],
        "Config": config["config"],
        "RootFS": {"Type": "layers", "Layers": config["rootfs"]["diff_ids"]},
    }
    return layout, metadata


def test_rewrite_changes_only_declared_environment_and_descriptors(tmp_path):
    layout, metadata = fixture_layout(tmp_path)
    _, before = image_manifest(layout)
    config_before = read_blob(layout, before["config"])
    receipt = neutralize(layout, metadata, {"NCCL_NET_PLUGIN": "spcx"})
    validate_layout(layout, receipt, metadata)
    _, after = image_manifest(layout)
    config_after = read_blob(layout, after["config"])
    assert after["layers"] == before["layers"]
    expected = copy.deepcopy(config_before)
    expected["config"]["Env"] = ["PATH=/usr/bin"]
    assert config_after == expected
    assert metadata["Config"]["Env"][-1] == "NCCL_NET_PLUGIN=spcx"
    assert receipt["inspection"]["Id"] != metadata["Id"]


def test_containerd_manifest_image_identity_is_supported(tmp_path):
    layout, metadata = fixture_layout(tmp_path)
    descriptor, _ = image_manifest(layout)
    metadata["Id"] = descriptor["digest"]
    receipt = neutralize(layout, metadata, {"NCCL_NET_PLUGIN": "spcx"})
    validate_layout(layout, receipt, metadata)


@pytest.mark.parametrize("change", ["identity", "layers", "default"])
def test_source_mismatch_fails_before_rewrite(tmp_path, change):
    layout, metadata = fixture_layout(tmp_path)
    defaults = {"NCCL_NET_PLUGIN": "spcx"}
    if change == "identity":
        metadata["Id"] = "sha256:" + "f" * 64
    elif change == "layers":
        metadata["RootFS"]["Layers"] = []
    else:
        defaults["NCCL_NET_PLUGIN"] = "none"
    with pytest.raises(ValueError):
        neutralize(layout, metadata, defaults)


def test_cached_foundation_does_not_export_or_recompile_again(tmp_path, monkeypatch):
    layout, metadata = fixture_layout(tmp_path)
    policy = tmp_path / "policy.json"
    source = "example.invalid/foundation@sha256:" + "b" * 64
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_image": source,
                "environment": {"NCCL_NET_PLUGIN": "spcx"},
            }
        )
    )
    calls = []

    def export(argv, *, input, check):
        calls.append(argv)
        destination = argv[argv.index("--output") + 1].split("dest=")[1].split(",")[0]
        shutil.copytree(layout, destination)
        assert input == f"FROM {source}\n".encode()

    monkeypatch.setattr(subprocess, "run", export)
    first = prepare(source, metadata, policy, tmp_path / "cache", "bounded-builder")
    assert first == prepare(
        source, metadata, policy, tmp_path / "cache", "bounded-builder"
    )
    assert len(calls) == 1
    assert "bounded-builder" in calls[0]
    assert first[0].startswith("oci-layout:///")
    assert first[1]["Config"]["Env"] == ["PATH=/usr/bin"]


@pytest.mark.skipif(
    not os.environ.get("LIL_FOUNDATION_BUILDER"),
    reason="Explicit CPU-only Docker qualification",
)
def test_buildkit_consumes_neutral_layout_without_adding_foundation_layers(tmp_path):
    builder = os.environ["LIL_FOUNDATION_BUILDER"]
    base_context = tmp_path / "base"
    base_context.mkdir()
    (base_context / "marker").write_text("foundation filesystem remains intact\n")
    layout = tmp_path / "oci"
    subprocess.run(
        [
            "docker",
            "buildx",
            "build",
            "--builder",
            builder,
            "--provenance=false",
            "-f",
            "-",
            "--output",
            f"type=oci,dest={layout},tar=false",
            str(base_context),
        ],
        input=b"FROM scratch\nCOPY marker /foundation-marker\nENV NCCL_NET_PLUGIN=spcx\n",
        check=True,
    )
    _, manifest = image_manifest(layout)
    config = read_blob(layout, manifest["config"])
    metadata = {
        "Id": manifest["config"]["digest"],
        "Config": config["config"],
        "RootFS": {"Type": "layers", "Layers": config["rootfs"]["diff_ids"]},
    }
    receipt = neutralize(layout, metadata, {"NCCL_NET_PLUGIN": "spcx"})
    validate_layout(layout, receipt, metadata)
    image = "local/lil-platform-env-test:" + uuid.uuid4().hex
    try:
        subprocess.run(
            [
                "docker",
                "buildx",
                "build",
                "--builder",
                builder,
                "--provenance=false",
                "-f",
                "-",
                "--build-context",
                f"neutral=oci-layout://{layout}@{receipt['manifest_digest']}",
                "--tag",
                image,
                "--load",
                str(base_context),
            ],
            input=b"FROM neutral\nCOPY marker /application-marker\n",
            check=True,
        )
        final = json.loads(
            subprocess.check_output(["docker", "image", "inspect", image])
        )[0]
        layers = final["RootFS"]["Layers"]
        assert layers[:-1] == metadata["RootFS"]["Layers"]
        assert not any(
            item.startswith("NCCL_NET_PLUGIN=")
            for item in final["Config"].get("Env") or []
        )
    finally:
        subprocess.run(["docker", "image", "rm", image], check=False)
