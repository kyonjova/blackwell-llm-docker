from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spark_release


def test_spark_tags_keep_the_release_identity():
    assert (
        spark_release.spark_image(
            "ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261006-c8c8feeaa8cc4a75"
        )
        == "ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-spark-20261006-c8c8feeaa8cc4a75"
    )
    assert (
        spark_release.spark_alias("ghcr.io/local-inference-lab/vllm:karmic-kraken-beta")
        == "ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-spark"
    )


def test_unexpected_tags_are_refused():
    with pytest.raises(ValueError, match="Unexpected release image tag"):
        spark_release.spark_image("ghcr.io/local-inference-lab/vllm:latest-1-2")


def test_every_assembled_role_has_a_bundle_script():
    import assemble_qwen38_runtime_bundle as assemble

    roles = set(assemble.EXPECTED_SCHEMAS) - {"foundation"} - assemble.OPTIONAL_ROLES
    assert roles == set(spark_release.BUNDLE_SCRIPTS)


@pytest.mark.parametrize("role", sorted(spark_release.BUNDLE_SCRIPTS))
def test_components_without_arm64_locks_are_detected(tmp_path, role):
    assert not spark_release.supports_arm64(role, tmp_path)
    lock = tmp_path / Path(spark_release.BUNDLE_SCRIPTS[role]).parent / "linux-arm64" / "runtime.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("platform=linux/arm64\n")
    assert spark_release.supports_arm64(role, tmp_path)


def test_unchanged_components_reuse_their_verified_bundle(tmp_path):
    bundle = tmp_path / "built" / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}")
    import subprocess

    subprocess.run("sha256sum manifest.json > SHA256SUMS", shell=True, cwd=bundle, check=True)
    assert spark_release.cached_bundle(tmp_path, "lmcache", "a" * 40) is None
    spark_release.keep_bundle(tmp_path, "lmcache", "a" * 40, bundle)
    cached = spark_release.cached_bundle(tmp_path, "lmcache", "a" * 40)
    assert cached is not None and (cached / "manifest.json").read_text() == "{}"
    (cached / "manifest.json").write_text('{"tampered": true}')
    assert spark_release.cached_bundle(tmp_path, "lmcache", "a" * 40) is None


def test_old_bundles_are_pruned(tmp_path):
    import os
    import subprocess

    bundle = tmp_path / "built" / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}")
    subprocess.run("sha256sum manifest.json > SHA256SUMS", shell=True, cwd=bundle, check=True)
    for index in range(spark_release.KEPT_BUNDLES + 2):
        commit = f"{index:040x}"
        spark_release.keep_bundle(tmp_path, "nccl", commit, bundle)
        os.utime(tmp_path / "bundles" / "nccl" / commit, (index, index))
    kept = sorted(path.name for path in (tmp_path / "bundles" / "nccl").iterdir())
    assert len(kept) == spark_release.KEPT_BUNDLES


def _releases(*assets):
    return [{"draft": False, "tag_name": "karmic-kraken-beta-" + "a" * 64,
             "assets": [{"name": "container-release.json"}, *({"name": a} for a in assets)]}]


def _assembly():
    return {"recipe_commit": "r" * 40, "components": {
        role: {"repository": f"local-inference-lab/{role}", "source_commit": "c" * 40}
        for role in spark_release.BUNDLE_SCRIPTS}}


@pytest.mark.parametrize(
    "assets, supported, expected",
    [
        ((), True, "karmic-kraken-beta-" + "a" * 64),
        ((), False, ""),
        (("container-release-linux-arm64.json",), True, ""),
        (("container-release-linux-arm64.failed.json",), True, ""),
    ],
)
def test_the_schedule_builds_only_supported_unbuilt_releases(monkeypatch, assets, supported, expected):
    import json

    def run(argv):
        return json.dumps(_releases(*assets) if argv[:2] == ["gh", "api"] else _assembly())

    monkeypatch.setattr(spark_release, "run", run)
    monkeypatch.setattr(spark_release, "has_arm64_support", lambda *args: supported)
    assert spark_release.select_release("local-inference-lab/blackwell-llm-docker") == expected


def test_a_commit_with_an_already_built_tree_is_not_rebuilt(tmp_path, monkeypatch):
    import subprocess

    bundle = tmp_path / "built" / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"source": {"commit": "' + "b" * 40 + '"}}')
    subprocess.run("sha256sum manifest.json > SHA256SUMS", shell=True, cwd=bundle, check=True)
    spark_release.keep_bundle(tmp_path / "cache", "flashinfer", "t" * 40, bundle)
    monkeypatch.setattr(spark_release, "source_tree", lambda repository, commit: "t" * 40)
    monkeypatch.setattr(spark_release, "checkout", lambda *args: pytest.fail("rebuilt"))
    component = {"repository": "local-inference-lab/flashinfer", "source_commit": "m" * 40}
    reused = spark_release.build_component("flashinfer", component, tmp_path / "work", tmp_path / "cache")
    assert reused == tmp_path / "cache" / "bundles" / "flashinfer" / ("t" * 40) / "bundle"
