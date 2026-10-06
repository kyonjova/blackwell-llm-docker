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

    roles = set(assemble.EXPECTED_SCHEMAS) - {"foundation"}
    assert roles == set(spark_release.BUNDLE_SCRIPTS)


@pytest.mark.parametrize("role", sorted(spark_release.BUNDLE_SCRIPTS))
def test_components_without_arm64_locks_are_detected(tmp_path, role):
    assert not spark_release.supports_arm64(role, tmp_path)
    lock = tmp_path / Path(spark_release.BUNDLE_SCRIPTS[role]).parent / "linux-arm64" / "runtime.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("platform=linux/arm64\n")
    assert spark_release.supports_arm64(role, tmp_path)
