"""User-facing defaults retain explicit text-only and non-speculative choices."""

import pytest

from runtime.launcher import resolve


def test_glm_defaults_to_three_mtp_proposals():
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", env={})
    assert plan.values["mode"] == "mtp"
    assert plan.values["draft-tokens"] == 3
    assert plan.values["speculative-config"]["num_speculative_tokens"] == 3
    assert plan.values["speculative-config"]["moe_backend"] == "marlin"
    assert plan.values["max-num-batched-tokens"] == 4096


@pytest.mark.parametrize("argv", [["--mode", "off"], ["--draft-tokens", "0"]])
def test_glm_non_speculative_override_remains_available(argv):
    plan = resolve("glm53-flash", env={}, argv=argv)
    assert plan.values["mode"] == "off"
    assert plan.values["draft-tokens"] == 0
    assert "speculative-config" not in plan.values


@pytest.mark.parametrize("preset", [None, "qwen38-tp2"])
def test_qwen_enables_vision_without_changing_table_or_mtp_policy(preset):
    plan = resolve("qwen38-flash-next", env={}, preset=preset)
    assert plan.values["language-model-only"] is False
    assert "--no-language-model-only" in plan.argv
    assert plan.values["draft-tokens"] == 3
    assert plan.environment["VLLM_PLE_CPU_OFFLOAD"] == "1"
    assert plan.values["max-num-batched-tokens"] == 6019
    assert "limit-mm-per-prompt" not in plan.values


def test_qwen_text_only_override_remains_available():
    plan = resolve("qwen38-flash-next", env={}, argv=["--language-model-only"])
    assert plan.values["language-model-only"] is True
    assert "--language-model-only" in plan.argv


def _interleave(plan):
    argv = plan.argv
    return {
        key: argv[argv.index(key) + 1]
        for key in ("--cp-kv-cache-interleave-size", "--dcp-kv-cache-interleave-size")
        if key in argv
    }


@pytest.mark.parametrize("dcp", ["2", "4"])
def test_qwen_dcp_derives_the_qsa_kv_interleave(dcp):
    plan = resolve("qwen38-flash-next", env={"TP": "4", "DCP": dcp})
    assert _interleave(plan) == {
        "--cp-kv-cache-interleave-size": "4",
        "--dcp-kv-cache-interleave-size": "4",
    }


def test_qwen_dcp1_and_explicit_interleave_are_unchanged():
    assert _interleave(resolve("qwen38-flash-next", env={"TP": "4"})) == {}
    plan = resolve(
        "qwen38-flash-next",
        env={
            "TP": "4",
            "DCP": "4",
            "CP_KV_CACHE_INTERLEAVE_SIZE": "8",
            "DCP_KV_CACHE_INTERLEAVE_SIZE": "8",
        },
    )
    assert _interleave(plan) == {
        "--cp-kv-cache-interleave-size": "8",
        "--dcp-kv-cache-interleave-size": "8",
    }
