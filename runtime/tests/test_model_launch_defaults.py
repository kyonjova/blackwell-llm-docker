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


@pytest.mark.parametrize(
    "profile", ["qwen38-flash-next", "glm53-flash", "ds4-flash", "ds41-flash"]
)
def test_mtp_drafts_sample_from_the_draft_distribution(profile):
    # Sampled requests (the models' default T=1) accept more draft tokens when
    # the drafter samples its distribution than when it proposes the argmax.
    config = resolve(profile, env={}).values["speculative-config"]
    assert config["draft_sample_method"] == "probabilistic"
    assert config["rejection_sample_method"] == "standard"


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


def _dcp_backend(plan):
    argv = plan.argv
    if "--dcp-comm-backend" not in argv:
        return None
    return argv[argv.index("--dcp-comm-backend") + 1]


@pytest.mark.parametrize(
    "dcp,backend", [("1", None), ("2", None), ("3", None), ("6", "ag_rs")]
)
def test_glm53_tp6_dcp6_uses_ag_rs(dcp, backend):
    # The B12X PCIe DCP all-to-all covers 2, 4 and 8 ranks; the generic
    # all-to-all prefills DCP6 about seven times slower than ag_rs.
    plan = resolve("glm53", env={"DCP": dcp}, preset="glm53-csf-tp6")
    assert _dcp_backend(plan) == backend


def test_glm53_explicit_dcp_backend_is_kept():
    plan = resolve(
        "glm53", env={"DCP": "6", "DCP_COMM_BACKEND": "a2a"}, preset="glm53-csf-tp6"
    )
    assert _dcp_backend(plan) == "a2a"


def _gather_env(plan):
    return {
        name: plan.environment.get(name)
        for name in ("VLLM_B12X_MLA_CKV_GATHER", "VLLM_DCP_INDEXER_KEY_GATHER")
    }


@pytest.mark.parametrize(
    "dcp,enabled", [("1", "0"), ("2", "1"), ("3", "1"), ("6", "1")]
)
def test_glm53_dcp_prefill_gathers_the_full_ckv_cache(dcp, enabled):
    plan = resolve("glm53", env={"DCP": dcp}, preset="glm53-csf-tp6")
    assert _gather_env(plan) == {
        "VLLM_B12X_MLA_CKV_GATHER": enabled,
        "VLLM_DCP_INDEXER_KEY_GATHER": enabled,
    }


def test_glm53_dcp_ckv_gather_can_be_turned_off():
    plan = resolve(
        "glm53", env={"DCP": "2", "DCP_CKV_GATHER": "0"}, preset="glm53-csf-tp6"
    )
    assert set(_gather_env(plan).values()) == {"0"}


def test_glm53_dcp_key_gather_needs_a_vllm_that_defines_it():
    plan = resolve(
        "glm53",
        env={"DCP": "2"},
        preset="glm53-csf-tp6",
        vllm_environment=frozenset({"VLLM_B12X_MLA_CKV_GATHER"}),
    )
    assert _gather_env(plan) == {
        "VLLM_B12X_MLA_CKV_GATHER": "1",
        "VLLM_DCP_INDEXER_KEY_GATHER": None,
    }
