"""GLM-5.3 precision and memory options: expert activations, router weights, lossless expert
scales, online MXFP8 per projection area, and the QAD TP2 preset that combines them."""

import json

import pytest

from runtime.launcher import (
    GLM53_MXFP8_AREAS,
    ROOT,
    ConfigError,
    deployment_presets,
    resolve,
)

TP2_KV = 8724152320
EXTERNAL = 201326592


def tp2(env=None, **kwargs):
    return resolve(
        "glm53-flash", "rtx-pro-6000-pcie", preset="glm53-tp2", env=env or {}, **kwargs
    )


def targets(plan):
    return (plan.values.get("quantization-config") or {}).get("targets", {})


def mtp_targets(plan):
    return (plan.values["speculative-config"].get("quantization_config") or {}).get(
        "targets", {}
    )


def test_manifest_groups_the_spark_projections_by_area():
    manifest = json.loads((ROOT / "glm53-mxfp8-targets.json").read_text())
    counts = {area: len(manifest["main"][area]) for area in GLM53_MXFP8_AREAS}
    assert counts == {"kda-attention": 306, "mla-attention": 88, "shared-experts": 126}
    assert {area: len(names) for area, names in manifest["mtp"].items()} == {
        "mla-attention": 8,
        "shared-experts": 3,
    }
    assert manifest["mtp_experts"] == ["model.layers.45.mlp.experts"]
    names = [name for area in GLM53_MXFP8_AREAS for name in manifest["main"][area]]
    assert len(names) == len(set(names)) == 520
    assert all(name.startswith("language_model.model.layers.") for name in names)
    assert all(
        ".shared_experts." in name for name in manifest["main"]["shared-experts"]
    )
    assert not any("kv_a_proj" in name for name in manifest["main"]["kda-attention"])


def test_qad_tp2_preset_loads_bf16_projections_as_mxfp8_with_lmcache():
    plan = tp2()
    values, env = plan.values, plan.environment
    assert values["model"] == "local-inference-lab/GLM-5.3-Flash-NVFP4"
    assert values["revision"] == "cfd47bd7680e68408924df09b179d5bed25b2ae9"
    assert (values["tensor-parallel-size"], values["decode-context-parallel-size"]) == (
        2,
        2,
    )
    assert values["cache-mode"] == "lmcache" and values["cache-l1-gib"] == 64
    assert values["cache-l2-enabled"] is False
    # The preset is sized without an external cache; LMCache keeps its own GPU buffers.
    assert values["kv-cache-memory-bytes"] == TP2_KV - EXTERNAL
    assert len(targets(plan)) == 520
    # vLLM's strict target check would refuse the separately quantized vision tower.
    assert values["quantization-config"]["strict_targets"] is False
    spec = values["speculative-config"]
    assert spec["method"] == "mtp" and spec["moe_backend"] == "b12x"
    assert spec["model"] == values["model"] and spec["revision"] == values["revision"]
    assert spec["quantization_config"]["strict_targets"] is True
    assert mtp_targets(plan)["model.layers.45.mlp.experts"] == "nvfp4_a16"
    assert sum(kind == "mxfp8" for kind in mtp_targets(plan).values()) == 11
    assert env["VLLM_B12X_MOE_FP4_FORCE_A16"] == "1"
    assert env["B12X_W4A16_FP32_TOPK_WEIGHTS"] == "1"
    assert env["VLLM_B12X_MOE_FP4_CSF"] == "1"
    assert env["VLLM_GLM53_VISION_MXFP8"] == "1"
    argv = plan.argv
    assert "--quantization-config" in argv
    for control in (
        "--expert-activations",
        "--router-weights",
        "--expert-scale-compression",
        "--mxfp8-kda-attention",
        "--mxfp8-vision",
        "--mtp-experts",
    ):
        assert not any(arg.startswith(control) for arg in argv), control


@pytest.mark.parametrize(
    ("env", "kv_cost", "main", "mtp"),
    [
        ({"MXFP8_KDA_ATTENTION": "0"}, 2264924160, 214, 12),
        ({"MXFP8_MLA_ATTENTION": "0"}, 627048448, 432, 4),
        ({"MXFP8_SHARED_EXPERTS": "0"}, 510001152, 394, 9),
        ({"EXPERT_SCALE_COMPRESSION": "off"}, 4496293888, 520, 12),
        ({"EXPERT_ACTIVATIONS": "fp4"}, 704643072, 520, 12),
        ({"MXFP8_VISION": "0"}, 268435456, 520, 12),
        ({"MTP_EXPERTS": "checkpoint"}, 5368709120, 520, 11),
    ],
)
def test_turning_a_memory_saving_off_shrinks_the_kv_cache(env, kv_cost, main, mtp):
    plan = tp2(env={**env, "CACHE_MODE": "vram"})
    assert plan.values["kv-cache-memory-bytes"] == TP2_KV - kv_cost
    assert plan.origins["kv-cache-memory-bytes"].startswith("derived:")
    assert len(targets(plan)) == main and len(mtp_targets(plan)) == mtp


def test_without_any_mxfp8_area_only_the_mtp_experts_are_quantized():
    plan = tp2(
        env={
            "MXFP8_KDA_ATTENTION": "0",
            "MXFP8_MLA_ATTENTION": "0",
            "MXFP8_SHARED_EXPERTS": "0",
        }
    )
    assert "quantization-config" not in plan.values
    assert mtp_targets(plan) == {"model.layers.45.mlp.experts": "nvfp4_a16"}


def test_vision_in_bf16_keeps_the_strict_target_check():
    assert (
        tp2(env={"MXFP8_VISION": "0"}).values["quantization-config"]["strict_targets"]
        is True
    )


def test_an_explicit_kv_size_is_used_as_given():
    plan = tp2(
        env={"EXPERT_SCALE_COMPRESSION": "off", "KV_CACHE_MEMORY_BYTES": "3000000000"}
    )
    assert plan.values["kv-cache-memory-bytes"] == 3000000000


def test_too_many_memory_costs_leave_no_room_for_the_kv_cache():
    env = {
        "EXPERT_SCALE_COMPRESSION": "off",
        "MXFP8_KDA_ATTENTION": "0",
        "MXFP8_MLA_ATTENTION": "0",
        "MXFP8_SHARED_EXPERTS": "0",
        "MXFP8_VISION": "0",
        "MAX_NUM_SEQS": "16",
    }
    with pytest.raises(
        ConfigError,
        match="leave .* GiB per GPU for the KV cache on the glm53-tp2 preset",
    ):
        tp2(env=env)


def test_expert_options_drive_their_environment_and_yield_to_explicit_variables():
    tp4 = resolve("glm53-flash", "rtx-pro-6000-pcie", env={})
    assert (
        tp4.values["expert-activations"] == "bf16"
        and tp4.values["router-weights"] == "fp32"
    )
    assert tp4.environment["VLLM_B12X_MOE_FP4_FORCE_A16"] == "1"
    assert tp4.environment["B12X_W4A16_FP32_TOPK_WEIGHTS"] == "1"
    assert "quantization-config" not in tp4.values
    fp4 = resolve("glm53-flash", "rtx-pro-6000-pcie", env={"EXPERT_ACTIVATIONS": "fp4"})
    assert fp4.environment["VLLM_B12X_MOE_FP4_FORCE_A16"] == "0"
    # FP32 router weights belong to the BF16-activation path.
    assert fp4.environment["B12X_W4A16_FP32_TOPK_WEIGHTS"] == "0"
    explicit = resolve(
        "glm53-flash", "rtx-pro-6000-pcie", env={"VLLM_B12X_MOE_FP4_FORCE_A16": "0"}
    )
    assert explicit.environment["VLLM_B12X_MOE_FP4_FORCE_A16"] == "0"
    lossless = resolve(
        "glm53-flash", "rtx-pro-6000-pcie", env={"EXPERT_SCALE_COMPRESSION": "lossless"}
    )
    assert lossless.environment["VLLM_B12X_MOE_FP4_CSF"] == "1"


def test_lossless_scale_compression_needs_b12x_without_expert_parallelism():
    with pytest.raises(
        ConfigError,
        match="Lossless expert-scale compression needs moe-backend b12x, no expert parallelism",
    ):
        resolve(
            "glm53-flash",
            "rtx-pro-6000-pcie",
            preset="glm53-tp3",
            env={"EXPERT_SCALE_COMPRESSION": "lossless"},
        )


def test_online_mxfp8_refuses_the_spark_checkpoint():
    with pytest.raises(
        ConfigError, match="Spark checkpoint already stores these projections in MXFP8"
    ):
        resolve(
            "glm53-flash",
            "rtx-pro-6000-pcie",
            preset="glm53-spark-tp2",
            env={"MXFP8_KDA_ATTENTION": "1"},
        )


def test_mxfp8_areas_work_on_the_default_tp4_profile_too():
    plan = resolve(
        "glm53-flash", "rtx-pro-6000-pcie", env={"MXFP8_SHARED_EXPERTS": "1"}
    )
    assert (
        len(targets(plan)) == 126
        and plan.values["quantization-config"]["strict_targets"] is True
    )
    # The default checkpoint's MTP experts stay as stored; only its shared expert follows.
    assert set(mtp_targets(plan).values()) == {"mxfp8"} and len(mtp_targets(plan)) == 3


def test_qad_tp2_falls_back_to_the_spark_recipe_on_an_older_vllm():
    older = tp2(
        vllm_environment=frozenset(
            {"VLLM_USE_V2_MODEL_RUNNER", "VLLM_GLM53_EMBED_HOST"}
        )
    )
    assert older.values["model"] == "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"
    assert (
        older.values["max-num-seqs"] == 4
        and older.values["kv-cache-memory-bytes"] == 4190109696
    )
    assert older.values["cache-mode"] == "vram"
    assert "quantization-config" not in older.values
    assert "quantization_config" not in older.values["speculative-config"]
    assert "VLLM_B12X_MOE_FP4_CSF" not in older.environment
    assert "VLLM_GLM53_VISION_MXFP8" not in older.environment


def test_preset_option_costs_are_validated():
    presets = deployment_presets()
    costs = presets["glm53-tp2"]["kv_bytes_for_options"]
    assert set(costs) <= {
        "expert-scale-compression",
        "mxfp8-kda-attention",
        "mxfp8-mla-attention",
        "mxfp8-shared-experts",
        "mxfp8-vision",
        "expert-activations",
        "mtp-experts",
    }
    assert all(
        isinstance(size, int)
        for by_value in costs.values()
        for size in by_value.values()
    )


@pytest.mark.parametrize(
    ("env", "option", "value", "kv_cost"),
    [
        ({"VLLM_B12X_MOE_FP4_CSF": "0"}, "expert-scale-compression", "off", 4496293888),
        ({"VLLM_B12X_MOE_FP4_FORCE_A16": "0"}, "expert-activations", "fp4", 704643072),
        ({"VLLM_GLM53_VISION_MXFP8": "0"}, "mxfp8-vision", False, 268435456),
    ],
)
def test_an_explicit_variable_decides_the_option_and_its_memory(
    env, option, value, kv_cost
):
    plan = tp2(env={**env, "CACHE_MODE": "vram"})
    assert plan.values[option] == value
    assert plan.origins[option].startswith("derived:")
    assert plan.values["kv-cache-memory-bytes"] == TP2_KV - kv_cost
    name, setting = next(iter(env.items()))
    assert plan.environment[name] == setting


def test_an_explicit_variable_contradicting_an_explicit_option_is_refused():
    with pytest.raises(ConfigError, match="VLLM_B12X_MOE_FP4_CSF=0 conflicts"):
        tp2(env={"VLLM_B12X_MOE_FP4_CSF": "0", "EXPERT_SCALE_COMPRESSION": "lossless"})


def test_an_explicit_quantization_config_replaces_the_mxfp8_areas():
    manifest = json.loads((ROOT / "glm53-mxfp8-targets.json").read_text())
    kda_only = {name: "mxfp8" for name in manifest["main"]["kda-attention"]}
    plan = tp2(
        env={
            "QUANTIZATION_CONFIG": json.dumps({"targets": kda_only}),
            "CACHE_MODE": "vram",
        }
    )
    assert plan.values["mxfp8-kda-attention"] is True
    assert plan.values["mxfp8-mla-attention"] is False
    assert plan.values["mxfp8-shared-experts"] is False
    assert plan.values["quantization-config"] == {"targets": kda_only}
    assert plan.values["kv-cache-memory-bytes"] == TP2_KV - 627048448 - 510001152
    with pytest.raises(ConfigError, match="explicit quantization-config conflicts"):
        tp2(env={"QUANTIZATION_CONFIG": "{}", "MXFP8_MLA_ATTENTION": "1"})


def test_an_explicit_mtp_draft_model_loads_as_stored():
    plan = tp2(env={"DRAFT_MODEL": "org/glm-mtp-draft", "CACHE_MODE": "vram"})
    spec = plan.values["speculative-config"]
    assert spec["model"] == "org/glm-mtp-draft"
    assert "quantization_config" not in spec
    assert plan.values["mtp-experts"] == "checkpoint"
    assert plan.values["kv-cache-memory-bytes"] == TP2_KV - 5368709120
    with pytest.raises(ConfigError, match="explicit draft-model conflicts"):
        tp2(env={"DRAFT_MODEL": "org/glm-mtp-draft", "MTP_EXPERTS": "nvfp4"})
