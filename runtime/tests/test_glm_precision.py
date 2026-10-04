"""GLM-5.3-Flash routed-expert precision: activations, router weights and the
activation precision of prefill, mapped to the B12X and vLLM variables."""

import pytest

from runtime import launcher
from runtime.launcher import GLM_A4_PREFILL_MIN_TOKENS, ConfigError, resolve

FORCE_A16 = "VLLM_B12X_MOE_FP4_FORCE_A16"
FP32_TOPK = "B12X_W4A16_FP32_TOPK_WEIGHTS"
MIN_TOKENS = "B12X_W4A16_A4_PREFILL_MIN_TOKENS"
TERMS = "B12X_W4A16_A4_PREFILL_TERMS"
OPTIONS = ("expert-activations", "router-weights", "prefill-activations")


def glm(env=None, **kwargs):
    return resolve("glm53-flash", "rtx-pro-6000-pcie", env=env or {}, **kwargs)


def precision_env(plan):
    return {
        name: plan.environment.get(name)
        for name in (FORCE_A16, FP32_TOPK, MIN_TOKENS, TERMS)
    }


def test_defaults_run_bf16_activations_fp32_router_weights_and_a16_prefill():
    plan = glm()
    assert {option: plan.values[option] for option in OPTIONS} == {
        "expert-activations": "bf16",
        "router-weights": "fp32",
        "prefill-activations": "a16",
    }
    assert precision_env(plan) == {
        FORCE_A16: "1",
        FP32_TOPK: "1",
        MIN_TOKENS: "0",
        TERMS: None,
    }
    assert plan.environment_origins[FORCE_A16] == "model:glm53-flash"
    # Control options configure the environment, not vLLM arguments.
    for option in OPTIONS:
        assert not any(argument.startswith(f"--{option}") for argument in plan.argv)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (
            {"PREFILL_ACTIVATIONS": "a8"},
            {
                FORCE_A16: "1",
                FP32_TOPK: "1",
                MIN_TOKENS: str(GLM_A4_PREFILL_MIN_TOKENS),
                TERMS: "2",
            },
        ),
        (
            {"PREFILL_ACTIVATIONS": "a4"},
            {
                FORCE_A16: "1",
                FP32_TOPK: "1",
                MIN_TOKENS: str(GLM_A4_PREFILL_MIN_TOKENS),
                TERMS: "1",
            },
        ),
        (
            {"ROUTER_WEIGHTS": "bf16", "PREFILL_ACTIVATIONS": "a8"},
            {FORCE_A16: "1", FP32_TOPK: "0", MIN_TOKENS: "1536", TERMS: "2"},
        ),
        # FP4 activations quantize prefill and decode alike; FP32 router
        # weights belong to the W4A16 combine.
        (
            {"EXPERT_ACTIVATIONS": "fp4"},
            {FORCE_A16: "0", FP32_TOPK: "0", MIN_TOKENS: "0", TERMS: None},
        ),
        (
            {"EXPERT_ACTIVATIONS": "fp4", "PREFILL_ACTIVATIONS": "a16"},
            {FORCE_A16: "0", FP32_TOPK: "0", MIN_TOKENS: "0", TERMS: None},
        ),
    ],
)
def test_options_set_their_variables(env, expected):
    plan = glm(env)
    assert precision_env(plan) == expected
    assert plan.environment_origins[MIN_TOKENS] == plan.origins["prefill-activations"]


def test_the_a4_prefill_threshold_is_1536_tokens():
    assert GLM_A4_PREFILL_MIN_TOKENS == 1536


@pytest.mark.parametrize("prefill", ["a8", "a4"])
@pytest.mark.parametrize("source", ["environment", "cli"])
def test_prefill_activations_need_bf16_expert_activations(prefill, source):
    with pytest.raises(
        ConfigError, match="prefill-activations needs expert-activations bf16"
    ):
        if source == "environment":
            glm({"EXPERT_ACTIVATIONS": "fp4", "PREFILL_ACTIVATIONS": prefill})
        else:
            glm(
                argv=[
                    "--expert-activations",
                    "fp4",
                    "--prefill-activations",
                    prefill,
                ]
            )


def test_a_default_prefill_mode_yields_to_explicit_fp4_activations(monkeypatch):
    """A profile or preset default of a8 or a4 gives way to an explicit fp4."""
    read = launcher.profile

    def a8_default(kind, identifier):
        result = read(kind, identifier)
        if kind == "model" and identifier == "glm53-flash":
            result["defaults"]["prefill-activations"] = "a8"
        return result

    monkeypatch.setattr(launcher, "profile", a8_default)
    assert glm().environment[TERMS] == "2"
    plan = glm({"EXPERT_ACTIVATIONS": "fp4"})
    assert plan.values["prefill-activations"] == "a16"
    assert plan.origins["prefill-activations"] == "derived:FP4 expert activations"
    assert precision_env(plan) == {
        FORCE_A16: "0",
        FP32_TOPK: "0",
        MIN_TOKENS: "0",
        TERMS: None,
    }


@pytest.mark.parametrize("value", ["a2", "fp8", ""])
def test_prefill_activations_accept_only_a16_a8_or_a4(value):
    with pytest.raises(ConfigError, match="prefill-activations must be one of"):
        glm({"PREFILL_ACTIVATIONS": value})


@pytest.mark.parametrize(
    ("env", "option", "value"),
    [
        ({FORCE_A16: "0"}, "expert-activations", "fp4"),
        ({FP32_TOPK: "0"}, "router-weights", "bf16"),
        ({MIN_TOKENS: "2048"}, "prefill-activations", "a4"),
        ({MIN_TOKENS: "2048", TERMS: "2"}, "prefill-activations", "a8"),
        ({MIN_TOKENS: "0"}, "prefill-activations", "a16"),
    ],
)
def test_an_explicit_variable_is_kept_and_decides_its_option(env, option, value):
    plan = glm(env)
    assert plan.values[option] == value
    for name, setting in env.items():
        assert plan.environment[name] == setting
        assert plan.environment_origins[name] == "environment"


def test_an_explicit_threshold_keeps_the_chosen_number_of_planes():
    plan = glm({"PREFILL_ACTIVATIONS": "a8", MIN_TOKENS: "4096"})
    assert plan.values["prefill-activations"] == "a8"
    assert plan.environment[MIN_TOKENS] == "4096"
    assert plan.environment[TERMS] == "2"


def test_explicit_fp4_activations_through_the_variable_refuse_a8_prefill():
    with pytest.raises(
        ConfigError, match="prefill-activations needs expert-activations bf16"
    ):
        glm({FORCE_A16: "0", "PREFILL_ACTIVATIONS": "a8"})


@pytest.mark.parametrize(
    ("env", "match"),
    [
        (
            {"EXPERT_ACTIVATIONS": "bf16", FORCE_A16: "0"},
            f"{FORCE_A16}=0 conflicts with expert-activations bf16",
        ),
        (
            {"ROUTER_WEIGHTS": "fp32", FP32_TOPK: "0"},
            f"{FP32_TOPK}=0 conflicts with router-weights fp32",
        ),
        (
            {"PREFILL_ACTIVATIONS": "a16", MIN_TOKENS: "1536"},
            f"{MIN_TOKENS}=1536 conflicts with prefill-activations a16",
        ),
        (
            {"PREFILL_ACTIVATIONS": "a4", TERMS: "2"},
            f"{TERMS}=2 conflicts with prefill-activations a4",
        ),
    ],
)
def test_an_explicit_variable_contradicting_an_explicit_option_is_refused(env, match):
    with pytest.raises(ConfigError, match=match):
        glm(env)


@pytest.mark.parametrize("profile_id", ["qwen38-flash-next", "ds4-flash", "ds41-flash"])
def test_the_options_belong_to_glm(profile_id):
    plan = resolve(profile_id, env={})
    assert not set(OPTIONS) & set(plan.values)
    assert MIN_TOKENS not in plan.environment and TERMS not in plan.environment
    with pytest.raises(ConfigError, match="applies to GLM-5.3-Flash only"):
        resolve(profile_id, env={"EXPERT_ACTIVATIONS": "bf16"})


def test_other_moe_backends_drop_the_b12x_options():
    tp3 = resolve("glm53-flash", "rtx-pro-6000-pcie", preset="glm53-tp3", env={})
    assert tp3.values["moe-backend"] == "flashinfer_cutlass"
    assert not set(OPTIONS) & set(tp3.values)
    assert not {FORCE_A16, FP32_TOPK, MIN_TOKENS, TERMS} & set(tp3.environment)
    with pytest.raises(
        ConfigError, match="prefill-activations applies to B12X routed experts"
    ):
        resolve(
            "glm53-flash",
            "rtx-pro-6000-pcie",
            preset="glm53-tp3",
            env={"PREFILL_ACTIVATIONS": "a8"},
        )


def test_a_vllm_without_the_a16_variable_is_left_alone():
    plan = glm(vllm_environment=frozenset({"VLLM_USE_V2_MODEL_RUNNER"}))
    assert FORCE_A16 not in plan.environment
    # B12X reads its own variables; the launcher does not check B12X versions.
    assert plan.environment[FP32_TOPK] == "1"
    assert plan.environment[MIN_TOKENS] == "0"


def test_the_spark_preset_follows_the_profile_precision():
    plan = resolve("glm53-flash", "rtx-pro-6000-pcie", preset="glm53-spark-tp2", env={})
    assert precision_env(plan) == {
        FORCE_A16: "1",
        FP32_TOPK: "1",
        MIN_TOKENS: "0",
        TERMS: None,
    }
    fp4 = resolve(
        "glm53-flash",
        "rtx-pro-6000-pcie",
        preset="glm53-spark-tp2",
        env={"EXPERT_ACTIVATIONS": "fp4"},
    )
    assert fp4.environment[FORCE_A16] == "0"
