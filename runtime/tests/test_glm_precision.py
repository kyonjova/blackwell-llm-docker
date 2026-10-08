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
# The detection itself, before any test replaces it.
INSTALLED_SEMANTIC_A4_PREFILL = launcher.installed_semantic_a4_prefill


@pytest.fixture(autouse=True)
def threshold_image(monkeypatch):
    """Expectations default to an image whose vLLM takes the token threshold,
    whatever vLLM the tests run next to; the newer switch has its own tests."""
    monkeypatch.setattr(launcher, "installed_semantic_a4_prefill", lambda: False)


def glm(env=None, **kwargs):
    return resolve("glm53-flash", "rtx-pro-6000-pcie", env=env or {}, **kwargs)


def precision_env(plan):
    return {
        name: plan.environment.get(name) for name in (FORCE_A16, FP32_TOPK, MIN_TOKENS)
    }


def test_defaults_run_bf16_activations_fp32_router_weights_and_a4_prefill():
    plan = glm()
    assert {option: plan.values[option] for option in OPTIONS} == {
        "expert-activations": "bf16",
        "router-weights": "fp32",
        "prefill-activations": "a4",
    }
    assert precision_env(plan) == {
        FORCE_A16: "1",
        FP32_TOPK: "1",
        MIN_TOKENS: str(GLM_A4_PREFILL_MIN_TOKENS),
    }
    # B12X's default of one activation plane is what a4 measured.
    assert TERMS not in plan.environment
    assert plan.environment_origins[FORCE_A16] == "model:glm53-flash"
    # Control options configure the environment, not vLLM arguments.
    for option in OPTIONS:
        assert not any(argument.startswith(f"--{option}") for argument in plan.argv)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (
            {"PREFILL_ACTIVATIONS": "a4"},
            {
                FORCE_A16: "1",
                FP32_TOPK: "1",
                MIN_TOKENS: str(GLM_A4_PREFILL_MIN_TOKENS),
            },
        ),
        # a16 keeps BF16 activations in prefill too.
        (
            {"PREFILL_ACTIVATIONS": "a16"},
            {FORCE_A16: "1", FP32_TOPK: "1", MIN_TOKENS: "0"},
        ),
        (
            {"ROUTER_WEIGHTS": "bf16", "PREFILL_ACTIVATIONS": "a4"},
            {FORCE_A16: "1", FP32_TOPK: "0", MIN_TOKENS: "1536"},
        ),
        # FP4 activations quantize prefill and decode alike; FP32 router
        # weights belong to the W4A16 combine.
        (
            {"EXPERT_ACTIVATIONS": "fp4"},
            {FORCE_A16: "0", FP32_TOPK: "0", MIN_TOKENS: "0"},
        ),
        (
            {"EXPERT_ACTIVATIONS": "fp4", "PREFILL_ACTIVATIONS": "a16"},
            {FORCE_A16: "0", FP32_TOPK: "0", MIN_TOKENS: "0"},
        ),
    ],
)
def test_options_set_their_variables(env, expected):
    plan = glm(env)
    assert precision_env(plan) == expected
    assert TERMS not in plan.environment
    assert plan.environment_origins[MIN_TOKENS] == plan.origins["prefill-activations"]


def test_the_a4_prefill_threshold_is_1536_tokens():
    assert GLM_A4_PREFILL_MIN_TOKENS == 1536


@pytest.mark.parametrize("source", ["environment", "cli"])
def test_prefill_activations_need_bf16_expert_activations(source):
    with pytest.raises(
        ConfigError, match="prefill-activations needs expert-activations bf16"
    ):
        if source == "environment":
            glm({"EXPERT_ACTIVATIONS": "fp4", "PREFILL_ACTIVATIONS": "a4"})
        else:
            glm(argv=["--expert-activations", "fp4", "--prefill-activations", "a4"])


def test_a_default_prefill_mode_yields_to_explicit_fp4_activations(monkeypatch):
    """A profile or preset default of a4 gives way to an explicit fp4."""
    read = launcher.profile

    def a4_default(kind, identifier):
        result = read(kind, identifier)
        if kind == "model" and identifier == "glm53-flash":
            result["defaults"]["prefill-activations"] = "a4"
        return result

    monkeypatch.setattr(launcher, "profile", a4_default)
    assert glm().environment[MIN_TOKENS] == "1536"
    plan = glm({"EXPERT_ACTIVATIONS": "fp4"})
    assert plan.values["prefill-activations"] == "a16"
    assert plan.origins["prefill-activations"] == "derived:FP4 expert activations"
    assert precision_env(plan) == {FORCE_A16: "0", FP32_TOPK: "0", MIN_TOKENS: "0"}


# a8 (two NVFP4 planes) measured slower than a4 without a quality gain.
@pytest.mark.parametrize("value", ["a8", "a2", "fp8", ""])
def test_prefill_activations_accept_only_a16_or_a4(value):
    with pytest.raises(ConfigError, match="prefill-activations must be one of"):
        glm({"PREFILL_ACTIVATIONS": value})


@pytest.mark.parametrize(
    ("env", "option", "value"),
    [
        ({FORCE_A16: "0"}, "expert-activations", "fp4"),
        ({FP32_TOPK: "0"}, "router-weights", "bf16"),
        ({MIN_TOKENS: "2048"}, "prefill-activations", "a4"),
        ({MIN_TOKENS: "0"}, "prefill-activations", "a16"),
    ],
)
def test_an_explicit_variable_is_kept_and_decides_its_option(env, option, value):
    plan = glm(env)
    assert plan.values[option] == value
    for name, setting in env.items():
        assert plan.environment[name] == setting
        assert plan.environment_origins[name] == "environment"


def test_an_explicit_threshold_tunes_a4():
    plan = glm({"PREFILL_ACTIVATIONS": "a4", MIN_TOKENS: "4096"})
    assert plan.values["prefill-activations"] == "a4"
    assert plan.environment[MIN_TOKENS] == "4096"


def test_the_planes_variable_is_left_to_the_operator():
    """B12X's two-plane experiment is not an option; a value set by hand is
    passed on without changing what the option describes."""
    plan = glm({"PREFILL_ACTIVATIONS": "a4", TERMS: "2"})
    assert plan.values["prefill-activations"] == "a4"
    assert plan.environment[TERMS] == "2"
    assert plan.environment_origins[TERMS] == "environment"


def test_explicit_fp4_activations_through_the_variable_refuse_a4_prefill():
    with pytest.raises(
        ConfigError, match="prefill-activations needs expert-activations bf16"
    ):
        glm({FORCE_A16: "0", "PREFILL_ACTIVATIONS": "a4"})


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
            {"PREFILL_ACTIVATIONS": "a4", MIN_TOKENS: "0"},
            f"{MIN_TOKENS}=0 conflicts with prefill-activations a4",
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
    assert MIN_TOKENS not in plan.environment
    with pytest.raises(ConfigError, match="applies to GLM-5.3-Flash only"):
        resolve(profile_id, env={"EXPERT_ACTIVATIONS": "bf16"})


def test_other_moe_backends_drop_the_b12x_options():
    tp3 = resolve("glm53-flash", "rtx-pro-6000-pcie", preset="glm53-tp3", env={})
    assert tp3.values["moe-backend"] == "flashinfer_cutlass"
    assert not set(OPTIONS) & set(tp3.values)
    assert not {FORCE_A16, FP32_TOPK, MIN_TOKENS} & set(tp3.environment)
    with pytest.raises(
        ConfigError, match="prefill-activations applies to B12X routed experts"
    ):
        resolve(
            "glm53-flash",
            "rtx-pro-6000-pcie",
            preset="glm53-tp3",
            env={"PREFILL_ACTIVATIONS": "a4"},
        )


def test_a_vllm_without_the_a16_variable_is_left_alone():
    plan = glm(vllm_environment=frozenset({"VLLM_USE_V2_MODEL_RUNNER"}))
    assert FORCE_A16 not in plan.environment
    # B12X reads its own variables; the launcher does not check B12X versions.
    assert plan.environment[FP32_TOPK] == "1"
    assert plan.environment[MIN_TOKENS] == str(GLM_A4_PREFILL_MIN_TOKENS)


SPARK = "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"


@pytest.mark.parametrize(
    "preset, env",
    [
        (None, {"MODEL": SPARK}),
        (None, {"MODEL": "/models/GLM-5.3-Flash-NVFP4-Spark"}),
    ],
)
def test_a4_prefill_is_refused_for_the_spark_checkpoint(preset, env):
    """The Spark checkpoint's pre-QAD experts lose accuracy with a4 prefill."""
    with pytest.raises(ConfigError, match="a4 is not qualified for .*Spark"):
        resolve(
            "glm53-flash",
            "rtx-pro-6000-pcie",
            preset=preset,
            env={**env, "PREFILL_ACTIVATIONS": "a4"},
        )


def test_a_default_a4_prefill_yields_to_the_spark_checkpoint():
    plan = glm({"MODEL": SPARK})
    assert plan.values["prefill-activations"] == "a16"
    assert plan.origins["prefill-activations"].startswith("derived:")
    assert plan.environment[MIN_TOKENS] == "0"


def test_an_explicit_threshold_still_runs_a4_on_the_spark_checkpoint():
    plan = glm({"MODEL": SPARK, MIN_TOKENS: "1536"})
    assert plan.values["prefill-activations"] == "a4"
    assert plan.environment[MIN_TOKENS] == "1536"


@pytest.mark.parametrize("preset", [None, "glm53-tp2"])
def test_the_qad_checkpoints_keep_a4_prefill(preset):
    plan = resolve(
        "glm53-flash",
        "rtx-pro-6000-pcie",
        preset=preset,
        env={"PREFILL_ACTIVATIONS": "a4"},
    )
    assert plan.values["prefill-activations"] == "a4"
    assert plan.environment[MIN_TOKENS] == str(GLM_A4_PREFILL_MIN_TOKENS)


SWITCH = "B12X_W4A16_A4_PREFILL"


@pytest.fixture
def semantic(monkeypatch):
    """An image whose vLLM picks the prefill rows itself."""
    monkeypatch.setattr(launcher, "installed_semantic_a4_prefill", lambda: True)


@pytest.mark.parametrize(
    ("env", "expected"),
    [({}, "1"), ({"PREFILL_ACTIVATIONS": "a16"}, "0"), ({SWITCH: "0"}, "0")],
)
def test_newer_images_turn_a4_prefill_on_with_one_switch(semantic, env, expected):
    plan = glm(env)
    assert plan.environment[SWITCH] == expected
    assert MIN_TOKENS not in plan.environment
    assert plan.values["prefill-activations"] == ("a4" if expected == "1" else "a16")


def test_newer_images_refuse_the_retired_threshold(semantic):
    with pytest.raises(ConfigError, match=f"{MIN_TOKENS} is no longer read"):
        glm({MIN_TOKENS: "1536"})


def test_newer_images_refuse_a_contradicting_switch(semantic):
    with pytest.raises(ConfigError, match=f"{SWITCH}=1 conflicts"):
        glm(
            {"PREFILL_ACTIVATIONS": "a16", SWITCH: "1"},
            argv=["--prefill-activations", "a16"],
        )


def test_the_switch_is_detected_from_the_installed_vllm(tmp_path, monkeypatch):
    utils = tmp_path / "vllm" / "utils"
    utils.mkdir(parents=True)
    (utils / "b12x.py").write_text(
        'value = os.environ.get("B12X_W4A16_A4_PREFILL", "0")'
    )
    monkeypatch.setattr(
        launcher,
        "installed_source",
        lambda package, relative: (
            (tmp_path / package / relative).read_text()
            if (tmp_path / package / relative).is_file()
            else None
        ),
    )
    assert INSTALLED_SEMANTIC_A4_PREFILL()
    (utils / "b12x.py").write_text('os.environ.get("B12X_W4A16_A4_PREFILL_MIN_TOKENS")')
    assert not INSTALLED_SEMANTIC_A4_PREFILL()
