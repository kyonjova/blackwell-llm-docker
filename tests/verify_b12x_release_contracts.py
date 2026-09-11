"""CPU-only graph-dispatch and quantization guards for the community image.

Run with the image's Python and B12X package. FakeTensor dispatch must not
execute a CUDA implementation. Keyword arguments follow the installed operator
schema; extending that schema must not invalidate this graph-interface check.
These checks do not qualify CUDA arithmetic or serving performance.
"""

import importlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.parametrize(
    "module,operation",
    [
        ("b12x._lib.dense_gemm", "dense_gemm_launch"),
        *[
            ("b12x.norm.mhc._kernels", operation)
            for operation in (
                "mhc_pre_partial_launch",
                "mhc_post_pre_partial_launch",
                "mhc_finalize_gram_launch",
                "mhc_prefill_tf32_project_launch",
            )
        ],
        *[
            ("b12x.moe.fused_moe._impl", operation)
            for operation in (
                "tp_moe_dynamic_launch",
                "tp_moe_compact_micro_launch",
            )
        ],
        *[
            ("b12x.moe._shared.kernels.w4a16.kernel", operation)
            for operation in (
                "w4a16_small_m_direct_launch",
                "w4a16_fused_moe_launch",
                "w4a16_fused_moe_calibrated_launch",
                "w4a16_topk_sum_launch",
            )
        ],
    ],
)
def test_launch_operator_fake_dispatch(module, operation):
    importlib.import_module(module)
    operator = getattr(torch.ops.b12x, operation).default
    primitives = {"int": 1, "bool": False, "float": 1.0, "str": "default"}
    assert not operator._schema.returns
    with FakeTensorMode():
        operands = {}
        for argument in operator._schema.arguments:
            kind = str(argument.type)
            if argument.has_default_value():
                operands[argument.name] = argument.default_value
            elif kind == "Tensor":
                operands[argument.name] = torch.empty((2, 128), dtype=torch.bfloat16)
            elif kind in primitives:
                operands[argument.name] = primitives[kind]
            elif kind.startswith("Optional["):
                operands[argument.name] = None
            else:
                raise AssertionError(f"Unsupported schema type {kind}: {argument.name}")
        assert operator(**operands) is None


@pytest.mark.parametrize(
    "quant_mode,intermediate_size,expected",
    [
        ("nvfp4", 640, True),
        ("nvfp4", 320, False),
        ("w4a16", 640, False),
        ("w4a8_mx", 640, False),
    ],
    ids=["qwen-target-tp1", "qwen-target-tp2", "qwen-draft", "mxfp4"],
)
def test_shared_scales_preserve_recipe_and_tile_guards(
    monkeypatch, quant_mode, intermediate_size, expected
):
    """An input-scale equality proof cannot override FP4 recipe/tile limits."""
    from b12x.moe.fused_moe import _impl

    monkeypatch.setenv("B12X_NVFP4_DYNAMIC_MATERIALIZED", "1")
    monkeypatch.setenv("B12X_DYNAMIC_WORK_SOURCE", "persistent_grid")
    assert (
        _impl._nvfp4_dynamic_materialized_enabled(
            quant_mode=quant_mode,
            activation="silu",
            routed_rows=4096 * 10,
            num_experts=512,
            k=2560,
            n=intermediate_size,
            share_input_across_experts=True,
            deterministic_output=False,
            planned_tile_m=128,
        )
        is expected
    )
