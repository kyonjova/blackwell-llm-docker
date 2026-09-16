#!/usr/bin/env python3
"""Check CPU custom-op mutation semantics and the serving dispatch fast path.

Run in a disposable build/validation process, never in a serving worker. The
test temporarily instruments schema expansion; no CUDA context is created.
PyTorch 2.13 needs the immutable-metadata backport. The pinned NGC 2.14 build
already implements direct mutation tracking without per-call schema expansion.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch


def verify() -> dict:
    import torch
    from torch._library import utils

    @torch.library.custom_op("lil_runtime_contract::mutate", mutates_args=("x",))
    def mutate(x: torch.Tensor, amount: int = 1) -> None:
        # External kernels do not perform a second ATen version increment.
        x.numpy()[...] += amount

    @torch.library.custom_op("lil_runtime_contract::mutate_list", mutates_args=("xs",))
    def mutate_list(xs: list[torch.Tensor], amount: int = 1) -> None:
        for x in xs:
            x.numpy()[...] += amount

    @torch.library.custom_op("lil_runtime_contract::optional", mutates_args=("x",))
    def optional(anchor: torch.Tensor, x: torch.Tensor | None = None) -> None:
        if x is not None:
            x.numpy()[...] += 2

    calls = 0
    fill_defaults = utils.fill_defaults

    def track(*args, **kwargs):
        nonlocal calls
        calls += 1
        return fill_defaults(*args, **kwargs)

    cases = 0
    with torch.no_grad(), patch.object(utils, "fill_defaults", track):
        x = torch.zeros(2)
        for call, amount in (
            (lambda: mutate(x), 1),
            (lambda: mutate(x=x, amount=3), 3),
            (lambda: torch.ops.lil_runtime_contract.mutate.default(x, 2), 2),
        ):
            before = x.clone()
            version = x._version
            call()
            torch.testing.assert_close(x, before + amount)
            if x._version != version + 1:
                raise RuntimeError(
                    "Mutable custom op must increment its input version once"
                )
            cases += 1
        xs = [torch.zeros(2), torch.ones(2)]
        before = [x.clone() for x in xs]
        versions = [x._version for x in xs]
        mutate_list(xs, 4)
        for x, reference, version in zip(xs, before, versions):
            torch.testing.assert_close(x, reference + 4)
            if x._version != version + 1:
                raise RuntimeError(
                    "Mutable tensor-list elements must each increment once"
                )
        cases += 1
        anchor, x = torch.zeros(2), torch.zeros(2)
        optional(anchor)
        optional(anchor, None)
        if anchor._version != 0:
            raise RuntimeError("Non-mutated arguments must retain their version")
        optional(anchor, x=x)
        torch.testing.assert_close(x, torch.full_like(x, 2))
        if x._version != 1 or anchor._version != 0:
            raise RuntimeError(
                "Optional mutation must track only the supplied mutable input"
            )
        cases += 3
    if calls:
        raise RuntimeError(f"Mutable custom-op dispatch expanded schemas {calls} times")

    class Schema:
        reads = 0

        @property
        def arguments(self):
            self.reads += 1
            return [
                SimpleNamespace(name="x", kwarg_only=False, default_value=None),
                SimpleNamespace(name="amount", kwarg_only=False, default_value=1),
                SimpleNamespace(name="enabled", kwarg_only=True, default_value=True),
            ]

    schema = Schema()
    args, kwargs = utils.fill_defaults(schema, ("input",), {})
    if tuple(args) != ("input", 1) or kwargs != {"enabled": True}:
        raise RuntimeError(
            "Schema expansion must preserve positional and keyword defaults"
        )
    if schema.reads != 1:
        raise RuntimeError(
            "Schema arguments must be materialized once, not once per argument"
        )
    return {
        "status": "qualified",
        "scope": "CPU mutation/version semantics and zero dispatch-time fill_defaults calls",
        "torch": torch.__version__,
        "cases": cases,
        "dispatch_schema_expansions": calls,
        "schema_argument_materializations": schema.reads,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
