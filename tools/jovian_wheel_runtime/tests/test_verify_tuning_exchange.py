"""Reject mismatched installed startup protocols before publishing an image."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "verify_qwen38_runtime",
    Path(__file__).resolve().parents[1] / "verify_qwen38_runtime.py",
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.mark.parametrize("producer_has_count", [False, True])
@pytest.mark.parametrize("consumer_uses_count", [False, True])
def test_package_gate_checks_both_sides_of_tuning_exchange(
    monkeypatch, producer_has_count, consumer_uses_count
):
    @dataclass
    class Requirement:
        key: str
        ranks: tuple
        assignment: dict
        latency_us: float
        candidate_index: int

        def __post_init__(self):
            assignment = self.assignment
            self.assignment = SimpleNamespace(to_dict=lambda: assignment)

    if producer_has_count:

        @dataclass
        class CountedRequirement(Requirement):
            rejected_count: int = 0

        requirement_type = CountedRequirement
    else:
        requirement_type = Requirement

    def ready(coordinator):
        (item,) = coordinator._last_progress.ready_tuning
        row = (
            item.key,
            item.ranks,
            item.assignment.to_dict(),
            item.latency_us,
            item.candidate_index,
        )
        return (row + (item.rejected_count,) if consumer_uses_count else row,)

    def authorize(rows, ranks):
        assert ranks == (0, 1)
        assert [row["global_rank"] for row in rows] == [0, 1]
        return (min((row["tuning"][0] for row in rows), key=lambda row: row[3]),)

    monkeypatch.setitem(
        VERIFIER.sys.modules,
        "b12x.preparation",
        SimpleNamespace(
            TuningRequirement=requirement_type,
        ),
    )
    monkeypatch.setitem(
        VERIFIER.sys.modules,
        "vllm.v1.worker.b12x_startup",
        SimpleNamespace(
            B12xPreparationCoordinator=SimpleNamespace(_ready_tuning=ready),
            _authorize_tuning=authorize,
        ),
    )
    if consumer_uses_count and not producer_has_count:
        with pytest.raises(AttributeError, match="rejected_count"):
            VERIFIER.verify_b12x_tuning_exchange()
    else:
        VERIFIER.verify_b12x_tuning_exchange()
