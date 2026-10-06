"""One tensor-parallel group across several machines (DGX Spark)."""

import pytest

from runtime import ConfigError
from runtime.launcher import configure_nodes


def run(values, replicas=1):
    derived = {}
    configure_nodes(values, replicas, lambda key, value, reason: derived.update({key: value}))
    return derived


def test_single_node_adds_nothing():
    assert run({"tensor-parallel-size": 2}) == {}


def test_worker_ranks_run_headless_and_rank_zero_serves():
    base = {"nnodes": 2, "master-addr": "10.200.0.12", "tensor-parallel-size": 2}
    assert run({**base, "node-rank": 1}) == {"headless": True}
    assert run({**base, "node-rank": 0}) == {}


@pytest.mark.parametrize(
    "values, replicas, message",
    [
        ({"nnodes": 2, "tensor-parallel-size": 2}, 1, "needs master-addr"),
        ({"nnodes": 2, "master-addr": "a", "node-rank": 2, "tensor-parallel-size": 2}, 1, "node-rank"),
        ({"nnodes": 2, "master-addr": "a", "tensor-parallel-size": 3}, 1, "split"),
        ({"nnodes": 2, "master-addr": "a", "tensor-parallel-size": 2}, 2, "replicas"),
        ({"node-rank": 1}, 1, "needs nnodes > 1"),
        ({"nnodes": 0}, 1, "at least 1"),
    ],
)
def test_inconsistent_placements_are_refused(values, replicas, message):
    with pytest.raises(ConfigError, match=message):
        run(values, replicas)
