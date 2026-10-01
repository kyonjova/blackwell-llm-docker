"""Replicas mode runs independent TP1 servers behind one proxy on one checkpoint."""

import pytest

from runtime import replicas
from runtime.launcher import ConfigError, resolve


def test_qwen_replicas_share_one_ple_table_and_stay_single_gpu():
    plan = resolve("qwen38-flash-next", env={"REPLICAS": "2"})

    assert plan.values["replicas"] == 2
    assert plan.environment["VLLM_PLE_TABLE_MEMORY"] == "shared"
    assert (
        plan.environment["VLLM_PLE_SHARED_TABLE_DIR"] == replicas.PLE_SHARED_DIRECTORY
    )
    assert "--replicas" not in plan.argv
    assert "VLLM_PLE_TABLE_MEMORY" not in resolve("qwen38-flash-next").environment
    explicit = resolve(
        "qwen38-flash-next", env={"REPLICAS": "2", "VLLM_PLE_TABLE_MEMORY": "disk"}
    )
    assert explicit.environment["VLLM_PLE_TABLE_MEMORY"] == "disk"
    assert resolve("qwen38-flash-next", preset="qwen38-tp1x2").values["replicas"] == 2


def test_a_vllm_without_shared_tables_keeps_a_table_per_replica():
    shared = resolve(
        "qwen38-flash-next",
        env={"REPLICAS": "2"},
        vllm_environment=frozenset({"VLLM_PLE_SHARED_TABLE_DIR"}),
    )
    private = resolve(
        "qwen38-flash-next",
        env={"REPLICAS": "2"},
        vllm_environment=frozenset({"VLLM_PLE_CPU_OFFLOAD"}),
    )

    assert shared.environment["VLLM_PLE_TABLE_MEMORY"] == "shared"
    assert "VLLM_PLE_TABLE_MEMORY" not in private.environment
    assert any("own host-RAM copy" in warning for warning in private.warnings)


@pytest.mark.parametrize(
    "env,error",
    [
        ({"REPLICAS": "2", "TP": "2"}, "tensor-parallel-size must be 1"),
        ({"REPLICAS": "2", "DCP": "2"}, "decode-context-parallel-size must be 1"),
        ({"REPLICAS": "0"}, "at least 1"),
    ],
)
def test_replicas_reject_multi_gpu_servers(env, error):
    with pytest.raises(ConfigError, match=error):
        resolve("qwen38-flash-next", env=env)


def test_each_replica_gets_one_gpu_and_a_loopback_port_behind_the_proxy():
    plan = resolve("qwen38-flash-next", env={"REPLICAS": "2", "PORT": "9000"})

    servers = replicas.replica_servers(
        plan, {"PATH": "/bin"}, ["/boot"], gpus=["3", "5"], ports=[41000, 41001]
    )

    assert [name for name, _, _ in servers] == [
        "Replica 0",
        "Replica 1",
        "Replica proxy",
    ]
    for (_, command, environment), gpu, port in zip(
        servers, ["3", "5"], [41000, 41001]
    ):
        assert command[0] == "/boot"
        assert command[command.index("--port") + 1] == str(port)
        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert environment["CUDA_VISIBLE_DEVICES"] == gpu
        assert environment["PATH"] == "/bin"
    proxy = servers[2][1]
    assert proxy[1:3] == ["-m", "runtime.replica_proxy"]
    assert proxy[proxy.index("--host") + 1] == "0.0.0.0"
    assert proxy[proxy.index("--port") + 1] == "9000"
    assert proxy[-2:] == ["--replica=127.0.0.1:41000", "--replica=127.0.0.1:41001"]
    with pytest.raises(ConfigError, match="needs 2 visible GPUs"):
        replicas.replica_servers(plan, {}, [], gpus=["3"], ports=[41000, 41001])


def test_visible_gpus_follow_cuda_visible_devices():
    assert replicas.visible_gpus({"CUDA_VISIBLE_DEVICES": "2, 7"}) == ["2", "7"]
    ports = replicas.loopback_ports(3)
    assert len(set(ports)) == 3
