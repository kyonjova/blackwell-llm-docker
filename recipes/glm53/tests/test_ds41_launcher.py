"""Native DS4.1 policy isolation and public Engram storage controls."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "serve-ds41-jovian.sh"


def launch(tmp_path, overrides=None, arguments=()):
    native = tmp_path / "serve-ds41-flash.sh"
    native.write_text(
        '#!/bin/bash\nexec "$PYTHON_BIN" -c '
        "'import json,os,sys; print(json.dumps({"
        '"args":sys.argv[1:],"env":dict(os.environ)}))' + '\' "$@"\n'
    )
    native.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "PYTHON_BIN": sys.executable,
        "VLLM_SOURCE_DIR": str(tmp_path),
        **(overrides or {}),
    }
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("memory", [None, "ram", "disk"])
def test_engram_storage_is_forwarded_without_general_cpu_offload(tmp_path, memory):
    result = launch(tmp_path, {} if memory is None else {"ENGRAM_TABLE_MEMORY": memory})
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["env"]["ENGRAM_TABLE_MEMORY"] == (memory or "disk")
    assert payload["env"]["HOST"] == "0.0.0.0"
    assert payload["env"]["MAX_NUM_BATCHED_TOKENS"] == "4096"
    assert payload["env"]["MODEL_PATH"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert "--cpu-offload-gb" not in payload["args"]


def test_invalid_engram_storage_is_rejected(tmp_path):
    result = launch(tmp_path, {"ENGRAM_TABLE_MEMORY": "ssd"})
    assert result.returncode == 2
    assert "ram or disk" in result.stderr


def test_glm_tuning_is_removed_but_cache_paths_are_preserved(tmp_path):
    result = launch(
        tmp_path,
        {
            "VLLM_GLM53_MTP_DRAFT_HEAD": "nvfp4",
            "B12X_PCIE_ONESHOT_THREADS": "512",
            "B12X_COMPILE_CACHE_DIR": "/cache/b12x",
            "VLLM_CACHE_ROOT": "/cache/vllm",
            "NCCL_GRAPH_FILE": "",
        },
    )
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)["env"]
    assert "VLLM_GLM53_MTP_DRAFT_HEAD" not in env
    assert "B12X_PCIE_ONESHOT_THREADS" not in env
    assert "NCCL_GRAPH_FILE" not in env
    assert env["B12X_COMPILE_CACHE_DIR"] == "/cache/b12x"
    assert env["VLLM_CACHE_ROOT"] == "/cache/vllm"


@pytest.mark.parametrize(
    "arguments",
    [
        ("--override-generation-config", '{"top_p":1}'),
        ('--override-generation-config={"top_p":1}',),
        ("--generation-config", "auto"),
    ],
)
def test_explicit_sampling_configuration_is_not_duplicated(tmp_path, arguments):
    result = launch(tmp_path, arguments=arguments)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["args"]
    assert argv[-len(arguments) :] == list(arguments)
    assert '{"temperature":1.0,"top_p":0.95}' not in argv


def test_sampling_defaults_are_explicit(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["args"]
    assert argv.count("--override-generation-config") == 1
    assert json.loads(argv[argv.index("--override-generation-config") + 1]) == {
        "temperature": 1.0,
        "top_p": 0.95,
    }


def test_operational_environment_and_nondefault_kernel_overrides_survive(tmp_path):
    overrides = {
        "VLLM_LOGGING_LEVEL": "DEBUG",
        "VLLM_WORKER_MULTIPROC_METHOD": "forkserver",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "1",
        "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
        "B12X_POLICY_MODE": "heuristic-only",
        "B12X_PCIE_ONESHOT_THREADS": "256",
        "VLLM_DISABLED_KERNELS": "SomeOtherKernel",
    }
    result = launch(tmp_path, overrides)
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)["env"]
    for name, value in overrides.items():
        assert env[name] == value


def test_prefill_capture_is_opt_in_without_disabling_decode_graphs(tmp_path):
    result = launch(tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["env"]["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
    assert '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' in payload["args"]


def test_model_thread_graph_and_lane_defaults_replace_image_inheritance(tmp_path):
    result = launch(
        tmp_path,
        {
            "OMP_NUM_THREADS": "1",
            "MAX_CUDAGRAPH_CAPTURE_SIZE": "256",
        },
    )
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)["env"]
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["MAX_CUDAGRAPH_CAPTURE_SIZE"] == "128"
    assert env["MAX_PARALLEL_PREFILLS"] == "1"


def test_nondefault_model_capacity_and_interleaving_overrides_survive(tmp_path):
    overrides = {
        "OMP_NUM_THREADS": "2",
        "MAX_CUDAGRAPH_CAPTURE_SIZE": "64",
        "MAX_PARALLEL_PREFILLS": "auto",
    }
    result = launch(tmp_path, overrides)
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)["env"]
    for key, value in overrides.items():
        assert env[key] == value


@pytest.mark.parametrize(
    "arguments", [(), ("--swa-block-size", "64"), ("--swa_block_size=32",)]
)
def test_swa_page_environment_respects_explicit_cli(tmp_path, arguments):
    result = launch(tmp_path, {"SWA_BLOCK_SIZE": "128"}, arguments)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)["args"]
    if arguments:
        assert argv[-len(arguments) :] == list(arguments)
        assert "128" not in argv
    else:
        assert argv.count("--swa-block-size") == 1
        assert argv[argv.index("--swa-block-size") + 1] == "128"
