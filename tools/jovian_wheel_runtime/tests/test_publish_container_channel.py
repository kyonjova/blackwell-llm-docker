from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import publish_container_channel as publisher
from publish_container_channel import verify_cache_test_report

QUALIFICATION_GPU = "GPU-2faf5385-78f7-dab0-5528-dfaca9cc8eb8"


def test_native_smoke_checks_disk_syscalls_with_serving_permissions():
    image = "ghcr.io/local-inference-lab/vllm@sha256:" + "a" * 64
    command = publisher.runtime_smoke_command(image, QUALIFICATION_GPU)
    before, after = command[: command.index(image)], command[command.index(image) + 1 :]
    assert before[before.index("--device") + 1] == f"nvidia.com/gpu={QUALIFICATION_GPU}"
    assert before[before.index("--security-opt") + 1] == "seccomp=unconfined"
    assert before[before.index("--ulimit") + 1] == "memlock=-1"
    assert "--require-gpu" in after and "--require-io-uring" in after


@pytest.mark.parametrize(
    "name",
    [
        "community-assembly-main.json",
        "community-assembly-beta.json",
        "community-assembly-karmic-kraken-beta.json",
        "manual-lock.json",
    ],
)
def test_publication_uses_canonical_assembly_asset_name(tmp_path, name):
    source = tmp_path / name
    payload = b'{"assembly_sha256":"example"}\n'
    source.write_bytes(payload)
    output = tmp_path / "publication"
    output.mkdir()
    staged = publisher.stage_assembly_lock(source, output)
    assert staged == output / "community-assembly.json"
    assert staged.read_bytes() == source.read_bytes() == payload


def gpu_snapshot(*, memory_mib=2, utilization="0 %", process_type=None):
    root = ET.Element("nvidia_smi_log")
    device = ET.SubElement(root, "gpu")
    ET.SubElement(device, "uuid").text = QUALIFICATION_GPU
    memory = ET.SubElement(device, "fb_memory_usage")
    ET.SubElement(memory, "used").text = f"{memory_mib} MiB"
    usage = ET.SubElement(device, "utilization")
    ET.SubElement(usage, "gpu_util").text = utilization
    processes = ET.SubElement(device, "processes")
    if process_type is not None:
        process = ET.SubElement(processes, "process_info")
        ET.SubElement(process, "pid").text = "1234"
        ET.SubElement(process, "type").text = process_type
    return root


@pytest.mark.parametrize("memory_mib", [0, 2, 3, 32])
def test_process_free_gpu_does_not_require_zero_vram(monkeypatch, memory_mib):
    snapshot = gpu_snapshot(memory_mib=memory_mib)
    calls = []

    def query(args):
        calls.append(args)
        return ET.tostring(snapshot)

    monkeypatch.setattr(publisher, "run", query)
    publisher.require_idle_gpu(QUALIFICATION_GPU)
    assert calls == [["nvidia-smi", "-i", QUALIFICATION_GPU, "--query", "--xml-format"]]


@pytest.mark.parametrize("process_type", ["C", "G", "C+G", "M"])
def test_resident_process_blocks_qualification_even_when_idle(
    monkeypatch, process_type
):
    snapshot = gpu_snapshot(memory_mib=0, process_type=process_type)
    monkeypatch.setattr(publisher, "run", lambda args: ET.tostring(snapshot))
    with pytest.raises(RuntimeError, match="GPU is busy; no workload was stopped"):
        publisher.require_idle_gpu(QUALIFICATION_GPU)


@pytest.mark.parametrize("utilization", ["1 %", "100 %"])
def test_gpu_activity_blocks_qualification_without_visible_processes(
    monkeypatch, utilization
):
    snapshot = gpu_snapshot(memory_mib=0, utilization=utilization)
    monkeypatch.setattr(publisher, "run", lambda args: ET.tostring(snapshot))
    with pytest.raises(RuntimeError, match="GPU is busy"):
        publisher.require_idle_gpu(QUALIFICATION_GPU)


@pytest.mark.parametrize("utilization", ["N/A", "", "0", "zero %", "101 %"])
def test_unknown_gpu_activity_fails_closed(monkeypatch, utilization):
    snapshot = gpu_snapshot(utilization=utilization)
    monkeypatch.setattr(publisher, "run", lambda args: ET.tostring(snapshot))
    with pytest.raises(RuntimeError, match="cannot verify.*activity"):
        publisher.require_idle_gpu(QUALIFICATION_GPU)


@pytest.mark.parametrize("processes_text", [None, "N/A", "Not Supported"])
def test_missing_process_visibility_fails_closed(monkeypatch, processes_text):
    snapshot = gpu_snapshot()
    device = snapshot.find("gpu")
    processes = device.find("processes")
    if processes_text is None:
        device.remove(processes)
    else:
        processes.text = processes_text
    monkeypatch.setattr(publisher, "run", lambda args: ET.tostring(snapshot))
    with pytest.raises(RuntimeError, match="cannot verify.*activity"):
        publisher.require_idle_gpu(QUALIFICATION_GPU)


@pytest.mark.parametrize("identity", ["missing", "mismatch", "multiple"])
def test_gpu_query_must_match_exact_device(monkeypatch, identity):
    snapshot = gpu_snapshot()
    if identity == "missing":
        snapshot.remove(snapshot.find("gpu"))
    elif identity == "multiple":
        snapshot.append(gpu_snapshot().find("gpu"))
    else:
        snapshot.find("gpu/uuid").text = "GPU-another-device"
    monkeypatch.setattr(publisher, "run", lambda args: ET.tostring(snapshot))
    with pytest.raises(RuntimeError, match="cannot verify.*identity"):
        publisher.require_idle_gpu(QUALIFICATION_GPU)


def test_gpu_query_requires_explicit_uuid():
    with pytest.raises(ValueError, match="explicit GPU UUID"):
        publisher.require_idle_gpu("14")


@pytest.mark.parametrize("skip_checkpoint", [False, True])
def test_qualification_requires_executed_cache_contract_groups(
    tmp_path, skip_checkpoint
):
    root = ET.Element("testsuites")
    for module in (
        "test_checkpoint_identity",
        "test_checkpoint_index",
        "test_checkpoint_storage",
        "test_fs_native_connector",
        "test_vllm_semantic_checkpoint_transfer",
    ):
        case = ET.SubElement(root, "testcase", classname=f"tests.v1.{module}")
        if skip_checkpoint and module == "test_checkpoint_storage":
            ET.SubElement(case, "skipped")
    report = tmp_path / "report.xml"
    ET.ElementTree(root).write(report)
    if skip_checkpoint:
        with pytest.raises(ValueError, match="not executed"):
            verify_cache_test_report(report)
    else:
        assert verify_cache_test_report(report) == {"passed": 5, "skipped": 0}


@pytest.mark.parametrize("exists", [False, True])
def test_release_upload_repairs_partial_release_before_publishing(
    monkeypatch, tmp_path, exists
):
    calls = []
    monkeypatch.setattr(
        publisher, "run", lambda args: b"beta-example\n" if exists else b""
    )
    monkeypatch.setattr(publisher, "execute", lambda args: calls.append(args))
    publisher.publish_release(
        "local-inference-lab/blackwell-llm-docker",
        {
            "release_tag": "beta-example",
            "image": "ghcr.io/local-inference-lab/vllm:beta-example",
        },
        "a" * 40,
        tmp_path / "notes.md",
        [tmp_path / "manifest.json"],
    )
    assert [call[2] for call in calls] == (
        ["upload", "edit"] if exists else ["create", "upload", "edit"]
    )
    if not exists:
        assert "--draft" in calls[0]
    assert "--clobber" in calls[-2]
    assert "--draft=false" in calls[-1]
