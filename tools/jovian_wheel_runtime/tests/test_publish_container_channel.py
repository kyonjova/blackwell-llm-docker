from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from publish_container_channel import verify_cache_test_report
import publish_container_channel as publisher


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
