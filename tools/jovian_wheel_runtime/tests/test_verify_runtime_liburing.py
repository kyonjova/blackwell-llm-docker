"""Verify disk-table dependency checks without requiring CUDA or host packages."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "verify_qwen38_runtime",
    Path(__file__).resolve().parents[1] / "verify_qwen38_runtime.py",
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_missing_headers_metadata_is_an_actionable_failure(monkeypatch):
    def missing(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(verifier.subprocess, "check_output", missing)
    with pytest.raises(RuntimeError, match="liburing-dev and pkg-config"):
        verifier.verify_liburing()


def test_rejects_liburing_version_drift(monkeypatch):
    monkeypatch.setattr(verifier.subprocess, "check_output", lambda *a, **k: "9.9\n")
    with pytest.raises(RuntimeError, match="expected '2.5'"):
        verifier.verify_liburing()


@pytest.mark.parametrize("require_queue", [False, True])
def test_build_probe_defers_only_syscall_policy(monkeypatch, require_queue):
    def metadata(command, **kwargs):
        return "2.5\n" if "--modversion" in command else "-I/usr/include -luring\n"

    calls = []
    monkeypatch.setattr(verifier.subprocess, "check_output", metadata)
    monkeypatch.setattr(
        verifier.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    result = verifier.verify_liburing(require_io_uring=require_queue)
    compiler, run = calls
    assert compiler[0][0][0] == "cc"
    assert "-luring" in compiler[0][0]
    assert "#include <liburing.h>" in compiler[1]["input"]
    assert "io_uring_queue_init" in compiler[1]["input"]
    assert compiler[1]["check"] and run[1]["check"]
    assert ("--queue" in run[0][0]) is require_queue
    assert result["queue_probe"] == ("pass" if require_queue else "not-requested")


def test_failed_queue_probe_cannot_report_success(monkeypatch):
    monkeypatch.setattr(
        verifier.subprocess,
        "check_output",
        lambda command, **kwargs: "2.5" if "--modversion" in command else "-luring",
    )

    def denied(command, **kwargs):
        if "--queue" in command:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(verifier.subprocess, "run", denied)
    with pytest.raises(subprocess.CalledProcessError):
        verifier.verify_liburing(require_io_uring=True)
