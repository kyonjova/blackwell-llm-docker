"""Detect import redirection and installed-code drift after wheel installation."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "verify_qwen38_runtime", Path(__file__).resolve().parents[1] / "verify_qwen38_runtime.py"
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.fixture
def installed_b12x(tmp_path, monkeypatch):
    package = tmp_path / "lib/python3.12/site-packages/b12x"
    package.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_bytes(b"standalone B12X")
    manifest = tmp_path / "share/lil-runtime/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"b12x_package": {
        "source": {"commit": "1" * 40},
        "files": {"b12x/__init__.py": hashlib.sha256(init.read_bytes()).hexdigest()},
    }}))
    monkeypatch.setattr(VERIFIER.sys, "prefix", str(tmp_path))
    monkeypatch.setitem(VERIFIER.sys.modules, "b12x", SimpleNamespace(__file__=str(init)))
    monkeypatch.setattr(
        VERIFIER.importlib.metadata, "distribution",
        lambda _: SimpleNamespace(files=[], entry_points=[]),
    )
    return init


def test_accepts_installed_wheel_payload(installed_b12x):
    VERIFIER.verify_b12x_package()


def test_rejects_redirected_import(installed_b12x, monkeypatch):
    monkeypatch.setitem(VERIFIER.sys.modules, "b12x", SimpleNamespace(
        __file__=str(installed_b12x.parent.parent / "flashinfer/b12x/__init__.py")
    ))
    with pytest.raises(RuntimeError, match="redirected"):
        VERIFIER.verify_b12x_package()


def test_rejects_overwritten_installed_code(installed_b12x):
    installed_b12x.write_bytes(b"snapshot shim")
    with pytest.raises(RuntimeError, match="payload mismatch"):
        VERIFIER.verify_b12x_package()
