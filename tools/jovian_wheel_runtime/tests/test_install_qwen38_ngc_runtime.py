import importlib.util
from pathlib import Path

import pytest


TOOL_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "install_qwen38_ngc_runtime", TOOL_DIR / "install_qwen38_ngc_runtime.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def manifest(source_image: str) -> dict[str, object]:
    foundation = [
        ("torch", "2.14"),
        ("torchvision", "0.29"),
        ("triton", "3.8"),
        ("triton_kernels", "1.0"),
        ("flash_attn", "2.7"),
    ]
    return {
        "schema": installer.SCHEMA,
        "components": {
            "foundation": {"source": {"image": source_image}},
            "vllm": {"source": {"commit": "1" * 40}},
        },
        "packages": [
            *(
                {
                    "component": "foundation",
                    "name": name,
                    "version": version,
                    "sha256": "1" * 64,
                }
                for name, version in foundation
            ),
            {
                "component": "vllm",
                "name": "vllm",
                "version": "0.1.dev1",
                "sha256": "2" * 64,
            },
        ],
    }


def test_accepts_exact_ngc_package_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "torch": "2.14",
        "torchvision": "0.29",
        "triton": "3.8",
        "triton-kernels": "1.0",
        "flash-attn": "2.7",
    }
    monkeypatch.setattr(installer.importlib.metadata, "version", versions.__getitem__)
    installer.require_ngc_foundation(manifest("ngc@example"), "ngc@example")


def test_rejects_another_ngc_image(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="different NGC image"):
        installer.require_ngc_foundation(manifest("ngc@example"), "ngc@other")


def test_rejects_ngc_package_version_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(installer.importlib.metadata, "version", lambda _: "unexpected")
    with pytest.raises(ValueError, match="does not match"):
        installer.require_ngc_foundation(manifest("ngc@example"), "ngc@example")


def test_application_requirements_exclude_ngc_foundation(tmp_path: Path) -> None:
    output = tmp_path / "requirements.txt"
    installer.write_application_requirements(manifest("ngc@example"), output)
    assert output.read_text() == f"vllm==0.1.dev1 --hash=sha256:{'2' * 64}\n"
