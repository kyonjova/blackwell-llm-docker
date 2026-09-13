from pathlib import Path

from tools.jovian_wheel_runtime.verify_runtime_lock_closure import main


def write_lock(path: Path, *packages: str) -> None:
    path.write_text("".join(f"{package}==1\n" for package in packages))


def test_accepts_complete_closure(monkeypatch, tmp_path: Path) -> None:
    foundation = tmp_path / "foundation.lock"
    runtime = tmp_path / "runtime.lock"
    resolved = tmp_path / "resolved.lock"
    write_lock(foundation, "numpy")
    write_lock(runtime, "uvloop")
    write_lock(resolved, "numpy", "uvloop", "torch")
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_runtime_lock_closure.py",
            "--foundation-lock",
            str(foundation),
            "--runtime-lock",
            str(runtime),
            "--resolved-lock",
            str(resolved),
        ],
    )
    assert main() == 0


def test_rejects_missing_dependency(monkeypatch, tmp_path: Path) -> None:
    foundation = tmp_path / "foundation.lock"
    runtime = tmp_path / "runtime.lock"
    resolved = tmp_path / "resolved.lock"
    write_lock(foundation, "numpy")
    write_lock(runtime, "uvloop")
    write_lock(resolved, "numpy", "uvloop", "pycountry")
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_runtime_lock_closure.py",
            "--foundation-lock",
            str(foundation),
            "--runtime-lock",
            str(runtime),
            "--resolved-lock",
            str(resolved),
        ],
    )
    assert main() == 1
