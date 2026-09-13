"""Apply hash-pinned Python dispatch corrections without rebuilding native code."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(manifest: Path, root: Path = Path("/")) -> None:
    specification = json.loads(manifest.read_text())
    if specification["schema"] != "local-inference-python-patches/v1":
        raise ValueError("Unsupported dependency patch manifest")
    entries = specification["patches"]
    targets = []
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Dependency target must be a relative path within root")
        target = root / relative
        if sha256(target) != entry["before_sha256"]:
            raise ValueError(f"Dependency source mismatch: {relative}")
        targets.append(target)
    # Validate every result before changing the installed package. Build-time
    # mismatch must fail rather than silently accept another dependency version.
    with tempfile.TemporaryDirectory(prefix="dependency-python-patches-") as tmp:
        staged = []
        for index, (entry, target) in enumerate(zip(entries, targets, strict=True)):
            candidate = Path(tmp) / str(index)
            candidate.write_bytes(target.read_bytes())
            subprocess.run(
                [
                    "patch",
                    "--batch",
                    "--forward",
                    str(candidate),
                    str(manifest.parent / entry["patch"]),
                ],
                check=True,
            )
            if sha256(candidate) != entry["after_sha256"]:
                raise ValueError(f"Patched dependency checksum mismatch: {target}")
            staged.append(candidate)
        for target, candidate in zip(targets, staged, strict=True):
            target.write_bytes(candidate.read_bytes())
            # Avoid importing an inherited timestamp-valid bytecode cache.
            for cached in (target.parent / "__pycache__").glob(target.stem + ".*.pyc"):
                cached.unlink()


if __name__ == "__main__":
    install(Path(__file__).with_name("dependency-python-patches.json"))
