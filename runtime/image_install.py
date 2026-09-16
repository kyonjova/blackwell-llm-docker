"""Install model profiles into a neutral runtime without modifying serving packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from runtime import ConfigError
from runtime.entrypoint import LEGACY_PROFILES
from runtime.launcher import ROOT
from runtime.packaging import make_contract, payload_sources


def install(
    metadata: dict,
    runtime_identity: str,
    bootstrap: str | None,
    *,
    destination: Path = Path("/opt/lil/runtime"),
    bin_directory: Path = Path("/usr/local/bin"),
    python_site: Path = Path("/opt/venv/lib/python3.12/site-packages"),
) -> dict:
    contract = make_contract(
        metadata, runtime_identity, [bootstrap] if bootstrap else []
    )
    if bootstrap and not Path(bootstrap).is_absolute():
        raise ConfigError("The CUDA/NCCL bootstrap must be an absolute executable path")
    if destination.exists() or destination.is_symlink():
        raise ConfigError(
            "Install profiles into a neutral foundation: runtime destination already exists"
        )
    launchers = {"lil-serve", "lil-entrypoint", *LEGACY_PROFILES}
    for name in launchers:
        target = bin_directory / name
        if target.exists() or target.is_symlink():
            raise ConfigError(f"Refusing to replace an inherited launcher: {target}")
    package_path = python_site / "lil-model-runtime.pth"
    if package_path.exists() or package_path.is_symlink():
        raise ConfigError("Refusing to replace an inherited runtime import path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bin_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".runtime-install-", dir=destination.parent
    ) as tmp:
        payload = Path(tmp) / "runtime"
        payload.mkdir()
        for relative, source in payload_sources().items():
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (payload / "image-contract.json").write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n"
        )
        staged_launchers = Path(tmp) / "launchers"
        staged_launchers.mkdir()
        for name in ("lil-serve", "lil-entrypoint"):
            target = staged_launchers / name
            shutil.copyfile(ROOT / name, target)
            target.chmod(0o755)
        for name in LEGACY_PROFILES:
            target = staged_launchers / name
            # Literal, repository-owned names only; caller arguments remain argv.
            target.write_text(
                f'#!/bin/sh\nexec /usr/local/bin/lil-entrypoint {name} "$@"\n'
            )
            target.chmod(0o755)
        python_site.mkdir(parents=True, exist_ok=True)
        created = []
        try:
            outputs = [
                (path, bin_directory / path.name) for path in staged_launchers.iterdir()
            ]
            staged_path = Path(tmp) / package_path.name
            staged_path.write_text(str(destination.parent.resolve()) + "\n")
            outputs.append((staged_path, package_path))
            for source, target in outputs:
                # Exclusive creation prevents overwriting files introduced after
                # preflight. Only files owned by this invocation are rolled back.
                with target.open("xb") as output:
                    created.append(target)
                    with source.open("rb") as input_file:
                        shutil.copyfileobj(input_file, output)
                target.chmod(source.stat().st_mode & 0o777)
            os.rename(payload, destination)
        except BaseException:
            for target in reversed(created):
                target.unlink()
            raise
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-inspect", type=Path, required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--runtime-manifest", type=Path)
    identity.add_argument("--runtime-lock-sha256")
    parser.add_argument("--bootstrap")
    args = parser.parse_args()
    metadata = json.loads(args.image_inspect.read_text())
    if isinstance(metadata, list):
        if len(metadata) != 1:
            parser.error("Inspect exactly one foundation image")
        metadata = metadata[0]
    digest = (
        args.runtime_lock_sha256
        or hashlib.sha256(args.runtime_manifest.read_bytes()).hexdigest()
    )
    print(json.dumps(install(metadata, digest, args.bootstrap), sort_keys=True))


if __name__ == "__main__":
    main()
