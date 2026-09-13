#!/usr/bin/env python3
"""Install a source-locked Qwen3.8 runtime over its immutable NGC foundation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath


SCHEMA = "local-inference-qwen38-cu134-runtime/v1"
FOUNDATION_COMPONENT = "foundation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(bundle: Path) -> None:
    checksum_file = bundle / "SHA256SUMS"
    if not checksum_file.is_file():
        raise ValueError("runtime bundle has no SHA256SUMS")
    for line in checksum_file.read_text().splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator:
            expected, separator, relative = line.partition(" *")
        path = PurePosixPath(relative)
        if (
            not separator
            or len(expected) != 64
            or path.is_absolute()
            or ".." in path.parts
        ):
            raise ValueError(f"invalid SHA256SUMS entry: {line!r}")
        candidate = bundle.joinpath(*path.parts)
        if not candidate.is_file() or sha256(candidate) != expected:
            raise ValueError(f"runtime bundle checksum mismatch: {candidate}")


def normalized_name(name: str) -> str:
    return name.lower().replace("_", "-")


def require_ngc_foundation(manifest: dict[str, object], source_image: str) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported Qwen3.8 runtime manifest")
    components = manifest.get("components")
    if not isinstance(components, dict):
        raise ValueError("runtime manifest has no component records")
    foundation = components.get(FOUNDATION_COMPONENT)
    if not isinstance(foundation, dict):
        raise ValueError("runtime manifest has no foundation component")
    source = foundation.get("source")
    if not isinstance(source, dict) or source.get("image") != source_image:
        raise ValueError("runtime bundle was built for a different NGC image")

    packages = manifest.get("packages")
    if not isinstance(packages, list):
        raise ValueError("runtime manifest has no package records")
    expected_foundation: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, dict) or package.get("component") != FOUNDATION_COMPONENT:
            continue
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise ValueError("foundation package record is incomplete")
        expected_foundation[normalized_name(name)] = version
    required = {"torch", "torchvision", "triton", "triton-kernels", "flash-attn"}
    if not required.issubset(expected_foundation):
        raise ValueError("runtime manifest has an incomplete NGC package contract")
    for name in sorted(required):
        actual = importlib.metadata.version(name)
        if actual != expected_foundation[name]:
            raise ValueError(
                f"NGC package {name} version {actual!r} does not match {expected_foundation[name]!r}"
            )


def write_application_requirements(
    manifest: dict[str, object], output: Path
) -> None:
    packages = manifest["packages"]
    assert isinstance(packages, list)
    lines: list[str] = []
    for package in packages:
        assert isinstance(package, dict)
        if package.get("component") == FOUNDATION_COMPONENT:
            continue
        name = package.get("name")
        version = package.get("version")
        digest = package.get("sha256")
        if not all(isinstance(value, str) for value in (name, version, digest)):
            raise ValueError("application package record is incomplete")
        lines.append(f"{name}=={version} --hash=sha256:{digest}\n")
    if not lines:
        raise ValueError("runtime manifest has no application packages")
    output.write_text("".join(lines))


def run(command: list[str], **kwargs: object) -> None:
    subprocess.run(command, check=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--venv", required=True, type=Path)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--entrypoint", required=True, type=Path)
    parser.add_argument("--verifier", required=True, type=Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    venv = args.venv
    if venv.exists():
        raise ValueError(f"runtime destination already exists: {venv}")
    verify_checksums(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    require_ngc_foundation(manifest, args.source_image)

    environment = os.environ.copy()
    environment["UV_NO_CONFIG"] = "1"
    environment["UV_LINK_MODE"] = "copy"
    with tempfile.TemporaryDirectory(prefix="qwen38-ngc-runtime-") as temporary:
        requirements = Path(temporary) / "requirements-application.txt"
        write_application_requirements(manifest, requirements)
        run(
            [
                str(args.uv),
                "venv",
                "--python",
                sys.executable,
                "--system-site-packages",
                str(venv),
            ],
            env=environment,
        )
        run(
            [
                str(args.uv),
                "pip",
                "install",
                "--python",
                str(venv / "bin/python"),
                "--require-hashes",
                "--no-deps",
                "-r",
                str(bundle / "qwen38-runtime.lock"),
            ],
            env=environment,
        )
        run(
            [
                str(args.uv),
                "pip",
                "install",
                "--python",
                str(venv / "bin/python"),
                "--no-index",
                "--find-links",
                str(bundle / "wheels"),
                "--no-deps",
                "--require-hashes",
                "-r",
                str(requirements),
            ],
            env=environment,
        )
    destination = venv / "bin/qwen38-ngc-runtime"
    shutil.copyfile(args.entrypoint, destination)
    destination.chmod(0o755)
    verifier = venv / "libexec/verify_qwen38_runtime.py"
    verifier.parent.mkdir()
    shutil.copyfile(args.verifier, verifier)
    verifier.chmod(0o755)
    print(f"qwen38_ngc_runtime={venv.resolve()} status=installed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        print(f"NGC runtime installation failed: {error}", file=sys.stderr)
        sys.exit(1)
