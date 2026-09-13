#!/usr/bin/env python3
"""Assemble one verified Qwen3.8 serving runtime from component bundles."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_SCHEMAS = {
    "foundation": "local-inference-jovian-foundation-bundle/v1",
    "nccl": "local-inference-nccl-cu134-release/v1",
    "flashinfer": "local-inference-flashinfer-wheel-release/v1",
    "b12x": "local-inference-b12x-wheel-release/v1",
    "vllm": "local-inference-vllm-wheel-release/v2",
    "lmcache": "local-inference-lmcache-wheel-release/v1",
    "instanttensor": "local-inference-instanttensor-wheel-release/v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(bundle: Path) -> None:
    checksum_file = bundle / "SHA256SUMS"
    if not checksum_file.is_file():
        raise ValueError(f"component bundle has no SHA256SUMS: {bundle}")
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
            raise ValueError(f"invalid SHA256SUMS entry in {bundle}: {line!r}")
        candidate = bundle.joinpath(*path.parts)
        if not candidate.is_file() or sha256(candidate) != expected:
            raise ValueError(f"checksum mismatch: {candidate}")


def wheel_metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(matches) != 1:
            raise ValueError(f"wheel must contain one METADATA file: {path}")
        message = email.parser.BytesParser().parsebytes(archive.read(matches[0]))
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise ValueError(f"wheel has incomplete package metadata: {path}")
    return name, version


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_value(manifest: dict[str, object], key: str) -> str | None:
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        return None
    value = runtime.get(key)
    return value if isinstance(value, str) else None


def validate_compatibility(manifests: dict[str, dict[str, object]]) -> None:
    foundation = manifests["foundation"]
    foundation_runtime = foundation.get("runtime")
    if not isinstance(foundation_runtime, dict):
        raise ValueError("foundation manifest has no runtime object")
    if foundation_runtime.get("python") != "3.12" or foundation_runtime.get("cuda") != "13.4.1":
        raise ValueError("foundation runtime must use Python 3.12 and CUDA 13.4.1")
    packages = foundation.get("packages")
    if not isinstance(packages, list):
        raise ValueError("foundation manifest has no package list")
    torch_versions = {
        package.get("version")
        for package in packages
        if isinstance(package, dict) and normalized_name(str(package.get("name"))) == "torch"
    }
    if len(torch_versions) != 1:
        raise ValueError("foundation manifest must identify exactly one PyTorch version")
    torch_version = next(iter(torch_versions))
    source = foundation.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("image"), str):
        raise ValueError("foundation manifest must identify its immutable source image")
    builder_image = source["image"]

    for role, manifest in manifests.items():
        if role in {"foundation", "nccl"}:
            continue
        python = runtime_value(manifest, "python")
        cuda = runtime_value(manifest, "cuda")
        pytorch = runtime_value(manifest, "pytorch")
        component_builder = runtime_value(manifest, "builder_image")
        if python != "3.12" or not cuda or not cuda.startswith("13.4"):
            raise ValueError(f"{role} does not declare the Python 3.12/CUDA 13.4 ABI")
        if pytorch != torch_version:
            raise ValueError(f"{role} was built against a different PyTorch distribution")
        if component_builder != builder_image:
            raise ValueError(f"{role} was built from a different foundation image")
    nccl_cuda = runtime_value(manifests["nccl"], "cuda")
    if not nccl_cuda or not nccl_cuda.startswith("13.4"):
        raise ValueError("NCCL component does not declare the CUDA 13.4 ABI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qwen-lock", required=True, type=Path)
    for role in EXPECTED_SCHEMAS:
        parser.add_argument(f"--{role}-bundle", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"output path already exists: {output}")
    qwen_lock = args.qwen_lock.resolve()
    if not qwen_lock.is_file():
        raise ValueError(f"Qwen dependency lock does not exist: {qwen_lock}")

    bundles = {
        role: getattr(args, f"{role}_bundle").resolve() for role in EXPECTED_SCHEMAS
    }
    manifests: dict[str, dict[str, object]] = {}
    for role, bundle in bundles.items():
        verify_checksums(bundle)
        manifest = json.loads((bundle / "manifest.json").read_text())
        if manifest.get("schema") != EXPECTED_SCHEMAS[role]:
            raise ValueError(f"unexpected {role} manifest schema: {manifest.get('schema')}")
        manifests[role] = manifest
    validate_compatibility(manifests)

    wheel_dir = output / "wheels"
    wheel_dir.mkdir(parents=True)
    packages: list[dict[str, str]] = []
    seen_packages: set[str] = set()
    seen_files: set[str] = set()
    for role, bundle in bundles.items():
        wheels = sorted((bundle / "wheels").glob("*.whl"))
        if not wheels:
            raise ValueError(f"component bundle contains no wheels: {bundle}")
        for wheel in wheels:
            name, version = wheel_metadata(wheel)
            package_key = normalized_name(name)
            if package_key in seen_packages or wheel.name in seen_files:
                raise ValueError(f"duplicate package or wheel filename: {wheel}")
            seen_packages.add(package_key)
            seen_files.add(wheel.name)
            destination = wheel_dir / wheel.name
            shutil.copyfile(wheel, destination)
            packages.append(
                {
                    "component": role,
                    "name": name,
                    "version": version,
                    "file": wheel.name,
                    "sha256": sha256(destination),
                }
            )

    packages.sort(key=lambda item: normalized_name(item["name"]))
    requirements = output / "requirements-local.txt"
    requirements.write_text(
        "".join(
            f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}\n"
            for item in packages
        )
    )
    foundation_bundle = bundles["foundation"]
    shutil.copyfile(foundation_bundle / "foundation-runtime.lock", output / "foundation-runtime.lock")
    shutil.copyfile(foundation_bundle / "foundation.lock", output / "foundation.lock")
    shutil.copyfile(qwen_lock, output / "qwen38-runtime.lock")
    tool_dir = Path(__file__).resolve().parent
    for name in ("install_qwen38_runtime.sh", "verify_qwen38_runtime.py"):
        shutil.copyfile(tool_dir / name, output / name)
        (output / name).chmod(0o755)

    component_records: dict[str, object] = {}
    for role, manifest in manifests.items():
        component_records[role] = {
            "manifest_sha256": sha256(bundles[role] / "manifest.json"),
            "source": manifest.get("source"),
        }
    complete_manifest = {
        "schema": "local-inference-qwen38-cu134-runtime/v1",
        "status": "research-only",
        "purpose": "Qwen3.8 Flash Next serving with vLLM on SM120 GPUs",
        "runtime": {"python": "3.12", "cuda": "13.4.1"},
        "components": component_records,
        "packages": packages,
    }
    (output / "manifest.json").write_text(
        json.dumps(complete_manifest, indent=2, sort_keys=True) + "\n"
    )
    checksum_paths = sorted(path for path in output.rglob("*") if path.is_file())
    with (output / "SHA256SUMS").open("w") as stream:
        for path in checksum_paths:
            stream.write(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n")
    print(output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"runtime bundle assembly failed: {error}", file=sys.stderr)
        sys.exit(1)
