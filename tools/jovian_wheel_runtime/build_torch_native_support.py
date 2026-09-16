#!/usr/bin/env python3
"""Build the native loader-support wheel for NVIDIA PyTorch 26.08.

The NVIDIA PyTorch wheel links to HPC-X and MKL libraries that are installed
outside Python's package tree in the NGC image.  NCCL-based vLLM serving does
not invoke MPI or UCC collectives, but the ELF loader must still resolve their
sonames during ``import torch``.  This wheel places the hash-locked closure in
``torch/lib``, where the existing PyTorch ``$ORIGIN`` runpath can resolve it.

The package qualifies loading and NCCL-based inference only.  It does not
claim a complete or supported HPC-X installation for MPI or UCC workloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_tree_mtime(root: Path, epoch: int) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if not path.is_symlink():
            os.utime(path, (epoch, epoch))
    os.utime(root, (epoch, epoch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    package = contract["package"]
    normalized_name = package["name"].replace("-", "_")
    dist_info_name = f"{normalized_name}-{package['version']}.dist-info"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    provenance: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="torch-native-support-") as temporary:
        stage = Path(temporary) / "payload"
        library_dir = stage / "torch" / "lib"
        license_dir = stage / dist_info_name / "licenses"
        library_dir.mkdir(parents=True)
        license_dir.mkdir(parents=True)

        for library in contract["libraries"]:
            source = Path(library["source"]).resolve()
            source_digest = sha256(source)
            if source_digest != library["sha256"]:
                raise RuntimeError(
                    f"source digest mismatch for {library['soname']}: "
                    f"{source_digest}"
                )
            destination = library_dir / library["soname"]
            shutil.copyfile(source, destination)
            shutil.copymode(source, destination)
            subprocess.run(
                ["patchelf", "--set-rpath", "$ORIGIN", str(destination)],
                check=True,
            )
            provenance.append(
                {
                    "soname": library["soname"],
                    "source": library["source"],
                    "source_sha256": source_digest,
                    "packaged_sha256": sha256(destination),
                }
            )

        for license_file in contract["licenses"]:
            shutil.copyfile(
                Path(license_file["source"]), license_dir / license_file["name"]
            )

        metadata = stage / dist_info_name / "METADATA"
        metadata.write_text(
            "Metadata-Version: 2.3\n"
            f"Name: {package['name']}\n"
            f"Version: {package['version']}\n"
            "Summary: Native loader closure for the NVIDIA PyTorch 26.08 wheel\n"
            "Requires-Python: >=3.12,<3.13\n"
            f"Requires-Dist: torch == {package['torch_version']}\n"
        )
        (stage / dist_info_name / "WHEEL").write_text(
            "Wheel-Version: 1.0\n"
            "Generator: local-inference-torch-native-support\n"
            "Root-Is-Purelib: false\n"
            "Tag: cp312-cp312-linux_x86_64\n"
        )
        normalize_tree_mtime(stage, args.source_date_epoch)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                "--dest-dir",
                str(args.output_dir),
                str(stage),
            ],
            check=True,
            env={**os.environ, "SOURCE_DATE_EPOCH": str(args.source_date_epoch)},
            stdout=sys.stderr,
        )

    wheels = list(args.output_dir.glob(f"{normalized_name}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one native-support wheel, found {len(wheels)}")
    result = {
        "schema": "local-inference-torch-native-support-build/v1",
        "status": "implemented",
        "contract": contract,
        "libraries": provenance,
        "wheel": wheels[0].name,
        "wheel_sha256": sha256(wheels[0]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
