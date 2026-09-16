"""Build the selected LMCache cuMem helper without recompiling its Python wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    record = manifest["auxiliary"]["lmcache_cumem"]
    if record["commit"] != manifest["components"]["lmcache"]["source"]["commit"]:
        raise ValueError("cuMem source must match the installed LMCache wheel revision")
    source = args.bundle / "auxiliary/lmcache-cumem"
    for name, expected in record["files"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((source / name).read_bytes()).hexdigest()
            != expected["sha256"]
        ):
            raise ValueError("cuMem build source checksum mismatch")
    output = Path("/opt/lmcache/lib/liblmcache_cumem_shareable.so")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "make",
            "-C",
            str(source),
            "-j2",
            "CUDA_HOME=/usr/local/cuda",
            f"TARGET={output}",
        ],
        check=True,
    )
    symbols = subprocess.check_output(["nm", "-D", str(output)], text=True)
    if not any(line.split()[-1:] == ["cuMemCreate"] for line in symbols.splitlines()):
        raise ValueError("cuMem helper must export cuMemCreate")
    license_path = Path("/opt/venv/share/lil-runtime/lmcache-cumem-LICENSE")
    shutil.copyfile(source / "LICENSE", license_path)
    receipt = {
        **record,
        "library": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (license_path.parent / "lmcache-cumem.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
