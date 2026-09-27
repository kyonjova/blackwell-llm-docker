"""Record the SHA-256 of every code file in the finished runtime image.

Runs as the last step of the image build, so the build's own dependency
patches are part of the reference. ``lil-bench`` compares a running
container against this file, and the release publishes the same file (its
hash is in container-release.json), so users' results show when the vLLM,
b12x or other code in their container was modified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata as metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOTS = (
    "/opt/venv/lib/python3.12/site-packages",
    "/usr/local/lib/python3.12/dist-packages/torch",
    "/usr/local/lib/python3.12/dist-packages/triton",
    "/opt/lil",
    "/usr/local/bin/lil-serve",
    "/usr/local/bin/lil-entrypoint",
    "/opt/venv/bin/lil-bench",
)
OUTPUT = Path("/opt/venv/share/lil-runtime/runtime-files.json.gz")


def skip(path: str) -> bool:
    # Bytecode caches are rebuilt by the interpreter; lil-bench ignores them too.
    return "/__pycache__/" in path or path.endswith((".pyc", ".pyo"))


def file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scan(roots) -> dict[str, str]:
    files = {}
    for root in roots:
        if os.path.isfile(root) and not os.path.islink(root):
            files[root] = file_hash(root)
            continue
        for directory, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in names:
                path = os.path.join(directory, name)
                if not os.path.islink(path) and not skip(path):
                    files[path] = file_hash(path)
    return files


def build(roots=ROOTS) -> dict:
    present = [root for root in roots if os.path.exists(root)]
    site = next((r for r in present if r.endswith("site-packages")), None)
    packages = {}
    if site:
        for dist in metadata.distributions(path=[site]):
            name = dist.metadata["Name"]
            if name:
                packages[name.lower()] = dist.version
    return {
        "schema": "lil-runtime-files/1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "roots": present,
        "packages": dict(sorted(packages.items())),
        "files": dict(sorted(scan(present).items())),
    }


def write(document: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(document, separators=(",", ":")).encode()
    # mtime=0 keeps the archive byte-identical for identical content.
    output.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    document = build()
    write(document, args.output)
    print(
        f"runtime file manifest: {len(document['files'])} files, {args.output.stat().st_size} bytes"
    )


if __name__ == "__main__":
    main()
