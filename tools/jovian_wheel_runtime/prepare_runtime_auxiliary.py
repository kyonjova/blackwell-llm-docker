"""Package the cuMem transport source from the exact LMCache wheel revision."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from container_channel import api

SOURCE_FILES = {
    "cumem_shareable_interposer.c": "csrc/cumem_ipc_interposer/cumem_shareable_interposer.c",
    "Makefile": "csrc/cumem_ipc_interposer/Makefile",
    "LICENSE": "LICENSE",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(manifest: dict, output: Path, cache: Path) -> dict:
    source = manifest["components"]["lmcache"]["source"]["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError(
            "cuMem interposer requires the immutable LMCache wheel source revision"
        )
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / source
    if not cached.exists():
        with tempfile.TemporaryDirectory(prefix="cumem-source-", dir=cache) as tmp:
            staged = Path(tmp)
            records = {}
            for name, path in SOURCE_FILES.items():
                entry = api(
                    f"repos/local-inference-lab/LMCache/contents/{path}?ref={source}"
                )
                if (
                    entry.get("type") != "file"
                    or entry.get("encoding") != "base64"
                    or entry.get("path") != path
                ):
                    raise ValueError(f"Unexpected LMCache source response: {path}")
                content = base64.b64decode(entry["content"])
                blob = hashlib.sha1(
                    b"blob " + str(len(content)).encode() + b"\0" + content
                ).hexdigest()
                if blob != entry["sha"]:
                    raise ValueError(f"LMCache git blob checksum mismatch: {path}")
                (staged / name).write_bytes(content)
                records[name] = {
                    "path": path,
                    "git_blob": blob,
                    "sha256": sha256(staged / name),
                }
            (staged / "source.json").write_text(
                json.dumps(
                    {
                        "repository": "local-inference-lab/LMCache",
                        "commit": source,
                        "files": records,
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            )
            os.rename(staged, cached)
    if cached.is_symlink():
        raise ValueError("cuMem source cache must not be a symlink")
    record = json.loads((cached / "source.json").read_text())
    if record["commit"] != source or set(record["files"]) != set(SOURCE_FILES):
        raise ValueError("cuMem source cache identity mismatch")
    for name, expected in record["files"].items():
        if (cached / name).is_symlink() or sha256(cached / name) != expected["sha256"]:
            raise ValueError(f"cuMem source cache checksum mismatch: {name}")
    destination = output / "auxiliary/lmcache-cumem"
    shutil.copytree(cached, destination)
    return record
