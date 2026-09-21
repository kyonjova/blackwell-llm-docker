#!/usr/bin/env python3
"""Resolve immutable model identities before enabling persistent checkpoints.

Repository names are resolved once to a Hugging Face commit; the caller must
pass the returned revision to the model loader. Local checkpoints are identified
by their weight and configuration bytes, not by their directory names. Local
model files must remain unchanged for the lifetime of the serving process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


_IDENTITY_CACHE_FORMAT = "local-checkpoint-identity/v1"


def _identity_cache_path(directory: Path) -> Path | None:
    """Use the persistent runtime cache when one is available and writable."""
    configured = os.environ.get("LIL_CHECKPOINT_IDENTITY_CACHE_DIR")
    root = Path(configured) if configured else Path("/cache/checkpoint-identities")
    if root.resolve() == directory:
        return None
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    key = hashlib.sha256(os.fsencode(directory)).hexdigest()
    return root / f"{key}.json"


def _cached_digests(
    cache_path: Path | None, directory: Path, metadata: list[list[object]]
) -> list[str] | None:
    if cache_path is None:
        return None
    try:
        cached = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        return None
    if (
        not isinstance(cached, dict)
        or cached.get("format") != _IDENTITY_CACHE_FORMAT
        or cached.get("directory") != str(directory)
        or cached.get("metadata") != metadata
    ):
        return None
    digests = cached.get("digests")
    if not isinstance(digests, list) or len(digests) != len(metadata):
        return None
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in digests
    ):
        return None
    return digests


def _write_cached_digests(
    cache_path: Path | None,
    directory: Path,
    metadata: list[list[object]],
    digests: list[str],
) -> None:
    if cache_path is None:
        return
    payload = {
        "format": _IDENTITY_CACHE_FORMAT,
        "directory": str(directory),
        "metadata": metadata,
        "digests": digests,
    }
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=cache_path.parent, prefix="identity-", delete=False
        ) as stream:
            temporary_path = stream.name
            json.dump(payload, stream, separators=(",", ":"))
        os.replace(temporary_path, cache_path)
        temporary_path = None
    except OSError:
        # Cache persistence is an optimization; identity verification still
        # succeeds by hashing the checkpoint on the next process start.
        pass
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def local_checkpoint_identity(directory: Path) -> str:
    """Identify local safetensors weights and model/tokenizer configuration.

    JSON, Jinja and tokenizer assets are included conservatively. File digests
    are reused only when the selected file list and metadata still match a
    previously verified checkpoint. Reading does not modify the checkpoint. A
    checkpoint that changes during this operation is rejected; callers must
    also keep it immutable during model loading and serving. Missing weights
    or unsupported weight formats are errors.
    """
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"Checkpoint must be a directory: {directory}")

    def files() -> list[Path]:
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and (
                path.suffix in {".safetensors", ".json", ".jinja", ".model"}
                or path.name in {"vocab.txt", "merges.txt", "added_tokens.txt"}
            )
        )

    selected = files()
    if not any(path.suffix == ".safetensors" for path in selected):
        raise ValueError("Persistent checkpoint identity requires safetensors weights")
    if directory / "config.json" not in selected:
        raise ValueError("Persistent checkpoint identity requires config.json")
    if any(directory.glob("*.bin")) or any(directory.glob("*.gguf")):
        raise ValueError("Ambiguous checkpoint: non-safetensors weights are present")

    def content_metadata(stat: os.stat_result) -> tuple[int, ...]:
        # Reading may update atime without changing the model. Inode, size,
        # mtime and ctime detect replacement or writes without rejecting reads.
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    before = {path: path.stat() for path in selected}
    metadata = [[path.name, list(content_metadata(before[path]))] for path in selected]
    cache_path = _identity_cache_path(directory)
    digests = _cached_digests(cache_path, directory, metadata)
    cache_hit = digests is not None
    if digests is None:
        digests = []
        for path in selected:
            with path.open("rb") as stream:
                digests.append(hashlib.file_digest(stream, "sha256").hexdigest())
    if files() != selected or any(
        content_metadata(path.stat()) != content_metadata(before[path])
        for path in selected
    ):
        raise ValueError("Checkpoint files changed while their identity was computed")
    if not cache_hit:
        _write_cached_digests(cache_path, directory, metadata, digests)
    manifest = [
        (path.name, before[path].st_size, digest)
        for path, digest in zip(selected, digests)
    ]
    encoded = json.dumps(manifest, separators=(",", ":")).encode()
    return hashlib.sha256(b"glm53-local-checkpoint-v1\0" + encoded).hexdigest()


def resolve_checkpoint(model: str, revision: str | None = None) -> dict[str, str]:
    """Return a content identity and, for a Hub repository, a pinned revision.

    Local files are hashed directly, including symlink targets. For a Hub model,
    config.json is resolved through the installed Hugging Face cache client so
    its standard authentication, offline and cache settings remain effective.
    The returned revision must be used for every weight/configuration download.
    """
    path = Path(model).expanduser()
    if path.exists():
        return {"identity": local_checkpoint_identity(path), "revision": ""}
    if path.is_absolute() or model.startswith(("./", "../", "~")):
        raise ValueError(f"Local checkpoint directory does not exist: {model}")

    from huggingface_hub import hf_hub_download

    config_path = Path(
        hf_hub_download(model, "config.json", revision=revision or "main")
    )
    # The Hub cache client returns snapshots/<commit>/config.json. Do not
    # resolve its symlink: the target blob path no longer contains the commit.
    commit = config_path.parent.name
    if (
        config_path.parent.parent.name != "snapshots"
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise ValueError("Hugging Face did not return an immutable snapshot path")
    return {"identity": commit, "revision": commit}


def resolve_serving_identity(
    model: str,
    revision: str | None,
    draft_model: str | None,
    draft_revision: str | None,
    speculation: str,
    source_lock: Path,
) -> dict[str, object]:
    """Identify weights and a build-verified serving source lock.

    ``speculation`` is none, mtp, or dflash. MTP shares target checkpoint weights;
    DFlash requires a separate checkpoint. The source lock must describe the
    image's installed code and native artifacts; source-code mounts are outside
    this launcher's immutable-image contract.
    """
    if speculation not in {"none", "mtp", "dflash"}:
        raise ValueError("Speculation must be none, mtp, or dflash")
    if speculation == "dflash" and not draft_model:
        raise ValueError("DFlash requires a draft model")
    lock_bytes = source_lock.read_bytes()
    if not lock_bytes.startswith(b"format=local-inference-source-lock/v1\n"):
        raise ValueError("Unrecognized serving source-lock format")
    target = resolve_checkpoint(model, revision)
    draft = {"identity": "", "revision": ""}
    if speculation == "mtp":
        draft = target
    elif speculation == "dflash":
        assert draft_model is not None
        draft = resolve_checkpoint(draft_model, draft_revision)
    return {
        "checkpoint_identity": {
            "target_revision": target["identity"],
            "draft_revision": draft["identity"],
            "source_revision": hashlib.sha256(lock_bytes).hexdigest(),
        },
        "model_revision": target["revision"],
        "draft_model_revision": draft["revision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--draft-model")
    parser.add_argument("--draft-revision")
    parser.add_argument(
        "--speculation", choices=("none", "mtp", "dflash"), required=True
    )
    parser.add_argument(
        "--source-lock", type=Path, default=Path("/opt/glm53-flash/source.lock")
    )
    args = parser.parse_args()
    result = resolve_serving_identity(
        args.model,
        args.revision,
        args.draft_model,
        args.draft_revision,
        args.speculation,
        args.source_lock,
    )
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
