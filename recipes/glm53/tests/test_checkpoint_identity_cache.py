"""Local checkpoint identity caching preserves content-sensitive namespaces."""

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


HELPER = Path(__file__).parents[1] / "glm53_checkpoint_identity.py"
SPEC = importlib.util.spec_from_file_location("glm53_checkpoint_identity", HELPER)
assert SPEC is not None and SPEC.loader is not None
identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(identity)


def checkpoint(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"test"}')
    (model / "model.safetensors").write_bytes(b"first weights")
    return model


def test_unchanged_checkpoint_reuses_verified_file_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = checkpoint(tmp_path)
    monkeypatch.setenv("LIL_CHECKPOINT_IDENTITY_CACHE_DIR", str(tmp_path / "cache"))
    expected = identity.local_checkpoint_identity(model)
    manifest = [
        (path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(model.iterdir())
    ]
    legacy = hashlib.sha256(
        b"glm53-local-checkpoint-v1\0"
        + json.dumps(manifest, separators=(",", ":")).encode()
    ).hexdigest()
    assert expected == legacy

    def reject_read(*_args, **_kwargs):
        raise AssertionError("Unchanged checkpoint contents were read again")

    monkeypatch.setattr(identity.hashlib, "file_digest", reject_read)
    assert identity.local_checkpoint_identity(model) == expected


def test_changed_checkpoint_invalidates_cached_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = checkpoint(tmp_path)
    monkeypatch.setenv("LIL_CHECKPOINT_IDENTITY_CACHE_DIR", str(tmp_path / "cache"))
    original = identity.local_checkpoint_identity(model)
    (model / "model.safetensors").write_bytes(b"different model weights")
    assert identity.local_checkpoint_identity(model) != original


def test_same_size_weight_replacement_invalidates_cached_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = checkpoint(tmp_path)
    monkeypatch.setenv("LIL_CHECKPOINT_IDENTITY_CACHE_DIR", str(tmp_path / "cache"))
    original = identity.local_checkpoint_identity(model)
    replacement = model / "replacement.safetensors"
    replacement.write_bytes(b"other weights")
    replacement.replace(model / "model.safetensors")
    assert identity.local_checkpoint_identity(model) != original


def test_added_configuration_invalidates_cached_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = checkpoint(tmp_path)
    monkeypatch.setenv("LIL_CHECKPOINT_IDENTITY_CACHE_DIR", str(tmp_path / "cache"))
    original = identity.local_checkpoint_identity(model)
    (model / "tokenizer_config.json").write_text('{"eos_token":"</s>"}')
    assert identity.local_checkpoint_identity(model) != original


def test_invalid_cache_entry_requires_content_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = checkpoint(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setenv("LIL_CHECKPOINT_IDENTITY_CACHE_DIR", str(cache))
    expected = identity.local_checkpoint_identity(model)
    entry = next(cache.glob("*.json"))
    entry.write_text(json.dumps({"format": "invalid"}))
    assert identity.local_checkpoint_identity(model) == expected
