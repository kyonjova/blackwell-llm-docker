"""Keep the cuMem transport source bound to its LMCache wheel revision."""

import base64
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_runtime_auxiliary as auxiliary


def manifest(commit="a" * 40):
    return {"components": {"lmcache": {"source": {"commit": commit}}}}


def test_source_download_is_cached_and_bound_to_commit(tmp_path, monkeypatch):
    requests = []

    def api(url):
        requests.append(url)
        path = url.split("/contents/", 1)[1].split("?", 1)[0]
        content = path.encode()
        return {
            "type": "file",
            "encoding": "base64",
            "path": path,
            "content": base64.b64encode(content).decode(),
            "sha": hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest(),
        }

    monkeypatch.setattr(auxiliary, "api", api)
    record = auxiliary.prepare(manifest(), tmp_path / "first", tmp_path / "cache")
    assert len(requests) == 3
    assert all(url.endswith("?ref=" + "a" * 40) for url in requests)
    assert (
        auxiliary.prepare(manifest(), tmp_path / "second", tmp_path / "cache") == record
    )
    assert len(requests) == 3
    (tmp_path / "cache" / ("a" * 40) / "Makefile").write_text("different source")
    with pytest.raises(ValueError, match="checksum mismatch"):
        auxiliary.prepare(manifest(), tmp_path / "third", tmp_path / "cache")


def test_moving_revision_cannot_select_auxiliary_source(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        auxiliary.prepare(
            manifest("integration/local-inference-lab"),
            tmp_path / "out",
            tmp_path / "cache",
        )
