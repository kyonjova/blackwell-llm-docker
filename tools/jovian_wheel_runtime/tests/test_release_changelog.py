from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_changelog as changelog


def fragment(identity="vllm-816", *, pull_requests=None):
    return {
        "schema": changelog.SCHEMA,
        "id": identity,
        "category": "fix",
        "summary": "Merge QSA selection across DCP ranks",
        "models": ["Qwen3.8-Flash-Next"],
        "compatibility": "No user action required.",
        "details": ["DCP1, DCP2, and DCP4 use the same selection contract."],
        "pull_requests": [816] if pull_requests is None else pull_requests,
        "requires": ["b12x-402"],
    }


def record(payload, blob="new-blob"):
    return {
        "fragment": payload,
        "path": f".lil/changes/{payload['id']}.json",
        "blob_sha": blob,
    }


def assembly():
    return {
        "assembly_sha256": "a" * 64,
        "channel": "karmic-kraken",
        "release_channel": "karmic-kraken-beta",
        "release_tag": "karmic-kraken-beta-example",
        "image": "ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-example",
        "changelog": {"required_components": ["vllm", "b12x"]},
        "components": {
            "vllm": {
                "repository": "local-inference-lab/vllm",
                "source_commit": "vllm-new",
            },
            "b12x": {
                "repository": "local-inference-lab/b12x",
                "source_commit": "b12x-new",
            },
        },
    }


def previous():
    return {
        "tag": "karmic-kraken-beta-prior",
        "url": "https://github.com/local-inference-lab/blackwell-llm-docker/releases/tag/prior",
        "image": "ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-prior",
        "digest": "ghcr.io/local-inference-lab/vllm@sha256:" + "b" * 64,
        "assembly": {
            "components": {
                "vllm": {
                    "repository": "local-inference-lab/vllm",
                    "source_commit": "vllm-old",
                },
                "b12x": {
                    "repository": "local-inference-lab/b12x",
                    "source_commit": "b12x-old",
                },
            }
        },
    }


def publication(tag, published_at, *, channel="karmic-kraken-beta"):
    candidate = {
        "release_channel": channel,
        "assembly_sha256": tag,
        "image": f"ghcr.io/example/runtime:{tag}",
    }
    receipt = {"status": "qualified", "assembly_sha256": tag}
    return {
        "tag_name": tag,
        "published_at": published_at,
        "created_at": "2026-09-22T01:00:00Z",
        "html_url": f"https://github.com/example/runtime/releases/tag/{tag}",
        "assets": [
            {"name": name, "state": "uploaded", "size": 1, "payload": payload}
            for name, payload in (
                ("community-assembly.json", candidate),
                ("container-release.json", receipt),
            )
        ],
    }


def install_release_pages(monkeypatch, pages):
    requests = []

    def fake_api(endpoint):
        requests.append(endpoint)
        page = int(endpoint.rsplit("page=", 1)[1])
        return pages[page - 1]

    monkeypatch.setattr(changelog, "api", fake_api)
    monkeypatch.setattr(
        changelog, "asset_bytes",
        lambda repository, asset: json.dumps(asset["payload"]).encode(),
    )
    return requests


def test_previous_publication_uses_publication_time_not_api_order(monkeypatch):
    entries = [
        publication("published-at-two", "2026-09-22T02:00:00Z"),
        publication("published-at-five", "2026-09-22T05:00:00Z"),
        publication("published-at-three", "2026-09-22T03:00:00Z"),
    ]
    install_release_pages(monkeypatch, [entries])
    result = changelog.previous_publication("example/runtime", assembly())
    assert result["tag"] == "published-at-five"


def test_previous_publication_checks_later_pages_before_selecting(monkeypatch):
    first_page = [publication("published-at-two", "2026-09-22T02:00:00Z")]
    first_page.extend({"draft": True} for _ in range(99))
    second_page = [publication("published-at-five", "2026-09-22T05:00:00Z")]
    requests = install_release_pages(monkeypatch, [first_page, second_page])
    result = changelog.previous_publication("example/runtime", assembly())
    assert result["tag"] == "published-at-five"
    assert len(requests) == 2


@pytest.mark.parametrize("unavailable", [
    "draft", "unpublished", "assembly", "incomplete", "unqualified", "channel",
])
def test_previous_publication_skips_ineligible_publications(monkeypatch, unavailable):
    candidate = publication("ineligible", "2026-09-22T06:00:00Z")
    if unavailable == "draft":
        candidate["draft"] = True
    elif unavailable == "unpublished":
        candidate["published_at"] = None
    elif unavailable == "assembly":
        candidate["tag_name"] = assembly()["release_tag"]
    elif unavailable == "incomplete":
        candidate["assets"][1]["state"] = "new"
    elif unavailable == "unqualified":
        candidate["assets"][1]["payload"]["status"] = "failed"
    elif unavailable == "channel":
        candidate["assets"][0]["payload"]["release_channel"] = "beta"
    install_release_pages(monkeypatch, [[
        candidate, publication("eligible", "2026-09-22T05:00:00Z"),
    ]])
    result = changelog.previous_publication("example/runtime", assembly())
    assert result["tag"] == "eligible"


def test_fragment_requires_human_identity_for_direct_changes():
    payload = fragment(pull_requests=[])
    with pytest.raises(ValueError, match="name at least one author"):
        changelog.validate_fragment(payload, "vllm", "vllm-816.json")
    payload["authors"] = ["@maintainer"]
    assert changelog.validate_fragment(payload, "vllm", "vllm-816.json") == payload


def test_loads_github_wrapped_base64_content(monkeypatch):
    payload = fragment()
    encoded = base64.encodebytes(json.dumps(payload).encode()).decode()

    def fake_api(endpoint):
        if "/contents/.lil/changes?" in endpoint:
            return [
                {
                    "name": "vllm-816.json",
                    "path": ".lil/changes/vllm-816.json",
                    "type": "file",
                    "sha": "blob-sha",
                }
            ]
        return {"encoding": "base64", "content": encoded}

    monkeypatch.setattr(changelog, "api", fake_api)
    assert changelog.load_fragments("example/vllm", "commit", "vllm") == {
        "vllm-816": record(payload, "blob-sha")
    }


def test_collects_only_fragments_added_since_previous_release(monkeypatch):
    vllm_change = record(fragment())
    b12x_payload = fragment("b12x-402", pull_requests=[402])
    b12x_payload["requires"] = []
    b12x_change = record(b12x_payload)
    historical = record(
        {
            **fragment("vllm-700", pull_requests=[700]),
            "requires": [],
        },
        "historical-blob",
    )
    by_commit = {
        "vllm-old": {"vllm-700": historical},
        "vllm-new": {"vllm-700": historical, "vllm-816": vllm_change},
        "b12x-old": {},
        "b12x-new": {"b12x-402": b12x_change},
    }
    monkeypatch.setattr(changelog, "previous_publication", lambda *args: previous())
    monkeypatch.setattr(
        changelog,
        "load_fragments",
        lambda repository, commit, component: by_commit[commit],
    )
    monkeypatch.setattr(
        changelog,
        "_pull_request",
        lambda repository, number: {
            "number": number,
            "url": f"https://github.com/{repository}/pull/{number}",
            "title": "Reviewed change",
            "author": "@contributor",
        },
    )

    result = changelog.collect_release_changelog(assembly(), "example/releases")

    assert [change["id"] for change in result["changes"]] == [
        "b12x-402",
        "vllm-816",
    ]
    assert result["previous_release"]["tag"] == "karmic-kraken-beta-prior"
    assert result["components"]["vllm"]["change_count"] == 1
    notes = changelog.render_release_notes(result)
    assert "Merge QSA selection across DCP ranks" in notes
    assert "[vllm #816]" in notes
    assert "vllm-new" not in notes


def test_rejects_changed_source_without_fragment(monkeypatch):
    monkeypatch.setattr(changelog, "previous_publication", lambda *args: previous())
    monkeypatch.setattr(changelog, "load_fragments", lambda *args: {})
    with pytest.raises(ValueError, match="source changed without"):
        changelog.collect_release_changelog(assembly(), "example/releases")


@pytest.mark.parametrize("mutation", ["delete", "modify"])
def test_published_fragments_are_immutable(monkeypatch, mutation):
    payload = fragment()
    old = record(payload, "old-blob")
    current = {} if mutation == "delete" else {payload["id"]: record(payload)}
    by_commit = {
        "vllm-old": {payload["id"]: old},
        "vllm-new": current,
        "b12x-old": {},
        "b12x-new": {},
    }
    monkeypatch.setattr(changelog, "previous_publication", lambda *args: previous())
    monkeypatch.setattr(
        changelog,
        "load_fragments",
        lambda repository, commit, component: by_commit[commit],
    )
    with pytest.raises(ValueError, match="fragments are immutable"):
        changelog.collect_release_changelog(assembly(), "example/releases")
