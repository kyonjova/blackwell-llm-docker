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
        changelog,
        "asset_bytes",
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


@pytest.mark.parametrize(
    "unavailable",
    [
        "draft",
        "unpublished",
        "assembly",
        "incomplete",
        "unqualified",
        "channel",
    ],
)
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
    install_release_pages(
        monkeypatch,
        [
            [
                candidate,
                publication("eligible", "2026-09-22T05:00:00Z"),
            ]
        ],
    )
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


def test_collects_recipe_fragments_since_the_previous_recipe(monkeypatch):
    """Launcher and profile changes in this repository reach the release notes."""
    recipe_payload = fragment("docker-71", pull_requests=[71])
    recipe_payload["requires"] = []
    vllm_payload = {**fragment(), "requires": []}
    by_source = {
        ("local-inference-lab/vllm", "vllm-old"): {},
        ("local-inference-lab/vllm", "vllm-new"): {"vllm-816": record(vllm_payload)},
        ("local-inference-lab/b12x", "b12x-old"): {},
        ("local-inference-lab/b12x", "b12x-new"): {
            "b12x-402": record(
                fragment("b12x-402", pull_requests=[402]) | {"requires": []}
            )
        },
        ("example/releases", "recipe-old"): {},
        ("example/releases", "recipe-new"): {"docker-71": record(recipe_payload)},
    }
    prior = previous()
    prior["assembly"]["recipe_commit"] = "recipe-old"
    current = assembly()
    current["recipe_commit"] = "recipe-new"
    monkeypatch.setattr(changelog, "previous_publication", lambda *args: prior)
    monkeypatch.setattr(
        changelog,
        "load_fragments",
        lambda repository, commit, component: by_source[(repository, commit)],
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

    result = changelog.collect_release_changelog(current, "example/releases")

    assert "docker-71" in [change["id"] for change in result["changes"]]
    assert result["components"]["docker"] == {
        "repository": "example/releases",
        "previous_commit": "recipe-old",
        "current_commit": "recipe-new",
        "change_count": 1,
    }
    assert "[docker #71](https://github.com/example/releases/pull/71)" in (
        changelog.render_release_notes(result)
    )


def test_reads_fragments_at_the_observed_branch_tip(monkeypatch):
    """A reused wheel still publishes fragments added after its source commit."""
    payload = {**fragment("lmcache-83", pull_requests=[83]), "requires": []}
    current = assembly()
    current["changelog"] = {"required_components": []}
    current["components"] = {
        "lmcache": {
            "repository": "local-inference-lab/LMCache",
            "source_commit": "lmcache-wheel",
            "observed_branch_commit": "lmcache-tip",
        }
    }
    prior = previous()
    prior["assembly"]["components"] = {
        "lmcache": {
            "repository": "local-inference-lab/LMCache",
            "source_commit": "lmcache-old",
        }
    }
    by_commit = {"lmcache-old": {}, "lmcache-tip": {"lmcache-83": record(payload)}}
    monkeypatch.setattr(changelog, "previous_publication", lambda *args: prior)
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

    result = changelog.collect_release_changelog(current, "example/releases")

    assert [change["id"] for change in result["changes"]] == ["lmcache-83"]


def test_docker_is_a_known_required_component_only_with_a_recipe(monkeypatch):
    monkeypatch.setattr(changelog, "previous_publication", lambda *args: None)
    monkeypatch.setattr(changelog, "load_fragments", lambda *args: {})
    current = assembly()
    current["changelog"] = {"required_components": ["docker"]}
    with pytest.raises(ValueError, match="unknown components"):
        changelog.collect_release_changelog(current, "example/releases")
    current["recipe_commit"] = "recipe-new"
    changelog.collect_release_changelog(current, "example/releases")


def _reviewed(monkeypatch):
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


def _canonical_sources(recipe_old, recipe_new, recipe_fragments, vllm_fragments):
    return {
        ("local-inference-lab/vllm", "vllm-old"): {},
        ("local-inference-lab/vllm", "vllm-new"): vllm_fragments,
        ("local-inference-lab/b12x", "b12x-old"): {},
        ("local-inference-lab/b12x", "b12x-new"): {},
        ("example/releases", recipe_old): {},
        ("example/releases", recipe_new): recipe_fragments,
    }


def test_fragment_waits_for_a_required_fragment_of_another_component(monkeypatch):
    """A shared recipe change that needs a beta-only vLLM fix must not fail
    the canonical publication; it waits until the channel contains the fix."""
    recipe_payload = {
        **fragment("docker-79", pull_requests=[79]),
        "requires": ["vllm-895"],
    }
    other_payload = {**fragment("docker-80", pull_requests=[80]), "requires": []}
    recipe = {
        "docker-79": record(recipe_payload),
        "docker-80": record(other_payload),
    }
    sources = _canonical_sources("recipe-old", "recipe-new", recipe, {})
    prior = previous()
    prior["assembly"]["recipe_commit"] = "recipe-old"
    current = assembly()
    current["recipe_commit"] = "recipe-new"
    current["changelog"] = {"required_components": []}
    monkeypatch.setattr(changelog, "previous_publication", lambda *args: prior)
    monkeypatch.setattr(
        changelog,
        "load_fragments",
        lambda repository, commit, component: sources[(repository, commit)],
    )
    _reviewed(monkeypatch)

    result = changelog.collect_release_changelog(current, "example/releases")

    assert [change["id"] for change in result["changes"]] == ["docker-80"]
    assert result["deferred"] == [
        {"id": "docker-79", "component": "docker", "missing": ["vllm-895"]}
    ]

    # The next release of the channel carries vllm-895, so docker-79 appears.
    later_sources = {
        ("local-inference-lab/vllm", "vllm-old"): {},
        ("local-inference-lab/vllm", "vllm-new"): {
            "vllm-895": record(
                {**fragment("vllm-895", pull_requests=[895]), "requires": []}
            )
        },
        ("local-inference-lab/b12x", "b12x-old"): {},
        ("local-inference-lab/b12x", "b12x-new"): {},
        ("example/releases", "recipe-new"): recipe,
    }
    later_prior = previous()
    later_prior["assembly"]["recipe_commit"] = "recipe-new"
    later_prior["deferred"] = ["docker-79"]
    monkeypatch.setattr(changelog, "previous_publication", lambda *args: later_prior)
    monkeypatch.setattr(
        changelog,
        "load_fragments",
        lambda repository, commit, component: later_sources[(repository, commit)],
    )

    later = changelog.collect_release_changelog(current, "example/releases")

    assert [change["id"] for change in later["changes"]] == ["docker-79", "vllm-895"]
    assert later["deferred"] == []


def test_previous_publication_reads_deferred_fragments(monkeypatch):
    release = publication("karmic-kraken-beta-prior", "2026-09-24T22:00:00Z")
    release["assets"].append(
        {
            "name": "release-changelog.json",
            "state": "uploaded",
            "size": 1,
            "payload": {"deferred": [{"id": "docker-79", "missing": ["vllm-895"]}]},
        }
    )
    install_release_pages(monkeypatch, [[release]])
    current = assembly()
    current["release_tag"] = "karmic-kraken-beta-next"
    found = changelog.previous_publication("example/releases", current)
    assert found["deferred"] == ["docker-79"]
