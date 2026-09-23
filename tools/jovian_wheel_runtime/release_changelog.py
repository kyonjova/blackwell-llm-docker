#!/usr/bin/env python3
"""Compile component change fragments into one container release changelog."""

from __future__ import annotations

import base64
import json
import re
import subprocess
from collections import defaultdict
from urllib.parse import quote

from container_channel import api, asset_bytes


SCHEMA = "local-inference-release-change/v1"
OUTPUT_SCHEMA = "local-inference-container-changelog/v1"
CHANGE_PATH = ".lil/changes"
# Launcher, profiles and image recipe changes, recorded in this repository.
RECIPE_COMPONENT = "docker"
CATEGORIES = {
    "breaking",
    "compatibility",
    "feature",
    "fix",
    "internal",
    "performance",
}
CATEGORY_TITLES = {
    "breaking": "Required changes",
    "feature": "Features",
    "fix": "Correctness and reliability",
    "performance": "Performance",
    "compatibility": "Compatibility",
    "internal": "Build and maintenance",
}
CATEGORY_ORDER = tuple(CATEGORY_TITLES)
ALLOWED_FIELDS = {
    "schema",
    "id",
    "category",
    "summary",
    "models",
    "compatibility",
    "details",
    "pull_requests",
    "authors",
    "requires",
    "evidence",
}


def _unique_strings(value: object, field: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or (required and not value):
        raise ValueError(f"{field} must be a{' non-empty' if required else ''} list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} entries must be non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} entries must be unique")
    return value


def validate_fragment(fragment: object, component: str, filename: str) -> dict:
    """Validate one contributor-authored fragment without rewriting its meaning."""
    if not isinstance(fragment, dict):
        raise ValueError(f"{filename}: fragment must be a JSON object")
    unknown = set(fragment) - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"{filename}: unknown fields: {sorted(unknown)}")
    if fragment.get("schema") != SCHEMA:
        raise ValueError(f"{filename}: unsupported fragment schema")
    identity = fragment.get("id")
    if not isinstance(identity, str) or not re.fullmatch(
        rf"{re.escape(component)}-[a-z0-9]+(?:-[a-z0-9]+)*", identity
    ):
        raise ValueError(
            f"{filename}: id must start with {component}- and use lowercase slugs"
        )
    if filename != f"{identity}.json":
        raise ValueError(f"{filename}: filename must match fragment id {identity}")
    if fragment.get("category") not in CATEGORIES:
        raise ValueError(f"{filename}: invalid category")
    for field in ("summary", "compatibility"):
        value = fragment.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{filename}: {field} must be a non-empty string")
    _unique_strings(fragment.get("models"), "models", required=True)
    for field in ("details", "authors", "requires", "evidence"):
        _unique_strings(fragment.get(field, []), field)
    pull_requests = fragment.get("pull_requests", [])
    if (
        not isinstance(pull_requests, list)
        or any(not isinstance(number, int) or number <= 0 for number in pull_requests)
        or len(pull_requests) != len(set(pull_requests))
    ):
        raise ValueError(
            f"{filename}: pull_requests must contain unique positive integers"
        )
    if not pull_requests and not fragment.get("authors"):
        raise ValueError(f"{filename}: direct changes must name at least one author")
    return fragment


def _optional_api(endpoint: str) -> object | None:
    try:
        return api(endpoint)
    except subprocess.CalledProcessError as error:
        message = (error.stderr or b"").decode(errors="replace")
        if "HTTP 404" in message:
            return None
        raise


def _fragment_entries(repository: str, commit: str) -> dict[str, dict]:
    endpoint = f"repos/{repository}/contents/{CHANGE_PATH}?ref={quote(commit, safe='')}"
    listing = _optional_api(endpoint)
    if listing is None:
        return {}
    if not isinstance(listing, list):
        raise ValueError(f"{repository}@{commit}: {CHANGE_PATH} is not a directory")
    entries: dict[str, dict] = {}
    for entry in listing:
        name = entry.get("name", "")
        if entry.get("type") != "file" or not name.endswith(".json"):
            raise ValueError(
                f"{repository}@{commit}: {CHANGE_PATH} may contain only JSON files"
            )
        entries[name] = entry
    return entries


def load_fragments(repository: str, commit: str, component: str) -> dict[str, dict]:
    """Load and validate every fragment at one exact component revision."""
    result: dict[str, dict] = {}
    for name, entry in _fragment_entries(repository, commit).items():
        endpoint = (
            f"repos/{repository}/contents/{quote(entry['path'], safe='/')}"
            f"?ref={quote(commit, safe='')}"
        )
        payload = api(endpoint)
        if payload.get("encoding") != "base64":
            raise ValueError(f"{repository}@{commit}:{entry['path']}: expected base64")
        # GitHub wraps base64 content with newlines. Remove transport whitespace
        # before retaining strict alphabet/padding validation.
        encoded = "".join(payload["content"].split())
        raw = base64.b64decode(encoded, validate=True)
        fragment = validate_fragment(json.loads(raw), component, name)
        identity = fragment["id"]
        if identity in result:
            raise ValueError(f"{repository}@{commit}: duplicate fragment {identity}")
        result[identity] = {
            "fragment": fragment,
            "path": entry["path"],
            "blob_sha": entry["sha"],
        }
    return result


def _uploaded_assets(release: dict) -> dict[str, dict]:
    return {
        item["name"]: item
        for item in release.get("assets", [])
        if item.get("state") == "uploaded" and item.get("size")
    }


def previous_publication(repository: str, assembly: dict) -> dict | None:
    """Find the most recent complete publication for the same release channel."""
    releases = []
    page = 1
    while True:
        batch = api(f"repos/{repository}/releases?per_page=100&page={page}")
        releases.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    # API listing order is not publication order, including across pages.
    # GitHub's UTC published_at timestamps share the same sortable format.
    releases.sort(
        key=lambda release: (release.get("published_at") or "", release.get("id", 0)),
        reverse=True,
    )
    for release in releases:
        if (
            release.get("draft")
            or not release.get("published_at")
            or release.get("tag_name") == assembly["release_tag"]
        ):
            continue
        assets = _uploaded_assets(release)
        lock = assets.get("community-assembly.json") or assets.get(
            f"community-assembly-{assembly['release_channel']}.json"
        )
        receipt_asset = assets.get("container-release.json")
        if lock is None or receipt_asset is None:
            continue
        candidate = json.loads(asset_bytes(repository, lock))
        receipt = json.loads(asset_bytes(repository, receipt_asset))
        if (
            candidate.get("release_channel") == assembly["release_channel"]
            and receipt.get("status") == "qualified"
            and receipt.get("assembly_sha256") == candidate.get("assembly_sha256")
        ):
            return {
                "tag": release["tag_name"],
                "url": release["html_url"],
                "image": candidate["image"],
                "digest": receipt.get("digest"),
                "assembly": candidate,
            }
    return None


def _pull_request(repository: str, number: int) -> dict:
    pull = api(f"repos/{repository}/pulls/{number}")
    if pull.get("number") != number:
        raise ValueError(f"{repository}#{number}: pull request identity mismatch")
    return {
        "number": number,
        "url": pull["html_url"],
        "title": pull["title"],
        "author": "@" + pull["user"]["login"],
    }


def _fragment_commit(source: dict) -> str:
    """Read fragments at the observed branch tip when the wheel was reused.

    The resolver reuses a wheel when later commits change no source paths,
    for example commits that only add change fragments.
    """
    return source.get("observed_branch_commit") or source["source_commit"]


def collect_release_changelog(assembly: dict, publication_repository: str) -> dict:
    """Compile immutable fragments added since the previous channel release."""
    previous = previous_publication(publication_repository, assembly)
    sources = dict(assembly["components"])
    previous_sources = dict((previous or {}).get("assembly", {}).get("components", {}))
    # The recipe ships in the image too; its fragments live in this repository.
    if assembly.get("recipe_commit"):
        sources[RECIPE_COMPONENT] = {
            "repository": publication_repository,
            "source_commit": assembly["recipe_commit"],
        }
        previous_recipe = (previous or {}).get("assembly", {}).get("recipe_commit")
        if previous_recipe:
            previous_sources[RECIPE_COMPONENT] = {
                "repository": publication_repository,
                "source_commit": previous_recipe,
            }

    required = set(assembly.get("changelog", {}).get("required_components", []))
    unknown_required = required - set(sources)
    if unknown_required:
        raise ValueError(
            f"changelog policy names unknown components: {sorted(unknown_required)}"
        )

    current_by_component: dict[str, dict[str, dict]] = {}
    added: list[tuple[str, str, dict]] = []
    component_ranges = {}
    for component, current in sources.items():
        repository = current["repository"]
        current_commit = _fragment_commit(current)
        current_fragments = load_fragments(repository, current_commit, component)
        current_by_component[component] = current_fragments
        previous_component = previous_sources.get(component) if previous else None
        previous_commit = None
        previous_fragments: dict[str, dict] = {}
        if previous_component and previous_component.get("repository") == repository:
            previous_commit = _fragment_commit(previous_component)
            previous_fragments = load_fragments(repository, previous_commit, component)
        deleted = set(previous_fragments) - set(current_fragments)
        modified = {
            identity
            for identity in set(previous_fragments) & set(current_fragments)
            if previous_fragments[identity]["blob_sha"]
            != current_fragments[identity]["blob_sha"]
        }
        if deleted or modified:
            raise ValueError(
                f"{component}: published change fragments are immutable; "
                f"deleted={sorted(deleted)}, modified={sorted(modified)}"
            )
        identities = sorted(set(current_fragments) - set(previous_fragments))
        if (
            previous is not None
            and component in required
            and previous_commit != current_commit
            and not identities
        ):
            raise ValueError(
                f"{component}: source changed without a new {CHANGE_PATH} fragment"
            )
        for identity in identities:
            added.append((component, repository, current_fragments[identity]))
        component_ranges[component] = {
            "repository": repository,
            "previous_commit": previous_commit,
            "current_commit": current_commit,
            "change_count": len(identities),
        }

    known_ids = {
        identity
        for fragments in current_by_component.values()
        for identity in fragments
    }
    changes = []
    for component, repository, record in added:
        fragment = record["fragment"]
        missing = set(fragment.get("requires", [])) - known_ids
        if missing:
            raise ValueError(
                f"{fragment['id']}: unknown required fragments: {sorted(missing)}"
            )
        changes.append(
            {
                **fragment,
                "component": component,
                "repository": repository,
                "fragment_path": record["path"],
                "fragment_blob_sha": record["blob_sha"],
                "pull_requests": [
                    _pull_request(repository, number)
                    for number in fragment.get("pull_requests", [])
                ],
            }
        )
    order = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    changes.sort(key=lambda item: (order[item["category"]], item["id"]))
    return {
        "schema": OUTPUT_SCHEMA,
        "assembly_sha256": assembly["assembly_sha256"],
        "channel": assembly["channel"],
        "release_channel": assembly["release_channel"],
        "image": assembly["image"],
        "previous_release": (
            {
                key: previous[key]
                for key in ("tag", "url", "image", "digest")
                if previous.get(key) is not None
            }
            if previous
            else None
        ),
        "components": component_ranges,
        "changes": changes,
    }


def render_release_notes(changelog: dict) -> str:
    """Render the human-facing portion of a GitHub container release."""
    previous = changelog.get("previous_release")
    heading = (
        f"Changes since [`{previous['image']}`]({previous['url']})"
        if previous
        else "Tracked component changes"
    )
    rows = [f"## {heading}", ""]
    if not changelog["changes"]:
        rows += ["No component changes are recorded for this assembly.", ""]
        return "\n".join(rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for change in changelog["changes"]:
        grouped[change["category"]].append(change)
    for category in CATEGORY_ORDER:
        if category not in grouped:
            continue
        rows += [f"### {CATEGORY_TITLES[category]}", ""]
        for change in grouped[category]:
            links = [
                f"[{change['component']} #{pull['number']}]({pull['url']})"
                for pull in change["pull_requests"]
            ]
            authors = sorted(
                set(change.get("authors", []))
                | {pull["author"] for pull in change["pull_requests"]}
            )
            suffix = []
            if links:
                suffix.append(", ".join(links))
            if authors:
                suffix.append("by " + ", ".join(authors))
            source = f" ({'; '.join(suffix)})" if suffix else ""
            rows.append(f"- **{change['summary']}**{source}")
            rows.append(f"  - Models: {', '.join(change['models'])}")
            if change["compatibility"] != "No user action required.":
                rows.append(f"  - Compatibility: {change['compatibility']}")
            rows.extend(f"  - {detail}" for detail in change.get("details", []))
            rows.extend(
                f"  - [Validation evidence]({url})"
                for url in change.get("evidence", [])
            )
        rows.append("")
    return "\n".join(rows)
