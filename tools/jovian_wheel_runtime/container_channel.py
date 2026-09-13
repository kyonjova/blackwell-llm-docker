#!/usr/bin/env python3
"""Resolve complete wheel releases into reproducible container build inputs.

Only configured repositories and branches are trusted. Events merely request a
rescan; their payloads never supply executable commands, URLs or source refs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from assemble_qwen38_runtime_bundle import (
    EXPECTED_SCHEMAS,
    ngc_foundation_manifest,
    validate_compatibility,
)


class PendingBuild(RuntimeError):
    """A required source revision has no complete published wheel yet."""


def run(args: list[str]) -> bytes:
    return subprocess.run(args, check=True, capture_output=True).stdout


def api(endpoint: str) -> object:
    return json.loads(run(["gh", "api", endpoint]))


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def release_assets(release: dict) -> tuple[dict, dict, dict]:
    """Select exactly one manifest and source archive with its checksum."""
    assets = release["assets"]
    manifests = [a for a in assets if a["name"] == "manifest.json"]
    archives = [a for a in assets if a["name"].endswith(".tar.zst")]
    if len(manifests) != 1 or len(archives) != 1:
        raise PendingBuild("release does not contain one manifest and source archive")
    checksums = [a for a in assets if a["name"] == archives[0]["name"] + ".sha256"]
    if len(checksums) != 1:
        raise PendingBuild("release archive checksum is not published")
    for item in [manifests[0], archives[0], checksums[0]]:
        if item.get("state") != "uploaded" or not item.get("size"):
            raise PendingBuild("release asset upload is incomplete")
        if PurePosixPath(item["name"]).name != item["name"]:
            raise ValueError("release contains an unsafe asset name")
    return manifests[0], archives[0], checksums[0]


def asset_bytes(repository: str, asset: dict) -> bytes:
    return run(
        [
            "gh",
            "api",
            f"repos/{repository}/releases/assets/{int(asset['id'])}",
            "-H",
            "Accept: application/octet-stream",
        ]
    )


def has_source_changes(comparison: dict, paths: list[str]) -> bool:
    # GitHub truncates comparison file lists at 300; incomplete evidence fails closed.
    files = comparison.get("files", [])
    if len(files) >= 300:
        return True
    return any(
        name == prefix or (prefix.endswith("/") and name.startswith(prefix))
        for file in files
        for name in (file["filename"], file.get("previous_filename", ""))
        for prefix in paths
    )


def select_component(role: str, config: dict) -> tuple[dict, dict]:
    repository = config["repository"]
    branch = api(f"repos/{repository}/commits/{quote(config['branch'], safe='')}")
    tip = branch["sha"]
    releases = api(f"repos/{repository}/releases?per_page=100")
    prefix = config["release_prefix"]
    candidates = [
        r
        for r in releases
        if not r["draft"]
        and re.fullmatch(re.escape(prefix) + r"[0-9a-f]{40}", r["tag_name"])
    ]
    # A delayed upload for an older source must never roll the channel backwards.
    candidates.sort(key=lambda r: r["created_at"], reverse=True)
    for release in candidates:
        commit = release["tag_name"][len(prefix) :]
        seed = config.get("bootstrap", {})
        seeded = tip == seed.get("branch_commit") and commit == seed.get(
            "release_commit"
        )
        comparison = {"status": "identical", "files": []}
        if commit != tip and not seeded:
            comparison = api(f"repos/{repository}/compare/{commit}...{tip}")
            if comparison["status"] not in {"ahead", "identical"}:
                continue
            if has_source_changes(comparison, config["source_paths"]):
                continue
        try:
            manifest_asset, archive, checksum = release_assets(release)
        except PendingBuild as error:
            raise PendingBuild(f"{role}: {error}") from error
        manifest_bytes = asset_bytes(repository, manifest_asset)
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != EXPECTED_SCHEMAS[role]:
            raise ValueError(f"{role}: unexpected release schema")
        if manifest.get("source", {}).get("commit") != commit:
            raise ValueError(f"{role}: release source identity mismatch")
        if role == "lmcache":
            receipt = manifest.get("community_source", {})
            expected = set(map(str, config["required_reviews"]))
            if (
                not expected <= set(receipt.get("required_reviews", {}))
                or receipt.get("source_commit") != commit
                or not receipt.get("verified_python_modules")
            ):
                raise ValueError(
                    "LMCache release lacks the complete community source proof"
                )
        checksum_data = asset_bytes(repository, checksum).decode().strip().split()
        if (
            len(checksum_data) != 2
            or not re.fullmatch(r"[0-9a-f]{64}", checksum_data[0])
            or checksum_data[1].lstrip("*") != archive["name"]
        ):
            raise ValueError(f"{role}: invalid archive checksum")
        return {
            "repository": repository,
            "branch": config["branch"],
            "release": release["tag_name"],
            "source_commit": commit,
            "observed_branch_commit": tip,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "archive": archive["name"],
            "archive_asset_id": archive["id"],
            "archive_sha256": checksum_data[0],
            "release_url": release["html_url"],
        }, manifest
    raise PendingBuild(
        f"{role}: no complete wheel for branch {config['branch']} at {tip}"
    )


def assembly_identity(channel: dict, components: dict, recipe_commit: str) -> str:
    """Bind the image to all component receipts and the complete build recipe."""
    return hashlib.sha256(
        canonical(
            {
                "channel": channel,
                "components": components,
                "recipe_commit": recipe_commit,
            }
        )
    ).hexdigest()


def resolve(config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config.get("schema") != "local-inference-container-channel/v1":
        raise ValueError("unsupported channel schema")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", config["channel"]):
        raise ValueError("invalid channel name")
    if not re.fullmatch(
        r"ghcr.io/local-inference-lab/[a-z0-9-]+", config["image_repository"]
    ):
        raise ValueError("image repository is outside the configured organization")
    components, manifests = {}, {}
    for role, component in config["components"].items():
        if not re.fullmatch(
            r"local-inference-lab/[A-Za-z0-9_-]+", component["repository"]
        ):
            raise ValueError(
                "component repository is outside the configured organization"
            )
        components[role], manifests[role] = select_component(role, component)
    manifests["foundation"] = ngc_foundation_manifest(
        config_path.parent / "foundation.lock"
    )
    validate_compatibility(manifests)
    root = config_path.resolve().parents[2]
    commit = run(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    identity = assembly_identity(config, components, commit)
    date = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    result = {
        "schema": "local-inference-container-assembly/v1",
        "channel": config["channel"],
        "recipe_commit": commit,
        "assembly_sha256": identity,
        "components": components,
        "image": f"{config['image_repository']}:{config['channel']}-beta-{date}-{identity[:16]}",
        "alias": f"{config['image_repository']}:{config['channel']}-beta",
        "release_tag": f"{config['channel']}-beta-{identity}",
        "status": "research-only",
    }
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def safe_extract(tar_path: Path, destination: Path) -> None:
    """Reject links, special files and traversal before unpacking trusted assets."""
    with tarfile.open(tar_path) as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe component archive member: {member.name}")
        archive.extractall(destination, members=members, filter="data")


def download(assembly: dict, destination: Path) -> None:
    """Fetch the exact locked asset IDs and verify bytes before extraction."""
    destination.mkdir(parents=True, exist_ok=False)
    for role, component in assembly["components"].items():
        with tempfile.TemporaryDirectory(prefix="component-download-") as temporary:
            compressed = Path(temporary) / component["archive"]
            with compressed.open("wb") as stream:
                subprocess.run(
                    [
                        "gh",
                        "api",
                        (
                            f"repos/{component['repository']}/releases/assets/"
                            f"{int(component['archive_asset_id'])}"
                        ),
                        "-H",
                        "Accept: application/octet-stream",
                    ],
                    stdout=stream,
                    check=True,
                )
            if digest(compressed) != component["archive_sha256"]:
                raise ValueError(f"{role}: locked archive checksum mismatch")
            tar_path = Path(temporary) / "component.tar"
            with tar_path.open("wb") as stream:
                subprocess.run(
                    ["zstd", "-dc", str(compressed)], stdout=stream, check=True
                )
            target = destination / role
            target.mkdir()
            safe_extract(tar_path, target)
            if digest(target / "manifest.json") != component["manifest_sha256"]:
                raise ValueError(f"{role}: archive and release manifests disagree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolver = commands.add_parser("resolve")
    resolver.add_argument("--config", type=Path, required=True)
    resolver.add_argument("--output", type=Path, required=True)
    downloader = commands.add_parser("download")
    downloader.add_argument("--assembly", type=Path, required=True)
    downloader.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        download(json.loads(args.assembly.read_text()), args.output)
        return
    try:
        assembly = resolve(args.config, args.output)
    except PendingBuild as error:
        print(f"Waiting for component build: {error}")
        if "GITHUB_OUTPUT" in os.environ:
            with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
                stream.write("ready=false\n")
        return
    print(f"Resolved {assembly['image']}")
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write("ready=true\n")
            stream.writelines(
                f"{key}={assembly[key]}\n" for key in ("image", "alias", "release_tag")
            )


if __name__ == "__main__":
    main()
