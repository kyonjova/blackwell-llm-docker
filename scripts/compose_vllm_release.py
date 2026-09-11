#!/usr/bin/env python3
"""Compose a release tree from a branch, pinned pull requests, and source patches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CompositionError(RuntimeError):
    pass


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if check and completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CompositionError(f"{' '.join(args)} failed: {detail}")
    return completed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_sha(repository: str, ref: str) -> str:
    result = _run(["git", "ls-remote", repository, ref])
    matches = [line.split()[0] for line in result.stdout.splitlines() if line]
    if len(matches) != 1 or not SHA_RE.fullmatch(matches[0]):
        raise CompositionError(
            f"expected exactly one SHA for {repository} {ref}, got {matches}"
        )
    return matches[0]


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CompositionError(f"cannot read manifest {path}: {exc}") from exc

    if manifest.get("schema_version") != 1:
        raise CompositionError("manifest schema_version must be 1")
    for key in ("name", "repository", "base_ref", "pull_requests"):
        if key not in manifest:
            raise CompositionError(f"manifest is missing {key!r}")
    if not manifest["base_ref"].startswith("refs/heads/"):
        raise CompositionError("base_ref must be a full refs/heads/... ref")

    strategy = manifest.get("composition_strategy", "merge")
    if strategy not in {"merge", "cherry_pick", "pinned_branch"}:
        raise CompositionError(
            "composition_strategy must be 'merge', 'cherry_pick', or 'pinned_branch'"
        )
    manifest["composition_strategy"] = strategy

    for key in ("base_commit", "expected_tree"):
        value = manifest.get(key)
        if value is not None and (
            not isinstance(value, str) or not SHA_RE.fullmatch(value)
        ):
            raise CompositionError(f"{key} must be a 40-character Git object ID")

    composition_ref = manifest.get("composition_ref")
    composition_commit = manifest.get("composition_commit")
    if strategy == "pinned_branch":
        if not isinstance(composition_ref, str) or not composition_ref.startswith(
            "refs/heads/"
        ):
            raise CompositionError(
                "pinned_branch composition requires a full refs/heads/... "
                "composition_ref"
            )
        if not isinstance(composition_commit, str) or not SHA_RE.fullmatch(
            composition_commit
        ):
            raise CompositionError(
                "pinned_branch composition requires a 40-character composition_commit"
            )
    elif composition_ref is not None or composition_commit is not None:
        raise CompositionError(
            "composition_ref and composition_commit require "
            "composition_strategy='pinned_branch'"
        )

    seen: set[int] = set()
    for entry in manifest["pull_requests"]:
        number = entry.get("number")
        head = entry.get("head")
        ref = entry.get("ref", f"refs/pull/{number}/head")
        if not isinstance(number, int) or number <= 0 or number in seen:
            raise CompositionError(f"invalid or duplicate PR number: {number!r}")
        if not isinstance(head, str) or not SHA_RE.fullmatch(head):
            raise CompositionError(f"PR #{number} must pin a 40-character head SHA")
        if ref != f"refs/pull/{number}/head":
            raise CompositionError(f"PR #{number} has unexpected ref {ref!r}")
        commits = entry.get("commits", [head])
        if not isinstance(commits, list) or not commits:
            raise CompositionError(f"PR #{number} commits must be a non-empty list")
        if any(
            not isinstance(commit, str) or not SHA_RE.fullmatch(commit)
            for commit in commits
        ):
            raise CompositionError(
                f"PR #{number} commits must contain 40-character commit SHAs"
            )
        if strategy == "merge" and commits != [head]:
            raise CompositionError(
                f"PR #{number} declares commits but composition_strategy is merge"
            )
        entry["ref"] = ref
        entry["commits"] = commits
        requires = entry.get("requires", [])
        if not isinstance(requires, list) or any(
            type(dependency) is not int or dependency <= 0 for dependency in requires
        ):
            raise CompositionError(f"PR #{number} requires must be a list of PR numbers")
        if not set(requires).issubset(seen):
            raise CompositionError(f"PR #{number} has unresolved review dependencies")
        seen.add(number)

    source_patches = manifest.get("source_patches", [])
    if not isinstance(source_patches, list):
        raise CompositionError("manifest source_patches must be a list")
    for index, entry in enumerate(source_patches):
        if not isinstance(entry, dict):
            raise CompositionError(f"source patch #{index + 1} must be an object")
        path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise CompositionError(
                f"source patch #{index + 1} must use a relative path"
            )
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise CompositionError(
                f"source patch #{index + 1} must pin a SHA-256 digest"
            )
    manifest["source_patches"] = source_patches
    return manifest


def verify_review_tree(manifest_path: Path, repository: Path) -> dict[str, Any]:
    """Prove pinned review heads reproduce a tree without editing a checkout.

    The object database must contain the audited public commits. This offline
    check proves composition, not publication or PR membership; the release PR
    audit must establish those separately.
    """
    manifest = _load_manifest(manifest_path)
    if manifest["composition_strategy"] != "merge" or manifest["source_patches"]:
        raise CompositionError(
            "Review-tree verification requires merges without source patches"
        )
    if not manifest.get("base_commit") or not manifest.get("expected_tree"):
        raise CompositionError(
            "Review-tree verification requires base_commit and expected_tree"
        )
    tip = manifest["base_commit"]
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Release Composer",
            "GIT_AUTHOR_EMAIL": "release-composer@local",
            "GIT_COMMITTER_NAME": "Release Composer",
            "GIT_COMMITTER_EMAIL": "release-composer@local",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        }
    )
    resolved = []
    _run(["git", "cat-file", "-e", f"{tip}^{{commit}}"], cwd=repository)
    for entry in manifest["pull_requests"]:
        head = entry["head"]
        _run(["git", "cat-file", "-e", f"{head}^{{commit}}"], cwd=repository)
        merged = _run(
            ["git", "merge-tree", "--write-tree", tip, head],
            cwd=repository,
            check=False,
        )
        if merged.returncode:
            raise CompositionError(
                f"PR #{entry['number']} conflicts with its declared predecessors. "
                "Publish the resolution in a review head before composing.\n"
                + merged.stdout
                + merged.stderr
            )
        tree = merged.stdout.splitlines()[0]
        tip = _run(
            [
                "git",
                "commit-tree",
                tree,
                "-p",
                tip,
                "-p",
                head,
                "-m",
                f"Compose reviewed source: PR #{entry['number']}",
            ],
            cwd=repository,
            env=env,
        ).stdout.strip()
        resolved.append({"number": entry["number"], "head": head, "tree": tree})
    tree = _run(["git", "rev-parse", f"{tip}^{{tree}}"], cwd=repository).stdout.strip()
    if tree != manifest["expected_tree"]:
        detail = _run(
            ["git", "diff", "--stat", manifest["expected_tree"], tree],
            cwd=repository,
        ).stdout
        raise CompositionError(
            f"Composed tree {tree} differs from expected tree "
            f"{manifest['expected_tree']}.\n{detail}"
        )
    return {
        "base_commit": manifest["base_commit"],
        "review_units": resolved,
        "result_commit": tip,
        "result_tree": tree,
        "matches_expected_tree": True,
    }


def compose(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    manifest = _load_manifest(manifest_path)
    repository = str(manifest["repository"])
    base_ref = str(manifest["base_ref"])
    composition_strategy = str(manifest["composition_strategy"])
    composition_ref = manifest.get("composition_ref")
    composition_commit = manifest.get("composition_commit")
    base_sha = manifest.get("base_commit") or _remote_sha(repository, base_ref)

    output_dir.mkdir(parents=True, exist_ok=True)
    patch_path = output_dir / "integration.patch"
    lock_path = output_dir / "integration.lock.json"

    with tempfile.TemporaryDirectory(prefix="source-release-compose-") as temp:
        checkout = Path(temp) / "checkout"
        _run(["git", "init", "--quiet", str(checkout)])
        _run(["git", "remote", "add", "origin", repository], cwd=checkout)
        _run(["git", "fetch", "--quiet", "--no-tags", "origin", base_sha], cwd=checkout)
        _run(["git", "checkout", "--quiet", "--detach", base_sha], cwd=checkout)

        merge_env = os.environ.copy()
        merge_env.update(
            {
                "GIT_AUTHOR_NAME": "Release Composer",
                "GIT_AUTHOR_EMAIL": "release-composer@local",
                "GIT_COMMITTER_NAME": "Release Composer",
                "GIT_COMMITTER_EMAIL": "release-composer@local",
                "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
                "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
            }
        )

        resolved_prs: list[dict[str, Any]] = []
        for entry in manifest["pull_requests"]:
            number = int(entry["number"])
            ref = str(entry["ref"])
            expected_head = str(entry["head"])
            remote_head = _remote_sha(repository, ref)
            if remote_head != expected_head:
                raise CompositionError(
                    f"PR #{number} moved: manifest pins {expected_head}, "
                    f"remote is {remote_head}; review and update the manifest"
                )

            local_ref = f"refs/remotes/origin/release-pr-{number}"
            _run(
                [
                    "git",
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    "origin",
                    f"{ref}:{local_ref}",
                ],
                cwd=checkout,
            )
            fetched_head = _run(
                ["git", "rev-parse", local_ref], cwd=checkout
            ).stdout.strip()
            if fetched_head != expected_head:
                raise CompositionError(
                    f"PR #{number} fetch mismatch: expected {expected_head}, got {fetched_head}"
                )

            commits = [str(commit) for commit in entry["commits"]]
            for commit in commits:
                belongs_to_pr = (
                    _run(
                        ["git", "merge-base", "--is-ancestor", commit, expected_head],
                        cwd=checkout,
                        check=False,
                    ).returncode
                    == 0
                )
                if not belongs_to_pr:
                    raise CompositionError(
                        f"PR #{number} does not contain declared commit {commit}"
                    )

            already_in_base = all(
                _run(
                    ["git", "merge-base", "--is-ancestor", commit, base_sha],
                    cwd=checkout,
                    check=False,
                ).returncode
                == 0
                for commit in commits
            )
            if composition_strategy == "pinned_branch":
                disposition = "recorded_in_composition"
            elif already_in_base:
                disposition = "already_in_base"
            elif composition_strategy == "merge":
                merge = _run(
                    [
                        "git",
                        "merge",
                        "--no-ff",
                        "--no-edit",
                        "-m",
                        f"release composition: PR #{number}",
                        expected_head,
                    ],
                    cwd=checkout,
                    env=merge_env,
                    check=False,
                )
                if merge.returncode:
                    detail = merge.stderr.strip() or merge.stdout.strip()
                    raise CompositionError(
                        f"PR #{number} conflicts with the clean stack: {detail}"
                    )
                disposition = "merged"
            else:
                for commit in commits:
                    cherry_pick = _run(
                        ["git", "cherry-pick", commit],
                        cwd=checkout,
                        env=merge_env,
                        check=False,
                    )
                    if cherry_pick.returncode:
                        detail = (
                            cherry_pick.stderr.strip() or cherry_pick.stdout.strip()
                        )
                        raise CompositionError(
                            f"PR #{number} commit {commit} conflicts with the clean "
                            f"stack: {detail}"
                        )
                disposition = "cherry_picked"

            resolved_prs.append(
                {
                    "number": number,
                    "ref": ref,
                    "head": expected_head,
                    "commits": commits,
                    "title": entry.get("title", ""),
                    "disposition": disposition,
                }
            )

        resolved_composition: dict[str, str] | None = None
        if composition_strategy == "pinned_branch":
            assert isinstance(composition_ref, str)
            assert isinstance(composition_commit, str)
            remote_composition = _remote_sha(repository, composition_ref)
            if remote_composition != composition_commit:
                raise CompositionError(
                    f"composition branch moved: manifest pins {composition_commit}, "
                    f"remote is {remote_composition}; review and update the manifest"
                )
            local_composition_ref = "refs/remotes/origin/release-composition"
            _run(
                [
                    "git",
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    "origin",
                    f"{composition_ref}:{local_composition_ref}",
                ],
                cwd=checkout,
            )
            base_is_ancestor = (
                _run(
                    [
                        "git",
                        "merge-base",
                        "--is-ancestor",
                        base_sha,
                        composition_commit,
                    ],
                    cwd=checkout,
                    check=False,
                ).returncode
                == 0
            )
            if not base_is_ancestor:
                raise CompositionError(
                    f"composition commit {composition_commit} does not descend "
                    f"from base {base_sha}"
                )
            _run(
                ["git", "checkout", "--quiet", "--detach", composition_commit],
                cwd=checkout,
            )
            resolved_composition = {
                "ref": composition_ref,
                "commit": composition_commit,
            }

        resolved_source_patches: list[dict[str, Any]] = []
        for index, entry in enumerate(manifest["source_patches"]):
            source_patch = (manifest_path.parent / entry["path"]).resolve()
            if not source_patch.is_file():
                raise CompositionError(
                    f"source patch #{index + 1} does not exist: {entry['path']}"
                )
            actual_digest = _sha256(source_patch)
            expected_digest = str(entry["sha256"])
            if actual_digest != expected_digest:
                raise CompositionError(
                    f"source patch #{index + 1} digest mismatch: "
                    f"expected {expected_digest}, got {actual_digest}"
                )
            apply_check = _run(
                ["git", "apply", "--check", str(source_patch)],
                cwd=checkout,
                check=False,
            )
            if apply_check.returncode:
                detail = apply_check.stderr.strip() or apply_check.stdout.strip()
                raise CompositionError(
                    f"source patch #{index + 1} does not apply to the composed tree: "
                    f"{detail}"
                )
            _run(["git", "apply", "--index", str(source_patch)], cwd=checkout)
            resolved_source_patches.append(
                {
                    "path": str(entry["path"]),
                    "sha256": expected_digest,
                    "title": entry.get("title", ""),
                }
            )

        # Unified-diff blobs encode empty context lines as a literal space.
        # Preserve that syntax; whitespace checks still cover source files.
        _run(
            ["git", "diff", "--check", base_sha, "--", ".", ":(exclude)*.patch"],
            cwd=checkout,
        )
        result_tree = _run(["git", "write-tree"], cwd=checkout).stdout.strip()
        if manifest.get("expected_tree") not in (None, result_tree):
            raise CompositionError(
                f"Composed tree {result_tree} differs from expected tree "
                f"{manifest['expected_tree']}"
            )
        patch = subprocess.run(
            ["git", "diff", "--binary", "--full-index", base_sha],
            cwd=checkout,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        patch_path.write_bytes(patch)

        _run(["git", "reset", "--hard", "--quiet", base_sha], cwd=checkout)
        _run(["git", "clean", "-fdx", "--quiet"], cwd=checkout)
        if patch:
            _run(["git", "apply", "--index", str(patch_path)], cwd=checkout)
        applied_tree = _run(["git", "write-tree"], cwd=checkout).stdout.strip()
        if applied_tree != result_tree:
            raise CompositionError(
                f"patch verification produced tree {applied_tree}, expected {result_tree}"
            )

    lock = {
        "schema_version": 1,
        "name": manifest["name"],
        "manifest": {
            "sha256": _sha256(manifest_path),
        },
        "base": {
            "repository": repository,
            "ref": base_ref,
            "commit": base_sha,
        },
        "composition_strategy": composition_strategy,
        "composition": resolved_composition,
        "pull_requests": resolved_prs,
        "source_patches": resolved_source_patches,
        "result": {
            "tree": result_tree,
            "patch": patch_path.name,
            "patch_sha256": _sha256(patch_path),
        },
    }
    temporary_lock = lock_path.with_suffix(".json.tmp")
    temporary_lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    temporary_lock.replace(lock_path)
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--verify-in",
        type=Path,
        help="Verify pinned review trees in an existing Git object database without checkout edits",
    )
    args = parser.parse_args()
    try:
        if args.verify_in is not None:
            lock = verify_review_tree(args.manifest, args.verify_in)
        else:
            if args.output_dir is None:
                parser.error("--output-dir is required unless --verify-in is used")
            lock = compose(args.manifest, args.output_dir)
    except CompositionError as exc:
        parser.error(str(exc))
    print(json.dumps(lock, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
