from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.compose_vllm_release import CompositionError, compose, verify_review_tree

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _commit(repo: Path, name: str, contents: str, message: str) -> str:
    (repo / name).write_text(contents)
    _git(repo, "add", name)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _repositories(tmp_path: Path) -> tuple[Path, Path, str]:
    source = tmp_path / "source"
    remote = tmp_path / "remote.git"
    source.mkdir()
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "Test")
    _git(source, "config", "user.email", "test@example.com")
    base = _commit(source, "model.py", "base\n", "base")
    _git(source, "branch", "-M", "dev/gilded-gnosis")
    _git(tmp_path, "init", "--quiet", "--bare", str(remote))
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "--quiet", "origin", "dev/gilded-gnosis")
    return source, remote, base


def _create_pr(
    source: Path,
    remote: Path,
    number: int,
    name: str,
    contents: str,
) -> str:
    _git(source, "checkout", "--quiet", "-B", f"pr-{number}", "dev/gilded-gnosis")
    head = _commit(source, name, contents, f"PR {number}")
    _git(source, "push", "--quiet", str(remote), f"HEAD:refs/pull/{number}/head")
    _git(source, "checkout", "--quiet", "dev/gilded-gnosis")
    return head


def _manifest(
    path: Path,
    remote: Path,
    prs: list[tuple[int, str]],
    source_patches: list[dict[str, str]] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test-release",
                "repository": str(remote),
                "base_ref": "refs/heads/dev/gilded-gnosis",
                "pull_requests": [
                    {"number": number, "head": head} for number, head in prs
                ],
                "source_patches": source_patches or [],
            }
        )
    )
    return path


def test_composition_resolves_branch_head_and_replays_patch(tmp_path: Path) -> None:
    source, remote, initial_base = _repositories(tmp_path)
    pr_head = _create_pr(source, remote, 1, "feature.py", "feature\n")

    advanced_base = _commit(
        source,
        "base-update.py",
        "advanced base\n",
        "advance base",
    )
    _git(source, "push", "--quiet", "origin", "dev/gilded-gnosis")
    assert advanced_base != initial_base

    manifest = _manifest(tmp_path / "manifest.json", remote, [(1, pr_head)])
    output = tmp_path / "output"
    lock = compose(manifest, output)

    assert lock["base"]["commit"] == advanced_base
    assert lock["pull_requests"][0]["disposition"] == "merged"
    assert (output / "integration.patch").is_file()

    replay = tmp_path / "replay"
    _git(tmp_path, "clone", "--quiet", str(remote), str(replay))
    _git(replay, "checkout", "--quiet", "--detach", advanced_base)
    _git(replay, "apply", "--index", str(output / "integration.patch"))
    assert _git(replay, "write-tree") == lock["result"]["tree"]
    assert (replay / "base-update.py").read_text() == "advanced base\n"
    assert (replay / "feature.py").read_text() == "feature\n"


def test_composition_lock_is_reproducible(tmp_path: Path) -> None:
    source, remote, _ = _repositories(tmp_path)
    pr_head = _create_pr(source, remote, 1, "feature.py", "feature\n")
    manifest = _manifest(tmp_path / "manifest.json", remote, [(1, pr_head)])

    first = tmp_path / "first"
    second = tmp_path / "second"
    compose(manifest, first)
    compose(manifest, second)

    assert (first / "integration.lock.json").read_bytes() == (
        second / "integration.lock.json"
    ).read_bytes()


def test_composition_preserves_unified_diff_context_whitespace(tmp_path: Path) -> None:
    source, remote, _ = _repositories(tmp_path)
    patch = "--- a/model.py\n+++ b/model.py\n@@ -1,2 +1,3 @@\n first\n \n+last\n"
    head = _create_pr(source, remote, 1, "kernel.patch", patch)
    manifest = _manifest(tmp_path / "manifest.json", remote, [(1, head)])
    lock = compose(manifest, tmp_path / "output")
    assert lock["result"]["tree"] == _git(source, "rev-parse", f"{head}^{{tree}}")


def test_composition_rejects_source_trailing_whitespace(tmp_path: Path) -> None:
    source, remote, _ = _repositories(tmp_path)
    head = _create_pr(source, remote, 1, "feature.py", "value = 1 \n")
    manifest = _manifest(tmp_path / "manifest.json", remote, [(1, head)])
    with pytest.raises(CompositionError, match="trailing whitespace"):
        compose(manifest, tmp_path / "output")


def test_composition_supports_a_base_without_pull_requests(tmp_path: Path) -> None:
    _, remote, base = _repositories(tmp_path)
    manifest = _manifest(tmp_path / "manifest.json", remote, [])
    output = tmp_path / "output"

    lock = compose(manifest, output)

    assert lock["base"]["commit"] == base
    assert lock["pull_requests"] == []
    assert lock["result"]["tree"] == _git(remote, "rev-parse", f"{base}^{{tree}}")
    assert (output / "integration.patch").read_bytes() == b""


def _pin_tree(path: Path, base: str, tree: str) -> Path:
    manifest = json.loads(path.read_text())
    manifest.update(base_commit=base, expected_tree=tree)
    path.write_text(json.dumps(manifest))
    return path


def test_review_tree_preserves_authors_and_dirty_checkout(tmp_path: Path) -> None:
    source, remote, base = _repositories(tmp_path)
    head = _create_pr(source, remote, 1, "feature.py", "feature\n")
    manifest = _pin_tree(
        _manifest(tmp_path / "manifest.json", remote, [(1, head)]),
        base,
        _git(source, "rev-parse", f"{head}^{{tree}}"),
    )
    (source / "model.py").write_text("user edits\n")
    _git(source, "add", "model.py")
    index_before = _git(source, "write-tree")
    first = verify_review_tree(manifest, source)
    second = verify_review_tree(manifest, source)
    assert first == second
    assert first["matches_expected_tree"]
    assert _git(source, "write-tree") == index_before
    assert _git(source, "rev-parse", "HEAD") == base
    assert (source / "model.py").read_text() == "user edits\n"
    assert _git(source, "rev-parse", f"{first['result_commit']}^2") == head
    assert (
        _git(source, "show", "-s", "--format=%an <%ae>", head)
        == "Test <test@example.com>"
    )


def test_review_tree_rejects_missing_source_changes(tmp_path: Path) -> None:
    source, remote, base = _repositories(tmp_path)
    head = _create_pr(source, remote, 1, "feature.py", "feature\n")
    manifest = _pin_tree(
        _manifest(tmp_path / "manifest.json", remote, []),
        base,
        _git(source, "rev-parse", f"{head}^{{tree}}"),
    )
    with pytest.raises(CompositionError, match="differs from expected tree"):
        verify_review_tree(manifest, source)


def test_review_tree_requires_public_conflict_resolution(tmp_path: Path) -> None:
    source, remote, base = _repositories(tmp_path)
    first = _create_pr(source, remote, 1, "model.py", "first\n")
    second = _create_pr(source, remote, 2, "model.py", "second\n")
    manifest = _pin_tree(
        _manifest(tmp_path / "manifest.json", remote, [(1, first), (2, second)]),
        base,
        _git(source, "rev-parse", f"{base}^{{tree}}"),
    )
    with pytest.raises(CompositionError, match="PR #2 conflicts"):
        verify_review_tree(manifest, source)
    assert _git(source, "status", "--porcelain") == ""


def test_review_tree_requires_ordered_dependencies(tmp_path: Path) -> None:
    source, remote, base = _repositories(tmp_path)
    head = _create_pr(source, remote, 1, "feature.py", "feature\n")
    path = _pin_tree(
        _manifest(tmp_path / "manifest.json", remote, [(1, head)]),
        base,
        _git(source, "rev-parse", f"{head}^{{tree}}"),
    )
    manifest = json.loads(path.read_text())
    manifest["pull_requests"][0]["requires"] = [2]
    path.write_text(json.dumps(manifest))
    with pytest.raises(CompositionError, match="unresolved review dependencies"):
        verify_review_tree(path, source)


def test_composition_pins_base_when_branch_advances(tmp_path: Path) -> None:
    source, remote, base = _repositories(tmp_path)
    tree = _git(source, "rev-parse", f"{base}^{{tree}}")
    _commit(source, "base-update.py", "advanced\n", "advance base")
    _git(source, "push", "--quiet", "origin", "dev/gilded-gnosis")
    manifest = _pin_tree(_manifest(tmp_path / "manifest.json", remote, []), base, tree)
    lock = compose(manifest, tmp_path / "output")
    assert lock["base"]["commit"] == base
    assert lock["result"]["tree"] == tree


@pytest.mark.parametrize("requires", [None, "1", [True], [-1], [{}]])
def test_review_tree_rejects_invalid_dependencies(tmp_path: Path, requires) -> None:
    source, remote, base = _repositories(tmp_path)
    head = _create_pr(source, remote, 1, "feature.py", "feature\n")
    path = _pin_tree(
        _manifest(tmp_path / "manifest.json", remote, [(1, head)]),
        base,
        _git(source, "rev-parse", f"{head}^{{tree}}"),
    )
    manifest = json.loads(path.read_text())
    manifest["pull_requests"][0]["requires"] = requires
    path.write_text(json.dumps(manifest))
    with pytest.raises(CompositionError, match="requires must be a list"):
        verify_review_tree(path, source)


def test_composition_applies_digest_locked_source_patch(tmp_path: Path) -> None:
    source, remote, base = _repositories(tmp_path)
    (source / "model.py").write_text("patched runtime\n")
    source_patch = tmp_path / "runtime.patch"
    source_patch.write_bytes(
        subprocess.run(
            ["git", "diff", "--binary", "--full-index"],
            cwd=source,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
    )
    _git(source, "restore", "model.py")
    digest = hashlib.sha256(source_patch.read_bytes()).hexdigest()
    manifest = _manifest(
        tmp_path / "manifest.json",
        remote,
        [],
        [
            {
                "path": source_patch.name,
                "sha256": digest,
                "title": "Patch the runtime fixture",
            }
        ],
    )

    output = tmp_path / "output"
    lock = compose(manifest, output)

    assert lock["base"]["commit"] == base
    assert lock["source_patches"] == [
        {
            "path": source_patch.name,
            "sha256": digest,
            "title": "Patch the runtime fixture",
        }
    ]
    assert (
        lock["result"]["patch_sha256"]
        == hashlib.sha256(source_patch.read_bytes()).hexdigest()
    )

    replay = tmp_path / "source-patch-replay"
    _git(tmp_path, "clone", "--quiet", str(remote), str(replay))
    _git(replay, "checkout", "--quiet", "--detach", base)
    _git(replay, "apply", "--index", str(output / "integration.patch"))
    assert _git(replay, "write-tree") == lock["result"]["tree"]
    assert (replay / "model.py").read_text() == "patched runtime\n"


def test_composition_rejects_source_patch_digest_mismatch(tmp_path: Path) -> None:
    _, remote, _ = _repositories(tmp_path)
    source_patch = tmp_path / "runtime.patch"
    source_patch.write_text("not a patch\n")
    manifest = _manifest(
        tmp_path / "manifest.json",
        remote,
        [],
        [{"path": source_patch.name, "sha256": "0" * 64}],
    )

    with pytest.raises(CompositionError, match="digest mismatch"):
        compose(manifest, tmp_path / "output")


def test_composition_rejects_moved_pr_head(tmp_path: Path) -> None:
    source, remote, _ = _repositories(tmp_path)
    pr_head = _create_pr(source, remote, 1, "feature.py", "feature\n")
    manifest = _manifest(tmp_path / "manifest.json", remote, [(1, "0" * 40)])

    with pytest.raises(CompositionError, match="PR #1 moved"):
        compose(manifest, tmp_path / "output")
    assert pr_head != "0" * 40


def test_composition_identifies_conflicting_pr(tmp_path: Path) -> None:
    source, remote, _ = _repositories(tmp_path)
    first = _create_pr(source, remote, 1, "model.py", "first\n")
    second = _create_pr(source, remote, 2, "model.py", "second\n")
    manifest = _manifest(tmp_path / "manifest.json", remote, [(1, first), (2, second)])

    with pytest.raises(CompositionError, match="PR #2 conflicts"):
        compose(manifest, tmp_path / "output")


def test_cherry_pick_composition_applies_declared_pr_commits(tmp_path: Path) -> None:
    source, remote, base = _repositories(tmp_path)
    pr_head = _create_pr(source, remote, 1, "feature.py", "feature\n")
    manifest = _manifest(tmp_path / "manifest.json", remote, [(1, pr_head)])
    manifest_data = json.loads(manifest.read_text())
    manifest_data["composition_strategy"] = "cherry_pick"
    manifest.write_text(json.dumps(manifest_data))

    lock = compose(manifest, tmp_path / "output")

    assert lock["composition_strategy"] == "cherry_pick"
    assert lock["pull_requests"][0]["commits"] == [pr_head]
    assert lock["pull_requests"][0]["disposition"] == "cherry_picked"
    assert lock["result"]["tree"] != _git(remote, "rev-parse", f"{base}^{{tree}}")


def test_pinned_branch_composition_records_exact_source_tree(tmp_path: Path) -> None:
    source, remote, base = _repositories(tmp_path)
    pr_head = _create_pr(source, remote, 1, "feature.py", "feature\n")
    _git(source, "checkout", "--quiet", "-B", "release-composition", base)
    _git(source, "cherry-pick", pr_head)
    composition_commit = _git(source, "rev-parse", "HEAD")
    composition_tree = _git(source, "rev-parse", "HEAD^{tree}")
    _git(source, "push", "--quiet", "origin", "release-composition")

    manifest = _manifest(tmp_path / "manifest.json", remote, [(1, pr_head)])
    manifest_data = json.loads(manifest.read_text())
    manifest_data.update(
        {
            "composition_strategy": "pinned_branch",
            "composition_ref": "refs/heads/release-composition",
            "composition_commit": composition_commit,
        }
    )
    manifest.write_text(json.dumps(manifest_data))

    output = tmp_path / "output"
    lock = compose(manifest, output)

    assert lock["composition"] == {
        "ref": "refs/heads/release-composition",
        "commit": composition_commit,
    }
    assert lock["pull_requests"][0]["disposition"] == "recorded_in_composition"
    assert lock["result"]["tree"] == composition_tree

    replay = tmp_path / "pinned-branch-replay"
    _git(tmp_path, "clone", "--quiet", str(remote), str(replay))
    _git(replay, "checkout", "--quiet", "--detach", base)
    _git(replay, "apply", "--index", str(output / "integration.patch"))
    assert _git(replay, "write-tree") == composition_tree


@pytest.mark.parametrize(
    ("component", "lock_sha256", "tree"),
    [
        (
            "vllm",
            "9f412dc26bde604e83b288bb10172bc8026035bbcb78c31fd530c381bba388bd",
            "936ed4829ed6b6a34b9052a7a2614333ee3b2623",
        ),
        (
            "sparkinfer",
            "c08e3a90ad260d0b817e3be3971aceb04ab9af6132a5175b3b77d4f94ce2b2d9",
            "f532ec965a70b710ba45e6f751fe5d7135001108",
        ),
    ],
)
def test_r5_release_archive_matches_lock(
    component: str,
    lock_sha256: str,
    tree: str,
) -> None:
    release_dir = (
        REPOSITORY_ROOT / "patches" / "releases" / "gilded-gnosis-v20-r5" / component
    )
    lock_bytes = (release_dir / "integration.lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    patch_bytes = (release_dir / lock["result"]["patch"]).read_bytes()

    assert hashlib.sha256(lock_bytes).hexdigest() == lock_sha256
    assert hashlib.sha256(patch_bytes).hexdigest() == lock["result"]["patch_sha256"]
    assert lock["result"]["tree"] == tree
