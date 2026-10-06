#!/usr/bin/env python3
"""Build and publish the linux/arm64 (DGX Spark) twin of a published release.

The amd64 publisher (publish_container_channel.py) releases an image from
component wheels built on the amd64 runners. This tool takes such a release,
rebuilds every component from the same source commits for linux/arm64
(LIL_WHEEL_PLATFORM=linux/arm64 in each component's bundle script), builds the
runtime image from the release's recipe commit with LIL_RUNTIME_PLATFORM=
linux/arm64, pushes it as <repository>:<image tag with -spark after the
channel>, and attaches container-release-linux-arm64.json to the release.

A component commit without linux/arm64 support in its bundle script fails the
build; the arm64 image never mixes amd64 wheels in. vLLM reuses the native
objects of an earlier arm64 build when csrc/, cmake/ and the other native inputs
are unchanged (VLLM_PRECOMPILED_BUNDLE), as its amd64 release does, and every
component reuses its verified arm64 bundle when its source commit was built
before (LIL_SPARK_CACHE).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PLATFORM = "linux/arm64"
RECEIPT = "container-release-linux-arm64.json"
BUNDLE_SCRIPTS = {
    "vllm": "tools/jovian_wheel_release/build_bundle.sh",
    "b12x": "ci/lil_wheels/build_bundle.sh",
    "flashinfer": "ci/lil_wheels/build_bundle.sh",
    "lmcache": "ci/lil_wheels/build_bundle.sh",
    "instanttensor": "ci/lil_wheels/build_bundle.sh",
    "nccl": "ci/lil_wheels/build_bundle.sh",
}
# Paths whose content decides whether vLLM's native objects can be reused.
VLLM_NATIVE_INPUTS = (
    "csrc",
    "cmake",
    "rust",
    "requirements",
    "CMakeLists.txt",
    "setup.py",
    "pyproject.toml",
    "vllm/vllm_flash_attn",
    "vllm/third_party",
    "tools/jovian_wheel_release/linux-arm64",
    "tools/jovian_wheel_release/normalize_wheel.py",
)


def run(argv: list[str], **kwargs) -> str:
    return subprocess.run(argv, check=True, capture_output=True, text=True, **kwargs).stdout


def execute(argv: list[str], **kwargs) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True, **kwargs)


def spark_image(image: str) -> str:
    """ghcr.io/x/vllm:karmic-kraken-beta-20261006-c8c8 -> ...:karmic-kraken-beta-spark-20261006-c8c8."""
    repository, tag = image.rsplit(":", 1)
    channel, date, identity = tag.rsplit("-", 2)
    if not (date.isdigit() and len(date) == 8 and len(identity) == 16):
        raise ValueError(f"Unexpected release image tag: {tag}")
    return f"{repository}:{channel}-spark-{date}-{identity}"


def spark_alias(alias: str) -> str:
    repository, tag = alias.rsplit(":", 1)
    return f"{repository}:{tag}-spark"


def release_asset(repository: str, tag: str, name: str, directory: Path) -> Path:
    execute(
        ["gh", "release", "download", tag, "--repo", repository, "--pattern", name,
         "--dir", str(directory), "--clobber"]
    )
    return directory / name


def checkout(repository: str, commit: str, destination: Path) -> None:
    execute(["git", "init", "--quiet", str(destination)])
    execute(["git", "-C", str(destination), "fetch", "--quiet", "--filter=blob:none",
             f"https://github.com/{repository}.git", commit])
    execute(["git", "-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    execute(["git", "-C", str(destination), "submodule", "update", "--quiet", "--init",
             "--recursive", "--depth", "1"])


def supports_arm64(role: str, source: Path) -> bool:
    script = BUNDLE_SCRIPTS[role]
    return (source / Path(script).parent / "linux-arm64" / "runtime.lock").is_file()


def reusable_vllm_bundle(source: Path, cache: Path) -> Path | None:
    """The newest cached arm64 vLLM bundle whose native inputs match this commit."""
    for candidate in sorted(cache.glob("*/bundle"), key=lambda p: p.stat().st_mtime, reverse=True):
        native_commit = json.loads((candidate / "manifest.json").read_text())["source"]["commit"]
        if native_commit and subprocess.run(
            ["git", "-C", str(source), "cat-file", "-e", f"{native_commit}^{{commit}}"],
            capture_output=True,
        ).returncode:
            subprocess.run(["git", "-C", str(source), "fetch", "--quiet", "--filter=blob:none",
                            "origin", native_commit], capture_output=True)
        same = subprocess.run(
            ["git", "-C", str(source), "diff", "--quiet", native_commit, "HEAD", "--",
             *VLLM_NATIVE_INPUTS],
            capture_output=True,
        )
        if same.returncode == 0:
            return candidate
    return None


# Bundles kept per component; a beta usually changes one or two components.
KEPT_BUNDLES = 4


def cached_bundle(cache: Path, role: str, commit: str) -> Path | None:
    """A verified arm64 bundle built earlier from the same source commit."""
    bundle = cache / "bundles" / role / commit / "bundle"
    if not (bundle / "SHA256SUMS").is_file():
        return None
    check = subprocess.run(["sha256sum", "--check", "--quiet", "SHA256SUMS"], cwd=bundle,
                           capture_output=True)
    return bundle if check.returncode == 0 else None


def keep_bundle(cache: Path, role: str, commit: str, bundle: Path) -> None:
    directory = cache / "bundles" / role
    directory.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{commit}-", dir=directory))
    shutil.copytree(bundle, staging / "bundle")
    target = directory / commit
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    entries = sorted(
        (path for path in directory.iterdir() if not path.name.startswith(".")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in entries[KEPT_BUNDLES:]:
        shutil.rmtree(stale)


def build_component(role: str, component: dict, work: Path, cache: Path) -> Path:
    cached = cached_bundle(cache, role, component["source_commit"])
    if cached is not None:
        print(f"Reusing the arm64 {role} bundle of {component['source_commit']}", flush=True)
        return cached
    source = work / "sources" / role
    checkout(component["repository"], component["source_commit"], source)
    if not supports_arm64(role, source):
        raise SystemExit(
            f"{role} {component['source_commit']} has no linux/arm64 bundle support "
            f"({Path(BUNDLE_SCRIPTS[role]).parent}/linux-arm64/runtime.lock)"
        )
    output = work / "components" / role
    environment = dict(os.environ, LIL_WHEEL_PLATFORM=PLATFORM)
    environment["GITHUB_REPOSITORY"] = component["repository"]
    if role == "vllm":
        if subprocess.run(["git", "-C", str(source), "remote", "get-url", "origin"],
                          capture_output=True).returncode:
            execute(["git", "-C", str(source), "remote", "add", "origin",
                     f"https://github.com/{component['repository']}.git"])
        native = reusable_vllm_bundle(source, cache / "vllm-native")
        if native is not None:
            print(f"Reusing arm64 vLLM native objects from {native}", flush=True)
            environment["VLLM_PRECOMPILED_BUNDLE"] = str(native)
    execute(["bash", BUNDLE_SCRIPTS[role], str(output)], cwd=source, env=environment)
    bundle = output / "bundle"
    if role == "vllm" and "VLLM_PRECOMPILED_BUNDLE" not in environment:
        keep = cache / "vllm-native" / component["source_commit"]
        if keep.exists():
            shutil.rmtree(keep)
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle, keep / "bundle")
    keep_bundle(cache, role, component["source_commit"], bundle)
    return bundle


FAILED = "container-release-linux-arm64.failed.json"


def has_arm64_support(repository: str, commit: str, path: str) -> bool:
    found = subprocess.run(
        ["gh", "api", f"repos/{repository}/contents/{path}?ref={commit}", "--silent"],
        capture_output=True,
    )
    return found.returncode == 0


def select_release(repository: str) -> str:
    """The newest published beta release to build, or "" when there is none.

    A release is skipped when it already has an arm64 receipt or a failure
    marker (a manual run with --release-tag retries it), and when one of its
    components or its recipe predates linux/arm64 support, so that the
    five-minute publisher schedule does not start a doomed build each time.
    """
    releases = json.loads(run(["gh", "api", f"repos/{repository}/releases?per_page=30"]))
    candidates = [
        release for release in releases
        if not release["draft"]
        and release["tag_name"].startswith("karmic-kraken-beta-")
        and any(asset["name"] == "container-release.json" for asset in release["assets"])
    ]
    if not candidates:
        return ""
    release = candidates[0]
    names = {asset["name"] for asset in release["assets"]}
    if RECEIPT in names or FAILED in names:
        return ""
    tag = release["tag_name"]
    assembly = json.loads(run(["gh", "release", "download", tag, "--repo", repository,
                               "--pattern", "community-assembly.json", "--output", "-"]))
    required = [(repository, assembly["recipe_commit"],
                 "tools/jovian_wheel_runtime/linux-arm64/foundation.lock")]
    for role, component in sorted(assembly["components"].items()):
        script = Path(BUNDLE_SCRIPTS[role])
        required.append((component["repository"], component["source_commit"],
                         str(script.parent / "linux-arm64" / "runtime.lock")))
    missing = [f"{repo}@{commit[:12]}" for repo, commit, path in required
               if not has_arm64_support(repo, commit, path)]
    if missing:
        print(f"{tag}: no linux/arm64 support in {', '.join(missing)}", file=sys.stderr)
        return ""
    return tag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true",
                        help="print the release tag to build (empty when none) and exit")
    parser.add_argument("--release-tag")
    parser.add_argument("--repository", default="local-inference-lab/blackwell-llm-docker")
    parser.add_argument("--work", type=Path)
    parser.add_argument(
        "--cache", type=Path,
        default=Path(os.environ.get("LIL_SPARK_CACHE", Path.home() / ".cache/lil-spark")),
    )
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.select:
        print(select_release(args.repository))
        return
    if not args.release_tag or args.work is None:
        parser.error("--release-tag and --work are required to build")
    args.work.mkdir(parents=True, exist_ok=False)
    assembly = json.loads(
        release_asset(args.repository, args.release_tag, "community-assembly.json", args.work).read_text()
    )
    amd64 = json.loads(
        release_asset(args.repository, args.release_tag, "container-release.json", args.work).read_text()
    )
    image = spark_image(amd64["image"])
    bundles = {
        role: build_component(role, component, args.work, args.cache)
        for role, component in sorted(assembly["components"].items())
    }
    recipe = args.work / "recipe"
    checkout(args.repository, assembly["recipe_commit"], recipe)
    tools = recipe / "tools/jovian_wheel_runtime"
    if not (tools / "linux-arm64" / "foundation.lock").is_file():
        raise SystemExit(f"recipe {assembly['recipe_commit']} has no linux/arm64 runtime locks")
    command = ["python3", str(tools / "build_local_runtime.py"),
               "--output", str(args.work / "runtime-build"), "--image", image]
    for role, bundle in bundles.items():
        command += [f"--{role}-bundle", str(bundle)]
    execute(command, env=dict(os.environ, LIL_RUNTIME_PLATFORM=PLATFORM))
    inspect = json.loads(run(["docker", "image", "inspect", image]))[0]
    if inspect["Architecture"] != "arm64":
        raise SystemExit(f"{image} is {inspect['Architecture']}, not arm64")
    receipt = {
        "schema": "local-inference-container-platform-release/v1",
        "platform": PLATFORM,
        "status": "research-only",
        "release_tag": args.release_tag,
        "assembly_sha256": assembly["assembly_sha256"],
        "recipe_commit": assembly["recipe_commit"],
        "amd64_image": amd64["image"],
        "image": image,
        "alias": spark_alias(assembly["alias"]),
        "layers": len(inspect["RootFS"]["Layers"]),
        "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "components": {
            role: {
                "repository": assembly["components"][role]["repository"],
                "source_commit": assembly["components"][role]["source_commit"],
                "manifest_sha256": run(["sha256sum", str(bundle / "manifest.json")]).split()[0],
            }
            for role, bundle in bundles.items()
        },
    }
    receipt_path = args.work / RECEIPT
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    if args.build_only:
        print(f"Built {image}; registry publication was not requested")
        return
    with tempfile.TemporaryDirectory(prefix="ghcr-spark-") as registry_config:
        registry = ["docker", "--config", registry_config]
        execute(registry + ["login", "ghcr.io", "--username", os.environ["GITHUB_ACTOR"],
                            "--password-stdin"], input=os.environ["GH_TOKEN"].encode(),
                stdout=subprocess.DEVNULL)
        execute(registry + ["push", image])
        inspect = json.loads(run(["docker", "image", "inspect", image]))[0]
        receipt["digest"] = next(
            item for item in inspect["RepoDigests"] if item.startswith(image.split(":")[0] + "@")
        )
        execute(["docker", "tag", image, receipt["alias"]])
        execute(registry + ["push", receipt["alias"]])
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    execute(["gh", "release", "upload", args.release_tag, "--repo", args.repository,
             "--clobber", str(receipt_path)])


if __name__ == "__main__":
    main()
