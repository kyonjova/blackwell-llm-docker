#!/usr/bin/env python3
"""Build and publish a source-addressed beta image after native GPU checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote

from container_channel import api, digest, download, run


def execute(args: list[str], **kwargs: object) -> None:
    subprocess.run(args, check=True, **kwargs)


def require_idle_gpu(gpu: str) -> None:
    if not gpu.startswith("GPU-"):
        raise ValueError("qualification GPU must be an explicit GPU UUID")
    fields = (
        run(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        .decode()
        .strip()
        .split(",")
    )
    if len(fields) != 2 or any(int(field.strip()) != 0 for field in fields):
        raise RuntimeError(f"qualification GPU is busy; no workload was stopped: {gpu}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    tools = Path(__file__).resolve().parent
    root = tools.parents[1]
    assembly = json.loads(args.assembly.read_text())
    commit = run(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    if commit != assembly["recipe_commit"]:
        raise ValueError("checkout is not the recipe selected by the assembly lock")
    if run(["git", "-C", str(root), "status", "--porcelain"]):
        raise ValueError("container build requires a clean recipe checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    bundles = args.output / "components"
    download(assembly, bundles)
    runtime = args.output / "runtime"
    command = [
        "python3",
        str(tools / "assemble_qwen38_runtime_bundle.py"),
        "--output",
        str(runtime),
        "--qwen-lock",
        str(tools / "qwen38-runtime.lock"),
        "--ngc-foundation-lock",
        str(tools / "foundation.lock"),
    ]
    for role in assembly["components"]:
        command += [f"--{role}-bundle", str(bundles / role)]
    execute(command)
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["assembly"] = assembly
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    checksum = runtime / "SHA256SUMS"
    checksum.write_text(
        "".join(
            f"{digest(path)}  {path.relative_to(runtime)}\n"
            for path in sorted(runtime.rglob("*"))
            if path.is_file() and path != checksum
        )
    )
    image = assembly["image"]
    execute(
        [
            str(tools / "run_serialized_build.sh"),
            str(tools / "build_qwen38_ngc_runtime_image.sh"),
            str(runtime),
            image,
        ]
    )
    base = manifest["components"]["foundation"]["source"]["image"]
    inspection = json.loads(run(["docker", "image", "inspect", base, image]))
    base_layers = inspection[0]["RootFS"]["Layers"]
    image_layers = inspection[1]["RootFS"]["Layers"]
    if (
        len(base_layers) != 67
        or len(image_layers) != 68
        or image_layers[:67] != base_layers
    ):
        raise ValueError(
            "container must retain the 67 foundation layers plus one application layer"
        )
    require_idle_gpu(args.gpu)
    execute(
        [
            "docker",
            "run",
            "--rm",
            "--device",
            f"nvidia.com/gpu={args.gpu}",
            image,
            "python",
            "/opt/venv/libexec/verify_qwen38_runtime.py",
            "--foundation",
            "ngc",
            "--require-gpu",
        ]
    )
    execute(
        [
            "docker",
            "run",
            "--rm",
            "--device",
            f"nvidia.com/gpu={args.gpu}",
            image,
            "python",
            "-c",
            (
                "from lmcache.lmcache_fs import LMCacheFSClient; "
                "from vllm.v1.core.kv_cache_manager import KVCacheManager; "
                "assert hasattr(KVCacheManager, 'reserve_external_boundary_checkpoint')"
            ),
        ]
    )
    # Cached native wheels must still satisfy the filesystem and checkpoint contracts.
    source = args.output / "lmcache-tests"
    execute(["git", "init", "--quiet", str(source)])
    execute(
        [
            "git",
            "-C",
            str(source),
            "fetch",
            "--quiet",
            "--depth=1",
            "https://github.com/local-inference-lab/LMCache.git",
            assembly["components"]["lmcache"]["source_commit"],
        ]
    )
    execute(
        [
            "git",
            "-C",
            str(source),
            "checkout",
            "--quiet",
            "--detach",
            assembly["components"]["lmcache"]["source_commit"],
        ]
    )
    execute(
        [
            "docker",
            "run",
            "--rm",
            "--ipc",
            "private",
            "--shm-size",
            "1g",
            "--device",
            f"nvidia.com/gpu={args.gpu}",
            "--mount",
            f"type=bind,src={source / 'tests'},dst=/qualification/tests,readonly",
            "-w",
            "/qualification",
            "-e",
            "LMCACHE_TRACK_USAGE=false",
            image,
            "python",
            "-m",
            "pytest",
            "--noconftest",
            "-q",
            "/qualification/tests/v1/multiprocess/test_checkpoint_identity.py",
            "/qualification/tests/v1/multiprocess/test_checkpoint_index.py",
            "/qualification/tests/v1/multiprocess/test_checkpoint_storage.py",
            "/qualification/tests/v1/storage_backend/test_fs_native_connector.py",
            "/qualification/tests/v1/test_vllm_semantic_checkpoint_transfer.py",
        ]
    )
    receipt = {
        **assembly,
        "status": "qualified",
        "qualification_scope": "Native GPU smoke and LMCache checkpoint/filesystem contract tests; "
        "model-serving performance and full GLM cache E2E remain unqualified.",
        "runtime_manifest_sha256": digest(manifest_path),
        "layers": 68,
    }
    (args.output / "container-release.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    if args.build_only:
        print(
            f"Build and native checks complete: {image}; registry publication was not requested"
        )
        return
    repository = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GH_TOKEN"]
    actor = os.environ["GITHUB_ACTOR"]
    # Isolate registry credentials without replacing the shared Buildx configuration.
    with tempfile.TemporaryDirectory(prefix="ghcr-publisher-") as registry_config:
        registry_docker = ["docker", "--config", registry_config]
        execute(
            registry_docker
            + ["login", "ghcr.io", "--username", actor, "--password-stdin"],
            input=token.encode(),
            stdout=subprocess.DEVNULL,
        )
        execute(registry_docker + ["push", image])
        inspect = json.loads(run(["docker", "image", "inspect", image]))[0]
        receipt["digest"] = next(
            item
            for item in inspect["RepoDigests"]
            if item.startswith(image.split(":")[0] + "@sha256:")
        )
        receipt_path = args.output / "container-release.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        # Serialize resolution/publication in Actions; an obsolete source cannot replace beta.
        superseded = False
        for component in assembly["components"].values():
            head = api(
                f"repos/{component['repository']}/commits/"
                f"{quote(component['branch'], safe='')}"
            )
            # New source is handled by the following scan. Immutable image remains usable.
            if component.get("observed_branch_commit") not in {None, head["sha"]}:
                superseded = True
        promote_alias = (
            not superseded and os.environ.get("GITHUB_REF") == "refs/heads/main"
        )
        if promote_alias:
            execute(["docker", "tag", image, assembly["alias"]])
            execute(registry_docker + ["push", assembly["alias"]])
        receipt["alias_updated"] = promote_alias
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    notes = args.output / "release-notes.md"
    rows = [
        "# Jovian Judgement wheel-built beta",
        "",
        f"Image: `{image}`",
        "",
        f"Digest: `{receipt['digest']}`",
        "",
        receipt["qualification_scope"],
        "",
        "| Component | Source commit | Wheel release |",
        "|---|---|---|",
    ]
    for role, component in assembly["components"].items():
        rows.append(
            f"| {role} | `{component['source_commit']}` | "
            f"[{component['release']}]({component['release_url']}) |"
        )
    notes.write_text("\n".join(rows) + "\n")
    execute(
        [
            "gh",
            "release",
            "create",
            assembly["release_tag"],
            "--repo",
            repository,
            "--target",
            commit,
            "--prerelease",
            "--title",
            image.rsplit(":", 1)[1],
            "--notes-file",
            str(notes),
            str(receipt_path),
            str(manifest_path),
            str(args.assembly),
        ]
    )


if __name__ == "__main__":
    main()
