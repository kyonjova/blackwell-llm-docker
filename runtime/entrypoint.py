"""Dispatch model-profile serving or an explicit command through the ABI bootstrap."""

from __future__ import annotations

import os
import sys

from runtime import ConfigError
from runtime.launcher import ROOT
from runtime.packaging import verify_contract

LEGACY_PROFILES = {
    "serve-glm53-flash.sh": "glm53-flash",
    "serve-glm53-flash-nvfp4-dflash2.sh": "glm53-flash",
    "serve-ds4-jovian.sh": "ds4-flash",
    "serve-ds4-flash.sh": "ds4-flash",
    "serve-ds41-jovian.sh": "ds41-flash",
    "serve-ds41-flash.sh": "ds41-flash",
    "serve-qwen38-flash-next-nvfp4.sh": "qwen38-flash-next",
}


def command(argv: list[str], environment: dict[str, str], contract: dict) -> list[str]:
    """Return argv without interpreting shell text or importing the serving engine."""
    args = list(argv)
    legacy = args[0] if args else None
    if legacy in LEGACY_PROFILES:
        args.pop(0)
        profile = LEGACY_PROFILES[legacy]
        if args and not args[0].startswith("-"):
            args = ["--model", args[0], *args[1:]]
        return [
            "/opt/venv/bin/python",
            "-m",
            "runtime.launcher",
            "--profile",
            profile,
            *args,
        ]
    if args and args[0] == "serve":
        args.pop(0)
    elif args and not args[0].startswith("-"):
        # Explicit python/bash/vllm commands retain CUDA/NCCL preparation, but
        # intentionally do not inherit model policy from PROFILE.
        return [*contract.get("bootstrap", []), *args]
    if not args and not environment.get("PROFILE") and not environment.get("PRESET"):
        args = ["--help"]
    return ["/opt/venv/bin/python", "-m", "runtime.launcher", *args]


def main() -> int:
    try:
        contract = verify_contract(ROOT / "image-contract.json")
        args = command(sys.argv[1:], dict(os.environ), contract)
        environment = dict(os.environ)
        if environment.get("NCCL_GRAPH_FILE") == "":
            environment.pop("NCCL_GRAPH_FILE")
        os.execvpe(args[0], args, environment)
    except (ConfigError, OSError) as error:
        print(f"Runtime entrypoint error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
