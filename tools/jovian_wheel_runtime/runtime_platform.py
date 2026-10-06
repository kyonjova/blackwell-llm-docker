"""Select the platform-specific locks of the runtime image build.

linux/amd64 keeps its locks next to the build tools. Another platform keeps
the same file names in a directory named after it, for example
tools/jovian_wheel_runtime/linux-arm64/foundation.lock, and its foundation
environment policy in runtime/platform-environment.<platform>.json.
LIL_RUNTIME_PLATFORM selects the platform; it defaults to linux/amd64.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT = "linux/amd64"
SUFFIXES = {"linux/amd64": "", "linux/arm64": "linux-arm64"}


def selected() -> str:
    platform = os.environ.get("LIL_RUNTIME_PLATFORM", DEFAULT)
    if platform not in SUFFIXES:
        raise ValueError(f"Unsupported runtime platform: {platform}")
    return platform


def lock_dir(tools: Path, platform: str) -> Path:
    suffix = SUFFIXES[platform]
    return tools / suffix if suffix else tools


def environment_policy(root: Path, platform: str) -> Path:
    suffix = SUFFIXES[platform]
    name = f"platform-environment.{suffix}.json" if suffix else "platform-environment.json"
    return root / "runtime" / name


def read_lock(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )
