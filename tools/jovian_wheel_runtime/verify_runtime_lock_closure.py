#!/usr/bin/env python3
"""Check external lock membership; installed component metadata is checked separately.

This check compares the public requirements resolver's output with explicit
locks. It does not certify local component wheels. The container build must
also run verify_installed_requirements.py, which traverses the actual installed
Requires-Dist metadata, including selected dependency extras.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# These distributions are supplied by immutable local component bundles.  The
# public resolver selects PyPI substitutes only because component wheels are
# not inputs to the external dependency lock compilation.
LOCALLY_PROVIDED = {
    "cuda-toolkit",
    "nvidia-nccl-cu13",
    "torch",
    "triton",
}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def package_names(path: Path) -> set[str]:
    packages: set[str] = set()
    for line in path.read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==", line)
        if match:
            packages.add(normalize(match.group(1)))
    return packages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-lock", required=True, type=Path)
    parser.add_argument("--runtime-lock", required=True, type=Path)
    parser.add_argument("--resolved-lock", required=True, type=Path)
    args = parser.parse_args()

    explicit = package_names(args.foundation_lock) | package_names(args.runtime_lock)
    required = package_names(args.resolved_lock)
    missing = sorted(required - explicit - LOCALLY_PROVIDED)
    if missing:
        print(
            "runtime locks omit resolved dependencies: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    print(f"runtime_lock_closure=PASS packages={len(required)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
