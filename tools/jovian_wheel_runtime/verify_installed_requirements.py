#!/usr/bin/env python3
"""Verify requirements against distributions visible to the runtime interpreter."""

from __future__ import annotations

import importlib.metadata
import sys

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def selected_distributions() -> dict[str, importlib.metadata.Distribution]:
    selected: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            selected.setdefault(canonicalize_name(name), distribution)
    return selected


def dependency_errors(
    selected: dict[str, importlib.metadata.Distribution],
) -> list[str]:
    errors: list[str] = []
    for package_name, distribution in sorted(selected.items()):
        for raw_requirement in distribution.requires or ():
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement as error:
                errors.append(f"{package_name}: invalid requirement {raw_requirement!r}: {error}")
                continue
            if requirement.marker is not None and not requirement.marker.evaluate(
                {"extra": ""}
            ):
                continue
            dependency_name = canonicalize_name(requirement.name)
            dependency = selected.get(dependency_name)
            if dependency is None:
                errors.append(f"{package_name} requires {requirement}, but it is not installed")
                continue
            if requirement.specifier and Version(dependency.version) not in requirement.specifier:
                errors.append(
                    f"{package_name} requires {requirement}, but {dependency_name} "
                    f"{dependency.version} is selected"
                )
    return errors


def main() -> int:
    selected = selected_distributions()
    errors = dependency_errors(selected)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(
            f"runtime dependency verification failed with {len(errors)} incompatibilities",
            file=sys.stderr,
        )
        return 1
    print(f"runtime_dependency_closure=PASS distributions={len(selected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
