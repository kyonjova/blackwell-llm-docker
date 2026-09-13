#!/usr/bin/env python3
"""Verify requirements against distributions visible to the runtime interpreter."""

from __future__ import annotations

import importlib.metadata
import sys
from collections.abc import Iterable
from pathlib import Path

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
    roots: Iterable[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    pending = list(selected if roots is None else roots)
    checked: set[str] = set()
    while pending:
        package_name = canonicalize_name(pending.pop())
        if package_name in checked:
            continue
        checked.add(package_name)
        distribution = selected.get(package_name)
        if distribution is None:
            errors.append(f"runtime root {package_name} is not installed")
            continue
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
            if requirement.specifier and not requirement.specifier.contains(
                Version(dependency.version), prereleases=True
            ):
                errors.append(
                    f"{package_name} requires {requirement}, but {dependency_name} "
                    f"{dependency.version} is selected"
                )
                continue
            pending.append(dependency_name)
    return errors


def main() -> int:
    selected = selected_distributions()
    runtime_prefix = Path(sys.prefix).resolve()
    roots = {
        name
        for name, distribution in selected.items()
        if Path(distribution.locate_file("")).resolve().is_relative_to(runtime_prefix)
    }
    if not roots:
        print(f"no distributions are installed under runtime prefix {runtime_prefix}", file=sys.stderr)
        return 1
    errors = dependency_errors(selected, roots)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(
            f"runtime dependency verification failed with {len(errors)} incompatibilities",
            file=sys.stderr,
        )
        return 1
    print(
        f"runtime_dependency_closure=PASS roots={len(roots)} "
        f"visible_distributions={len(selected)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
