"""Package Python-only LMCache changes with ABI-matched compiled extensions.

The reference image must contain the same native sources and build policy.
Compiled payloads are copied byte-for-byte; wheel tags and RECORD describe the
resulting platform wheel. Native-source changes fail rather than reusing an
incompatible binary.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import subprocess
import zipfile

NATIVE_INPUTS = (
    "csrc",
    "rust",
    "setup_extensions",
    "CMakeLists.txt",
    "setup.py",
    "pyproject.toml",
    "MANIFEST.in",
    "requirements",
)


def native_identity(root: Path) -> dict[str, str]:
    return {
        name: subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", f"HEAD:{name}"], text=True
        ).strip()
        for name in NATIVE_INPUTS
    }


def platform_wheel(
    files: dict[str, bytes], native: dict[str, bytes], tag: str
) -> dict[str, bytes]:
    """Return complete wheel members with consistent platform metadata and hashes."""
    if not native or any(
        not path.startswith("lmcache/") or not path.endswith(".so") for path in native
    ):
        raise ValueError("Expected compiled extensions in the lmcache package")
    wheel_paths = [name for name in files if name.endswith(".dist-info/WHEEL")]
    if len(wheel_paths) != 1 or tag != "cp312-cp312-linux_x86_64":
        raise ValueError("Expected one CPython 3.12 Linux x86_64 wheel")
    result = {**files, **native}
    wheel_path = wheel_paths[0]
    lines = [
        line
        for line in files[wheel_path].decode().splitlines()
        if line and not line.startswith(("Tag:", "Root-Is-Purelib:"))
    ]
    result[wheel_path] = (
        "\n".join(lines + ["Root-Is-Purelib: false", f"Tag: {tag}"]) + "\n"
    ).encode()
    record = wheel_path.removesuffix("WHEEL") + "RECORD"
    rows = io.StringIO(newline="")
    writer = csv.writer(rows, lineterminator="\n")
    for name, data in sorted(result.items()):
        if name != record:
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .rstrip(b"=")
                .decode()
            )
            writer.writerow([name, f"sha256={digest}", len(data)])
    writer.writerow([record, "", ""])
    result[record] = rows.getvalue().encode()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    args = parser.parse_args()
    if native_identity(args.source) != native_identity(args.reference_source):
        raise ValueError(
            "LMCache native build inputs differ; provide compatible compiled artifacts"
        )
    wheels = list(args.wheel_dir.glob("*.whl"))
    if len(wheels) != 1 or not wheels[0].name.endswith("-py3-none-any.whl"):
        raise ValueError("Expected one Python-only LMCache wheel")
    source = wheels[0]
    reference_metadata = list(
        args.reference_package.parent.glob("lmcache-*.dist-info/WHEEL")
    )
    if len(reference_metadata) != 1:
        raise ValueError("Expected one installed LMCache distribution")
    tags = [
        line.removeprefix("Tag: ")
        for line in reference_metadata[0].read_text().splitlines()
        if line.startswith("Tag: ")
    ]
    if tags != ["cp312-cp312-linux_x86_64"]:
        raise ValueError(f"Unsupported reference wheel tags: {tags}")
    native = {
        "lmcache/" + str(path.relative_to(args.reference_package)): path.read_bytes()
        for path in args.reference_package.rglob("*.so")
    }
    with zipfile.ZipFile(source) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    output = source.with_name(
        source.name.removesuffix("-py3-none-any.whl") + f"-{tags[0]}.whl"
    )
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in platform_wheel(files, native, tags[0]).items():
            archive.writestr(name, data)
    source.unlink()
    print(f"Preserved {len(native)} compiled LMCache artifacts in {output.name}")


if __name__ == "__main__":
    main()
