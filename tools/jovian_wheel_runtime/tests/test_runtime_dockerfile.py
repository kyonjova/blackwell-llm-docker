"""Keep application installation in one layer over the immutable foundation."""

import re
from pathlib import Path


def test_application_stage_has_one_filesystem_instruction():
    recipe = (Path(__file__).resolve().parents[1] / "Dockerfile.runtime").read_text()
    application = recipe.split("FROM ${SOURCE_IMAGE} AS runtime", 1)[1]
    # WORKDIR can create a directory layer even when a preceding RUN populated
    # that path. Runtime imports use a .pth entry and need no working directory.
    assert re.findall(r"^(RUN|COPY|ADD|WORKDIR)\b", application, re.MULTILINE) == ["RUN"]
    assert 'ENTRYPOINT ["/usr/local/bin/lil-entrypoint"]' in application
    assert "runtime.image_install" in application
    assert "install_runtime_auxiliary.py" in application


def test_quack_kernel_dependency_is_locked_and_verified():
    tool_dir = Path(__file__).resolve().parents[1]
    lock = (tool_dir / "ngc-runtime-overlay.lock").read_text()
    verifier = (tool_dir / "verify_qwen38_runtime.py").read_text()
    recipe = (tool_dir / "Dockerfile.runtime").read_text()
    assert "quack-kernels==0.6.4" in lock
    assert "e77c5d1f1299b0b38487fe8737df6c6975daa16bca7f7eb883bd1a74d09e7e78" in lock
    assert '"quack-kernels": "0.6.4"' in verifier
    assert "import quack" in verifier
    assert "--overlay-lock /source/tools/jovian_wheel_runtime/ngc-runtime-overlay.lock" in recipe
