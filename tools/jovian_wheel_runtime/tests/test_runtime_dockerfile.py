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
