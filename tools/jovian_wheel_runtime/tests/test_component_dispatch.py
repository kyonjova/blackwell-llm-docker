"""Execute the credentialed workflow command against a local GitHub CLI stub."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github/workflows/component-container-dispatch.yml"
)


@pytest.mark.parametrize(
    ("token", "failures", "expected_calls", "success"),
    [
        ("", 0, 0, False),
        ("test-only-secret", 0, 1, True),
        ("test-only-secret", 1, 2, True),
        ("test-only-secret", 3, 3, False),
    ],
)
def test_dispatch_has_fixed_destination_and_bounded_retries(
    tmp_path, token, failures, expected_calls, success
):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    job = workflow["jobs"]["dispatch"]
    assert workflow["permissions"] == {}
    assert job["runs-on"] == "ubuntu-24.04"
    assert len(job["steps"]) == 1
    step = job["steps"][0]
    assert step["env"]["GH_HOST"] == "github.com"
    stub = tmp_path / "gh"
    stub.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$*" >> "$CALLS"\n'
        'count=$(wc -l < "$CALLS")\n'
        "[[ $count -gt $FAILURES ]]\n"
    )
    stub.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/bash\nexit 0\n")
    sleep.chmod(0o755)
    calls = tmp_path / "calls"
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "GH_TOKEN": token,
            "CALLS": str(calls),
            "FAILURES": str(failures),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert (result.returncode == 0) is success
    actual = calls.read_text().splitlines() if calls.exists() else []
    assert (
        actual
        == [
            "api --hostname github.com --method POST "
            "repos/local-inference-lab/blackwell-llm-docker/actions/workflows/"
            "community-container-release.yml/dispatches -f ref=main"
        ]
        * expected_calls
    )
    assert "test-only-secret" not in result.stdout + result.stderr
    assert ("::error::" in result.stderr) is not success
