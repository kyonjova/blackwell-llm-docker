"""The configurator reads parameter-docs.yaml with a strict YAML parser."""

import yaml

from runtime.launcher import ROOT


def test_parameter_docs_parse_and_keep_every_text():
    data = yaml.safe_load((ROOT / "parameter-docs.yaml").read_text())
    entries = {**data["options"], **data["environment"]}
    assert entries
    for name, entry in entries.items():
        # Every parameter has a summary; "why" exists where a profile chose a value.
        assert isinstance(entry.get("summary"), str) and entry["summary"].strip(), name
        assert "why" not in entry or isinstance(entry["why"], str), name
    # A ": " or " #" inside a plain scalar breaks parsing or silently drops the
    # rest of the line, so such texts must be folded blocks.
    source = (ROOT / "parameter-docs.yaml").read_text().split("\n")
    for index, line in enumerate(source):
        stripped = line.strip()
        if not stripped.startswith(("summary: ", "why: ")) or stripped.endswith(": >-"):
            continue
        text = [stripped.split(": ", 1)[1]]
        for following in source[index + 1 :]:
            if not following.startswith("      "):
                break
            text.append(following.strip())
        joined = " ".join(text)
        assert ": " not in joined and " #" not in joined, line
