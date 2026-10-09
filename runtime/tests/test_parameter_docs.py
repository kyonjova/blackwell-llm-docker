"""The configurator reads parameter-docs.yaml with a strict YAML parser."""

import re

import yaml

from runtime.launcher import ROOT

DOCS = ROOT / "parameter-docs.yaml"
PROFILES = {
    yaml.safe_load(path.read_text())["id"]
    for path in (ROOT / "profiles").glob("*.yaml")
    if path.stem != "common"
}
PRESETS = set(yaml.safe_load((ROOT / "presets.yaml").read_text())["presets"])


def entries() -> dict:
    data = yaml.safe_load(DOCS.read_text())
    return {**data["options"], **data["environment"]}


def test_parameter_docs_parse_and_keep_every_text():
    docs = entries()
    assert docs
    for name, entry in docs.items():
        # Every parameter has a summary; "why" exists where a profile chose a value.
        assert isinstance(entry.get("summary"), str) and entry["summary"].strip(), name
        assert "why" not in entry or isinstance(entry["why"], str), name
        assert set(entry) <= {"group", "summary", "why", "profiles", "presets"}, name
    # A ": " or " #" inside a plain scalar breaks parsing or silently drops the
    # rest of the line, so such texts must be folded blocks.
    source = DOCS.read_text().split("\n")
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


def test_model_notes_name_known_profiles_and_presets():
    for name, entry in entries().items():
        for field, known in (("profiles", PROFILES), ("presets", PRESETS)):
            if field not in entry:
                continue
            notes = entry[field]
            assert isinstance(notes, dict) and notes, (name, field)
            assert set(notes) <= known, (name, field, sorted(set(notes) - known))
            for ident, text in notes.items():
                assert isinstance(text, str) and text.strip(), (name, ident)
    # Every note is a folded block, so its text cannot break the parser.
    source = DOCS.read_text().split("\n")
    inside = False
    for line in source:
        if line in ("    profiles:", "    presets:"):
            inside = True
            continue
        if not line.startswith("      "):
            inside = False
        if inside and not line.startswith("        "):
            assert re.fullmatch(r"      [a-z0-9-]+: >-", line), line


def test_generic_why_names_no_model():
    """The generic text shows for every model; model-specific facts go in notes."""
    names = sorted(PROFILES | PRESETS, key=len, reverse=True)
    pattern = re.compile(
        r"GLM|Qwen|DS4|DeepSeek|MiMo|Kimi|Spark|"
        + "|".join(re.escape(name) for name in names)
    )
    for name, entry in entries().items():
        found = pattern.search(entry.get("why", ""))
        assert not found, (name, found and found.group())
