"""Runtime chat templates: provenance, launcher selection and multi-turn reuse."""

import hashlib
import json

import pytest

from runtime import ConfigError
from runtime.launcher import ROOT, resolve
from runtime.packaging import payload_hashes

GLM_TEMPLATE = ROOT / "templates" / "glm53-flash.jinja"
# zai-org/GLM-5.3-Flash chat_template.jinja; the same file ships in
# local-inference-lab/GLM-5.3-Flash-NVFP4 and GLM-5.3-Flash-NVFP4-Spark.
GLM_OFFICIAL_SHA256 = "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5"
# The assistant branch, after its <think> block; the only line that differs.
THINK_END = "{{ '<think></think>' }}\n{%- endif -%}\n"
OFFICIAL_CONTENT = THINK_END + (
    "{%- if content.strip() -%}\n{{ content.strip() }}\n{%- endif -%}\n"
)
VERBATIM_CONTENT = THINK_END + "{{- content -}}\n"


def official_glm_template() -> str:
    text = GLM_TEMPLATE.read_text()
    header, separator, body = text.partition("-#}\n")
    assert separator and header.startswith("{#-")
    assert body.count(VERBATIM_CONTENT) == 1
    return body.replace(VERBATIM_CONTENT, OFFICIAL_CONTENT)


def test_glm_template_is_the_official_one_with_verbatim_content():
    official = official_glm_template()
    assert hashlib.sha256(official.encode()).hexdigest() == GLM_OFFICIAL_SHA256
    assert "content.strip()" not in GLM_TEMPLATE.read_text().partition("-#}\n")[2]


def test_template_is_installed_with_the_profiles():
    assert "templates/glm53-flash.jinja" in payload_hashes()


@pytest.mark.parametrize("preset", [None, "glm53-spark-tp2"])
def test_glm_serves_the_runtime_template(preset):
    plan = resolve("glm53-flash", env={}, preset=preset)
    assert plan.values["chat-template"] == "runtime:templates/glm53-flash.jinja"
    index = plan.argv.index("--chat-template")
    assert plan.argv[index + 1] == str(GLM_TEMPLATE)
    assert plan.argv.count("--chat-template") == 1


def test_other_profiles_keep_the_checkpoint_template():
    for identifier in ("qwen38-flash-next", "ds4-flash", "ds41-flash", "mimo26-flash"):
        assert "--chat-template" not in resolve(identifier, env={}).argv


def test_operator_can_restore_or_replace_the_template(tmp_path):
    plan = resolve("glm53-flash", env={"CHAT_TEMPLATE": "checkpoint"})
    assert "--chat-template" not in plan.argv
    custom = tmp_path / "custom.jinja"
    custom.write_text("{{ messages }}")
    for plan in (
        resolve("glm53-flash", env={"CHAT_TEMPLATE": str(custom)}),
        resolve("glm53-flash", env={}, argv=["--chat-template", str(custom)]),
    ):
        index = plan.argv.index("--chat-template")
        assert plan.argv[index + 1] == str(custom)


@pytest.mark.parametrize(
    "value", ["runtime:templates/missing.jinja", "runtime:templates/../launcher.py"]
)
def test_unknown_runtime_template_is_rejected(value):
    with pytest.raises(ConfigError, match="Unknown runtime chat template"):
        resolve("glm53-flash", env={"CHAT_TEMPLATE": value})


# Rendering needs jinja2 (a transformers dependency), which the profile
# resolver itself does not; CI installs it for these checks.
jinja2 = pytest.importorskip("jinja2")


def render(template: str, messages: list[dict], **kwargs) -> str:
    """Render like transformers.apply_chat_template."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(
        value, ensure_ascii=False, indent=None, separators=None, sort_keys=False
    ):
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    return environment.from_string(template).render(messages=messages, **kwargs)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }
]
READ_CALL = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "read", "arguments": {"filePath": "/repo/a.py", "limit": 200}},
}
READ_TEXT = (
    "<tool_call>read<arg_key>filePath</arg_key><arg_value>/repo/a.py</arg_value>"
    "<arg_key>limit</arg_key><arg_value>200</arg_value></tool_call>"
)
# (text generated after the prompt's <think>, the stop token, and the message
# vLLM's glm45/glm47 parsers return for it).
TURNS = {
    "answer_trailing_newline": (
        "Plan.</think>Use move_to_end.\n",
        "<|user|>",
        {"reasoning_content": "Plan.", "content": "Use move_to_end.\n"},
    ),
    "answer_code_block": (
        "Plan.</think>Fixed:\n\n```python\nx = 1\n```\n",
        "<|user|>",
        {"reasoning_content": "Plan.", "content": "Fixed:\n\n```python\nx = 1\n```\n"},
    ),
    "answer_leading_newline": (
        "Plan.\n</think>\nDone.",
        "<|user|>",
        {"reasoning_content": "Plan.\n", "content": "\nDone."},
    ),
    "no_reasoning": ("</think>Done.\n", "<|user|>", {"content": "Done.\n"}),
    "tool_call_with_content": (
        "Read it.</think>Let me read the file.\n" + READ_TEXT,
        "<|observation|>",
        {
            "reasoning_content": "Read it.",
            "content": "Let me read the file.\n",
            "tool_calls": [READ_CALL],
        },
    ),
    "tool_call_after_whitespace": (
        "Read it.</think>\n" + READ_TEXT,
        "<|observation|>",
        {"reasoning_content": "Read it.", "content": "\n", "tool_calls": [READ_CALL]},
    ),
}
KWARGS = {"reasoning_effort": "high", "clear_thinking": False, "tools": TOOLS}


@pytest.mark.parametrize("name", TURNS)
def test_next_prompt_extends_the_generated_tokens(name):
    """The whole response stays a prefix of the next request's prompt."""
    raw, stop, message = TURNS[name]
    history = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Why is the new entry evicted?"},
    ]
    if "tool_calls" in message:
        follow = [{"role": "tool", "tool_call_id": "call_0", "content": "file text"}]
    else:
        follow = [{"role": "user", "content": "And in get?"}]
    template = GLM_TEMPLATE.read_text()
    first = render(template, history, add_generation_prompt=True, **KWARGS)
    second = render(
        template,
        [*history, {"role": "assistant", **message}, *follow],
        add_generation_prompt=True,
        **KWARGS,
    )
    assert first.endswith("<|assistant|><think>")
    assert second.startswith(first + raw + stop)
    # The checkpoint's template drops whitespace the model generated.
    if message.get("content", "") != message.get("content", "").strip():
        official = render(
            official_glm_template(),
            [*history, {"role": "assistant", **message}, *follow],
            add_generation_prompt=True,
            **KWARGS,
        )
        assert not official.startswith(first + raw + stop)


@pytest.mark.parametrize("clear_thinking", [False, True])
def test_whitespace_free_turns_render_like_the_official_template(clear_thinking):
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Read a.py"},
        {"role": "assistant", "reasoning_content": "Plan.", "tool_calls": [READ_CALL]},
        {"role": "tool", "tool_call_id": "call_0", "content": "print(1)"},
        {"role": "assistant", "reasoning_content": "Done.", "content": "It prints 1."},
        {"role": "user", "content": "Thanks"},
    ]
    kwargs = {**KWARGS, "clear_thinking": clear_thinking}
    assert render(
        GLM_TEMPLATE.read_text(), messages, add_generation_prompt=True, **kwargs
    ) == render(official_glm_template(), messages, add_generation_prompt=True, **kwargs)
