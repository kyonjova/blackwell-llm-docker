"""Execute materialized launch values without inheriting model-profile defaults.

The document includes all resolved options and environment variables. Only
image-specific JIT paths and checkpoint-cache identities are bound at execution.
LMCache service arguments remain derived from the visible cache controls, with
native geometry checked against the exported values before any process starts.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

import yaml

from runtime import ConfigError
from runtime.cache import configure as configure_cache
from runtime.launcher import (
    JIT_PATHS,
    NAME,
    ROOT,
    SECRET,
    LaunchPlan,
    UniqueLoader,
    _redact,
    convert,
    execute,
    make_argv,
    parse_native,
    read_yaml,
    validate,
)

SCHEMA = "lil-explicit-launch/v1"
GENERATED_OPTIONS = {
    "kv-transfer-config",
    "kv-offloading-backend",
    "kv-offloading-size",
    "enable-cumem-allocator",
    "max-num-scheduled-tokens",
}
TOKEN = re.compile(r"hf_[A-Za-z0-9]{15,}|github_pat_[A-Za-z0-9_]{15,}")
NATIVE_ENV_CONTROLS = {
    "VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE": "target-page-size",
    "VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE": "recurrent-page-size",
    "LMCACHE_CUMEM_BROKER_DIR": "cache-broker-directory",
}


def reject_secrets(document: dict) -> None:
    """Require callers to supply credentials separately from portable exports."""
    encoded = json.dumps(document, sort_keys=True, allow_nan=False)
    if encoded != json.dumps(
        _redact(document), sort_keys=True, allow_nan=False
    ) or TOKEN.search(encoded):
        raise ConfigError("Explicit launch documents must not contain credentials")
    for argument in document.get("passthrough", []):
        if argument.startswith("--") and SECRET.search(argument.partition("=")[0]):
            raise ConfigError("Explicit launch documents must not contain credentials")


def document(
    plan: LaunchPlan,
    *,
    image_environment: dict | None = None,
    specs: dict | None = None,
) -> dict:
    """Create a complete data-only launch document from an already resolved plan."""
    specs = specs or read_yaml(ROOT / "options.yaml")
    values = copy.deepcopy(plan.values)
    if plan.profile == "glm53-flash" and values["cache-mode"] == "lmcache":
        # The native row budget includes speculative slots. The service needs
        # the separate target budget to avoid adding those slots a second time.
        values["cache-target-tokens"] = values["max-num-scheduled-tokens"]
    result = {
        "schema": SCHEMA,
        "profile": plan.profile,
        "hardware": plan.hardware,
        "options": values,
        "environment": {**(image_environment or {}), **plan.environment},
        "passthrough": list(plan.passthrough),
        "vllm_defaults": sorted(
            key
            for key, spec in specs.items()
            if key not in values and not spec.get("control")
        ),
        "runtime_bindings": {
            "UNBOUND-RUNTIME": "Runtime lock from the selected immutable image",
            "checkpoint_identity": "Verified target/draft revisions before opening external storage",
        },
    }
    reject_secrets(result)
    return result


def load_plan(
    data: dict,
    *,
    runtime_identity: str,
    incoming_environment: dict[str, str] | None = None,
) -> LaunchPlan:
    """Validate a materialized configuration and bind its runtime-owned paths."""
    expected = {
        "schema",
        "profile",
        "hardware",
        "options",
        "passthrough",
        "vllm_defaults",
        "runtime_bindings",
    }
    if (
        not isinstance(data, dict)
        or set(data)
        not in (expected | {"environment"}, expected | {"environment_keys"})
        or data.get("schema") != SCHEMA
    ):
        raise ConfigError("Unsupported or incomplete explicit launch document")
    if not re.fullmatch(r"[0-9a-f]{64}", runtime_identity):
        raise ConfigError(
            "Explicit execution requires the image's immutable runtime lock"
        )
    for key in ("profile", "hardware"):
        if not isinstance(data[key], str) or not re.fullmatch(
            r"[a-z][a-z0-9-]*", data[key]
        ):
            raise ConfigError(f"Invalid explicit {key}")
    if not isinstance(data["options"], dict):
        raise ConfigError("Explicit options must be a mapping")
    if not isinstance(data["passthrough"], list) or not all(
        isinstance(item, str) for item in data["passthrough"]
    ):
        raise ConfigError("Explicit passthrough must be an argument list")
    reject_secrets(data)
    specs = read_yaml(ROOT / "options.yaml")
    values = copy.deepcopy(data["options"])
    for key, value in values.items():
        if key in specs:
            if convert(key, value, specs[key]) != value:
                raise ConfigError(f"Explicit --{key} must use its typed resolved value")
        elif key not in GENERATED_OPTIONS:
            raise ConfigError(f"Unknown explicit option: {key}")
    consumed = {alias for spec in specs.values() for alias in spec["env"]}
    if "environment_keys" in data:
        names = data["environment_keys"]
        if (
            not isinstance(names, list)
            or not all(isinstance(key, str) for key in names)
            or len(set(names)) != len(names)
        ):
            raise ConfigError("Explicit environment_keys must be unique names")
        missing = set(names) - set(incoming_environment or {})
        if missing:
            raise ConfigError(
                "Missing exported environment values: " + ", ".join(sorted(missing))
            )
        environment = {key: incoming_environment[key] for key in names}
    else:
        if not isinstance(data["environment"], dict):
            raise ConfigError("Explicit environment must be a mapping")
        environment = dict(data["environment"])
    reject_secrets({"environment": environment})
    for key, value in environment.items():
        if not NAME.fullmatch(key) or not isinstance(value, str) or "\x00" in value:
            raise ConfigError(
                "Explicit ENV needs valid names and NUL-free string values"
            )
        if key in consumed and (
            key not in NATIVE_ENV_CONTROLS
            or value != values.get(NATIVE_ENV_CONTROLS[key])
        ):
            # GLM's page controls are also native ENV consumers. Retain only
            # their matching resolved values; other aliases duplicate options.
            raise ConfigError(f"Set {key} through explicit options, not an ENV alias")
        if "UNBOUND-RUNTIME" in value:
            if key not in JIT_PATHS:
                raise ConfigError(f"Unbound runtime placeholder is not valid in {key}")
            environment[key] = value.replace("UNBOUND-RUNTIME", runtime_identity)
    if environment.get("NCCL_GRAPH_FILE") == "":
        raise ConfigError("NCCL_GRAPH_FILE must be absent or name a real graph file")
    managed, passthrough = parse_native(data["passthrough"], specs)
    if managed:
        raise ConfigError("Managed options belong in the explicit options mapping")
    origins = {key: "explicit:materialized configuration" for key in values}
    env_origins = {key: "explicit:materialized configuration" for key in environment}
    native_before = {
        key: value
        for key, value in values.items()
        if not specs.get(key, {}).get("control")
    }
    env_before = dict(environment)
    if (
        data["profile"] == "glm53-flash"
        and values.get("cache-mode") == "lmcache"
        and values.get("cache-transfer-mode") == "lmcache_driven"
    ):
        interposer = "/opt/lmcache/lib/liblmcache_cumem_shareable.so"
        preload = environment.get("LD_PRELOAD", "")
        if preload == interposer:
            environment.pop("LD_PRELOAD")
        elif preload.startswith(interposer + ":"):
            environment["LD_PRELOAD"] = preload[len(interposer) + 1 :]
    try:
        service = configure_cache(
            values, origins, environment, env_origins, data["profile"], runtime_identity
        )
        validate(values, environment, data["profile"])
    except KeyError as error:
        raise ConfigError(
            f"Explicit document is missing required option: {error.args[0]}"
        ) from error
    native_after = {
        key: value
        for key, value in values.items()
        if not specs.get(key, {}).get("control")
    }
    if native_after != native_before or environment != env_before:
        raise ConfigError(
            "Explicit native options/ENV conflict with cache controls; regenerate the export"
        )
    return LaunchPlan(
        data["profile"],
        data["hardware"],
        values,
        environment,
        origins,
        env_origins,
        make_argv(values, passthrough),
        passthrough,
        [],
        service,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config", type=Path, help="YAML or JSON explicit launch document"
    )
    source.add_argument(
        "--config-json", help="Inline explicit launch document for docker run"
    )
    source.add_argument(
        "--config-yaml", help="Inline explicit YAML for Docker or Compose"
    )
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    try:
        from runtime.packaging import verify_contract

        contract = verify_contract(ROOT / "image-contract.json")
        if args.config:
            data = read_yaml(args.config)
        elif args.config_yaml:
            data = yaml.load(args.config_yaml, Loader=UniqueLoader)
        else:
            data = json.loads(args.config_json)
        plan = load_plan(
            data,
            runtime_identity=contract["runtime_lock_sha256"],
            incoming_environment=dict(os.environ),
        )
        if args.print_config:
            print(json.dumps(plan.public(), indent=2, allow_nan=False))
        else:
            execute(plan, ROOT / "image-contract.json")
    except (ConfigError, OSError, ValueError, TypeError, yaml.YAMLError) as error:
        print(f"Explicit launch configuration error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
