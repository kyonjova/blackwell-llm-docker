"""Validate profile CLI and scheduler contracts without loading model weights."""

import inspect
import json

from runtime.launcher import ROOT, profile, resolve


def validate_scheduler(args):
    """Exercise native cross-field validators independently of checkpoint access."""
    from vllm.config.scheduler import SchedulerConfig

    parameters = inspect.signature(SchedulerConfig).parameters
    values = {
        name: value
        for name, value in vars(args).items()
        if name in parameters and value is not None
    }
    # Automatic model lengths require checkpoint metadata. A concrete context
    # validates scheduling relationships without claiming model qualification.
    if not isinstance(values.get("max_model_len"), int) or values["max_model_len"] <= 0:
        values["max_model_len"] = 131072
    values["is_encoder_decoder"] = False
    SchedulerConfig(**values)


def main():
    # Imports are deliberately outside the configuration resolver: this test
    # uses the native serving installation but never loads model weights.
    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    ServeSubcommand().subparser_init(parser.add_subparsers())
    cases = []
    for path in sorted((ROOT / "profiles").glob("*.yaml")):
        if path.stem == "common":
            continue
        for mode in profile("model", path.stem)["modes"]:
            for cache in (
                ("vram", "lmcache")
                if path.stem in {"glm53-flash", "ds4-flash", "ds4-vision"}
                else ("vram",)
            ):
                plan = resolve(
                    path.stem,
                    env={"SPECULATOR": mode, "CACHE_MODE": cache},
                    runtime_identity="a" * 64,
                )
                args = parser.parse_args(plan.argv[3:])
                validate_scheduler(args)
                cases.append({"profile": path.stem, "mode": mode, "cache": cache})
    print(json.dumps({"native_cli_cases": cases, "result": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
