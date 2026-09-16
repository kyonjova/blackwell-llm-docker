"""Check generated profile arguments against the installed vLLM parser, without serving."""

import json

from runtime.launcher import ROOT, profile, resolve


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
                parser.parse_args(plan.argv[3:])
                cases.append({"profile": path.stem, "mode": mode, "cache": cache})
    print(json.dumps({"native_cli_cases": cases, "result": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
