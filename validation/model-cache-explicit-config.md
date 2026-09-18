# Model cache profiles and materialized launch configuration

Status: implemented. Full model-serving and external-cache restore qualification
is pending; configuration checks do not establish cache-hit correctness or speed.

## Profile behavior

Qwen 3.8 Flash Next uses the engine-driven recurrent-checkpoint connector. It
stores attention cache and recurrent state at the same request boundary, requires
DCP1 and retains the profile's 6019-token scheduler budget. Periodic retention
and independently restored recurrent chunks are rejected.

DeepSeek V4.1 Flash uses the engine-driven multi-group connector, retaining its
256-token attention pages, 128-token sliding-window pages, 4096-token batch
budget and 0.95 GPU-memory fraction. Its deployment context limit is selected
from the model and available KV capacity (`max-model-len: -1`), with 32 request
slots. Engram RAM/disk placement is independent of external KV-cache placement.

Both profiles offer CPU L1 and optional persistent L2 storage. Qwen/DS4.1 cache
namespaces include their recurrent/sliding-window layout. Existing GLM and DS4
namespace construction is unchanged. An image profile that declares external
cache unsupported still rejects it, including through manual overrides.

The RTX PRO PCIe hardware profile disables two-shot all-reduce for GLM and
DS4.1, matching the qualification commands. One-shot and other collective paths
remain available.

## Explicit execution interface

`python -m runtime.explicit --config FILE` accepts a YAML/JSON document with
schema `lil-explicit-launch/v1`. `--config-yaml` and `--config-json` accept the
same data inline. `--print-config` validates and displays the native command
without starting serving.

The document contains resolved options, passthrough arguments and either an
environment mapping or a list of names supplied by the container environment.
Model defaults are not applied again. Runtime-specific JIT paths bind to the
verified image contract; external-cache identity binds to verified checkpoint
revisions before processes start. Cache controls must agree with the materialized
native arguments and ENV. Credentials and conflicting managed aliases are rejected.

`vllm_defaults` names options intentionally omitted from the native command;
it is not an enumeration of every internal serving-library default. GLM DFlash
exports preserve the separate 4096 target-token budget and 4320 native-row budget
without adding speculative slots twice.

## Evidence

- The shared runtime suite passes 198 CPU tests. Coverage includes seven
  model/speculation cases, two hardware profiles, GPU-only/CPU-L1/persistent-L2
  cache plans, invalid inputs, installed-runtime CLI parity and one bootstrap
  invocation at execution.
- Native vLLM CLI and scheduler validation passes 26 profile/mode/cache cases
  using the installed KK runtime image
  `sha256:79d8d57177e54435586a36f9e3ae5a609bf4e07d62f86ed7108f7cca192b7035`
  with the profile source mounted read-only. GPU0 was exposed for CUDA platform
  discovery; no model weights or inference workload were loaded.
- Before deployment qualification, run cold requests, native-prefix reuse,
  external restore after clearing GPU prefix state, and persistent-L2 restore
  across server restart for each model. These checks remain pending.
