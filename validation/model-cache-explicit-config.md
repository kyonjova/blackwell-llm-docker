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
Derived cache options are recomputed from an empty derived-option set before
comparison. Values left over from another cache mode, missing generated values
and native passthrough attempts cannot change the selected cache contract.

`vllm_defaults` names options intentionally omitted from the native command;
it is not an enumeration of every internal serving-library default. GLM DFlash
exports preserve the separate 4096 target-token budget and 4320 native-row budget
without adding speculative slots twice.

## Evidence

- The shared runtime suite passes 198 CPU tests. Coverage includes seven
  model/speculation cases, two hardware profiles, GPU-only/CPU-L1/persistent-L2
  cache plans, invalid inputs, installed-runtime CLI parity and one bootstrap
  invocation at execution.
- Generated-cache validation and the serving-probe contracts bring the combined
  runtime/container suite to 411 passing CPU tests. The cache validator rejects
  backend, size, allocator, connector and scheduled-token edits before execution;
  valid materialized configurations preserve their native arguments.
- Native vLLM CLI and scheduler validation passes 26 profile/mode/cache cases
  using the installed KK runtime image
  `sha256:79d8d57177e54435586a36f9e3ae5a609bf4e07d62f86ed7108f7cca192b7035`
  with the profile source mounted read-only. GPU0 was exposed for CUDA platform
  discovery; no model weights or inference workload were loaded.
- Before deployment qualification, run cold requests, native-prefix reuse,
  external restore after clearing GPU prefix state, and persistent-L2 restore
  across server restart for each model. These checks remain pending.

## Serving cache probe

Status: implemented; the probe's 31 CPU contract tests pass. Model-serving cache
qualification remains pending.

`tools/jovian_wheel_runtime/qualify_model_cache.py` sends a long catalog prompt
with an isolated cache salt at temperature 1. It verifies two catalog answers,
including a changed user turn sharing the system prompt. It records responses,
raw metric samples, elapsed time, GPU reset acknowledgements and container image
identities. It does not compare stochastic output token sequences or infer cache
reuse from latency alone. The default fixture tests text state. `--fixture vision`
places a red/blue image before a long shared text prefix, then asks for the left
or right color. Reusing at least 4096 prefix tokens therefore includes the
image-dependent attention/recurrent state. Both fixtures require factual answers
after cold, GPU-prefix, CPU and persistent restores; the vision fixture has no
external image URL or image-library dependency.

Use a dedicated endpoint: this test clears its GPU prefix cache, but never clears
external objects, aborts active requests, or starts/stops a container. It rejects
busy endpoints, absent evidence counters, metric resets and additional completed
requests inside a measurement. The minimum reusable prefix is 4096 tokens.

For a server on port 5058 with the profile's derived cache metrics port 15060:

```bash
python tools/jovian_wheel_runtime/qualify_model_cache.py prime \
  --dedicated-endpoint --base-url http://127.0.0.1:5058 \
  --cache-metrics-url http://127.0.0.1:15060/metrics \
  --container cache-qualification --model GLM-5.3-Flash \
  --persistent --output-dir /tmp/glm-cache-qualification
```

The directory must not exist. `--persistent` requires completed L2 stores in
addition to cold, GPU-prefix and CPU-L1 checks. Omit it for a CPU-only cache.
The `prime` receipt qualifies those stages only, not restart restoration.

After restarting the dedicated serving and LMCache processes while retaining
their persistent directory, run:

```bash
python tools/jovian_wheel_runtime/qualify_model_cache.py restore \
  --dedicated-endpoint --base-url http://127.0.0.1:5058 \
  --cache-metrics-url http://127.0.0.1:15060/metrics \
  --output-dir /tmp/glm-cache-qualification
```

The probe requires changed container start times and unchanged image IDs for
both roles. A separate cache container is selected with `--cache-container` in
`prime`; otherwise both processes are expected in the serving container. The
restored request must show external token hits, zero GPU-prefix hits and actual
L2 object loads. Receipts are never overwritten, including failed runs.

Run both stages separately for GLM MTP3 and DFlash2, DS4 text and Vision,
DS4.1 DSpark, and Qwen MTP3. Use each profile's advertised model ID and native
chat-template settings; `--thinking-kwargs` accepts an explicit JSON object.
For a vision-capable deployment, run a separate `prime --fixture vision` receipt
and its `restore` stage. Restore retains the saved fixture and request. These
bounded checks do not replace serving-speed or general model-quality evaluation.
