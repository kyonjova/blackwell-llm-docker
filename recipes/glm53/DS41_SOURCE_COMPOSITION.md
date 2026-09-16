# DeepSeek-V4.1 serving source composition

Status: implemented. GPU release qualification applies only to the immutable
image identified by its published source lock and measurement receipt.

The common image retains GLM, Qwen and DeepSeek integrations. Its DS4.1
entrypoint is `/usr/local/bin/serve-ds41-jovian.sh`; the default entrypoint is GLM.
The build uses one pinned runtime layer and one committed source/dependency
layer, not `FROM` a chain of community releases.

## Review units

| Repository | Base | Additional review units |
|---|---|---|
| vLLM | `dev/jovian-judgement` at `b40673cd` | #740 seeded warmup; #743 owned attention output and bounded indexing; #742 optional prefill capture; #744 official reasoning budgets |
| B12X | `master` at `05d2c43b` | #364 host metadata; #367 projection kernels and profiles; #366 visible score tiles; #365 mHC tiles |
| LMCache | `29bc5a2efde737c436b04499eb62cd1776cebeec` | No additional source delta |

Both source repositories publish `release/ds41-optimized-r37` as the composed
build ref. These refs contain the review commits without merging them into JJ
or master. The vLLM #742/#743 CED test additions share an insertion point;
the composition retains both additions. Every optimization runtime file was
compared byte-for-byte with its GPU-qualified source before adding the
separate reasoning-budget correction.

B12X base `05d2c43b` includes #363's small-tile input-pair barrier correction
and the TP3 prefill-head fix. TP3 is not implied to be serving-qualified by
including that source.

## Python dependency corrections

`dependency-python-patches.json` pins every input and output SHA-256. The
installer verifies all inputs and patched outputs before changing installed
files. A different dependency revision fails the build. Native CUDA, Torch,
FlashInfer, NCCL and LMCache binaries are not rebuilt or replaced by these patches.
The dependency manifest also participates in the JIT-cache fingerprint.

- `torch-schema-enumeration.patch` is equivalent to yingru's existing
  [PyTorch #195110](https://github.com/pytorch/pytorch/pull/195110): linear-time
  schema expansion. No duplicate upstream PR is created.
- `torch-mutable-argument-metadata.patch` retains immutable schema metadata at
  custom-op registration, preserving live mutable tensors and version counters.
- `cutlass-sentinel-identity.patch` detects missing arguments by identity,
  without invoking equality on tensor or DSL operands.

The latter two corrections were prepared by Martin Vit with OpenAI Codex
assistance. Upstream PyTorch submission requires the human review prescribed
by that repository; this recipe is the reviewed deployment unit meanwhile.

CPU parity covered sixteen mutation/output/version cases and four public
`torch.library.opcheck` suites. CuTe binding covered eight ordinary result/error
cases and an opaque operand whose equality must not run. Fifty projection GPU
tests passed with the identity binder. These are operation contracts, not a
general model-quality claim.

## Launcher controls and compatibility

`ENGRAM_TABLE_MEMORY=ram|disk` selects host-resident complete ngram tables or
native SSD row reads. The default is disk. This is unrelated to KV or LMCache.
Serving defaults to TP4/DCP1, DSpark K7, batch 4096, and `0.0.0.0:8000`.

`VLLM_USE_BREAKABLE_CUDAGRAPH=0` is the default: ordinary decode graphs remain
enabled, while 4096-token prefill runs eager. Set it to 1 for optional prefill
capture. Paired source qualification measured about +1.3% prefill throughput
at a roughly 11% KV-capacity cost; this is not a capacity-neutral default.

The DS4.1 wrapper preserves general `VLLM_*` and `B12X_*` settings, including
`VLLM_LOGGING_LEVEL`. It removes only the explicitly listed values inherited
from the image's GLM tuning. Different-value overrides remain intact. Docker
does not identify whether a caller explicitly supplied a value identical to
an image default, so listed exact values are treated as inherited. The native
DS4.1 script separately owns its documented architecture/thread/communication
settings. This wrapper does not promise to override that native contract.

Named reasoning budgets follow the publisher: low=50, high=75, max=100; high is
the default. The accepted xhigh alias remains 75. Explicit numeric budgets in
`chat_template_kwargs` remain available, including 50 for matched comparisons
against releases whose default high rendered 50. Do not attribute changed
reasoning length or acceptance to kernel execution speed.
