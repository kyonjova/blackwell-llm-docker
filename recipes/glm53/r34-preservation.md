# R34 runtime preservation and NVFP4 activation correctness

Status: **qualified for source/artifact preservation and the focused tests below**.
Audit date: 2026-09-11.
The audit compares the published R34 image with the public PR composition in
[vLLM issue #731](https://github.com/local-inference-lab/vllm/issues/731).
It does not claim a complete serving qualification for every model or topology.

Recorded evidence: [installed-file checks](evidence/r34-preservation/artifact-summary.json),
[independent merge residuals](evidence/r34-preservation/source-preservation.json),
[decompressed profile checks](evidence/r34-preservation/profile-preservation.json),
and [runtime metadata classification](evidence/r34-preservation/runtime-metadata.json).

## Conclusion

No required R34 serving implementation was found missing from the composed
image. The composition retains the R34 launchers, cache implementation,
speculation paths and retained B12X tuning, with the pinned JJ/master changes.
It is **not** the same source tree as R34 and must not be described as such.

One intentional mathematical difference is essential: the composition fixes
the missing SwiGLU clamp in R34's split NVFP4 MoE kernel. Preserving R34's
serving functionality does not mean preserving that defect. The correction is
already merged in [B12X #353](https://github.com/local-inference-lab/b12x/pull/353)
and is present in the pinned B12X base; no additional fix PR is required.

## Immutable comparison boundary

| Artifact | Identity |
|---|---|
| Published R34 | `localinferencelab/vllm@sha256:d2d13141fc158f3e5f989930c4be4637eaf28322307288724c2faa7cd4e9bcc7` |
| Composed image, local only | image ID `sha256:d3508d5afd3c616db55efb5d18490e12370fdd0f9c6821f434136e4f4fc21c6a` |
| R34 source-lock SHA-256 | `e7b5712d12676c8daf0a000398cfa2d57eedb3e290cedf611fcee28fe2413dd0` |
| Composition source-lock SHA-256 | `7be25b839e3c34b68e5f54dbec6d12b2cb540277966e49c424e082e57c448e88` |

| Component | R34 commit | Composed commit | Installed tracked files, R34 / composition |
|---|---|---|---:|
| vLLM | `c496604123b1f4441007b952a7ee37ab12c8f6ad` | `de982a50c6a3e4718e5cf9f00423a92192718da1` | 6,835 / 6,870 |
| B12X | `59d51a36a942d56a9c36265855cdc7856fa7712e` | `98086604c86ec1e78977e5023ce282ecb97ab8a7` | 881 / 1,015 |
| LMCache | `29bc5a2efde737c436b04499eb62cd1776cebeec` | same | 2,423 / 2,423 |

Every tracked file and executable mode matches its declared Git tree in both
images: **20,447 checks, zero mismatches**. Both source-lock hashes match the
OCI labels. This checks actual installed files, not just the labels.

## Independent source-preservation check

For each repository, `git merge-tree --write-tree R34_COMMIT PINNED_BASE`
combines the released source and the base from the public review manifest.
The result is compared with the composed commit. This is independent of
replaying the PR manifest against its own expected tree.

| Area | Residual review and result |
|---|---|
| vLLM | Eleven residual paths: native-import isolation from #730; equivalent connector-method and field ordering; documentation/formatting; test maintenance and additional pending-import lane coverage. No R34 runtime implementation is removed. |
| B12X split MoE | Retains route/compute splitting, fast preparation, low-SMEM controls and numerical-recipe propagation. Uses the merged clamp and scratch-sizing helper; removes a duplicated allocation branch rather than removing allocation. |
| B12X MHC | Retains capacity-based partial grouping while preserving upstream's `pre_mix` condition for Gram computation. |
| Embedded GPU policies | Decompressed three-way comparison passes for WS, Max-Q and GB10. Each retains the R34 MHC component and the upstream/unchanged values of every other component; no unresolved policy field. |
| B12X tests/evidence | Uses upstream's split-kernel tests and benchmark evidence. Four Qwen/recipe guard cases pass when replayed from R34. Historical positional fake-operator calls require the installed schemas; the recipe provides 15 keyword/schema-based CPU checks. |
| LMCache | Zero residual differences; complete source tree equals R34. |

The positional fake-dispatch tests are not evidence of a CUDA failure: two
historical calls reject the composed operator's extended argument schema
before execution. The corresponding schema-bound tests pass without changing
the operators or weakening an arithmetic tolerance.

## Runtime and packaging preservation

- Both images have **two filesystem layers** and the exact same runtime-base
  DiffID, `sha256:cb220f94ebc323749ae8e960eecd0076d01b4a23ee7d2dabdb463b44fdc37d68`.
- Launch scripts, entrypoint, command, ports, healthcheck and volume settings
  are identical. Eleven environment differences identify source-derived JIT
  cache namespaces; serving defaults do not change.
- Native library overrides are byte-identical. The shared base and unchanged
  overrides preserve CUDA, PyTorch, FlashInfer, FlashKDA and compatible native
  extensions. No native library was substituted by this composition.
- Outside vLLM/B12X, differences are build-cache files, generated bytecode,
  source-lock metadata and LMCache wheel/build timestamps. The installed
  LMCache native and Python payloads are identical despite a different wheel
  archive hash. The wheel `RECORD` changes only its `uv_cache.json` timestamp
  entry.
- Of the differing bytecode entries, 791 differ only in their header while
  their source is identical; 34 accompany changed/added source. There are no
  unexplained bytecode differences. Generated vLLM version metadata identifies
  the respective source revision.

## SwiGLU correctness proof

Derek Yates (D-Rock) reported that R33's split NVFP4 phase-1 kernel omitted
the model's SwiGLU limit. R34 uses the same B12X revision. GLM's configured
limit is 10: clamp the gate above the limit and the up projection on both
sides before `silu(gate) * up`. Omitting this changes model semantics even
when all outputs remain finite.

An independent test on RTX PRO 6000 Workstation GPU
`GPU-a2f726b2-60db-8102-abef-abf76c457fe0`, SM120, driver 610.57.04, exercises
production fused-MoE dispatch. It uses a tight limit of 2 to guarantee clamping
on the synthetic domain (512 rows, 8 experts, K256, N128, top-k 2, seed 42).
Unclamped and clamped outputs are checked against their respective NVFP4
oracles; the gates require finite output, cosine above 0.9999, BF16 error
bounds and a measurable clamp effect.

| Arm | Result |
|---|---|
| Immutable R34, unmodified runtime | **FAIL:** cosine 0.929721, RMSE 0.365815, maximum absolute error 3.113281 |
| Same R34 with only 13 clamp/parameter-propagation lines in two B12X files | **PASS** |
| Immutable composed image, unmodified runtime | **PASS** |

Raw test receipts: [R34 failure](evidence/r34-preservation/r34-swiglu-gpu.log),
[13-line-only correction](evidence/r34-preservation/r34-minimal-swiglu-gpu.log),
[composed-image clamp](evidence/r34-preservation/composition-swiglu-gpu.log),
[four GPU dispatch cases](evidence/r34-preservation/composition-dispatch-gpu.log),
and [15 CPU contracts](evidence/r34-preservation/composition-contracts-cpu.log).

The corrected B12X merge is `8648fae3bc19c164b46098c1e97882efd8443876`, an
ancestor of the composed revision. The split-materialized path stays enabled;
the fix does not switch to a slower backend or add another kernel launch.

The composed image also passes all four production split-dispatch tests,
including CUDA-graph/scratch reuse and unsupported-domain behavior. Fifteen
CPU graph-interface/quantization-guard checks pass. No model server or GPU
clock configuration was changed for these focused checks.

### Mandatory image gate

Inside the B12X source directory of an image with an assigned test GPU, run:

```bash
/opt/venv/bin/python -m pytest -p no:cacheprovider -q \
  tests/moe/test_nvfp4_split_dispatch_policy.py::TestNvfp4SplitDispatch
```

Mount [the recipe's CPU contract check](../../tests/verify_b12x_release_contracts.py)
read-only and run it with the same image's Python. No model download is needed.
Do not accept a build solely because `swiglu_limit` appears in the source or
because a finite-output smoke test passes.

## Qualification limits

The [matched R34/composition serving measurements](review-qualification.md)
remain the performance evidence: DFlash2 K7 TP4/DCP1 C1, C8 and cold 32K
prefill meet the stated bounded 2% screen. They are not a complete performance
matrix, and R34 is not a valid clamped-math correctness oracle.

The GPU reproducer proves the clamp defect and its correction. It does **not**
prove that every reported 20K–168K runaway has that cause. D-Rock's production-
shaped churn evidence is separately reported evidence, not a soak repeated by
this audit. A field replay or controlled production canary is still needed to
establish causality for those incidents. No claim is made that LMCache itself
caused the incorrect activations.

The published R34 digest remains unchanged and contains the defect. The source
composition is a corrected build input, not a retroactive correction of R34.
