# Serving dependency corrections

Status: **implemented** in the CUDA 13.4 container recipe. The build applies
hash-locked Python corrections and verifies the imported files. These checks
do not imply full-model correctness or throughput qualification.

## PyTorch and CuTe contract

| Required behavior | CUDA 13.3 / PyTorch 2.13 recipe | CUDA 13.4 / NGC PyTorch 2.14 recipe |
|---|---|---|
| Materialize operator schema arguments once during default expansion | `torch-schema-enumeration.patch` | Same hash-locked patch |
| Track mutable custom-op inputs without expanding the complete schema per call | `torch-mutable-argument-metadata.patch` | Native direct mutation tracking; do not apply the 2.13 patch |
| Identify an unbound CuTe argument by sentinel identity, without invoking tensor equality | `cutlass-sentinel-identity.patch` | Same patch for both the system and venv CuTe copies |

The patch manifests are
[`recipes/glm53/dependency-python-patches.json`](../recipes/glm53/dependency-python-patches.json)
for the CUDA 13.3 recipe and
[`tools/jovian_wheel_runtime/ngc-python-patches.json`](../tools/jovian_wheel_runtime/ngc-python-patches.json)
plus [`ngc-venv-python-patches.json`](../tools/jovian_wheel_runtime/ngc-venv-python-patches.json)
for CUDA 13.4. Each application requires the exact input hash and checks the
result hash. A changed upstream file fails the build; it is not patched with
fuzz or silently accepted. The manifests retain contributor attribution.

The NGC 2.14 `CustomOpDef._register_mutation_version_bump` implementation visits
only mutated argument indices and keys. Dispatcher-normalized arguments are
passed directly to version tracking. It has no per-call `fill_defaults` call.
This satisfies the hot-path purpose of the 2.13 backport without requiring
byte-identical implementations across PyTorch versions.

`verify_torch_operator_contract.py` executes seven CPU mutation cases with
positional/named inputs, a tensor list, and supplied/omitted optional tensors.
It checks values, version increments, zero dispatch-time schema expansions,
and one schema-argument materialization when default expansion is requested.
It runs in a disposable build/verification process, not a serving worker.
The container verifier also checks the exact imported dependency hashes.

## Qualification evidence

On 2026-09-16, the CPU contract passed on both the R38 PyTorch 2.13 image
(`5ae584725299` image-ID prefix) and the locally available CUDA 13.4 beta image
(`55350bb5fff8` image-ID prefix). Each reported seven passing mutation cases,
zero dispatch-time schema expansions and one schema-argument materialization.
No GPU, model request or serving-process mutation was involved. This is a
comparison of those local immutable images, not a claim about a moving tag.

## Build ownership

Corrections belong to the shared runtime build, before any model is selected.
Model profiles never patch site-packages at startup. A profile-only change
does not rebuild PyTorch or native component wheels. The recipe/source identity
includes the patch manifests and verifier; their edits trigger container
assembly, not component recompilation.

Changing the NGC foundation requires inspection of all three operations,
updated file-hash contracts, CPU semantic tests and affected model validation.
A wheel installation alone does not apply a patch to NGC's system PyTorch;
the image recipe is the installation boundary for this foundation.
