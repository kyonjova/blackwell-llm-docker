# GLM Spark TP2 external checkpoint cache

Status: qualified for the configuration and synthetic text fixture below.

The GLM runtime profile accepts TP2 engine-driven, request-boundary checkpoint
transfer. CPU memory and disk hold complete target, recurrent and speculative
checkpoint bundles. The profile rejects TP2 aligned/direct transfer; TP4 and
TP8 behavior is unchanged.

## Conditions

- Target: `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`.
- TP2, DCP2, MTP3 with probabilistic drafting and standard rejection.
- Two RTX PRO 6000 Blackwell Max-Q GPUs; CUDA 13.4.1 and PyTorch 2.14.
- FP8 attention KV, B12X attention/MoE/linear/KDA prefill, full-and-piecewise graphs.
- 3,072-token target scheduling and cache-object budget; four sequence slots.
- 3,996 MiB GPU KV per rank; 12 MiB allocator large segments.
- CPU cache: 16 GiB, initially 2 GiB; persistent disk capacity: 64 GiB.
- Temperature 1, top-p 0.95, profile-default reasoning, unique cache salt.
- GPU-prefix reset before each external restore; no other requests on the endpoint.

## Whole-model results

| Request | GPU-hit tokens | External-hit tokens | Disk objects loaded | Answer |
|---|---:|---:|---:|---|
| Cold lookup | 0 | 0 | 0 | COBALT |
| Identical GPU-prefix hit | 16,287 | 0 | 0 | COBALT |
| Identical lookup after GPU reset | 0 | 16,287 | 0 | COBALT |
| Changed question after GPU reset | 0 | 16,281 | 0 | AMBER |
| Changed question after serving and cache restart | 0 | 16,287 | 28 | AMBER |

Both process identities changed for the disk test; the image and persistent
volume were retained. The restored server uses the native NVIDIA 615.71.09
driver libraries, without a compatibility-library override. The measured cache
configuration reports a 924,672-token API limit. These are functional cache
checks, not throughput measurements or a full-context memory-stress qualification.

The validation command is
`tools/jovian_wheel_runtime/qualify_model_cache.py prime|restore` with a dedicated
endpoint and the cache service's `/metrics` URL. All five stages are qualified.
The runtime and container-contract suites pass 438 tests; they also exercise
TP2/DCP1 configuration resolution and rejection of unsupported transfer modes.
Whole-model TP2/DCP1 cache behavior is not inferred from those CPU tests.

## Artifact identities

The tested image configuration ID is
`sha256:23e4538cfdc18a7c63fa8763b71bddfe4519bb48d5a4e67e1acf2ddbbfcbed8f`.
Its vLLM source tree is identical to integration commit
`bd76814003bd83d35d3092aa95aa29cfcd967ec3`; B12X is
`eea3ced11fc14625b683d3c575cf2d702ad3706a` and LMCache is
`688bee14e157b64623d93c07fc0d4db93470e12f`.

The fixture/metrics receipts are retained with these SHA-256 identities:

- `prime.json`: `91160349f6511fe53a0b5f73edc732efdc22d31044ff72a92487699fc480521c`
- `restore.json`: `cefe0ccd8997aaa2cf0475767a4552cecf28b90ee2fda8c7a722dea2a38f3e3c`

## Compatibility limits

GLM image-bearing requests still use the native vision path, but external
recurrent checkpoint reuse for them is unsupported and recomputes. External
cache geometry and transfer buffers can reduce the auto-fit context limit
relative to VRAM-only serving. Public-image qualification remains a separate
release check; these results do not identify an untested registry digest.
