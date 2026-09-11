# Qualification of the public Jovian source composition

Status: **qualified for the bounded checks below**, measured 2026-09-11.
This is not a DockerHub release or a full GLM/Qwen/DeepSeek qualification.

The [review manifests](review-composition.md) reproduce complete source trees
from public GitHub refs with no additional patches. Network fetch, ordered
merge and generated-patch replay passed for 32 vLLM, 3 B12X and 9 LMCache
review units. All 44 were open and non-draft at the audit.

## Attributed source mirrors

| Component | Complete Git source | Tested commit |
|---|---|---|
| vLLM | [voipmonitor/vllm](https://github.com/voipmonitor/vllm/tree/integration/jovian-reviewed-sources-20260911) | `de982a50c6a3e4718e5cf9f00423a92192718da1` |
| B12X | [voipmonitor/b12x](https://github.com/voipmonitor/b12x/tree/integration/jovian-reviewed-sources-20260911) | `98086604c86ec1e78977e5023ce282ecb97ab8a7` |
| LMCache | [local-inference-lab/LMCache](https://github.com/local-inference-lab/LMCache/tree/release/jovian-fp4-fs-ledger-r33-20260910) | `29bc5a2efde737c436b04499eb62cd1776cebeec` |

The candidate has two filesystem layers and source-lock SHA-256
`7be25b839e3c34b68e5f54dbec6d12b2cb540277966e49c424e082e57c448e88`.
Its local image ID and the public R34 digest are recorded with
[measured cells, prefill samples and prefix checks](review-qualification.json).
CUDA, FlashInfer and compatible native artifacts are unchanged. Serving used
no source bind mounts.

## Matched serving comparison

Conditions: the same physical RTX PRO 6000 Workstation GPUs 4–7, TP4/DCP1,
VRAM offset **+6000 MHz**, graphics offset 0, 600 W limit. These are not stock
measurements. Both images use the same target and MXFP8 draft snapshots,
DFlash2 K7 probabilistic/standard sampling, temperature 1, top-p 0.95,
FP8 target KV, B12X MoE/attention/all-reduce, FlashKDA recurrent prefill,
V2 runner, 4096-token scheduler budget, OMP1, NCCL16/2MiB and
`FULL_AND_PIECEWISE` graphs.

| Metric | Published R34 | Public review composition | Change |
|---|---:|---:|---:|
| C1 output tok/s | 254.85 | 255.05 | +0.08% |
| C1 verifier steps/s | 97.37 | 97.71 | +0.34% |
| C8 aggregate output tok/s | 803.70 | 792.84 | −1.35% |
| C8 aggregate verifier steps/s | 308.74 | 310.74 | +0.65% |
| Cold 32K prefill tok/s | 16,872 | 16,694 | −1.05% |

Decode uses llm-decode-bench 0.4.29, zero initial context, a 15-second warmup,
one 30-second cell per concurrency and an 8192-output-token request limit.
Both arms have zero API errors. C8 emitted tokens per verifier step change
2.603 → 2.552; output throughput alone is not a pure execution-speed metric.

Prefill measures exact 32,768-token, one-output-token requests for 30 seconds
after warmup. All 17 reference and 16 candidate samples report 32,768 locally
computed tokens and no cache or external-transfer tokens. Reference range:
16,423–16,956 tok/s; candidate: 16,651–16,768.

Conclusion: these observations meet a bounded 2% screening threshold. One
decode cell per arm does not establish statistical equivalence, a general
speedup or performance at other modes/concurrencies.

## Correctness evidence and limits

- Final vLLM composition: 393 CPU tests pass for fairness, boundary admission,
  sparse cleanup, connector configuration and native-import isolation.
  Recipe/composer suites: 199 passed.
- Composed GLM indexer/checkpoint GPU suites: 73 passed, including restore
  byte offsets above 2 GiB.
- B12X policy/scratch/scale/alias: 70 passed. Exact GLM M8 graph oracle and
  eight MHC specialization cases pass. Six two-GPU PCIe PDL torture cases
  pass, including epoch rollover and multiple streams.
- LMCache filesystem/native-adapter tests: 116 passed. Its complete source
  tree is identical to R34, not merely patch-equivalent.
- Packaged DFlash: three identical 8192-token requests restore exactly 8192
  tokens and emit identical greedy token IDs. Response continuation matches
  an independently cold control. Shared SYSTEM reuse, C4 document lookups,
  assistant continuation, tool history and changed instruction values pass.

The fused B12X MHC post/pre oracle has the same two-element BF16 discrepancy
on master, R34 and the composed source: absolute error 0.0078125 versus its
0.004 tolerance. No tolerance was weakened. This discrepancy is not fixed or
introduced by the composition.

No fresh no-spec/MTP3 performance matrix, DCP4 external-cache restart matrix,
TP8, NVFP4 target-KV, Qwen or DeepSeek serving qualification is claimed.
The strict-JSON/concurrent-MTP/LMCache issue tracked by vLLM #726 remains open.
