# Public review composition for Jovian serving

Status: **implemented; exact source-tree composition qualified**. The bounded
[GPU serving evidence and source mirrors](review-qualification.md) are recorded
separately; serving correctness does not follow merely from a successful merge.

The manifests in this directory describe complete vLLM, B12X and LMCache source
trees as a pinned base plus ordered public pull-request heads. They contain no
`source_patches` or private conflict-resolution input. A review head owns every
required resolution. The resulting tree is checked against the manifest's
`expected_tree` before an integration artifact is accepted.

| Component | Base branch | Base commit | Manifest |
|---|---|---|---|
| vLLM | `dev/jovian-judgement` | `a6571d0a602ed42525a7a77cf960f39ca7f8d7a8` | [32 review units](review-stack.vllm.json) |
| B12X | `master` | `00b69ac22e21413622c4ecd98f607a2c3e015161` | [3 review units](review-stack.b12x.json) |
| LMCache | `dev` | `7ed4675404a31f4ffafd98975899dc83832ba965` | [9 review units](review-stack.lmcache.json) |

## Verification

Run from the repository root:

```bash
uv run --no-project scripts/compose_vllm_release.py \
  recipes/glm53/review-stack.vllm.json --output-dir ./review-vllm
uv run --no-project scripts/compose_vllm_release.py \
  recipes/glm53/review-stack.b12x.json --output-dir ./review-b12x
uv run --no-project scripts/compose_vllm_release.py \
  recipes/glm53/review-stack.lmcache.json --output-dir ./review-lmcache
```

The composer fetches public GitHub PR refs, rejects moved heads or unresolved
dependencies, merges in the declared order and verifies the complete Git tree.
It emits a generated `integration.patch` and `integration.lock.json`; the patch
is an output, not an extra integration input. Patch replay is checked against
the same tree. Embedded `.patch` files retain meaningful unified-diff context
spaces; ordinary source files remain subject to whitespace validation.

For a checkout whose object database already contains the pinned commits, the
offline check does not edit its branch, index or working files:

```bash
uv run --no-project scripts/compose_vllm_release.py \
  recipes/glm53/review-stack.vllm.json --verify-in ./vllm-source
```

An offline proof checks composition only, not publication. Use the network
verification when accepting public review heads. The declared order is a
supported composition contract, not a claim that all subsets or merge orders
are valid.

## Review ownership

- vLLM #664 owns scheduling, checkpoint-readiness polling and prefill-lane
  release. Apply #664 and #553 before #709's atomic checkpoint import.
- vLLM #729 preserves logprobz's local DCP1 checkpoint admission implementation
  and supersedes #721. Apply #718 first. The local admission feature remains
  disabled for DCP4 and external connectors.
- vLLM #708 retains packed recurrent export and the checkpoint-restore test
  above a 2 GiB byte offset; #676 remains the separate partial-prefix proof.
- B12X #356 retains MadeBy561's capacity-based MHC policy. It is not the
  source-split scheduling proposal in #310. The policy preserves the master
  schemas and non-MHC profile components.
- LMCache #49 incorporates Derek Yates's filesystem ledger contribution from
  #67 together with bounded object names. Do not apply #67 separately.
- B12X #353/#354 and the behavior reviewed in #317 are represented in the
  pinned master source. vLLM #727 is represented in pinned JJ. They are not
  additional merge inputs; inclusion alone does not imply that GitHub marked
  each original PR merged.

Merge commits preserve contributor history. Source bundles must be built from
the complete attributed Git composition, not a squashed filesystem copy. The
two-layer Docker recipe authenticates the resulting source trees and compatible
native binaries independently of the PR titles.

## Compatibility boundary

The serving composition retains the R34 cache, speculative decoding, model
launchers, sampling defaults and B12X deployment default while incorporating
the declared JJ/master additions. LMCache's composed tree is exactly the R34
tree. No vLLM native ABI or CUDA runtime change is required by these JJ merges.

vLLM #730 isolates quantization configuration discovery from model-specific
native layer imports. JJ's standalone B12X imports are preserved; actual
DeepSeek V4.1 dispatch still requires its native backend. Import isolation is
not a fallback implementation or a DeepSeek V4.1 serving qualification.

The published R34 image and its source refs in [the build guide](README.md) are
immutable reference artifacts. These review manifests identify a distinct
composition. They do not change the contents of an already-published image.
