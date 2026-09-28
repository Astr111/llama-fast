---
title: Pending decisions
type: topic
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/common/common.h, src/src/llama-triattention.cpp, src/ggml/src/ggml-cuda/CMakeLists.txt, llama-fast-src.zip]
tags: [planning, decisions]
---

# Pending decisions

[[open-questions]] classifies the vault's unresolved items by *what it would take to close them* and finds roughly ten that are not research at all — they are choices. This page states them as choices: the options, what each one costs, and a recommendation with its reason. Nothing here is a finding; every claim behind a recommendation lives on the linked page.

Nothing on this page requires a build, a run, or the V100 — which is why it is the highest-value work available on the current machine.

---

## D1. What do we do about the two confirmed scoring defects before anything is measured again?

**Why it is first.** [[ta-8-offset-max-zero-nan]] makes every eviction score NaN in the shipped configuration; [[ta-9-rope-scope-mismatch]] applies the wrong inverse rotation with a frequency-dependent error; [[ta-1-wht-inversion-256]] skips the inversion entirely at `head_dim=256`; [[ta-2-budget-starvation]] starves the budget to zero on long prompts. Four independent mechanisms corrupt *which keys survive*, and all four are live in the documented profile. Any measurement of eviction quality taken now measures their sum.

| Option | Cost | Consequence |
| :--- | :--- | :--- |
| **A. Fix TA-8 only, then re-measure** | two lines: guard `n_offsets == 0`, give `offset_max` the doc's 65536 | Scores become defined again; the other three remain. Cheapest real improvement in the project |
| **B. Fix TA-8 + TA-9, then re-measure** | A is small; TA-9 needs the inverse parameterised by `n_rot` and the MRoPE sections, cross-checked against the design doc | Eviction decisions become defensible; needs [[source-triattention]] as the reference, and the partial-RoPE layout makes it non-trivial |
| **C. Port the `Release/` WHT fix first** | unavailable — that checkout is **not on this machine** ([[ta-1-wht-inversion-256]]) | Blocked |
| **D. Measure anyway, note the confounds** | free | Produces numbers nobody can attribute; the vault already has one set of these ([[performance-profile]]) |

**Recommendation: A now, B as the follow-on, and treat TA-2 (roadmap item 4) as part of the same change.** Reason: A is the only option that makes *any* subsequent number interpretable while costing almost nothing, and it is the one defect of the four whose fix is a guard rather than a design.

## D2. Which Walsh-Hadamard implementation is canonical?

**Why it matters.** Five exist: `set-rows.cu`'s encode butterfly, `turbo-wht.cu`'s `k_turbo_wht_f32` (the only one reachable as a ggml op), `turbo-quant.cuh`'s sequential `turbo_fwht_128` (no callers), the CUDA `cooperative_fwht_128` in the scoring kernel, and `fwht.cu`'s hint-driven side path. Roadmap item 6 says "unify" without naming a winner ([[walsh-hadamard-transform]], [[turbo-wht]]).

| Option | Cost | Consequence |
| :--- | :--- | :--- |
| **A. `k_turbo_wht_f32` is canonical; others must match it** | documentation plus deleting the dead sequential pair | The op is the only one carrying the InnerQ placement contract in comments; a single stated contract is what downstream work needs |
| **B. The encoder butterfly is canonical** | documentation | It is the hot path, but it is not an op — nothing outside `set-rows.cu` can be held to it |
| **C. Each is canonical in its own domain** | documentation only | Honest but weak: three implementations of one transform with no oracle is how [[tq-4-wht-numerical-mismatch]] arose |

**Recommendation: A.** Reason: a contract has to live somewhere an outside reader can reach, and the op is that place; the butterfly then becomes an optimisation obligated to match the op, not a second source of truth.

## D3. Does a TurboQuant KV type get a case in `test-backend-ops`?

**Why it matters.** Zero `turbo` matches in `src/tests/test-backend-ops.cpp`; the registered custom tests cover the *weight* types only ([[build-and-verify]], [[quantized-kernel-units]]). A regression in the KV path today would surface only end-to-end, on hardware nobody has measured.

| Option | Cost | Consequence |
| :--- | :--- | :--- |
| **A. Add the case now** | blocked — nothing can be compiled here | — |
| **B. Commit to adding it with the first tree that builds** | a decision, recorded | Turns "no coverage" into "coverage pending environment", which is a plan rather than a gap |
| **C. Accept end-to-end-only coverage** | free | Every KV regression costs a full agent-run benchmark to find |

**Recommendation: B**, with the note that it should be a CPU-only case if possible — the CPU path is readable and runnable anywhere, and it is the reference semantics for the format ([[cpu-path]]).

## D4. Do we restore `template-instances/` into the working tree?

**Why it matters.** The directory is empty while CMake globs it, and the 138 files plus `generate_cu_files.py` exist in `llama-fast-src.zip`. This is either deliberate publication hygiene or an incomplete copy — the vault cannot tell which from the tree alone ([[codebase-map]]).

| Option | Cost | Consequence |
| :--- | :--- | :--- |
| **A. Restore from the archive and commit** | one `unzip -n` line plus a CMake re-run | The tree becomes buildable in principle; the diff to the published repo becomes large and must be explained |
| **B. Leave it, document the restore command** | free | Publication hygiene preserved; every future builder hits the same wall — but now with the exact command in hand ([[build-and-verify]]) |
| **C. Restore into a scratch directory only** | free | Builders work; the published tree stays as-is |

**Recommendation: B, with C available.** Reason: it is a publication decision, not a technical one — and the exact command is already recorded, so the cost of B is one line in the build page.

## D5. What happens to `turbo-rotation-data-32.h` and the dead `PTQ1_0` id?

**Why it matters.** The `-32` variant is **included by nothing** in this checkout ([[rotation-data]]); `GGML_TYPE_PTQ1_0` exists, is declared, and is used by no artifact ([[conversion-and-packing]]). Both are the kind of thing that gets "cleaned up" by someone who does not know why it is there.

| Option | Cost | Consequence |
| :--- | :--- | :--- |
| **A. Delete both** | free here | An out-of-tree checkout or the external packer may depend on either — neither can be checked from this repo |
| **B. Keep both, document them as externally-consumed** | one paragraph each | Survives the cleanup; the unknown stays unknown |
| **C. Investigate before deciding** | needs the checkouts / the packer | Blocked |

**Recommendation: B.** Reason: the vault has a rule for exactly this case — an artifact whose consumer is outside the repository is documented, not deleted. Deleting on the strength of "we found no caller here" is the same reasoning that would have deleted the rotation tables.

## D6. Is the README's benchmark table worth re-stating, or should it be retired?

**Why it matters.** Every published figure is Ampere; the target is Volta; two of the three profiles differ by ~3 % end-to-end while differing ~25 % in VRAM, and [[kv-accounting]] shows the tokens-per-GB column is consistent with a *dense* 64-layer model rather than this model's 16 KV-bearing layers.

| Option | Cost | Consequence |
| :--- | :--- | :--- |
| **A. Re-measure on the target** | needs the V100 | The only option that produces a number worth publishing |
| **B. Annotate the README table as Ampere-only** | one line | Readers stop assuming the numbers describe the deployment target; the vault already carries the analysis ([[benchmarks]]) |
| **C. Leave as-is** | free | The table stays the project's most-quoted and least-applicable artifact |

**Recommendation: B until A is possible.** Reason: it costs one line and removes the single most misleading claim in the repository's public surface.

---

## What is *not* a pending decision

- **The vol/eviction defects in [[open-questions]] class C/D** — those need hardware or artifacts, not choices. Recording them as decisions would be a way of not doing them.
- **Anything the pipeline would decide** — the BMad architecture stage was paused at the user's instruction; the six roadmap items are its input, not this page's.

## See also

[[open-questions]] · [[roadmap]] · [[overview]] · [[scoring-correctness]] · [[build-and-verify]] · [[walsh-hadamard-transform]] · [[cpu-path]] · [[kv-accounting]] · [[benchmarks]]
