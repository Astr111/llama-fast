---
title: Roadmap
type: topic
status: current
updated: 2026-09-28
sources: [state.md, AGENTS.md]
verified: []
tags: [planning, backlog]
---

# Roadmap

## Bottom line

Six pending action items, inherited verbatim from [[source-state-md]] §5. In dependency order they resolve into three tracks, and the first track is the one that unblocks the measured bottleneck:

| # | Item | Issues | Track |
| :-- | :--- | :--- | :--- |
| 1 | Port the dynamic `wht_group` fix from `Release/` into this repo | [[ta-1-wht-inversion-256]] | correctness — eviction quality |
| 2 | Implement `vec_dot_turbo3_0` and register the types in MMQ/MMVQ dispatch | [[tq-1-missing-gemm-kernels]], [[gemm-dispatch]] | throughput — removes the 38.8 % MAGMA fallback |
| 3 | Move InnerQ host/device state out of `turbo-quant.cuh` | [[tq-2-innerq-host-state]], [[tq-3-innerq-multigpu]], [[tq-6-innerq-race]] | correctness — quantization fidelity |
| 4 | Implement dynamic `min_history_budget` in `triattention_prune_impl` | [[ta-2-budget-starvation]] | correctness — long-prompt behaviour |
| 5 | Batch `triattention_dequant_kv_head` transfers instead of per-cell D2H | [[ta-3-cpu-fallback-transfers]] | latency — CPU fallback only |
| 6 | Unify the two WHT implementations | [[tq-4-wht-numerical-mismatch]] | correctness — future-proofing |

Items 1 and 2 are the current planning focus: they were selected as the first [[overview|initiative]] to run through the repository's delivery pipeline (`AGENTS.md` → *BMad Delivery Pipeline*), because together they address the measured 38.8 % GPU-time consumer and the unported correctness fix that confounds its measurement.

## Evidence

- Items 1–6 as written by the user: [[source-state-md]] §5, each already cross-referenced to its issue page above.
- Item 2's mechanism is [[gemm-dispatch]].
- Item 3 is a three-issue bundle because InnerQ's state problems share one root cause: host/device state living in a header instead of a translation unit or a backend context.
- Item 1's target fix is described as already existing in an external `Release/` checkout. **That checkout was absent from the path state.md named** — `/home/ms/llama-fast/` has no `Release/` directory, checked 2026-09-28 — so item 1 was filed as a port from an unreachable source. **Updated 2026-09-29:** a sibling checkout exists at `/home/ms/llama-fast-dev/Release/src/`, its scoring kernel carries the dynamic `wht_group` fix at lines 224-226, and this repository has no `wht_group` at all. The source is reachable — but that checkout is reported as slated for deletion, so the fix should be captured before it is removed ([[ta-1-wht-inversion-256]] holds the three lines verbatim as a fallback).

- Item 5's payoff, item 6's premise, and item 4's regime all still hold as filed.

### Re-baselined after verification (2026-09-28)

Every premise above was re-read against this checkout while filing the `issues/` pages. Three of them moved:

| Item | Was filed as | What the code shows |
| :--- | :--- | :--- |
| 1 — port the WHT fix | a copy from a known-good checkout | the source checkout is **not on this machine**; the bug is present here ([[ta-1-wht-inversion-256]]) but the fix must be re-derived, not copied |
| 2 — TurboQuant GEMM | the fix for the 38.81 % cuBLAS/MAGMA cost | the `magma_sgemmEx_kernel` attribution has **no counterpart in `src/`** (no MAGMA path; the fallback is cuBLAS with F16 compute type), and the turbo types are **absent from `ggml_cuda_device_supports_op`** as well as from the kernel dispatch — so the scheduling question comes before the kernel question ([[gemm-dispatch]]) |
| 6 — unify the WHTs | a numerical divergence between two implementations | three implementations exist and the two in question agree by construction; `turbo_rotate_forward{,_64}` have no callers ([[tq-4-wht-numerical-mismatch]], [[walsh-hadamard-transform]]) |

The practical consequence for item 2: writing `vec_dot_turbo3_0` is necessary but not obviously sufficient — the allow-list gap has to be closed too, and whether the profitable path is a `mul_mat` kernel or the fused-attention path that already carries native turbo dots (`vec_dot_fattn_vec_KQ_turbo{3,2,4}_0`) is now an open design question rather than a given.

### Re-baselined again (2026-09-28, after the placement verdict)

Item 2 has since lost its premise altogether. [[device-placement]] establishes that the turbo KV types **force flash attention on**, that the fused kernel dequantises the blocks internally, and that **no turbo-typed `MUL_MAT` is built during decode** — so `ggml_cuda_mul_mat` never sees the compressed cache, and the "missing GEMM → cuBLAS/MAGMA → 38.81 %" story that justified item 2 does not run. What survives is narrower and still worth doing, but it is a different task:

| Item 2, as filed | Item 2, re-scoped |
| :--- | :--- |
| Implement `vec_dot_turbo3_0` and register the types so attention stops falling back to cuBLAS | The fallback is not happening in decode. The types' absence from `vecdotq.cuh` and the dispatch tables is a real gap, but it only bites on paths where a turbo `MUL_MAT` is actually constructed — and on those paths the scheduler sends the node to **CPU** ([[ta-3-cpu-fallback-transfers]]), not to cuBLAS |
| Justified by 38.81 % of GPU time | The 38.81 % is unexplained; its profiler symbol exists nowhere in this tree, and the mechanism it implied is refuted. Re-profiling on the target is a prerequisite, not a follow-up |

Consequence for sequencing: **item 2 should not start before (a) [[ta-8-offset-max-zero-nan]] is fixed and (b) a trustworthy profile exists.** Otherwise it is optimisation aimed by an attribution that has been withdrawn.


## Not on the list, but visible in the wiki

Findings that surfaced while filing the sources and are not among the six items — candidates, not commitments:

- **Volta is unbenchmarked** ([[benchmarks]]): there is no V100 number for any profile, and the published CUDA 12.4 bundle carries only `sm_86` ([[v100-sxm2]]).
- **The working tree cannot build as shipped** ([[codebase-map]]): `src/ggml/src/ggml-cuda/template-instances/` is empty while CMake globs it; the 138 files live only in `llama-fast-src.zip`. Nothing else on this list can be measured until that is restored.
- **Calibration may be broken at its entry point** ([[triattention-calibrate]]): `offset_max` defaults to 0, which may make scores NaN, and the README's calibration example may not run at all. `[INFERENCE]` from code, never executed — cheap to confirm, and it sits upstream of items 1 and 4.
- **`freq_scale_sq` is provably 1.0** ([[ta-5-freq-scale-dead-code]]) — unlisted; either dead code or a disabled precision feature.
- **Config validation is absent** ([[ta-7-config-validation]]) — unlisted; the starvation in item 4 would at least be announced to users by it.
- **Overlap double-counting** ([[ta-6-overlap-double-counting]]) — explicitly deferred by the user, recorded here so it is not rediscovered.
- **InnerQ's 128-channel ceiling** ([[tq-7-innerq-max-channels]]) — unlisted; blocks full equalization for the target model's `head_dim=256`.

## Open questions

- Sequencing: item 2 changes the numbers that item 1's fix is meant to be measured against. Doing 1 before re-profiling leaves the same confound as today ([[performance-profile]]).
- Which checkout is authoritative for item 1? [[source-state-md]] §2 names four; the wiki has verified none of them.
- Whether item 5 is worth doing at all if the GPU path is healthy: it is a fallback-only cost, so its priority depends on whether the CPU path is reachable in production.

## See also

[[overview]] · [[performance-profile]] · [[codebase-map]] · [[source-state-md]] · [[source-agents-md]] · [[open-questions]] · [[ta-8-offset-max-zero-nan]] · [[ta-9-rope-scope-mismatch]]
