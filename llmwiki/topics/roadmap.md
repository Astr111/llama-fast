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
- Item 2's payoff estimate is [[performance-profile]]'s 38.81 % / 157 ms MAGMA share; item 2's mechanism is [[gemm-dispatch]].
- Item 1's target fix is described as already existing in an external `Release/` checkout — `[UNVERIFIED]` from inside this repository; porting it requires that checkout to be reachable.
- Item 3 is a three-issue bundle because InnerQ's state problems share one root cause: host/device state living in a header instead of a translation unit or a backend context.

## Not on the list, but visible in the wiki

Findings that surfaced while filing the sources and are not among the six items — candidates, not commitments:

- **Volta is unbenchmarked** ([[benchmarks]]): there is no V100 number for any profile, and the source build example omits `sm_70`.
- **`freq_scale_sq` is provably 1.0** ([[ta-5-freq-scale-dead-code]]) — unlisted; either dead code or a disabled precision feature.
- **Config validation is absent** ([[ta-7-config-validation]]) — unlisted; the starvation in item 4 would at least be announced to users by it.
- **Overlap double-counting** ([[ta-6-overlap-double-counting]]) — explicitly deferred by the user, recorded here so it is not rediscovered.
- **InnerQ's 128-channel ceiling** ([[tq-7-innerq-max-channels]]) — unlisted; blocks full equalization for the target model's `head_dim=256`.

## Open questions

- Sequencing: item 2 changes the numbers that item 1's fix is meant to be measured against. Doing 1 before re-profiling leaves the same confound as today ([[performance-profile]]).
- Which checkout is authoritative for item 1? [[source-state-md]] §2 names four; the wiki has verified none of them.
- Whether item 5 is worth doing at all if the GPU path is healthy: it is a fallback-only cost, so its priority depends on whether the CPU path is reachable in production.

## See also

[[overview]] · [[performance-profile]] · [[codebase-map]] · [[source-state-md]] · [[source-agents-md]]
