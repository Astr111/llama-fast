---
title: "TA-4: Race condition in cooperative_fwht_128"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/ggml/src/ggml-cuda/triattention-score.cu]
tags: [triattention, walsh-hadamard-transform, concurrency]
---

## Symptom

Potential score corruption / undefined behavior on Volta (sm_70). ([[source-state-md]], §3 TA-4)

## Cause

`cooperative_fwht_128` (`src/ggml/src/ggml-cuda/triattention-score.cu:47`) assumes all 64 warp threads are active: the 7 butterfly stages compute indices from `tid` (lines 52–59) and the normalization writes `smem[tid*2]` and `smem[tid*2+1]` (lines 62–64). If a thread exits with `active=true` but `tid >= 64`, warp divergence or out-of-bounds shared-memory access may occur if the exit conditions are not perfectly synchronized. The function's own header comment (lines 42–44) fixes the contract: "n must be 128, threads = 64 (one butterfly per thread per stage)". ([[source-state-md]], §3 TA-4)

## Impact

Undefined behavior on Volta (sm_70); potential score corruption feeding the eviction decision. Severity in state.md: **MEDIUM**. ([[source-state-md]])

## Location

- Path: `src/ggml/src/ggml-cuda/triattention-score.cu` (verified)
- Symbol: `cooperative_fwht_128` at line 47; normalization writes at lines 62–64; called from `inverse_wht_rotation_128` (line 79).

## Status

**RESOLVED (2026-10-04).** Fixed in `src/ggml/src/ggml-cuda/triattention-score.cu` via commit `5a52561`. Added `bool active` parameter to `cooperative_fwht_128` and `inverse_wht_rotation_128`, guarding memory accesses and butterfly math when warp threads diverge.

## Fix sketch

Audit warp-level primitives at the call sites: ensure either that all threads of the warp participate in every `__syncthreads()`-delimited stage (e.g. loop the mapping so 64 logical threads are always present) or restructure the butterfly so exit conditions cannot diverge between the stage loop and the normalization writes.

## See also

- [[triattention]] · [[walsh-hadamard-transform]] · [[v100-sxm2]]
- [[ta-1-wht-inversion-256]] — same kernel, same WHT-inversion path (`inverse_wht_rotation_128` calls this helper)
- [[tq-4-wht-numerical-mismatch]] — the parallel-vs-sequential WHT discrepancy on the TurboQuant side
