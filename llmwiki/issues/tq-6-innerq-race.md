---
title: "TQ-6: InnerQ calibration counter keyed on threadIdx.x"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/ggml/src/ggml-cuda/set-rows.cu, src/ggml/src/ggml-cuda/turbo-quant.cuh]
tags: [innerq, turboquant, correctness]
---

## Symptom

The InnerQ calibration token count can over-count or under-count if the block dimensions change. ([[source-state-md]], §4 TQ-6)

## Cause

**Verified.** In `k_set_rows_turbo3()` (`src/ggml/src/ggml-cuda/set-rows.cu:237`, template parameter `GROUP_SIZE`, `__launch_bounds__(128)` at line 236) the thread index `const int j = threadIdx.x;` (line 259) doubles as the within-group element index, and the calibration block at `set-rows.cu:288-295` increments the counter exactly once per group through whichever thread happens to be lane 0:

```c
if (d_innerq_calibrating) {
    atomicAdd(&d_innerq_sq_accum[j], x[j] * x[j]);
    if (j == 0) atomicAdd(&d_innerq_count, 1);
}

if (d_innerq_active) {
    x[j] *= d_innerq_scale[j];
}
```

The mapping is one block per rotation group (`const int64_t g = blockIdx.x;` decodes to `(i_grp, i01, i02, i03)` at lines 261-269) and `blockDim.x == GROUP_SIZE` at both launch sites (`set-rows.cu:575` with 128 threads, `:581` with 64), so "thread 0" and "one thread per group" coincide today. The identity is an unstated invariant: any launch where `blockDim.x > GROUP_SIZE`, or where one block covers several groups, breaks it — `j == 0` would then be either one thread too few or a lane whose `x[j]` is not the first element of a full group. The same construct is repeated in `k_set_rows_turbo2()` (`set-rows.cu:659`) and `k_set_rows_turbo4()` (`set-rows.cu:1002`), and the tail kernels do not touch the counter at all (see [[tq-5-tail-elements]]).

`d_innerq_count` is compared against `innerq_target_tokens` in `turbo_innerq_check_finalize()` (`turbo-quant.cuh:280-287`) and divided into the accumulated squares in `turbo_innerq_finalize()` (`turbo-quant.cuh:193-198`, `:211-215`), so a miscount corrupts the RMS estimate directly.

## Impact

Wrong RMS estimate → wrong per-channel `scale`/`scale_inv` (`scale[i] = pow(mean_rms / rms[i], strength)` clamped to `[0.5, 2.0]`, `turbo-quant.cuh:217-226`) or premature/late finalization. Not exploitable by user input today; it is a fragility that turns a future block-shape change into silent accuracy loss. Severity in state.md: **MEDIUM**.

## Location

- Path: `src/ggml/src/ggml-cuda/set-rows.cu` (verified) — `k_set_rows_turbo3()` at 237; `j = threadIdx.x` 259; group decode 261-269; calibration block 288-295 with `if (j == 0) atomicAdd(&d_innerq_count, 1);` at 290; launches 575/581 (128/64 threads); same pattern at 657-664 (turbo2) and 1000-1007 (turbo4).
- Path: `src/ggml/src/ggml-cuda/turbo-quant.cuh` (verified) — counter consumers at 193-198 and 280-287; scale formula 217-226.

## Status

Unresolved. Not listed among the §5 action items of [[source-state-md]] (it is covered implicitly by the TQ-2/TQ-3 state refactor).

## Fix sketch

Per state.md §4 TQ-6: replace `if (j == 0)` with an explicit `if (threadIdx.x == 0)` guard (and keep `atomicAdd(&d_innerq_sq_accum[j], …)` keyed on the element index), in all three kernels. If a future launch ever packs several groups per block, the counter must instead be incremented once per group — e.g. `if (j % GROUP_SIZE == 0)` with a per-group base index — so the invariant is expressed rather than assumed.

## See also

- [[innerq]] · [[turboquant]] · [[quantization]]
- [[tq-2-innerq-host-state]] · [[tq-3-innerq-multigpu]] · [[tq-7-innerq-max-channels]]
- [[tq-5-tail-elements]] — the tail kernels that deliberately skip this counter
