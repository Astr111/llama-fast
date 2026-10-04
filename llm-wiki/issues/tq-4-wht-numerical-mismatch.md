---
title: "TQ-4: Sequential vs parallel WHT — three copies, no numerical mismatch demonstrated"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/ggml/src/ggml-cuda/turbo-quant.cuh, src/ggml/src/ggml-cuda/set-rows.cu, src/ggml/src/ggml-cuda/turbo-wht.cu, src/ggml/src/ggml-cuda/ggml-cuda.cu, src/src/llama-graph.cpp]
tags: [turboquant, walsh-hadamard-transform, innerq, correctness]
---

## Symptom

state.md warns that the sequential single-thread `turbo_fwht_128()` and the parallel shared-memory butterfly in `set-rows.cu` reorder floating-point operations, so any use of the sequential variant outside `set-rows.cu` would numerically mismatch the encoding process and silently lose accuracy. ([[source-state-md]], §4 TQ-4)

## Cause

**Verified facts.** `turbo_fwht_128()` (`src/ggml/src/ggml-cuda/turbo-quant.cuh:88-106`) is a plain triple loop `for (h = 1; h < 128; h *= 2)` over element pairs, writing `x[j] = a + b; x[j + h] = a - b`, then scaling by `inv_sqrt_128 = 0.08838834764831845f` (lines 100-103). Its 64-wide sibling `turbo_fwht_64()` is at lines 108-125 (normalizing by `0.125f`), and `turbo_rotate_forward()` / `turbo_rotate_forward_64()` (`:127-131`, `:135-139`) wrap them as signs1 → FWHT → signs2.

`set-rows.cu` does the same transform in shared memory inside `k_set_rows_turbo3()`: `WHT_STAGE_SHARED(h)` defined at line 332 and invoked for h = 1, 2, 4, 8, 16, 32 (and 64 for `GROUP_SIZE == 128`) at lines 337-343, with the same `a + b` / `a - b` writes and a `__syncthreads()` per stage; the normalization `x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2[j]` is at lines 346-351 with `inv_sqrt_group = 0.08838834764831845f` for 128 and `0.125f` for 64.

**Correction.** The claimed "floating-point operation ordering differs" is not substantiated: both implementations run the same radix-2 stages in the same order and touch each element with the same sequence of operations (`a+b`, `a-b`, then the same normalization constant, then the same sign array), so thread-level parallelism changes nothing about per-element operation order. The 64-wide paths use the same constants on both sides (line 346 vs `turbo-quant.cuh:119-121`).

What *is* verified: (a) `turbo_rotate_forward()` and `turbo_rotate_forward_64()` have **no callers anywhere under `src/`** — `turbo_fwht_128()`/`turbo_fwht_64()` are reached only from those two dead wrappers, so that implementation is obsolete dead code and the stated impact is latent by construction. (b) The tree carries **three** implementations of the same transform, of which two are live: the dead pair above; the encode-side butterfly in `set-rows.cu:324-351` (live — it runs on every KV write); and `k_turbo_wht_f32<direction, group_size>` in `turbo-wht.cu:31-96`, which is the **real rotation op**, reached from the graph via `ggml_turbo_wht()` (`src/src/llama-graph.cpp:2707`, `:2785`, `:2985`, `:3108`, `:3299`) → `GGML_OP_TURBO_WHT` (`ggml-cuda.cu:2092-2093`) → `ggml_cuda_turbo_wht()` (`turbo-wht.cu:116-174`). So the correct framing is "one obsolete copy, one live encoder, one live rotation op", not "two variants of the same file". The live pair also owns the InnerQ scale placement — forward applies `scale_inv` *before* signs+WHT (`turbo-wht.cu:49-53`) while the inverse applies it *after* (`:90-93`) — which is the property that must stay consistent with `set_rows`' encode-side `x[j] *= d_innerq_scale[j]` (`set-rows.cu:294-295`).

## Impact

A latent maintenance hazard: three implementations of the same WHT must be kept in lockstep, and a future dequant/rotate caller could legitimately reach the dead sequential variant and get a *different convention* (not a numerically different result from the same convention). No numerical mismatch is demonstrable from the code as written. Severity in state.md: **HIGH**; re-assessed here as a maintainability risk rather than a live accuracy loss.

## Location

- Path: `src/ggml/src/ggml-cuda/turbo-quant.cuh` (verified) — `turbo_fwht_128()` 88-106, `turbo_fwht_64()` 108-125, `turbo_rotate_forward()` 127-131, `turbo_rotate_forward_64()` 135-139 (no callers).
- Path: `src/ggml/src/ggml-cuda/set-rows.cu` (verified) — encoding butterfly: `WHT_STAGE_SHARED` macro 332, stages 337-343, normalization 346-351.
- Path: `src/ggml/src/ggml-cuda/turbo-wht.cu` (verified) — `k_turbo_wht_f32<direction, group_size>` 31-96; `WHT_STAGE` macro 66, stages 70-76, normalization 80; InnerQ forward 49-53, inverse 90-93; launches 151-161.
- Path: `src/ggml/src/ggml-cuda/ggml-cuda.cu` (verified) — `GGML_OP_TURBO_WHT` dispatch at 2092-2093.

## Status

Unresolved in the sense that the duplication remains; the specific defect claim is corrected above. Listed as action item "Unify WHT Implementations (TQ-4)" (§5.6 of [[source-state-md]]).

## Fix sketch

Per state.md §4 TQ-4 and §5.6: unify the implementations. Concretely, either delete the dead `turbo_rotate_forward()` / `turbo_rotate_forward_64()` / `turbo_fwht_128()` / `turbo_fwht_64()` from `turbo-quant.cuh` and route every rotation through `k_turbo_wht_f32` (`turbo-wht.cu:31`), or keep a single in-place butterfly and have both `k_set_rows_turbo3` and `k_turbo_wht_f32` call it. Either way, the InnerQ placement contract (scale on encode, `scale_inv` on Q pre-rotation, `scale_inv` after inverse WHT for V) must be stated in one place.

## See also

- [[walsh-hadamard-transform]] · [[turboquant]] · [[innerq]] · [[quantization]]
- [[tq-1-missing-gemm-kernels]] — the sibling defect in the same quantization path
- [[tq-5-tail-elements]] — tails bypass the rotation entirely
- [[ta-1-wht-inversion-256]] · [[ta-4-cooperative-fwht-race]] — the TriAttention-side WHT bugs (same family of 128/64 element transforms)
