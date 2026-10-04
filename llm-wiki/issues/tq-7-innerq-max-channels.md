---
title: "TQ-7: INNERQ_MAX_CHANNELS = 128 hard limit"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/ggml/src/ggml-cuda/turbo-innerq.cuh, src/ggml/src/ggml-cuda/turbo-innerq.cu, src/ggml/src/ggml-cuda/turbo-quant.cuh, src/ggml/src/ggml-cuda/set-rows.cu, src/ggml/src/ggml-cuda/turbo-wht.cu, src/src/llama-kv-cache.cpp]
tags: [innerq, turboquant, kv-cache]
---

## Symptom

state.md reports that the InnerQ channel limit is hardcoded to 128 and that models with `head_dim=256` (such as the primary target, Ternary-Bonsai-2-27B) therefore cannot fully utilise InnerQ equalization across all channels. ([[source-state-md]], §4 TQ-7)

## Cause

**Verified: the constant and its blast radius.** `#define INNERQ_MAX_CHANNELS 128` (`src/ggml/src/ggml-cuda/turbo-innerq.cuh:7`) is the array bound for every piece of InnerQ state: the device arrays `d_innerq_scale`, `d_innerq_scale_inv`, `d_innerq_sq_accum` (`turbo-quant.cuh:147-149`), the host working buffers `zeros`/`sq_accum`/`rms`/`scale`/`scale_inv` in `turbo_innerq_init()` and `turbo_innerq_finalize()` (`turbo-quant.cuh:177`, `191`, `205`, `215-216`), the publish loop (`turbo-innerq.cu:16-21`), and the per-model scale tensor plus its uploads on the llama side — `ggml_new_tensor_1d(ctx, GGML_TYPE_F32, INNERQ_MAX_CHANNELS)` (`src/src/llama-kv-cache.cpp:378`), the identity fills at `:434-436` and `:543-545`, and the publish consumer at `:3124`. The llama layer even re-defines the constant under an `#ifndef` guard (`llama-kv-cache.cpp:24-25`), so the limit is duplicated across modules.

**Correction — the claimed impact does not hold here.** InnerQ scales are indexed by *position within a WHT group*, not by absolute channel: the encoder applies `x[j] *= d_innerq_scale[j]` where `j = threadIdx.x ∈ [0, GROUP_SIZE)` (`set-rows.cu:259`, `:288-295`), and the rotation op applies `scale_inv[t % group_size]` (`turbo-wht.cu:50`, `:92`). The group size is pinned to 128 for every turbo tensor: `wht_group = 128` is always written into the set-rows `op_params` (`llama-kv-cache.cpp:1583-1587`, `1637-1638`, `1663-1664`), head_dim is padded to the next multiple of 128 (`llama-kv-cache.cpp:324-346`), and both `set_rows_cuda_turbo3` (`set-rows.cu:553-555`) and `ggml_cuda_turbo_wht` (`turbo-wht.cu:132-134`) clamp/assert `group_size ∈ {64, 128}`. `QK_TURBO3 = QK_TURBO2 = QK_TURBO4 = 128` (`src/ggml/src/ggml-common.h:324`, `:343`, `:374`). For a 256-wide head that means two 128-element rotation groups per head, each indexed by the same 128-entry table — **every channel is covered**; the equalization profile is simply shared between the two halves of a head [INFERENCE from the indexing, not observed in a run]. The constant would only truncate if a single group larger than 128 existed, which the two clamps above currently forbid.

What is genuinely latent is the coupling: the 128 in `turbo-innerq.cuh:7`, the 128 fallback in `llama-kv-cache.cpp:24-25`, the 128-element tensor at `llama-kv-cache.cpp:378`, and the `group_size` clamps in `set-rows.cu:555` / `turbo-wht.cu:133` are four independent literals that must agree. The guard that was presumably meant to catch a bad combination, `const bool multi_group_per_head = (group_size < 128);` (`turbo-quant.cuh:268-277`), only disables InnerQ when the group is *smaller* than 128 — never, given `wht_group = 128`.

## Impact

No live truncation of InnerQ channels in this checkout. Residual risks: (a) four unsynchronised copies of "128" across two modules, and (b) the `multi_group_per_head` heuristic in `turbo-quant.cuh:268` is effectively dead, so a future non-128 group size would not be caught by it. Severity in state.md: **MEDIUM**; re-assessed here as primarily a coupling/maintainability issue.

## Location

- Path: `src/ggml/src/ggml-cuda/turbo-innerq.cuh` (verified) — `#define INNERQ_MAX_CHANNELS 128` at line 7; `extern` host array sized by it at line 11.
- Path: `src/ggml/src/ggml-cuda/turbo-quant.cuh` (verified) — device arrays 147-149, host buffers 177/191/205/215-216, `multi_group_per_head` heuristic 268-277.
- Path: `src/ggml/src/ggml-cuda/turbo-innerq.cu` (verified) — publish loop bounded by the constant at 16-21.
- Path: `src/ggml/src/ggml-cuda/set-rows.cu` (verified) — `group_size` clamp 553-555; per-group-index scaling at 288-295.
- Path: `src/ggml/src/ggml-cuda/turbo-wht.cu` (verified) — `scale_inv[t % group_size]` at 50 and 92; `GGML_ASSERT(group_size == 64 || group_size == 128)` at 133.
- Path: `src/src/llama-kv-cache.cpp` (verified) — duplicated constant 24-25; scale tensor 378; identity fills 434-436 and 543-545; publish consumer 3124.

## Status

Unresolved. Not listed among the §5 action items of [[source-state-md]]; raised in §4 TQ-7.

## Fix sketch

Per state.md §4 TQ-7 ("maximum channel limit is hardcoded to 128"): the limit should track the supported group size rather than being an independent literal. Concretely: keep one definition (the `#define` at `turbo-innerq.cuh:7`) and add a `static_assert` tying it to the largest `GROUP_SIZE` the kernels accept (`set-rows.cu:555`, `turbo-wht.cu:133`), size the llama-side tensor from that same value instead of the local `#ifndef` fallback (`llama-kv-cache.cpp:24-25`, `:378`), and renew the `multi_group_per_head` guard in `turbo-quant.cuh:268` so that a group size above `INNERQ_MAX_CHANNELS` disables InnerQ with a warning rather than indexing past the arrays. If a group larger than 128 is ever wanted, all four sites must grow together.

## See also

- [[innerq]] · [[turboquant]] · [[quantization]] · [[kv-cache]]
- [[tq-2-innerq-host-state]] · [[tq-3-innerq-multigpu]] · [[tq-6-innerq-race]] — the other InnerQ defects
- [[ternary-bonsai-2-27b]] — the `head_dim=256` target named in state.md
- [[tq-5-tail-elements]] — the other group-boundary defect
