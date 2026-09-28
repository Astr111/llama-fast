---
title: "TQ-5: Tail elements skip WHT and InnerQ — path unreachable in this checkout"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/ggml/src/ggml-cuda/set-rows.cu, src/ggml/src/ggml-cuda/turbo-wht.cu, src/ggml/src/ggml-cuda/dequantize.cuh, src/ggml/src/ggml-cuda/turbo-quant.cuh, src/ggml/src/ggml-common.h, src/src/llama-kv-cache.cpp]
tags: [turboquant, quantization, walsh-hadamard-transform]
---

## Symptom

state.md reports that when `head_dim % GROUP_SIZE != 0` the leftover channels are quantized without WHT rotation and without InnerQ scaling, that dequantization has no tail handling, and that models with non-aligned `head_dim` (GLM 576) therefore carry systematic error in those channels. ([[source-state-md]], §4 TQ-5)

## Cause

**Verified in the kernel.** `k_set_rows_turbo3_tail()` (`src/ggml/src/ggml-cuda/set-rows.cu:422-533`) loads the tail element (`const float val = src_row[tail_start + j]`), computes an L2 norm over the tail group, then takes the branch commented `// ---- Normalize (no WHT!) ----` and quantizes straight to a centroid — there is no butterfly stage, and the kernel never reads `d_innerq_calibrating`, `d_innerq_scale`, or any other InnerQ symbol. The read path is symmetric: `k_turbo_wht_copy_tail()` (`turbo-wht.cu:100-114`) copies tail elements through unchanged, launched under the comment "Pass through tail elements unchanged (no rotation)" at `turbo-wht.cu:165-173`; the header comment at `turbo-wht.cu:10-11` states the same ("Tail elements are left unchanged (identity)"). And `dequantize.cuh:163-168` (`dequantize_turbo3_0`) does indeed contain no tail-specific branch.

**Correction — the path is unreachable here.** The launcher computes `const int tail_size = (int)(ne00 % group_size);` (`set-rows.cu:559`) immediately after asserting `GGML_ASSERT(ne00 % group_size == 0);` (`set-rows.cu:556`), so `tail_size` is always `0` and the guarded second launch at `set-rows.cu:591-598` (itself further guarded by `GGML_ASSERT(tail_size % QK_TURBO3 == 0)` at line 592) can never execute. `group_size` comes from `dst->op_params` with a clamp to `{64, 128}` (`set-rows.cu:553-555`), and QK_TURBO3 = QK_TURBO2 = QK_TURBO4 = 128 (`src/ggml/src/ggml-common.h:324`, `:343`, `:374`) — i.e. the storage block is exactly one rotation group, so no remainder below the group size can be block-aligned. Upstream of the kernel, the KV cache pads K and V `head_dim` to the next multiple of 128 for turbo types (`src/src/llama-kv-cache.cpp:324-346`) and always writes `wht_group = 128` into the set-rows `op_params` (`llama-kv-cache.cpp:1583-1587`, `1637-1638`, `1663-1664`), so GLM's 576 is padded to 640 before the kernel sees it. The same `tail_size == 0` structure holds for `set_rows_cuda_turbo2` (`set-rows.cu:898-904`, launch 933-940). Note also that the "block size 32" comments this defect's premise rests on are stale: `turbo-quant.cuh:5` and the block comment at `ggml-common.h:318-323` still say 32, while `QK_TURBO3` is 128 (`ggml-common.h:324-325`).

## Impact

No live impact in this checkout: the tail kernels and the tail pass-through kernel are dead code, and no model shape reaches them while `wht_group` is forced to 128 and the padding at `llama-kv-cache.cpp:324-346` is in place. The residual risk is that the *contract* is implicit — relaxing the padding, the `group_size` clamp, or the `ne00 % group_size == 0` assertion would silently reactivate a path whose tails are unrotated and uncalibrated. Severity in state.md: **MEDIUM**.

## Location

- Path: `src/ggml/src/ggml-cuda/set-rows.cu` (verified) — `k_set_rows_turbo3_tail()` at 422-533 (norm + "Normalize (no WHT!)" + quantize; no InnerQ); `group_size` clamp 553-555; `GGML_ASSERT(ne00 % group_size == 0)` 556; `tail_size` 559; guarded launch 591-598; turbo2 twins at 775-946.
- Path: `src/ggml/src/ggml-cuda/turbo-wht.cu` (verified) — `k_turbo_wht_copy_tail()` 100-114, launch 167-173; `group_size` assert and `groups_per_head` 132-134.
- Path: `src/ggml/src/ggml-cuda/dequantize.cuh` (verified) — `dequantize_turbo4_0` 154-159, `dequantize_turbo3_0` 163-168, `dequantize_turbo2_0` 171-176; no tail branch.
- Path: `src/ggml/src/ggml-common.h` (verified) — `QK_TURBO3` 324, `QK_TURBO3_GROUP` 325, `QK_TURBO4` 343, `QK_TURBO2` 374; stale "32" comments 318-323.
- Path: `src/src/llama-kv-cache.cpp` (verified) — turbo head_dim padding to a multiple of 128 at 324-346; `wht_group = 128` into `op_params` at 1583-1587, 1637-1638, 1663-1664.

## Status

Unresolved in state.md, but not reachable in this checkout — the effective status is "dead code with an implicit contract", so the useful work is deletion or an explicit assertion rather than a correctness fix. Not listed among the §5 action items of [[source-state-md]].

## Fix sketch

Per state.md §4 TQ-5 ("requires padding or partial-group WHT"): the padding half already exists at `llama-kv-cache.cpp:324-346`, which is why the tail path is dead. The concrete remaining work is either (a) delete `k_set_rows_turbo3_tail` / `k_set_rows_turbo2_tail` (`set-rows.cu:422`, `:775`) and `k_turbo_wht_copy_tail` (`turbo-wht.cu:100`) together with their launches, or (b) if a partial group is ever wanted for a group size greater than `QK_TURBO3`, relax `GGML_ASSERT(ne00 % group_size == 0)` (`set-rows.cu:556`), pad the partial group to `QK_TURBO3` inside the kernel, and apply the same rotation on the read path — the InnerQ calibration loop indexed by `threadIdx.x` (`set-rows.cu:288-295`) would need matching bounds.

## See also

- [[turboquant]] · [[quantization]] · [[walsh-hadamard-transform]] · [[kv-cache]]
- [[tq-4-wht-numerical-mismatch]] — the same rotation, three implementations
- [[tq-1-missing-gemm-kernels]] — the sibling defect in the quantization path
