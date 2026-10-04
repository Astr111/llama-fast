---
title: "TA-3: CPU fallback synchronous D2H transfers"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/src/llama-triattention.cpp]
tags: [triattention, kv-cache, performance]
---

## Symptom

When the CPU fallback path runs (GPU init failed), the pipeline stalls for 15–30 seconds instead of milliseconds. ([[source-state-md]], §3 TA-3)

## Cause

`triattention_dequant_kv_head` (`src/src/llama-triattention.cpp:540`) calls `ggml_backend_tensor_get(k_tensor, quant_buf.data(), tensor_offset, head_bytes)` per KV cell at line 572, inside the `for (ci = 0; ci < n_cells; ci++)` loop at line 566 — one synchronous `cudaMemcpy` per cell rather than a single bulk transfer. For `n_decode=4096` and `head_dim=256` this is ~4096 separate ~1KB device-to-host transfers. ([[source-state-md]], §3 TA-3)

## Impact

15–30 second pipeline stalls on the CPU scoring path for realistic decode lengths. Severity in state.md: **HIGH**. ([[source-state-md]])

## Location

- Path: `src/src/llama-triattention.cpp` (verified)
- Symbol: `triattention_dequant_kv_head` at line 540; per-cell `ggml_backend_tensor_get` at line 572.

## Status

Unresolved. ([[source-state-md]], §3 TA-3)

## Fix sketch

Per state.md §3 TA-3 and §5.5: refactor `triattention_dequant_kv_head` to batch the transfers — either one bulk `cudaMemcpy` covering the contiguous candidate cell range (cells may need compaction into a staging buffer first), or enforce strict GPU-only operation so the fallback path is never taken.

## See also

- [[triattention]] · [[kv-cache]]
- [[ta-2-budget-starvation]] — same prune pipeline; useless CPU/GPU scoring for `B=0`
