---
title: "TA-6: Overlap double-counting risk"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/src/llama-triattention.cpp]
tags: [triattention, kv-eviction, robustness]
---

## Symptom

None today — latent risk for future patches. ([[source-state.md]], §3 TA-6)

## Cause

In the protection loop of `triattention_prune_impl` (`src/src/llama-triattention.cpp:1134–1146`), `if (is_prefix || is_recent)` at line 1140 correctly counts each token at most once. But `recent_threshold` (line 1128: `max_pos - divide_length + 1`) and the prefix range (`occupied_positions[i] < prefix_length`, line 1137) can overlap: recent tokens may also lie inside the prefix when `prefix_length + divide_length` covers the whole context. Future patches computing a `max_protected` bound must account for this overlap or they will overestimate protected cells. ([[source-state.md]], §3 TA-6)

## Impact

No current defect; risk of overestimated `max_protected` (and thus a shrunken decode budget — see [[ta-2-budget-starvation]]) in future patches. Severity in state.md: **LOW**. ([[source-state.md]])

## Location

- Path: `src/src/llama-triattention.cpp` (verified)
- Symbol: `triattention_prune_impl`, protection loop lines 1134–1146; `recent_threshold` at line 1128.

## Status

Documented for future patching. ([[source-state.md]], §3 TA-6)

## Fix sketch

When introducing `max_protected`, compute the protected set as a union (as the current `is_prefix || is_recent` test does) rather than summing `prefix_length` and `divide_length` — e.g. clamp the prefix bound at `min(prefix_length, recent_threshold)` so overlap is counted once.

## See also

- [[triattention]] · [[kv-eviction]]
- [[ta-2-budget-starvation]] — same loop; the budget that `max_protected` would shrink
- [[ta-7-config-validation]] — related config-geometry hazard
