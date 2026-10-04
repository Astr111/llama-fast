---
title: "TA-7: Missing configuration validation"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/src/llama-triattention.cpp]
tags: [triattention, configuration, robustness]
---

## Symptom

Users get no warning when their prompt length guarantees immediate history eviction. ([[source-state.md]], §3 TA-7)

## Cause

`triattention_init` (`src/src/llama-triattention.cpp:630`) validates the calibration file (head_dim mismatch at line 642, n_kv_heads mismatch at line 651, rope_theta warning at line 659) but performs no check on the physical compatibility of the `triattention_config` fields `budget`, `prefix_length`, and `divide_length` stored into the state (lines 671–673). There is no warning or error path for configurations where the prefix alone consumes the budget — the condition that triggers [[ta-2-budget-starvation]]. ([[source-state.md]], §3 TA-7)

## Impact

Silent misconfiguration: configs that guarantee immediate history eviction (e.g. large `prefix_length` vs `budget`) pass initialization without any diagnostic. Severity in state.md: **LOW**. ([[source-state.md]])

## Location

- Path: `src/src/llama-triattention.cpp` (verified)
- Symbol: `triattention_init` at line 630; calibration-only validation at lines 642–660; unvalidated config copy at lines 671–673.

## Status

Unresolved. Needs warning/error emission. ([[source-state.md]], §3 TA-7)

## Fix sketch

Add a validation step in `triattention_init` that warns (or errors) when `prefix_length + divide_length >= budget` — the exact geometry that zeroes the decode budget in `triattention_prune_impl` — using the same `fprintf(stderr, "[TriAttention] ...")` convention as the existing calibration checks.

## See also

- [[triattention]]
- [[ta-2-budget-starvation]] — the failure this validation would surface
- [[ta-6-overlap-double-counting]] — the other config-geometry hazard in the protection logic
