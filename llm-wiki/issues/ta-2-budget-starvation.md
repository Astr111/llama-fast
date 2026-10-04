---
title: "TA-2: Budget starvation on long prefixes"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/src/llama-triattention.cpp]
tags: [triattention, kv-eviction, budget]
---

## Symptom

TriAttention degenerates into a pure sliding window: on long system prompts all historical tokens between the prefix and the recent window are instantly evicted. Agent benchmarks lose reasoning context; speculative decoding acceptance rate plummets. GPU/CPU scoring pipelines execute uselessly for `B=0`. ([[source-state-md]], §3 TA-2)

## Cause

In `triattention_prune_impl` (`src/src/llama-triattention.cpp:1077`), the eviction budget is `decode_budget = (budget > n_protected) ? (budget - n_protected) : 0` at line 1150. `n_protected` counts prefix-protected plus recent-window cells (loop at lines 1134–1146, `is_prefix` at 1136, `is_recent` vs `recent_threshold` at lines 1128/1138). When `prefix_length + divide_length >= budget`, `n_protected` reaches the full budget and `decode_budget` drops to 0 — for the configured budget 4096 and a >3500-token system prompt, every middle token is evicted at once. ([[source-state-md]], §3 TA-2)

## Impact

Loss of all mid-context history on long agent prompts; the eviction mechanism reduces to the 512-token recent window (`divide_length`). Severity in state.md: **HIGH**. ([[source-state-md]])

## Location

- Path: `src/src/llama-triattention.cpp` (verified)
- Symbol: `triattention_prune_impl` at line 1077; the `decode_budget` computation at line 1150; protection loop at lines 1134–1146.

## Status

**RESOLVED (2026-10-04).** Fixed in `src/src/llama-triattention.cpp` via commit `f858af6`.
- Implemented dynamic `min_history_budget` in `triattention_prune_impl`: `decode_budget` is guaranteed to retain a floor of historical tokens (up to $\min(n_{decode}, \max(budget / 8, 32))$) so that long prompts exceeding `budget - divide_length` do not completely starve historical context into a sliding window.


## Fix sketch

Implement the dynamic `min_history_budget` logic in `triattention_prune_impl` (state.md §5.4): clamp `n_protected` (or floor `decode_budget`) so a minimum share of the budget is always reserved for scored historical tokens, instead of letting the prefix fully consume `budget - divide_length`.

## See also

- [[triattention]] · [[kv-eviction]]
- [[ta-6-overlap-double-counting]] — future `max_protected` patches in the same protection-counting loop
- [[ta-7-config-validation]] — no warning is emitted for configs that guarantee this starvation
