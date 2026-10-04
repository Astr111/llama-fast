---
title: "TA-8: offset_max defaults to 0, making every eviction score NaN"
type: issue
status: current
updated: 2026-09-28
sources: [state.md, TRIATTENTION.md]
verified: [src/common/common.h, src/common/common.cpp, src/common/arg.cpp, src/src/llama-triattention.cpp, src/src/llama-context.cpp, src/ggml/src/ggml-cuda/triattention-score.cu, scripts/start_server_turbo.sh, scripts/run_cli.sh]
tags: [triattention, kv-eviction, correctness, ship-blocker]
---

# TA-8: `offset_max = 0` makes every score NaN

> **Not in either inventory.** Found 2026-09-28 while resolving the open question called H1 on [[scoring-correctness]]; confirmed from code, never executed. It is a *new* defect, and it is worse than [[ta-1-wht-inversion-256]] in one respect: it fires on the shipped launch configuration.

## Symptom

In the configuration both launch scripts actually use, every key's score is **NaN**, so TriAttention's eviction order is undefined. Nothing in the runtime complains.

## Cause

A four-step chain, each step verified:

1. `offset_max` defaults to `0` — `src/common/common.h:755`. It is registered with no override (`src/common/arg.cpp:4714-4720`) and passed unchanged into the TriAttention config (`src/common/common.cpp:1409` → `src/src/llama-context.cpp:4394`) with no guard.
2. `triattention_build_offsets` returns `0` for `d = 1 <= 0` — `src/src/llama-triattention.cpp:333-339` — so `n_offsets = 0` (`:683-684`), again unguarded.
3. Both launch scripts **omit** `--triattention-offset-max` *and* `--triattention-agg` (`scripts/start_server_turbo.sh:35`, `scripts/run_cli.sh:24`), and the default aggregate is `mean` (`common.h:758`; `src/docs/TRIATTENTION.md:87`).
4. The mean over zero offsets is not zero, it is undefined:
   - **CPU path:** `total_score = 0.0f` (`:462`), the offset loop is skipped (`:465`), then it is multiplied by `1.0f / 0.0f` = `+inf` (`:450`, `:501-503`) → `0 × inf` = **NaN** for every key.
   - **GPU path:** `total_score = sum / (float)n_offsets` = `0.0f / 0.0f` = **NaN** (`src/ggml/src/ggml-cuda/triattention-score.cu:291-296`). The GPU is the path attempted first (`llama-triattention.cpp:1162-1166`), so this is the normal case, not the fallback.

The NaN then reaches the selection comparator (`scores[a] > scores[b]`, `:867-870`), where it breaks the strict weak ordering `std::partial_sort` requires — undefined behaviour, not merely wrong order.

`--triattention-agg max` avoids the NaN but not the degeneracy: it yields a constant score for every key (CPU `0.0f`; GPU `-1e30f`, `triattention-score.cu:280-287`), i.e. an arbitrary eviction order under a different name.

## Impact

**CRITICAL (new).** In the profile the project ships and documents (`turbo3` K + `q8_0` V, budget 4096, window 512), eviction selects essentially arbitrary tokens. This is a plausible co-cause of the quality and acceptance-rate behaviour attributed elsewhere to [[ta-1-wht-inversion-256]] and [[ta-2-budget-starvation]] — all three damage *which* keys survive, and only one of them was known. It also means any measurement of the other two is confounded until this is fixed.

## Location

- `src/common/common.h:755` — the default
- `src/src/llama-triattention.cpp:333-339`, `:683-684` — `n_offsets = 0`
- `src/src/llama-triattention.cpp:450`, `:462`, `:465`, `:501-503` — the CPU mean
- `src/ggml/src/ggml-cuda/triattention-score.cu:291-296` — the GPU mean
- `src/src/llama-triattention.cpp:867-870` — the comparator that receives NaN
- `scripts/start_server_turbo.sh:35`, `scripts/run_cli.sh:24` — the flags that are not passed

## Status

**RESOLVED (2026-10-04).** Fixed across `src/common/common.h`, `src/src/llama-triattention.cpp`, and `src/ggml/src/ggml-cuda/triattention-score.cu` via commit `585fe5d`.
- Implemented Alternative 3 (comprehensive defense):
  1. Default `triattention_offset_max = 65536` in `common.h` ensures 17 geometric offsets are generated when CLI arguments are omitted.
  2. Guard in `triattention_init` logs warning and sets `disable_trig = true` (norm-only fallback) if `n_offsets == 0` when `budget > 0`.
  3. Arithmetic fail-safes in both CPU (`triattention_score_keys`) and GPU (`k_triattention_score_half`) scoring routines prevent division by zero or NaN creation when `n_offsets == 0`.


## Fix sketch

Three options, in order of how much they preserve:

1. **Guard the aggregate** — treat `n_offsets == 0` as a configuration error at init (this is also what [[ta-7-config-validation]] asks for) and refuse to enable pruning.
2. **Give `offset_max` a working default** — the doc's own value is 65536 (`TRIATTENTION.md`, CLI table), which is what the code's `0` should probably have been; confirm the intended geometry before changing it, since the offsets define the geometric time-horizon set.
3. **Make the mean well-defined** — a `n_offsets == 0` branch that scores on the other terms alone, rather than dividing by zero.

Option 1 plus 2 is the smallest change that makes the shipped scripts correct.

## See also

[[scoring-correctness]] · [[triattention]] · [[triattention-calibrate]] · [[ta-1-wht-inversion-256]] · [[ta-2-budget-starvation]] · [[ta-7-config-validation]] · [[roadmap]]
