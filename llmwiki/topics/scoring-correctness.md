---
title: Scoring correctness
type: topic
status: current
updated: 2026-09-28
sources: [TRIATTENTION.md, state.md, README.md]
verified: [src/common/common.h, src/common/common.cpp, src/common/arg.cpp, src/src/llama-triattention.cpp, src/src/llama-triattention.h, src/src/llama-context.cpp, src/src/models/qwen35.cpp, src/src/llama-model.cpp, src/src/llama-model.h, src/src/turbo-rotation-data.h, src/ggml/src/ggml-cuda/triattention-score.cu, src/docs/TRIATTENTION.md, src/docs/development/HOWTO-add-model.md, scripts/start_server_turbo.sh, scripts/run_cli.sh]
tags: [triattention, kv-eviction, scoring, correctness]
---

# Scoring correctness

The four-step scoring pipeline (invert RoPE → invert WHT → combine with calibrated query statistics → aggregate over offsets) is the arithmetic that decides which KV keys get evicted. This page decides three hypotheses that earlier waves left open. All three concern the **pre-rotation and aggregation stages**, not the WHT itself ([[walsh-hadamard-transform]], [[ta-1-wht-inversion-256]]).

## Bottom line

| Hyp | Question | Verdict |
| :-- | :--- | :--- |
| **H1** | Is `n_offsets == 0` reachable via the documented CLI, and does the default `mean` aggregate then emit NaN? | **CONFIRMED** — reachable by plain defaults; `mean` yields NaN on both the CPU and GPU paths, `max` yields a constant degenerate tie. |
| **H2** | Does the scorer's RoPE inverse match the target model's MRoPE? | **CONFIRMED defect** — the scorer treats all 256 head dims as two rotary halves with pairs `(f, f+128)` and `ω_f = θ^(−2f/256)`, while qwen35 rotates only 64 dims with MRoPE sections; the inverse is not an inverse of the model's forward rotation. |
| **H3** | Is `freq_scale_sq` really always 1.0? | **Dead code (b)** — `cos²(ω·0)+sin²(ω·0) ≡ 1` by construction; the paper's `fscale²_f = 1/ω²` appears nowhere in `src/`. Not intentionally disabled (no flag), not active elsewhere. |

Two of these are **new, unlisted defects**, and each deserves its own issue page:

- **H1 is the worst.** With defaults (`src/common/common.h:755` → `offset_max = 0`, `:758` → `agg = mean`) the CPU path computes `0.0f · (1.0f/0.0f) = NaN` (`src/src/llama-triattention.cpp:450`, `:501-503`) and the GPU path `0.0f/0.0f = NaN` (`triattention-score.cu:296`) for **every candidate key**. The launch scripts pass no `--triattention-offset-max` and no `--triattention-agg`, so the shipped workflow runs exactly this configuration. NaNs then flow into the selector's `std::partial_sort` comparator `scores[a] > scores[b]` (`llama-triattention.cpp:867-870`), which is not a strict weak ordering over NaN ⇒ undefined selection. `[INFERENCE]` — the comparator requirement is from the C++ standard, not read from a file. H1 belongs with [[ta-7-config-validation]] in spirit but is a distinct runtime failure with this page as its warrant.
- **H2 is the second new defect.** It is stronger than [[triattention-calibrate]]'s open "pre/post-RoPE" question: even a pre-RoPE capture would not fix it, because the scorer has no `n_rot`, no `sections`, and a single `positions[]` array — it cannot represent a partial/MRoPE basis at all.

**What this means for the roadmap.** All three verdicts live on the eviction-quality track of [[roadmap]] (items 1 and 4: the [[ta-1-wht-inversion-256]] port and the dynamic `min_history_budget` of [[ta-2-budget-starvation]]), because they change what "eviction quality" means before either of those items can be measured. Specifically:

- **H1** touches item 4 directly (long-prompt behaviour — eviction that runs on NaN scores is eviction that is meaningless in the exact regime item 4 governs) and item 1 (any eviction-quality benchmark made against this build is measuring the NaN/constant path, not real scoring). It should be a new issue page, not merely appended to [[ta-7-config-validation]].
- **H2** touches items 1 and 4 (it is a second, independent reason eviction quality cannot be trusted) and item 6's premise (the basis the scorer inverts around a broken RoPE, next to the broken WHT — [[walsh-hadamard-transform]]). It deserves a new issue page.
- **H3** resolves the status of the already-listed candidate [[ta-5-freq-scale-dead-code]] from "either dead code or disabled" to "provably dead code"; it is not on the six-item list but biases every measurement of items 1 and 4, because the shipped scoring is not the formula the paper commits to.

## Evidence

### H1 — `n_offsets == 0` is reachable and the `mean` aggregate emits NaN

**Reachability (no guard anywhere).** The default is `int32_t triattention_offset_max = 0` (`src/common/common.h:755`); the flag `--triattention-offset-max` registers with that value and never overrides it (`src/common/arg.cpp:4714-4720`). The value travels unchanged: `common_init_result` passes `params.triattention_offset_max` through (`src/common/common.cpp:1409`) into `llama_triattention_init` → `cfg.offset_max = (uint32_t)offset_max` with **no validation** (`src/src/llama-context.cpp:4358-4394`). `triattention_build_offsets` yields zero elements for 0 because the loop starts at `d = 1 ≤ offset_max` (`llama-triattention.cpp:333-339`), so `state->n_offsets = triattention_build_offsets(state->offsets, cfg->offset_max)` is 0 (`:683-684`) — again with no `n_offsets == 0` guard (a `n_offsets`-wide grep finds no check). The header comment "default: 65536" (`llama-triattention.h:133`) and the design document's 65536 (`src/docs/TRIATTENTION.md:84`) both contradict the code default of 0 — a further confirmation the shipped default is what the user actually gets. `scripts/start_server_turbo.sh:35` and `scripts/run_cli.sh:24` pass `--triattention-stats/budget/window/protect-prefill` and **never** `--triattention-offset-max` or `--triattention-agg`, so the documented CLI and both launch scripts reproduce the zero case.

**NaN for the default `mean`.** `triattention_score_keys` computes `inv_n_offsets = 1.0f/(float)n_offsets` (`llama-triattention.cpp:450`), then `total_score` is initialized to `0.0f` (`:462`); the offset loop (`:465`) never executes, and for `TRIATTENTION_AGG_MEAN` it does `total_score *= inv_n_offsets` (`:501-503`) ⇒ `0.0f · (1.0f/0.0f = +inf) = NaN` under IEEE-754. The GPU kernel's mean branch does `sum / (float)n_offsets = 0.0f/0.0f = NaN` directly (`triattention-score.cu:296`). The GPU path is the default attempt: `triattention_init_gpu` is invoked unconditionally and only falls back to CPU if GPU init fails (`llama-triattention.cpp:1162-1166`, `:885-916`). `mean` is the documented default (`src/docs/TRIATTENTION.md:87`; `triattention_agg = 0` at `common.h:758`; `TRIATTENTION_AGG_MEAN = 0` is "Paper default" at `llama-triattention.h:88`). `[INFERENCE]` — the arithmetic readings are standard IEEE-754 float semantics, not observed in a run; the GPU expression is unambiguous regardless of compiler `-ffast-math` since the input is a literal 0.

**`max` is not NaN but is degenerate.** On the CPU path the max branch never fires either, so `total_score` stays `0.0f` and every key scores exactly 0; on the GPU path `max_score` initialises at `-1e30f` (`triattention-score.cu:280-287`) so every key scores exactly −1e30. Both are exact ties ⇒ arbitrary eviction. A secondary hazard: `cudaMalloc(&state->d_offsets, config->n_offsets * sizeof(float))` alloc