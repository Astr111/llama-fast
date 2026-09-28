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

The scoring pipeline (invert RoPE → invert WHT → combine with calibrated query statistics → aggregate over offsets) is the arithmetic that decides which KV keys get evicted. This page decides three hypotheses earlier waves left open. All three concern the **pre-rotation and aggregation stages**, not the WHT itself ([[walsh-hadamard-transform]], [[ta-1-wht-inversion-256]]).

## Bottom line

| Hyp | Question | Verdict |
| :-- | :--- | :--- |
| **H1** | Is `n_offsets == 0` reachable via the documented CLI, and does the default `mean` aggregate then emit NaN? | **CONFIRMED** — reachable by plain defaults; `mean` yields NaN on both the CPU and GPU paths; `max` yields a constant degenerate tie. |
| **H2** | Does the scorer's RoPE inverse match the target model's MRoPE? | **CONFIRMED defect** — the scorer treats all 256 head dims as two rotary halves with pairs `(f, f+128)` and `ω_f = θ^(−2f/256)`, while qwen35 rotates only 64 dims under MRoPE sections; the inverse is not an inverse of the model's forward rotation. |
| **H3** | Is `freq_scale_sq` really always 1.0? | **Dead code (b)** — `cos²(ω·0)+sin²(ω·0) ≡ 1` by construction; the paper's `fscale²_f = 1/ω²` appears nowhere in `src/`. Not intentionally disabled (no flag), not active elsewhere. |

Two of these are **new, unlisted defects**, and each deserves its own issue page:

- **H1 is the worst of the three.** Under plain defaults (`src/common/common.h:755` → `offset_max = 0`; `:758` → `agg = mean`) the CPU path computes `0.0f · (1.0f/0.0f) = NaN` (`src/src/llama-triattention.cpp:450`, `:501-503`) and the GPU path computes `0.0f/0.0f = NaN` (`triattention-score.cu:296`) for **every candidate key**. Both launch scripts omit `--triattention-offset-max` and `--triattention-agg`, so the shipped workflow runs exactly this configuration. The NaNs then reach the selector's `std::partial_sort` comparator `scores[a] > scores[b]` (`llama-triattention.cpp:867-870`), which is not a strict weak ordering over NaN ⇒ undefined selection (`[INFERENCE]` — standard-library contract, not read from a file). This belongs with [[ta-7-config-validation]] in spirit but is a distinct runtime failure, and it warrants its own issue page.
- **H2 is the second new defect.** It is stronger than [[triattention-calibrate]]'s open "pre/post-RoPE capture" question: even a pre-RoPE capture would not fix it, because the scorer takes no `n_rot`, no `sections`, and a single `positions[]` array — it cannot represent a partial/MRoPE basis at all.

**What this means for the roadmap.** All three verdicts sit on the eviction-quality track of [[roadmap]] — item 1 (port the [[ta-1-wht-inversion-256]] fix) and item 4 (dynamic `min_history_budget`, [[ta-2-budget-starvation]]) — because each changes what "eviction quality" means before either item can be measured.

- **H1** touches **item 4** directly: eviction that runs on NaN scores is meaningless in exactly the long-prompt regime item 4 governs. It also taints **item 1**, since any eviction-quality benchmark against this build measures the NaN/constant path, not real scoring.
- **H2** touches **items 1 and 4** (a second, independent reason eviction quality cannot be trusted) and **item 6**'s premise: the scorer inverts around a wrong RoPE next to the wrong WHT ([[walsh-hadamard-transform]]).
- **H3** resolves the already-filed candidate [[ta-5-freq-scale-dead-code]] from "either dead code or disabled" to "provably dead code". It is not on the six-item list, but it biases every measurement of items 1 and 4, because the shipped scoring is not the formula the paper commits to.

## Evidence

<!-- APPEND -->
