---
title: KV eviction
type: concept
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/src/llama-triattention.cpp, src/src/llama-triattention.h, src/src/llama-kv-cache.cpp, src/ggml/src/ggml-cuda/triattention-score.cu, src/common/arg.cpp]
tags: [kv-cache, eviction, triattention]
---

## Definition

Bounding KV-cache memory by a fixed cell count instead of by context length: when the cache fills, score every resident token, keep the best `budget` of them, and free the rest. The cache stops being an append-only log and becomes a reservoir with a quality-ordered replacement policy.

In this project the policy is TriAttention, and the whole mechanism is four parameters plus a scoring function:

| Parameter | Field | Meaning |
| :--- | :--- | :--- |
| `budget` | `triattention_config::budget` (`llama-triattention.h:131`) | cells to retain after pruning |
| `divide_length` | `llama-triattention.h:132` | pruning interval in decode tokens; also the width of the recent window |
| `prefix_length` | `triattention_state::prefix_length` (`llama-triattention.h:155`) | prompt length; protected when `protect_prefill` |
| `offset_max` | `llama-triattention.h:133` | largest geometric future offset used for scoring |

`prefix_length` is not a config field — it is measured at runtime from the first multi-token batch containing position 0 (`llama-kv-cache.cpp:1354-1363`), so it is the real prompt length of the request in flight. The CLI exposes `budget` as `--triattention-budget` (`src/common/arg.cpp:4698`), `divide_length` as `--triattention-window` / `--triattention-divide-length` (`:4706`), and `offset_max` as `--triattention-offset-max` (`:4715`); `--triattention-stats` supplies the calibration file that enables the whole mechanism (`:4690`).

### The scheme

1. **Score.** `triattention_score_keys` (`llama-triattention.cpp:434`) walks the dequantised, un-rotated keys and accumulates, per frequency `f` of the RoPE basis, a trigonometric term `amp · freq_scale_sq · cos(ω_f · Δ + φ)` (`llama-triattention.cpp:487`) and a position-independent norm term `extra_weight · freq_scale_sq · |k_f|` (`llama-triattention.cpp:491`). Amplitude is `||E[q_f]|| · |k_f|` and the phase comes from `E[q_f] · conj(k_f)` (`:474-482`) — the query statistics come from a calibration file, not from the live query. The cosine is evaluated at a set of geometric offsets `Δ + δ`, `δ ∈ {1,2,4,…,offset_max}`, and aggregated by max or mean (`:494-503`); this is the "trigonometric series" — a prediction of how much attention this key *will* attract at a range of future distances, not how much it attracted in the past.
2. **Norm-based selection.** The `|k_f|` term is the part of the score that does not depend on position at all; with `disable_trig = true` it becomes the entire score (`:506-513`), reducing eviction to "keep the keys with the largest weighted norms". The weight `extra_weight` is the MLR norm excess `E[||q_f||] − ||E[q_f]||`, built at init (`llama-triattention.cpp:344-358`, applied in `triattention_precompute_head_derived`).
3. **Protect.** Before any selection, the pruning routine counts two protected classes — prefix cells when `protect_prefill` is set, and everything at or after `recent_threshold = max_pos − divide_length + 1` (`llama-triattention.cpp:1128`, classification loop `:1136-1146`). The recent window is mandatory regardless of the flag: the comment at `:1114-1121` says evicting the highest-position token would make the server's position counter inconsistent.
4. **Select and drop.** Protected cells are removed from the candidate set; the remainder is scored, optionally z-normalised per head (`zscore_normalize`, `:829`) and combined across sampled heads, then `top_k_indices` (`:851`) fills `keep_indices` (`:1345`, `:1399`, `:1421`) and every candidate outside it has its cell position invalidated (`:1431-1439`).
5. **Trigger.** Either every `divide_length` tokens once the cache is at budget, or only once it has grown to `budget + divide_length` (`triattention_should_prune`, `:803`, cases at `:811` / `:816`, `TRIATTENTION_TRIGGER_INTERVAL` / `_SLACK`).

## Why it matters here

Eviction is orthogonal to [[quantization]]: quantisation shrinks each cell, eviction removes cells. Together they are what make the target's context length physical on 16 GB.

The tension is structural, and it is the one thing to understand about this design:

- The budget is a **fixed constant**, but the protected set is **proportional to the prompt**. Prefix protection reserves `prefix_length` cells and the recent window reserves `divide_length`; the budget for scored history is whatever is left: `decode_budget = (budget > n_protected) ? (budget − n_protected) : 0` (`llama-triattention.cpp:1150`). At the shipped configuration — budget 4096, window 512, and a system prompt above ~3500 tokens — the prefix plus the window consume the entire budget and `decode_budget` collapses to zero.
- When that happens TriAttention **degenerates into a plain sliding window**: the only survivors are the prompt head and the last 512 tokens, and everything between them is evicted in one pass. The scoring pipeline still runs and still costs GPU/CPU time, for `B = 0` selections. That is [[ta-2-budget-starvation]].
- Nothing warns about it: `triattention_init` does not compare `budget`, `prefix_length` and `divide_length` (`[[ta-7-config-validation]]`), and `prefix_length` is only known once the first request arrives, so a config that is fine for one prompt starves on the next.

The contrast with a sliding window is therefore sharper than "a special case": a sliding window is *defined* by a fixed recent span and no importance notion; TriAttention carries a scoring apparatus whose entire purpose is to keep distant-but-relevant tokens, and starvation silently discards exactly the population that apparatus exists to select. The measured symptom is not memory or speed but quality — long agent runs lose reasoning context and speculative acceptance falls ([[source-state-md]] §3 TA-2).

Two further couplings:

- Eviction happens *after* the token is written, so the write path (`SET_ROWS` + WHT) and the scoring path (`triattention-score.cu`) must agree on the key representation. The GPU scorer inverts the WHT only for `padded_hd == 128` (`triattention-score.cu:225`), so at `head_dim=256` it scores rotated keys — eviction is effectively random ([[ta-1-wht-inversion-256]]).
- When the GPU scorer is unavailable the CPU fallback dequantises cell-by-cell with a synchronous copy per cell (`triattention_dequant_kv_head`), turning one prune into thousands of D2H transfers ([[ta-3-cpu-fallback-transfers]]).

## Tradeoffs

- **Score fidelity vs eviction cost.** Scoring is O(cells × frequencies × offsets) per prune and is the only part of the design with a tunable accuracy/cost knob (`offset_max`, aggregation mode, `disable_trig`). The recorded 0.91 % GPU-time share ([[source-state-md]] §1.3) says the cost side is not the problem — but that figure is from a run whose budget may already have been starved, where the work produced `B = 0`.
- **Fixed budget vs varying prompt.** A constant `budget` makes memory predictable and the KV tensor statically sized, which is what allows CUDA-graph capture. The cost is that the residual history budget swings with the prompt, all the way to zero ([[ta-2-budget-starvation]]).
- **Recent window vs positional consistency.** Keeping the last `divide_length` cells is not a quality choice but a correctness requirement of the server's position bookkeeping (`llama-triattention.cpp:1114-1121`) — it removes degrees of freedom from the policy for reasons that have nothing to do with attention.
- **Calibration coupling.** Scores depend on a `.triattention` file carrying head statistics and `head_dim`/`n_kv_heads`/`rope_theta`; a mismatch aborts init or warns (`triattention_init`, `:630`, validation at `:645-661`). The eviction policy is a property of the checkpoint, not of the code.
- **Recompute vs retain.** Evicted tokens cannot be recovered without re-prefilling. Every eviction decision is irreversible within a sequence, so a scoring error is a permanent loss of context rather than a temporary slowdown.

## Open questions

- Does the 0.91 % TriAttention GPU-time share reflect a healthy prune or a starved one? The two are indistinguishable in the recorded profile ([[performance-profile]]).
- What is the actual `n_protected` distribution on the workloads the project targets? `[UNVERIFIED]` — no logged prune with the counters `[prefix=%lld, recent=%d]` (`llama-triattention.cpp:1449-1452`) has been captured in the wiki.
- Whether `prefix_length` being set only on the first batch containing position 0 behaves correctly across sequences with different prompt lengths in one server session — `[UNVERIFIED]`.

## See also

[[triattention]] · [[kv-cache]] · [[quantization]] · [[ta-2-budget-starvation]] · [[ta-1-wht-inversion-256]] · [[ta-6-overlap-double-counting]] · [[ta-7-config-validation]] · [[performance-profile]]
