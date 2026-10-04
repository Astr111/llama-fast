---
title: Source — TRIATTENTION-API.md
type: source
status: current
updated: 2026-09-28
sources: [TRIATTENTION-API.md]
verified: [src/docs/TRIATTENTION-API.md, src/include/llama.h, src/src/llama-triattention.h, src/src/llama-triattention.cpp, src/src/llama-context.cpp, src/src/llama-kv-cache.cpp, src/ggml/include/ggml-cuda.h, src/ggml/src/ggml-cuda/triattention-score.cu, calibration/bonsai-27b.triattention]
tags: [triattention, api, kv-eviction, documentation]
---

# Source — `TRIATTENTION-API.md`

The first-party API reference for TriAttention: the public C entry point, the internal C++ lifecycle and scoring functions, the position-tracking hooks, the data structures, the CUDA scoring API, and the enums. It is the companion to [[source-triattention]] and is the document [[triattention-calibrate]] leans on for the runtime side of the calibration contract.

Its **enums and the CUDA section are accurate; its function signatures are not.** Several declarations are written as they were intended rather than as they are in `src/src/llama-triattention.h`, and the drift is systematic enough to be a finding in itself (the header is the source of truth; anyone coding against this document will not compile).

## Summary

The document defines a three-layer API: a public `llama_triattention_init()` in `include/llama.h` that wires TriAttention onto a `llama_context`; an internal C++ API in `src/llama-triattention.h` (`triattention_init` / `triattention_free`, the RoPE-inversion and scoring kernels, the pruning pipeline `triattention_prune_impl`, four position-tracking hooks, `triattention_print_stats`); and a CUDA scoring API in `ggml/include/ggml-cuda.h` that scores keys on-device and copies back only one float per position. It documents the three structs (`triattention_config`, `triattention_head_stats`, `triattention_calibration`), the mode/trigger/agg enums, and the parameter semantics of each call.

## Key claims

| Claim | Section |
| :--- | :--- |
| `bool llama_triattention_init(ctx, stats_path, budget, divide_length, offset_max, mode, trigger, agg, seed, normalize, protect_prefill, disable_mlr, disable_trig, enable_logging)` — "Must be called after context creation and before inference begins"; returns `true`/`false` | §C API (include/llama.h) |
| `triattention_init(stats_path, cfg, kv_size, rope_theta, head_dim, n_kv_heads)` and `triattention_free(state)` are the internal lifecycle | §Core lifecycle |
| `triattention_invert_rope` converts dequantized K from post-RoPE to pre-RoPE **in place**; `rope_style` 0=half, 1=interleaved | §Scoring functions |
| `triattention_score_keys` is the scoring kernel: pre-RoPE K + per-head `stats` + `omega` + `freq_scale_sq` + geometric `offsets` + `key_positions` + `round_start` → `[n_keys]` scores, with `agg` (mean/max) and `disable_trig` (norm-only ablation) | §Scoring functions |
| `triattention_prune_impl` is the "full pruning pipeline (called from KV cache)" and takes the K tensors, a layer map, head/layer dimensions, `kv_size`, and **output** arrays `evicted_cells` / `n_evicted`; returns `void` | §Scoring functions |
| Four hooks — `on_token_added`, `on_cell_removed`, `on_position_shift`, `on_reset` — must be called by the KV cache whenever cells change; they maintain the O(1) position lookup | §Position tracking hooks |
| `triattention_print_stats(state)` prints total calls, total tokens evicted, average timing | §Statistics |
| `triattention_config` fields: `budget`, `divide_length`, `offset_max`, `mode`, `trigger`, `agg`, `seed`, `normalize`, `protect_prefill`, `disable_mlr`, `disable_trig`, `enable_logging` | §Structs |
| `triattention_head_stats`: `q_mean_real`, `q_mean_imag`, `q_abs_mean` (measured), plus `q_mean_abs` and `extra_weight` precomputed at init | §Structs |
| `triattention_calibration`: `head_dim`, `num_layers`, `num_attn_heads`, `num_kv_heads`, `num_kv_groups`, `freq_count`, `n_sampled`, `rope_theta`, `rope_style`, `model_name`, `head_stats`, `sample_layer`, `sample_head` | §Structs |
| CUDA path: `triattention_gpu_init` uploads calibration; `triattention_gpu_score_head` launches one block per cache position with `freq_count` cooperating threads (dequant, inverse WHT, inverse RoPE, score); only the score array comes back to the host | §CUDA GPU Scoring API |
| `triattention_mode` 0/1/2 = `GLOBAL` (union-based), `PER_KV_HEAD`, `PER_LAYER_HEAD`; `triattention_trigger` 0/1 = `INTERVAL`, `SLACK`; `triattention_agg` 0/1 = `MEAN`, `MAX` | §Enums |

## Discrepancies worth holding

### The signatures drift from the header — systematically

Comparing §C API and §Internal C++ API against `src/src/llama-triattention.h` and `src/src/llama-context.cpp`:

| Documented | Actual | Where |
| :--- | :--- | :--- |
| `bool llama_triattention_init(...)`, param `normalize` | `int32_t llama_triattention_init(...)`, returns `0` on success / `-1` on failure; param named `normalize_scores` | `src/include/llama.h:827`, `src/src/llama-context.cpp:4354` |
| `triattention_invert_rope(float * k, const float * omega, const int32_t * positions, uint32_t n_keys, …)` — **in place** | `triattention_invert_rope(float * out, const float * post_rope_k, const int32_t * positions, const float * omega, uint32_t n_keys, head_dim, freq_count, rope_style)` — out-of-place, two buffers, and `positions`/`omega` ordered opposite to the document | `src/src/llama-triattention.h:241` |
| `triattention_prune_impl(state, k_tensors, layer_map, n_layers, n_kv_heads, padded_head_dim, kv_size, evicted_cells, n_evicted)` → `void` | `triattention_prune_impl(state, ggml_tensor * const * k_tensors, uint32_t n_layers, const int32_t * layer_map, uint32_t kv_size)` → `int32_t` (cells evicted, or `-1`); argument order differs and four documented parameters — `n_kv_heads`, `padded_head_dim`, and both output arrays — are not in the real signature | `src/src/llama-triattention.h:388`, `src/src/llama-triattention.cpp:1077` |
| `triattention_print_stats(const triattention_state * state)` | `triattention_print_stats(const triattention_state * state, FILE * stream)` | `src/src/llama-triattention.h:364`, `src/src/llama-triattention.cpp:1464` |
| `triattention_calibration.model_name` is `char *`; fields `sample_layer` / `sample_head` | `char model_name[256]` fixed array; fields are `sampled_layer` / `sampled_head` | `src/src/llama-triattention.h:110-128` |
| `triattention_config.budget/divide_length/offset_max` are `int32_t`; field `normalize` | `uint32_t` for all three; field `normalize_scores` | `src/src/llama-triattention.h:130-147` |
| (not documented) | `triattention_should_prune(state, n_used)` exists and is the live trigger decision | `src/src/llama-triattention.h:325`, `src/src/llama-triattention.cpp:803` |

The parameter *semantics* the document gives — head_dim / freq_count / rope_style, the trigger arithmetic, the hook duties — are otherwise correct.

### The "full pruning pipeline" is one of two implementations

The document presents `triattention_prune_impl` as the pipeline. There is a second, larger `triattention_prune(state, llama_kv_cache *)` (`src/src/llama-triattention.cpp:934`) that takes the cache object and does its own enumeration/scoring/selection; it is declared in the header (`:315`) and **called from nowhere in the tree**. The live call chain is `llama_kv_cache::triattention_try_prune()` (`src/src/llama-kv-cache.cpp:3015`) → `triattention_prune_impl`. The document does not mention the dead variant, so nothing here is contradicted — but a reader would not learn that half of the pruning code is unreachable.

### `GLOBAL` is not union-based in the code

The enums section describes mode 0 as `TRIATTENTION_MODE_GLOBAL` — "Union-based global selection", and the header comment (`src/src/llama-triattention.h:60-63`) spells out the intended algorithm: each sampled head picks top-B, the union is taken, then top-B is chosen from the union by combined max-over-heads score. The implementation at `src/src/llama-triattention.cpp:1329-1345` does only the last two steps: it computes `combined[i] = max over heads` and then `top_k_indices(combined, decode_budget)`. No per-head top-B preselection, no union stage. Either the union step was elided or it is documented aspiration. `[INFERENCE]` — the code path is unambiguous, but whether the omission is deliberate is not recorded anywhere.

### Runtime semantics the document does not mention

- The recent-window protection is mandatory and independent of `protect_prefill`: `triattention_prune_impl` always protects positions `>= max_pos − divide_length + 1` (`:1128`), so eviction candidates are fewer than "all non-prompt tokens". See [[ta-2-budget-starvation]].
- Config validation is absent: `triattention_init` checks the calibration file's `head_dim`, `n_kv_heads` and `rope_theta` only (`src/src/llama-triattention.cpp:642-660`); nothing checks `budget` against `divide_length`/`prefix_length` ([[ta-7-config-validation]]).
- `triattention_config`'s documented defaults in the header comment (budget 2048, divide_length 128, offset_max 65536, normalize `false`, seed 0) are the design values; the CLI defaults are `0`, `0`, `0`, `true`, `-1` (`src/common/common.h:752-766`). See [[source-triattention]].

### The CUDA section is faithful

Unlike the C++ section, §CUDA GPU Scoring API matches `src/ggml/include/ggml-cuda.h` call for call: `triattention_gpu_init` (`:67`), `triattention_gpu_score_head` (`:75`), `triattention_gpu_scores_to_host` (`:90`), `triattention_gpu_upload_cells` (`:96`), `triattention_gpu_alloc_scores` / `_free_dev` / `_free` (`:104-106`). The "one block per cache position, `freq_count` threads" claim matches the launch at `src/ggml/src/ggml-cuda/triattention-score.cu:350-351` (`grid(n_cells)`, `block(fc)`); the supported-type switch is at `:364-385` and covers `TURBO2_0/TURBO3_0/TURBO4_0/Q8_0/F16/F32` — note **BF16 is accepted by the CPU dequant path but absent from the GPU kernel**, a gap neither document flags. The document's file path is again relative to the fork's `src/` root: `ggml/include/ggml-cuda.h` → `src/ggml/include/ggml-cuda.h`.

## Pages derived

[[triattention]] · [[triattention-calibrate]] · [[kv-eviction]] · [[kv-cache]] · [[walsh-hadamard-transform]]
[[ta-2-budget-starvation]] · [[ta-7-config-validation]] · [[ta-1-wht-inversion-256]]
[[turboquant]] · [[overview]]

## Provenance

- raw path: `llm-wiki/raw/TRIATTENTION-API.md`
- sha256: `a1e0e46e85a3aca72ebd7607e9acd31a493e5bdca46cfbbe02d309d214b2c130`
- ingested: 2026-09-28
