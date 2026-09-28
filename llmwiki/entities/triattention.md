---
title: TriAttention
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md, TRIATTENTION.md, TRIATTENTION-API.md]
verified:
  - src/src/llama-triattention.h
  - src/src/llama-triattention.cpp
  - src/src/llama-kv-cache.cpp
  - src/src/llama-kv-cache.h
  - src/src/llama-context.cpp
  - src/src/llama-graph.cpp
  - src/src/turbo-rotation-data.h
  - src/include/llama.h
  - src/common/arg.cpp
  - src/common/common.h
  - src/common/common.cpp
  - src/ggml/include/ggml-cuda.h
  - src/ggml/src/ggml-cuda/triattention-score.cu
  - src/ggml/src/ggml-cuda/triattention-score.cuh
  - src/ggml/src/ggml-cuda/turbo-quant.cuh
  - src/ggml/src/ggml-cuda/dequantize.cuh
  - src/tools/triattention-calibrate/triattention-calibrate.cpp
  - src/tools/triattention-calibrate/CMakeLists.txt
  - calibration/bonsai-27b.triattention
  - scripts/start_server_turbo.sh
  - scripts/run_cli.sh
tags: [triattention, kv-eviction, kv-cache, calibration, cuda]
---

# TriAttention

## What it is

**Calibration-guided KV cache eviction**: it scores every token resident in the KV cache and drops the lowest-scoring ones, keeping the cache inside a fixed `budget`. It is the "reduce token count" axis of the fork, paired with [[turboquant]]'s "reduce bits per token" axis — the header's own framing is `TurboQuant compresses each KV entry to 2-4 bits … TriAttention evicts unimportant entries entirely … Combined: ~40x effective KV memory reduction` (`src/src/llama-triattention.h:9-13`). The scoring rule follows arXiv 2604.04921 (*TriAttention: Trigonometric KV Cache Eviction*, cited at `:4`).

The system has two halves that run at different times and in different processes:

1. an **offline calibration pass** that records pre-RoPE query statistics per (layer, attention head, frequency band) into a `.triattention` file, and
2. a **runtime pruning pass** that uses those statistics to predict which cached keys a future query distribution would attend to, and evicts the rest.

Pruning is invoked from the **KV cache**, not from the graph: `llama_kv_cache::apply_ubatch()` → `triattention_should_prune()` → `triattention_try_prune()` → `triattention_prune_impl()` (`src/src/llama-kv-cache.cpp:1351-1375`, `:3015-3069`). `src/src/llama-graph.cpp` never references TriAttention — it only knows the TurboQuant rotation op, which is why the two stacks meet only through the cache tensors.

## How it works

### 1. Offline calibration

`src/tools/triattention-calibrate/triattention-calibrate.cpp` decodes a text corpus through the model with an eval callback installed (`params.cb_eval = triattention_calibrate_cb_eval`, `:191`). The callback keeps tensors whose name contains `Qcur-<il>` (`parse_qcur_layer`, `:44`), i.e. the **reshaped 3-D Q tensor before RoPE** with shape `[head_dim, n_head, n_tokens]` (`:61-70`), and accumulates, for every `(layer, head, freq band f < head_dim/2)`:

```c
const float re = q_ptr[k];            // ":126-127" — Half layout: real in [0, fc), imag in [fc, 2fc)
const float im = q_ptr[k + fc];
h_acc.sum_real[k] += re;  h_acc.sum_imag[k] += im;  h_acc.sum_abs[k] += sqrtf(re*re + im*im);
```

It writes the format documented at `src/src/llama-triattention.h:26-60`: header (magic `0x54524941`, version `1`, `head_dim`, `num_layers`, `num_attn_heads`, `num_kv_heads`, `rope_theta` as f64, `rope_style`, `n_sampled`, `freq_count = head_dim/2`, name length + name) followed by, per sampled head, `layer_idx`, `head_idx`, and four `f32[freq_count]` arrays `q_mean_real`, `q_mean_imag`, `q_abs_mean`, `r_f`. `r_f` is validation-only: the runtime loader reads and discards it (`src/src/llama-triattention.cpp:243-251`).

Two hardcoded choices in the writer: `rope_style = 0` always (`:299`) and `model_name = "Bonsai-2-27B-PQ2_0"` always (`:296`) regardless of the model being calibrated.

The loader `triattention_load_calibration()` (`src/src/llama-triattention.cpp:107-285`) validates the header (`freq_count == head_dim/2` at `:170`; non-zero head counts that divide evenly at `:178`), validates each sampled `(layer, head)` index against the header (`:219-224`), and drops `r_f`.

### 2. Init-time precomputation

`triattention_init()` (`:630`) — reached through `llama_triattention_init()` (`src/src/llama-context.cpp:4354`, declared `src/include/llama.h:827`) and `llama_kv_cache::init_triattention()` (`src/src/llama-kv-cache.cpp:2995`, which supplies `kv_size`, `rope_freq_base_train`, `n_embd_head_k(0)`, `n_head_kv(0)`) — checks model/calibration agreement:

- `head_dim` must match exactly (`:642-649`),
- `n_kv_heads` must match exactly (`:650-655`),
- `rope_theta` only warns when it differs by more than 1 % (`:657-661`).

It then builds four arrays, all of which are used verbatim by the scoring loop:

| Array | Formula | Site |
| :--- | :--- | :--- |
| `omega[f]` | `rope_theta^(-2f/head_dim)` | `:309-315` |
| `freq_scale_sq[f]` | `cosf(omega[f]*0)^2 + sinf(omega[f]*0)^2` — **always `1.0f`** | `:320-330` |
| `offsets[d]` | geometric `{1, 2, 4, …, offset_max}` | `:333-343` |
| `extra_weight[f]` | `E[‖q_f‖] − ‖E[q_f]‖` (MLR norm excess), clamped ≥ 0 for `disable_mlr=false` | `:345-367` |

State also carries `cell_positions[kv_size]` (`int32_t`, `-1` = empty) — the O(1) absolute-position lookup that makes RoPE inversion correct after previous evictions — plus scratch buffers `dequant_buf`, `unrot_buf`, `score_buf[n_sampled × kv_size]`, `combined_buf`, and `keep_indices[budget]` (`:675-703`).

### 3. Runtime scoring

For each cached key, the CPU path runs three stages per sampled head:

1. **Dequantize** the K row for one KV head: `triattention_dequant_kv_head()` (`:540-627`) walks the candidate cells and, for every cell, calls `ggml_backend_tensor_get()` for that one head's bytes (`:572`), dequantizes per type (turbo3/turbo4/turbo2/q8_0/f16/bf16/f32), then multiplies each 128-element block by the dense inverse rotation `TURBO_ROTATION_RT` via `matvec_128()` (`:90-103`, call at `:619-621`) — but only when `need_wht_inv` is true, which is `k_type == TURBO2_0 || k_type == TURBO3_0` (`:1261`, `:899`).
2. **Invert RoPE**: `triattention_invert_rope()` (`:382-432`) turns post-RoPE keys back into pre-RoPE keys using each cell's absolute position, for both the `half` (`:395-406`) and `interleaved` (`:407-420`) layouts.
3. **Score**: `triattention_score_keys()` (`:434-538`), one `(layer, attention head)` at a time. Per frequency band `f`, with `k_re = k[f]`, `k_im = k[f+freq_count]` (the code always scores in `half` layout — `:465-467`):

```c
amp  = stats->q_mean_abs[f] * |k_f|;                       // ‖E[q_f]‖ · |k_f|
phi  = atan2f(q_im*k_re - q_re*k_im, q_re*k_re + q_im*k_im);   // angle(E[q_f] · conj(k_f))
offset_score += amp * freq_scale_sq[f] * cosf(omega[f]*delta + phi)     // trig term
              + stats->extra_weight[f] * freq_scale_sq[f] * |k_f|;      // norm-excess term
```

with `delta = (round_start − key_position) + offsets[d]`, `round_start = state->absolute_position`. `freq_scale_sq` being identically 1 makes the trig term a plain cosine — see [[ta-5-freq-scale-dead-code]]. Aggregation over the geometric offsets is `mean` (default, `total_score *= 1.0f/n_offsets`) or `max` (`:493-502`). `disable_trig` skips the trig term and keeps only `extra_weight × |k|` (`:504-514`). Note that the norm term sits *inside* the offset loop, so under `agg=max` it participates in the max rather than adding a constant.

### 4. The GPU scoring kernel

`src/ggml/src/ggml-cuda/triattention-score.cu` reimplements stages 1-3 as one kernel: grid `(n_cells, 1, 1)`, block `(freq_count, 1, 1)`, `smem = (hd + fc) * sizeof(float)` (`:352`). Per block (= one cache position) it dequantizes the head row into shared memory (`dequant_head_to_smem`, `:93-160`, one branch per supported `K_TYPE`), applies the inverse WHT rotation (`inverse_wht_rotation_128`, `:72-85`, built on `cooperative_fwht_128`, `:47-69`), applies inverse RoPE (`:233-256`), computes the same per-frequency score, then block-reduces to one float per position (`:304-321`). The launch dispatcher switches on `cfg.k_type` and passes `NEED_WHT_INV = true` only for `TURBO2_0`/`TURBO3_0` (`:366-392`).

Two things matter for the target model:

- The inverse-rotation block is `for (b = 0; b < padded_hd; b += 128) { if (f < 64) { /* empty body */ } }` followed by `if (padded_hd == 128 && f < 64) inverse_wht_rotation_128(k_smem, f);` (`:212-230`). `Ternary-Bonsai-2-27B` has `head_dim = 256`, so `padded_hd = 256` and **no inversion happens at all** — [[ta-1-wht-inversion-256]]. The CPU fallback does not share this bug: its `for (b = 0; b < padded_hd; b += 128)` loop calls `matvec_128` for every 128-block (`src/src/llama-triattention.cpp:619-621`).
- The kernel gets `hd = cfg.head_dim` as its `padded_hd` argument and derives the per-head byte offset as `ggml_row_size(cfg.k_type, kv_head_idx * hd)` (`:368`), i.e. from the *unpadded* head dimension, while the CPU path uses `padded_hd = ceil(hd/128)*128` (`:947`, `:1092`). For `head_dim = 256` the two coincide (256 is already 128-aligned); for a padded model they would not. `[INFERENCE]` — no padded-head model was exercised.

The device ↔ host boundary is small: `triattention_gpu_init()` uploads the calibration arrays once (`:397-455`), each head's kernel writes into a packed device score buffer, and a single `triattention_gpu_scores_to_host()` copy brings `n_sampled × n_decode` floats back (`triattention-score.cu:485-495`, called at `src/src/llama-triattention.cpp:1218`). The GPU path passes `kt->data` straight to the kernel (`:1203`), i.e. it assumes the K tensor lives in a CUDA buffer; neither `triattention_init_gpu()` nor `triattention_prune_impl()` checks the tensor's backend.

### 5. Trigger, protection and budget arithmetic

`triattention_should_prune()` (`:803-820`) is called once per `apply_ubatch`:

```c
case TRIATTENTION_TRIGGER_INTERVAL:   // default
    return n_used >= state->cfg.budget && state->absolute_position > 0 &&
           (state->absolute_position % state->cfg.divide_length) == 0;
case TRIATTENTION_TRIGGER_SLACK:
    return n_used >= (state->cfg.budget + state->cfg.divide_length);
```

`prefix_length` is latched on the first multi-token batch that contains position 0 (`src/src/llama-kv-cache.cpp:1354-1363`), which is what makes prompt protection meaningful.

`triattention_prune_impl()` (`:1077-1459`) is the workhorse — the header's *other* entry point, `triattention_prune(state, kv)`, is a stub that enumerates cells and always `return 0` (`:934-1060`, comments at `:1048-1059`), and has no callers anywhere under `src/`.

The real algorithm, in order:

1. **Enumerate** occupied cells from `cell_positions` (`:1096-1111`); bail out if `n_occupied <= budget`.
2. **Partition** into protected and candidate cells. Two protection classes are counted together into `n_protected`: prompt prefix (`protect_prefill && pos < prefix_length`) and the recent window (`pos >= max_pos − divide_length + 1`) (`:1127-1146`). The recent window is not cosmetic: the comment at `:1118-1122` records that evicting the highest-position token would break the server's `seq_pos_max` bookkeeping.
3. `decode_budget = (budget > n_protected) ? (budget − n_protected) : 0` (`:1150`); bail if `n_decode <= decode_budget`.
4. **Score** every candidate cell for every sampled head — GPU path at `:1166-1238` (one kernel launch per sampled head, all on the default stream), CPU fallback at `:1244-1301`.
5. **Combine** per mode (below) and select `decode_budget` winners with `top_k_indices()` (`:851-883`, a `std::partial_sort` on descending score).
6. **Evict**: every candidate not in the keep-set gets `cell_positions[cell] = -1` (`:1424-1438`); the caller then calls `cells.rm(i)` for those cells and rewinds the ring head (`src/src/llama-kv-cache.cpp:3040-3058`). Position *gaps* are deliberately left in place (`:3059-3066`).

### 6. The three modes — as implemented

| Mode | CLI | `combined[i]` | Selection |
| :--- | :--- | :--- | :--- |
| `global` (0, default) | `--triattention-mode global` | max over *all* sampled heads, after optional per-head z-score and per-head tie-break noise | one global top-`decode_budget` (`:1307-1346`) |
| `per-kv-head` (1) | `--triattention-mode per-kv-head` | first pass: max within each KV head's group, folded into a running max ⇒ still the max over all sampled heads; if `normalize` is set, the array is z-scored and `combined` is **recomputed as the max over all sampled heads** | one global top-`decode_budget` (`:1347-1400`) |
| `per-layer-head` (2) | `--triattention-mode per-layer-head` | **mean** over all sampled heads (optionally after z-score) | one global top-`decode_budget` (`:1401-1422`) |

So the code produces a **single global keep-set in all three modes**; the only differences are the aggregation used to rank tokens and when normalisation is applied. The two documented behaviours that are *not* implemented are "each KV head independently selects its own top-B tokens" and "each (layer, KV head) pair selects independently" (`src/docs/TRIATTENTION.md` §*Pruning modes*, `src/docs/TRIATTENTION-API.md` §*triattention_mode*). The reason is structural: the KV cache stores all KV heads of a layer in one row of one tensor, so cell-level `cells.rm()` cannot express a per-head keep-set. The comments at `:1374-1376` (`// simplified: we use the per-KV-head max as the combined score`) and the empty `if (cfg.normalize_scores) { … }` body inside the per-KV-head loop (`:1381-1384`) are the fingerprints.

### 7. Knobs, as the parser defines them

`src/common/arg.cpp:4690-4831` registers the flags; defaults come from `src/common/common.h:749-766`. Eight of the flags carry a `LLAMA_ARG_TRIATTENTION_*` environment fallback.

| Flag(s) | Default | Notes |
| :--- | :--- | :--- |
| `--triattention-stats PATH` | `""` | enables the feature; `LLAMA_ARG_TRIATTENTION_STATS` |
| `--triattention-budget N` | `0` | `LLAMA_ARG_TRIATTENTION_BUDGET`; **see the hazards below** |
| `--triattention-window N` / `--triattention-divide-length N` | `0` | same option, both spellings; sets both fields; `LLAMA_ARG_TRIATTENTION_WINDOW` |
| `--triattention-offset-max N` | `0` | `LLAMA_ARG_TRIATTENTION_OFFSET_MAX`; **see the hazards below** |
| `--triattention-mode` | `global` | `global` \| `per-kv-head` \| `per-layer-head` |
| `--triattention-trigger` | `interval` | `interval` \| `slack` |
| `--triattention-agg` | `mean` | `mean` \| `max` |
| `--triattention-seed N` | `-1` | RNG seed for tie-break noise; `-1` disables it |
| `--triattention-normalize` / `--triattention-no-normalize` | `true` | z-score per head before selection |
| `--triattention-protect-prefill` / `--triattention-no-protect-prefill` | `true` | prefix protection |
| `--triattention-disable-mlr` / `--triattention-no-mlr` | `false` | algebraically replaces `extra_weight` with `q_abs_mean` |
| `--triattention-disable-trig` / `--triattention-no-trig` | `false` | norm-only scoring |
| `--triattention-log` | `false` | per-prune stderr line |
| `--triattention-calibrate PATH`, `--triattention-calibrate-out PATH` | `""` | parsed into `params.triattention_calibrate` / `_out`, **never read anywhere** |

`src/common/common.cpp:1403-1424` auto-initialises when `--triattention-stats` is non-empty **or** `budget > 0`, deriving `divide_len = divide_length > 0 ? divide_length : window` and warning (rather than failing) when init fails.

### 8. Verified configuration hazards (not in the issue inventory)

All three follow from the defaults above and are stated with the code that produces them. The arithmetic consequences are `[INFERENCE]` — they were checked as float32 arithmetic, not by running the engine (no CUDA toolkit on this machine, per [[codebase-map]]).

- **`--triattention-offset-max` defaults to `0`, which yields `n_offsets == 0`.** `triattention_build_offsets()` is `for (d = 1; d <= offset_max; d *= 2) offsets[n++] = d;` (`:333-343`), so `offset_max = 0` writes nothing and `triattention_init()` stores `n_offsets = 0` (`:684`). `triattention_score_keys()` then computes `inv_n_offsets = 1.0f/0.0f = +inf` (`:450`) and multiplies a zero accumulator by it (`:502`), and the GPU kernel divides by the same zero (`triattention-score.cu:296`): both produce `NaN` importance for every token, after which `top_k_indices()`'s `a > b` comparator is false for every pair. Neither the shipped `scripts/start_server_turbo.sh` nor `scripts/run_cli.sh` passes this flag, and the parser's own help string prints `(default: 0)`.
- **`--triattention-window` defaults to `0`, which is a divisor in the trigger.** `triattention_should_prune()` computes `absolute_position % cfg.divide_length` (`:812-813`); with the parser default of `0` and a stats file supplied, the first prune check is an integer modulo by zero. `[INFERENCE]` on the consequence (UB; `SIGFPE` on x86-64).
- **`budget = 0` with a stats file evicts every unprotected token.** `keep_indices` is `new uint32_t[0]`, `decode_budget` is `0`, `top_k_indices(…, k = 0)` selects nothing, and the keep-set stays empty (`:1424-1438`) — a total-history eviction on the first trigger. This is the `decode_budget = 0` regime of [[ta-2-budget-starvation]], reached by default rather than by a long prompt.

`> Contradiction (2026-09-28):` `src/docs/TRIATTENTION.md` §*CLI arguments* documents `--triattention-budget` as `2048`, `--triattention-window` as `128`, `--triattention-offset-max` as `65536`, `--triattention-seed` as `0` and `--triattention-normalize` as `off`. `src/common/common.h:752-766` holds `0`, `0`, `0`, `-1` and `true`. The doc is recorded, not overwritten; the parser defaults are what the binary ships, and `--triattention-offset-max`/`--triattention-window` are the two where the difference is not cosmetic (see above).

`> Contradiction (2026-09-28):` [[source-readme]] *Example 4* invokes the calibration tool as `llama-triattention-calibrate -m model.gguf --triattention-calibrate corpus.txt --triattention-calibrate-out my_model.triattention`. The tool parses those two flags (they are registered for `LLAMA_EXAMPLE_CLI`) but never reads the resulting fields — repo-wide, `params.triattention_calibrate` and `params.triattention_calibrate_out` have no consumer outside `arg.cpp`. Its own `print_usage()` (`triattention-calibrate.cpp:150-166`) advertises `-f, --file PATH` (which sets `params.prompt`) and `-o, --output PATH` (which sets `params.out_file`), and `main()` gates on exactly those two (`:174-186`). The README's invocation therefore exits with `no calibration text provided (use -f FNAME)`.

`> Contradiction (2026-09-28):` `src/docs/TRIATTENTION.md` §*Quick start* and §*Architecture* name `scripts/calibrate-triattention.py` and `scripts/validate-calibration.py` as the calibration tooling. No such files exist (`scripts/` holds only the llama.cpp upstream scripts; a repo-wide search for both names finds only the doc). The calibration tool that ships is the C++ binary `llama-triattention-calibrate` built from `src/tools/triattention-calibrate/` (`CMakeLists.txt` target `llama-triattention-calibrate`); it is the one described above.

`> Contradiction (2026-09-28):` `src/docs/TRIATTENTION-API.md` lists `triattention_prune_impl()` as taking `(state, k_tensors, layer_map, n_layers, n_kv_heads, padded_head_dim, kv_size, evicted_cells, n_evicted)`, `triattention_invert_rope()` as in-place on `k`, and `llama_triattention_init()` as returning `bool`. The headers declare `triattention_prune_impl(state, k_tensors, n_layers, layer_map, kv_size)` (`src/src/llama-triattention.h:392`), an out-of-place `triattention_invert_rope(out, post_rope_k, …)` (`:250-259`), and `int32_t llama_triattention_init(…)` (`src/include/llama.h:827`). The struct and enum listings in the same document do match the header.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/src/llama-triattention.h` (393 lines) | `.triattention` format block `:26-60`; `TRIATTENTION_MODE_*`/`_TRIGGER_*`/`_AGG_*` enums `:66-101`; `triattention_head_stats`/`_calibration`/`_config`/`_state` structs `:107-216`; C API `:224-380`; `triattention_prune_impl()` `:392` |
| `src/src/llama-triattention.cpp` (1492 lines) | `triattention_load_calibration` `:107`; `triattention_build_omega` `:309`; `triattention_build_freq_scale_sq` `:320`; `triattention_build_offsets` `:333`; `triattention_precompute_head_derived` `:345`; `triattention_invert_rope` `:382`; `triattention_score_keys` `:434`; `triattention_dequant_kv_head` `:540`; `triattention_init` `:630`; `triattention_should_prune` `:803`; `zscore_normalize` `:829`; `top_k_indices` `:851`; `triattention_init_gpu` `:885`; `triattention_prune` (stub) `:934`; `triattention_prune_impl` `:1077`; `triattention_print_stats` `:1464` |
| `src/ggml/src/ggml-cuda/triattention-score.cu` (542 lines) | `cooperative_fwht_128` `:47`; `inverse_wht_rotation_128` `:72`; `dequant_head_to_smem` `:107`; `triattention_score_kernel` `:168`; `launch_score_kernel` `:325`; `triattention_gpu_init` `:397`; `triattention_gpu_score_head` `:457`; utility API `:490+` |
| `src/ggml/src/ggml-cuda/triattention-score.cuh` (19 lines) | include-only header; the API lives in `ggml-cuda.h` |
| `src/ggml/include/ggml-cuda.h:44-106` | `triattention_gpu_state`, `triattention_gpu_head_calib`, `triattention_gpu_config` and the seven `triattention_gpu_*` entry points |
| `src/src/llama-kv-cache.cpp` / `.h` | include `:19`; hooks `:517-528`, `:580`, `:605`, `:791`, `:1302-1307`; trigger `:1351-1375`; `init_triattention` `:2995`; `triattention_try_prune` `:3015`; `has_triattention` `:3071`; state member `llama-kv-cache.h:344` |
| `src/src/llama-context.cpp:4354-4407` | `llama_triattention_init()` — mode/trigger/agg enums cast from `int32_t`, then `kv->init_triattention()` |
| `src/include/llama.h:818-829` | public declaration |
| `src/common/arg.cpp:4690-4831`, `src/common/common.h:749-766`, `src/common/common.cpp:1403-1424` | CLI + env + auto-init |
| `src/src/turbo-rotation-data.h` (4103 lines) | `TURBO_ROTATION_RT` `:3` and `TURBO_ROTATION_R` `:2054`, the dense 128×128 `f32` rotations the CPU fallback multiplies by |
| `src/tools/triattention-calibrate/triattention-calibrate.cpp` (364 lines) + `CMakeLists.txt` | the `llama-triattention-calibrate` binary |
| `calibration/bonsai-27b.triattention` | the shipped profile — 789,571 B, sha256 `6dff56bab1e4…`; byte-identical to `src/bonsai-27b.triattention` |
| `scripts/start_server_turbo.sh`, `scripts/run_cli.sh` | pass `--triattention-stats … --triattention-budget 4096 --triattention-window 512 --triattention-protect-prefill`, gated on the stats file existing and `DISABLE_TRIATTENTION != 1` |

**The shipped profile, decoded.** Reading the header of `calibration/bonsai-27b.triattention` directly gives `magic 0x54524941`, `version 1`, `head_dim 256`, `num_layers 64`, `num_attn_heads 24`, `num_kv_heads 4`, `rope_theta 1.0e7`, `rope_style 0`, `n_sampled 384`, `freq_count 128`, `name_len 19`, name `Bonsai-2-27B-PQ2_0`; the first per-head record is `(layer 3, head 0)`. `384 = 16 attention layers × 24 heads` matches the hybrid layout of the target model (only 16 of its 64 blocks keep a KV cache — [[ternary-bonsai-2-27b]]), and `freq_count = 128 = head_dim/2` is the value the loader checks at `:170`. The same profile ships inside both release archives at `<root>/calibration/` ([[codebase-map]]).

## Known issues

- [[ta-1-wht-inversion-256]] — **CRITICAL**: the GPU scoring kernel's inverse-WHT block is empty and its guard is `padded_hd == 128`, so the model this page serves (`head_dim = 256`) is scored on keys that still carry the TurboQuant rotation. The CPU fallback loop does invert every 128-block.
- [[ta-2-budget-starvation]] — `decode_budget = budget − n_protected` collapses to `0` once prefix + recent window consume the budget, turning eviction into a sliding window.
- [[ta-3-cpu-fallback-transfers]] — `triattention_dequant_kv_head` issues one synchronous `ggml_backend_tensor_get()` per KV cell.
- [[ta-4-cooperative-fwht-race]] — the 64-thread contract of `cooperative_fwht_128` versus the `freq_count = head_dim/2` launch config; see [[walsh-hadamard-transform]] for why the two guards as written are mutually exclusive.
- [[ta-5-freq-scale-dead-code]] — `freq_scale_sq` is built from `cosf(omega[f] * 0.0f)`, hence identically `1.0f`.
- [[ta-6-overlap-double-counting]] — prefix and recent-window ranges can overlap in the protection loop; today's `is_prefix || is_recent` counts once, a future `max_protected` must too.
- [[ta-7-config-validation]] — `triattention_init` validates the calibration file but not `budget`/`prefix_length`/`divide_length`; this is the family the three hazards in §*Verified configuration hazards* belong to (none of them is filed as an issue yet).
- Three modes, one keep-set: `per-kv-head` and `per-layer-head` both reduce to a single global top-`decode_budget` selection (`:1354-1427`), contrary to both first-party docs. Not filed.
- `triattention_prune()` (`:934`) — the public entry declared in the header — is a no-op that returns `0` and has no callers; only `triattention_prune_impl()` does work.

## See also

[[overview]] · [[v100-sxm2]] · [[ternary-bonsai-2-27b]] · [[kv-cache]] · [[kv-eviction]] · [[walsh-hadamard-transform]] · [[turboquant]] · [[innerq]] · [[quantization]] · [[gemm-dispatch]] · [[performance-profile]] · [[codebase-map]] · [[roadmap]] · [[upstream-lineage]] · [[benchmarks]] · [[triattention-calibrate]]
[[source-readme]] · [[source-state-md]] · [[source-triattention]] · [[source-triattention-api]]
[[ta-1-wht-inversion-256]] · [[ta-2-budget-starvation]] · [[ta-3-cpu-fallback-transfers]] · [[ta-4-cooperative-fwht-race]] · [[ta-5-freq-scale-dead-code]] · [[ta-6-overlap-double-counting]] · [[ta-7-config-validation]]
