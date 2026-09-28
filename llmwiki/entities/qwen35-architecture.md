---
title: Qwen3.5 architecture (qwen35)
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: ["/home/ms/Загрузки/Ternary models/Ternary-Bonsai-2-27B-PQ2_0.gguf", calibration/bonsai-27b.triattention, scripts/start_server_turbo.sh, scripts/run_cli.sh, scripts/start_server_baseline.sh, src/src/models/qwen35.cpp, src/src/models/qwen3next.cpp, src/src/llama-arch.cpp, src/src/llama-arch.h, src/src/llama-model.cpp, src/src/llama-model.h, src/src/llama-hparams.cpp, src/src/llama-hparams.h, src/src/llama-memory-hybrid.cpp, src/src/llama-memory-hybrid.h, src/src/llama-memory-recurrent.cpp, src/src/llama-memory-recurrent.h, src/src/llama-kv-cache.cpp, src/src/llama-kv-cache.h, src/src/llama-graph.cpp, src/src/llama-context.cpp, src/src/llama-triattention.cpp, src/src/llama-triattention.h, src/src/llama-vocab.cpp, src/ggml/include/ggml.h, src/ggml/src/ggml-common.h, src/ggml/src/ggml-cuda/triattention-score.cu, src/tools/triattention-calibrate/triattention-calibrate.cpp]
tags: [architecture, hybrid, ssm, kv-cache, model]
---

# Qwen3.5 architecture (`qwen35`)

## What it is

The architecture of the only model this fork serves. Neither [[source-state-md]] nor [[source-readme]] ever names it — they say "27B" and `head_dim=256` — but the GGUF declares `general.architecture = qwen35` (Qwen 3.5 lineage; the loader also knows `qwen35moe` and the closely related `qwen3next`, `src/src/llama-arch.cpp:41-42`), and the loader maps 64 blocks with `n_embd = 5120` to `LLM_TYPE_27B` by name (`src/src/models/qwen35.cpp:32`).

It is a **hybrid attention / state-space model**, and that single fact governs everything the rest of the project measures: **only 16 of its 64 blocks keep a KV cache.** The other 48 are Gated-DeltaNet (linear-attention) blocks whose state is a fixed-size tensor that does not grow with context.

| Property (read from the file) | Value |
| :--- | :--- |
| `qwen35.block_count` | 64 |
| `qwen35.embedding_length` / `feed_forward_length` | 5120 / 17408 |
| `qwen35.attention.head_count` / `head_count_kv` | 24 / 4 (GQA group 6) |
| `qwen35.attention.key_length` / `value_length` | 256 / 256 |
| `qwen35.context_length` | 262144 |
| `qwen35.rope.freq_base` | 1e7 |
| `qwen35.rope.dimension_count` | 64 (**partial RoPE**: 64 of 256 head dims) |
| `qwen35.rope.dimension_sections` | `[11, 11, 10, 0]` (multimodal RoPE) |
| `qwen35.full_attention_interval` | 4 |
| `qwen35.attention.recurrent_layers` | *absent* → the interval rule builds the schedule |
| `qwen35.nextn.predict_layers` | *absent* → no MTP block; `n_layer() = n_layer_all = 64` |
| `qwen35.ssm.*` | `inner_size` 6144, `state_size` 128, `group_count` 16, `conv_kernel` 4, `time_step_rank` 48 |

## How it works

### The block schedule: interval 4, derived in code

The rule lives in the architecture's own loader, not only in metadata (`src/src/models/qwen35.cpp:20-29`):

```cpp
if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
    uint32_t full_attn_interval = 4;
    ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
    }
}
```

`is_recr(i)` is therefore true except when `(i+1) % 4 == 0`, i.e. the **full-attention layers are 3, 7, 11, …, 63** (0-based) — 16 of them, interleaved every fourth block. `qwen35.attention.recurrent_layers` is absent from this file, so this derived path is the one that runs; the metadata key `full_attention_interval = 4` matches the code default. The same rule, same default, in the sibling architectures (`src/src/models/qwen3next.cpp:22-26`, `src/src/models/qwen35moe.cpp:25-29`).

Architecture-level defaults bracket it: `llm_arch_is_recurrent(QWEN35) = true` (`src/src/llama-arch.cpp:1068-1104`) fills `is_recr_impl` with 1 for every layer (`src/src/llama-model.cpp:1422`), and the rule above then punches 16 holes in it. `llm_arch_is_hybrid(QWEN35) = true` (`src/src/llama-arch.cpp:1082-1104`) is what routes the model to the hybrid memory container instead of a plain KV cache. `llm_arch_supports_rs_rollback(QWEN35) = true` (`src/src/llama-arch.cpp:1118-1131`) is the recurrent-state rollback hook speculative decoding needs ([[speculative-decoding]]).

Verified against the file's own tensor names: **16 layers carry `attn_k.weight`/`attn_v.weight`/`attn_q.weight`/`attn_output.weight`** and **48 layers carry `ssm_*`**, with no overlap — the two sets are complementary and cover all 64 blocks.

| | Full-attention block (layers 3, 7, … 63 — 16 blocks) | Gated-DeltaNet block (the other 48) |
| :--- | :--- | :--- |
| Projections | `attn_q` `[5120, 12288]` (query **and** gate, interleaved), `attn_k` `[5120, 1024]`, `attn_v` `[5120, 1024]`, `attn_output` `[6144, 5120]` | `attn_qkv` `[5120, 10240]`, `attn_gate` `[5120, 6144]` |
| Norms | `attn_q_norm` `[256]`, `attn_k_norm` `[256]` | `ssm_norm` `[128]` |
| Recurrence | — (KV cache) | `ssm_conv1d` `[4, 10240]`, `ssm_dt` `[48]`, `ssm_a` `[48]`, `ssm_alpha`/`ssm_beta` `[5120, 48]`, `ssm_out` `[6144, 5120]` |
| Shared in every block | `attn_norm`, `post_attention_norm`, `ffn_gate`/`ffn_up` `[5120, 17408]`, `ffn_down` `[17408, 5120]` | same |

The full-attention path is a Qwen3-Next-style block: one fused Q+gate projection split by view offsets, per-head RMS norms on Q and K, then `ggml_rope_multi` on both, then attention, then a **sigmoid gate on the attention output** before `attn_output` (`src/src/models/qwen35.cpp:322-410`; the shapes above are the GGUF's). The GDN path builds the pre-normed QKV, the gate `z`, `beta = sigmoid(...)`, `alpha` → softplus → `gate = softplus(alpha + dt) * ssm_a`, a short conv with kernel width 4 over `conv_dim = 2·2048 + 6144 = 10240`, and a recurrent scan into per-layer state (`src/src/models/qwen35.cpp:411-560`; the graph class derives from `llm_build_delta_net_base`).

### Memory: 64 blocks, 16 KV caches, 48 fixed states

The hybrid memory container splits the two: the KV cache is constructed with the filter `!hparams.is_recr(il)` and the recurrent memory with `is_recr(il)` (`src/src/llama-memory-hybrid.cpp:44-64`), which for `qwen35` is spelled out as `il < hparams.n_layer() && !hparams.is_recr(il)` / `… && hparams.is_recr(il)` (`src/src/llama-model.cpp:2787-2793`). The KV cache honours the filter while allocating its per-layer tensors (`src/src/llama-kv-cache.cpp:194`, `:394`), and reports the resulting count in its own log line (`:455-459`, `"(%6u cells, %3d layers, …)"`).

**KV side — 16 layers.** With `n_embd_k_gqa = n_embd_head_k · n_head_kv = 256 · 4 = 1024` (`src/src/llama-hparams.cpp:131-141`) and the same for V, one token costs **2048 elements per attention layer**, i.e. **32 768 elements per token across the model** — a quarter of the 131 072 a dense 64-layer model of the same shape would hold.

Bytes, from the block layouts in code (`block_turbo3_0` = 2 + 32 + 16 = 50 B per 128 values, `block_turbo2_0` = 2 + 32 = 34 B per 128, `q8_0` = 2 + 32 = 34 B per 32 — `src/ggml/src/ggml-common.h:324-333`, `:374-381`, sizes taken from the `static_assert`s; the prose comments directly above the `turbo2`/`turbo3` structs still describe a 32-value block — 14 B and 10 B — which contradicts both `QK_TURBO3 = QK_TURBO2 = 128` and the asserted struct sizes, so they are not used here):

| Profile | K row (1024) | V row (1024) | Per layer | Per token (×16) | 16K context | Tokens / 1 GiB (KV only) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `f16` K + `f16` V | 2048 B | 2048 B | 4096 B | **64 KiB** | **1 GiB** | 16 384 |
| `turbo3` K + `q8_0` V (the speed profile) | 400 B | 1088 B | 1488 B | **23.25 KiB** | 372 MiB | 45 100 |
| `turbo3` K + `turbo2` V (the max-compression profile) | 400 B | 272 B | 672 B | **10.5 KiB** | 168 MiB | 99 864 |
| *dense-64 model:* `f16` | — | — | 4096 B | 256 KiB | 4 GiB | 4 096 |
| *dense-64 model:* `turbo3` + `turbo2` | — | — | 2688 B | 42 KiB | 672 MiB | 24 966 |

**The load-bearing consequence.** The published capacity column ([[benchmarks]], [[source-readme]]) reproduces the **dense-64** rows of that table, not this model's:

| Published figure | Bytes/token it implies | Closest match |
| :--- | ---: | :--- |
| baseline `~4 000 tok/GB` | 268 435 B | dense-64 `f16` — 256 KiB = 262 144 B (**+2 %**) |
| max-mem `~25 200 tok/GB` | 42 609 B | dense-64 `turbo3`+`turbo2` — 42 KiB = 43 008 B (**−1 %**) |
| speed `~20 000 tok/GB` | 53 687 B | neither: dense-64 `turbo3`+`q8_0` is 93 KiB (11 275 tok/GiB), the 16-layer cache 23.25 KiB (45 100) |

So the two endpoints of the published column are the *dense-64-layer* KV arithmetic — which is also what the repo's own baseline banner states outright: `"KV Cache: FP16 (256 KB/token, 4K tok/GB VRAM)"` (`scripts/start_server_baseline.sh:25`). **This model's cache is a quarter of that** (16 of 64 layers), so the same VRAM buys ~4× the tokens the published column implies. Any capacity planning done from those numbers under-states this model's context capacity by that factor; the derived 16-layer figures are 64 KiB/token (`f16`, 16 384 tok/GiB) and 10.5 KiB/token (`turbo3`+`turbo2`, 99 864 tok/GiB). The third figure does not fit either arithmetic and is recorded as an open question below.

**SSM side — 48 layers.** The recurrent state is F32 (`recurrent_type_k`/`recurrent_type_v` = `GGML_TYPE_F32` in both hybrid constructors, `src/src/llama-model.cpp:2795-2830`) and its size is fixed per layer and per sequence (`src/src/llama-hparams.cpp:183-228`):

| Component | Formula | Elements | Bytes |
| :--- | :--- | ---: | ---: |
| Conv ring (`n_embd_r`) | `(d_conv − 1) · (d_inner + 2·n_group·d_state)` = `3 · (6144 + 4096)` | 30 720 | 120 KiB |
| Recurrent state (`n_embd_s`) | `d_state · d_inner` = `128 · 6144` | 786 432 | 3 MiB |
| **Per SSM layer, per sequence** | | **817 152** | **3.1 MiB** |
| **All 48 layers, one sequence** | | 39 223 296 | **~150 MiB** |

This is the part of the model's memory that **does not respond to context length**: it is allocated at `rs_size = max(1, n_seq_max)` per sequence (`src/src/llama-model.cpp:2809-2810`, `:2828-2829`) and never grows or shrinks with tokens. Context-length pressure therefore acts only on the 16 KV layers, while every concurrent sequence adds its own copy of the 48-layer SSM state on top of its usual share of the KV cache.

### Where TriAttention meets this structure

TriAttention is attached to the KV cache object (`src/src/llama-kv-cache.cpp:2995-3010`), which for this model exists on only 16 layers — and the plumbing already knows it. `triattention_try_prune()` builds its `k_tensors` array and `layer_map` from `layers` (the filter-selected set), and the pruner maps a sampled calibration layer to an internal cache index, zeroing the score row of any sampled head whose layer is not in the cache (`src/src/llama-kv-cache.cpp:3015-3035`; `src/src/llama-triattention.cpp:1182-1231`).

The shipped calibration profile is scoped to match. Parsed from `calibration/bonsai-27b.triattention`: `head_dim = 256`, `num_layers = 64`, `num_attn_heads = 24`, `num_kv_heads = 4`, `rope_theta = 1e7`, `rope_style = 0` (half), `n_sampled = 384`, `freq_count = 128`, name `Bonsai-2-27B-PQ2_0` — and the 384 sampled `(layer, head)` pairs fall on **exactly the 16 attention layers, 24 heads each** (3, 7, … 63). No sampled layer lacks a cache entry.

The consequence is about *accounting*, not correctness: the budget is a single scalar over the cache's used **cells** — one eviction removes a position, and a position is shared across every layer in the cache (`triattention_should_prune()` compares `n_used` against `cfg.budget`, `src/src/llama-triattention.cpp:803-820`; triggered at `src/src/llama-kv-cache.cpp:1373-1374`; cell metadata is per-stream, `v_cells[0]`, `:3040`). So a budget of 4096 tokens is a budget over a cache that is one quarter the size a dense 64-layer model would carry for the same token count, and each evicted position reclaims 32 768 elements of K+V rather than 2048. [[triattention]] owns the scoring design and [[ta-2-budget-starvation]] owns the failure mode on long prefixes; both apply unchanged here.

### Partial RoPE: 64 of 256 dimensions

`n_rot` is read from `qwen35.rope.dimension_count = 64` against a 256-wide head (`src/src/llama-model.cpp:1484-1490`), and the attention path applies `ggml_rope_multi` with `n_rot` and the four-section MRoPE layout in both the trunk and the MTP graph (`src/src/models/qwen35.cpp:365-378`, `:715-718`). `32 = 64/2` = `11+11+10+0`: the sections account for exactly the rotary span, which is a useful internal consistency check on the metadata. The rope type is `LLAMA_ROPE_TYPE_IMROPE` for this architecture (`src/src/llama-model.cpp:3243-3247`). The remaining **192 of 256 dimensions per head are never rotated**, so they carry no positional information — which is the design intent of partial RoPE, and also a trap for anything that assumes a fully-rotary head.

Two things in the tree do assume that, and this page records the mismatch rather than resolving it:

- **TriAttention.** Its calibration invariant is `freq_count == head_dim / 2` (`src/src/llama-triattention.cpp:170-173`), the file carries `freq_count = 128` for `head_dim = 256`, and the scoring kernel pairs `k_smem[f]` with `k_smem[f + freq_count]` for `f ∈ [0, freq_count)` — i.e. it treats all 256 dimensions as 128 rotary pairs (`src/ggml/src/ggml-cuda/triattention-score.cu:186-224` inverse-RoPE step, and the per-cell launch `block(fc)`). The model rotates 64. **Whether the scoring path is meant to see the model's MRoPE layout is an open question** — with `dimension_count = 64`, the pre-RoPE domain TriAttention scores in exists only over a quarter of each head.
- **The KV rotation/quantization stack.** TurboQuant's rotation and the fork's own attention K/V rotation both operate on 128-element groups of the full `head_dim` (`QK_TURBO3 = 128`, `src/ggml/src/ggml-common.h:324-325`; `attn_rot_k` gated on `hparams.n_embd_head_k() % 64 == 0`, `src/src/llama-kv-cache.cpp:462-486`). They are dimension-agnostic, so the non-rotary remainder gets rotated and quantized like the rest — which the cache design accepts by construction but which no source analyses. `[UNVERIFIED]` — whether the non-rotary dims need different treatment is not settled anywhere in the repo.

Also worth noting for [[kv-cache]]: because only 16 layers participate, every per-layer KV knob in the launch scripts (`-ctk`/`-ctv`, `TURBO_LAYER_ADAPTIVE`, boundary-layer precision) acts on a quarter as many tensors as the same flags would on a dense model — the env-var adaptive modes that key off `hparams.n_layer()` still count 64 (`src/src/llama-kv-cache.cpp:283-295`), i.e. they reason in model layers while every KV action lands on the 16 attention layers.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `/home/ms/Загрузки/Ternary models/Ternary-Bonsai-2-27B-PQ2_0.gguf` | the `qwen35.*` key set, the 64-block tensor layout, the 16/48 split |
| `src/src/models/qwen35.cpp` | hparams + the interval rule `:4-35`; per-block tensor construction, attention vs GDN branch at `:70` (`:37-128`); the trunk graph and its recurrent/attention dispatch `:137-215`; full-attention builder `:322-402`; GDN builder `:403-584`; dense FFN builder `:585-600`; MTP graph `:601-771` |
| `src/src/llama-arch.cpp`, `src/src/llama-arch.h` | arch name `qwen35` `:41-42`; `full_attention_interval` key `:255`; ssm keys `:334-339`; ssm tensor names `:481-503`; recurrent/hybrid/rollback classification `:1068-1131` |
| `src/src/llama-model.cpp` | arch dispatch `:320-323`; rope/head-dim parsing incl. `n_rot` `:1478-1512`; rope type `:3243-3247`; hybrid memory construction and the QWEN35 filters `:2772-2830`; `is_recr`-driven block sizes `:656-663` |
| `src/src/llama-hparams.cpp` / `.h` | `is_recr()` `:231-236`, `n_layer()` `:301-303`, `n_rot()` `:85-91`, `n_embd_k_gqa`/`n_embd_v_gqa` `:131-141`, recurrent sizes `n_embd_r()` `:183-205` / `n_embd_s()` `:207-229` |
| `src/src/llama-memory-hybrid.cpp` / `.h` | KV vs recurrent memory split `:44-64`; `get_attn()` / `get_recr()` context accessors `:132-133` |
| `src/src/llama-memory-recurrent.cpp` / `.h` | per-layer state tensors `cache_r_l%d`/`cache_s_l%d` `:100-107`; allocation from `n_embd_r()`/`n_embd_s()` |
| `src/src/llama-kv-cache.cpp` / `.h` | layer filter on allocation `:194`, `:394`; layer/state log line `:455-459`; attention-rotation gating `:462-508`; TriAttention integration `:2995-3075`; prune trigger `:1373-1374` |
| `src/src/llama-triattention.cpp` / `.h` | calibration format + `freq_count == head_dim/2` invariant `:170-173`; sample→cache-layer mapping and zeroing `:1182-1231`; init signature `:209-215`; trigger predicate `:803-820` |
| `src/tools/triattention-calibrate/triattention-calibrate.cpp` | the profile writer: `head_dim/2` → `freq_count` `:277-278`, header fields `:314-321` |
| `calibration/bonsai-27b.triattention` | the shipped 16-layer/24-head profile, 789 571 B |
| `scripts/start_server_turbo.sh`, `scripts/run_cli.sh` | the served KV configuration: `-ctk turbo3 -ctv q8_0` (both), TriAttention budget 4096 / window 512 (both); `-c 32768` (server) vs `-c 16384` (CLI) |
| `scripts/start_server_baseline.sh` | FP16-KV baseline and its `256 KB/token` banner `:25` |

## Known issues

The 16-of-64 structure is what makes the following apply (or not) — the issue pages own the detail:

- [[ta-1-wht-inversion-256]] — **triggered by this model**: `head_dim = 256` is what skips the scoring kernel's inverse rotation. Recorded on [[ternary-bonsai-2-27b]] as well, because it is a `head_dim` consequence rather than a hybrid consequence.
- [[ta-2-budget-starvation]] — the aggregate budget meets a 16-layer cache; see *Where TriAttention meets this structure* for the accounting, not the failure mode.
- [[tq-7-innerq-max-channels]] — 256-wide heads against a 128-channel limit; a per-head property of this architecture.
- [[ta-3-cpu-fallback-transfers]] — sized by `n_decode × head_dim = 256`; again a head-width property.
- [[tq-1-missing-gemm-kernels]] — the served K/V types have no native GEMM path; this architecture contributes `n_embd_k_gqa = 1024` per layer ([[performance-profile]]).
- [[tq-5-tail-elements]] — **does not bite this model**: the tail case needs `head_dim % GROUP_SIZE != 0` with `GROUP_SIZE ∈ {64, 128}` (`src/ggml/src/ggml-cuda/set-rows.cu:257`, `src/ggml/src/ggml-common.h:325`), and `256 = 2 × 128` is exactly aligned.
- The 48 SSM layers are **not** covered by either defect inventory. No TA/TQ issue concerns the recurrent-state tensors, their F32 dtype, or their per-sequence allocation — and no benchmark separates GDN-block cost from attention-block cost, so the 0.91 % attribution for TriAttention in [[source-state-md]] says nothing about what the other 48 blocks spend.

### Open questions

- **Does the prompt-processing cost follow the 16/48 split?** 48 blocks run a recurrent scan and a width-4 conv instead of attention; nothing in the repo profiles them separately ([[performance-profile]]).
- **The published capacity column and the derived cache do not reconcile.** Per the table above, the published baseline (`~4 000 tok/GB`) and max-mem (`~25 200 tok/GB`) figures are the dense-64-layer arithmetic to within 1–2 %, while this model has 16 KV layers; and the published *speed* figure (`~20 000 tok/GB`) matches neither the dense-64 (11 275) nor the 16-layer (45 100) arithmetic. Separately, the measured 11 800 MiB at 16K ctx ([[benchmarks]]) leaves ~4.9 GiB of non-weight VRAM against a ~1 GiB `f16` cache. Whether the column describes pure KV, whole-process VRAM, or a differently configured cache is not settled by anything in the repo. `[UNVERIFIED]`; recorded, not resolved.
- **Does anything depend on the non-rotary 192 dimensions?** TriAttention's `freq_count = head_dim/2` and its `(f, f + freq_count)` pairing do (see above); nothing else in the tree distinguishes the 64 rotary dims from the rest. `[UNVERIFIED]`.
- **`rope_style = 0` ("half") in the calibration versus MRoPE with sections `[11,11,10,0]` in the model.** The header's own comment defines `0 = half, 1 = interleaved` (`src/src/llama-triattention.h:36`), and the model's rope type is `IMROPE`; whether the two agree on the rotary layout of a 64-of-256 head is not established here.
- Does the `n_layer()`-based layer-adaptive logic in the KV cache (`TURBO_LAYER_ADAPTIVE`, `src/src/llama-kv-cache.cpp:283-295`) do anything sensible when 48 of the 64 counted layers have no cache? Unanswered.

## See also

[[ternary-bonsai-2-27b]] · [[prism-hadamard-weight-fold]] · [[triattention]] · [[turboquant]] · [[walsh-hadamard-transform]] · [[kv-cache]] · [[kv-eviction]] · [[quantization]] · [[benchmarks]] · [[performance-profile]] · [[v100-sxm2]] · [[speculative-decoding]] · [[qwen3-dflash-draft]] · [[gemm-dispatch]] · [[overview]] · [[source-state-md]] · [[ta-2-budget-starvation]]
