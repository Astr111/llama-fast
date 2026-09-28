---
title: Hybrid memory (attention + recurrent)
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/src/llama-memory-hybrid.cpp, src/src/llama-memory-hybrid.h, src/src/llama-memory-recurrent.cpp, src/src/llama-memory-recurrent.h, src/src/llama-hparams.cpp, src/src/llama-hparams.h, src/src/llama-model.cpp, src/src/models/qwen35.cpp, src/src/llama-graph.cpp]
tags: [hybrid, ssm, kv-cache, memory, architecture]
---

# Hybrid memory (attention + recurrent)

## What it is

`llama_memory_hybrid` is the cache container that lets one model hold **two kinds of state at once**: a per-token KV cache for its attention blocks and a fixed-size recurrent state for its linear-attention ([[gated-delta-net]]) blocks. The class states its own purpose in the header: it "utilizes instances of `llama_memory_recurrent` and `llama_kv_cache` to support models where each layer may be either attention-based or recurrent" (`src/src/llama-memory-hybrid.h:20-22`).

It owns exactly two children and nothing else — `std::unique_ptr<llama_kv_cache> mem_attn` and `std::unique_ptr<llama_memory_recurrent> mem_recr` (`src/src/llama-memory-hybrid.h:72-73`), exposed through `get_mem_attn()` / `get_mem_recr()` (`src/src/llama-memory-hybrid.cpp:132-133`). For [[qwen35-architecture]] that split is 16 KV layers and 48 recurrent ones, and it is the structural reason the model's context-memory cost is a quarter of a dense 64-layer model of the same shape.

## How it works

### Dispatch is a construction-time layer filter, not a runtime branch

Neither child is told "you are the attention half". Each is given a layer predicate and allocates state only for the layers that pass it. The constructor supplies the complementary pair when the caller passes none:

```cpp
filter_attn == nullptr ?
    [&](int32_t il) { return !hparams.is_recr(il); } : filter_attn,
...
filter_recr == nullptr ?
    [&](int32_t il) { return hparams.is_recr(il); } : filter_recr
```
(`src/src/llama-memory-hybrid.cpp:48-64`)

For `qwen35` the caller installs explicit filters that additionally bound the index, because the two halves must be exact complements over the *executed* layer range: `il < hparams.n_layer() && !hparams.is_recr(il)` for the KV cache and `il < hparams.n_layer() && hparams.is_recr(il)` for the recurrent memory (`src/src/llama-model.cpp:2787-2793`). Both children therefore enumerate layers `0..63` and skip the ones that reject them; a layer rejected by both would silently hold no state at all, which this pair rules out.

`is_recr()` itself is a lookup, not a computation: `return is_recr_impl[il]` (`src/src/llama-hparams.cpp:231-233`). The array is pre-filled to **0** — `llm_arch_is_recurrent(QWEN35)` is false (`src/src/llama-arch.cpp:1068-1080`, which lists MAMBA/MAMBA2/RWKV6/RWKV6QWEN2/RWKV7/ARWKV7 only), so `src/src/llama-model.cpp:1422` writes 0 — and the architecture's interval rule then assigns every entry, which is what produces the split: `is_recr_impl[i] = (i < n_layer()) && ((i + 1) % full_attention_interval != 0)`, default interval 4 (`src/src/models/qwen35.cpp:21-27`). Attention layers are consequently **3, 7, 11, … 63** — the `(i+1) % 4 == 0` positions.

### What each block type maintains

| | Full-attention block (layers 3, 7, … 63 — 16) | Recurrent block (the other 48) |
| :--- | :--- | :--- |
| State | per-token K/V rows in `llama_kv_cache`; grows with tokens | `cache_r_l%d` conv ring + `cache_s_l%d` recurrence matrix; **one cell per sequence** |
| State tensor shape | `[n_embd_k_gqa=1024, kv_size]`, same for V | `r`: `[n_embd_r=30720, rows]`; `s`: `[n_embd_s=786432, rows]`, rows `= size·(1 + n_rs_seq)` |
| Grows with `-c` / `-n`? | yes | **no** — sized at context construction |
| Quantizable | yes — [[turboquant]] K/V types, [[triattention]] eviction | no — F32 tensors, no eviction path |

The two per-layer state tensors are created as plain 2-D tensors and named per layer (`src/src/llama-memory-recurrent.cpp:99-107`):

```cpp
const uint32_t n_rows = mem_size * (1 + n_rs_seq);
ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), n_rows);
ggml_format_name(r, "cache_r_l%d", i);
ggml_format_name(s, "cache_s_l%d", i);
```

`r` is the short-conv ring buffer, `s` the recurrent matrix. `nullptr` entries mark the layers the filter rejected, and every consumer checks for them — the state writer skips null layers outright (`src/src/llama-memory-recurrent.cpp:928-930`, with the comment "skip null layers (read_data will handle this by checking `r_l` and `s_l` for null)").

### How the state is sized

Both widths come from `llama_hparams`, and for `qwen35` the last branch of each function is the one that runs (the earlier branches are the RWKV, LFM2, Kimi-KDA and MiniMax shapes):

- `n_embd_r()` = `(ssm_d_conv − 1) · (ssm_d_inner + 2·ssm_n_group·ssm_d_state)` (`src/src/llama-hparams.cpp:203-205`)
- `n_embd_s()` = `ssm_d_state · ssm_d_inner` (`src/src/llama-hparams.cpp:227-228`)

With the values in the task brief that is `3 · (6144 + 2·16·128)` = **30 720** for the conv ring and `128 · 6144` = **786 432** for the state. The conv figure is self-checking against the graph, which reshapes the same buffer to `[conv_kernel_size − 1, conv_channels, n_seqs]` with `conv_channels = d_inner + 2·n_group·d_state = 10 240` (`src/src/models/qwen35.cpp:479-481`); the state figure is `S_v · S_v · H = 128 · 128 · 48`, the per-value-head matrix [[gated-delta-net]] describes.

Allocation is `size = mem_size` cells (`src/src/llama-memory-recurrent.cpp:32`) with `cells.resize(mem_size)` (`:39`) and one row per cell, so the whole recurrent cache is `mem_size · (1 + n_rs_seq)` rows of each tensor. The startup log line reports the result per type: `"size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): …, S (%s): …"` (`src/src/llama-memory-recurrent.cpp:119-127`).

`rs_size` for this model is `max(1, n_seq_max)` — **one cell per sequence** — and `n_rs_seq` widens every cell into `1 + n_rs_seq` row-groups so that speculative decoding can roll the recurrent state back by up to `n_rs_seq` tokens ([[speculative-decoding]], [[request-lifecycle]]). The rollback design leaks into batching: the trailing `1 + n_rs_seq` tokens of each sequence must stay inside one ubatch, tagged `[TAG_RECURRENT_ROLLBACK_SPLITS]` in the source (`src/src/llama-memory-recurrent.cpp:431-434`).

### Memory cost per sequence

Per SSM layer, per sequence, at F32 (the type both hybrid constructors pass for `recurrent_type_k`/`recurrent_type_v`):

| Component | Elements | Bytes |
| :--- | ---: | ---: |
| Conv ring (`n_embd_r`) | 30 720 | 120 KiB |
| Recurrent state (`n_embd_s`) | 786 432 | 3 MiB |
| **Per layer** | **817 152** | **3.117 MiB** |
| **All 48 layers** | 39 223 296 | **≈ 149.6 MiB** |

and this is multiplied by `(1 + n_rs_seq)` when rollback snapshots are enabled, because the widening is per cell, not per cache. It does not depend on the token count, on `-c`, or on `-n`: every extra concurrent sequence adds its own ~150 MiB on top of its share of the KV cache. That is the trade the architecture makes — [[kv-cache]] is the only context-scaling state in the model.

### Interaction with the KV cache and TriAttention

The KV half is a full `llama_kv_cache`, and the [[turboquant]] / [[triattention]] machinery lives on it, not here. The hybrid context forwards the TurboQuant entry points straight to the attention context and returns `nullptr` for the other half:

```cpp
ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_forward() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_forward() : nullptr;
}
```
(`src/src/llama-memory-hybrid.cpp`, same shape for `get_turbo_rot_inverse()` and `get_turbo_innerq_scale_inv()`; declared at `src/src/llama-memory-hybrid.h:119-122`)

So the rotation tensors, the InnerQ inverse scale and the eviction scorer all exist **only on the attention side**; the 48 recurrent blocks are untouched by the quantization stack and by [[kv-eviction]]. The consequence for TriAttention is about accounting, not correctness: its budget is a single scalar over cache **cells**, and a cell is a position shared across the layers in the cache — here 16 rather than 64 — so one eviction reclaims 16 · 2048 = 32 768 K+V elements instead of the 131 072 a dense 64-layer model would release, while remaining an aggregate figure applied to a quarter-sized cache. That mismatch is the subject of [[ta-2-budget-starvation]]; the arithmetic is on [[qwen35-architecture]].

### Batching and state movement

`llama_memory_hybrid_context` builds **one child context per half** from the *same* ubatch vector — `ctx_attn(new llama_kv_cache_context(...))` and `ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches))` — and combines their status with `llama_memory_status_combine` in every constructor (`src/src/llama-memory-hybrid.cpp:94-113`). Because one ubatch has to be acceptable to both halves, the recurrent half's rollback constraint above can force a split that the KV half alone would not need.

`state_write`/`state_read` on the hybrid delegate to both children. The recurrent half's serializer walks the layers, skips the null (attention) ones, writes a row size and a type per layer, and then the per-cell ranges — a format it validates on the way back in, erroring on mismatched row/embedding sizes (`src/src/llama-memory-recurrent.cpp:926-951`, `:1089-1170`). Practically, saving a sequence therefore moves ~150 MiB of F32 recurrent state plus the KV cache.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/src/llama-memory-hybrid.h` / `.cpp` | the container; children `mem_attn`/`mem_recr` `:72-73`; filter defaults `:48-64`; accessors `:132-133`; TurboQuant delegation; per-half context construction `:94-113` |
| `src/src/llama-memory-recurrent.h` / `.cpp` | `cache_r_l%d`/`cache_s_l%d` allocation `:99-107`; `size`/`cells`/`rs_idx`/`n_rs_seq`; rollback split `:431-434`; `find_slot` `:494`; sizing helpers `:719-741`; state serialization `:891-951`, `:1089-1170` |
| `src/src/llama-hparams.cpp` / `.h` | `n_embd_r()` `:183-205`, `n_embd_s()` `:207-228`; `is_recr_impl` `:168` and `is_recr()` `:231-236` |
| `src/src/llama-model.cpp` | hybrid construction and the `QWEN35` filters `:2772-2793`; `is_recr_impl` pre-fill `:1422`; `is_recr`-driven block sizing `:656-663` |
| `src/src/models/qwen35.cpp` | the interval rule that decides which half each layer joins `:21-27`; the per-block attention/GDN dispatch `:199-201` |
| `src/src/llama-kv-cache.cpp` | the attention half proper — allocation under the filter, TriAttention attachment (see [[kv-cache]]) |

## Known issues

- [[ta-2-budget-starvation]] — an aggregate eviction budget meeting a 16-layer cache; the *accounting* half of that mismatch is above, the failure mode is on the issue page.
- **Untracked:** the recurrent-state relocation in `build_rs` is ordered before the GDN read that consumes it, and the source itself records this as a real multi-sequence hazard with the correct fix left as a follow-up. See [[gated-delta-net]] *Known issues* for the citation; no issue page covers it.
- **Untracked:** nothing in either defect inventory ([[source-state-md]]) touches the recurrent side at all — not its F32 dtype, not its `(1 + n_rs_seq)` widening, not its per-sequence allocation. A four-sequence server pays ~600 MiB for SSM state alone, and no benchmark in the repo separates that from the KV cost.
- Recurrent state is not evictable and not quantized: [[kv-eviction]] and [[turboquant]] simply have no purchase on the 48 linear-attention blocks, so no amount of KV tuning changes what they hold.

## See also

[[qwen35-architecture]] · [[gated-delta-net]] · [[kv-cache]] · [[kv-eviction]] · [[triattention]] · [[turboquant]] · [[ternary-bonsai-2-27b]] · [[speculative-decoding]] · [[request-lifecycle]] · [[overview]] · [[ta-2-budget-starvation]]
