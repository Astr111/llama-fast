---
title: DeepSeek-V4 KV cache (DSV4)
type: entity
status: current
updated: 2026-09-29
sources: []
verified: [src/src/llama-kv-cache-dsv4.h, src/src/llama-kv-cache-dsv4.cpp, src/src/llama-kv-cache.h, src/src/llama-kv-cache.cpp, src/src/llama-kv-cache-iswa.h, src/src/llama-model.cpp, src/src/llama-context.cpp, src/src/llama-arch.h, src/src/llama-triattention.cpp, src/src/CMakeLists.txt]
tags: [kv-cache, dsv4, deepseek, hybrid, fork]
---

# DeepSeek-V4 KV cache (DSV4)

## What it is

The **second KV-cache engine** in this tree: class `llama_kv_cache_dsv4` (`src/src/llama-kv-cache-dsv4.h:88`), a 78 KB implementation (`src/src/llama-kv-cache-dsv4.cpp`) that no page of this vault had mentioned. It is **not** a variant of `llama_kv_cache` — it derives from `llama_memory_i` and *owns* ordinary caches.

> **Verdict: a sibling cache engine for a different architecture, and it is reachable.** It is selected by **architecture, not by a flag**: `src/src/llama-model.cpp:2705-2720` constructs `llama_kv_cache_dsv4` for `LLM_ARCH_DEEPSEEK4` (`src/src/llama-arch.h:87`) whenever that model's context is not an MTP draft context. The same architecture's MTP context gets a plain `llama_kv_cache_iswa` instead (`llama-model.cpp:2683-2704`), and the sibling arch `LLM_ARCH_DFLASH` with `dsv4_hc_mult > 0` likewise gets an ISWA cache (`llama-model.cpp:2724-2749`) — so DSV4 is one of four memory policies chosen in the same `switch`, not an orphan file.

Reachability in detail:

| Question | Answer | Evidence |
| :--- | :--- | :--- |
| Compiled? | Yes, in the core library | `src/src/CMakeLists.txt:30` |
| Constructed? | Yes — `new llama_kv_cache_dsv4(...)` | `src/src/llama-model.cpp:2706` |
| Selected by | `model.arch == LLM_ARCH_DEEPSEEK4`, non-MTP context | `src/src/llama-model.cpp:2705`, `src/src/llama-arch.h:87` |
| Has a model loader and graph? | Yes — `llama_model_deepseek4` and `graph_dsv4` | `src/src/models/deepseek4.cpp`, `src/src/models/models.h:1394`, `src/src/llama-graph.cpp:14` |
| Dead-code guards or "for reference" comments? | None; only two `FIXME`s | `src/src/llama-kv-cache-dsv4.h:85-86` |

So it is **not** a fork-of-fork and **not** dead code: it is the KV engine of the DeepSeek-V4 architecture that this fork also carries. It is simply not the engine of *this project's* target: the vault's subject is the qwen35/ternary-bonsai hybrid stack ([[qwen35-variants]], [[hybrid-memory]], [[ternary-bonsai-2-27b]]), whose attention cache is `llama_kv_cache_iswa`/`llama_kv_cache`. Every claim in the vault about cell layout, TurboQuant cell typing and eviction was written against `src/src/llama-kv-cache.cpp`; DSV4 is a *different* `llama_memory_i` and, as shown below, none of the TriAttention machinery reaches it.

## How it works

`llama_kv_cache_dsv4` is a **composite**: a normal raw token cache plus three compressed K-only caches, plus three compressor-state buffers, all built in its constructor (`src/src/llama-kv-cache-dsv4.cpp:1211-1335`). Its contexts (`llama_kv_cache_dsv4_context`) fan out to one context per sub-cache (`:2001-2011`, `:2069-2081`).

| Member | Type | Role | Evidence |
| :--- | :--- | :--- | :--- |
| `kv_raw` | `llama_kv_cache_iswa` | raw token cache, SWA; DSV4 attention reads only the SWA half | `:1255-1258`; header comment `:177-178` |
| `kv_csa` | `llama_kv_cache` | compressed K, layers whose `dsv4_compress_ratios[il] == 4` | `:1292-1295`, `:1271-1276` |
| `kv_hca` | `llama_kv_cache` | compressed K, layers whose ratio `== 128` | `:1300-1303`, `:1278-1283` |
| `kv_lid` | `llama_kv_cache` | "lightning indexer" compressed K, `n_head_kv = 1`, NEOX RoPE, `indexer_head_size` | `:1263-1269`, `:1308-1311` |
| `csa_state` / `hca_state` / `lid_state` | `llama_dsv4_comp_state` | compressor ring state, **F32**, not quantized | `:1328-1350`, `:964-966` |

Mechanism, point by point:

- **Compression ratios are fixed constants**, not per-model: `DSV4_CSA_RATIO = 4`, `DSV4_HCA_RATIO = 128` (`:18-19`). Each layer is assigned to CSA, HCA or neither by its GGUF-declared ratio (`hparams.dsv4_compress_ratios`, `src/src/llama-hparams.h:282`).
- **Compressed capacity** is `max(1, ceil(kv_size / ratio))` rows, rounded up to a multiple of 256 (`dsv4_comp_size` `:28-30`; `GGML_PAD(..., 256u)` at `:1294`, `:1302`, `:1310`).
- **K-only storage** is forced by a helper that abuses the MLA path: `dsv4_make_k_only` sets `n_embd_head_k_mla_impl = n_embd_head_k()` and `n_embd_head_v_mla_impl = n_embd_head_k()`, because "`llama_kv_cache` uses `hparams.is_mla()` to allocate K-only storage" (`:883-886`), applied to all four hp sets (`:1253`, `:1260-1261`, `:1269`).
- **Compressed rows are graph outputs, not token writes.** The compressed-cache context deliberately implements no `apply()`: "DSV4 compressed KV rows are graph outputs, not normal token KV writes. Keep a small context that exposes K tensors without generic `apply()` semantics" (`llama-kv-cache-dsv4.h:236-238`; implementation `llama-kv-cache-dsv4.cpp:1951-1999` exposes only `get_k` / `cpy_k` / `*_k_rot`).
- **Compressor state** is a separate F32 ring: `kv` and `score` tensors of shape `[n_embd_state, state_size, n_stream*(1 + n_rs_seq)]` (`:964-966`), with `state_size = 2*ratio` for CSA/LID and `ratio` for HCA, and `n_embd_state = 2*n_embd_head_k()` (CSA), `n_embd_head_k()` (HCA), `2*indexer_head_size` (LID) (`:1328-1350`). The `1 + n_rs_seq` planes are **rollback snapshots** driven by `comp_plan` (`state_persist_*`, `state_restore_*`, `state_snapshot_*`, `llama-kv-cache-dsv4.h:273-300`; planning `:653-700`). `n_rs_seq` comes from `cparams.n_rs_seq` (`llama-model.cpp:2717`) — this is a rollback feature, **not** eviction, and must not be confused with TriAttention's `triattention_state`.
- **Streams.** Multi-sequence operation is a layout-stream dimension: `n_stream = unified ? 1 : n_seq_max` (`:907`), with `dsv4_stream_offset` mapping `seq_id` to a buffer slice (`:51-60`). The raw half takes the passed `unified` flag (`unified_raw`), but the compressed half is hard-coded to non-unified — `const bool unified_compressed = false;` (`:1290`), which is exactly the header's `FIXME` ("currently the cache only supports non-unified mode even if unified flag is passed", `:85`).
- **Its own serialization format**: magic `0x34565344` ("DSV4"), `DSV4_STATE_VERSION 1`, `DSV4_STATE_MODE_FULL/PARTIAL`, `DSV4_K_CACHE_STATE_VER 2`, `DSV4_COMP_STATE_VER 1` (`:21-26`) — `state_write`/`state_read` on the composite and on `llama_dsv4_comp_state` (`llama-kv-cache-dsv4.h:44-45`, `:111-112`).
- **Buffers are zeroed after construction** so that attention reading compressed rows the current graph did not overwrite sees zeros rather than uninitialized memory (`:1330-1335` and the comment there).
- Coupled (multi-seq) ubatches are rejected for embedding batches and otherwise split into a raw-write ubatch (`dsv4_build_raw_write_ubatch`, `:82-98`).

### Cell layout and quantized-cell handling versus `llama-kv-cache.cpp`

- **No cell machinery of its own.** All four sub-caches are ordinary `llama_kv_cache` / `llama_kv_cache_iswa` instances, and the composite forwards `type_k`/`type_v` straight into their constructors (`:1255-1257`, `:1292-1294`, `:1300-1302`, `:1308-1310`). There is therefore **no second quantized-cell implementation**: the TurboQuant key types and their quantizer state (`llama-turbo-quant.cpp`, [[turboquant]]) are reached through the same `llama_kv_cache` cells as on the target path.
- **What differs is granularity, not typing.** The three compressed caches hold one K row per *block* of 4 or 128 tokens and store no V at all; the `n_stream` layout dimension is exactly the stream/`n_seq_max` dimension the [[kv-cache]] page describes, so stream bookkeeping is the same concept applied to a new tensor.
- **The F32 compressor state is un-quantized** (`GGML_TYPE_F32`, `:965-966`) — the only DSV4 structure that is entirely outside the [[turboquant]] type system, and the reason DSV4 costs extra cache memory per layer beyond the compressed K rows.
- **Bug-suspicious detail, stated plainly:** `kv_lid` and `lid_state` are both constructed with `filter_csa` rather than a distinct indexer filter (`:1311`, `:1349`), so the "lightning indexer" cache is populated for exactly the CSA layers. `[UNVERIFIED]` whether that is intended (indexer taps the same layers) or a copy-paste defect — no comment says either way, and the graph side (`llm_graph_input_dsv4::get_lid()`) was not read.

## Where it lives

- Implementation: `src/src/llama-kv-cache-dsv4.cpp` (declared at `src/src/CMakeLists.txt:30`), header `src/src/llama-kv-cache-dsv4.h`.
- Selection: `src/src/llama-model.cpp:2705-2720` (`LLM_ARCH_DEEPSEEK4`, non-MTP); arch enum `src/src/llama-arch.h:87`.
- Graph side: `src/src/llama-graph.cpp:14`, `:811-947`, `src/src/llama-graph.h:43-45`, `:640-715`, `:1360`; model side `src/src/models/deepseek4.cpp` and `src/src/models/models.h:1394`.
- Hyperparameters it consumes: `dsv4_o_group_count`, `dsv4_compress_ratios`, `dsv4_hc_mult`, `dsv4_hc_sinkhorn_iters`, `dsv4_hc_eps`, `dsv4_hash_layer_count`, `dsv4_compress_rope_base` (`src/src/llama-hparams.h:269-282`).
- One arch-side side-effect worth knowing: `llama_model_n_swa` returns 0 for `LLM_ARCH_DEEPSEEK4` because the rollback stream cannot be expressed as an SWA rollback (`src/src/llama-model.cpp:3042-3046`).
- Siblings in the same family, for orientation: `llama-kv-cache-iswa.{h,cpp}`, `llama-kv-cache-dsa{,-iswa}.*`, `llama-kv-cache-msa.*` (all under `src/src/`).

## Known issues

### TriAttention cannot attach to this cache — the latch is absent here, not broken

This is the load-bearing finding for the vault's eviction claims:

1. The only way to install TriAttention on a cache is `llama_triattention_init` (`src/src/llama-context.cpp:4362-4407`), which resolves the target by `dynamic_cast<llama_kv_cache *>(ctx->get_memory())` (`:4373`), with a fallback cast to `llama_memory_hybrid` that takes `get_mem_attn()` (`:4374-4377`).
2. A DSV4 context's memory is a `llama_kv_cache_dsv4`, which derives from `llama_memory_i` (`llama-kv-cache-dsv4.h:88`), as does `llama_kv_cache_iswa` (`llama-kv-cache-iswa.h:14`). Neither cast succeeds, so the call logs "memory is not a KV cache (recurrent models not supported)" and returns `-1` (`llama-context.cpp:4379-4381`).
3. `llama_kv_cache_dsv4` exposes no TriAttention API of its own (`llama-kv-cache-dsv4.h:140-153` lists `get_raw`/`get_csa`/`get_hca`/`get_lid`/`get_*_state`/`get_n_rs_seq`/`get_rs_idx` only), and no other call site of `init_triattention` exists in the tree.

**Consequences, stated for the eviction pages:**

- The DSV4 cache is never given a `triattention_state`; `triattention_st` stays `nullptr` in all four of its sub-caches. It therefore has **no** `prefix_length` latch, no `budget`, no `should_prune` trigger, and no eviction hooks — `triattention_try_prune()` is only ever reached from `llama_kv_cache::apply_ubatch()` when a state exists (`src/src/llama-kv-cache.cpp:1350-1376`, `:3015-3073`).
- The `prefix_length` latch that [[ta-10-prefix-length-global-latch]] documents is **base-class code and is present here as code** — it lives in `llama_kv_cache::apply_ubatch` (`src/src/llama-kv-cache.cpp:1350-1376`), which every sub-cache inherits — but it is **unreachable in practice**, because the state it latches into can never be installed. The same is true of the `on_cell_removed` / `on_token_added` / `on_position_shift` hooks (e.g. `src/src/llama-kv-cache.cpp:580`, `:605`, `:791`, `:1302-1307`); even if a sub-cache's `apply` path ran them, each is a no-op under a null state (`src/src/llama-triattention.cpp:767` and, `[INFERENCE]` from the same guard pattern, its siblings).
- **Does this invalidate [[ta-2-budget-starvation]], [[ta-10-prefix-length-global-latch]] or [[kv-eviction]]? No — but it scopes them.** Those claims are about `llama_kv_cache` with TriAttention enabled, which remains exactly how the target (hybrid qwen35 via `llama_memory_hybrid::get_mem_attn`, or a plain `llama_kv_cache`) reaches it. What DSV4 shows is that "the KV cache" in this fork is not one object: there is a second, composite engine whose sub-caches own the same base-class cells and the same latch code while being structurally outside TriAttention's reach. Any future claim phrased as "the cache evicts X" must name the architecture it holds for.
- Corollary worth recording as an open question: DSV4's compressed block caches have **no eviction mechanism at all** — they are fixed-capacity ring/compressed stores sized from `kv_size`, with only the rollback planes for state. Whether DeepSeek-V4 relies on its own compressors instead of eviction is a design question this file raises and does not answer.

### Other

- Two `FIXME`s in the header: non-unified compressed mode only (`:85`; confirmed by the hard-coded `unified_compressed = false` at `:1290`), and "we currently conflate token_pos and buffer contents", with a link to an upstream PR discussion (`:86`). Whether upstream or fork authored this file is `[UNVERIFIED]` from this tree alone; the FIXME's `ggml-org/llama.cpp/pull/25521` reference `[INFERENCE]` points at upstream provenance.
- `dsv4_build_raw_write_ubatch` throws for coupled embedding ubatches (`:86-88`) — a hard failure mode for multi-sequence embedding batches on this arch, not a degraded path.
- Because it duplicates the cache surface, this file is a second place any future change to cell bookkeeping, serialization versions or quantized-cell access must be made; the vault's [[kv-cache]] page describes only the other one.

## See also

[[kv-cache]] · [[kv-eviction]] · [[triattention]] · [[turboquant]] · [[hybrid-memory]] · [[qwen35-variants]] · [[ta-10-prefix-length-global-latch]] · [[ta-2-budget-starvation]] · [[quantization]]
