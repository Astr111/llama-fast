---
title: Loading and Batching
type: topic
status: current
updated: 2026-09-29
sources: [state.md, README.md]
verified: [src/src/llama-batch.cpp, src/src/llama-batch.h, src/src/llama-context.cpp, src/src/llama-kv-cache.cpp, src/src/llama-triattention.h, src/src/llama-mmap.cpp, src/src/llama-model-loader.cpp, src/src/llama-model-saver.cpp, src/src/llama-model-saver.h, src/src/llama-adapter.cpp, src/src/llama-graph.cpp, src/src/llama.cpp, src/ggml/src/gguf.cpp, src/ggml/src/ggml.c, src/ggml/include/ggml.h, src/common/arg.cpp]
tags: [batching, loading, mmap]
---

# Loading and Batching

## Bottom line

Two things happen before a token ever reaches the graph that [[request-lifecycle]] and [[forward-pass]] take for granted: the weights become memory, and the caller's token list becomes **ubatches**.

**Loading.** There is no "load" step that copies 1085 MiB of `PQ2_0` weights into a private buffer by default. `--load-mode auto` mmaps the GGUF read-only and points each tensor's `data` at an offset inside the mapping (`src/src/llama-model-loader.cpp:1416-1419`). The bytes are then demand-paged by the kernel, which is why the *same* 1085 MiB model reads at **949 MiB/s** the first time and **14 179 MiB/s** the second ([[first-live-measurements]]): the first read is a disk read, the second is a page-cache copy. That ratio is not a property of the firmware or the fork — it is this subsystem's whole behaviour, and it is selected by `--load-mode` ([[runtime-switches]]).

**Batching.** A `llama_batch` is a flat caller-owned array with no structure the engine trusts; it is sanitised into a `llama_ubatch` by `llama_batch_allocr`, and *where the ubatch boundaries fall is decided by the KV cache, not by the caller or the scheduler*: `llama_kv_cache::init_batch` picks `split_simple` for a single-stream cache and `split_equal(n_ubatch, /*sequential=*/true, /*n_keep_tail=*/0)` otherwise (`src/src/llama-kv-cache.cpp:887`). Because the only per-ubatch hooks are `apply_ubatch()` and, at its tail, the TriAttention trigger (`src/src/llama-kv-cache.cpp:966` and `:1373-1375`), **the split rule is the clock that TriAttention runs on** — `triattention_should_prune()` is a function of `n_used`, and `n_used` only advances at a ubatch boundary. That is the mechanism behind [[ta-2-budget-starvation]]; whether [[ta-10-prefix-length-global-latch]] latches for the same reason is not established here.

The fork's own quantised types survive this layer untouched: `PQ2_0` is a fully registered ggml type ([[prismml-weight-kernels]]), so both the loader and the saver treat it as an opaque block format and never dequantise it on the way in or out.

## Evidence

### `llama_batch` — what the caller hands over

A `llama_batch` is six parallel arrays and a count: `n_tokens`, `token[]` *or* `embd[]`, `pos[]`, `n_seq_id[]`, `seq_id[][]`, `logits[]` (`src/src/llama-batch.cpp:945-973`). Nothing in it is required. `llama_batch_get_one()` returns a struct with **only** `n_tokens` and `token` set — every other field `nullptr` (`src/src/llama-batch.cpp:931-943`) — and that is the form the simple decode path uses.

`llama_batch_allocr::init()` then fills and checks (`src/src/llama-batch.cpp:25-391`):

- token ids must be in `[0, vocab.n_tokens())` (`:50-56`);
- sequence ids must be in `[0, n_seq_max)` (`:59-63`);
- missing `n_seq_id`/`seq_id` default to a single sequence `0` (`:73-88`);
- missing `pos` is **inferred from the memory module**: `pos[i] = p0[seq_id]` and `p0[seq] = pos[i] + 1` (`:104-113`);
- missing `logits` defaults to *all* tokens when `output_all`, else only the **last** token (`:122-128`); a caller who passes an all-zero `logits` array gets a warning and all-outputs behaviour (`:133-145`).

The check that matters most downstream is the **position-continuity** rule: with one position per token, `init()` requires

```
seq_pos_min(s) == memory->seq_pos_max(s) + 1      // src/src/llama-batch.cpp:283-303
```

i.e. the first token of each sequence in the batch must continue *exactly* where the KV cache ended. A hole in the sequence is a hard error, not a warning. The M-RoPE case (`n_pos_per_embd > 1`) is looser: forward jumps only, `X < Y` for tokens and `X <= Y` for embeddings (`:249-275`). `llama-kv-cache.cpp:3059-3063` records why this is safe to keep: TriAttention's evictions leave position gaps, but its recent-token protection keeps `seq_pos_max` unchanged, so the `Y = X + 1` validation still passes.

Pairs of sequences sharing at least one token are flagged coupled (`has_cpl`, `:163-181`), and a coupled batch cannot be split sequentially — that is an error telling the user to try `-kvu` (`:511-514`).

### `llama_ubatch` — what the graph actually consumes

A `llama_ubatch` is the rectangular form of the same data: `n_tokens = n_seq_tokens * n_seqs`, with `n_seqs` **sequence sets** rather than raw sequences, plus `n_seqs_unq`, `n_pos`, and a `b_equal_seqs` flag (`src/src/llama-batch.h:13-52`). The pointers are valid only for the lifetime of the ubatch's `shared_ptr<data_t>` (`:54-62`), so a ubatch cannot be stashed past the decode step that produced it.

`equal_seqs()` is the structural promise the graph and the memory module branch on; `llama_kv_cache::set_input_k_idxs` asserts on it (`src/src/llama-kv-cache.cpp:2229`), and the encode path explicitly wishes for a split mode that always makes it true (`src/src/llama-context.cpp:1644-1646`).

### Who splits, and what forces a boundary

The caller sets a *ceiling*: `cparams.n_ubatch = min(n_batch, params.n_ubatch ?: n_batch)` (`src/src/llama-context.cpp:338`). The split itself happens inside the memory module via `memory->init_batch(*balloc, cparams.n_ubatch, output_all)` (`src/src/llama-context.cpp:1977`), which for the standard KV cache is:

```cpp
auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch)
                            : balloc.split_equal(n_ubatch, true, 0);
// src/src/llama-kv-cache.cpp:887
```

The three splitters and their constraints (`src/src/llama-batch.cpp`):

| Splitter | Used by | Boundary rule |
| :--- | :--- | :--- |
| `split_simple` (`:476-508`) | single-stream KV, encoder | take the first unused token, then extend with tokens belonging to the **same sequence set**; stop when `idxs.size() >= n_ubatch` |
| `split_seq` (`:681-721`) | sequence-set-wise | same, but the token's set must be a *superset* of the set being built (`:711`) |
| `split_equal` (`:510-650`) | multi-stream KV | build non-overlapping sequence sets, then grow them **in lockstep** |

`split_equal` is where boundaries are really forced, and the three constraints are:

1. **Set count** — a set is refused once it would exceed `n_ubatch` (`:547-549`).
2. **Lockstep growth** — `can_expand` requires every participating sequence set to still have an unused token, i.e. a ubatch is truncated to the shortest set (`:574-603`); the explicit stop is `(idxs_per_seq[0].size() + 1) * n_seqs > n_ubatch` (`:600-602`).
3. **Tail preservation** — optional `n_keep_tail`: sequences are cut so their last `n_keep_tail` tokens always land in one ubatch (`:605-650`), with `GGML_ASSERT(n_ubatch > n_keep_tail)` (`:609`). **The KV cache passes `0`** (`llama-kv-cache.cpp:887`), so this mechanism is inactive on the default path.

Encoders sidestep splitting entirely: non-causal encoding asserts `n_ubatch >= n_tokens` and processes in one shot (`src/src/llama-context.cpp:1649`, `:1951`).

The number of ubatches per `llama_decode` is therefore `ceil(n_tokens_all / n_ubatch)`, further bounded by memory availability — the loop re-runs `init_batch` after a cache update or defrag and can legitimately return `-2` or `1` instead of decoding (`src/src/llama-context.cpp:1976-2013`).

### Why the boundary is semantically load-bearing here

`apply_ubatch()` is the only place the KV cache absorbs a graph result, and its last act is the TriAttention check:

```cpp
if (triattention_should_prune(triattention_st, n_used)) {
    triattention_try_prune();
}
// src/src/llama-kv-cache.cpp:1373-1375
```

`triattention_should_prune()` is documented as a pure function of occupancy — "SLACK mode: `n_used >= (budget + divide_length)`" (`src/src/llama-triattention.h:324-327`). `n_used` grows once per ubatch, so **the ubatch boundary is the sampling interval of the prune decision**. Two consequences follow:

- A prompt fed in one `llama_decode` with a large `n_ubatch` reaches the eviction trigger fewer times than the same prompt streamed in small batches, even though the final cache contents are identical. Prefix-processing strategy is therefore an *input* to eviction behaviour, which is why [[ta-2-budget-starvation]] and [[ta-10-prefix-length-global-latch]] both turn on where the boundary falls.
- `apply_ubatch` runs *before* the current ubatch's keys are in the graph's output (`src/src/llama-kv-cache.cpp:966` is called from the same batch loop that the graph follows), so pruning always scores the cache as of the previous ubatch — the open question already recorded on [[request-lifecycle]].

Observability: `LLAMA_BATCH_DEBUG=1` turns on `ubatch_print()`, which dumps `equal_seqs`, `n_tokens`, `n_seq_tokens`, `n_seqs`, positions and per-token sequence membership for every ubatch (`src/src/llama-batch.cpp:13-14`, `:846-921`). That is the cheapest way to see the boundary rule act on a real prompt.

### Loading: two paths, and what each costs

`--load-mode` selects between them in one place:

```cpp
this->use_mmap      = load_mode == LLAMA_LOAD_MODE_MMAP || ... MMAP_MLOCK || ... AUTO;
this->use_direct_io = load_mode == LLAMA_LOAD_MODE_DIRECT_IO;
// src/src/llama-model-loader.cpp:562-563
```

| `--load-mode` | Mechanism | Start-up IO cost | Where the bytes live afterwards |
| :--- | :--- | :--- | :--- |
| `auto` / `mmap` (default) | `mmap(PROT_READ, MAP_SHARED)` of the GGUF; tensor `data` points into the mapping | one demand-paged read per weight actually used — uncached this is the measured **949 MiB/s** | page cache; reclaimable under memory pressure |
| `none` | `llama_file` reads via `fread`/`::read` into backend buffers | one sequential read of the whole file at the same uncached rate | anonymous memory (plus an async pinned-memory upload path for GPU) — not reclaimable |
| `dio` | same read path with `O_DIRECT` | same rate, **every run** — the page cache is bypassed | anonymous memory; no cache pollution |
| `mlock` / `mmap+mlock` | plus `mlock()` on the mapping, grown per tensor | as above | pinned in RAM; cannot be paged out |

The mmap path in detail (`src/src/llama-mmap.cpp:444-473`): `posix_fadvise(POSIX_FADV_SEQUENTIAL)` before mapping, `MAP_POPULATE` when prefetching the whole file, then `posix_madvise(POSIX_MADV_WILLNEED)` over a prefix. The loader passes `prefetch ? -1 : 0`, where `-1` means "the whole file" (`src/src/llama-model-loader.cpp:1374`). Tensor attachment is a pointer arithmetic step, no copy: `cur->data = (uint8_t *) mapping->addr() + w->offs` (`:1416-1419`), and offloaded/unused fragments are `munmap`ped afterwards (`:1408-1411`, `:1699-1706`). `mlock` is grown incrementally as each tensor is attached, not all at once (`:1583-1586`).

The read path costs an `open` failure mode that mmap does not: `O_DIRECT` is attempted at file-open time (`src/src/llama-mmap.cpp:198-201`) and needs alignment, which is why the file object exposes `read_alignment()` and `has_direct_io()` (`:408-409`). The loader also builds an async upload backend from pinned memory *only* when not mmapping and not checking tensors (`src/src/llama-model-loader.cpp:1470-1474`), so `--load-mode none` is the mode with a real host-to-device transfer pipeline.

### Making start-up IO predictable

- **Page cache is the 15× speedup, and it is free.** A second run of the same model at the same size lands in the 14 179 MiB/s regime ([[first-live-measurements]]) because nothing on the mmap path copies anything — the win is purely that the kernel already holds the pages.
- **To make the *first* run cheap, prefetch; to make *every* run identical, bypass.** `MAP_POPULATE`/`WILLNEED` front-loads the disk read into `llama_model_load` (`src/src/llama-mmap.cpp:455-467`); `-lm dio` removes the cache from the equation so the second run pays the same 949 MiB/s as the first.
- **To stop eviction, lock.** `--load-mode mlock` (or `mmap+mlock`) is the only mode that makes resident size a guarantee rather than a hope (`src/src/llama-mlock` via `src/src/llama-mmap.cpp:642-643`, grown at `src/src/llama-model-loader.cpp:1583-1586`), at the cost of the full model in locked RAM — 1085 MiB here.
- **Do not mix old flags.** `--mmap`/`--mlock`/`--direct-io` are deprecated aliases that write the same enum, and combining them with `--load-mode` warns that only the last one on the command line wins (`src/common/arg.cpp:891-901`; alias definitions `:2682-2726`; the enum's own documentation `:2708-2726`).
- **`mmap` plus a CPU weight override is a downgrade.** The loader warns exactly once that `--override-tensor` to CPU with mmap enabled should really be `--load-mode none` (`src/src/llama-model-loader.cpp:1195-1199`), because the override breaks the "point into the file" property that makes mmap cheap.

### Saving: can it write a `PQ2_0` model back out?

`llama_model_save_to_file()` is the single production entry point, and it is three calls: build the KV block from the `llama_model`, collect the tensors, write (`src/src/llama.cpp:495-499`). In this tree nothing else uses it — `src/tools/quantize` contains no reference to `llama_model_saver` (grep: no matches), so the saver is a library capability, not the quantiser's writer.

Three findings:

1. **The target's architecture is not excluded.** `llama_model_saver_supports_arch()` returns `true` by default and blacklists 19 architectures (`src/src/llama-model-saver.cpp:15-36`); the ctor asserts on that predicate (`:43`). Nothing in the qwen3-family list ([[ternary-bonsai-2-27b]], [[quantization]]) is on it.
2. **`PQ2_0` passes the writer unchanged.** The GGUF writer validates only `0 <= type < GGML_TYPE_COUNT`, i.e. `< 144` (`src/ggml/src/gguf.cpp:713-719`), and `GGML_TYPE_PQ2_0 = 142` is well inside that range with a registered `type_traits` entry (`src/ggml/include/ggml.h:437-441`, `src/ggml/src/ggml.c:692-698`). Tensor payloads are written as raw `ggml_nbytes` bytes, via `ggml_backend_tensor_get()` if the tensor sits in a backend buffer, else `memcpy` (`src/ggml/src/gguf.cpp:1628-1638`). So the weight **bytes** and their `PQ2_0` block layout round-trip exactly, and saving a GPU-resident model works but pays a full D2H copy.
3. **Metadata is the weak half, not the tensor types.** `add_kv_from_model()` writes a hard-coded upstream key list and explicitly leaves `general.quantization_version`, `general.alignment` and `general.file_type` unset (`src/src/llama-model-saver.cpp:185-187`); `add_tensors_from_model()` walks a fixed list of model-level tensors plus every pointer slot of each `llama_layer` (`:418-450`), skipping `nullptr`s and de-duplicating by name with an assert for the three `rope_*` tensors (`:136-148`).

Consequence: a save/reload cycle of a `PQ2_0` GGUF keeps the weights bit-exact but loses any Prism-private KV that is not in that hard-coded list. The reload is not fatal, because the loader can infer the file type from the per-tensor types as a fallback (`GGML_TYPE_PQ2_0 → LLAMA_FTYPE_MOSTLY_PQ2_0`, `src/src/llama-model-loader.cpp:777-778`, with a legacy-ftype name for GGUF packed before the `Q2_0_G128`→`PQ2_0` rename, `:41-46`).

> `[UNVERIFIED]` The absence of Prism-specific KV keys was checked against the lines of `add_kv_from_model()` that were read, not against every line of the 268-line body. Treat "the saver carries no Prism metadata" as highly likely rather than established.

### Adapters: LoRA attaches by name and does not read weight bytes

`llama_adapter_lora_init_impl()` loads the adapter GGUF, pairs every `*.lora_a`/`*.lora_b` tensor by name suffix (`src/src/llama-adapter.cpp:265-290`), and resolves them against the model at graph-build time by **name lookup only** — `get_weight(w)` searches `ab_map` for `w->name` (`:138-147`). Attachment validates shape against the model tensor (`:355-374`) and falls back to a CPU buffer type when the model tensor lives in a repacking "extra" buffer (`:337-350`).

So the adapter never interprets the model weight's bytes: `build_lora_mm()` computes the base product with the model's own type and adds a separate low-rank term,

```cpp
ggml_tensor * res    = ggml_mul_mat(ctx0, w, cur_mm);          // model type (PQ2_0 here)
...
ggml_tensor * ab_cur = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur));  // F32 adapter
res = ggml_add(ctx0, res, ab_cur);
// src/src/llama-graph.cpp:1579-1601
```

There is exactly one hard type restriction on the base weight in this file — the NVFP4 asserts (`down->type != GGML_TYPE_NVFP4` etc., `src/src/llama-graph.cpp:1817-1820`) — and no equivalent for `PQ2_0`. An adapter does not require a `PQ2_0` kernel, and a `PQ2_0` weight does not require a modified adapter.

**The one real interaction is the Hadamard fold, and it looks wrong.** When a weight is in the fork's `hadamard_rotations` map, the base product is computed against the *rotated* activation `cur_mm`, while the LoRA branch still reads the **unrotated** `cur` (`src/src/llama-graph.cpp:1550-1577` vs `:1593-1600`). If the fold means the stored weight is `W·H` and the activation is rotated to match, the adapter's delta must be rotated into the same basis; multiplying it by the original `cur` computes the delta in the wrong frame. `[INFERENCE]` — forced by those two reads; I did not check whether any weight in the target model is both folded and covered by an adapter, which is the condition for the defect to bite.

### What this page adds over [[request-lifecycle]] and [[forward-pass]]

[[forward-pass]] explains what the graph does with a ubatch once it exists, and [[request-lifecycle]] explains why a `llama_decode()` is called at each step. Neither says what a ubatch *is*, who chose its size, or why a boundary lands where it does — so neither can explain why the identical prompt decodes differently when its `n_ubatch` or stream count changes, which is the mechanism this page supplies. On the loading side, [[request-lifecycle]] and [[forward-pass]] both start from weights that are already resident and never mention mmap, so the measured 949-vs-14 179 MiB/s gap has until now had no home; it is a property of `--load-mode`, and this page names the mode that selects each behaviour.

## Open questions

- `hadamard_rotations` is consulted per weight in `build_lora_mm` (`src/src/llama-graph.cpp:1550-1577`). Which weights in a `PQ2_0` target are in that map, and is any of them LoRA-adapted? Unanswered here; it decides whether the basis mismatch above is a live defect or a dead branch.
- `split_equal`'s `n_keep_tail` is passed `0` by the KV cache (`src/src/llama-kv-cache.cpp:887`). Is that deliberate (the tail protection is TriAttention's job) or an unported upstream fix?
- `add_tensors_from_model()` walks `llama_layer` as a flat array of tensor pointers (`src/src/llama-model-saver.cpp:436-441`). That is only sound if every member is a pointer; the layout was not read. `[UNVERIFIED]`
- Which `n_ubatch` the server actually runs with, and whether the server's prompt chunking is aligned to it — not read here. See [[open-questions]].
- Does `-lm dio` work on the filesystem holding the 1085 MiB `PQ2_0` GGUF? `O_DIRECT` needs alignment (`src/src/llama-mmap.cpp:198-201`, `:408-409`) and was never exercised on this machine.

## See also

[[request-lifecycle]] · [[forward-pass]] · [[runtime-switches]] · [[kv-cache]] · [[hybrid-memory]] · [[ta-2-budget-starvation]] · [[ta-10-prefix-length-global-latch]] · [[quantization]] · [[prismml-weight-kernels]] · [[first-live-measurements]] · [[ternary-bonsai-2-27b]] · [[open-questions]]
