---
title: KV cache
type: concept
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/src/llama-kv-cache.cpp, src/src/llama-kv-cache.h, src/src/llama-graph.cpp, src/ggml/src/ggml-common.h]
tags: [kv-cache, attention, memory]
---

## Definition

The per-layer store of past keys and values that autoregressive attention reads on every new token. Each generated token appends one row to K and one to V; each generated token also attends over *all* rows written so far. The cache therefore converts an O(1)-per-token recomputation into an O(1)-per-token read, and in exchange makes memory and per-token work grow linearly with context length.

Concretely, the cost of step `t` is:

- **Memory** — `2 · n_layers · n_kv_heads · head_dim · dtype_size · t` bytes, allocated up front as a fixed-capacity tensor, not grown on demand. In this tree the tensors are created once at the maximum size (`src/src/llama-kv-cache.cpp:351-352`, `ggml_new_tensor_3d(ctx, layer_type_k, n_embd_k_gqa_eff, kv_size, n_stream)`) and reported at construction as `K (turbo3): X MiB, V (q8_0): Y MiB` (`:445-451`). `n_layers` here means the layers that actually hold a cache: the target model is hybrid, and only the attention blocks contribute ([[ternary-bonsai-2-27b]]).
- **Attention** — one score for every cached key, i.e. `O(t)` work and `O(t)` bytes of cache traffic per token. Nothing in the architecture removes this term; a transformer's attention is dense over the cache by construction.
- **Cache traffic per token** — the whole K and V of every layer is re-read each step. This is why the *dtype* of the cache matters for throughput as much as for capacity.

The distinguishing property versus an ordinary tensor: the cache is the only activation whose size is a function of user behaviour (conversation length) rather than of model geometry. Everything in this project that touches memory is ultimately a statement about this tensor.

## Why it matters here

The decode-time degradation recorded for this project is a KV-cache effect, not a scheduling effect. [[source-state-md]] §1.1 measures decode at **15.75 ms → 19.43 ms** as context grows, and explicitly attributes the growth to "unbounded KV attention costs as context grows" while concluding that CUDA-graph reuse is a constant one-time saving (~25 µs against a 16–64 ms step) rather than something that compounds. A 25 µs launch-overhead win cannot explain a 3.7 ms regression; an attention pass whose working set grew from thousands to tens of thousands of rows can.

Against that curve the project applies three independent levers, and it is worth keeping them separate because they fail in different ways:

1. **Quantization — shrink each cell.** `turbo3` keys and `q8_0` values ([[quantization]]). This reduces bytes per key deterministically and is the only lever that helps from the first token. Its failure mode is not slowness but *wrong arithmetic*: a type with no native dot product still has to be multiplied, and the fallback dequantises the whole cache ([[gemm-dispatch]], [[tq-1-missing-gemm-kernels]]). On Volta the missing integer tensor cores make that fallback land on FP16 cuBLAS with no fast counterweight ([[v100-sxm2]]).
2. **Eviction — remove cells.** TriAttention scores resident keys and keeps a fixed `budget` ([[kv-eviction]], [[triattention]]). This is the only lever that attacks the `O(t)` term in attention rather than its constant factor — it caps `t` itself. Its failure mode is *silent context loss*: a fixed budget against a variable-length prompt can starve to zero retained history, degenerating into a sliding window ([[ta-2-budget-starvation]]).
3. **Graph scheduling — remove launch overhead, not cache work.** CUDA graphs eliminate per-node launch cost by replaying a captured graph ([[cuda-graphs]]). This is what makes the many small ops of a decode step cheap to *submit*; it does nothing about the attention term above, which the source of truth is explicit about. It also constrains the design: a statically sized cache tensor is exactly what makes capture possible, which is part of why `budget` is a constant.

The two levers that address the cache proper are consequently in tension with the third's requirement for static shapes, and the measured bottleneck sits in a fourth place entirely — the arithmetic consuming the quantised cache.

## Tradeoffs

- **Memory vs context.** Fixed-capacity allocation for the maximum `kv_size` gives predictable memory and capturable graphs; it also means the configured `kv_size` is the hard ceiling on usable context *before* eviction, and the eviction budget is the effective ceiling after.
- **Read bandwidth vs arithmetic.** A smaller dtype reduces cache traffic but adds a decode step in the inner loop. When the dtype has a native fused dot (fused attention's `vec_dot_fattn_vec_KQ_turbo*`, [[quantization]]) this is a clear win; when it does not, the decode is paid once over the whole tensor instead of once per element of the dot ([[gemm-dispatch]]).
- **Keys and values are not symmetric.** Keys are consumed by a score (dot with Q) and values by a weighted sum, but only keys are *scored for eviction* and only keys need un-rotation under TriAttention. Splitting the dtypes (`-ctk` / `-ctv`) exploits this asymmetry: `q8_0` values skip the inverse WHT entirely ([[source-readme]], *SPEED PROFILE*).
- **Determinism of position.** Eviction cannot evict the most recent `divide_length` tokens without breaking the server's position counter (`llama-triattention.cpp:1114-1121`); the cache's representation of "where am I in the sequence" is a positional index, not a monotone token count, and any compaction must preserve it.
- **Sharing.** Caches can be shared or copied across sequences and streams (`llama-kv-cache.cpp:116-122` forces a matching `kv_size`; `n_stream` is a tensor dimension at `:351`), so cache policy is also a concurrency decision, not purely a memory one.

## Open questions

- How much of the 15.75 → 19.43 ms slope is attention over *more* keys versus attention over *worse* keys once WHT inversion is broken ([[ta-1-wht-inversion-256]])? The two are confounded in the measurement ([[performance-profile]]).
- What is the effective context ceiling at the shipped configuration once [[ta-2-budget-starvation]] collapses the retained history to the 512-token window?
- Does the non-flash attention branch (`ggml_mul_mat(ctx0, k, q)`, `llama-graph.cpp:2730`) ever run on the turbo-typed cache in practice, or is flash attention always selected? Determines whether the cache is consumed by the fused kernels exclusively — `[UNVERIFIED]` ([[gemm-dispatch]]).

## See also

[[kv-eviction]] · [[quantization]] · [[triattention]] · [[turboquant]] · [[cuda-graphs]] · [[gemm-dispatch]] · [[performance-profile]] · [[overview]] · [[kv-cache-dsv4]]
