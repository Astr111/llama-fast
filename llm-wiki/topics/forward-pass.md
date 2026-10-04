---
title: Forward Pass — one token, id to logits
type: topic
status: current
updated: 2026-09-29
sources: [state.md, README.md]
verified: [src/src/llama-graph.cpp, src/src/models/qwen35.cpp, src/src/llama-kv-cache.cpp, src/src/llama-model.cpp, src/common/sampling.cpp, src/ggml/src/ggml-cuda/rope.cu, src/ggml/src/ggml-cuda/norm.cu, src/ggml/src/ggml-cuda/ggml-cuda.cu, src/ggml/src/ggml-cuda/fwht.cu, src/ggml/src/ggml-cuda/turbo-wht.cu, src/ggml/src/ggml-cuda/turbo-quant.cuh, src/ggml/src/ggml-cuda/triattention-score.cu, src/ggml/include/ggml.h]
tags: [synthesis, graph, kv-cache, architecture]
---

# Forward Pass — one token, id to logits

## Bottom line

A token is an `I32` id in `inp_tokens`; the engine turns it into a row lookup, one residual stream, 64 blocks (16 attention, 48 Gated-DeltaNet), a final norm, and an untied, Hadamard-folded output projection — and the id that comes back out is chosen **outside** the graph by the sampler ([[sampling]], [[tokenizer]]). Two of this fork's three custom stacks live inside that chain: TurboQuant rotates Q and the attention output with in-graph `ggml_turbo_wht` ops and quantizes K/V during the cache write (`set_rows`), while TriAttention prunes the cache *before* the graph is built ([[request-lifecycle]]) and CUDA graphs capture the finished per-token graph ([[cuda-graphs]]).

Three facts the rest of the vault depends on:

1. **The residual stream carries the token with no pooling and no scaling.** `build_inp_embd` emits raw gathered rows (plus the Hadamard inverse; see 2) straight into `inpL` — for `qwen35`, `n_embd_inp == n_embd = 5120`, so no padding, and `f_embedding_scale` (Granite-only) is 0.
2. **`prism.hadamard.inverse_weight_names = ["token_embd.weight"]` means the embedding table *is* folded** — it is the one weight whose fold is undone on the *gather* side rather than the matmul side. The table stores latent (rotated) rows; the graph applies `llama_mul_mat_hadamard` (rotation, then signs) to the looked-up rows before they enter the stream (`src/src/llama-graph.cpp:2398-2414`).
3. **The head is untied.** `output.weight` is a separate `PQ2_0` tensor `[5120, 248320]`, also folded ([[prism-hadamard-weight-fold]]); the tied variant would be `prism.hadamard.version = 2` + `tied_output = true`, which this file is not ([UNVERIFIED] only in the sense that nothing in this repo exercises it — see [[prism-hadamard-weight-fold]]).

## Evidence

### 1. Input ids → embeddings

`build_inp_embd` (`src/src/llama-graph.cpp:2417`) declares both graph inputs: `inp_tokens` (I32, `[n_tokens]`, marked input; `t_inp_tokens`, `:2425-2428`) and `inp_embd` (F32, `[n_embd_inp, n_tokens]`, `:2430-2432`). At build time `ggml_build_forward_select` picks one of the two based on whether the ubatch carries tokens or raw embeddings (`:2475`). The token path (`build_embd_rows`, `:2398`) is:

```
cur = ggml_get_rows(ctx0, tok_embd, ids);          // [5120, n_tokens]
cur = llama_mul_mat_hadamard(ctx0, cur, rot);      // H first …
if (signs) cur = ggml_mul(ctx0, cur, signs);       // … then signs
```

The second and third lines are the inverse of the checkpoint-time fold: `token_embd.weight` is stored in the rotated ("latent") basis, and because a row lookup is a gather, not a matmul, the un-fold happens on the gathered rows (`llama-graph.cpp:2401-2414`; loader contract `src/src/llama-model.cpp:1328-1341` — "tensors consumed by row lookup store latent rows and need the inverse transform applied to the lookup result instead"). Any other name in `inverse_weight_names` is refused at load (`llama-model.cpp:1334-1338`). Note the order differs from the weight path (sign-then-rotation there, rotation-then-sign here); the two are halves of the same symmetric pair `(s, H)` ([[prism-hadamard-weight-fold]]).

No pooling and no normalisation follow: `n_embd_inp == n_embd` for `qwen35` (no pad at `:2462`), the scale at `:2486-2494` is skipped (`f_embedding_scale == 0`), and there is no `n_embd_pooled` key in the file, so `t_embd_pooled` is never built. The result is `t_inp_embd` (`:2483`), named `model.input_embed` in the trunk (`src/src/models/qwen35.cpp`, `cb(inpL, "model.input_embed", -1)`).

### 2. Embedding → the 64-block stack

The trunk is `llama_model_qwen35::graph::graph`, derived from `llm_build_delta_net_base` (`src/src/models/qwen35.cpp:137-138`), and reads hybrid inputs via `build_inp_mem_hybrid()` (`:178`, which builds both `inp_rs` and the attention input — `llm_graph_context::build_inp_mem_hybrid`, `src/src/llama-graph.cpp:3818-3827`). Per layer (`for il in 0..n_layer`, `qwen35.cpp:204`):

1. `attn_norm` — RMSNorm over `inpL` (`build_norm`, `llm_graph_context::build_norm` at `src/src/llama-graph.cpp:1671`).
2. **Routing on `hparams.is_recr(il)`** (`qwen35.cpp:199-202`): recurrent → `build_layer_attn_linear(inp->get_recr(), cur, il)` (the 48 GDN layers); else → `build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il)` (the 16 KV layers at 3, 7, … 63). The interval-4 schedule lives in `qwen35.cpp:20-29`; see [[qwen35-architecture]].
3. Residual: `cur = ggml_add(cur, inpSA)` → `attn_residual` (`qwen35.cpp:212-215`).
4. `attn_post_norm` (RMSNorm), then the dense FFN (`build_layer_ffn`), then `cur = ggml_add(cur, ffn_residual)` → `post_ffn` (`qwen35.cpp:216-231`). `inpL = cur` closes the loop.

Every weight that multiplies the stream is consumed through `build_lora_mm`/`build_lora_mm_id`, which applies the fold's activation-side transform (sign flip → blockwise FWHT; memoised per activation so shared Q/K/V inputs pay once) before the folded matmul (`src/src/llama-graph.cpp:1546-1590`, [[prism-hadamard-weight-fold]]).

### 3. Inside an attention block (the 16 KV-bearing layers)

`build_layer_attn` (`src/src/models/qwen35.cpp:322-402`):

- **QKV projection.** A single fused Q+gate projection: `Qcur_full = build_lora_mm(wq, cur)` gives `[12288, n_tokens]` = 24 heads × 256 (query) + 24 × 256 (gate), interleaved; `Qcur` is a strided view of the first half `[256, 24, n_tokens]`, `gate` a view of the second half (`qwen35.cpp:330-356`). `Kcur`/`Vcur` come from `build_lora_mm(wk/wv, …)` at `[256, 4, n_tokens]` (GQA, 4 KV heads; `qwen35.cpp:340-353`). Per-head RMS norms: `attn_q_norm` → `Qcur_normed`, `attn_k_norm` → `Kcur_normed` (`:335-348`).
- **Partial / MRoPE.** `ggml_rope_multi` on Q and K only, with `n_rot = 64` (from `rope.dimension_count = 64`; `src/src/llama-model.cpp:1484-1490`), sections `[11, 11, 10, 0]` (32 = 64/2 pairs), rope type `IMROPE` (`qwen35.cpp:366-380`; `llama-model.cpp:3243-3247`). The other 192 of 256 head dims are never rotated; V is not rotated at all.
- **KV write.** `build_attn` (the KV-cache FA path, around `src/src/llama-graph.cpp:2955-2998`) emits `mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il)` / `cpy_v(...)` (`:2964-2965`, hook `k_cache_in` at `:2960-2962`). In `llama_kv_cache::cpy_k` (`src/src/llama-kv-cache.cpp:1526`): the optional K mean-centre bias is subtracted (`k_bar`, `:1535-1544`), head dims are zero-padded to 128 (`:1548-1556`), and the write is one `ggml_set_rows(ctx, k, k_cur, k_idxs)` (`:1581`) with the WHT group size (128) stashed in `op_params` for the CUDA kernel (`:1583-1587`; `cpy_v` `:1593-1664`). `k_cur`/`v_cur` stay F32; the cache tensors `layers[ikv].k` / `.v` carry the turbo types and the quantisation happens inside the `set_rows` backend op. The served profile is `-ctk turbo3 -ctv q8_0` ([[turboquant]]; `scripts/start_server_turbo.sh`). Whether the K/V Walsh–Hadamard rotation is folded into that quantiser or applied upstream is a joint open question with [[request-lifecycle]] `[UNVERIFIED]` — the graph itself only ever rotates Q and the output (below).
- **Q rotation.** `q = ggml_turbo_wht(ctx0, q, 0, 128, innerq_scale)` immediately before the attention read, guarded on `k->type ∈ {TURBO2_0, TURBO3_0, TURBO4_0}` (`src/src/llama-graph.cpp:2976-2985`); the InnerQ scale comes from the memory context (`get_turbo_innerq_scale_inv`, [[innerq]]).
- **The read.** `mctx_cur->get_k()/get_v()` (`:2971-2972`) feed `build_attn_mha` (`:2650`), which for `cparams.flash_attn` calls the **fused `ggml_flash_attn_ext`** directly on the quantized cache tensors (`:2690`; `LLM_FUSED_OP_FLASH_ATTN` at `:2692`, `GGML_PREC_F32` at `:2695`) — the kernel dequantises the turbo blocks internally (flash attention is *forced* by the turbo KV types, [[overview]] correction). On the output, the inverse rotation `ggml_turbo_wht(..., inverse=1, 128, innerq_scale)` un-rotates the V contribution (`:2707`; non-FA path `:2785`), and any padded head dims are sliced back off (`:2991-2998`). `kq_scale = 1/√256 = 1/16` (`qwen35.cpp:360-362`).
- **Gate + output.** `attn_pregate` → `sigmoid(gate)` → `attn_gated` → `build_lora_mm(attn_output, …)` (`[6144, 5120]`, folded) → `attn_output` (`qwen35.cpp:386-397`), then the residual and FFN of step 2.

### 4. Inside a recurrent block (the 48 SSM/GDN layers)

The builder is `build_layer_attn_linear` (`src/src/models/qwen35.cpp:403-584`); the math is owned by [[gated-delta-net]] and not restated here. The graph nodes, in order: `qkv_mixed`/`z` from `attn_qkv`/`attn_gate` (`build_qkvz`, `qwen35.cpp:220-239`) → gates `beta` (`beta_sigmoid`), `alpha`, `a_softplus = softplus(α + ssm_dt)`, `gate = a_softplus · ssm_a` (`qwen35.cpp:424-462`) → short causal conv via `build_conv_state` + `ggml_ssm_conv` (`conv_output_raw`) + `ggml_silu` (`conv_output_silu`; channels = `d_inner + 2·n_group·d_state` = 10 240, `qwen35.cpp:467-498`) → Q/K/V view split and joint l2 norm (`qk_conv_l2`, `:503-533`) → the recurrence: `build_recurrent_attn` (`src/src/delta-net-base.cpp:532-593`) dispatches to the fused `ggml_gated_delta_net` (K = 1, autoregressive) or chunked path; live state is read per sequence via `build_rs_cache_view` (`state_cache_view`, rows mode; `llm_graph_context::build_rs` at `src/src/llama-graph.cpp:3641`, `build_rs_cache_view` `:3720`) → gated RMSNorm over `z` (`build_norm_gated`, `ssm_norm`) → reshape `final_output` → `build_lora_mm(ssm_out, …)` (`linear_attn_out`, `qwen35.cpp:551-560`). The per-layer state is F32, `[128, 128, 48]` per sequence, written back into the hybrid recurrent cache ([[hybrid-memory]]).

### 5. After the stack: norm, head, logits

`cur = inpL` → `build_norm(output_norm, …)` → `result_norm` (`qwen35.cpp:274-283`; also `res->t_embd`, the embedding output path) → **the LM head** `cur = build_lora_mm(model.output, cur, model.output_s)` → `result_output` → `res->t_logits` (`qwen35.cpp:287-291`). `set_outputs` marks `t_logits` as a graph output (`src/src/llama-graph.cpp:1385-1388`) and the caller copies it out via `get_logits()` (`src/src/llama-context.cpp:2092-2094`).

**Tied versus untied.** This model is **untied**: `output.weight` exists as a distinct 337 715 200-byte `PQ2_0` tensor `[5120, 248320]` (verified in [[prism-hadamard-weight-fold]]), so the loader's tied fallback — `output = create_tensor(…, TENSOR_NOT_REQUIRED)` then "if output is NULL, init from the input tok embed" (`qwen35.cpp:46-50`) — does not fire; the head is folded and consumed through `build_lora_mm` like every other weight (`src/src/llama-model.cpp:1295-1297`: "the output head is built through build_lora_mm in every arch").

**`--reasoning-budget` / `-n` do not gate the head.** The head node is unconditional in the graph. `-n` only bounds how many tokens the caller requests, and the reasoning-budget knobs are a *sampling-chain* stage (`common_reasoning_budget_init`, `src/common/sampling.cpp:312`; applied first in the chain at `:632`), which forces backend sampling off (`:421-424`) because the reasoning-budget and backend samplers are mutually exclusive — see [[sampling]].

### 6. Where the three custom stacks sit

| Stack | Where in the chain | Consequence |
| :--- | :--- | :--- |
| [[triattention]] | **Before graph build**: `apply_ubatch()` → `triattention_try_prune()` (`src/src/llama-kv-cache.cpp:1373-1374`) evicts cells of the 16 KV layers | Evicted positions are simply absent from the flash-attention read; the graph topology does not change ([[request-lifecycle]]) |
| [[turboquant]] | KV **write** (`set_rows` onto turbo-typed cache tensors, WHT group in `op_params`) and KV **read** (`ggml_flash_attn_ext` dequantising internally); Q pre-rotation and output un-rotation are in-graph `ggml_turbo_wht` ops (`llama-graph.cpp:2985`, `:2707`) | K/V are stored quantized + rotated; only Q and the attention output are ever touched by the custom op; turbo types force the fused attention path |
| [[cuda-graphs]] | Captures the **whole per-token graph** after build, around graph compute | Graph replay amortises the build; the saving is constant, not cumulative ([[cuda-graphs]]) |

### The chain, end-to-end

```mermaid
flowchart TD
    T["token id, I32"] --> INP["inp_tokens (ggml_set_input)"]
    INP --> GR["build_embd_rows: ggml_get_rows(token_embd.weight)"]
    GR --> INV["llama_mul_mat_hadamard: H*z then signs (latent-table inverse)"]
    INV --> INPL["inpL = model.input_embed, F32 5120 x n_tokens"]
    INPL --> NORM["attn_norm (RMSNorm)"]
    NORM --> ROUTE{"is_recr(il)?"}
    ROUTE -->|"48x GDN"| GDN["build_layer_attn_linear: qkv + z, gates, conv4 + silu, l2-norm, ggml_gated_delta_net scan, gated RMSNorm, ssm_out"]
    ROUTE -->|"16x attention"| ATT["build_layer_attn: Q+gate proj, Q/K RMSNorm, rope_multi 64-of-256 [11,11,10,0], turbo_wht(Q), cpy_k/cpy_v set_rows, ggml_flash_attn_ext, inv turbo_wht, gate, attn_output"]
    GDN --> RES["attn_residual: add inpL"]
    ATT --> RES
    RES --> FFN["attn_post_norm, FFN, post_ffn residual"]
    FFN --> NORM
    FFN --> FN["output_norm (RMSNorm)"]
    FN --> HEAD["build_lora_mm(output.weight) — folded, untied"]
    HEAD --> LOG["result_output = logits, F32 248320 x n_outputs"]
    LOG --> SMP["sampling (outside the graph) -> token id -> text (tokenizer)"]
```

### Load-bearing tensor shapes (model metadata)

| Tensor | Shape | Type | Notes |
| :--- | :--- | :--- | :--- |
| `token_embd.weight` | `[5120, 248320]` | `PQ2_0` | latent rows; Hadamard-inverted after lookup |
| `embd` / `inpL` | `[5120, n_tokens]` | `F32` | no pooling, no scale |
| `Qcur` | `[256, 24, n_tokens]` | `F32` | 24 heads × 256, partial-RoPE on 64 |
| `Kcur`, `Vcur` | `[256, 4, n_tokens]` | `F32` | 4 KV heads × 256 (GQA, group 6) |
| K cache (per KV layer) | `[1024, kv_size, n_stream]` | `TURBO3_0` (served) | 1024 = 4 × 256; 16 layers only; 2048 elements/token with V |
| V cache (per KV layer) | `[1024, kv_size, n_stream]` | `q8_0` (served) | flash-attention (non-transposed) layout |
| SSM state (per GDN layer/seq) | `[128, 128, 48]` | `F32` | 786 432 elements; fixed, context-independent |
| `result_output` (logits) | `[248320, n_outputs]` | `F32` | head is untied + folded |

Shapes from `[[qwen35-architecture]]` (GGUF-verified) and `[[prism-hadamard-weight-fold]]` (tensor list); `kv_size` is the configured cell count (`-c`).

## Open questions

- **Where exactly is K/V rotated on write?** The graph rotates only Q and the attention output with `ggml_turbo_wht`; `cpy_k`/`cpy_v` emit plain `set_rows` with a WHT group size in `op_params` (`src/src/llama-kv-cache.cpp:1581-1587`). The rotate+quantise step must therefore live in the CUDA `set_rows` kernel — read kernel-side here, not re-derived; joint with [[request-lifecycle]]'s identical open question `[UNVERIFIED]`.
- **Non-rotary dims are rotated and quantized wholesale.** RoPE covers 64 of 256 head dims, yet TurboQuant's 128-element groups and the WHT rotation act on the full head; the 192 non-positional dims are treated like the rest by design (`[[qwen35-architecture]]`, itself `[UNVERIFIED]` on whether that is intended).
- **Tied head is dead code for this artifact.** The loader supports `version 2`/`tied_output` with a latent embedding reused as the head (`src/src/llama-model.cpp:1345-1356`), but no artifact here exercises it; nothing in the repo states whether a tied variant of Bonsai-2-27B exists.

## The kernels behind the rotations and the norms

Two unrelated linear maps are called "rotation" in this fork, and they are never the same kernel. The positional rotation (RoPE) is the **attention path**; the TurboQuant rotation (a Walsh–Hadamard transform) is the **cache path**, applied on top of the already-roped row. What [[first-live-measurements]] sees as a doubling of `rms_norm_mul_rope_f32` is the point where the two contracts meet: the fused norm+rope kernel is the last kernel in the chain that still belongs to RoPE, and whether it can absorb the cache write determines which *instantiation* of it runs.

### 1. Two rotations, two families of kernels

| Rotation | Kernel symbols | Defined | Reached from |
| :--- | :--- | :--- | :--- |
| RoPE — dims `[0, n_rot)` of each head, attention path | `rope_norm<has_ff,T,D>`, `rope_neox<has_ff,T,D>`, `rope_multi<has_ff,T>`, `rope_vision<has_ff,T>`; fused form `rms_norm_mul_rope_f32<block_size,has_ff,D>` | `src/ggml/src/ggml-cuda/rope.cu:44`, `:123`, `:200`, `:294`, `:711` | `ggml_cuda_op_rope_impl` (`rope.cu:536`), dispatched from `GGML_OP_ROPE` (`ggml-cuda.cu:2309-2311`); the fused form only through `ggml_cuda_try_fuse` (`ggml-cuda.cu:4093-4101`) |
| TurboQuant WHT — all `group_size` dims, KV write and turbo read path | `k_turbo_wht_f32<direction,group_size>` (`turbo-wht.cu:151-169`); `fwht_cuda<N,T,has_signs>` / `fwht_cuda_block` / `fwht_cuda_smem` (`fwht.cu:234-296`); in-kernel `turbo_rotate_forward` (`turbo-quant.cuh:125-128`), `inverse_wht_rotation_128` (`triattention-score.cu:72`) | `src/ggml/src/ggml-cuda/turbo-wht.cu`, `fwht.cu`, `turbo-quant.cuh` | `GGML_OP_TURBO_WHT` (`ggml-cuda.cu:2092-2094` → `ggml_cuda_turbo_wht`, `turbo-wht.cu:117`) for the graph-side Q pre-rotation and the attention-output inverse; `SET_ROWS` for the write side |

They compose rather than substitute. A K row on its way into a turbo cache is: head RMSNorm → **RoPE** (`rope_multi` for this model) → optional mean-centre `ggml_sub` → optional 128-alignment `ggml_pad` → **WHT + quantisation inside `set_rows`** (`src/src/llama-kv-cache.cpp:1526-1587`; the WHT group is handed to the backend in `op_params`, `:1583-1587`, and the rotation itself is `turbo_rotate_forward`'s `signs1 → FWHT → signs2`, `turbo-quant.cuh:125-128`). On the read side Q gets the same WHT forward (`llama-graph.cpp:2976-2985`) and the attention output the inverse (`:2700-2708`), both as separate launches of `k_turbo_wht_f32` — which is why the phrase "no `turbo_wht` kernel appears" in [[first-live-measurements]] is a *name* observation: `ggml_turbo_wht` is the wrapper, the kernel symbol is `k_turbo_wht_f32<...>`. The one kernel where both contracts sit in a single launch is `rms_norm_mul_rope_f32`, and it serves RoPE only — it has no signs vector, no group size and no InnerQ parameter (`rope.cu:711-725`, [[innerq]]), so it cannot perform the WHT; a turbo-typed destination is not merely refused by the fusion predicate, it is unrepresentable in that kernel (`dst_type` is cast to `D ∈ {float, half}`, `rope.cu:924-939`).

### 2. The fused kernel is a fusion, not an op

`rms_norm_mul_rope_f32` has no `GGML_OP` of its own. `ggml_cuda_can_fuse` recognises exactly two subgraphs (`ggml-cuda.cu:3083-3111`):

- `RMS_NORM → MUL → ROPE` (the Q-side shape: per-head norm, weight, RoPE), and
- `RMS_NORM → MUL → ROPE → VIEW → SET_ROWS` (the K/V-side shape: the same three nodes plus `cpy_k`'s flattening view and the cache write, `src/src/llama-kv-cache.cpp:1577-1581`).

The second form is the interesting one: it writes the roped row **straight into the KV cache**, because the driver takes its destination from the `SET_ROWS` tensor — `dst_d = set_rows->data; dst_type = set_rows->type;` and `row_indices`/`set_rows_stride` come from `set_rows->src[1]` and the row stride (`rope.cu:867-876`). Both instantiations are the same binary; only the template argument `D` differs (`rms_norm_mul_rope_cuda<D>`, `rope.cu:790`, launched as `<256,false,D>` or `<1024,false,D>` after `ncols < 1024`, `:822-848`). The kernel name's `f32` refers to its *input*; `D` is the output type and is the parameter the profile's `<(int)256,…>` abbreviation hides.

### 3. Why a quantized cache moves the launch to the other instantiation

The five-op form is gated on the destination type (`ggml_cuda_should_fuse_rope_set_rows`, `ggml-cuda.cu:2668-2698`):

```
if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) return false;   // :2679-2681
if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX)              return false;   // :2695-2698
```

A `TURBO3_0` cache row is neither F32 nor F16, so under `-ctk turbo3` the K chain falls to the three-op form, which writes a **F32 intermediate** (`dst_type == GGML_TYPE_F32` branch, `rope.cu:924-931`), and the cache write becomes a separate `set_rows` launch. Two further differences in `cpy_k` reinforce that: it inserts `ggml_sub` for the mean-centre bias (`src/src/llama-kv-cache.cpp:1535-1543`, [[source-kv-mean-center]]) and `ggml_pad` for 128-alignment (`:1551-1558`) *between* the ROPE node and the VIEW, so the five nodes are no longer consecutive and `ggml_can_fuse_subgraph` cannot match them anyway.

Consequence for the trace, stated as precisely as the evidence allows. Under FP16 KV the K-side chain is served by the **`D = half` instantiation** writing the cache row itself; under turbo KV it is served by the **`D = float` instantiation** writing a buffer that `set_rows` then rotates and quantises. The launches quoted in [[first-live-measurements]] (`rms_norm_mul_rope_f32<(int)256,…>`: 108 instances / 0.31 ms FP16, 216 / 0.83 ms turbo) therefore most plausibly are: FP16 = two symbols of the same kernel (the F32 Q-side chain, 108; the F16 cache-writing chain, 108 — only one row shown), turbo = one merged F32 row (Q-side + K-side = 216), the F16 symbol having disappeared with the F16 cache. Two measurements fit that reading and not "RoPE is being run twice per head": (a) the neighbouring rows are *byte-identical* across the two traces (`rms_norm_f32<(int)1024,…>` 219/219, `quantize_q8_1` 472/472), so no norm was un-fused — only re-instantiated; (b) the per-instance cost moves exactly as an F32 destination would (0.31 ms / 108 = 2.9 µs → 0.83 ms / 216 = 3.8 µs). `[UNVERIFIED]` the published trace shows one row per configuration, so the alternative — that the turbo graph genuinely contains twice as many norm+rope chains — cannot be excluded from it. Nothing in `src/src/llama-graph.cpp` supports that alternative: every KV-type-conditional branch there inserts only `ggml_cont`, `ggml_pad`, `ggml_turbo_wht` and slicing (`:2700-2708`, `:2976-2985`, `:2993-2997`). The decisive observation is an nsys table that lists the `float` and `half` instantiations of this kernel separately: a FP16 run showing 108 + 108 on two rows means the total launch count is unchanged (216 in both) and only its split moved.

Magnitude, for scale: +0.52 ms of fused-kernel time against a +170 ms wall-clock penalty (2.36 s → 2.53 s at n = 128, [[first-live-measurements]]) — 0.3 %. The rotation contract is visible in the trace; it is not what costs the 11 %.

### 4. Partial / MRoPE: the kernel is parameterised, and that is the whole point

All four RoPE kernels take the rotation span at runtime, from the node's `op_params`, not from their type or their launch shape (`ggml_cuda_op_rope_impl`, `rope.cu:580-604`):

| Parameter | Source | Used as |
| :--- | :--- | :--- |
| `n_dims` (= `n_rot`, 64 here) | `op_params[1]` (`rope.cu:580`) | the kernel bounds test `if (i0 < n_offs \|\| i0 >= n_offs + n_dims)` (`rope.cu:97`, `:170`, `:239`) — everything outside is copied through, in place |
| `n_offs` | `op_params[15]` (`rope.cu:584`) | offset of the rotated window (0 for this model; the fused kernel refuses non-zero, `ggml-cuda.cu:2745-2748`) |
| `sections[4]` | `memcpy(&sections.v, op_params + 11, 4*sizeof(int))` (`rope.cu:604`) | the sector map in `rope_multi` (`rope.cu:220`, `:251-275`); the kernels assert at least one non-zero section (`:611-613`) |
| frequency exponent | `theta_scale = powf(freq_base, -2.0f/n_dims)` (`rope.cu:388`, `:432`, `:477`, `:811`) | θ^(−2f/`n_dims`), *not* θ^(−2f/head_dim) |

So 64-of-256 with sections `[11, 11, 10, 0]` is expressible and *is* expressed: only `[0,64)` rotates (in NEOX pairing — `x[i0/2 + n_offs/2]` against `x[i0/2 + n_offs/2 + n_dims/2]`, `rope.cu:192-196`, `:286-290`), the 192 remaining dims are copied unrotated, and the exponent is computed over 64. `src/ggml/include/ggml.h:1879-1884` documents the same contract from the graph side — `MROPE n_dims = 16 → [ttttyyxxttttyyxx00]`, `IMROPE n_dims = 16 → [ttyxttyxttyxttyx00]`, with "idx used for theta: [0123… until n_dims/2], not reset for each section". That is the model-side truth [[ta-9-rope-scope-mismatch]] must be inverted against, and it is why that fix sketch is a parameterisation problem: the scorer's inverse pairs `(f, f+128)` over all 256 dims with θ^(−2f/256), three independent deviations from this table.

One scope fence worth recording: **the fused kernel cannot be the forward path for this model.** `ggml_cuda_should_fuse_rms_norm_mul_rope` accepts `NORMAL` (0) and `NEOX` (2) only (`ggml-cuda.cu:2735-2738`); `IMROPE` is 40 (`ggml.h:254`), so every qwen35 Q/K RoPE goes through `rope_multi` (`rope.cu:650-663`) with sections, unfused — and the fused kernel would also drop the sections entirely, since its driver never reads `op_params[11..14]` (`rope.cu:878-903`). The 108/216 measurement is from `Ternary-Bonsai-4B`, a NORMAL/NEOX model; it therefore constrains the fused path, not the partial-RoPE path, which this project's 27B never takes a fused kernel for.

### 5. `GGML_CUDA_FWHT_LEGACY`

The switch ([[runtime-switches]]) lives in exactly one place in this codebase: the tail of `fwht_launch`, `src/ggml/src/ggml-cuda/fwht.cu:273-286`.

```
static const bool legacy = getenv("GGML_CUDA_FWHT_LEGACY") != nullptr;
if (legacy) {  // 512/1024/2048 -> fwht_cuda<NN,…>, 4096/8192 -> fwht_cuda_smem<NN,…>
}              // default   -> fwht_cuda_block<NN, 256, …> for 512…8192
```

Widths 64/128/256 always use `fwht_cuda<N,T,has_signs>` (`fwht.cu:228-244`) and are unaffected. It does **not** appear in `rope.cu`, `norm.cu`, `turbo-wht.cu`, or in the `SET_ROWS`/`ROPE`/`RMS_NORM`/`TURBO_WHT` dispatch cases read here. And the widths it does select are wider than the KV path ever uses: `ggml_cuda_turbo_wht` asserts `group_size == 64 || group_size == 128` (`turbo-wht.cu:129-131`) and `cpy_k` always writes a WHT group of 128 (`src/src/llama-kv-cache.cpp:1585-1586`). So on this model the switch cannot change the rotation the cache sees — it is an A/B knob for a WHT width the TurboQuant path does not reach ([[turbo-wht]], [[walsh-hadamard-transform]]). Its one neighbouring use outside the KV path is the calibration/scoring side, `triattention-score.cu:72`'s own cooperative 128-point butterfly, which is hand-written and does not call `fwht_launch` at all.

## See also

[[qwen35-architecture]] · [[hybrid-memory]] · [[gated-delta-net]] · [[turboquant]] · [[triattention]] · [[sampling]] · [[request-lifecycle]] · [[prism-hadamard-weight-fold]] · [[tokenizer]] · [[cuda-graphs]] · [[walsh-hadamard-transform]]