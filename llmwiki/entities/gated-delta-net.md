---
title: Gated DeltaNet
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/src/models/qwen35.cpp, src/src/llama-graph.cpp, src/src/delta-net-base.cpp, src/src/llama-graph.h, src/src/llama-hparams.cpp, src/src/llama-hparams.h, src/src/llama-model.cpp, src/ggml/src/ggml-cpu/ops.cpp, src/ggml/src/ggml-cpu/ggml-cpu.c, src/ggml/src/ggml-cuda/gated_delta_net.cu, src/ggml/src/ggml-metal-ops.cpp, src/ggml/src/ggml-opencl/CMakeLists.txt]
tags: [ssm, linear-attention, recurrence, gdn, architecture]
---

# Gated DeltaNet

## What it is

Gated DeltaNet (GDN) is the **linear-attention recurrence implemented by the 48 non-attention blocks of [[qwen35-architecture]]**. The model code names it outright: "Linear attention layer (gated delta net)" (`src/src/models/qwen35.cpp:199-201`). It is a delta-rule recurrent net: each block keeps a fixed-size matrix per value head, decays it by a learned gate each step, and updates it with a rank-1 correction weighted by the prediction error — no softmax, no per-token key/value storage. That is what makes those 48 blocks immune to context growth ([[hybrid-memory]]).

In ggml it is one op, `GGML_OP_GATED_DELTA_NET` (`ggml_gated_delta_net`), with native CPU, CUDA, Metal, OpenCL, Hexagon and ET backends. This page records what the *code* computes, from the CPU reference implementation and the `qwen35` graph.

## How it works

### The per-token recurrence (from the CPU reference op)

The CPU implementation `ggml_compute_forward_gated_delta_net_one_chunk` (`src/ggml/src/ggml-cpu/ops.cpp:10942-11120`) is the ground truth for the math. Per token `t`, per head `iv1`, per sequence `iv3`, with state matrix `S` stored **transposed** (`s_out[j*S_v + i] = S[i][j]`, i.e. buffer row `j` is state column `j`):

1. **Decay.** `S ← exp(g0) · S` (the scalar-gate branch; the KDA branch decays per channel: `S[:,i] *= exp(g[i])`, selected when the gate's `ne[0] == S_v`). `exp(g0) < 1` is the forget factor.
2. **Delta rule.** `δ[j] = (v[j] − (Sᵀk)[j]) · β` — the error between the value and what the state *would* have retrieved for `k`.
3. **Update.** `S ← S + k ⊗ δ` (outer-product, `S[i][j] += k[i]·δ[j]`).
4. **Readout.** `o[j] = (Sᵀq)[j] · scale`, with `scale = 1/√S_v`.

The op can also apply the gates itself when fed pre-activation values: `β = σ(β_raw)` and `g = a[h] · softplus(g_raw + dt_bias[h])` (the `raw_gates` path, `ops.cpp:10955-10959`, set via `ggml_gated_delta_net_set_raw_gates`). When `K > 1` snapshot slots are requested, the op output is `[attention scores | K state snapshots]`, slot 0 being the most recent state (`ops.cpp:11023-11039`) — this is the speculative-decoding rollback mechanism ([[speculative-decoding]]).

### The qwen35 block, tensor by tensor

The `qwen35` variant is built in `build_layer_attn_linear` (`src/src/models/qwen35.cpp:411-560`), and reads its hyperparameters from the `ssm_*` metadata (`src/src/models/qwen35.cpp:8-15`):

| Tensor | Shape (qwen35) | Role in the block |
| :--- | :--- | :--- |
| `attn_qkv` (`wqkv`) | `[5120, 10240]` | projects the input into QKV; also fed through the conv |
| `attn_gate` (`wqkv_gate`) | `[5120, 6144]` | `z`, the output gate |
| `ssm_beta` | `[5120, 48]` | per value head, `β = σ(β·x)` — the delta-rule learning rate |
| `ssm_alpha` | `[5120, 48]` | per value head, `α = α·x` |
| `ssm_dt` | `[48]` | bias added to `α` before softplus |
| `ssm_a` | `[48]` | per-head log decay; `g = softplus(α + ssm_dt) · ssm_a`, and the source comment marks it `-A_log.exp()` (`qwen35.cpp:438-441`) — negative, so `exp(g)` is a decay in `(0, 1)` |
| `ssm_conv1d` | `[4, 10240]` | depthwise causal short conv, kernel width `ssm_conv_kernel = 4` |
| `ssm_norm` | `[128]` | grouped RMS norm over `z`: `RMSNorm(x) · silu(z)` in one `ggml_swiglu_split` (`build_norm_gated`, `qwen35.cpp:245-252`) |
| `ssm_out` | `[6144, 5120]` | output projection back to `n_embd = 5120`; also the PrismML Hadamard fold target ([[prism-hadamard-weight-fold]]) |

Sequence of operations in the block:

1. **Project + gate** (`build_qkvz`, `qwen35.cpp:220-239`): `qkv_mixed = wqkv·x`, `z = wqkv_gate·x`.
2. **Gates**: `β = σ(ssm_beta·x)`; `α = ssm_alpha·x`; `gate = softplus(α + ssm_dt) · ssm_a` (`qwen35.cpp:411-441`). Unless `GGML_GDN_RAW_GATES_DISABLE` is set and the backend supports raw gates, the raw projections are handed to the fused op instead (`qwen35.cpp:420-436`).
3. **Short conv**: the conv state ring (`n_embd_r` — see [[hybrid-memory]]) is laid out as `[kernel−1, channels, n_seqs]` with `channels = d_inner + 2·n_group·d_state = 6144 + 4096 = 10240` (`build_conv_state`, `src/src/delta-net-base.cpp:454-471`; `qwen35.cpp:479-481`); `ggml_ssm_conv` then `ggml_silu` (`qwen35.cpp:487-490`).
4. **Split + normalize**: the 10 240-wide conv output is viewed as Q and K (16 key heads × `ssm_d_state = 128`), and V (48 value heads × `head_v_dim = d_inner/48 = 128`) (`qwen35.cpp:493-516`); Q and K are l2-normalized jointly (`qwen35.cpp:518-532`). `num_k_heads = ssm_n_group = 16`, `num_v_heads = ssm_dt_rank = 48`.
5. **The recurrence**: `build_recurrent_attn` (`src/src/delta-net-base.cpp:532-593`) dispatches to the chunked path (multi-token), the autoregressive path (single token), or the fused path, which calls `ggml_gated_delta_net` once with `K = 1` (`delta-net-base.cpp:373-452`). For qwen35 the live state is read per sequence directly out of the cache in "rows mode" (op `src[6]`) when `n_rs_seq > 0` — CPU and Metal only; `GGML_GDN_STATE_GATHER=1` restores the gathered path (`qwen35.cpp:449-465`; the cache view helper is `build_rs_cache_view`, `src/src/llama-graph.h:1443-1447`).
6. **Gate + project**: `RMSNorm_gated(output, ssm_norm, z)`, reshape, then `ssm_out·` and the residual path continues as in any block (`qwen35.cpp:551-560`).

The state the block maintains is `[S_v=128, S_v=128, H=48]` per sequence — exactly `n_embd_s = ssm_d_state · ssm_d_inner = 128 · 6144 = 786 432` elements — written back into the cache by a `ggml_cpy` into `ssm_states_all`'s per-sequence row group (`delta-net-base.cpp:559-568`). CUDA fuses that scatter into the op's epilogue (`src/ggml/src/ggml-cuda/gated_delta_net.cu`, cache-fusion matcher at `src/ggml/src/ggml-cuda.cu:2754-2802, 3391-3403`), and Metal's `SET_ROWS` fold does the same (`src/ggml/src/ggml-metal-ops.cpp:1923-2001`).

### What context (`-c`, `-n`) means for a recurrent block

Nothing about size: one `[128, 128, 48]` state per sequence is allocated once, at cache construction (`rs_size = max(1, n_seq_max)` — [[hybrid-memory]]), and no `-n 8192`, `-c 32768` or any other length knob resizes it. Tokens only drive *sequential state updates*: every token must pass through the recurrence in order, which is exactly why a multi-token prompt uses the chunked variant while decode uses the K=1 autoregressive one (`delta-net-base.cpp:430-452`). A recurrent block therefore has no notion of "position in context"; its memory does not grow, and no part of [[kv-eviction]] applies to it. The only context-sensitive state in the model is the 16-layer [[kv-cache]].

### `gdn_v_grouped` — the flagged flag

`prism.hadamard.gdn_v_grouped` is read into `hadamard_gdn_v_grouped` (`src/src/llama-model.cpp:1267`) and its only observed consumer is the PrismML Hadamard weight-fold, applied when the fold rewrites `.ssm_out.` weights: it groups the 48 value heads (`n_v = hparams.ssm_dt_rank`) into 16 groups (`n_k = hparams.ssm_n_group`), asserting `n_k > 0 && n_v % n_k == 0 && weight->ne[0] % n_v == 0` before folding (`src/src/llama-model.cpp:2121-2130`). The loader therefore *trusts* that 48 heads group evenly as 3-per-group over the 16 key-head groups.

The graph never cross-checks that assumption: `build_layer_attn_linear` only verifies divisibility when it repeats Q/K to 48 heads on the non-fused paths (`qwen35.cpp:537-544`), and nothing compares the fold's grouping with the actual 48-head layout of `ssm_out`. If the flag disagreed with the weights, they would load and the graph would still run — folded-but-mis-grouped weights are invisible to every assertion in the repo. The flag's semantics beyond that fold (e.g. any effect on the recurrence itself) is `[UNVERIFIED]`: a single pass over these files found no other consumer.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/src/models/qwen35.cpp` | the full GDN block: projections `:220-239`, gates `:411-441`, raw-gates toggle `:420-436`, conv state `:479-481`, Q/K/V split and l2 norm `:493-532`, recurrence dispatch `:449-465`, gated norm `:245-252`, output projection `:551-560` |
| `src/src/delta-net-base.cpp` | shared GDN plumbing: chunked/autoregressive/fused selection `:430-452`; fused op call with raw gates `:404-407`; conv-state layout `:454-471`; state write-back `:532-593` |
| `src/src/llama-graph.h` | `build_rs_cache_view` rows-mode state read `:1443-1447` |
| `src/ggml/src/ggml-cpu/ops.cpp` | CPU reference: the recurrence `:10942-11120`, dispatcher `:11157-11168` |
| `src/ggml/src/ggml-cpu/ggml-cpu.c` | op dispatch `:2138-2140` |
| `src/ggml/src/ggml-cuda/gated_delta_net.cu` / `gated_delta_net.cuh` | CUDA kernel, `S_v`-specialized launches (`16/32/64/128`), rows-mode and raw-gates support |
| `src/ggml/src/ggml-cuda.cu` | GDN→cache-copy fusion `:2754-2802`, `:3391-3403` |
| `src/ggml/src/ggml-metal-ops.cpp` | Metal `SET_ROWS` scatter fold `:1923-2001` |
| `src/ggml/src/ggml-et/…`, `ggml-opencl/CMakeLists.txt`, `ggml-hexagon/ggml-hexagon.cpp` | further backend ports |
| `src/src/llama-hparams.cpp` / `.h` | the `ssm_*` hyperparameters and `n_embd_r()`/`n_embd_s()` sizing `:183-228` |
| `src/src/llama-model.cpp` | `gdn_v_grouped` key and fold `:1267`, `:2121-2130` |

## Known issues

- **No TA/TQ issue covers the GDN path.** The defect inventory ([[source-state-md]]) concerns the KV stack; the 48 recurrent blocks have no issue pages, no kernel-benchmark rows, and no separated profiling — the 0.91 % TriAttention attribution says nothing about what the other 48 blocks spend.
- **Untracked multi-sequence hazard in the state path:** `build_rs` relocates/zeroes recurrent state *before* the GDN op that reads it; the source records the read-before-write hazard explicitly and leaves the correct reordering as a follow-up (`src/src/llama-graph.cpp:3733-3745`). A same-step overlapping row is read before it is relocated. No issue page exists.
- **`gdn_v_grouped` is trusted, not checked** — see above. A mis-grouped fold would load silently.
- Recurrent state is F32 and `(1 + n_rs_seq)`-widened, so speculative decoding multiplies ~150 MiB of SSM state per sequence by `(1 + n_rs_seq)` ([[hybrid-memory]]).

## See also

[[qwen35-architecture]] · [[hybrid-memory]] · [[kv-cache]] · [[kv-eviction]] · [[triattention]] · [[ternary-bonsai-2-27b]] · [[prism-hadamard-weight-fold]] · [[walsh-hadamard-transform]] · [[speculative-decoding]] · [[overview]]
