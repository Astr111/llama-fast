---
title: Turbo WHT (GGML_OP_TURBO_WHT)
type: entity
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/ggml/src/ggml-cuda/turbo-wht.cu, src/ggml/src/ggml-cuda/turbo-wht.cuh, src/ggml/src/ggml-cuda/fwht.cu, src/ggml/src/ggml-cuda/fwht.cuh, src/ggml/src/ggml-cuda/ggml-cuda.cu, src/ggml/src/ggml-cuda/turbo-quant.cuh, src/ggml/src/ggml-cuda/set-rows.cu, src/ggml/src/ggml-cuda/triattention-score.cu, src/ggml/src/ggml.c, src/ggml/include/ggml.h, src/src/llama-graph.cpp, src/src/llama-kv-cache.cpp]
tags: [walsh-hadamard-transform, turboquant, cuda, kv-cache, quantization]
---

# Turbo WHT (`GGML_OP_TURBO_WHT`)

## What it is

The **live rotation operator of the TurboQuant stack, exposed as a first-class ggml op**. Where [[walsh-hadamard-transform]] is the hub for the transform family, this page is the runtime op that actually executes on the graph's rotation sites: `GGML_OP_TURBO_WHT` (enum `src/ggml/include/ggml.h:586`, name `"TURBO_WHT"` at `src/ggml/src/ggml.c:1126`), implemented once for CUDA in `src/ggml/src/ggml-cuda/turbo-wht.cu` and dispatched at `src/ggml/src/ggml-cuda/ggml-cuda.cu:2092-2093` → `ggml_cuda_turbo_wht`.

It is a **signed, normalized, radix-2 Hadamard butterfly with both directions in one kernel**: `direction = 0` forward = `signs1 → butterfly → 1/√n → signs2` (Q pre-rotation), `direction = 1` inverse = `signs2 → butterfly → 1/√n → signs1` (attention-output un-rotation). The kernel's own header states the convention (`turbo-wht.cu:8`) and that it mirrors the CPU implementation in `ggml-cpu/ops.cpp` (`:12-16`).

## How it works

### Op contract

Built by `ggml_turbo_wht(ctx, a, direction, group_size, scale)` (`src/ggml/src/ggml.c:6634`; declaration `src/ggml/include/ggml.h:2695`):

- asserts `ggml_is_contiguous(a)` and `a->type == GGML_TYPE_F32` (`ggml.c:6640-6641`), `direction == 0 || direction == 1` (`:6642`);
- `group_size == 0` means auto-detect: `(a->ne[0] % 128 == 0) ? 128 : 64` (`:6644-6647`), then `GGML_ASSERT(group_size == 64 || group_size == 128)` and `GGML_ASSERT(a->ne[0] % group_size == 0)` (`:6648-6649`);
- the result is a fresh 4-D F32 tensor of `a->ne` (`:6651`), `op = GGML_OP_TURBO_WHT`, `src[0] = a`, `src[1] = scale` — the InnerQ `scale_inv`, explicitly allowed to be `NULL` ("`NULL` = no scaling", `:6654-6655`);
- **op_params layout**: `op_params[0..3] = direction` (int), `op_params[4..7] = group_size` (int) (`:6657-6659`), read back with the same two `memcpy`s in the kernel (`turbo-wht.cu:122-125`).

CUDA support predicate (`ggml-cuda.cu:5493-5495`): `src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && src[0]->ne[0] % 64 == 0`.

**`wht_group` is a different channel — do not confuse it with this op's params.** `src/src/llama-kv-cache.cpp:1586-1587`, `:1637-1638` and `:1663-1664` write `int32_t wht_group = 128` into the op_params of the **`ggml_set_rows` result** produced by `cpy_k` / `cpy_v` ("always 128 with padding" — the head dim is zero-padded to the next multiple of 128 first, `:1630-1634`), not into a `TURBO_WHT` node. That value is consumed by the encode kernels: `set-rows.cu:551-554` and `:899` (`memcpy(&group_size, dst->op_params, sizeof(int))`, with the comment at `:551` naming `llama-kv-cache.cpp` as the writer). So `wht_group` belongs to the `GGML_OP_SET_ROWS` rotation, and `GGML_OP_TURBO_WHT` never reads it. ([[source-kv-mean-center]] is the other K-cache-side contract.)

### The kernel

One launch per op: grid = `n_groups = groups_per_head * n_heads`, block = `group_size` threads (128 or 64), with four template instantiations (`direction` × `group_size`) selected on the host. Per block:

1. load the group into `__shared__ float x[group_size]` (`turbo-wht.cu:42`), where `base = head_idx * head_dim + grp_in_head * group_size` (`:40`) — groups are **tiled across the head**, so `head_dim` need not equal `group_size`;
2. *(forward only)* `x[t] *= scale_inv[t % group_size]` (`:48-53`);
3. multiply by `TURBO_WHT_SIGNS1[t]` (forward) or `TURBO_WHT_SIGNS2[t]` (inverse), 64-wide arrays when `group_size == 64` (`:55-58`);
4. butterfly via the `WHT_STAGE(h)` macro (`:66-68`): `if (t % (2h) < h) { a = x[t]; b = x[t+h]; x[t] = a+b; x[t+h] = a-b; } __syncthreads();` — **stages `WHT_STAGE(1) (2) (4) (8) (16) (32)`, plus `WHT_STAGE(64)` only when `group_size == 128`** (`:70-76`). Seven stages at 128, six at 64, `log2(group_size)` as documented (`:15`), one `__syncthreads()` per stage;
5. normalize by `inv_sqrt` — `0.08838834764831845f` (1/√128) or `0.125f` (1/64) (`:80`) — and multiply by the *second* sign array (`SIGNS2` forward, `SIGNS1` inverse) (`:82-88`);
6. *(inverse only)* `result *= scale_inv[t % group_size]` (`:90-92`).

The InnerQ placement contract is stated in-code and enforced by the branch order: forward scales **before** signs+butterfly (Q pre-rotation), inverse scales **after** (`:18-20`, `:48-50`, `:90-92`). See [[innerq]]. `t % group_size` is a no-op in both places (threads == `group_size`).

**Tail elements.** `k_turbo_wht_copy_tail` (`:100-112`) copies the `head_dim % group_size` trailing elements of each head unchanged ("only the full groups within each head are processed", `:10`). With the op as built it is **unreachable**: `ggml_turbo_wht` asserts `a->ne[0] % group_size == 0` (`ggml.c:6649`) and the CUDA support predicate requires `ne[0] % 64 == 0` with `group_size ∈ {64,128}`, so `tail_size` is always 0. The tail convention question therefore lives entirely on the encode side ([[tq-5-tail-elements]]).

### Where in the graph it is inserted

Five emission sites, all in `src/src/llama-graph.cpp`, all guarded on the KV type being `TURBO2_0`/`TURBO3_0`/`TURBO4_0`:

| Direction | Sites | Tensor | Position |
| :--- | :--- | :--- | :--- |
| forward, `group_size = 128` | `:2985`, `:3108`, `:3299` | `q`, after per-head padding to a multiple of 128 (`:2979-2986`, `:3102-3108`, `:3293-3299`) and a `ggml_cont` | **immediately before** `build_attn_mha(q, k, v, ...)` (`:2988`, `:3111`, `:3302`) — the three attention builders (FA, MHA, ISWA) |
| inverse, `group_size = turbo_group` | `:2707` (FA), `:2785` (non-FA) | the attention output (`cur` / `kqv`), after the attention compute, still inside the attention builder (`:2777` "inverse WHT on attention output (non-FA path)") | e.g. `cur = ggml_turbo_wht(ctx0, cur, 1, turbo_group, innerq_scale)` |

Group size on the inverse side is derived, not constant: `group_src = k_is_turbo ? k : v` and `turbo_group = (group_src->ne[0] % 128 == 0) ? 128 : 64` (`:2702-2704`, `:2780-2782`), emitted only if `cur->ne[0] % turbo_group == 0`. The comment at `:2699` gives the reason: "group size must come from K (which determines the WHT rotation), not V". The forward side is hard-wired `128` because Q has already been padded to a 128 multiple.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/ggml/src/ggml-cuda/turbo-wht.cu` | `k_turbo_wht_f32<direction, group_size>` (`:23`), `WHT_STAGE` (`:66`), stages (`:70-76`), normalization (`:80-88`), InnerQ forward (`:49-53`) / inverse (`:90-93`), tail copy (`:100`), dispatcher `ggml_cuda_turbo_wht` (`:117`) |
| `src/ggml/src/ggml-cuda/turbo-wht.cuh` | the one declaration `void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);` |
| `src/ggml/src/ggml.c`, `src/ggml/include/ggml.h` | enum `:586` / name `:1126`; builder `ggml_turbo_wht()` `:6634` / `:2695`; op_params = {direction, group_size}, `src[1]` = InnerQ `scale_inv` |
| `src/ggml/src/ggml-cuda/ggml-cuda.cu` | dispatch `:2092-2093`; support predicate `:5493-5495` |
| `src/src/llama-graph.cpp` | the five call sites (`:2707`, `:2785`, `:2985`, `:3108`, `:3299`) |
| `src/ggml/src/ggml-cuda/turbo-quant.cuh` | the sign arrays the kernel indexes: `TURBO_WHT_SIGNS1/2[128]` (`:47`, `:58`), `TURBO_WHT_SIGNS1_64/2_64[64]` (`:71`, `:78`) |
| `src/src/llama-kv-cache.cpp` | `wht_group = 128` written onto the `set_rows` result in `cpy_k`/`cpy_v` (`:1586-1587`, `:1637-1638`, `:1663-1664`) |

### Sibling implementations in the same tree

| File | Symbol(s) | Who reaches it | Status |
| :--- | :--- | :--- | :--- |
| `turbo-wht.cu` / `.cuh` | `k_turbo_wht_f32<dir, gs>`, `ggml_cuda_turbo_wht` | `ggml-cuda.cu:2092-2093` ← nodes from `ggml_turbo_wht()` ← the five graph sites above | **live — the rotation op** |
| `turbo-quant.cuh` | `turbo_fwht_128` (`:88`), `turbo_fwht_64` (`:108`), wrappers `turbo_rotate_forward` (`:127`) / `turbo_rotate_forward_64` (`:135`) | **nobody.** Repo-wide `grep -rn "turbo_rotate_forward" src/` returns only those two definition lines; `turbo_fwht_128/64` are reached only from those wrappers (`:129`, `:137`) | **dead code** — confirmed, matching the sibling report |
| `triattention-score.cu` | `cooperative_fwht_128` (`:47`), `inverse_wht_rotation_128` (`:72`) | `inverse_wht_rotation_128` is called at `:226` inside the scoring kernel; `cooperative_fwht_128` from it at `:79` | **live, separate path** (TriAttention key scoring), not the op |
| `fwht.cu` / `.cuh` | `ggml_cuda_op_fwht`, `ggml_cuda_op_fwht_signed` → `fwht_dispatch` → `fwht_cuda` (warp-per-row) / `fwht_cuda_smem` / `fwht_cuda_block` | `ggml-cuda.cu:1823` (mul_mat, hint `GGML_HINT_SRC0_IS_HADAMARD` on `src1`) and `:3502` (fused sign-flip + FWHT-hint matmul, gated at `:3492`, described at `:3482`) | **live but a different facility** — see below |
| `set-rows.cu` | the encode butterfly inside `k_set_rows_turbo3/turbo2/turbo4` | `GGML_OP_SET_ROWS` from `llama_kv_cache::cpy_k`/`cpy_v`; reads `wht_group` from its own op_params (`:551-554`, `:899`) | live on the **encode** path; not `GGML_OP_TURBO_WHT` |

### `fwht.cu` — what it is, and is it this fork's turbo path?

It is **not a ggml op at all**: this tree contains no `GGML_OP_FWHT` (grep for `GGML_OP_FWHT` across `src/ggml/src/ggml.c` and `src/ggml/include/ggml.h` returns nothing). `fwht.cu` is a CUDA-backend helper facility with exactly two entry points, and both are reached only from `ggml-cuda.cu` via the matmul hint `GGML_HINT_SRC0_IS_HADAMARD` (`src/ggml/include/ggml.h:452`):

- `ggml-cuda.cu:1823` — in the mul_mat path, `if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst))`: an unsigned FWHT applied to a matrix operand;
- `ggml-cuda.cu:3492-3502` — the fused variant, gated on `ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD` and called as `ggml_cuda_op_fwht_signed(*cuda_ctx, x, signs, mm)` (comment at `:3482`: "Hadamard sign flip + reshape + FWHT-hint matmul").

So it serves the **Hadamard-folded-weight matmul path** ([[prism-hadamard-weight-fold]]), not the turbo KV rotation. Reasons it cannot stand in for the turbo rotation without an API change: `fwht_dispatch` applies **at most one** sign array, and that on the input side only (`signs_row = signs + (r % n_blk) * N`, `n_blk = signs_t->ne[0] / n`), with a single scalar `scale = 1 / sqrtf(n)`; the turbo rotation needs **two distinct ±1 arrays bracketing** the butterfly (`signs1` before, `signs2` after). Its widths also differ: register kernels for `n = 64/128/256` (one warp per row, 4 rows per block), one-block-per-row kernels for `512…8192` (`FWHT_BLOCK_THREADS 256`, `NE = N/NT` registers per thread), the old shared-memory path reachable only with `GGML_CUDA_FWHT_LEGACY=1`, and it accepts `F16` or `F32` input with `F32` output. Whether `fwht.cu` originated upstream or was added by this fork is `[UNVERIFIED]` — no git access here; what is verified is that nothing in the turbo path calls it.

## Known issues

- [[tq-4-wht-numerical-mismatch]] — the duplication issue. **This reading confirms rather than changes its verdict**, and extends its map: the op's stages are the same ascending `h ∈ {1,2,4,8,16,32,64}` with the same `a+b` / `a−b` pair convention (`t > t+h` split by `t % (2h) < h`) and the same `1/√n` constants as the encode butterfly, so the "no demonstrable FP-order mismatch" conclusion holds for the live op too. Two additions: (a) the op contributes *more* dead weight than the issue lists — the tail-copy kernel (`turbo-wht.cu:100`) is unreachable through the op, and `t % group_size` is a no-op; (b) the tree's WHT-family inventory is larger than "three live implementations plus one dead pair": `fwht.cu` carries three more kernel shapes (six instantiations) and `turbo-quant.cuh`'s dead pair wraps a fourth butterfly — so the "unify the WHT implementations" fix sketch has a wider map, while `fwht.cu` itself is *not* a unification target for the rotation (one sign array vs two, see above).
- [[ta-1-wht-inversion-256]] — worth keeping the two mechanisms apart: the `head_dim = 256` inversion defect is on the **TriAttention scoring** side (`inverse_wht_rotation_128`'s single-128-block, `padded_hd == 128` guard — see [[walsh-hadamard-transform]]). The **op** un-rotates group-by-group and is dimension-agnostic, since it tiles heads with `base = head_idx * head_dim + grp_in_head * group_size` (`turbo-wht.cu:40`) and loops over `groups_per_head = head_dim / group_size`. "The WHT inversion is broken at `head_dim=256`" ([[overview]]) is therefore true of the eviction-scoring path only, not of the graph's rotation op.
- [[tq-5-tail-elements]] — tails bypass the rotation on the encode side; on the op side the tail branch is dead by construction (builder assert + support predicate), so `TURBO_WHT` never sees a partial group.
- [[ta-4-cooperative-fwht-race]] — a thread-count contract problem in the scoring helper, not in this op (the op's block size always equals its group size).

## Open questions

- Is `scale_inv` sized per head or per group? The kernel indexes it as `scale_inv[t % group_size]` (`:50`, `:92`) while the graph passes `group_size = 128` even when a head is 256 wide (two groups per head), and `turbo-quant.cuh:262` states "InnerQ only works when each WHT group = one head (group_size == head_dim)". Whether the two are consistent at `head_dim = 256` is `[UNVERIFIED]` — settling it requires reading the `turbo_innerq_scale_inv` allocation, which this page did not do.
- Should `GGML_OP_TURBO_WHT` absorb the encode-side butterfly (then `wht_group` and its op-params channel disappear) or should `set-rows` be the single home? The op is the only implementation that documents the InnerQ placement contract in comments, which argues for it; the encode path is the hot one, which argues against.
- The CPU fallback (`ggml_compute_forward_turbo_wht_f32` in `ggml-cpu/ops.cpp`, per [[walsh-hadamard-transform]]) is not covered by this page's reading — its tail/InnerQ behaviour is `[UNVERIFIED]` here.

## See also

[[walsh-hadamard-transform]] · [[turboquant]] · [[tq-4-wht-numerical-mismatch]] · [[ta-1-wht-inversion-256]] · [[prism-hadamard-weight-fold]] · [[innerq]] · [[kv-cache]] · [[quantization]] · [[ta-4-cooperative-fwht-race]] · [[tq-5-tail-elements]] · [[source-state-md]]
