---
title: Walsh-Hadamard Transform (WHT / FWHT)
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/ggml/src/ggml-cuda/turbo-quant.cuh, src/ggml/src/ggml-cuda/turbo-wht.cu, src/ggml/src/ggml-cuda/turbo-wht.cuh, src/ggml/src/ggml-cuda/turbo-innerq.cu, src/ggml/src/ggml-cuda/turbo-innerq.cuh, src/ggml/src/ggml-cuda/set-rows.cu, src/ggml/src/ggml-cuda/triattention-score.cu, src/ggml/src/ggml-cuda/ggml-cuda.cu, src/ggml/src/ggml-cuda/common.cuh, src/ggml/src/ggml-cpu/ops.cpp, src/ggml/src/ggml.c, src/ggml/include/ggml.h, src/src/llama-graph.cpp, src/src/llama-triattention.cpp, src/src/llama-kv-cache.cpp, src/src/turbo-rotation-data.h]
tags: [walsh-hadamard-transform, turboquant, triattention, quantization, cuda]
---

# Walsh-Hadamard Transform (WHT / FWHT)

## What it is

The **rotation applied to keys and queries before low-bit quantization**, and its inverse applied to the value/output side afterwards. It is the load-bearing step of both custom stacks: [[turboquant]] needs it to make 2/3-bit per-channel quantization well-conditioned, and [[triattention]] needs its *inverse* to score keys that are stored rotated.

The transform is a signed, normalized, radix-2 Hadamard butterfly: `R = (1/√n) · D₂ · H · D₁`, with `D₁`/`D₂` fixed ±1 diagonal matrices. The sign arrays are compiled in as `TURBO_WHT_SIGNS1[128]` / `TURBO_WHT_SIGNS2[128]` (`src/ggml/src/ggml-cuda/turbo-quant.cuh:47`, `:58`, "seed=42") and their 64-wide truncations `TURBO_WHT_SIGNS1_64` / `TURBO_WHT_SIGNS2_64` (`:71`, `:78`). Normalization constants are `0.08838834764831845f = 1/√128` and `0.125f = 1/64`, identical in every implementation that applies them.

This page is the **hub for three issues** — [[ta-1-wht-inversion-256]], [[ta-4-cooperative-fwht-race]], [[tq-4-wht-numerical-mismatch]] — because each of them presupposes the mechanism below and none of them is intelligible without it.

## How it works

### The transform itself

For `n = 128` (the primary group size) the butterfly runs `log₂ 128 = 7` stages: at stage `h ∈ {1,2,4,8,16,32,64}`, every pair `(x[j], x[j+h])` within an aligned `2h` block becomes `(a+b, a−b)`. The C++ comment on the sequential variant states the cost as "896 ops for n=128" (`turbo-quant.cuh:86-88`). After the stages, every element is multiplied by `1/√n` and then by the second sign array — so a full forward rotation is `signs1 → butterfly → ·(1/√n) → signs2` and the inverse is `signs2 → butterfly → ·(1/√n) → signs1` (both orders are implemented, see the table below).

### Five implementations, three of them live

The tree carries the same transform **five separate times**. The repo-wide call graph, verified by reading each site and by grepping `turbo_fwht|turbo_rotate_forward|inverse_wht_rotation|cooperative_fwht` across `src/`:

| # | Site | Form | Status |
| :-- | :--- | :--- | :--- |
| 1 | `turbo-quant.cuh:88` `turbo_fwht_128` / `:108` `turbo_fwht_64`, wrapped by `turbo_rotate_forward()` `:127` and `turbo_rotate_forward_64()` `:135` | sequential triple loop, one thread, in-place on a register array | **dead** — `turbo_rotate_forward`/`_64` have **no callers anywhere under `src/`** (repo-wide grep returns only their definitions) |
| 2 | `set-rows.cu:332` `WHT_STAGE_SHARED(h)` inside `k_set_rows_turbo3` (`:237`), stages at `:337-343`, normalization `x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2[j]` at `:346-351` | parallel shared-memory butterfly, one block per group, `GROUP_SIZE` threads, `__syncthreads()` per stage | **live — the encoding path.** Every KV write runs it. The turbo2 encoder mirrors it (`k_set_rows_turbo2`, `:606+`, `WHT_STAGE_SHARED_T2` defined `:701`, stages `:705-712`); turbo4 has its own `WHT_STAGE_SHARED_T4` (`:1040`) and is dispatched through the same file (`:1129`) |
| 3 | `turbo-wht.cu:23` `k_turbo_wht_f32<direction, group_size>` | parallel butterfly over `__shared__ float x[group_size]`, **both directions in one kernel**, plus InnerQ scaling hooks (`:49-53` forward, `:90-93` inverse); tail elements copied unchanged by `k_turbo_wht_copy_tail` (`:100`) | **live — the graph-visible rotation op**, reached as `GGML_OP_TURBO_WHT` (`ggml-cuda.cu:2092-2093`) from `ggml_turbo_wht()` (`ggml.c:6634`; declaration `ggml.h:2695`) |
| 4 | `triattention-score.cu:47` `cooperative_fwht_128(smem, tid)` + `:72` `inverse_wht_rotation_128(smem, tid)` | parallel shared-memory butterfly with a **hard 64-thread contract** and a 128-element block contract | **live** in the GPU scoring kernel — this is the *only* inverse WHT on the TriAttention path |
| 5 | `ops.cpp:12253` `ggml_compute_forward_turbo_wht_f32` with `turbo_wht_s1`/`turbo_wht_s2` (`:12250-12251`), dispatched at `:12336-12340` | CPU scalar butterfly | **live** for CPU-offloaded graphs |

There is also a **sixth, non-butterfly** form: the precomputed 128×128 rotation matrices `TURBO_ROTATION_RT` and `TURBO_ROTATION_R` (`src/src/turbo-rotation-data.h:3`, `:2054`), consumed as dense matrix–vector products by the TriAttention **CPU** fallback (`matvec_128(TURBO_ROTATION_RT, dequant_tmp.data() + b, final_dst + b)`, `src/src/llama-triattention.cpp:90`, `:619-620`) and uploaded as tensors for the device path (`src/src/llama-kv-cache.cpp:429-430`, `:538-539`).

### The call graph, in one line

`llama-graph.cpp` requests the forward rotation (`direction = 0`) on Q at `:2985`, `:3108`, `:3299` — all three guarded by the Q padding block at `:2979-2986` — and the **inverse** (`direction = 1`) on the attention output at `:2707` (inside `build_attn_mha`, the FlashAttention path, guarding on `v->type` at `:2700`) and at `:2785` (the non-FA path). All of these become `GGML_OP_TURBO_WHT` tensors and execute implementation #3 on CUDA, #5 on CPU. The encode side (#2) is reached through `GGML_OP_SET_ROWS`, not through the WHT op at all — that asymmetry is why the same math exists twice.

### The TriAttention side, and the two guards that matter

`triattention_score_kernel<..., bool NEED_WHT_INV, ...>` dequantizes one KV head into shared memory (`dequant_head_to_smem`, `triattention-score.cu:93`, called at `:207`), and then:

- The `NEED_WHT_INV` block (`:211-230`) iterates `for (uint32_t b = 0; b < padded_hd; b += 128)` (`:213`) with an **empty body** (`:215-222`, comments only), followed by `if (padded_hd == 128 && f < 64) inverse_wht_rotation_128(k_smem, f);` (`:225-227`) and the comment "For head_dim > 128, we'd need multiple passes" (`:228`). With `padded_hd == 256` — which is exactly `((256+127)/128)*128` for the target model (`llama-triattention.cpp:949-950`, and `head_dim=256` per [[ternary-bonsai-2-27b]]) — the guard is false and **no inversion happens at all**. That is [[ta-1-wht-inversion-256]].
- The launch config is `dim3 block(fc, 1, 1)` with `fc = cfg.freq_count = head_dim / 2` (`triattention-score.cu:351`, invariant asserted at `src/src/llama-triattention.cpp:170`). So `head_dim=128 → 64` threads, `head_dim=256 → 128` threads. `cooperative_fwht_128` documents "n must be 128, threads = 64 (one butterfly per thread per stage)" (`:42-44`) and writes `smem[tid*2]` / `smem[tid*2+1]` (`:62-64`), i.e. it addresses `0..127` only for `tid < 64`. That is the contract [[ta-4-cooperative-fwht-race]] presupposes.

### InnerQ placement — the contract the rotation carries

`turbo-wht.cu` states it in-code: for the forward (Q pre-rotation) direction the per-channel `scale_inv` is applied **before** signs+WHT (`:49-53`); for the inverse (V un-rotation) it is applied **after** WHT+signs (`:90-93`). The encoder applies `d_innerq_scale` on the *unscaled* values before the butterfly (`set-rows.cu`, InnerQ calibrate/apply steps preceding the WHT at `:288-297`), and publishes `scale_inv` to the host through `turbo_innerq_publish()` (`turbo-quant.cuh:249` → `turbo-innerq.cu:15`). See [[innerq]].

### Do the copies disagree numerically?

[[source-state-md]] §4 TQ-4 asserts the sequential `turbo_fwht_128` and the `set-rows.cu` butterfly "differ in floating-point operation ordering". Read side by side, they do **not**: both run the same seven `h` stages in the same ascending order and apply the same `a+b`/`a−b` to the same pairs, then the same `inv_sqrt_128`, then the same sign array (`turbo-quant.cuh:88-103` vs `set-rows.cu:332-351`). Thread-level parallelism changes which thread performs an operation, not the order of operations on a given element. The demonstrated problem is therefore **duplication, not discrepancy**: three live implementations plus one dead pair plus a matrix form, with the tail convention (below) differing from the group convention. [[tq-4-wht-numerical-mismatch]] reaches the same conclusion; this page carries the mechanism it rests on.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/ggml/src/ggml-cuda/turbo-quant.cuh` | sign arrays `:47/:58/:71/:78`; sequential `turbo_fwht_128` `:88`, `turbo_fwht_64` `:108`; dead wrappers `turbo_rotate_forward` `:127`, `_64` `:135`; InnerQ device/host statics `:147-157` |
| `src/ggml/src/ggml-cuda/turbo-wht.cu` | the live rotation op `k_turbo_wht_f32` `:23-96` (`WHT_STAGE` `:66`, stages `:70-76`, normalization `:80-89`), tail copy `:100`, dispatcher `ggml_cuda_turbo_wht` `:117-174`, launches `:151-161` |
| `src/ggml/src/ggml-cuda/set-rows.cu` | encoding butterfly in `k_set_rows_turbo3` `:237`/`:324-351`, launches `:575`/`:581`; tail kernel `k_set_rows_turbo3_tail` `:422` (comment `:415-418`: tails bypass WHT entirely) |
| `src/ggml/src/ggml-cuda/triattention-score.cu` | `cooperative_fwht_128` `:47`, `inverse_wht_rotation_128` `:72`, dead block `:213-223`, guard `:225-227`, launch config `:351-358` |
| `src/ggml/src/ggml-cuda/ggml-cuda.cu` | `GGML_OP_TURBO_WHT` → `ggml_cuda_turbo_wht` `:2092-2093` |
| `src/ggml/src/ggml.c`, `src/ggml/include/ggml.h` | op builder `ggml_turbo_wht()` `:6634` / `:2695` (op_params = direction, group_size; `src[1]` = InnerQ scale) |
| `src/ggml/src/ggml-cpu/ops.cpp` | CPU fallback `:12247-12340` |
| `src/src/llama-graph.cpp` | Q forward rotation (`direction=0`) `:2985`, `:3108`, `:3299` with padding guard `:2979-2986`; V/attn-output inverse (`direction=1`) `:2707` (FA) and `:2785` (non-FA) |
| `src/src/llama-triattention.cpp` | `matvec_128` `:90`, matrix inverse `:619-620`, `padded_hd` `:949-950`, `freq_count == head_dim/2` check `:170` |
| `src/src/turbo-rotation-data.h`, `src/src/llama-kv-cache.cpp` | dense `R`/`Rᵀ` `:3`/`:2054`; uploads `:429-430`, `:538-539` |

## Known issues

- [[ta-1-wht-inversion-256]] — **CRITICAL**: the inverse rotation is skipped entirely when `padded_hd > 128`, so the target model (`head_dim=256`) is pruned on un-inverted keys.
- [[ta-4-cooperative-fwht-race]] — the 64-thread contract of `cooperative_fwht_128` versus the `freq_count = head_dim/2` launch. **Mechanism note (verified from the two guards):** `f < 64` can only exclude a thread when `freq_count > 64`, and `padded_hd == 128` forces `head_dim ≤ 128` hence `freq_count ≤ 64`; the two conditions are therefore mutually exclusive as written, so `cooperative_fwht_128` is never called with `tid ≥ 64`. The genuine defect is the *mirror image*: at `head_dim = 256` the block has 128 threads, so any fix for [[ta-1-wht-inversion-256]] that simply relaxes the guard to `padded_hd % 128 == 0` would call a 128-element-block helper from 128 threads and write `smem[255]` in a 128-element region.
- [[tq-4-wht-numerical-mismatch]] — three live implementations and one dead pair of the same transform; the claimed FP-order mismatch is not demonstrable, so the issue is unification, not a live accuracy loss.
- [[tq-5-tail-elements]] — elements beyond the last full group get **no rotation and no InnerQ** on the encode side (`set-rows.cu:415-418`, tail kernel `:422`) and are copied unchanged by the op (`turbo-wht.cu:100`). The graph compensates by also skipping the Q rotation for such tensors (`llama-graph.cpp:2979-2986` "the graph guards on `ne[0] % 128`"), which keeps `<Q_tail, K_tail>` in the original space.
- Dead code: `turbo_rotate_forward()`/`turbo_rotate_forward_64()` and the sequential `turbo_fwht_128`/`turbo_fwht_64` they wrap are unreachable; the InnerQ host/device statics that live beside them in `turbo-quant.cuh:147-157` are a *second* copy of state that also exists in `turbo-innerq.cu` — see [[tq-2-innerq-host-state]] and [[tq-3-innerq-multigpu]] for which copy is authoritative.

> Contradiction (2026-09-28): [[source-state-md]] §3 TA-4 describes the race as "`active=true` but `tid >= 64`" inside `cooperative_fwht_128`. The code at `triattention-score.cu:225` and `:351` does not admit that state (see the mechanism note above). The state.md claim is recorded, not overwritten; the reachable defect is the head_dim=256 thread-count mismatch that [[ta-1-wht-inversion-256]]'s fix would expose.

## Open questions

- Which of the five implementations is intended to be canonical? The unification in [[roadmap]] item 6 does not name one. `k_turbo_wht_f32` is the only one reachable from ggml ops and the only one carrying the InnerQ placement contract in comments; the encoder butterfly is the only one on the hot path.
- `padded_hd` is computed as a 128-multiple in `llama-triattention.cpp:949-950`, so a 256-wide head is *never* a single 128-element block. Does the Release-side "dynamic `wht_group`" fix ([[source-state-md]] §5.1, unportable here) re-run `inverse_wht_rotation_128` per 128-block, and with how many threads? `[UNVERIFIED]` — that checkout is not present in this repository.
- The dense-matrix path (`TURBO_ROTATION_RT`) is a different arithmetic formulation (128×128 f32 matvec) from the butterflies. Whether it agrees bit-for-bit with the butterfly on the same input is `[UNVERIFIED]` — reconciling it would require running both, which this page did not do.

## See also

[[turboquant]] · [[triattention]] · [[innerq]] · [[quantization]] · [[kv-cache]] · [[ta-1-wht-inversion-256]] · [[ta-4-cooperative-fwht-race]] · [[tq-4-wht-numerical-mismatch]] · [[tq-5-tail-elements]] · [[turbo-wht]] · [[v100-sxm2]] · [[source-state-md]]
