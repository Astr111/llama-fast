---
title: CPU Path
type: entity
status: current
updated: 2026-09-28
sources: []
verified:
  - src/ggml/src/ggml-turbo-quant.c
  - src/ggml/src/ggml-cpu/ops.cpp
  - src/ggml/src/ggml-quants.c
  - src/ggml/src/ggml-cpu/ggml-cpu.c
  - src/ggml/src/ggml-cuda/triattention-score.cu
  - src/ggml/src/ggml-cuda/triattention-score.cuh
tags: [cpu, quantization, walsh-hadamard-transform, triattention]
---

# CPU Path

## What it is

Everything the fork's custom operations do when the work is not on the GPU: the reference codec for `turbo2`/`turbo3`/`turbo4` (`src/ggml/src/ggml-turbo-quant.c`), the CPU twins of the rotation op (`GGML_OP_TURBO_WHT`) and of the plain WHT, the turbo `vec_dot` entries registered in the ggml CPU type traits, and — outside ggml — the host-side TriAttention scoring fallback that [[ta-3-cpu-fallback-transfers]] tracks.

It matters for two reasons. The ggml-level CPU machinery is the **only readable statement of the intended encode semantics** of the turbo formats: it runs the whole pipeline (norm → normalize → signed WHT → nearest centroid → corrected norm) in plain scalar code, so a future GPU kernel can be diffed against it. And the host-side TriAttention scoring is the degraded path that stalls 15–30 s per prune round ([[ta-3-cpu-fallback-transfers]]). Both are reachable in ways this page pins down; neither is "the same thing, just slower" — `turbo4` on the CPU is a *different codec* (see the verdict below).

## How it works

### 1. Entry points

| CPU entry point | What it implements | file:line | GPU counterpart it mirrors |
| :--- | :--- | :--- | :--- |
| `quantize_row_turbo3_0_ref` / `quantize_row_turbo2_0_ref` | full encode: group L2 norm → normalize → signed WHT (`turbo_cpu_fwht`) → nearest centroid → **corrected norm** `grp_norm/recon_norm` written to every block | `ggml-turbo-quant.c:240-310` (t3; group-size read `:249-252`, corrected norm `:305-310`), `:333-395` (t2; read `:344-348`) | `k_set_rows_turbo3` / `k_set_rows_turbo2` (`set-rows.cu:237` / `:608`, WHT `:324-351` / `:692-725`); same corrected norm at `set-rows.cu:402-407` |
| `quantize_row_turbo4_0_ref` | **alternate codec**: dense Gaussian-QR rotation (`turbo_rotation`, seed 42) instead of the WHT; default `TURBO4_USE_4BIT` → 4-bit centroids, no QJL (`#else` branch keeps legacy 3-bit + QJL) | `ggml-turbo-quant.c:429` (rotation init `:67-117`) | `k_set_rows_turbo4` (`set-rows.cu:952`, WHT `:1039-1062`) — **divergent**, see §2 |
| `dequantize_row_turbo3_0` / `_turbo2_0` / `_turbo4_0` | centroid × norm unpack (t3/t2); t4 additionally inverse-rotates with `matvec(turbo_rotation_t, …)` — again Gaussian, not WHT | `ggml-turbo-quant.c`, after `:310` (t3), after the t2 quantize (`:397-410`), `:539` (t4) [bounds unverified] | `dequantize_turbo3_0`/`_turbo2_0`/`_turbo4_0` (`dequantize.cuh:154-176`) |
| `quantize_turbo3_0` / `_turbo2_0` / `_turbo4_0` | row-batched wrappers for host-side conversion (`llama-quantize`) | `ggml-turbo-quant.c` (one after each dequantize) | none — quantization is host-side by construction |
| `turbo_cpu_fwht` | signed WHT butterfly: `x·s1` → `log2(g)` butterfly → `x·(1/√g)·s2`; sign arrays `turbo_cpu_s1/s2` | `ggml-turbo-quant.c:219-240`, arrays `:202-215` | `turbo_fwht_128`/`_64` (`turbo-quant.cuh:88-139`) plus the inline WHT inside the `set-rows` kernels |
| `ggml_compute_forward_turbo_wht_f32` | `GGML_OP_TURBO_WHT` graph op: forward (`direction 0`, Q pre-rotation) and inverse (`direction 1`, attention output), `group_size` from `op_params`, InnerQ `scale_inv` applied pre-rotation (fwd) / post (inv), 64-group uses the first 64 signs, tail elements copied unchanged | `ops.cpp:12253-12331` (own copy of the sign arrays `:12250-12251`; type switch `:12336-12342`; dispatch case `ggml-cpu.c:2157-2160`) | `ggml_cuda_turbo_wht` (`turbo-wht.cu:117-174`) |
| `ggml_compute_forward_fwht` | **plain unsigned** WHT (×`1/√n`, no sign arrays, SIMD butterfly) — substituted for `MUL_MAT` when src0 is flagged `GGML_HINT_SRC0_IS_HADAMARD` (WHT-domain weights) | `ops.cpp:12066-12140` (dispatch `:12142-12160`; hook inside `ggml_compute_forward_mul_mat`, `ggml-cpu.c:1332-1337`) | none verified in this scope [UNVERIFIED] |
| `ggml_vec_dot_turbo3_0_f32` / `_turbo2_0` / `_turbo4_0` | dequantize turbo row to f32, scalar f32 dot with the operand row; registered as `.vec_dot` in the ggml CPU type traits | `ggml-cpu.c:3528` / `:3548` / `:3567`; traits table `:440-457` (`vec_dot_type = F32`, `nrows = 1`) | fused FA KQ helpers `vec_dot_fattn_vec_KQ_turbo3_0` etc. (`fattn-common.cuh:333` / `:387` / `:436`) — those dequantize inline in registers |
| CPU TriAttention scoring | **none in ggml-cpu** — `grep triattention` over `ops.cpp` + `ggml-cpu.c` returns zero; scoring kernels exist only in CUDA. The CPU *fallback* is host code: `triattention_dequant_kv_head` (`src/src/llama-triattention.cpp:540`), one `ggml_backend_tensor_get` per KV cell (`:566`/`:572`) | n/a (grep result) / per [[ta-3-cpu-fallback-transfers]] | `triattention_score_kernel` (`triattention-score.cu:167-`) |

`src/ggml/src/ggml-quants.c` contains **no** turbo entries at all (grep: zero matches) — `ggml-turbo-quant.c` is a separate translation unit and the only home of the codec.

### 2. Reference-semantics verdict

**Faithful for `turbo2`/`turbo3`; not a reference at all for `turbo4`; and never a performance model.**

- For t2/t3 the CPU path is a complete, readable statement of the format: the encode pipeline and the corrected-norm formula match the CUDA encoder point for point (`:305-310` vs `set-rows.cu:402-407`), the WHT stage is the same `(1/√g)·D₂·H·D₁` with the same seed-42 sign arrays, and the dequant is a trivial centroid × norm. A future GPU port can be checked against it.
- It is scalar and stack-buffer-bound on purpose — the codec quantizes per group into `float buf[128]`, and the dot below uses a 16 KiB f32 scratch with an `n ≤ 4096` assert — so it is ground truth for *values*, never for speed.
- **`turbo4` is a different codec on the CPU**: `turbo_init_rotation` (seed 42) synthesises a dense random Gaussian matrix and QR-orthogonalises it (`:67-117`), and both `quantize_row_turbo4_0_ref` and `dequantize_row_turbo4_0` rotate with it via `matvec` — the CUDA `turbo4` encoder uses the signed WHT (`set-rows.cu:1039-1062`). The CPU t4 was the legacy 3-bit+QJL design (its `#else` branch survives); it cannot validate the GPU codec. Known in [[turboquant]] §8.
- Two stray comments misdescribe the code: `dequantize_row_turbo3_0` says "Stub — Metal shader handles dequant on GPU" above a fully implemented body, and the t4 dequant says "TODO: add proper 4-bit centroid table to C code (currently only in Metal)" directly above a defined `CENTROIDS_4BIT` table. Both are stale; the code is complete either way.

### 3. `ggml_vec_dot_turboN_0_f32`

All three live as `static` functions in `ggml-cpu.c` (`:3528`, `:3548`, `:3567`) and are reachable only through the ggml CPU type-traits `vec_dot` slots (`:440-457`). Each asserts `nrc == 1`, dequantizes the turbo operand into `float tmp[4096]` (`GGML_ASSERT(n <= 4096)` — a 16 KiB stack scratch; the sibling report of an "int8 scratch" is the packed `qs` block payload, not the scratch, which is f32), then runs a scalar `sum += tmp[i] * y[i]` loop. Interpretation: this is a **dequantize-then-dot reference implementation** — no quantized-arithmetic tricks, no SIMD, no block-norm factoring — so it is exactly the arithmetic a GPU port must reproduce, and the 4096 cap just mirrors the maximum head dim. It is the "CPU reference dot product" the vault keeps referencing.

**Reachable in a normal GPU run — yes.** The comment at `ggml-cpu.c:3526-3527` documents the trigger: "Used by CPU flash attention for models with D not supported by CUDA FA (e.g. D=192)". When the CUDA fused-attention kernel cannot handle the head dim, the flash-attention node is scheduled to the CPU backend inside an otherwise-GPU graph, and this `vec_dot` runs there. CUDA FA never uses it (its turbo KQ dots are the fused in-register helpers). So the ggml CPU path is *not* gated on GPU-init failure.

### 4. Reachability and the TA-3 cost

Two distinct "CPU paths":

1. **ggml-level (codec, `TURBO_WHT` op, `vec_dot`) — production-reachable alongside the GPU.** CPU FA for head dims CUDA FA rejects (above); any op the CUDA backend's `supports_op` denies (turbo `GET_ROWS` aborts on CUDA and `supports_op` admits turbo only for `SET_ROWS` — [[turboquant]] §5); and all conversion/quantization, which is host-side by definition.
2. **Host-side TriAttention scoring — only on GPU-init failure.** This is the [[ta-3-cpu-fallback-transfers]] path: `triattention_dequant_kv_head` pulls every candidate KV cell to host with one synchronous `ggml_backend_tensor_get` each — ~4096 separate ~1 KiB D2H transfers at `n_decode=4096`/`head_dim=256` — costing 15–30 s per prune round (severity HIGH, [[source-state-md]]). The host scorer is also the only consumer of the dense `turbo_rotation` tensors ([[turboquant]] §3), i.e. the CPU TA fallback inverts rotation with the Gaussian matrix, not the WHT.

So the TA-3 worth on this path is dominated by the transfer pattern, not the CPU arithmetic itself; and the ggml CPU machinery exists in normal runs regardless of GPU health.

### 5. WHT-stage structure: same as `[[turbo-wht]]`

Yes for everything the fork actually rotates. The CPU encode path and the `TURBO_WHT` op both implement the signed WHT exactly as the `[[turbo-wht]]`/`[[walsh-hadamard-transform]]` structure: `D₁` signs → `log2(g)` butterfly → `(1/√g)·D₂`, with two independent copies of the identical 128-element seed-42 arrays (`turbo_cpu_s1/s2` at `ggml-turbo-quant.c:202-215`; `turbo_wht_s1/s2` at `ops.cpp:12250-12251`; sampled prefixes match in this read, and both files declare the arrays "must match" the CUDA/Metal shaders). The **third variant** is not a signed-WHT variant at all: the plain `ggml_compute_forward_fwht` (`ops.cpp:12066`, unsigned, ×1/√n) used by the `GGML_HINT_SRC0_IS_HADAMARD` matmul path, and the CPU `turbo4` codec's dense Gaussian-QR rotation, which replaces the WHT entirely.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/ggml/src/ggml-turbo-quant.c` | the codec: `turbo_cpu_fwht` `:219-240`, sign arrays `:202-215`, centroid tables, `quantize_row_turbo{3,2,4}_0_ref`, `dequantize_row_turbo{3,2,4}_0`, `quantize_turbo{3,2,4}_0` wrappers, the WHT-group global `:19` (no writers inside `src/ggml` — grep finds only the declaration and the two reads `:249`/`:344`; [[turboquant]] puts the setter in the llama-layer `SET_ROWS` path) |
| `src/ggml/src/ggml-cpu/ops.cpp:12253-12342` | `ggml_compute_forward_turbo_wht_f32` + dispatch |
| `src/ggml/src/ggml-cpu/ops.cpp:12066-12160` | plain `ggml_compute_forward_fwht` (unsigned WHT) |
| `src/ggml/src/ggml-cpu/ggml-cpu.c:440-457` | ggml CPU type traits: `.from_float`, `.vec_dot`, `.vec_dot_type`, `.nrows` for the three turbo types |
| `src/ggml/src/ggml-cpu/ggml-cpu.c:3528, 3548, 3567` | `ggml_vec_dot_turbo{3,2,4}_0_f32` |
| `src/ggml/src/ggml-cpu/ggml-cpu.c:1332-1337, 2157-2160` | the `GGML_HINT_SRC0_IS_HADAMARD` hook inside `ggml_compute_forward_mul_mat`; the `GGML_OP_TURBO_WHT` dispatch case |
| `src/ggml/src/ggml-quants.c` | nothing turbo (verified by grep) |
| `src/src/llama-triattention.cpp:540` | host-side TA scoring fallback (per [[ta-3-cpu-fallback-transfers]]; not re-read this pass) |
| `src/ggml/src/ggml-cuda/triattention-score.cu` / `.cuh` | the GPU scoring kernel, for contrast — the only scoring kernels in the tree |

## Known issues

- [[ta-3-cpu-fallback-transfers]] — per-cell synchronous D2H transfers on the host-scoring path; 15–30 s stalls at realistic decode lengths.
- CPU `turbo4` is the legacy Gaussian-QR codec, not the WHT codec the GPU encodes with ([[turboquant]] §8) — no issue filed upstream of this page.
- The CPU vec_dot caps at `n ≤ 4096` (`ggml-cpu.c:3536` etc.) and is scalar; it is a reference, not a performance path — relevant to [[tq-1-missing-gemm-kernels]] only as ground truth.
- Stale comments ("Stub — Metal…", "TODO…only in Metal") on complete CPU code (§2).
- `turbo3_cpu_wht_group_size` has no setter inside `src/ggml`; unset, group size is inferred from row 128-alignment (`ggml-turbo-quant.c:251-253`) while the GPU path pins 128 via KV padding — a divergence surface for group semantics.
- The `GGML_HINT_SRC0_IS_HADAMARD` plain-WHT path has no verified CUDA counterpart in this scope.

## See also

[[quantization]] · [[turboquant]] · [[turbo-wht]] · [[walsh-hadamard-transform]] · [[ta-3-cpu-fallback-transfers]] · [[backend-parity]] · [[tq-1-missing-gemm-kernels]] · [[triattention]] · [[ta-1-wht-inversion-256]]