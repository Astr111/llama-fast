---
title: Quantized kernel units
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/ggml/src/ggml-cuda/mmf.cu, src/ggml/src/ggml-cuda/mmvf.cu, src/ggml/src/ggml-cuda/mmvq.cu, src/ggml/src/ggml-cuda/mmvq.cuh, src/ggml/src/ggml-cuda/mmq.cu, src/ggml/src/ggml-cuda/mmq.cuh, src/ggml/src/ggml-cuda/mma.cuh, src/ggml/src/ggml-cuda/mmq-config-pascal.cuh, src/ggml/src/ggml-cuda/mmq-config-ampere.cuh, src/ggml/src/ggml-cuda/mmq-hopper-q1.cu, src/ggml/src/ggml-cuda/ggml-cuda.cu, src/ggml/src/ggml-cuda/vecdotq.cuh, src/ggml/src/ggml-cuda/convert.cu, src/ggml/src/ggml-cuda/dequantize.cuh, src/ggml/src/ggml-cuda/turbo-quant.cuh, src/ggml/src/ggml-cuda/turbo-wht.cu, src/ggml/src/ggml-cuda/fattn-common.cuh, src/ggml/src/ggml-cuda/fattn.cu, src/ggml/src/ggml-cuda/set-rows.cu, src/ggml/src/ggml-cuda/triattention-score.cu, src/tests/CMakeLists.txt]
tags: [cuda, gemm, quantization, kernels]
---

# Quantized kernel units

## What it is

The four CUDA translation units that actually compute a matrix product, and the per-type tables inside them that decide *how*. `[[gemm-dispatch]]` covers the runtime chain that picks one of them; this page covers the units themselves — what arithmetic each performs, what hardware it assumes, and which quantized types each one's table really contains.

| Unit | Entry point / file | Computes | Hardware class it targets | Quantized types in its table |
| :--- | :--- | :--- | :--- | :--- |
| **MMF** | `ggml_cuda_mul_mat_f` / `mul_mat_f_switch_rows_per_block` — `mmf.cu` | matrix × matrix in a low-precision *float* type (`F32`/`F16`/`BF16`, `Vals_per_T` 1/2/2) | tensor cores: F32→TF32 needs Ampere+ or AMD MFMA; F16→Volta/Turing+ or AMD WMMA; BF16→Ampere+ | **none.** Three cases only; `default:` calls `GGML_ABORT("unsupported type: %s")` (`mmf.cu:128-129`), and `ggml_cuda_should_use_mmf` returns `false` at its default (`mmf.cu:188-189`) |
| **MMVF** | `ggml_cuda_mul_mat_vec_f` — `mmvf.cu` | matrix × vector for the same three float types, with GLU fusion (SiLU/SwiGLU, `ggml_cuda_op_silu_single`) and an `ncols_dst` block-size ladder (32…256) | tensor-core-capable GPUs for F16/BF16; float4 dot with `type_acc` = half or float | **none.** Same three types twice (`mmvf.cu:701-722`, `:760-781`), both `default:` → `GGML_ABORT("unsupported type")`; `ggml_cuda_should_use_mmvf` default → `false` (`mmvf.cu:866-867`) |
| **MMVQ** | `ggml_cuda_mul_mat_vec_q` / `mul_mat_vec_q_switch_type` — `mmvq.cu` (declarations in `mmvq.cuh`) | quantized weights × q8_1-quantized activations, one weight row per warp, `vec_dot_q_cuda_t` per block, batches up to `MMVQ_MAX_BATCH_SIZE` = 8 (`mmvq.cuh:3`) | every CUDA arch that has integer `dp4a`/`__dp4a` dot products; no tensor cores required, so Pascal through Blackwell | 25 entries: `Q1_0`, `Q2_0`, **`PQ2_0`**, **`PTQ1_0`**, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `MXFP4`, `NVFP4`, `Q2_K`…`Q6_K`, `IQ1_S`, **`IQ1_M`**, `IQ2_XXS/XS/S`, `IQ3_XXS/S`, `IQ4_NL/XS` (`mmvq.cu:11-40`); `default: return nullptr` (`:38`). **No `TURBO*` case** |
| **MMQ** | `ggml_cuda_mul_mat_q` / `mul_mat_q_case<type>` — `mmq.cu` + `mmq.cuh` | tiled quantized GEMM: an integer `dp4a` tile path and an MMA tile path (shared-memory tiles `I`×`J`, `MMQ_ITER_K` = 256, `MMQ_ITER_K_FP4` = 512), q8_1 activations (`MMQ_Q8_1_DS_LAYOUT_D4` for `Q1_0`/`Q2_0`/`PQ2_0`/`PTQ1_0`, `mmq.cuh:65-70`) | `dp4a` path: any arch; MMA path: Turing+ (`mma.cuh` fragments); arch config chosen per family (see below) | 24 entries, the `DECL_MMQ_CASE` list (`mmq.cuh:1676-1704`) = the MMVQ set **minus `IQ1_M`**; includes **`PQ2_0`** and, behind `#if !defined(GGML_USE_HIP)`, **`PTQ1_0`** (`:1678-1681`); `default: GGML_ABORT("fatal error")` (`mmq.cu:88-90`). **No `TURBO*` case** |

The per-type util functions are a second table inside MMQ, separate from the case list:

- `dp4a` layout: `ggml_cuda_mmq_load_tiles_*` + a type-specific VDR, e.g. `VDR_PQ2_0_Q8_1_MMQ` and `VDR_PTQ1_0_Q8_1_MMQ` (`mmq.cuh:573-583`).
- MMA layout: `PQ2_0`/`PTQ1_0` pass **`-1`** as the vector-dot id and reuse the generic MMA q8_0 dot with the D4 layout (`mmq.cuh:749-759`) — the custom types ride the generic path there, not a bespoke one.
- `mmq_get_dp4a_tile_x_sizes` maps `Q1_0`, `Q2_0`, `PQ2_0` and `PTQ1_0` all to the `MMQ_DP4A_TXS_Q8_0` layout (`mmq.cuh:400-409`), `default: tile_x_sizes{0, 0, 0}` (`:430`).
- `Q1_0`/`Q2_0`/`PQ2_0` are the only types with `async_buffer_y` — double-buffered Y tiles on DGX Spark (`mmq.cuh:932-935`, `:1491-1492`).

**No turbo type appears in any of these tables.** This was checked per table, case-insensitively, over the whole `src/ggml/src/ggml-cuda/` directory: the string `turbo` occurs in `convert.cu`, `dequantize.cuh`, `set-rows.cu`, `turbo-quant.cuh`, `turbo-wht.cu`, `triattention-score.cu`, `fattn.cu`, `fattn-common.cuh`, `fattn-vec.cuh`, `triattention`, and in `ggml-cuda.cu` (TURBO_WHT), but **never** in `mmf.cu`, `mmvf.cu`, `mmvq.cu`, `mmq.cu`, `mmq.cuh`, `vecdotq.cuh`, or any `mmq-config-*.cuh`. Turbo KV types have native dots only inside fused attention (`vec_dot_fattn_vec_KQ_turbo{2,3,4}_0`) and dequantize/round-trip kernels — see [[gemm-dispatch]], [[tq-1-missing-gemm-kernels]], [[turboquant]].

## How it works

**MMA fragment shapes (`mma.cuh`).** MMQ's tensor-core path is written against explicit PTX shapes, not wmma: `m16n8k16`/`m16n8k32` for `s8×s8→s32` (`mma.cuh:924`, `:946`), `m16n8k16` for `f16×f16→f16/f32` (`:977`, `:1163`), `m16n8k8` for `tf32` (`:1089`), `m16n8k16` for `bf16` (`:1187`), Volta's `m8n8k4` (`:1372`, `:1392`), and Turing fallbacks built by stacking `m8n8k8`/`m8n8k16` when the wider atom does not exist (`:928-934`, `:950-962`, `:981-987`, `:1009-1021`, `:1167-1173`, `:1211-1223`). Blackwell adds block-scaled `m16n8k64` for `MXFP4`/`NVFP4` (`:1136-1148`). A new quantized type that needs a fragment shape the file does not have must add one here; the custom PrismML types did not, because they reuse the existing integer and fp16 atoms.

**Arch split for the MMQ config.** `ggml_cuda_mmq_get_config` (`mmq.cuh:235-261`) is an if-ladder: AMD→CDNA/RDNA4/RDNA3.5/RDNA3/RDNA2; `cc == GGML_CUDA_CC_DGX_SPARK`→`gb10`; `blackwell_mma_available(cc)`→`blackwell`; `ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA`→**`ampere`**; else→**`pascal`**. The device-side mirror picks on `__CUDA_ARCH__` (`:263-289`). So a V100 (sm_70 ≥ Volta) is served by `mmq-config-ampere.cuh`, and the *pascal* file only serves pre-Volta devices — nothing in the tree selects it for Volta.

- Pascal rows are `CASE(type, 256, 2, 64, J, layout, MMQ_ITER_K, false, …)`; Ampere rows are `CASE(type, 256, 1, 128, J, layout, MMQ_ITER_K, true, …)` — different `I` (64 vs 128) and the boolean that `ggml_cuda_mmq_get_config(...).use_mma_data_layout()` reads is `false` on Pascal, `true` on Ampere. A `false` there routes to the `dp4a` util-func table (`mmq.cuh:559-600`).
- `mmq-config-pascal.cuh` contains `Q1_0`, `Q2_0`, **`PQ2_0`**, `Q4_0`…`Q8_0`, all k-quants, all IQ-quants, `MXFP4`, `NVFP4` — and **not `PTQ1_0`**; `mmq-config-ampere.cuh` contains `PTQ1_0` explicitly (`:53-68`).

**Is the `PTQ1_0` asymmetry deliberate or incidental?** The code makes it *mechanically* deliberate, though nothing says so in a comment:

- `ggml_cuda_should_use_mmq` gates `PTQ1_0` on Turing: `case GGML_TYPE_PTQ1_0: mmq_supported = turing_mma_available(cc);` (`mmq.cu:374-377`), while `Q1_0`/`Q2_0`/`PQ2_0` fall into the unconditional `mmq_supported = true` group (`:379-386`). `ggml_cuda_should_use_mmvq` likewise special-cases `PTQ1_0` only for `cc >= GGML_CUDA_CC_TURING` (`mmvq.cu:298-300`).
- The Pascal config branch is reachable only for `cc < Volta`, where `turing_mma_available` is false. `PTQ1_0` rows there would be unreachable, so their absence is consistent rather than forgotten — and `PQ2_0`'s *presence* is consistent for the same reason (it is not Turing-gated). `[INFERENCE]`: the intent is a Turing+ policy for `PTQ1_0`; the source states the gate but not the rationale for the config asymmetry.
- Note `PTQ1_0` does have a working `dp4a` MMQ implementation in the source (`mmq.cuh:580-583`), and its `dp4a` tile sizes are defined (`:407-408`) — the exclusion comes from the eligibility gate, not from a missing kernel. On CUDA, `PTQ1_0`'s MMQ path is also on by default at *every* batch (`MMQ_PTQ1_0_MAX_BATCH_SIZE (1 << 30)`, overridable by `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH`) because the fp16-dequantize-then-cuBLAS fallback is the source of its extra error on CUDA (`mmq.cuh:10`, `mmq.cu:427-431`).

**The Hopper-only path.** `mmq-hopper-q1.cu` is a fifth unit, gated far more tightly: a Hopper (`sm_90a`) `wgmma` MMQ implementation for `Q1_0` (and reachable for `PQ2_0` from the call site), built only when the CUTLASS include directory is provided (`#if defined(GGML_USE_HOPPER_Q1)`, `mmq-hopper-q1.cu:17-24`), using `GMMA::MMA_64x64x32_S32S8S8_SS_TN` with tiles `bM=bN=bK=128`, an fp32→int8 per-128 activation quantizer (`quant_act_per128`) and a one-time repack of `block_q1_0` into dense bit words plus fp32 scales.

The call site inside `ggml_cuda_mul_mat` (`ggml-cuda.cu:1866-1873`) is:

```c
if ((src0->type == GGML_TYPE_Q1_0 || src0->type == GGML_TYPE_PQ2_0) && ne11 >= 128
        && ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)
        && ggml_cuda_mul_mat_q1_hopper(ctx, src0, src1, dst)) {
    // handled by the opt-in Hopper wgmma path (returns false to fall through when unsupported)
    return;
}
```

Three consequences for the V100 target ([[v100-sxm2]]): (1) the path needs `ne11 >= 128`, so single-token decode never touches it — it is a prefill/batched path; (2) it requires Hopper `wgmma`, which sm_70 does not have, and the build gate is a compile-time CUTLASS option rather than a clean `cc` check, so on a Volta build it cannot be selected at all — Volta gets the standard MMQ path (the *ampere* config, above) or MMVQ at batch ≤ 8; (3) it covers `Q1_0`/`PQ2_0` **weights** only and does nothing for turbo KV types. The file header also warns it is **not bit-identical to standard MMQ** ("activations use a per-128-K int8 absmax scale, coarser than q8_1's per-32"). Whether the runtime `cc == sm_90` guard lives inside `ggml_cuda_mul_mat_q1_hopper` was not read — `[UNVERIFIED]`. The header comment names `GGML_CUDA_HOPPER_Q1` / `GGML_HOPPER_Q1_DISABLE` while the `#if` tests `GGML_USE_HOPPER_Q1`; the definitions were not read.

**Where a new `vec_dot_turbo*` would have to be registered.** A turbo KV dot product is not a one-line addition; the tables agree with each other by hand, so the same type must be added in each of them or the dispatch lands on cuBLAS (or aborts below `ne11 = 8`):

1. `vecdotq.cuh` — the dot body itself (block layout, rotation, VDR). This file currently contains **no** `turbo` at all, so there is nothing to extend, only to write; the block structs and helpers exist next door in `turbo-quant.cuh` ([[turboquant]], [[turbo-wht]]).
2. `mmvq.cu:11` `get_vec_dot_q_cuda` — a `case` returning the new function pointer; its `default` is `nullptr` (`:38`), which is only safe because the host predicate already refused the type.
3. `mmvq.cu:1194` `mul_mat_vec_q_switch_type` — a `case` launching `mul_mat_vec_q_switch_ncols_dst<TURBO*>`; without it the `default:` at `:1353-1354` calls `GGML_ABORT("fatal error")`. Plus `get_vdr_mmvq` (`:42-69`), the `get_mmvq_mmid_max_batch_*` ladder (`:126-256`), the MMVQ parameter-table selector (`:407+`), and the eligibility predicate `ggml_cuda_should_use_mmvq` (`:293-380`).
4. MMQ — `MMQ_Q8_1_DS_LAYOUT` switch (`mmq.cuh:65-70`), `mmq_get_dp4a_tile_x_sizes` (`:400-431`), the `dp4a` util-func switch (`:559-600`) and the MMA one (`:748+`), the `DECL_MMQ_CASE` list (`:1676-1704`), the host switch with its `GGML_ABORT` default (`mmq.cu:10-91`), and the `ggml_cuda_should_use_mmq` allow-list (`:374-409`, `default: false`).
5. The arch configs — rows in `mmq-config-{pascal,ampere,blackwell,cdna,rdna*}.cuh`; a type missing from a config makes `ggml_cuda_mmq_get_config` return `GGML_TYPE_COUNT`, which compiles the tile code out (`mmq.cuh:1063-1067`) and skips the type in the `J`-search (`:1588-1592`).
6. `ggml_backend_cuda_device_supports_op`'s `GGML_OP_MUL_MAT` allow-list, or the scheduler may never place the node on CUDA — see [[tq-1-missing-gemm-kernels]] and the `supports_op` section of [[gemm-dispatch]].

`mma.cuh` would need work only if the type demanded a new fragment shape; the existing set (`m16n8k16/32` s8, `m16n8k16` f16/bf16, `m16n8k8` tf32, `m8n8k4` Volta, `m16n8k64` block-scaled fp4) already covers a 2–4-bit integer dot after dequantization.

**Test coverage of the units.** The only quantized-type kernel tests in this tree are the three weight-quant tests registered at `src/tests/CMakeLists.txt:339-341`: `test-pq2-row-shapes`, `test-ptq1_0-cuda-dot`, `test-ptq1_0-element-map` — all PrismML *weight* types, and notably the only ones registered un-indented among their neighbours. There is no turbo KV-type test anywhere in the repository ([[build-and-verify]]), which matches the fact that there is no turbo GEMM kernel to test.

## Where it lives

- `src/ggml/src/ggml-cuda/mmf.cu` — MMF (`mul_mat_f_switch_rows_per_block`, `ggml_cuda_should_use_mmf`).
- `src/ggml/src/ggml-cuda/mmvf.cu` — MMVF (`mul_mat_vec_f_cuda`, `ggml_cuda_should_use_mmvf`).
- `src/ggml/src/ggml-cuda/mmvq.cu` — MMVQ kernels, `get_vec_dot_q_cuda`, `get_vdr_mmvq`, `mul_mat_vec_q_switch_type`, `ggml_cuda_should_use_mmvq`; `mmvq.cuh` — declarations and `MMVQ_MAX_BATCH_SIZE`.
- `src/ggml/src/ggml-cuda/mmq.cu` — host dispatch, type switch, `ggml_cuda_should_use_mmq`; `mmq.cuh` — tiles, `instr`/util-func tables, `DECL_MMQ_CASE`, `mmq_get_dp4a_tile_x_sizes`, `ggml_cuda_mmq_get_config`.
- `src/ggml/src/ggml-cuda/mmq-config-{pascal,ampere,blackwell,cdna,rdna2,rdna3,rdna3-5,rdna4}.cuh` — per-arch tile rows.
- `src/ggml/src/ggml-cuda/mma.cuh` — PTX fragment shapes.
- `src/ggml/src/ggml-cuda/mmq-hopper-q1.cu` — Hopper `wgmma` `Q1_0`/`PQ2_0` path, called from `ggml-cuda.cu`.
- `src/ggml/src/ggml-cuda/vecdotq.cuh` — the shared quantized dot products (no turbo).

## Known issues

- Turbo KV types exist in no unit here: `ggml_cuda_mul_mat` falls through them to cuBLAS — [[tq-1-missing-gemm-kernels]], [[gemm-dispatch]].
- Adding a type means editing ~6 independent tables per unit; nothing cross-checks them, so a type can be present in one and absent in the next (MMVQ has `IQ1_M`, MMQ's list does not). Below `ne11 = 8` the disagreement is an abort, above it a silent cuBLAS fallback — [[gemm-dispatch]].
- `PTQ1_0` is Turing-gated but has a complete `dp4a` MMQ implementation; on pre-Turing hardware (Pascal, and any Volta-less build) that path is unreachable, and the Pascal config omits its rows. Deliberate in effect, undocumented in intent.
- The Hopper `Q1_0` path is opt-in, `ne11 >= 128`, not bit-identical to standard MMQ, and its documented flag spellings (`GGML_CUDA_HOPPER_Q1`, `GGML_HOPPER_Q1_DISABLE`) do not match the `#if` actually used (`GGML_USE_HOPPER_Q1`).
- Kernel correctness for the PrismML weight types is covered only by three weight-only tests; the turbo KV types have no kernel test because they have no kernel — [[build-and-verify]], [[quantization]].

## See also

[[gemm-dispatch]] · [[prismml-weight-kernels]] · [[turboquant]] · [[tq-1-missing-gemm-kernels]] · [[v100-sxm2]] · [[quantization]] · [[build-and-verify]] · [[turbo-wht]] · [[kv-cache]] · [[performance-profile]]
