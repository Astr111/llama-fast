---
title: "TQ-1: Missing native GEMM kernels — turbo2/3/4 fall back to cuBLAS"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/ggml/src/ggml-cuda/ggml-cuda.cu, src/ggml/src/ggml-cuda/mmq.cu, src/ggml/src/ggml-cuda/mmvq.cu, src/ggml/src/ggml-cuda/mmvq.cuh, src/ggml/src/ggml-cuda/mmf.cu, src/ggml/src/ggml-cuda/convert.cu, src/ggml/src/ggml-cuda/quantize.cu, src/ggml/src/ggml-cuda/common.cuh, src/ggml/src/ggml-cuda/vecdotq.cuh, src/ggml/src/ggml.c]
tags: [turboquant, gemm-dispatch, quantization, performance]
---

## Symptom

Profiling shows a cuBLAS-family GEMM dominating GPU time — `magma_sgemmEx_kernel<float, __nv_bfloat16>` at 38.81 % (157 ms) against TriAttention's 0.91 % — while the custom quantized kernels sit idle. ([[source-state-md]], §1.3, §4 TQ-1)

## Cause

**Verified dispatch audit.** `ggml_cuda_should_use_mmq()` (`src/ggml/src/ggml-cuda/mmq.cu:366`) enumerates `PTQ1_0`, `Q1_0`, `Q2_0`, `PQ2_0`, `Q4_0/1`, `Q5_0/1`, `Q8_0`, the K-quants (`Q2_K`–`Q6_K`), the `IQ*` family, `MXFP4` and `NVFP4` (switch at line 373); `TURBO2_0`/`TURBO3_0`/`TURBO4_0` are absent, so the `default:` at line 407 sets `mmq_supported = false` (line 408) and the function returns `false` at lines 412-414. The MMQ claim in state.md holds exactly.

For MMVQ the state.md wording is true but misleading. `ggml_cuda_should_use_mmvq()` (`src/ggml/src/ggml-cuda/mmvq.cu:293`) has **no per-type list for Volta**: the per-type `switch`es exist only for Ada (line 300), Blackwell, DGX Spark and CDNA; every other NVIDIA path falls through to the closing `return ne11 <= MMVQ_MAX_BATCH_SIZE;` (with `MMVQ_MAX_BATCH_SIZE 8` at `src/ggml/src/ggml-cuda/mmvq.cuh:3`). Turbo types pass the initial `ggml_is_quantized()` gate because their type traits set `.is_quantized = true` (`src/ggml/src/ggml.c:708-731`; `ggml_is_quantized()` itself at `src/ggml/src/ggml.c:1399`). So on a V100 the function *returns true* for a turbo `src0` with a small batch — the type gate that actually rejects turbo is the **kernel** dispatch: `mul_mat_vec_q_switch_type()` (`mmvq.cu:1194`, switch at 1202) lists the same quantized types and ends in `GGML_ABORT("fatal error")` at lines 1353-1354. There is no `vec_dot_turbo*` entry in `get_vec_dot_q_cuda()` (`mmvq.cu:12-39`) nor anywhere in `vecdotq.cuh` (grep for `TURBO`/`turbo` finds none). Ergo a turbo `MUL_MAT` cannot be served by MMVQ; it would abort, not fall back. (For completeness, the `src1` quantization that path relies on is type-agnostic: `quantize_row_q8_1_cuda()` is dispatched from `src0->type` but ignores it — `GGML_UNUSED(type_src0)` at `src/ggml/src/ggml-cuda/quantize.cu:638-654`.)

**What `ggml_cuda_mul_mat()` actually does** (`src/ggml/src/ggml-cuda/ggml-cuda.cu:1819`). In order it tries MMVF (line 1853), the transposed-MMVF special case, `ggml_cuda_should_use_mmf()` (line 1860 — rejects everything quantized at `mmf.cu:135-137`), `ggml_cuda_should_use_mmvq()` (line 1864), the opt-in Hopper path (line 1868), `ggml_cuda_should_use_mmq()` (line 1874), and finally `ggml_cuda_mul_mat_cublas()` (line 1878). Every cuBLAS/MMQ/MMF gate rejects turbo; the only branch that accepts it is the last one. `ggml_cuda_mul_mat_cublas()` (`ggml-cuda.cu:1624`) sees `ggml_is_quantized(src0->type)` and switches the compute type to `GGML_TYPE_F16` because `fast_fp16_hardware_available()` is true for any NVIDIA `cc >= PASCAL && cc != 610` (`common.cuh:310`) — the V100 qualifies. It then calls `ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>()` (line 1411), whose `traits::convert()` is `ggml_get_to_fp16_cuda()` (F16 traits at `ggml-cuda.cu:1397-1411`, function at `convert.cu:605`). That switch *does* handle turbo: lines 664-669 return `dequantize_block_cont_cuda<QK_TURBO3, QR_TURBO3, dequantize_turbo3_0>` (and the turbo2/turbo4 siblings; helper at `convert.cu:340`). The entire `src0` tensor is materialized as fp16 on every call and handed to `cublasSgemm` / `cublasGemmEx` / `cublasGemmStridedBatchedEx` (`ggml-cuda.cu:1542-1575`).

So the mechanism the state.md fix sketch implies is confirmed: **there is no native turbo GEMM path**; the compressed cache is dequantized wholesale and multiplied by a generic GEMM.

> Correction (2026-09-28): two details of the state.md prose do not survive verification. (1) TURBO types are not rejected *by* `ggml_cuda_should_use_mmvq()` on this architecture — that function returns true; the rejection lives in `mul_mat_vec_q_switch_type()`'s abort. (2) MAGMA is not referenced anywhere in this checkout (grep for `magma`/`MAGMA` hits only `llmwiki/` and unrelated data files); the in-tree fallback is cuBLAS (`cublasGemmEx`), and the chosen compute type is F16, not bf16. The literal profiler symbol name and its bf16 template argument therefore cannot be explained from this tree [UNVERIFIED] and the causal link from the 157 ms to a turbo `MUL_MAT` is [INFERENCE] — the standard graph consumes turbo KV through the fused attention kernels and `GGML_OP_TURBO_WHT` rather than through `ggml_mul_mat`.

## Impact

> **Correction (2026-09-28) — the impact claim does not survive.** A full read of the placement path ([[device-placement]]) establishes that **no turbo-typed `MUL_MAT` is built during decode at all**: the turbo KV types force flash attention on (`src/src/llama-context.cpp:3882-3887` → `cparams.flash_attn`, `:320`), so `build_attn_mha` always takes the fused `ggml_flash_attn_ext` branch (`src/src/llama-graph.cpp:2673`, `:2699`) and the quantized blocks are dequantized **inside** `ggml_cuda_flash_attn_ext` (`fattn-vec.cuh:87-98`, `fattn.cu:338-369`), which CUDA accepts via `ggml_cuda_flash_attn_ext_supported` (`fattn.cu:647-649`, type list `:382-402`). `ggml_cuda_mul_mat` therefore **never sees a turbo tensor in the decode path**, and the chain "missing GEMM → cuBLAS/MAGMA fallback → 38.81 %" is not the mechanism. The types' absence from the dispatch tables is a **real fact with an unproven consequence**; the 38.81 % still needs an attribution, and `MAGMA` appears nowhere in this tree.

What remains true: the types are absent from `ggml_cuda_should_use_mmq`/`mmvq` and from `vecdotq.cuh` ([[quantized-kernel-units]]); if a turbo `MUL_MAT` *were* built — e.g. through the non-flash attention branch, which a capability mismatch can select — the CUDA allow-list (`ggml-cuda.cu:5244-5270`) would reject it and `ggml_backend_sched_split_graph` would push it to the **CPU** backend with a graph split and D2H copies (`ggml-backend.cpp:1399-1419`), i.e. the [[ta-3-cpu-fallback-transfers]] mechanism, not an abort. Severity as recorded in state.md: **CRITICAL**; as verified here: **unproven**.

## Location

- Path: `src/ggml/src/ggml-cuda/ggml-cuda.cu` (verified) — `ggml_cuda_mul_mat()` at line 1819; MMVF/MMF/MMVQ/MMQ gates at 1853/1860/1864/1874; cuBLAS fallback at 1878; `ggml_cuda_mul_mat_cublas()` at 1624 (compute-type choice 1626-1628); `ggml_cuda_mul_mat_cublas_impl` at 1411; F16 traits (`convert` = `ggml_get_to_fp16_cuda`) at 1397-1411; GEMM launches at 1542-1575.
- Path: `src/ggml/src/ggml-cuda/mmq.cu` (verified) — `ggml_cuda_should_use_mmq()` at 366; type switch at 373; `default: mmq_supported = false` at 407-408; early return at 412. No turbo case.
- Path: `src/ggml/src/ggml-cuda/mmvq.cu` (verified) — `ggml_cuda_should_use_mmvq()` at 293 (no turbo case, but no type list on Volta either); `get_vec_dot_q_cuda()` at 12-39 (no turbo); `mul_mat_vec_q_switch_type()` at 1194, switch 1202, `GGML_ABORT` 1353-1354 (no turbo).
- Path: `src/ggml/src/ggml-cuda/convert.cu` (verified) — `ggml_get_to_fp16_cuda()` at 605 with `GGML_TYPE_TURBO3_0`/`TURBO2_0`/`TURBO4_0` at 664-669 (also the fp32/bf16 twins at 735-741, 772-778, 836-842).
- Path: `src/ggml/src/ggml-cuda/mmf.cu` (verified) — `ggml_cuda_should_use_mmf()` rejects quantized types at 133-136.
- Symbols: no `vec_dot_turbo3_0` / `vec_dot_turbo2_0` / `vec_dot_turbo4_0` exists on the CUDA side; the only CUDA turbo dot products are the fused-attention helpers `vec_dot_fattn_vec_KQ_turbo{3,2,4}_0` (`src/ggml/src/ggml-cuda/fattn-common.cuh:333/387/436`).

## Status

Unresolved and unchanged in this checkout: no turbo case in MMQ, in the MMVQ vec-dot/kernel tables, or in `vecdotq.cuh`. The only working GEMM route for a turbo tensor is cuBLAS after a full fp16 dequantization. Listed as action item "Implement TurboQuant GEMM (TQ-1)" (§5.2 of [[source-state-md]]).

## Fix sketch

Per state.md §4 TQ-1 and §5.2: implement `vec_dot_turbo3_0()` (and the 2/4-bit siblings) in `vecdotq.cuh` over the existing `dequantize_turbo3_0()` blocks (`dequantize.cuh:163-176`), then wire it in at three concrete hook points — `get_vec_dot_q_cuda()` (`mmvq.cu:12`), `mul_mat_vec_q_switch_type()` (`mmvq.cu:1202`), and the `mmq_supported` switch (`mmq.cu:373`) — or, alternatively, write a dedicated dequant+GEMM kernel and dispatch it from `ggml_cuda_mul_mat()` before the `ggml_cuda_mul_mat_cublas()` fallback at `ggml-cuda.cu:1878`. Note that adding the case to `should_use_mmvq` alone is not enough (it already returns true on Volta); the kernel dispatch is the missing link.

## See also

- [[turboquant]] · [[gemm-dispatch]] · [[quantization]] · [[kv-cache]]
- [[performance-profile]] — the 38.8 % figure and the confound with eviction quality
- [[v100-sxm2]] — no INT tensor cores; why the fallback is catastrophic here
- [[tq-4-wht-numerical-mismatch]] · [[tq-5-tail-elements]] — the two other defects in the same encode/decode pair
- [[overview]] · [[roadmap]]
