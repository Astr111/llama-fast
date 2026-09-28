---
title: GEMM dispatch
type: concept
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/ggml/src/ggml-cuda/ggml-cuda.cu, src/ggml/src/ggml-cuda/mmq.cu, src/ggml/src/ggml-cuda/mmq.cuh, src/ggml/src/ggml-cuda/mmvq.cu, src/ggml/src/ggml-cuda/mmvq.cuh, src/ggml/src/ggml-cuda/mmvf.cu, src/ggml/src/ggml-cuda/mmf.cu, src/ggml/src/ggml-cuda/vecdotq.cuh, src/ggml/src/ggml-cuda/convert.cu, src/ggml/src/ggml-cuda/fattn.cu, src/ggml/src/ggml-cuda/fattn-common.cuh, src/ggml/src/ggml-cuda/common.cuh, src/ggml/src/ggml.c, src/ggml/CMakeLists.txt, src/src/llama-graph.cpp]
tags: [gemm, cuda, dispatch, quantization]
---

## Definition

The runtime decision, made per `GGML_OP_MUL_MAT` node, about *which* CUDA kernel computes the matrix product. ggml does not ask the operands what they want; it asks a fixed chain of type predicates, in order, and takes the first that says yes. Everything the chain does not recognise lands on the generic fused-multiply BLAS path.

The chain lives in `ggml_cuda_mul_mat` (`src/ggml/src/ggml-cuda/ggml-cuda.cu:1819`) and is, in order:

| Step | Predicate | Kernel | Line |
| :--- | :--- | :--- | :--- |
| 0 | argument types wrong, or view with dirty padding | `ggml_cuda_mul_mat_cublas` | `ggml-cuda.cu:1833` |
| 1 | `should_use_mmvf` (F32/F16/BF16 only, defined at `mmvf.cu:786`) | `mul_mat_vec_f` | `ggml-cuda.cu:1840` |
| 2 | `should_use_mmf` (dequantised types only; returns `false` for any quantised type, `mmf.cu:134`) | `mul_mat_f` | `ggml-cuda.cu:1860` |
| 3 | `should_use_mmvq` (`mmvq.cu:293`) | `mul_mat_vec_q` | `ggml-cuda.cu:1864` |
| 4 | `should_use_mmq` (`mmq.cu:366`) | `mul_mat_q` | `ggml-cuda.cu:1874` |
| 5 | nothing matched | `ggml_cuda_mul_mat_cublas` | `ggml-cuda.cu:1878` |

`ggml_cuda_mul_mat_id` (`ggml-cuda.cu:1912`) mirrors the same ordering for the MoE path.

The two quantised predicates are **allow-lists**, not capability queries:

- `ggml_cuda_should_use_mmq` switches on `type` (`mmq.cu:373`), sets `mmq_supported = true` for the listed types (`mmq.cu:405`), `false` for `default` (`mmq.cu:408`), then adds arch gates — the `turing_mma_available` fast-path (`mmq.cu:438-439`), a 48 KiB shared-memory floor (`mmq.cu:417-422`), dp4a batch limits.
- `ggml_cuda_should_use_mmvq` first rejects anything not `ggml_is_quantized` (`mmvq.cu:294`), then on non-Ada/Blackwell/CDNA NVIDIA falls straight through to `return ne11 <= MMVQ_MAX_BATCH_SIZE` (`MMVQ_MAX_BATCH_SIZE == 8`, `mmvq.cuh:3`) at `mmvq.cu:381`. This one is a *deny-if-not-quantised* gate, so it admits a quantised type that no kernel implements — which matters: for a turbo `src0` on Volta with `ne11 <= 8` the predicate says yes, and the very next hop, `mul_mat_vec_q_switch_type`, has no turbo case and hits `GGML_ABORT("fatal error")` (`mmvq.cu:1354`). The cuBLAS fallback is therefore reachable for a quantised-but-unimplemented type only at `ne11 > 8`; below that the predicate's answer and the kernel table disagree.

Choosing a kernel is necessary but not sufficient: the kernel has to exist for that type too. `mul_mat_vec_q_switch_type` (`mmvq.cu:1194`) and `ggml_cuda_mul_mat_q_switch_type` (`mmq.cu:9`) are independent `switch`es over the same allow-lists, both ending in `GGML_ABORT("fatal error")` (`mmvq.cu:1354`, `mmq.cu:89`), and the device-side `vec_dot` table `get_vec_dot_q_cuda` (`mmvq.cu:11`) returns `nullptr` at its `default` (`mmvq.cu:38`).

## Why it matters here

TurboQuant's KV types are absent from both allow-lists, and this is the mechanism behind the project's dominant cost.

- `GGML_TYPE_TURBO2_0`, `TURBO3_0`, `TURBO4_0` appear nowhere in the `mmq.cu:373` allow-list, nowhere in `mmvq.cu`'s `vec_dot` table (`mmvq.cu:11-38`), and nowhere in `vecdotq.cuh` (zero occurrences of `TURBO` in that file). `ggml_is_quantized` does return `true` for them (`src/ggml/src/ggml.c:1399`, with `.is_quantized = true` at `ggml.c:708/716/724`), so step 3's gate passes a type it cannot actually compile a kernel for; the type is excluded only upstream by the MMQ allow-list and downstream by the switch tables.
- Consequence of a miss: step 5, `ggml_cuda_mul_mat_cublas` (`ggml-cuda.cu:1624`). A quantised `src0` is given `compute_type = GGML_TYPE_F16` (`ggml-cuda.cu:1626-1627`), the tensor is **dequantised into a temporary F16 buffer** — `traits::convert` / `convert_func(src0->data, src0_alloc.get(), …)` at `ggml-cuda.cu:1455-1457`, with the converter table in `src/ggml/src/ggml-cuda/convert.cu:605` — the GEMM runs in cuBLAS (`cublasSgemm` `ggml-cuda.cu:1547`, `cublasGemmEx` `:1554`, `cublasGemmStridedBatchedEx` `:1569`, `cublasGemmBatchedEx` `:1607`), and the result is converted back to F32 (`ggml-cuda.cu:1618-1620`). That is the "dequantize-then-GEMM" pattern: full-precision materialisation of a deliberately compressed tensor, plus an extra round-trip allocation.

> Contradiction (2026-09-28): [[source-state-md]] §1.3/§4 TQ-1 attributes 38.81 % (157 ms) of GPU time to a `magma_sgemmEx_kernel<float, __nv_bfloat16>`, and calls the fallback "cuBLAS/MAGMA". A case-insensitive search for `magma` across `src/` returns **no match of any kind** in this tree, and CMake exposes only `GGML_CUDA_FORCE_CUBLAS` (`src/ggml/CMakeLists.txt:205`), not a MAGMA vendor option. The fallback that exists here is cuBLAS. Whether the profiled symbol came from a different checkout or a differently linked BLAS is `[UNVERIFIED]`; the *symbol name* is not reproducible from this repository.

Two further gates decide whether the gap is even reachable, and they are the reason [[tq-1-missing-gemm-kernels]] and this page must be read together:

- `ggml_backend_cuda_device_supports_op` (`ggml-cuda.cu:5155`) lists, for `GGML_OP_MUL_MAT` / `MUL_MAT_ID`, only F16/F32, the classic quants, k-quants, IQ-quants, MXFP4/NVFP4 and BF16 (`ggml-cuda.cu:5240-5275`) — **no turbo type**. Turbo appears in that function only for `GGML_OP_SET_ROWS` (`ggml-cuda.cu:5327`), and separately for `GGML_OP_TURBO_WHT` (`ggml-cuda.cu:5493`, `:2092`). A turbo-typed operand in a plain `MUL_MAT` is therefore not claimed by the CUDA backend at all, and the scheduler is free to place that node on another backend instead of reaching the cuBLAS tail. That path — turbo `src0` → dequantise → cuBLAS — is consequently `[UNVERIFIED]` as the actual source of the measured time, even though the predicate gap is verified.
- The fused attention kernels *do* carry native turbo dots: `vec_dot_fattn_vec_KQ_turbo{3,2,4}_0` (`src/ggml/src/ggml-cuda/fattn-common.cuh:968-974`) behind `ggml_cuda_fattn_kv_type_supported` (`fattn.cu:382-400`) and the mixed turbo/q8_0 case list (`fattn.cu:493-507`). Attention over a turbo KV cache that goes through `ggml_flash_attn_ext` (`src/src/llama-graph.cpp:2690`) never consults the dispatch chain above. The chain is reached only by the non-flash branch, which builds `ggml_mul_mat(ctx0, k, q)` directly on the cache tensors (`src/src/llama-graph.cpp:2730`).

So the accurate summary is: dispatch is a closed allow-list, turbo types are on it nowhere, and *where* the resulting miss lands depends on which attention path the graph took.

## Tradeoffs

- **Allow-lists vs capability queries.** A `switch` is compile-time dispatch with no vtable and no host branching per call — cheap and debuggable. The price is that a type a backend *can partially* handle (fused attention, `SET_ROWS`, `TURBO_WHT`) is indistinguishable, to the dispatcher, from one it cannot handle at all.
- **Fused quantised dot vs generic BLAS.** MMQ/MMVQ dequantise inside the inner loop and multiply against a Q8_1-quantised activation; the BLAS tail dequantises the whole operand once and pays for the temporary in HBM bandwidth. For a KV cache read once per token at growing context, the second is the worse shape — but it is also the only universal fallback, which is why every unrecognised type converges on it.
- **Arch gates are part of dispatch.** On Volta the interesting gates are the negative ones: `turing_mma_available` is false (`common.cuh:348`) and `ampere_mma_available` is false (`common.cuh:356`), so MMQ's fast path and BF16 MMA are both out; `fp16_mma_hardware_available` is true (`common.cuh:316-320`). A missing kernel cannot be compensated by an integer MMA unit, because the hardware does not have one ([[v100-sxm2]]).
- **Cost is invisible until profiled.** The dispatch chain returns a function pointer; a miss is not an error, not a warning and not a slow path in the source — it is a different library. Nothing in the build or the model loader says "this type has no GEMM kernel".

## Open questions

- Which checkpoint's profiler run produced `magma_sgemmEx_kernel`, and against which BLAS linkage? Not reproducible from this tree.
- Is the turbo-typed `MUL_MAT` node in the non-flash attention branch actually placed on CUDA, or does `supports_op` push it to CPU? That determines whether the cuBLAS-dequant path is hot at all.
- `ggml_cuda_should_use_mmvq`'s fall-through admits any quantised type; whether adding turbo to the MMQ allow-list alone would suffice without a `vec_dot_turbo*` in `vecdotq.cuh` is untested here.

## See also

[[tq-1-missing-gemm-kernels]] · [[quantization]] · [[turboquant]] · [[kv-cache]] · [[v100-sxm2]] · [[performance-profile]] · [[codebase-map]]
