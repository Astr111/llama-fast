---
title: Device placement of TurboQuant nodes
type: topic
status: current
updated: 2026-09-28
sources: []
verified: [src/ggml/src/ggml-backend.cpp, src/ggml/src/ggml.c, src/ggml/src/ggml-cuda/ggml-cuda.cu, src/ggml/src/ggml-cuda/fattn.cu, src/ggml/src/ggml-cuda/fattn-vec.cuh, src/ggml/src/ggml-cuda/fattn-common.cuh, src/ggml/src/ggml-cuda/set-rows.cu, src/ggml/src/ggml-cuda/convert.cu, src/ggml/src/ggml-cuda/triattention-score.cu, src/src/llama-graph.cpp, src/src/llama-context.cpp, src/src/llama-kv-cache.cpp, src/src/llama-triattention.cpp]
tags: [quantization, kv-cache, gemm-dispatch]
---

# Device placement — does a TurboQuant compute node ever run on the GPU?

## Bottom line

**The turbo KV cache is never consumed by a `MUL_MAT` node: it is read directly by the fused flash-attention kernels, and those kernels do run on the GPU (verdict c).** Turbo cache types are *forced* to use flash attention — `src/src/llama-context.cpp:3882-3887` warns "turbo cache types require flash_attn — enabling automatically" and sets `flash_attn_type = ENABLED`, which flows into `cparams.flash_attn` (`src/src/llama-context.cpp:320`). With flash attention on (and no KQ bias), `build_attn_mha` takes the fused branch `ggml_flash_attn_ext(ctx0, q, k, v, ...)` (`src/src/llama-graph.cpp:2673`, `:2699`) — there is no `ggml_mul_mat(k, q)` / `ggml_mul_mat(v, kq)` anywhere on the turbo cache. The quantised K/V blocks are dequantised inside the attention kernel (`src/ggml/src/ggml-cuda/fattn-vec.cuh:87-98` — `vec_dot_KQ_turbo3_0`/`dequantize_V_turbo*`; dispatch tables in `src/ggml/src/ggml-cuda/fattn.cu:338-369` list every turbo×turbo and turbo×q8_0 combination).

**What this does to the TQ-1 story:** the causal chain in [[tq-1-missing-gemm-kernels]] — "turbo KV types have no native matmul kernels, so `ggml_cuda_mul_mat` falls through to cuBLAS/MAGMA and burns 38.81 % of GPU time" — **does not survive** for the attention path. `ggml_cuda_mul_mat` never sees a turbo tensor in decode in this checkout; the "missing GEMM" omission in the allow-list (`src/ggml/src/ggml-cuda/ggml-cuda.cu:5244-5270` — every `GGML_TYPE_TURBO*` absent, `default: return false`) is real but **dead on the hot path**. The 38.81 % attribution must come from something else (see Open questions). What survives is the narrower claim that the *cache-type predicate* rejects turbo `MUL_MAT` on CUDA, and the KV-write side (`set_rows`, `turbo_wht`) which does have CUDA kernels.

## Evidence

- **The force-enable gate.** `src/src/llama-context.cpp:3882-3887`: if flash attention is disabled and `type_k`/`type_v` is `TURBO2_0/3_0/4_0`, it is enabled automatically; `src/src/llama-context.cpp:320`: `cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED`. So the turbo cache cannot reach the non-FA branch — except when a model has KQ bias or is Grok (see Open questions).
- **The graph.** `src/src/llama-graph.cpp:2673` — `use_flash_attn = cparams.flash_attn && kq_b == nullptr`. Fused branch at `:2699` (`ggml_flash_attn_ext`); the only turbo node after it is the inverse WHT `ggml_turbo_wht(cur, 1, ...)` on the F32 output (`:2700-2708`). The non-FA branch (`:2777-2787` region) contains the `mul_mat(k,q)`/`mul_mat(v,kq)` nodes — but it is unreachable for a turbo cache.
- **Placement.** `ggml_backend_sched_split_graph` (`src/ggml/src/ggml-backend.cpp:1055`) is the function that assigns backends. Pass 3 (`:1210-1211`) assigns unassigned nodes to the backend with the most supported inputs among those for which `ggml_backend_supports_op` holds; pass 4 (`:1280` loop) assigns leftovers to the first backend that supports the op (`ggml_backend_sched_set_if_supported`, `:1047-1051`) and then `GGML_ASSERT(*cur_backend_id != -1)` — abort if *no* registered backend supports the node. `GGML_OP_FLASH_ATTN_EXT` is accepted by CUDA: `src/ggml/src/ggml-cuda/ggml-cuda.cu:5604-5605` → `ggml_cuda_flash_attn_ext_supported` (`src/ggml/src/ggml-cuda/fattn.cu:647-649` → `ggml_cuda_get_best_fattn_kernel`), which accepts `TURBO2_0/3_0/4_0` for K and V (`src/ggml/src/ggml-cuda/fattn.cu:382-402`, mixed-type check `:493-497`, turbo head-dim guard `:502-505`). So the fused node stays on the GPU and is executed by `ggml_cuda_flash_attn_ext` (`src/ggml/src/ggml-cuda/ggml-cuda.cu:2369-2371`).
- **The rejection mechanism, for completeness.** If a turbo `MUL_MAT` ever *were* built, the CUDA predicate (`src/ggml/src/ggml-cuda/ggml-cuda.cu:5244-5270`, `default: return false`) rejects it; the scheduler would then hand the node to the CPU backend (which supports any `MUL_MAT`), splitting the graph at that boundary and copying the split inputs D2H (`src/ggml/src/ggml-backend.cpp:1399-1419`, pass 5) — the exact [[ta-3-cpu-fallback-transfers]] mechanism. It is not the abort path: `GGML_ASSERT` fires only if no backend at all supports the op.
- **`turbo_wht`'s own predicate** is shape-based, not type-named: `src/ggml/src/ggml-cuda/ggml-cuda.cu:5493-5495` — `src[0]` F32, dst F32, `src[0]->ne[0] % 64 == 0`. It never mentions `GGML_TYPE_TURBO*`; any F32 tensor whose head dim is a multiple of 64 passes. Note this governs the inverse-WHT on the attention *output*, not the cache itself.
- **KV write side stays on GPU**: `set_rows` dispatch has turbo cases (`src/ggml/src/ggml-cuda/set-rows.cu:1241-1246`); TriAttention scoring kernels read turbo K blocks directly (`src/ggml/src/ggml-cuda/triattention-score.cu:118-141`, `:365-383`).

## Open questions

- What actually produced the recorded 38.81 % ([[performance-profile]])? The profile predates this checkout's force-enable, was measured with `--flash-attn off` (impossible with turbo types since the force-enable at `llama-context.cpp:3882-3887`), or attributes GPU time to an op that is not the attention matmul. None of these have been read out of the repo — the number itself lives only in prose.
- Fallback edge: models with KQ bias (`kq_b != nullptr`) or Grok (`llama-context.cpp:3860-3863` disables FA, though the later force-enable at `:3882` may re-enable it) could in principle reach the non-FA branch with a turbo cache, pushing the turbo `MUL_MAT` to CPU with D2H split copies — mechanism (b). Whether any configured model in this repo hits it is unverified.
- Does the fused FA vec/tile kernel with turbo types exist for the `head_dim=256` (WHT-rotated) shape on `sm_70`? `fattn.cu:338-369` lists 256-wide cases, but [[v100-sxm2]]-specific dispatch was not checked.

## See also

[[gemm-dispatch]] · [[tq-1-missing-gemm-kernels]] · [[quantized-kernel-units]] · [[turboquant]] · [[request-lifecycle]] · [[v100-sxm2]] · [[performance-profile]] · [[ta-3-cpu-fallback-transfers]]
