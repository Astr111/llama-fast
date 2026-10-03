# Project State: llama-fast (Custom llama.cpp Inference Engine)

**Date:** 2026-09-28
**Hardware Target:** Tesla V100-SXM2-16GB (Volta sm_70, 900 GB/s HBM2, no INT Tensor Cores)
**OS / Stack:** Ubuntu 22.04 LTS, CUDA 12.4
**Models:**
*   **Target:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` (`head_dim=256`)
*   **Draft:** `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
**Key Optimizations:** Speculative Decoding (`draft-dflash`, max 5), TurboQuant KV Cache (`turbo3` Keys + `q8_0` Values), TriAttention KV Eviction (budget 4096, window 512), CUDA Graphs.
**Benchmarks:** Terminal-Bench 2.0 (MR Subset 39), Harbor 0.1.x, Nsight Systems.

---

## 1. Executive Summary & Resolved Issues

### Performance Profiling Conclusions
1.  **CUDA Graph Reuse:** Does *not* accumulate speed improvements over time. Launch overhead reduction is a constant one-time saving (~25µs vs 16-64ms). Observed decode degradation (15.75ms -> 19.43ms) correlates with unbounded KV attention costs as context grows, not graph reuse count.
2.  **Decode Speed Growth (40 -> 69 tok/s):** Traced to speculative decoding (`draft-dflash`). As the agent generates repetitive CLI/code patterns, context warms up and acceptance rates grow, shifting effective throughput from ~31.6 t/s to ~38.1+ t/s.
3.  **Nsight Profiling Anomaly:** TriAttention consumes only 0.91% of GPU time, but `magma_sgemmEx_kernel<float, __nv_bfloat16>` consumes 38.81% (157ms). This is caused by TurboQuant types lacking native GEMM kernels (see Issue TQ-1).

---

## 2. Active Codebases

*   **Publication Repository (Source of Truth):** `/home/ms/Загрузки/llama-fast/`
*   **Active Working Copy (Broken WHT):** `/home/ms/llama-fast/llama.cpp/`
*   **Fixed Release Copy:** `/home/ms/llama-fast/Release/`
*   **TurboQuant Fork:** `/home/ms/llama-cpp-turboquant/`

---

## 3. Known Issues: TriAttention

### TA-1. [CRITICAL] WHT Inversion Bug for `head_dim=256`
*   **File:** `ggml/src/ggml-cuda/triattention-score.cu` (publication repository)
*   **Description:** The GPU scoring kernel previously skipped Walsh-Hadamard Transform (WHT) inversion for models with `head_dim=256` due to `if (padded_hd == 128 && f < 64)`.
*   **Impact:** KV cache was scored using non-inverted keys, resulting in random eviction and severe generation quality degradation.
*   **Status:** **RESOLVED (2026-10-04)**. Ported dynamic `wht_group` calculation and `inverse_wht_rotation_128` per 128-element group from `/home/ms/llama-fast-dev/Release/` to `src/ggml/src/ggml-cuda/triattention-score.cu` (Commit `5a52561`).

### TA-2. [HIGH] Budget Starvation on Long Prefixes
*   **File:** `src/src/llama-triattention.cpp` (`triattention_prune_impl`)
*   **Description:** The formula `decode_budget = budget - n_protected` dropped to zero when `prefix_length + divide_length >= budget`.
*   **Impact:** TriAttention previously degenerated into a pure sliding window on long system prompts.
*   **Status:** **RESOLVED (2026-10-04)**. Implemented dynamic `min_history_budget` in `triattention_prune_impl` guaranteeing a floor of historical tokens during pruning (Commit `f858af6`).

### TA-3. [HIGH] CPU Fallback Synchronous D2H Transfers
*   **File:** `src/llama-triattention.cpp` (`triattention_dequant_kv_head`)
*   **Description:** If GPU initialization fails, the CPU fallback path calls `ggml_backend_tensor_get()` (synchronous `cudaMemcpy`) for *every single KV cell individually* inside a nested loop.
*   **Impact:** For `n_decode=4096` and `head_dim=256`, this triggers ~4096 separate 1KB D2H transfers, causing 15–30 second pipeline stalls instead of milliseconds.
*   **Status:** Unresolved. Requires batched transfers or strict GPU-only enforcement.

### TA-4. [MEDIUM] Race Condition in `cooperative_fwht_128`
*   **File:** `ggml/src/ggml-cuda/triattention-score.cu`
*   **Description:** Shared memory WHT rotation assumed all 64 warp threads are active. If `active=true` but `tid >= 64`, warp divergence or shared memory OOB access could occur on Volta sm_70.
*   **Impact:** Undefined behavior on Volta (sm_70); potential score corruption.
*   **Status:** **RESOLVED (2026-10-04)**. Ported `bool active` parameter and guarded shared memory loads, stores, and butterfly updates in `cooperative_fwht_128` and `inverse_wht_rotation_128` (Commit `5a52561`).

### TA-5. [MEDIUM] Dead Code: `freq_scale_sq` Always 1.0
*   **File:** `src/llama-triattention.cpp` (calibration)
*   **Description:** Computed via `cosf(omega[f] * 0.0f)` and `sinf(omega[f] * 0.0f)`, which always yields `c=1.0, s=0.0` -> `freq_scale_sq = 1.0`. Trigonometric scaling is effectively disabled.
*   **Impact:** Loss of scoring precision; implementation diverges from the paper's formulas.
*   **Status:** Unresolved. Needs verification if intentional simplification or init bug.

### TA-6. [LOW] Overlap Double-Counting Risk
*   **File:** `src/llama-triattention.cpp`
*   **Description:** While `if (is_prefix || is_recent)` correctly counts each token once, future patches calculating `max_protected` must account for overlap between `recent_threshold` and `prefix_length` to avoid overestimating protected cells.
*   **Status:** Documented for future patching.

### TA-7. [LOW] Missing Configuration Validation
*   **File:** `src/llama-triattention.cpp` (`triattention_init`)
*   **Description:** No check for physical incompatibility between `budget`, `prefix_length`, and `divide_length`. Users get no warning if their prompt length guarantees immediate history eviction.
*   **Status:** Unresolved. Needs warning/error emission.

### TA-8. [CRITICAL] `offset_max = 0` Causing NaN Eviction Scores
*   **Files:** `src/common/common.h`, `src/src/llama-triattention.cpp`, `src/ggml/src/ggml-cuda/triattention-score.cu`
*   **Description:** `offset_max` defaulted to 0, producing `n_offsets = 0`. Division by zero in CPU/GPU mean aggregation yielded `NaN` for every key's eviction score, violating strict weak ordering in `std::partial_sort` (UB) and making eviction pseudo-random.
*   **Impact:** Complete corruption of KV cache eviction ordering in default launch configurations.
*   **Status:** **RESOLVED (2026-10-04)**. Set default `triattention_offset_max = 65536` in `common.h` (generating 17 geometric offsets), added initialization guard in `triattention_init` (warn + fallback to norm scoring if `n_offsets == 0`), and added arithmetic fail-safes in CPU/GPU scoring kernels (Commit `585fe5d`).

### TA-9. [HIGH] RoPE Scope and Frequency Mismatch in Scorer
*   **Files:** `src/src/llama-kv-cache.cpp`, `src/src/llama-triattention.h`, `src/src/llama-triattention.cpp`, `src/ggml/include/ggml-cuda.h`, `src/ggml/src/ggml-cuda/triattention-score.cu`
*   **Description:** The TriAttention scorer previously inverted RoPE over all 256 dimensions with exponent $\theta^{-2f/256}$, whereas the model rotates only $n_{rot} = 64$ dimensions with exponent $\theta^{-2f/64}$, causing growing angular distortion $\theta^{3f/128}$ and erroneous rotation of unrotated dimensions.
*   **Impact:** Inverted keys differed drastically from true pre-RoPE keys, corrupting trigonometric importance evaluation.
*   **Status:** **RESOLVED (2026-10-04)**. Dynamic `n_rot` passed from `hparams.n_rot(0)` to `triattention_init`, `omega` computed via `n_rot` with 0.0f padding, and inverse RoPE selectively applied only to channels $f < n_{rot}/2$ in both CPU (`triattention_invert_rope`) and GPU (`triattention_score_kernel`).

### TA-10. [HIGH] `prefix_length` Stale Per-Context Latch
*   **Files:** `src/src/llama-kv-cache.cpp`
*   **Description:** `prefix_length` was latched once on the very first prompt batch containing position 0 and never reset when a sequence slot was recycled via `seq_rm(id, -1, -1)`. Subsequent longer prompts had the middle of their prompt incorrectly treated as evictable.
*   **Impact:** New prompt tokens were evicted during subsequent requests in persistent server environments.
*   **Status:** **RESOLVED (2026-10-04)**. Reset `prefix_length = 0` on full sequence clear in `seq_rm` and updated `prefix_length` whenever a prompt batch contains position 0 (Commit `f858af6`).

---

## 4. Known Issues: TurboQuant

### TQ-1. [CRITICAL -> RE-SCOPED] Missing Native GEMM Kernels -> cuBLAS/MAGMA Fallback
*   **Files:** `ggml-cuda/mmq.cu`, `ggml-cuda/mmvq.cu`, `ggml-cuda/ggml-cuda.cu`
*   **Description:** `GGML_TYPE_TURBO2_0`, `TURBO3_0`, and `TURBO4_0` are not listed in `ggml_cuda_should_use_mmq()` or `ggml_cuda_should_use_mmvq()`. However, TurboQuant KV cache forces Flash Attention (`params.flash_attn_type = ENABLED`), routing execution to `ggml_cuda_flash_attn_ext` where dequantization is performed fused inside the attention kernel (`fattn-vec.cuh`). No turbo `MUL_MAT` is constructed on the decode path.
*   **Impact:** The absence in MMQ/MMVQ is dead code on the decode hot path; the 38.8% MAGMA figure was a profiling misattribution (MAGMA does not exist in the codebase).
*   **Status:** **RE-SCOPED / NOT A BOTTLENECK**. No custom GEMM kernels needed for attention decode.

### TQ-2. [HIGH] InnerQ Host State Thread Safety Violation
*   **File:** `ggml-cuda/turbo-quant.cuh` (lines 154-157)
*   **Description:** Variables `innerq_enabled`, `innerq_target_tokens`, `innerq_strength`, `innerq_initialized` are declared as `static` (file-scope) in a header included by multiple translation units (`set-rows.cu`, `dequantize.cuh`, etc.). Each TU gets its own independent copy.
*   **Impact:** Calibration triggered in one TU may never be visible to `turbo_innerq_check_finalize()` in another TU. InnerQ may silently fail to activate depending on compilation/link order.
*   **Status:** Unresolved. Must move host-side state to a dedicated `.cu` file with `extern` declarations.

### TQ-3. [HIGH] InnerQ Device State Multi-GPU Unsafe
*   **File:** `ggml-cuda/turbo-quant.cuh` (lines 147-152)
*   **Description:** Device variables (`d_innerq_scale`, `d_innerq_sq_accum`, etc.) are `static __device__` in a header. `cudaMemcpyToSymbol` only targets the *current* device.
*   **Impact:** In multi-GPU (tensor parallelism) setups, scales upload to only one GPU. Others use zero/stale values, corrupting quantization.
*   **Status:** Unresolved. Requires explicit per-device init loop or binding to `ggml_backend_cuda_context`.

### TQ-4. [HIGH] Sequential vs Parallel WHT Numerical Mismatch
*   **File:** `ggml-cuda/turbo-quant.cuh` (lines 88-106) vs `ggml-cuda/set-rows.cu` (lines 332-343)
*   **Description:** `turbo_fwht_128()` in the header is a sequential single-thread loop. `set-rows.cu` uses a parallel shared-memory butterfly with `__syncthreads()`. Floating-point operation ordering differs.
*   **Impact:** If `turbo_rotate_forward()` is ever called outside `set-rows.cu` (e.g., in a future dequant kernel), results will numerically mismatch the encoding process, causing silent accuracy loss.
*   **Status:** Unresolved. Needs unified WHT implementation or strict usage boundaries.

### TQ-5. [MEDIUM] Tail Elements: No Rotation, No InnerQ
*   **File:** `ggml-cuda/set-rows.cu` (kernel `k_set_rows_turbo3_tail`)
*   **Description:** When `head_dim % GROUP_SIZE != 0`, tail elements are quantized without WHT rotation or InnerQ scaling. Dequantization in `dequantize.cuh` has no special handling for tails.
*   **Impact:** Models with non-aligned `head_dim` (e.g., GLM 576) suffer systematic quantization errors in tail channels. InnerQ calibration ignores these channels.
*   **Status:** Unresolved. Requires padding or partial-group WHT.

### TQ-6. [MEDIUM] InnerQ Calibration Race Condition
*   **File:** `ggml-cuda/set-rows.cu` (lines 288-290)
*   **Description:** `if (j == 0) atomicAdd(&d_innerq_count, 1);` relies on `j` mapping exactly to one thread per block. While true for `GROUP_SIZE=128` with 128 threads, changes to block dimensions could cause overcount/undercount.
*   **Impact:** Incorrect RMS estimates -> wrong InnerQ scales.
*   **Status:** Unresolved. Should use `threadIdx.x == 0` explicitly.

### TQ-7. [MEDIUM] `INNERQ_MAX_CHANNELS = 128` Hard Limit
*   **File:** `ggml-cuda/turbo-innerq.cuh`
*   **Description:** Maximum channel limit is hardcoded to 128.
*   **Impact:** Models with `head_dim=256` (like Ternary-Bonsai-2-27B) cannot fully utilize InnerQ equalization across all channels.
*   **Status:** Unresolved.

---

## 5. Pending Action Items

1.  **[DONE] Port WHT Fix (TA-1 & TA-4):** Ported dynamic `wht_group` and `active` guard in `cooperative_fwht_128` from `/home/ms/llama-fast-dev/Release/...` to publication repo (Commit `5a52561`).
2.  **[DONE] Fix offset_max NaN Bug (TA-8):** Set default `triattention_offset_max = 65536`, added init validation guard, and kernel fail-safes (Commit `585fe5d`).
3.  **[DONE] Resolve Prefix Stale Latch (TA-10):** Dynamic update on prompt pos 0 and clear on `seq_rm` (Commit `f858af6`).
4.  **[DONE] Apply TriAttention Budget Scaling (TA-2):** Implemented dynamic `min_history_budget` logic in `triattention_prune_impl` to prevent context starvation (Commit `f858af6`).
5.  **[CLOSED / RE-SCOPED] Implement TurboQuant GEMM (TQ-1):** Not required; turbo KV cache is consumed by fused flash attention kernels (`ggml_cuda_flash_attn_ext`), not `mul_mat`.
6.  **Refactor InnerQ State (TQ-2, TQ-3):** Move `static` host/device variables out of `turbo-quant.cuh` into proper TU-scoped or context-scoped storage to ensure thread safety and multi-GPU support.
7.  **Batch CPU Fallback Transfers (TA-3):** Refactor `triattention_dequant_kv_head` to use bulk `cudaMemcpy` instead of per-cell synchronous transfers.
8.  **Fix RoPE Phase Inversion Scope (TA-9):** Parameterize scorer RoPE inverse by actual model rotation dimension (`n_rot = 64`) rather than assuming full 256 dimensions.
