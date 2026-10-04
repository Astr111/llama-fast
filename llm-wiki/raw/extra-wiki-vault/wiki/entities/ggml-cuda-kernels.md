---
title: "GGML CUDA Kernels"
type: "entity"
tags: ["cuda", "gpu", "kernels", "ggml", "vecdotq", "gemv"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-prismml]]", "[[source-src-triattention]]", "[[source-src-turboquant]]"]
status: "active"
---

# GGML CUDA Kernels

The **GGML CUDA Kernels** subsystem in `llama-fast` contains hardware-accelerated device kernels implemented in NVIDIA CUDA (`/src/ggml/src/ggml-cuda/`) powering low-bit inference.

## Key Kernel Implementations

### 1. PrismML Dequantization & GEMV (`convert.cu`, `vecdotq.cuh`)
- `dequantize_row_ptq1_0_cuda`: Block-level asynchronous unpacking of base-3 packed trits into fp32/fp16 destination arrays.
- `vec_dot_ptq1_0_q8_1_multi`: Multi-column GEMV kernel executing inner products between ternary weights (`PTQ1_0`) and 8-bit activations (`Q8_1`).
- `vec_dot_pq2_0_q8_1`: High-speed 2-bit sub-block dot product kernel.

### 2. TriAttention Scoring Kernel (`triattention-score.cu`)
- Computes importance scores directly in VRAM.
- Grid: `(n_cells, n_offsets, 1)` with `Block: (freq_count, 1, 1)` (one thread per frequency pair).
- Includes cooperative Fast Walsh-Hadamard Transform (`cooperative_fwht_128`) in shared memory to invert [[turboquant-kv-cache]] rotations before scoring.

### 3. TurboQuant Walsh-Hadamard Kernel (`turbo-wht.cu`)
- Implements forward and inverse $O(d \log d)$ WHT rotations across batch dimensions using warp-synchronous butterfly exchanges.

## See Also
- Weight codec: [[prismml-weight-quantization]]
- KV cache compression: [[turboquant-kv-cache]]
- Scoring algorithm: [[triattention-scoring-eviction]]
- Runtime execution: [[cuda-graph-concurrency]]
