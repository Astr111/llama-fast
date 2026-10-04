---
title: "PrismML Weight Quantization (PTQ1_0 & PQ2_0)"
type: "concept"
tags: ["quantization", "prismml", "ternary", "weights", "cuda-kernels", "inference"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-prismml]]"]
status: "active"
---

# PrismML Weight Quantization (PTQ1_0 & PQ2_0)

**PrismML Weight Quantization** encompasses the proprietary low-bit weight representation formats and execution kernels developed for the [[bonsai-model]] family in llama.cpp, targeting sub-2-bit and ternary model weights.

## 1. PTQ1_0: Lossless Ternary Encoding (1.75 bpw)
Designed for ternary neural network checkpoints where weights are constrained to $\{-1, 0, +1\}$ (such as BitNet b1.58 architectures).

### Base-3 Trit Packing Mechanism
Standard binary packing requires 2 bits per ternary element (wasting $25\%$ of representation space, since $2^2 = 4 > 3$). PrismML achieves optimal theoretical compression using **base-3 radix packing**:
- $3^5 = 243 \le 256 \implies$ Exactly **5 trits per 8-bit byte**!
- In a block of $QK = 128$ elements:
  - 24 bytes in `qs` store $24 \times 5 = 120$ trits.
  - 2 bytes in `qh` store $2 \times 4 = 8$ trits ($3^4 = 81 \le 256$).
  - 2 bytes store fp16 block delta scale `d`.
  - **Total block size**: $24 + 2 + 2 = 28$ bytes for 128 weights.
  - **Effective Bitrate**: $\frac{28 \times 8}{128} = 1.75$ bits per weight (bpw).

### Device Decoding Implementation
CUDA and Vulkan shaders unpack trits using modular multiplication:
```cuda
static __device__ __forceinline__ int ptq1_0_trit(const block_ptq1_0 * x, const int e) {
    uint8_t b; int n;
    if (e < 80)       { b = x->qs[e & 15];              n = e >> 4; }
    else if (e < 120) { const int t = e - 80; b = x->qs[16 + (t & 7)]; n = t >> 3; }
    else              { const int t = e - 120; b = x->qh[t & 1];       n = t >> 1; }
    uint32_t v = b;
    for (int i = 0; i < 4; ++i) if (i < n) v = (v * 3) & 0xFF;
    return (int)((v * 3) >> 8) - 1; // Evaluates to -1, 0, or +1
}
```

## 2. PQ2_0: Sub-2-Bit Format (2.125 bpw)
- Group size: 128 elements.
- Structure: `uint8_t qs[32]` (2 bits/element) + `ggml_half d` (fp16 scale).
- Total block size: 34 bytes for 128 elements.
- Effective Bitrate: $\frac{34 \times 8}{128} = 2.125$ bpw.

## 3. Hardware Kernels & Hopper Acceleration
- **CUDA GEMV (`vecdotq.cuh`)**: Specialized vector dot product `vec_dot_ptq1_0_q8_1` pairing 1.75 bpw weights with 8-bit quantized activation vectors (`Q8_1`).
- **Hopper SM90a Tensor Cores**: Optional `GGML_CUDA_HOPPER_Q1` build flag enabling WGMMA (Warpgroup Matrix Multiply-Accumulate) for extreme-throughput prefill.
- **CPU SIMD**: Generic and AVX2/AVX-512 vector dot routines in `quants.c`.

## Synergy with KV Compression
When combined with [[turboquant-kv-cache]] and [[triattention-scoring-eviction]], PrismML allows 27-billion parameter models to execute entirely within 8GB to 12GB of VRAM.

## See Also
- Entity: [[prismml]]
- Model family: [[bonsai-model]]
- Hardware kernels: [[ggml-cuda-kernels]]
- Source code analysis: [[source-src-prismml]]
- Deep synthesis: [[synergy-prismml-turboquant-triattention]]
