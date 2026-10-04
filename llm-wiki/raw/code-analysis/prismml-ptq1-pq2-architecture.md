# Codebase Extract: PrismML Low-Bit Weight Kernels (PTQ1_0 & PQ2_0)

**Source Path**: `/src/ggml/src/ggml-common.h`, `/src/ggml/src/ggml-cuda/convert.cu`, `/src/ggml/src/ggml-cuda/vecdotq.cuh`, `/src/ggml/src/ggml-cpu/quants.c`  
**Components**: `PTQ1_0` (Ternary 1.75 bpw), `PQ2_0` (Sub-2-bit 2.125 bpw)

## 1. Struct Definitions (`ggml-common.h`)

### `block_pq2_0` (Group size 128, 2 bits per element)
```c
#define QK_PQ2_0 128
typedef struct {
    ggml_half d;                 // 2 bytes fp16 delta scale
    uint8_t qs[QK_PQ2_0 / 4];    // 32 bytes (2 bits per element)
} block_pq2_0;                   // Total: 34 bytes for 128 elements (2.125 bpw)
```

### `block_ptq1_0` (Group size 128, Base-3 Trit Packing)
```c
#define QK_PTQ1_0 128
typedef struct {
    uint8_t qs[(QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5]; // 24 bytes (5 trits per byte -> 120 trits)
    uint8_t qh[QK_PTQ1_0/64];                   // 2 bytes (4 trits per byte -> 8 trits)
    ggml_half d;                                // 2 bytes fp16 delta scale
} block_ptq1_0;                                 // Total: 28 bytes for 128 elements (1.75 bpw)
```

## 2. Base-3 Trit Encoding & Decoding Algorithm
Because $3^5 = 243 < 256$, a single 8-bit byte packs 5 ternary digits ($\{-1, 0, +1\}$).
The remaining 8 elements in the 128 block are packed into `qh` ($3^4 = 81 < 256$, 4 trits per byte across 2 bytes).

CUDA Device Decoding (`ggml-cuda/common.cuh` & `dequantize.cuh`):
```cuda
static __device__ __forceinline__ int ptq1_0_trit(const block_ptq1_0 * x, const int e) {
    uint8_t b; int n;
    if (e < 80)       { b = x->qs[e & 15];              n = e >> 4; }
    else if (e < 120) { const int t = e - 80; b = x->qs[16 + (t & 7)]; n = t >> 3; }
    else              { const int t = e - 120; b = x->qh[t & 1];       n = t >> 1; }
    uint32_t v = b;
    for (int i = 0; i < 4; ++i) if (i < n) v = (v * 3) & 0xFF;
    return (int)((v * 3) >> 8) - 1; // Unpacks trit value in {-1, 0, +1}
}
```

## 3. Vector Dot Product Kernels
- CUDA: `vec_dot_ptq1_0_q8_1` and multi-column GEMV `vec_dot_ptq1_0_q8_1_multi` in `vecdotq.cuh`.
- Hopper SM90a optimization: `GGML_CUDA_HOPPER_Q1` enabling asynchronous Tensor Core WGMMA (Warpgroup Matrix Multiply-Accumulate) for prefill.
- CPU: AVX2 / AVX-512 / ARM Neon vector routines in `quants.c`.
