---
title: "Source Summary: PrismML Codebase Analysis"
type: "source"
tags: ["source-summary", "prismml", "ptq1_0", "pq2_0", "codebase"]
created: 2026-10-04
updated: 2026-10-04
raw_path: "raw/code-analysis/prismml-ptq1-pq2-architecture.md"
status: "active"
---

# Source: PrismML Codebase Analysis

- **Focus**: Low-bit weight quantization, base-3 trit packing, and CUDA GEMV kernels
- **Analyzed Files**: `/src/ggml/src/ggml-common.h`, `/src/ggml/src/ggml-cuda/convert.cu`, `/src/ggml/src/ggml-cuda/vecdotq.cuh`, `/src/ggml/src/ggml-cpu/quants.c`

## Executive Summary
Detailed investigation of PrismML's ternary (`PTQ1_0`, 1.75 bpw) and sub-2-bit (`PQ2_0`, 2.125 bpw) weight representations in llama.cpp. Documents how base-3 arithmetic packs 5 trits into 8 bits, reducing memory overhead compared to naive 2-bit storage, and examines the CUDA and CPU dequantization and dot product pipelines.

## Key Technical Takeaways
1. **Base-3 Radix Packing**: $3^5 = 243 \le 256$, enabling 5 ternary digits per byte in `qs[24]`, and 4 trits per byte in `qh[2]`, yielding 28 bytes per 128 weights.
2. **CUDA Implementation**: Unpacking is vectorized via modular multiplications; dot products with `Q8_1` activations are executed in warp-synchronous GEMV kernels.
3. **Hopper sm_90a Acceleration**: Optional WGMMA tensor core acceleration for prefill workloads (`GGML_CUDA_HOPPER_Q1`).

## Affected Wiki Notes
- Concepts: [[prismml-weight-quantization]], [[cuda-graph-concurrency]].
- Entities: [[prismml]], [[bonsai-model]], [[ggml-cuda-kernels]].
- Syntheses: [[synergy-prismml-turboquant-triattention]], [[llama-fast-execution-pipeline]].
