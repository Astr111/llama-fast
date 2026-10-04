---
title: "Source Summary: TurboQuant Codebase Analysis"
type: "source"
tags: ["source-summary", "turboquant", "wht", "polarquant", "qjl", "codebase"]
created: 2026-10-04
updated: 2026-10-04
raw_path: "raw/code-analysis/turboquant-wht-integration.md"
status: "active"
---

# Source: TurboQuant Codebase Analysis

- **Focus**: PolarQuant centroid optimization, randomized WHT rotation, and computation graph integration
- **Analyzed Files**: `/src/ggml/src/ggml-turbo-quant.c`, `/src/ggml/src/ggml-cuda/turbo-wht.cu`, `/src/src/llama-graph.cpp`, `/src/ggml/include/ggml.h`

## Executive Summary
Investigation of the TurboQuant KV cache compression subsystem in llama-fast. Details the three operational formats (`turbo2_0`, `turbo3_0`, `turbo4_0`), the mathematical mechanism of Fast Walsh-Hadamard Transform rotation for outlier mitigation, and the exact integration points inside `llama-graph.cpp`.

## Key Technical Takeaways
1. **Inner Product Invariance**: Rotating Queries ($Q$) by forward WHT and storing Keys ($K$) in rotated form preserves dot product values while spreading outlier activation mass.
2. **PolarQuant Centroids**: Uses pre-computed Lloyd-Max centroids fitted to Gaussian distributions rather than linear quantization buckets.
3. **Graph Lifecycle**: Forward WHT on Queries before dot product; inverse WHT on Attention Output to restore unrotated activations.

## Affected Wiki Notes
- Concepts: [[turboquant-kv-cache]], [[walsh-hadamard-transform]].
- Entities: [[turboquant]], [[llama-fast]], [[ggml-cuda-kernels]].
- Syntheses: [[synergy-prismml-turboquant-triattention]], [[llama-fast-execution-pipeline]].
