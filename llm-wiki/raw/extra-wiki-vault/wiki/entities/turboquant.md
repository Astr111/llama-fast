---
title: "TurboQuant"
type: "entity"
tags: ["algorithm", "kv-cache", "polarquant", "qjl", "compression"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-turboquant]]"]
status: "active"
---

# TurboQuant

**TurboQuant** is a vector quantization framework introduced at ICLR 2026 (arXiv 2504.19874) designed for aggressive Key-Value (KV) cache compression in Transformer inference.

## Key Capabilities
- **2-bit, 3-bit, and 4-bit KV Cache**: Reduces cache footprint by 4x to 8x compared to fp16 baselines.
- **PolarQuant**: Replaces uniform grids with mathematically optimal centroids fitted to Gaussian activation distributions.
- **Walsh-Hadamard Transform (WHT)**: Integrates $O(d \log d)$ randomized orthogonal rotations to neutralize activation outliers.
- **Quantized Johnson-Lindenstrauss (QJL)**: Incorporates 1-bit residual error projections to ensure unbiased inner-product estimation.

## Implementation in llama-fast
Exposed via CLI arguments:
`--cache-type-k turbo2 / turbo3 / turbo4`
`--cache-type-v turbo2 / turbo3 / turbo4`

## See Also
- Applied mechanism: [[turboquant-kv-cache]]
- Mathematical rotation: [[walsh-hadamard-transform]]
- Inference engine: [[llama-fast]]
- Source code analysis: [[source-src-turboquant]]
