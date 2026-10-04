---
title: "PrismML"
type: "entity"
tags: ["organization", "engine", "low-bit", "quantization", "prismml"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-prismml]]"]
status: "active"
---

# PrismML

**PrismML** is an engineering and AI research team specializing in extreme low-bit LLM quantization, sub-2-bit inference kernels, and the [[bonsai-model]] series.

## Technical Innovations in llama.cpp
- **Fork Development**: Main line behind the `prism` / `prism-v7` branch of llama.cpp.
- **PTQ1_0 Codec**: Lossless base-3 packing for ternary checkpoints (1.75 bpw), packing 5 trits per byte.
- **PQ2_0 Codec**: Group-128 2-bit quantization format (2.125 bpw) with high-efficiency vector dot products.
- **Hopper SM90a Integration**: `GGML_CUDA_HOPPER_Q1` WGMMA path for accelerating prefill phases on NVIDIA H100/H200 GPUs.

## Key Projects & Models
- [[bonsai-model]]: High-parameter models trained and quantized in ternary and sub-2-bit formats.
- Integration with [[turboquant-kv-cache]] and [[triattention-scoring-eviction]] in `llama-fast`.

## See Also
- Weight quantization: [[prismml-weight-quantization]]
- Hardware kernels: [[ggml-cuda-kernels]]
- Source code analysis: [[source-src-prismml]]
