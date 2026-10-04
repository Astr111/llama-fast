---
title: "Source Summary: TriAttention Codebase Analysis"
type: "source"
tags: ["source-summary", "triattention", "eviction", "rope", "codebase"]
created: 2026-10-04
updated: 2026-10-04
raw_path: "raw/code-analysis/triattention-codebase-deep-dive.md"
status: "active"
---

# Source: TriAttention Codebase Analysis

- **Focus**: Trigonometric key scoring, RoPE inversion, GPU shared-memory WHT inversion, and KV eviction
- **Analyzed Files**: `/src/src/llama-triattention.h`, `/src/src/llama-triattention.cpp`, `/src/ggml/src/ggml-cuda/triattention-score.cu`

## Executive Summary
Comprehensive trace of TriAttention KV cache pruning in llama.cpp. Analyzes the binary calibration schema (`.triattention`), the mathematical formula modeling future attention interactions across geometric offsets $D=\{1, \dots, 65536\}$, and the GPU scoring kernel that inverts both TurboQuant WHT rotation and RoPE position shifts.

## Key Technical Takeaways
1. **RoPE Inversion**: Post-RoPE keys are transformed back to pre-RoPE base vectors using complex trigonometric rotation before scoring.
2. **Trigonometric Series Scoring**: Evaluates complex amplitude $\text{amp}_f$ and phase angle $\phi_f$ against future query expectations.
3. **On-GPU Cooperative WHT**: The scoring kernel executes an in-place 7-stage butterfly transform in CUDA shared memory to handle TurboQuant-compressed keys without roundtripping to host RAM.

## Affected Wiki Notes
- Concepts: [[triattention-scoring-eviction]], [[tri-attention-mechanism]], [[kv-cache-eviction]].
- Entities: [[llama-fast]], [[bonsai-model]], [[ggml-cuda-kernels]].
- Syntheses: [[synergy-prismml-turboquant-triattention]], [[memory-bounded-llm-inference]].
