---
title: "Bonsai Model Family"
type: "entity"
tags: ["model", "llm", "bonsai", "ternary", "27b", "prismml"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-prismml]]", "[[source-src-triattention]]"]
status: "active"
---

# Bonsai Model Family

The **Bonsai Model Family** comprises large language models engineered by [[prismml]] trained natively or quantized into ternary ($\{-1, 0, +1\}$) and sub-2-bit parameter representations.

## Key Checkpoints
- **Bonsai-27B**: 27-billion parameter dense model operating natively in `PTQ1_0` (1.75 bpw) and `PQ2_0` (2.125 bpw).
- **Memory Footprint**: While an fp16 27B model requires $> 54\text{ GB}$ of VRAM, Bonsai-27B in `PTQ1_0` requires less than $6.5\text{ GB}$ of weight memory.

## Calibration for TriAttention
Bonsai models ship with dedicated binary attention calibration profiles (such as `calibration/bonsai-27b.triattention`):
- Measures complex expectation values $E[q_f]$ across all attention layers and heads.
- Enables zero-shot [[triattention-scoring-eviction]] during inference.

## See Also
- Creator organization: [[prismml]]
- Weight quantization: [[prismml-weight-quantization]]
- Attention eviction: [[triattention-scoring-eviction]]
