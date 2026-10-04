---
title: "TurboQuant KV Cache Quantization"
type: "concept"
tags: ["kv-cache", "turboquant", "polarquant", "qjl", "wht", "compression"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-turboquant]]"]
status: "active"
---

# TurboQuant KV Cache Quantization

**TurboQuant** is an algorithm for compressing Large Language Model Key-Value (KV) caches down to 2, 3, or 4 bits per element without catastrophic perplexity loss, combining Polar Vector Quantization with the [[walsh-hadamard-transform]].

## 1. Supported Quantization Formats

| Format | GGML Type | Block Structure | Memory Footprint | Description |
| :--- | :--- | :--- | :--- | :--- |
| **turbo2_0** | `GGML_TYPE_TURBO2_0` (45) | fp16 norm + 2-bit centroids | 10 B / 32 elem (2.5 bpw) | Pure 2-bit PolarQuant |
| **turbo3_0** | `GGML_TYPE_TURBO3_0` (43) | fp16 norm + 2-bit qs + 1-bit sign | 14 B / 32 elem (3.5 bpw) | 2-bit PolarQuant + WHT signs |
| **turbo4_0** | `GGML_TYPE_TURBO4_0` (44) | fp16 norm + qjl_scale + 3-bit qs + QJL signs | 68 B / 128 elem (4.25 bpw) | 3-bit PolarQuant + 1-bit QJL |

## 2. PolarQuant Centroid Optimization
Instead of naive uniform linear quantization (which distorts Gaussian token distributions), TurboQuant calculates L2-optimal centroids:
- **2-bit Centroids**: $\{-0.133462, -0.039994, +0.039994, +0.133462\}$ (scaled by $1/\sqrt{d}$).
- **3-bit Centroids**: 8 Lloyd-Max centroids for Gaussian distribution $\mathcal{N}(0, 1/128)$.
- **Norm Correction**: Stored fp16 scale is multiplied by the ratio of original vector norm to reconstructed centroid norm, guaranteeing unbiased expectation.

## 3. Quantized Johnson-Lindenstrauss (QJL) Projection
In `turbo4_0`, a 1-bit residual error projection is added using an orthogonal random projection matrix $S_{\text{QJL}}$:
$$\text{sign} = \text{sign}(S_{\text{QJL}} \cdot (x - \hat{x}))$$
This acts as an unbiased estimator for residual inner-product errors, pushing effective accuracy close to fp16 baseline even under long contexts.

## 4. Integration with Attention Graph
TurboQuant relies on the [[walsh-hadamard-transform]] to eliminate outlier channels before quantization:
- Forward WHT applied to Queries ($Q$) in `llama-graph.cpp`.
- Cached Keys ($K$) and Values ($V$) stored in rotated and quantized state.
- Inverse WHT applied to Attention Output ($cur$) to restore baseline activations.

## See Also
- Mathematical transformation: [[walsh-hadamard-transform]]
- Context pruning partner: [[triattention-scoring-eviction]]
- Engine implementation: [[llama-fast]]
- Source code analysis: [[source-src-turboquant]]
