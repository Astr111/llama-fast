---
title: "Walsh-Hadamard Transform in Attention"
type: "concept"
tags: ["wht", "hadamard", "rotation", "math", "outliers", "kv-cache"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-turboquant]]"]
status: "active"
---

# Walsh-Hadamard Transform in Attention

The **Fast Walsh-Hadamard Transform (FWHT)** is an $O(d \log d)$ orthogonal matrix multiplication technique used in [[turboquant-kv-cache]] to disperse activation outliers across all latent dimensions before quantization.

## The Problem: Outlier Dimensions
In Transformer activations, a small subset of latent channels often exhibit extreme numerical magnitudes ($\times 50$ to $\times 100$ larger than average). In low-bit quantization, these outliers force the quantization grid scale to expand, squashing $99\%$ of non-outlier channels into zero or near-zero buckets and destroying attention fidelity.

## The Solution: Randomized WHT Rotation
A randomized orthogonal matrix $R \in \mathbb{R}^{d \times d}$ rotates the representation space:
$$R = S_2 \cdot H_d \cdot S_1 \cdot \frac{1}{\sqrt{d}}$$
Where:
- $S_1, S_2 \in \{-1, +1\}^d$ are diagonal random sign flips (seeded deterministically with 42).
- $H_d$ is the recursive Sylvester-Hadamard matrix. For $d=128$, FWHT requires exactly $7$ butterfly stages.

### Properties
1. **Outlier Diffusion**: Spreads the energy of outlier dimensions uniformly across all $d$ dimensions according to the Central Limit Theorem. Rotated coordinates closely follow a Gaussian distribution $\mathcal{N}(0, \sigma^2)$, matching the optimal Lloyd-Max centroids in [[turboquant-kv-cache]].
2. **Inner Product Preservation**: Because $R$ is orthogonal ($R^T R = I$):
   $$\langle R Q, R K \rangle = (R Q)^T (R K) = Q^T (R^T R) K = Q^T K = \langle Q, K \rangle$$
   Therefore, attention logits $\frac{Q K^T}{\sqrt{d_k}}$ are mathematically invariant under WHT rotation!

## Hardware Implementation
- **CPU**: In-place butterfly loops in `ggml-turbo-quant.c` and `ggml-cpu.c`.
- **CUDA**: `cooperative_fwht_128` implemented in GPU shared memory using warp-synchronous butterfly exchanges in `triattention-score.cu` and `turbo-wht.cu`.

## See Also
- Applied KV cache quantization: [[turboquant-kv-cache]]
- Scoring interaction: [[triattention-scoring-eviction]]
