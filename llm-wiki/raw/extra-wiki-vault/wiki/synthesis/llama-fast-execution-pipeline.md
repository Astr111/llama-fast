---
title: "Synthesis: llama-fast Execution Pipeline"
type: "synthesis"
tags: ["synthesis", "execution-pipeline", "llama-fast", "cuda", "runtime"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-prismml]]", "[[source-src-triattention]]", "[[source-src-turboquant]]"]
status: "active"
---

# Synthesis: llama-fast Execution Pipeline

A step-by-step trace of tensor computation during prefill and autoregressive decode in `llama-fast`, highlighting the interaction of [[prismml-weight-quantization]], [[turboquant-kv-cache]], and [[triattention-scoring-eviction]].

## 1. Prefill Phase (Prompt Processing)

```text
[Input Prompt Tokens]
         │
         ▼
[Embedding Lookup]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Transformer Layer                                      │
│                                                        │
│ 1. Forward LayerNorm                                   │
│ 2. Q, K, V Projections via PrismML GEMM               │
│    • Weights: PTQ1_0 (1.75 bpw) / PQ2_0 (2.125 bpw)   │
│    • Hopper SM90a: Async WGMMA Tensor Cores           │
│ 3. RoPE Position Encoding on Q and K                   │
│ 4. TurboQuant Ingestion:                               │
│    • Rotate Q via forward WHT (ggml_turbo_wht, dir=0)  │
│    • Rotate K and V via forward WHT                   │
│    • Quantize K and V into PolarQuant centroids        │
│    • Store into KV Cache (turbo2, turbo3, or turbo4)  │
│ 5. Attention Computation:                             │
│    • Dot product <Q_rot, K_rot> (inner product match)  │
│    • Softmax scaling                                  │
│    • Multiply by V_rot                                │
│ 6. Output Un-rotation:                                 │
│    • Inverse WHT on attention output (dir=1)          │
│ 7. FFN Projection via PrismML PTQ1_0 GEMM              │
└────────────────────────────────────────────────────────┘
```

## 2. Autoregressive Decode Phase (Token Generation)

During single-token decoding, latency is memory-bandwidth bounded:
1. **Activation Normalization**: Immediate LayerNorm on the new token vector.
2. **Q, K, V Projection**: Computed via `vec_dot_ptq1_0_q8_1` / `vec_dot_pq2_0_q8_1` (`[[ggml-cuda-kernels]]`).
3. **KV Cache Append**: New Key and Value vectors are rotated by [[walsh-hadamard-transform]], quantized, and appended to the active cache buffer.
4. **Attention Evaluation**: The rotated query vector $R Q$ computes dot products with all active rotated keys $R K$ stored in the KV cache.
5. **Attention Output Un-rotation**: Inverse WHT is applied to restore standard activation coordinates.
6. **CUDA Graph Replay**: The entire forward iteration is replayed with near-zero launch latency under [[cuda-graph-concurrency]].

## 3. The TriAttention Eviction Event

When the token decode counter reaches the configured trigger condition (e.g. cache size exceeds budget $B$ by `divide_length` tokens):

```text
[KV Cache Capacity Limit Reached]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ TriAttention Pruning Routine                           │
│                                                        │
│ 1. Identify Candidate Cells (excluding protected sink  │
│    and sliding-window tokens)                          │
│ 2. GPU Scoring Kernel Launch (triattention-score.cu):  │
│    • Shared memory cooperative FWHT inverts TurboQuant │
│      rotation on cached keys                           │
│    • Inverse RoPE recovers base pre-RoPE key vectors   │
│    • Trigonometric interaction calculated across       │
│      geometric horizons D = {1, 2, 4, ..., 65536}      │
│    • MLR norm term added for uncalibrated variance     │
│ 3. Top-B Keep-Set Selection (Global or Per-KV-Head)    │
│ 4. Eviction: Cells below threshold are freed           │
│ 5. Re-packing / Slot Compaction                        │
└────────────────────────────────────────────────────────┘
         │
         ▼
[KV Cache bounded at budget B; Decode continues seamlessly]
```

## Related Notes
- Technical synergy: [[synergy-prismml-turboquant-triattention]]
- Quantization mechanics: [[prismml-weight-quantization]], [[turboquant-kv-cache]]
- Eviction algorithm: [[triattention-scoring-eviction]]
