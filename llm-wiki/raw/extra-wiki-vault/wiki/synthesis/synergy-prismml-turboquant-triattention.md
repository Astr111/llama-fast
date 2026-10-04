---
title: "Synthesis: Synergy of PrismML, TurboQuant, and TriAttention"
type: "synthesis"
tags: ["synthesis", "prismml", "turboquant", "triattention", "architecture", "optimization"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-prismml]]", "[[source-src-triattention]]", "[[source-src-turboquant]]"]
status: "active"
---

# Synthesis: Synergy of PrismML, TurboQuant, and TriAttention

An architectural synthesis of the three foundational optimization layers in `llama-fast`: **PrismML** weight compression, **TurboQuant** Key-Value cache quantization, and **TriAttention** KV cache eviction.

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                          LLAMA-FAST OPTIMIZATION TRIFECTA                 │
└───────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│     PrismML     │         │   TurboQuant    │         │  TriAttention   │
│  Weight Kernels │         │ KV Quantization │         │ Cache Eviction  │
├─────────────────┤         ├─────────────────┤         ├─────────────────┤
│ • PTQ1_0 (1.75) │         │ • turbo2 / 3 / 4│         │ • RoPE Inversion│
│ • PQ2_0  (2.125)│         │ • WHT Rotation  │         │ • Trig Scoring  │
│ • Base-3 Trit   │         │ • PolarQuant    │         │ • Attention Sink│
│   Packing (5/B) │         │ • QJL Error Est │         │ • Sliding Window│
└─────────────────┘         └─────────────────┘         └─────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
Reduces static model       Reduces bits/token in       Reduces total token
VRAM footprint (8x-9x)     KV cache from 16 to 2-4b    count in cache (4x-10x)
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │ ~40x Effective KV Memory Reduction  │
                  │ + 8x Weight Memory Reduction        │
                  │ = 27B LLM Inference in < 8 GB VRAM  │
                  └─────────────────────────────────────┘
```

## 1. Dimensional Breakdown

### Layer 1: Model Weights (PrismML)
- **Target**: Static parameter storage in High Bandwidth Memory (HBM).
- **Mechanism**: [[prismml-weight-quantization]] replaces 16-bit floating-point weights with lossless base-3 ternary digits (`PTQ1_0`, 1.75 bpw) or sub-2-bit groups (`PQ2_0`, 2.125 bpw).
- **Result**: An fp16 27B parameter model (normally requiring 54 GB VRAM) is compressed to **~6.5 GB**, fitting comfortably on consumer graphics cards (e.g. RTX 4070/4080/4090).

### Layer 2: Per-Token KV Memory (TurboQuant)
- **Target**: Dynamic KV tensor size per token in sequence length $L$.
- **Mechanism**: [[turboquant-kv-cache]] rotates activation channels using randomized [[walsh-hadamard-transform]] to eliminate outlier spikes, then quantizes to optimal Gaussian centroids (`turbo2`, `turbo3`, or `turbo4`).
- **Result**: Reduces the memory per cached token from 16 bits down to 2–4 bits (a **4x to 8x compression factor**).

### Layer 3: Context Token Footprint (TriAttention)
- **Target**: Total sequence length $L$ in the KV cache.
- **Mechanism**: [[triattention-scoring-eviction]] evaluates pre-RoPE trigonometric interaction series across geometric future horizons $D=\{1, 2, 4, \dots, 65536\}$. Unimportant intermediate tokens are evicted, while attention sinks and local sliding window tokens are strictly preserved.
- **Result**: Context token count is capped at a fixed budget (e.g. $B = 2048$ tokens), preventing linear memory exhaustion during infinite-turn conversations.

## 2. Kernel Synergy & GPU Execution
The three components are not merely stacked together; they are deeply coupled inside the CUDA runtime:
1. **On-Chip Dequantization & WHT Inversion**: In `triattention-score.cu`, when keys are stored in `turbo2` or `turbo3`, the scoring kernel does not decompress data to host RAM. It performs dequantization and executes cooperative 7-stage Fast Walsh-Hadamard Transform (`cooperative_fwht_128`) directly in GPU shared memory before computing trigonometric importance scores.
2. **Asynchronous Dispatch & Graph Reuse**: Under [[cuda-graph-concurrency]], forward execution passes are compiled into single-launch CUDA graphs, overlapping ternary GEMV operations with WHT attention projections.

## 3. Practical Implications
By simultaneously attacking:
1. Weight footprint (PrismML: 1.75 bpw)
2. Token bit-width (TurboQuant: 2–4 bpw)
3. Token count (TriAttention: bounded budget)

`llama-fast` achieves an unprecedented combined **~40x reduction in operational memory bandwidth and VRAM overhead**, demonstrating how hardware-aware algorithms enable massive model capabilities in resource-constrained environments.

## Related Notes
- Concepts: [[prismml-weight-quantization]], [[turboquant-kv-cache]], [[triattention-scoring-eviction]], [[walsh-hadamard-transform]], [[cuda-graph-concurrency]]
- Entities: [[prismml]], [[turboquant]], [[bonsai-model]], [[llama-fast]], [[ggml-cuda-kernels]]
- Pipeline trace: [[llama-fast-execution-pipeline]]
