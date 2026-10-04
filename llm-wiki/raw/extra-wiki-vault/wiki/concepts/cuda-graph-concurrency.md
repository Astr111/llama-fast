---
title: "CUDA Graph Reuse and Stream Concurrency"
type: "concept"
tags: ["cuda", "gpu", "concurrency", "cuda-graphs", "latency", "streams"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-prismml]]"]
status: "active"
---

# CUDA Graph Reuse and Stream Concurrency

**CUDA Graph Reuse and Stream Concurrency** is an architectural optimization in `llama-fast` (controlled via `GGML_CUDA_GRAPH_OPT=1`) designed to eliminate CPU kernel launch bottlenecks during autoregressive token decoding.

## 1. The Kernel Launch Bottleneck
During decode generation:
- Each token triggers dozens of micro-kernels (matrix-vector multiplications, layer norms, RoPE shifts, softmax, residual adds).
- At high token generation rates (e.g. $> 100$ tokens/sec), the CPU overhead of dispatching individual CUDA kernel launches (`cudaLaunchKernel`) exceeds the actual execution duration of the kernels on the GPU streaming multiprocessors.

## 2. CUDA Graph Capture and Instantiation
With `GGML_CUDA_GRAPH_OPT=1`:
1. **Graph Capture**: The entire forward pass execution graph for a single decode step is captured into an immutable CUDA Graph (`cudaGraph_t`).
2. **Graph Instantiation**: The driver compiles the graph once into an executable graph instance (`cudaGraphExec_t`).
3. **Zero-Overhead Replay**: Subsequent decode steps replay the instantiated graph via a single call to `cudaGraphLaunch`, shifting control entirely to GPU hardware schedulers and reducing CPU-GPU dispatch latency to near zero.

## 3. Concurrent Multi-Stream Execution
- Attention head projections ($Q, K, V$) are dispatched to independent CUDA streams, executing concurrently across GPU SMs.
- KV cache updates and dequantization of [[prismml-weight-quantization]] weights overlap with activation normalization.

## See Also
- Inference engine: [[llama-fast]]
- Kernel ecosystem: [[ggml-cuda-kernels]]
- Hardware synthesis: [[memory-bounded-llm-inference]]
