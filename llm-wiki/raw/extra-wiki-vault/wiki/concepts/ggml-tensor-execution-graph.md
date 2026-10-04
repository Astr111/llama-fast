---
title: "GGML Tensor Execution Graph"
type: "concept"
tags: ["ggml", "cgraph", "tensor", "memory-planning", "scheduler", "runtime"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-prismml]]", "[[source-src-turboquant]]"]
status: "active"
---

# GGML Tensor Execution Graph

The **GGML Tensor Execution Graph** (`struct ggml_cgraph`) is the foundational Directed Acyclic Graph (DAG) data structure that organizes, schedules, and evaluates neural network operations in llama.cpp.

## 1. Graph Life Cycle
1. **Creation**: Initialized via `ggml_new_graph(ctx)` or allocated inside `llama_context`.
2. **Expansion**: `ggml_build_forward_expand(cgraph, target_node)` traverses backward via Depth-First Search (DFS), populating `cgraph->nodes` in topological order.
3. **Memory Planning**: Evaluated by `ggml_gallocr` to statically bind buffer offsets, overlapping non-concurrent tensor lifetimes.
4. **Partitioning**: Split into heterogeneous device subgraphs by `ggml_backend_sched`.
5. **Execution**: Dispatched to GPU kernels or CPU worker threads via `ggml_backend_sched_graph_compute_async`.

## 2. In-Place Operations & Views
GGML optimizes memory through zero-copy mechanisms:
- **Tensor Views (`view_src`, `view_offs`)**: Created via `ggml_view_1d/2d/3d/4d`. Points into memory of an existing tensor with modified strides (`nb`) without allocating new memory.
- **In-Place Operators**: Ops like `ggml_add_inplace` or `ggml_rms_norm_inplace` write directly to their primary input buffer when safe.

## See Also
- Deep synthesis & hook points: [[ggml-inference-architecture-and-hooks]]
- Backend hardware acceleration: [[ggml-cuda-kernels]], [[cuda-graph-concurrency]]
- Execution pipeline: [[llama-fast-execution-pipeline]]
