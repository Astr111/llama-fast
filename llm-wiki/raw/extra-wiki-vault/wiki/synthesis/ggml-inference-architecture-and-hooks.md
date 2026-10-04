---
title: "GGML Inference Architecture and Hook Points"
type: "synthesis"
tags: ["ggml", "inference", "hooks", "cgraph", "cuda", "runtime", "architecture"]
created: 2026-10-04
updated: 2026-10-04
sources: ["[[source-src-prismml]]", "[[source-src-turboquant]]", "[[source-src-triattention]]"]
status: "mature"
---

# GGML Inference Architecture and Hook Points

A comprehensive technical reference detailing the internal execution pipeline of **GGML** during Large Language Model inference, memory management, computation graph construction, and precise hook points for runtime interception.

---

## 1. Memory Architecture & Core Data Structures

GGML strictly separates **metadata descriptors** from **raw tensor payloads**:

### 1.1 `ggml_context` and `ggml_init_params`
- A `ggml_context` is an arena allocator for struct descriptors (`ggml_tensor`, `ggml_cgraph`).
- During inference graph building, `no_alloc = true` is set in `ggml_init_params`. The context allocates **zero bytes** of tensor data; it only tracks the DAG topology and tensor metadata.

### 1.2 `struct ggml_tensor`
Defined in `ggml.h`, the tensor descriptor contains:
```c
struct ggml_tensor {
    enum ggml_type type;                    // Data type (F32, F16, Q4_K, PTQ1_0, TURBO3_0, etc.)
    struct ggml_backend_buffer * buffer;   // Backend memory buffer owning data
    int64_t ne[GGML_MAX_DIMS];             // Number of elements per dimension (ne[0]..ne[3])
    size_t  nb[GGML_MAX_DIMS];             // Strides in bytes
    enum ggml_op op;                       // Operation producing this tensor (GGML_OP_ADD, MUL_MAT, etc.)
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]; // Operator-specific parameters
    int32_t flags;                         // GGML_TENSOR_FLAG_INPUT, OUTPUT, PARAM, COMPUTE
    struct ggml_tensor * src[GGML_MAX_SRC];// Input operand tensors
    struct ggml_tensor * view_src;         // If view, pointer to base tensor
    size_t               view_offs;        // Byte offset within view_src->data
    void * data;                           // Pointer to allocated memory on device/host
    char name[GGML_MAX_NAME];              // Tensor identifier (e.g. "cur", "ffn_up-0")
    void * extra;                          // Backend-specific private context (CUDA events, etc.)
};
```

**Strides Calculation (`nb`)**:
- Dimension order is column-major: `ne[0]` is contiguous row elements, `ne[1]` is rows.
- `nb[0] = ggml_type_size(type)`
- `nb[1] = nb[0] * (ne[0] / ggml_blck_size(type)) + padding`
- `nb[i] = nb[i-1] * ne[i-1]`

### 1.3 `struct ggml_cgraph`
Defined in `ggml-impl.h`, representing the linearized Directed Acyclic Graph:
```c
struct ggml_cgraph {
    int size;                               // Maximum node capacity
    int n_nodes;                            // Number of operator nodes to compute
    int n_leafs;                            // Number of leaf nodes (inputs, constants, weights)
    struct ggml_tensor ** nodes;            // Array of operator tensors in topological order
    struct ggml_tensor ** leafs;            // Array of leaf tensors
    struct ggml_hash_set  visited_hash_set; // Dedup during graph expansion
    enum ggml_cgraph_eval_order order;      // Topological evaluation order
};
```

---

## 2. Graph Construction & Memory Planning

### 2.1 Forward Expansion (`ggml_build_forward_expand`)
- Calling `ggml_build_forward_expand(cgraph, output_tensor)` performs recursive Depth-First Search (DFS) starting from the output tensor down to the leaves.
- Tensors are topologically sorted into `cgraph->nodes`.

### 2.2 Static Memory Planning (`ggml_gallocr`)
- LLM inference exhibits predictable tensor lifespans: an activation is created, consumed by 1 or 2 downstream layers, and never used again.
- `ggml_gallocr` analyzes the lifespan of every node in `cgraph`. It reuses the same physical memory offsets for non-overlapping activations.
- As a result, peak memory during inference equals only the largest simultaneous working set, not the sum of all layers.

### 2.3 Heterogeneous Backend Scheduling (`ggml_backend_sched`)
- `ggml_backend_sched_split_graph` splits the graph across available devices (e.g. CUDA GPU and CPU).
- Where a tensor computed on GPU is needed by CPU (or vice-versa), the scheduler automatically injects asynchronous copy nodes (`ggml_backend_tensor_copy_async`).

---

## 3. End-to-End Inference Execution Flow

```text
[Input Tokens / Prompt]
         │
         ▼
llama_decode() -> llama_build_graph()
  │  (Creates ggml_context with no_alloc = true)
  │  (Builds embeddings, layers, attention, FFN, and logits)
         │
         ▼
ggml_backend_sched_alloc_graph(sched, gf)
  │  (Assigns buffer memory offsets via static planning)
         │
         ▼
ggml_backend_sched_graph_compute_async(sched, gf)
  │  ┌─────────────────────────────────────────────────────────────┐
  │  │ Loop over nodes in cgraph:                                  │
  │  │ 1. cb_eval(node, ask = true)  <-- PRE-EXECUTION HOOK        │
  │  │ 2. Backend executes kernel (CUDA stream / CPU threadpool)   │
  │  │ 3. cb_eval(node, ask = false) <-- POST-EXECUTION HOOK       │
  │  └─────────────────────────────────────────────────────────────┘
         │
         ▼
ggml_backend_tensor_get(res_logits, ...) -> Sampler -> Next Token
```

---

## 4. The 5 Essential Hook Points

For developers looking to inspect, modify, or extend GGML inference, five primary interception mechanisms exist:

### Hook Point 1: Evaluation Callback (`cb_eval`)
The primary runtime hook exposed via `llama_context_params.cb_eval`.

```cpp
// Hook function signature
bool my_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        // PRE-EXECUTION: t is about to be computed
        // Return true to compute, false to skip
        return true;
    } else {
        // POST-EXECUTION: t has just completed execution on device!
        // Inspect or modify tensor data:
        if (strcmp(t->name, "result_output") == 0 || strstr(t->name, "ffn_out")) {
            std::vector<float> data(ggml_nelements(t));
            ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
            // Apply custom modification, steering vectors, or logging
            // ggml_backend_tensor_set(t, modified.data(), 0, ggml_nbytes(t));
        }
        return true;
    }
}

// Registration in context setup:
llama_context_params cparams = llama_context_default_params();
cparams.cb_eval = my_eval_callback;
cparams.cb_eval_user_data = nullptr;
llama_context * ctx = llama_init_from_model(model, cparams);
```

### Hook Point 2: Graph Construction Callback (`llm_graph_cb`)
Intercepts tensors as the computational DAG is being generated inside `llama-graph.cpp`. Allows replacing an entire branch with a custom subgraph before memory allocation.

```cpp
llm_graph_cb cb = [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
    // Intercept specific layer activation
    if (strcmp(name, "cur_norm") == 0 && il == 10) {
        // Modify cur->op, add custom residual, or attach metadata
    }
};
```

### Hook Point 3: Custom Operators (`ggml_map_custom1/2/3` and `ggml_custom_4d`)
Injects arbitrary C++ or CUDA functions directly into the computation graph as native nodes.

```cpp
void my_custom_activation(struct ggml_tensor * dst, const struct ggml_tensor * src,
                          int ith, int nth, void * userdata) {
    // ith: thread index, nth: total worker threads
    float * d = (float *)dst->data;
    const float * s = (const float *)src->data;
    int64_t n = ggml_nelements(src);

    int64_t start = (n * ith) / nth;
    int64_t end   = (n * (ith + 1)) / nth;

    for (int64_t i = start; i < end; ++i) {
        d[i] = s[i] > 0 ? s[i] : 0.1f * s[i]; // Custom LeakyReLU
    }
}

// In graph building:
struct ggml_tensor * custom_layer = ggml_map_custom1(ctx, input_tensor, my_custom_activation,
                                                     GGML_N_TASKS_MAX, nullptr);
```

### Hook Point 4: Direct Backend Buffer Interception (`ggml_backend_tensor_get / set`)
Synchronous and asynchronous primitives for reading and writing device tensors (compatible with CUDA, Metal, and Host RAM):
```cpp
// Asynchronously write activations or steering vectors into device tensor:
ggml_backend_tensor_set_async(backend, tensor, host_buf, 0, n_bytes);

// Synchronously read tensor data from GPU to Host:
ggml_backend_tensor_get(tensor, host_buf, 0, n_bytes);
```

### Hook Point 5: Abort Callback (`ggml_abort_callback`)
Stops computation mid-graph if a specific condition (e.g. timeout, EOS token detection, safety trigger) occurs:
```cpp
bool my_abort_check(void * data) {
    return stop_signal_received.load();
}
// Set on context:
ggml_set_abort_callback(ctx, my_abort_check, nullptr);
```

---

## Related Notes
- Concepts: [[ggml-tensor-execution-graph]], [[cuda-graph-concurrency]], [[prismml-weight-quantization]], [[turboquant-kv-cache]], [[triattention-scoring-eviction]]
- Entities: [[llama-fast]], [[ggml-cuda-kernels]], [[prismml]], [[turboquant]]
- Synthesis: [[llama-fast-execution-pipeline]], [[memory-bounded-llm-inference]]
