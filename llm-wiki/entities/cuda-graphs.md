---
title: CUDA graphs and concurrent streams
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/ggml/src/ggml-cuda/ggml-cuda.cu, src/ggml/src/ggml-cuda/common.cuh, src/ggml/src/ggml-cuda/CMakeLists.txt, src/ggml/CMakeLists.txt, src/CMakeLists.txt, src/ggml/src/ggml-backend.cpp, src/ggml/src/ggml-backend-impl.h, scripts/start_server_turbo.sh, scripts/run_cli.sh]
tags: [cuda, cuda-graphs, streams, scheduling, performance]
---

# CUDA graphs and concurrent streams

## What it is

The backend's **launch-overhead and stream-parallelism layer**, switched on by `GGML_CUDA_GRAPH_OPT=1` ([[source-readme]], *Engine & Concurrency Environment Variables*: "Enables concurrent stream execution for attention projections and CUDA graph reuse").

It is two independent mechanisms behind one flag, and conflating them is the classic misreading of this subsystem:

1. **CUDA graph capture/replay** — the standard llama.cpp graph cache, which records a whole `ggml_cgraph` into a `cudaGraphExec_t` and replays it with one `cudaGraphLaunch` instead of N kernel launches.
2. **Concurrent streams for the Q/K/V projections** — a graph rewrite that finds the `attn_norm` fork, interleaves the three projection branches so their tensors stay alive simultaneously, and executes them on separate non-blocking streams joined by events.

Mechanism (2) is what `GGML_CUDA_GRAPH_OPT` actually gates; mechanism (1) is compiled in by a *different* switch (`GGML_CUDA_GRAPHS`, on by default for llama builds: `src/CMakeLists.txt:171-173`, surfaced as `GGML_CUDA_USE_GRAPHS` at `src/ggml/src/ggml-cuda/CMakeLists.txt:149-150`). Both launch scripts export the env var unconditionally:

- `scripts/start_server_turbo.sh` — `export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"`
- `scripts/run_cli.sh` — same line, same default

## How it works

### The recorded finding (the reason this page exists)

[[source-state-md]] §1.1 is the project's own conclusion from Nsight Systems profiling, and it is a *negative* result worth quoting precisely:

> "**CUDA Graph Reuse:** Does *not* accumulate speed improvements over time. Launch overhead reduction is a constant one-time saving (~25 µs vs 16–64 ms). Observed decode degradation (15.75 ms → 19.43 ms) correlates with unbounded KV attention costs as context grows, not graph reuse count." ([[source-state-md]] §1.1)

So: graph reuse buys a **constant** ~25 µs per graph launch against a 16–64 ms budget, and the measured decode slowdown as context grows is **not** attributable to it. The degradation tracks the cost of attention over an ever longer KV cache — the same growing cost that [[triattention]] and [[turboquant]] exist to bound, and that [[tq-1-missing-gemm-kernels]]'s cuBLAS/MAGMA fallback amplifies (38.81 % / 157 ms of GPU time per the same section). The `15.75 → 19.43 ms` figures are recorded verbatim and are **`[UNVERIFIED]` against this tree** — they are a measurement, not something the code can confirm. The `[[benchmarks]]` page carries no CUDA-graph ablation either.

Practical consequence: the concurrency work is a **fixed-cost** optimisation. It cannot be the explanation for any growth-over-time curve, and it cannot rescue a decode budget dominated by attention arithmetic.

### Graph capture and replay

`ggml_backend_cuda_graph_compute` (`src/ggml/src/ggml-cuda/ggml-cuda.cu:4534`) is the entry point, registered in the backend interface as `.graph_compute` (`:4877`) and called by the scheduler. Sequence:

1. `ggml_cuda_graph_get_key(cgraph)` returns `cgraph->nodes[0]` as the cache key (`:2593-2595`) — one graph object per distinct first node.
2. `ggml_cuda_graph_set_enabled()` (`:4518-4532`) disables graphs outright for `cc < GGML_CUDA_CC_VOLTA` (`:4522`, i.e. below `sm_70`; `GGML_CUDA_CC_VOLTA = 700` at `common.cuh:52`) and for `GGML_CUDA_DISABLE_GRAPHS` (`common.cuh:1298-1301`).
3. `ggml_cuda_graph_check_compability()` + `ggml_cuda_graph_update_required()` (`:2597-2637`) compare the cached `node_props` (`common.cuh:1290-1296`) — node shapes, strides and **source data pointers** — and set `cuda_graph_update_required`.
4. **Warmup gate** (`:4554-4570`): with `warmup_complete == false`, the graph is only captured after a second consecutive call whose properties did **not** change (`:4556-4560`); if properties change later the warmup resets (`:4563-4566`). This is why capture does not thrash as KV-cache pointers move.
5. Capture is a `cudaStreamBeginCapture(..., cudaStreamCaptureModeRelaxed)` (`:4585`) around `ggml_cuda_graph_evaluate_and_capture()` (`:4143`), closed by `cudaStreamEndCapture` (`:4482`); instantiation at `:4497`, `cudaGraphExecUpdate` (with re-instantiate on failure) in `ggml_cuda_graph_update_executable()` (`:2639-2665`), and finally one `cudaGraphLaunch` (`:4503`).

### The concurrent-stream rewrite

`ggml_backend_cuda_graph_optimize()` (`:4618-4841`) is registered as `.graph_optimize` (`:4878`) and invoked by the scheduler before compute (`src/ggml/src/ggml-backend.cpp:1470` → `ggml_backend_graph_optimize()` `:559-563`). Order of operations:

1. Read the env var **once** into a function-local `static bool` (`:4630-4638`) — `getenv("GGML_CUDA_GRAPH_OPT")` with `atoi(env) == 1`; anything else leaves the function immediately (`:4635-4637`).
2. `stream_context.reset()` clears the cached plan (`:4639-4640`; `ggml_cuda_stream_context::reset` at `common.cuh:1451-1453`), then bail unless `use_cuda_graph && ggml_backend_cuda_get_device_count() == 1` (`:4642-4644`) — the optimisation is single-GPU only.
3. Build `fan_out` / `node_indices` maps over the graph, skipping no-op nodes (`:4646-4685`). The dependence test `depends_on()` also treats two tensors as dependent when they share a `view_src` (`:4663-4673`).
4. Candidate forks are constrained hard: `min_fan_out = max_fan_out = 3` (`:4696-4697`) and the root node's **name must contain `attn_norm`** (`:4706-4708`, with the `// TODO: make this more generic` note). So this is a QKV-specific rewrite, not a general scheduler.
5. Find the join node (first node depending on ≥ 2 branches, `:4733-4751`), collect one node list per branch, and require that the region between fork and join contains **exactly** the branch nodes — otherwise the fork is skipped with a debug log (`:4802-4810`).
6. Build a `ggml_cuda_concurrent_event` (`common.cuh:1305-1446`) carrying `n_streams`, a `stream_mapping` node→stream, the `fork_event`/`join_events`, and `original_order`, then **interleave** the branches in `cgraph->nodes` so their tensors' lifetimes overlap and ggml's allocator cannot recycle them (`:4825-4858`; the worked example in the comment at `:4826-4828` turns `[attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn]` into `[attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]`).

Safety valve: `ggml_cuda_concurrent_event::is_valid()` (`common.cuh:1346-1430`) rejects a plan when two streams' write ranges overlap or when a node consumes a source belonging to a different branch (`:1368-1420`).

### Execution

- Fork: on entering the region, the main stream records `fork_event` and every child stream does `cudaStreamWaitEvent` (`ggml-cuda.cu:4199-4212`, `cudaEventRecord` at `:4207`).
- Dispatch: `cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node]` per node, and `cuda_ctx->stream()` resolves through `stream(device, curr_stream_no)` (`:4301-4307`, `common.cuh:1528-1537`).
- Join: at the join node, each child records its `join_event` and the main stream waits on all of them, then `curr_stream_no = 0` (`:4292-4307`).
- Streams are created lazily and **non-blocking**: `cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking)` (`common.cuh:1532`), up to `GGML_CUDA_MAX_STREAMS = 8` (`common.cuh:178`).
- Before the concurrent region runs, the original node order is restored inside the sorted positions so intra-stream fusion still fires (`:4279-4283`, the `original_order` field collected at `:4812-4817`).

Note the interaction with capture: the interleaved `cgraph` is what gets captured, so the stream/event topology (not just the kernels) is baked into the `cudaGraphExec_t`.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_graph_evaluate_and_capture` `:4143` (stream context use `:4149-4151`, fork `:4206-4211`, join `:4292-4307`, stream selection `:4303-4307`, end capture/instantiate/launch `:4482-4503`); key `:2593`; update-required `:2597`; `ggml_cuda_graph_update_executable` `:2639`; `ggml_cuda_graph_set_enabled` `:4518`; `graph_compute` `:4534-4588` (warmup `:4554-4570`, capture begin `:4585`); `ggml_backend_cuda_graph_optimize` `:4618-4860` (env `:4630-4638`, single-GPU guard `:4642`, fan-out `:4646-4697`, `attn_norm` filter `:4706-4708`, join `:4733-4751`, restore order `:4279-4283`, interleave `:4824-4858`); interface slots `:4877-4878` |
| `src/ggml/src/ggml-cuda/common.cuh` | `GGML_CUDA_MAX_STREAMS` `:178`; `ggml_cuda_graph` struct `:1272-1303` (`warmup_complete` `:1287`, `is_enabled()` `:1298-1301`); `ggml_cuda_concurrent_event` `:1305-1446` (`is_valid()` `:1346`); `ggml_cuda_stream_context` `:1448-1453`; stream creation `:1532` |
| `src/ggml/CMakeLists.txt`, `src/ggml/src/ggml-cuda/CMakeLists.txt`, `src/CMakeLists.txt` | `GGML_CUDA_GRAPHS` option `:212`; llama default ON `src/CMakeLists.txt:171-173`; `GGML_CUDA_USE_GRAPHS` define `CMakeLists.txt:149-150` |
| `src/ggml/src/ggml-backend.cpp`, `src/ggml/src/ggml-backend-impl.h` | `ggml_backend_graph_optimize()` `:559-563`, scheduler call `:1470`; interface field `:139` |
| `scripts/start_server_turbo.sh`, `scripts/run_cli.sh` | `GGML_CUDA_GRAPH_OPT` defaulted to `1` in both launch scripts |

## Known issues

- **No issue page tracks this subsystem**, which is itself a finding: the recorded behaviour is a limitation, not a defect. The one number the project has — ~25 µs constant versus a 16–64 ms budget — is filed here rather than as a bug.
- [[tq-1-missing-gemm-kernels]] — the cost that *does* dominate decode. Graph reuse cannot mask it, and the observed 15.75 → 19.43 ms growth is the same signature ([[source-state-md]] §1.1, §1.3).
- [[ta-2-budget-starvation]] — the other mechanism that makes attention cost grow with context: if eviction degenerates to a sliding window, the cache accounting that should bound this cost does not bound it.
- [[v100-sxm2]] — `sm_70` is the *floor* the graph path supports (`cc < GGML_CUDA_CC_VOLTA` disables it), so Volta is graph-capable; but the repo's own fat-binary scan found no arch markers for the CUDA 12.4 artifact, so whether the shipped Volta build exercises this code is open on that page, not here.
- Scope limits found in the code and worth stating plainly: the concurrency rewrite is single-GPU only (`:4642`), fan-out exactly 3 (`:4696-4697`), and rooted on a node whose name contains `attn_norm` (`:4708`). It is not a general multi-stream scheduler and will silently do nothing for any other graph shape.

## Open questions

- Is the ~25 µs figure a per-launch constant (so benefit ∝ number of graph launches, but never grows within a run) or a one-time amortisation? [[source-state-md]] §1.1 says "constant one-time saving"; the code shows a single `cudaGraphLaunch` per `graph_compute` call, which is consistent with either reading. `[UNVERIFIED]` — resolving it needs a profile.
- `GGML_CUDA_GRAPH_OPT=1` is read **once** into a function-local `static` (`:4630-4638`), so changing the env var mid-process has no effect. Not documented anywhere in the sources; noted here because it surprises.
- The graph cache is keyed on `nodes[0]` only (`:2593-2595`). Two different graphs sharing a first node would contend for one `ggml_cuda_graph`; the `node_props` comparison is what makes that safe, but the failure mode (warmup reset instead of a wrong replay) is not documented in the sources. `[UNVERIFIED]` — not exercised.
- Does the concurrent-stream path actually engage for the target model's attention? It requires a node literally named `*attn_norm*` with fan-out 3 and single-GPU execution; the archived Nsight data in [[source-state-md]] §1.3 does not report a stream-count or graph-launch metric, so there is no direct evidence of engagement here. `[UNVERIFIED]`.

## See also

[[overview]] · [[performance-profile]] · [[benchmarks]] · [[roadmap]] · [[v100-sxm2]] · [[tq-1-missing-gemm-kernels]] · [[speculative-decoding]] · [[kv-cache]] · [[source-state-md]] · [[source-readme]] · [[codebase-map]]
