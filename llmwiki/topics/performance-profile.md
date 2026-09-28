---
title: Performance profile
type: topic
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: []
tags: [profiling, nsight, performance]
---

# Performance profile

## Bottom line

Three findings from Nsight Systems and long agent runs, recorded in [[source-state-md]] §1:

1. **CUDA-graph reuse is a constant, not a compounding, saving.** It removes ~25 µs of launch overhead against a 16–64 ms per-step cost — a one-time win, not something that improves as it runs.
2. **Decode slows down as context grows, and graphs are not the cause.** 15.75 ms → 19.43 ms per decode step correlates with unbounded KV attention cost as the cache grows.
3. **The dominant cost is wrong for the product.** `magma_sgemmEx_kernel<float, __nv_bfloat16>` takes **38.81 % (157 ms)** of GPU time; TriAttention takes **0.91 %**. The KV-compression feature is not the bottleneck — the *matmul path feeding on compressed types* is.

The causal chain for (3) is mechanical and checkable: [[gemm-dispatch]] shows what `ggml_cuda_mul_mat` does with a type it has no kernel for, and [[tq-1-missing-gemm-kernels]] shows `turbo2_0`/`turbo3_0`/`turbo4_0` missing from both dispatch predicates, leaving a dequantize-then-BLAS path. On Volta there is no INT tensor core to make that path cheap ([[v100-sxm2]]).

## Evidence

| Observation | Value | Source |
| :--- | :--- | :--- |
| `magma_sgemmEx_kernel<float, __nv_bfloat16>` share of GPU time | 38.81 % (157 ms) | [[source-state-md]] §1.3 |
| TriAttention share of GPU time | 0.91 % | [[source-state-md]] §1.3 |
| Graph-reuse saving | ~25 µs vs 16–64 ms launch overhead | [[source-state-md]] §1.1 |
| Decode step, short → long context | 15.75 ms → 19.43 ms | [[source-state-md]] §1.1 |
| Decode throughput, cold → warmed | 40 → 69 tok/s observed; 31.6 → 38.1+ t/s effective, attributed to rising speculative acceptance | [[source-state-md]] §1.2 |
| Peak decode, RTX 3090 | 68.68 tok/s baseline / 68.04 turbo `t3+q8` / 67.68 turbo `t3+t2` | [[source-readme]] *Tests Results* |
| End-to-end 10-task agent time, RTX 3090 | 875.89 s → 632.07 s (1.39×) | [[source-readme]] *Tests Results* |

The `float, __nv_bfloat16` instantiation is itself a clue: the attention score computation is being handed `bf16` and a float accumulator, i.e. it has already left the compressed domain before it reaches BLAS.

## Open questions

- **Confounded variables.** Context growth simultaneously raises attention cost *and* exercises the eviction path; a broken WHT inversion ([[ta-1-wht-inversion-256]]) degrades *which* keys survive, which changes both quality and the acceptance rate. The 15.75 → 19.43 ms curve does not separate these.
- Has the profile been re-taken since the `Release/` WHT fix? Every number above predates a port that has not happened in this tree.
- No V100 profile exists in any source. `magma_sgemmEx` share on Volta is the number that actually matters and is unmeasured.
- Is TriAttention's 0.91 % measured under starvation ([[ta-2-budget-starvation]]), where the pruning work is executed for `B=0` and does nothing useful? If so it is a cost with no benefit.

## See also

[[overview]] · [[benchmarks]] · [[gemm-dispatch]] · [[tq-1-missing-gemm-kernels]] · [[cuda-graphs]] · [[speculative-decoding]] · [[triattention]] · [[v100-sxm2]]
