---
title: Overview
type: topic
status: current
updated: 2026-09-28
sources: [state.md, README.md, AGENTS.md, llmwiki.txt]
verified: []
tags: [synthesis]
---

# Overview — llama-fast

## Bottom line

`llama-fast` is a **llama.cpp fork that exists to make a 27B ternary-quantized model fit and run fast on a Tesla V100**, by shrinking the KV cache along three independent axes at once: **quantize it** ([[turboquant]] — `turbo3` keys, `q8_0` values), **evict from it** ([[triattention]] — scoring and dropping low-importance keys under a fixed budget), and **schedule it better** ([[cuda-graphs]] — concurrent streams and graph reuse). A draft model plus `draft-dflash` speculative decoding supplies the throughput on top.

> **Correction (2026-09-28).** The paragraph below was written before [[device-placement]] was read. It no longer holds: turbo-typed KV **forces flash attention**, the fused kernel dequantizes the blocks internally, and **no turbo `MUL_MAT` is built during decode** — so the cuBLAS/MAGMA fallback is not the mechanism behind the recorded 38.81 %, and `MAGMA` appears nowhere in this tree. What the project is actually stuck at is now a different question, and the most defensible candidate is the *combination* of the three confirmed eviction defects ([[ta-8-offset-max-zero-nan]], [[ta-9-rope-scope-mismatch]], [[ta-1-wht-inversion-256]]) plus the unmeasured nature of everything on the target hardware. See [[roadmap]] and [[open-questions]].

The architecture is sound and the published numbers are real, but the project is currently stuck at a **single bottleneck**: TurboQuant's KV types have **no native matmul kernels**, so `ggml_cuda_mul_mat` falls through to cuBLAS/MAGMA. On the V100 — which has no INT tensor cores to compensate — that fallback consumes **38.81 % (157 ms) of GPU time** while the TriAttention machinery the project is actually proud of costs 0.91 %. The KV-cache compression is working; the arithmetic that consumes the compressed cache is not.

Two further structural risks sit behind that one:

1. **The correctness-critical path is only fixed in a checkout outside this repository.** The WHT inversion bug that governs eviction quality at `head_dim=256` ([[ta-1-wht-inversion-256]]) is fixed in `Release/` and not ported here ([[source-state-md]] §2, §5).
2. **The published benchmarks were measured on an RTX 3090, not the V100 target** ([[source-readme]] *Tests Results*). Ampere has INT tensor cores and a different memory subsystem; the 1.39× and the 25 200 tok/GB figures are not evidence about Volta. See [[benchmarks]].

## Evidence

- Deployment target, optimizations, and the two defect inventories: [[source-state-md]].
- Four integrated optimizations, lineage, CLI surface, and measurements: [[source-readme]].
- Profitability of the whole approach rests on KV memory, which is the subject of [[kv-cache]] and [[kv-eviction]].
- The `verification` chain for the dominant cost is [[gemm-dispatch]] → [[tq-1-missing-gemm-kernels]] → [[performance-profile]].
- Repo boundaries and where the custom code lives: [[codebase-map]].

## Where the work stands

| Subsystem | State |
| :--- | :--- |
| WHT / rotation (`head_dim=256`) | broken in this checkout, fixed elsewhere — [[ta-1-wht-inversion-256]], [[ta-4-cooperative-fwht-race]], [[tq-4-wht-numerical-mismatch]] |
| TriAttention eviction | functional; starves to a sliding window on long prefixes — [[ta-2-budget-starvation]] |
| TurboQuant GEMM | missing kernels, MAGMA fallback, 38.8 % of GPU time — [[tq-1-missing-gemm-kernels]] |
| InnerQ calibration | unsafe state, single-GPU assumptions — [[tq-2-innerq-host-state]], [[tq-3-innerq-multigpu]], [[tq-6-innerq-race]], [[tq-7-innerq-max-channels]] |
| CPU fallback path | per-cell synchronous D2H transfers — [[ta-3-cpu-fallback-transfers]] |
| CUDA graphs / concurrency | works; saving is constant, not cumulative — [[cuda-graphs]] |
| Validation | no configuration guard rails — [[ta-7-config-validation]], [[ta-5-freq-scale-dead-code]] |

## Open questions

- Does `sm_70` actually build from source here? The README's native build example omits it while its own CUDA 12.4 release ships it — see [[codebase-map]].
- Which of the four checkouts in [[source-state-md]] §2 exists on this machine right now, and which one is the real source of truth for the custom kernels?
- Is the MAGMA fallback the *only* cause of the decode degradation recorded at growing context, or does eviction quality (a broken WHT inversion) contribute? The two are confounded in the current measurements.
- What does a V100-native (no INT tensor cores) kernel strategy look like for `turbo2/3/4`, and does it change the recommended `ctk`/`ctv` defaults?

## See also

[[performance-profile]] · [[roadmap]] · [[codebase-map]] · [[upstream-lineage]] · [[benchmarks]] · [[request-lifecycle]] · [[triattention]] · [[turboquant]] · [[walsh-hadamard-transform]] · [[v100-sxm2]] · [[gemm-dispatch]]
