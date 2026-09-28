---
title: Source — state.md
type: source
status: current
updated: 2026-09-28
sources: [state.md]
verified: []
tags: [profiling, issues, roadmap]
---

# Source — `state.md`

The user's working state document: the profiling conclusions, the checkouts in play, and the defect inventory for TriAttention and TurboQuant. This is the wiki's densest single source — most of the `issues/` vault derives from it.

## Summary

Records a deployment target of **Tesla V100-SXM2-16GB** (Volta `sm_70`, 900 GB/s HBM2, **no INT tensor cores**) on Ubuntu 22.04 / CUDA 12.4, serving `Ternary-Bonsai-2-27B-PQ2_0.gguf` with `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` as draft model, using `draft-dflash` speculative decoding (max 5), TurboQuant `turbo3` keys + `q8_0` values, TriAttention eviction (budget 4096, window 512), and CUDA graphs.

Three profiling conclusions: CUDA-graph reuse is a **constant** one-time saving (~25 µs vs 16–64 ms launch overhead), not cumulative; decode degradation 15.75 ms → 19.43 ms tracks growing KV attention cost, not reuse count; and `magma_sgemmEx_kernel` consumes **38.81 % (157 ms)** of GPU time while TriAttention consumes only 0.91 %, because TurboQuant types have no native GEMM kernels.

Then two defect inventories: **TA-1…TA-7** (TriAttention, from a CRITICAL WHT-inversion bug at `head_dim=256` to missing config validation) and **TQ-1…TQ-7** (TurboQuant, from the missing native GEMM kernels to a hardcoded 128-channel InnerQ limit).

## Key claims

| Claim | Where |
| :--- | :--- |
| Four codebases in play: publication repo (this one), a broken working copy, a fixed `Release/` copy, and a TurboQuant fork — all outside this repository | §2 Active Codebases |
| TA-1 fix exists in `Release/` and **is not ported here** | §3 TA-1, §5 item 1 |
| TriAttention budget starvation at `prefix_length + divide_length >= budget` collapses eviction to a sliding window | §3 TA-2 |
| CPU fallback dequant does one synchronous D2H copy **per KV cell** (~4096 copies for `n_decode=4096`) | §3 TA-3 |
| `freq_scale_sq` is computed via `cosf/sinf(ω·0)` → always 1.0, disabling trigonometric scaling | §3 TA-5 |
| TurboQuant types are absent from `ggml_cuda_should_use_mmq()`/`mmvq()`, forcing a cuBLAS/MAGMA fallback | §4 TQ-1 |
| Host-side InnerQ state is file-scope `static` in a header included by several translation units; device state is `static __device__` and breaks on multi-GPU | §4 TQ-2, TQ-3 |
| Six pending action items, in priority order | §5 |

## Discrepancies worth holding

- §2 names four checkouts by absolute path; none of them is inside this repository. Only the paths themselves are claimed here — that they hold the described code is `[UNVERIFIED]` until the wiki reads them, and the deliverable fix for TA-1 lives in one of them.
- Hub dims and channel limits from this document (`head_dim=256`, InnerQ max 128) are the hinge for three separate issues; [[walsh-hadamard-transform]] holds the mechanism.

## Pages derived

[[overview]] · [[performance-profile]] · [[roadmap]] · [[benchmarks]]
[[triattention]] · [[turboquant]] · [[innerq]] · [[walsh-hadamard-transform]] · [[v100-sxm2]] · [[ternary-bonsai-2-27b]] · [[qwen3-dflash-draft]] · [[speculative-decoding]] · [[cuda-graphs]]
[[ta-1-wht-inversion-256]] · [[ta-2-budget-starvation]] · [[ta-3-cpu-fallback-transfers]] · [[ta-4-cooperative-fwht-race]] · [[ta-5-freq-scale-dead-code]] · [[ta-6-overlap-double-counting]] · [[ta-7-config-validation]]
[[tq-1-missing-gemm-kernels]] · [[tq-2-innerq-host-state]] · [[tq-3-innerq-multigpu]] · [[tq-4-wht-numerical-mismatch]] · [[tq-5-tail-elements]] · [[tq-6-innerq-race]] · [[tq-7-innerq-max-channels]]

## Provenance

- raw path: `llmwiki/raw/state.md`
- sha256: `a5b15a9a7d973d1fa7e3feaa82cdc3003f5fbde59f68ffccd0ece0bb97fb40d0`
- ingested: 2026-09-28
