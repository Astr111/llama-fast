---
title: Source — README.md
type: source
status: current
updated: 2026-10-06
sources: [README.md]
verified: []
tags: [release, documentation, benchmarks, pareto]
---

# Source — `README.md`

Release documentation for llama-fast: what the engine integrates, where it came from, how it is launched, and what it measured.

## Summary

A **llama.cpp** engine integrating four optimizations: PrismML ternary/sub-2-bit weight kernels (`PQ2_0`, `PTQ1_0`), TurboQuant KV-cache vector quantization (`turbo3_0`, `turbo4_0`, `turbo2_0`) with Polar WHT rotation, TriAttention KV pruning by trigonometric scoring, and CUDA concurrency/graph reuse (`GGML_CUDA_GRAPH_OPT=1`).

It documents release binaries, GPU matrices, manual CLI parameters, and optimal Pareto-frontier configurations on Tesla V100 (16GB) combining `Ternary-Bonsai-2-27B-PQ2_0` with `Qwen3.8-27B-DFlash2-Q4_K_M` reaching 75.92 tok/s.

## Key claims

| Claim | Where |
| :--- | :--- |
| Four integrated optimizations, with `turbo3`/`q8_0` described as the speed profile | Top section; Quick Start guide |
| Optimal Pareto configuration with `Q4_K_M` drafter reaches **75.92 tok/s** (+21% over Q8_0) with 100% logic accuracy and 100% 50k NIAH recall | Optimal Pareto-Front Configuration |
| Upstream lineage: PrismML llama.cpp fork, `atomicmilkshake/llama-cpp-turboquant`, TriAttention paper arXiv:2604.04921 (Mao et al., MIT/NVIDIA/Zhejiang, April 2026), llama.cpp base | Original Projects & Research Citations |
| TriAttention is driven by a `.triattention` calibration profile plus `--triattention-budget` (recommended 2048–4096) and `--triattention-window` (recommended 512–768) | CLI Argument Reference |
| Bounded deterministic reasoning via `--reasoning-budget 4096`, `--reasoning-budget-message` and medium effort | Production & Pareto configs |

## Provenance

- raw path: `llm-wiki/raw/README.md`
- sha256: `a57549448dd3b15a281b821c46dc8e53fcee8fad6c7275f745f0292f56d82fc1`
- ingested: 2026-10-06
