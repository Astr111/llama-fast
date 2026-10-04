---
title: Source — README.md
type: source
status: current
updated: 2026-09-28
sources: [README.md]
verified: []
tags: [release, documentation, benchmarks]
---

# Source — `README.md`

Release documentation for llama-fast: what the engine integrates, where it came from, how it is launched, and what it measured.

## Summary

A **llama.cpp** engine integrating four optimizations: PrismML ternary/sub-2-bit weight kernels (`PQ2_0`, `PTQ1_0`), TurboQuant KV-cache vector quantization (`turbo3_0`, `turbo4_0`, `turbo2_0`) with Polar WHT rotation, TriAttention KV pruning by trigonometric scoring, and CUDA concurrency/graph reuse (`GGML_CUDA_GRAPH_OPT=1`).

It documents a `Release/` layout — clean fork source, `calibration/`, `scripts/`, and prebuilt `build/cuda13` (universal, `sm_75`–`sm_120`) and `build/cuda12.4` (legacy, `sm_61`–`sm_86`, the Volta-capable one) trees — plus a manual CLI reference for KV-cache types and TriAttention pruning flags, four production command examples, and native build instructions.

## Key claims

| Claim | Where |
| :--- | :--- |
| Four integrated optimizations, with `turbo3`/`q8_0` described as the speed profile (eliminates 64 inverse-WHT kernels per token) | Top section; Quick Start guide |
| Upstream lineage: PrismML llama.cpp fork, `atomicmilkshake/llama-cpp-turboquant`, TriAttention paper arXiv:2604.04921 (Mao et al., MIT/NVIDIA/Zhejiang, April 2026), llama.cpp base | Original Projects & Research Citations |
| `build/cuda12.4` targets `sm_61;sm_70;sm_75;sm_80;sm_86`; `build/cuda13` targets `sm_75…sm_120` (no `sm_70`) | GPU Compatibility Matrix |
| TriAttention is driven by a `.triattention` calibration profile plus `--triattention-budget` (recommended 2048–4096) and `--triattention-window` (recommended 512) | CLI Argument Reference |
| Measured on **RTX 3090 24GB**: 875.89 s → 632.07 s (1.39×), VRAM 11 800 MiB → 8 122 MiB at 16K ctx, ~25 200 tok/GB in the max-compression profile, peak decode 68.68 / 68.04 / 67.68 tok/s | Tests Results |
| Native build example passes `-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100;120"` — **`sm_70` is absent** | Native Compilation from Source |

## Discrepancies worth holding

- The benchmark platform is an **RTX 3090 (Ampere, INT tensor cores)**; the deployment target recorded in [[source-state-md]] is a **Tesla V100 (Volta, no INT tensor cores)**. The published speedups do not transfer by assumption — see [[benchmarks]].
- The native build example omits `sm_70` even though `build/cuda12.4` ships it. Whether that is an oversight in the docs or a real capability gap is an open question in [[codebase-map]].

## Pages derived

[[source-state-md]] · [[overview]] · [[benchmarks]] · [[upstream-lineage]] · [[ternary-bonsai-2-27b]] · [[qwen3-dflash-draft]] · [[v100-sxm2]] · [[triattention]] · [[turboquant]] · [[cuda-graphs]] · [[kv-cache]] · [[roadmap]]

## Provenance

- raw path: `llm-wiki/raw/README.md`
- sha256: `05dfb403b9dc8661f9037ea00463bfc5199d14c02a9352f5a147e4660999f08b`
- ingested: 2026-09-28
