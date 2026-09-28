---
title: Benchmarks
type: topic
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: []
tags: [benchmarks, evaluation]
---

# Benchmarks

## Bottom line

The project is measured by two different, non-overlapping suites, and **neither was run on the deployment hardware**.

- **[[source-readme]]** reports a 10-task agent benchmark on an **RTX 3090 (Ampere, 24 GB)** with the Pi Agent: 875.89 s baseline → 632.07 s in the `turbo3`+`q8_0` speed profile (**1.39×**), and VRAM 11 800 MiB → 8 122 MiB at 16K context, ~20 000–25 200 tokens per GB, with peak decode essentially flat (68.68 → 68.04 tok/s).
- **[[source-state-md]]** names the evaluation *harnesses* — Terminal-Bench 2.0 (MR Subset 39), Harbor 0.1.x, Nsight Systems — but records no results from them, only profiling conclusions.

The honest reading: compression and end-to-end agent speed are supported by evidence on Ampere; **the V100 target has no published benchmark at all**, and the peak-decode column shows the speed profile buys no raw decode throughput over baseline — its win is the 1.39× on multi-task agent time, i.e. context retention, not tokens per second.

## Evidence

| Metric | Baseline (FP16 KV) | `t3+q8_0` | `t3+turbo2` |
| :--- | :---: | :---: | :---: |
| 10-task total time | 875.89 s | **632.07 s** | 846.83 s |
| Speedup | 1.00× | **1.39×** | 1.03× |
| VRAM @ 16K ctx | ~11 800 MiB | 8 122 MiB | **7 914 MiB** |
| Tokens / 1 GB VRAM | ~4 000 | ~20 000 | **~25 200** |
| Peak decode | **68.68 tok/s** | 68.04 tok/s | 67.68 tok/s |

Platform: NVIDIA GeForce RTX 3090 (24 GB), Pi Agent, `Ternary-Bonsai-2-27B-PQ2_0`. Source: [[source-readme]] *Tests Results*.

Note the inversion in the last row: **the compression profiles are slower at peak decode** (−0.9 % and −1.5 %) and win only on the aggregate task time — consistent with [[performance-profile]]'s finding that the per-step cost is dominated by the matmul fallback, not by KV reading.

## Open questions

- **No V100 numbers exist.** Every figure above is Ampere, which has INT tensor cores the target lacks ([[v100-sxm2]]). Until a Volta run is recorded, the README's claims are unvalidated for the deployment target.
- The two profiles are **within 3 % of each other on all three end-to-end metrics** while differing ~25 % in VRAM. Whether `t3+q8_0` is genuinely the "speed profile" on Volta (where inverse-WHT elimination has different relative cost) is untested.
- Terminal-Bench 2.0 / Harbor results are named but not reported anywhere in the sources; no baseline or threshold is recorded.
- The 10-task figure has no variance, no repeats, and no seed — a single run, so the 1.03× of the max-compression profile is inside any plausible noise band.

## See also

[[overview]] · [[performance-profile]] · [[v100-sxm2]] · [[source-readme]] · [[source-state-md]] · [[roadmap]]
