---
title: "TA-5: Dead code — freq_scale_sq always 1.0"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/src/llama-triattention.cpp]
tags: [triattention, correctness, dead-code]
---

## Symptom

Loss of scoring precision; the implementation diverges from the paper's formulas (paper Section 3.2, "frequency scaling factor"). ([[source-state.md]], §3 TA-5)

## Cause

`triattention_build_freq_scale_sq` (`src/src/llama-triattention.cpp:320`) computes `float c = cosf(omega[f] * 0.0f); float s = sinf(omega[f] * 0.0f);` (lines 325–326) and then `freq_scale_sq[f] = c * c + s * s` (line 327). Since the argument is always zero, `c=1.0, s=0.0` always, so `freq_scale_sq = 1.0` for every frequency — trigonometric scaling is effectively disabled. The in-code comment (lines 321–324) acknowledges this and defers real scaling to future YaRN support. The constant array is built at line 680 in `triattention_init` and consumed by the scoring paths (lines 487, 491, 511). ([[source-state.md]], §3 TA-5)

## Impact

Scoring precision loss; divergence from the paper's frequency-scaling formulas. Severity in state.md: **MEDIUM**. ([[source-state.md]])

## Location

- Path: `src/src/llama-triattention.cpp` (verified)
- Symbol: `triattention_build_freq_scale_sq` at line 320; zero-argument `cosf/sinf` at lines 325–326; call site in `triattention_init` at line 680.

## Status

Unresolved. Needs verification whether this is an intentional simplification or an initialization bug. ([[source-state.md]], §3 TA-5)

## Fix sketch

Determine intent against the paper's Section 3.2 formula: if frequency scaling is required, replace the position-0 evaluation with the actual frequency-dependent scaling factors (e.g. YaRN scaling) instead of `cosf(omega[f] * 0.0f)`; if intentional, replace the dead trig with a constant `1.0f` and document the simplification.

## See also

- [[triattention]]
- [[ta-7-config-validation]] — same `triattention_init` path
