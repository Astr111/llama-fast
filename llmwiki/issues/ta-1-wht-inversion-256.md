---
title: "TA-1: WHT inversion bug for head_dim=256"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/ggml/src/ggml-cuda/triattention-score.cu]
tags: [triattention, kv-eviction, walsh-hadamard-transform, correctness]
---

## Symptom

Severe generation quality degradation: the KV cache is scored using non-inverted keys, so eviction decisions are effectively random. ([[source-state-md]], §3 TA-1)

## Cause

The GPU scoring kernel skips the Walsh-Hadamard Transform inversion for models with `head_dim=256`. The guard is `if (padded_hd == 128 && f < 64)` at `src/ggml/src/ggml-cuda/triattention-score.cu:225` inside the `NEED_WHT_INV` block — when `padded_hd` is 256 the condition is false and the inversion body stays empty. The surrounding loop (`for (uint32_t b = 0; b < padded_hd; b += 128)`, line 213) remaps threads per 128-element block but performs no rotation; the comment at lines 219–221 explicitly acknowledges "we need to handle the case where padded_hd > 128" and only the single-block case calls `inverse_wht_rotation_128` (defined at line 72). Scoring then proceeds on keys still carrying the TurboQuant WHT rotation. ([[source-state-md]], §3 TA-1)

## Impact

Random eviction and severe generation quality degradation on any model with `head_dim=256` — including the primary target `Ternary-Bonsai-2-27B-PQ2_0.gguf` (`head_dim=256`). Severity in state.md: **CRITICAL**. ([[source-state-md]])

## Location

- Path: `src/ggml/src/ggml-cuda/triattention-score.cu` (verified)
- Symbols: `triattention_score_kernel` region — `NEED_WHT_INV` block, `if (padded_hd == 128 && f < 64)` at line 225; `inverse_wht_rotation_128` at line 72; `cooperative_fwht_128` at line 47.

## Status

state.md claims a fix exists in `/home/ms/llama-fast/Release/` using a dynamic `wht_group` calculation, not yet ported to this repository. **That path does not exist on this machine** (`/home/ms/llama-fast/` contains no `Release/` directory — checked 2026-09-28), so the claim is unverifiable here. In this repository the bug is present and unfixed. Not yet ported per state.md; see also action item "Port WHT Fix" (§5.1 of [[source-state-md]]).

## Fix sketch

Per state.md §3 TA-1 and §5.1: replace the hardcoded `padded_hd == 128` single-block guard with a dynamic `wht_group` calculation, so `inverse_wht_rotation_128` is applied to every 128-element block within the `b` loop when `padded_hd > 128`. The multi-block scaffolding already exists in the loop at `triattention-score.cu:213`.

## See also

- [[triattention]] · [[kv-eviction]] · [[walsh-hadamard-transform]]
- [[ta-4-cooperative-fwht-race]] — same kernel, same 64-thread WHT helpers
- [[tq-4-wht-numerical-mismatch]] — WHT forward/inverse implementation divergence on the TurboQuant side
