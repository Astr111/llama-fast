---
title: "TA-9: the scoring kernel inverts RoPE over 256 dims while the model rotates 64"
type: issue
status: current
updated: 2026-10-04
sources: [state.md, TRIATTENTION.md]
verified: [src/src/llama-triattention.cpp, src/src/llama-triattention.h, src/src/llama-kv-cache.cpp, src/src/models/qwen35.cpp, src/src/llama-model.cpp, src/src/llama-model.h, src/src/turbo-rotation-data.h, src/ggml/include/ggml-cuda.h, src/ggml/src/ggml-cuda/triattention-score.cu, src/docs/development/HOWTO-add-model.md, src/docs/TRIATTENTION.md]
tags: [triattention, rope, correctness]
---

# TA-9: RoPE inverse applied over the wrong dimension range

> **Not in either inventory.** Found 2026-09-28 while resolving the open question called H2 on [[scoring-correctness]]; confirmed from code, never executed.

## Symptom

TriAttention's scoring asks how much attention a cached key would receive from future queries. It restores the key's pre-RoPE form before measuring. On this model it restores the wrong thing: the inverse rotation is applied to a different set of dimensions, and with a different frequency exponent, than the forward rotation used when the key was written.

## Cause

The forward and inverse rotations disagree on two parameters — **how many dimensions rotate** and **the frequency exponent**:

| | Dimensions rotated | Exponent |
| :--- | :--- | :--- |
| **Model** (forward) | `n_rot = 64` of 256 (`hparams.n_rot()` from `LLM_KV_ROPE_DIMENSION_COUNT`, `src/src/llama-model.h:862`, `src/src/llama-model.cpp:1486`), applied by `ggml_rope_multi` with the GGUF's MRoPE sections (`src/src/models/qwen35.cpp:6`, `:368-377`) | θ^(−2f/**64**) |
| **Scorer** (inverse) | **all 256**, pairwise `(f, f+128)`, with no `n_rot` or sections parameter (`src/src/llama-triattention.cpp:382`) | θ^(−2f/**head_dim**) = θ^(−2f/256) over `freq_count = head_dim/2 = 128` (`:309-312`, `:676-677`) |

The upstream contract confirms that dimensions from `n_dims` to the end are copied through unrotated (`src/docs/development/HOWTO-add-model.md:169-172`), so only `[0, 64)` of the 256 dimensions carry rotation — while the scorer rotates all 256. The CUDA kernel implements the same wrong pairing (`src/ggml/src/ggml-cuda/triattention-score.cu:9-10`, `:230-260`, `:296`).

The angle error is not a constant offset: the two exponents differ, so a given frequency `f` is off by a factor of θ^(3f/128). The mismatch therefore grows with `f` — which is exactly the quantity the scoring formula weights by frequency band, so the error is structured, not noise.

**Not compensated anywhere.** `src/src/turbo-rotation-data.h` contains no omega, `freq_count`, `head_dim` or rope symbol (`grep`) — the scorer's omega comes from `build_omega` alone. Nor do the two errors cancel across the calibration boundary: the calibration tool pairs frequencies the same way, but the captured `q` statistic is post-RoPE and left untouched, while `k` is pushed through a map that does not match it.

## Impact

**HIGH (new).** The scorer compares a post-RoPE query against a key restored with the wrong inverse, so the measured "would this token be attended to?" differs from reality by a frequency-dependent rotation. TriAttention's selection is systematically wrong even when it is not NaN ([[ta-8-offset-max-zero-nan]]) and even with a correct WHT inversion ([[ta-1-wht-inversion-256]]). The "it only needs *a* basis" defence does **not** apply: the same linear map is required on both operands, and the distance-horizon semantics the score is built on depend on the exact angles.

## Location

- `src/src/llama-triattention.cpp:309-312` — omega exponent uses `head_dim`, not `n_rot`
- `src/src/llama-triattention.cpp:676-677` — `freq_count = head_dim / 2`
- `src/src/llama-triattention.cpp:382` — `triattention_invert_rope`, pairwise over all dimensions, no `n_rot`/sections
- `src/ggml/src/ggml-cuda/triattention-score.cu:9-10`, `:230-260`, `:296` — the same pairing on the GPU
- `src/src/models/qwen35.cpp:368-377` — the forward rotation that sets the true geometry

## Status

**RESOLVED (2026-10-04)**. Parameterized TriAttention with dynamic `n_rot`:
1. In `src/src/llama-kv-cache.cpp`, pass `hparams.n_rot(0)` to `triattention_init` (fallback to `head_dim` if 0).
2. In `src/src/llama-triattention.h` & `src/src/llama-triattention.cpp`, added `n_rot` to `triattention_state` and `triattention_init`.
3. In `triattention_build_omega`, compute `omega[f] = rope_theta^(-2f / n_rot)` for `f < n_rot / 2`, and 0.0f for `f >= n_rot / 2`.
4. In `triattention_invert_rope` (CPU) and `triattention_score_kernel` (GPU), only invert RoPE for `f < n_rot / 2`, copying channels `f >= n_rot / 2` as-is.
5. In `src/ggml/include/ggml-cuda.h` & `src/ggml/src/ggml-cuda/triattention-score.cu`, added `n_rot` to `triattention_gpu_config` and dispatched it to the scoring kernel.

## Fix sketch

Make the inverse a parameterised function of the forward geometry rather than a fixed 128-pair butterfly: pass `n_rot` and the RoPE section layout into `triattention_invert_rope` and its CUDA twin, build omega from `n_rot` rather than `head_dim`, and rotate only the dimensions the model actually rotated. Note the partial-RoPE complication — sections `[11, 11, 10, 0]` mean the rotating 64 dimensions are not a contiguous prefix pair-set, so the correct inverse is the model's own `ggml_rope_multi` geometry inverted, not a new convention. Cross-check against the reference implementation in the paper/design doc before changing anything: [[source-triattention]] describes the intended statistic.

## See also

[[scoring-correctness]] · [[triattention]] · [[qwen35-architecture]] · [[walsh-hadamard-transform]] · [[forward-pass]] · [[ta-1-wht-inversion-256]] · [[ta-8-offset-max-zero-nan]] · [[ta-5-freq-scale-dead-code]] · [[roadmap]]
