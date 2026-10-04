---
title: "TA-11: the shipped calibration profile records post-RoPE Q, which the scorer cannot use"
type: issue
status: current
updated: 2026-09-28
sources: [state.md, TRIATTENTION.md, TRIATTENTION-API.md]
verified: [src/tools/triattention-calibrate/triattention-calibrate.cpp, src/src/models/qwen35.cpp, src/src/llama-context.cpp, src/src/llama-triattention.cpp, src/src/llama-triattention.h, src/src/llama-graph.cpp, calibration/bonsai-27b.triattention]
tags: [triattention, calibration, correctness, ship-blocker]
---

# TA-11: the calibration captures a basis the scorer does not consume

> **Not in either inventory.** Found 2026-09-28 while resolving what was ranked **#1** in [[open-questions]]; confirmed from code, never executed. It sits upstream of every eviction decision the shipped profile informs.

## Symptom

`calibration/bonsai-27b.triattention` was produced from a statistic the scorer cannot use. TriAttention's score has a phase term built from the angle between a query direction and a key; the profile's query direction is measured **after** RoPE, while the keys it will later be compared against are deliberately restored **before** RoPE.

## Cause

Three reads settle it:

1. **The collector** matches one exact name pattern — `"Qcur-" + <layer>` (`src/tools/triattention-calibrate/triattention-calibrate.cpp:42-52`) against names formatted `%s-%d` (`src/src/llama-context.cpp:2795`).
2. **In the `qwen35` graph**, exactly one tensor carries that name: `cb(Qcur, "Qcur", il)` at `src/src/models/qwen35.cpp:379` — and that is the **return value of `ggml_rope_multi`** (`:367-371`). The pre-RoPE tensors are named `Qcur_full` (`:335`), `Qcur_reshaped` (`:340`) and `Qcur_normed` (`:344`); none contains the substring `Qcur-`. The MTP builder uses `mtp_Qcur_full` / `mtp_Qcur_normed` (`:687`, `:695`). So no pre-RoPE view is captured under that name in any `qwen35` variant path.
3. **The `Qcur` name is reused pre-RoPE only in the generic graph builder** (`src/src/llama-graph.cpp:1742`, `:1776`) — which this architecture does not use for the attention path.

The scorer, meanwhile, is built on pre-RoPE keys by design: `pre_rope_k` is the output of `triattention_invert_rope` (`src/src/llama-triattention.cpp:382`, used at `:444-445`), and the phase comes from `phi = atan2(E[q_f] · conj(k_f))` (`:480-482`) feeding `cos(omega·delta + phi)` (`:484-490`).

Mixing the two bases corrupts the score in a way that does not cancel:

- the **phase** term is position-dependent and therefore scattered by the mean over offsets, not merely shifted;
- the **norm** term is understated, because only `E[‖q_f‖]` is basis-invariant — the mean of a rotated vector has smaller magnitude than the mean of its norms.

This is **independent of [[ta-9-rope-scope-mismatch]]**, which is about the geometry of the inverse map (256 dimensions at θ^(−2f/256) versus the model's 64 rotating dimensions). Neither fix cures the other: TA-9 corrects *how* the key is rotated back, TA-11 corrects *what the query was measured in*.

## Impact

**CRITICAL (new).** Every score computed from the shipped profile is wrong in a position-dependent way — the input data is invalid, not the arithmetic. It also means the profile cannot be regenerated correctly until the capture hook is moved: any new `.triattention` file produced by the same tool on this architecture inherits the defect.

In the shipped configuration it is **masked** by [[ta-8-offset-max-zero-nan]]: with `agg=mean` and zero offsets every score is NaN regardless, so this defect cannot be observed until TA-8 is fixed. That makes TA-8 the gate for investigating the other two, and it makes all three part of one repair.

## Location

- `src/tools/triattention-calibrate/triattention-calibrate.cpp:42-52` — the name matcher
- `src/src/models/qwen35.cpp:335`, `:340`, `:344`, `:367-371`, `:379` — the tensor names and the one that is post-RoPE
- `src/src/llama-context.cpp:2795` — the `%s-%d` naming convention
- `src/src/llama-graph.cpp:1742`, `:1776` — where `Qcur` *is* pre-RoPE (generic builder, unused here)
- `src/src/llama-triattention.cpp:382`, `:444-445`, `:480-490` — the scorer's pre-RoPE key and its phase/norm terms
- `calibration/bonsai-27b.triattention` — the affected artifact (789 571 B; 384 records = 16 layers × 24 heads, consistent with this model's full-attention layers)

## Status

**Open, unlisted, unfixed.** Decided by reading; nothing was built or run ([[build-and-verify]]). The magnitude of the corruption is unmeasured — how much the scattered phase and understated norm change the *kept* token set is not quantified anywhere.

## Fix sketch

Move the capture to a pre-RoPE view: either register the callback against a tensor the graph names pre-RoPE (the generic builder's `Qcur` is exactly that, so the fix is to make `qwen35` emit the same name for that tensor), or extend the collector to accept the architecture's pre-RoPE names explicitly. Then regenerate the profile and compare kept-token sets against the current one — the diff is the defect's magnitude. Cross-check the intended statistic against the project's own description before changing anything: [[source-triattention]] is the statement of what `E[q_f]` is supposed to be.

## See also

[[triattention-calibrate]] · [[scoring-correctness]] · [[triattention]] · [[ta-8-offset-max-zero-nan]] · [[ta-9-rope-scope-mismatch]] · [[qwen35-architecture]] · [[open-questions]]
