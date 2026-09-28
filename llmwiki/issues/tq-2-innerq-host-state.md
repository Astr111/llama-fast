---
title: "TQ-2: InnerQ host state duplicated per translation unit"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/ggml/src/ggml-cuda/turbo-quant.cuh, src/ggml/src/ggml-cuda/turbo-innerq.cuh, src/ggml/src/ggml-cuda/turbo-innerq.cu, src/ggml/src/ggml-cuda/set-rows.cu, src/src/llama-kv-cache.cpp]
tags: [innerq, turboquant, correctness]
---

## Symptom

state.md reports that InnerQ calibration can silently fail to activate depending on compilation/link order. ([[source-state-md]], §4 TQ-2)

## Cause

**Verified: the declaration is exactly as described.** `src/ggml/src/ggml-cuda/turbo-quant.cuh:154-157` declares four host-side variables at file scope with internal linkage:

```c
static int  innerq_enabled       = 0;  // host: 0=off, 1=calibrating, 2=active
static int  innerq_target_tokens = 0;
static float innerq_strength     = 0.5f;
static bool  innerq_initialized  = false;
```

and the host functions that own them — `turbo_innerq_init()` (`turbo-quant.cuh:160`), `turbo_innerq_finalize()` (`:189`), `turbo_innerq_check_finalize()` (`:256`), `turbo_innerq_is_active()` (`:291`) — are `static` too. The header is included by many translation units: directly by `turbo-wht.cu:1`, `triattention-score.cu:17`, `convert.cu:3`, `set-rows.cu:3`, `fattn-common.cuh:6` and `dequantize.cuh:3`, and transitively by everything that includes `dequantize.cuh` (`cpy.cu`, `getrows.cu`, `lightning-indexer.cu`, `convert.cu`, `triattention-score.cu`) and `fattn-common.cuh` (`fattn.cu`, `fattn-vec.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.cuh`). Each of those TUs gets its own private copy of the four variables and of the four functions.

**But the reported impact does not hold in this checkout.** The only callers of `turbo_innerq_check_finalize()` — which is what triggers `turbo_innerq_init()` and, at the token target, `turbo_innerq_finalize()` — are the three `set_rows_cuda_turbo{3,2,4}` launchers in `set-rows.cu:569`, `914` and `1129`. `turbo_innerq_is_active()` has no callers at all. Calibration state is therefore both produced and consumed inside a single TU (`set-rows.cu`), and cannot be "seen" by a different TU's copy. The cross-TU path that *does* carry InnerQ results has already been factored out: `turbo_innerq_finalize()` calls `turbo_innerq_publish()` (`turbo-quant.cuh:189-253`, call at its tail), defined in `turbo-innerq.cu:15-22`, which writes the true extern `g_innerq_finalized` / `g_innerq_scale_inv_host` declared in `turbo-innerq.cuh:10-11`. `src/src/llama-kv-cache.cpp:28-32` declares those two symbols plus `turbo_innerq_needs_tensor_update()` / `turbo_innerq_mark_tensor_updated()` for the CUDA build (and provides local stubs at lines 34-37 otherwise) and consumes them at `llama-kv-cache.cpp:3121-3125`.

> Correction (2026-09-28): state.md's impact clause — "Calibration triggered in one TU may never be visible to `turbo_innerq_check_finalize()` in another TU … may silently fail to activate depending on compilation/link order" — is not reachable here: the triggering and the observation both live in `set-rows.cu`, and the header's own comment (`turbo-innerq.cuh:3-5`) confirms the design intent, with the shared host state already living in `turbo-innerq.cu`. What remains true is the duplication itself (and `turbo-quant.cuh:5`'s "block size 32" header comment is likewise stale — see [[tq-5-tail-elements]]).

## Impact

Latent hazard, not an active bug: four host variables and four functions plus the device arrays are duplicated into every TU that includes the header, and any future caller of `turbo_innerq_is_active()`/`turbo_innerq_check_finalize()` from a TU other than `set-rows.cu` would silently read `innerq_enabled == 0` and observe InnerQ as disabled. Severity in state.md: **HIGH**.

## Location

- Path: `src/ggml/src/ggml-cuda/turbo-quant.cuh` (verified) — `static` host state at lines 154-157; `turbo_innerq_init()` 160, `turbo_innerq_finalize()` 189, `turbo_innerq_check_finalize()` 256, `turbo_innerq_is_active()` 291; `#include "turbo-innerq.cuh"` at line 12.
- Path: `src/ggml/src/ggml-cuda/turbo-innerq.cu` (verified) — `g_innerq_finalized` / `g_innerq_scale_inv_host` at lines 5-11, `turbo_innerq_publish()` 15-22, `turbo_innerq_needs_tensor_update()` 24-26, `turbo_innerq_mark_tensor_updated()` 28-30.
- Path: `src/ggml/src/ggml-cuda/turbo-innerq.cuh` (verified) — `extern` declarations at lines 10-11, function prototypes 14/18/21, header comment 3-5 stating host state lives in the `.cu`.
- Path: `src/ggml/src/ggml-cuda/set-rows.cu` (verified) — the only call sites, lines 569, 914, 1129.
- Path: `src/src/llama-kv-cache.cpp` (verified) — guarded extern-block at 28-37; consumer at 3121-3125.

## Status

**Not fixed, and not vestigial.** The tree currently holds *two* InnerQ state homes: the file-scope statics inside `turbo-quant.cuh` (lines 147-157) and the host-side publish mirror in the new module `turbo-innerq.cu` / `turbo-innerq.cuh` (831 bytes; `extern`s at `turbo-innerq.cuh:10-11`, definitions at `turbo-innerq.cu:5-11`). Only the second is externalized; the first is still the live calibration path, because `set-rows.cu:569/914/1129` call the header's `turbo_innerq_check_finalize()`, so this is live code, not dead code with a replacement. What the second module proves is that the cross-TU boundary can be crossed properly (that is how `llama-kv-cache.cpp:3121-3125` learns the scales) — it does not remove the duplicated header state.

Partly addressed relative to state.md's description: the cross-TU *result* path is externalized (`turbo-innerq.cu`), while the *control* state is still per-TU. Listed as action item "Refactor InnerQ State (TQ-2, TQ-3)" (§5.3 of [[source-state-md]]).

## Fix sketch

Per state.md §4 TQ-2 and §5.3: move `innerq_enabled`, `innerq_target_tokens`, `innerq_strength` and `innerq_initialized` — together with `turbo_innerq_init()` / `turbo_innerq_finalize()` / `turbo_innerq_check_finalize()` / `turbo_innerq_is_active()` — out of `turbo-quant.cuh` into `turbo-innerq.cu`, and replace them with `extern` declarations in `turbo-innerq.cuh`, mirroring the existing `g_innerq_finalized` pattern (lines 10-11). The device arrays (`turbo-quant.cuh:147-152`) are the companion change; see [[tq-3-innerq-multigpu]].

## See also

- [[innerq]] · [[turboquant]] · [[quantization]]
- [[tq-3-innerq-multigpu]] — the device-side half of the same refactor
- [[tq-6-innerq-race]] · [[tq-7-innerq-max-channels]] — the other InnerQ defects
- [[source-state-md]] §4 TQ-2, §5.3
