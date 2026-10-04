---
title: "TQ-3: InnerQ device state is not multi-GPU safe"
type: issue
status: current
updated: 2026-09-28
sources: [state.md]
verified: [src/ggml/src/ggml-cuda/turbo-quant.cuh, src/ggml/src/ggml-cuda/turbo-innerq.cu, src/ggml/src/ggml-cuda/set-rows.cu, src/src/llama-kv-cache.cpp]
tags: [innerq, turboquant, multi-gpu, correctness]
---

## Symptom

In a multi-GPU (tensor-parallel / layer-split) deployment, InnerQ scales reach only one GPU; the others quantize with uncalibrated (zero) state. ([[source-state-md]], §4 TQ-3)

## Cause

**Verified: the declaration and the upload are exactly as described.** `src/ggml/src/ggml-cuda/turbo-quant.cuh:147-152` holds the device-side state as header-scope `static __device__` arrays:

```c
static __device__ float d_innerq_scale[INNERQ_MAX_CHANNELS];
static __device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS];
static __device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS];
static __device__ int   d_innerq_count;
static __device__ int   d_innerq_active;       // 0 = scales are identity, 1 = scales applied
static __device__ int   d_innerq_calibrating;  // 1 = accumulating K² stats
```

Two properties combine. First, `static` at namespace scope gives each including TU (including `set-rows.cu`, whose kernels are the only readers) its own internal-linkage copy. Second, a `__device__` symbol has one instantiation *per device context*, and `cudaMemcpyToSymbol` without a stream/context argument targets the **current** device only. The uploads in `turbo_innerq_init()` (`turbo-quant.cuh:160-186`) and `turbo_innerq_finalize()` (`:189-253`) are single, unlooped calls — `cudaMemcpyToSymbol(d_innerq_sq_accum, …)` / `(d_innerq_count, …)` / `(d_innerq_active, …)` / `(d_innerq_calibrating, …)` at lines 179-182, and `(d_innerq_calibrating, …)` / `(d_innerq_scale, …)` / `(d_innerq_scale_inv, …)` / `(d_innerq_active, …)` at lines 240-244. There is no `ggml_cuda_set_device()` and no per-device loop anywhere in the file. `cudaDeviceSynchronize()` at line 243 synchronizes every device but copies nothing to them.

Note the asymmetry that makes this silent: the **host** control variable `innerq_enabled` (`turbo-quant.cuh:154`) is one piece of host state shared by all devices, while `d_innerq_calibrating` / `d_innerq_active` / the scale arrays exist once per device. After finalization the host believes InnerQ is active (`innerq_enabled == 2`) while every device other than the one that ran the upload still has `d_innerq_active == 0`, so those devices skip `x[j] *= d_innerq_scale[j]` in `k_set_rows_turbo3` (`set-rows.cu:286-289`) without warning.

## Impact

On a split deployment the secondary devices silently skip InnerQ equalization (and, if they run the first `set_rows` before the current device, skip calibration accumulation entirely), producing a mismatch between the equalized and non-equalized K halves. Severity in state.md: **HIGH**. The multi-device consequence itself is reasoned from the code as read — no multi-GPU run was observed here [INFERENCE]; the single-device-only upload is verified directly.

## Location

- Path: `src/ggml/src/ggml-cuda/turbo-quant.cuh` (verified) — device arrays at lines 147-152; uploads at 179-182 (init) and 240-244 (finalize); `cudaDeviceSynchronize()` at 243.
- Path: `src/ggml/src/ggml-cuda/set-rows.cu` (verified) — device-state readers in `k_set_rows_turbo3` at 288-295 (`d_innerq_calibrating`, `d_innerq_sq_accum[j]` (289), `d_innerq_count` (290), `d_innerq_active` (294), `d_innerq_scale[j]` (295)); launchers call `turbo_innerq_check_finalize()` at 569/914/1129.
- Path: `src/src/llama-kv-cache.cpp` (verified) — per-model InnerQ scale tensor `turbo_innerq_scale_inv` created at line 378 and uploaded from `g_innerq_scale_inv_host` at 3121-3125.

## Status

Unresolved. Note the two-home split: the device arrays that the kernels actually read live in `turbo-quant.cuh:147-152` (internal linkage per TU, one instantiation per device context), while the published host mirror lives in `turbo-innerq.cu:5-11`; neither home is per-device, so the single-`cudaMemcpyToSymbol` upload in the header's helpers is still the only write path to the accelerators. Listed as action item "Refactor InnerQ State (TQ-2, TQ-3)" (§5.3 of [[source-state-md]]).

## Fix sketch

Per state.md §4 TQ-3 and §5.3: replace the "upload once to the current device" pattern with an explicit per-device initialization loop — iterate the devices known to `ggml_backend_cuda_context`, select each with `ggml_cuda_set_device()` before every `cudaMemcpyToSymbol(d_innerq_*, …)` call, and repeat for `d_innerq_calibrating` / `d_innerq_active`; or bind the arrays to the backend context instead of using free-standing device symbols. The host-side half of the refactor (moving the control variables out of the header) is [[tq-2-innerq-host-state]].

## See also

- [[innerq]] · [[turboquant]] · [[quantization]]
- [[tq-2-innerq-host-state]] — host-side duplication of the same state
- [[tq-6-innerq-race]] · [[tq-7-innerq-max-channels]]
- [[v100-sxm2]] — the target device; see also [[source-state-md]] §4 TQ-3
