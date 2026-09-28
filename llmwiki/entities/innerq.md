---
title: InnerQ
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified:
  - src/ggml/src/ggml-cuda/turbo-quant.cuh
  - src/ggml/src/ggml-cuda/turbo-innerq.cu
  - src/ggml/src/ggml-cuda/turbo-innerq.cuh
  - src/ggml/src/ggml-cuda/set-rows.cu
  - src/ggml/src/ggml-cuda/turbo-wht.cu
  - src/ggml/src/ggml-cuda/turbo-wht.cuh
  - src/ggml/src/ggml-cuda/convert.cu
  - src/ggml/src/ggml-cuda/dequantize.cuh
  - src/ggml/src/ggml-cuda/fattn-common.cuh
  - src/ggml/src/ggml-cuda/triattention-score.cu
  - src/ggml/src/ggml-cuda/ggml-cuda.cu
  - src/src/llama-kv-cache.cpp
  - src/src/llama-kv-cache.h
  - src/src/llama-graph.cpp
  - src/src/llama-memory-hybrid.cpp
  - src/src/llama-memory.h
  - src/common/arg.cpp
  - src/common/common.h
tags: [innerq, turboquant, quantization, kv-cache, cuda]
---

# InnerQ

## What it is

**InnerQ is the per-channel equalization stage inside [[turboquant]]'s encoder.** Before a group of K (or V) channels is rotated and quantized, InnerQ multiplies each channel by its own scale so that the channels' variances become comparable; the query side multiplies by the reciprocal, which leaves every dot product unchanged. It is not a separate ggml type and has no CLI flag — it is an *additional* first step of the `SET_ROWS` write path, enabled or disabled by environment variables.

The in-code statement of the invariant is one line: `Math: <Q/s, s*K> = <Q, K> preserves dot products` (`src/ggml/src/ggml-cuda/turbo-quant.cuh:143`). The scale is applied to K and V at encode time (`x[j] *= d_innerq_scale[j]`, `set-rows.cu:294-295`, `:663-664`, `:1006-1007`) and its reciprocal is applied on the query pre-rotation and on the value un-rotation (`turbo-wht.cu:49-53`, `:90-93`).

Why it helps: the centroid codebooks are Lloyd-Max tables tuned for a fixed variance (`TURBO_CENTROIDS_2BIT`/`_3BIT`/`_4BIT`, `turbo-quant.cuh:23-25`, `:33-36`, `:297-302`, all documented as `N(0, 1/128)`), and the encoder normalises each group by its L2 norm before rotating. Without equalization, a group whose energy sits in a few high-variance channels rotates into a vector that is still far from isotropic, so the same codebook spends most of its levels on the distribution's tail. Equalizing channel RMS *before* the rotation pushes the rotated vector towards the isotropic distribution the codebooks assume. [INFERENCE] — the mechanism is stated in the header comment and follows from the codebook constants; no measurement of the quality delta exists in this tree.

## How it works

### The four-stage lifecycle

| Stage | Where | What happens |
| :--- | :--- | :--- |
| **init** | `turbo_innerq_init()`, `turbo-quant.cuh:160-187` | reads `TURBO_INNERQ` (token/step budget) and `TURBO_INNERQ_STRENGTH` (default `0.5f`, clamped to `(0,1]`, `:173-174`); zeroes `d_innerq_sq_accum`, `d_innerq_count`, `d_innerq_active` and sets `d_innerq_calibrating = 1` via four `cudaMemcpyToSymbol` calls (`:178-182`); `innerq_enabled = 1` |
| **accumulate** | inside all three encoders, `set-rows.cu:287-296`, `:656-665`, `:999-1008` | per group, per channel `j = threadIdx.x`: `atomicAdd(&d_innerq_sq_accum[j], x[j]*x[j])`, and one `atomicAdd(&d_innerq_count, 1)` per block — on the **unscaled** inputs, taken before the scale is applied |
| **finalize** | `turbo_innerq_check_finalize()` `:256-289` → `turbo_innerq_finalize()` `:189-253` | gated on `d_innerq_count >= innerq_target_tokens`; computes per-channel `rms[i] = sqrt(sq_accum[i]/count)`, then `scale[i] = clamp(powf(mean_rms/rms[i], strength), 0.5, 2.0)` and `scale_inv[i] = 1/scale[i]` (`:211-223`); **auto-disables** if the channels are already balanced (`max_ratio < 1.2` and `min_ratio > 1/1.2`, `:228-237`); otherwise uploads `scale`/`scale_inv`, `cudaDeviceSynchronize()`, sets `d_innerq_active = 1`, `innerq_enabled = 2`, and calls `turbo_innerq_publish(scale_inv, group_size)` (`:238-250`) |
| **apply** | encoders `set-rows.cu:294-295` (`if (d_innerq_active) x[j] *= d_innerq_scale[j];`), rotation op `turbo-wht.cu:49-53` / `:90-93` | the encoder scales K and V; the graph's `GGML_OP_TURBO_WHT` scales Q (forward direction, *before* signs+WHT) and the attention output (inverse direction, *after* WHT+signs) by the published `scale_inv` |

The counter and the accumulator are shared by **every** turbo `SET_ROWS` — K writes and V writes alike, whichever of turbo2/turbo3/turbo4 — because they all live in the same translation unit (see *Where it lives*). `d_innerq_count` is incremented once per `(row, group)` pair, so it counts group-writes rather than tokens: for the target model's K cache (`n_embd_k_gqa = 4 × 256`, group size 128) the kernel grid is 8 groups per row (`set-rows.cu:263-264`, `:632-633`), i.e. eight increments per token written.

### The knobs

There is **no CLI flag** for InnerQ — `src/common/arg.cpp` and `src/common/common.h` contain no occurrence of `innerq`. The two knobs are environment variables read once per translation unit:

| Variable | Read at | Meaning |
| :--- | :--- | :--- |
| `TURBO_INNERQ=N` | `turbo-quant.cuh:164-170` | `N > 0` starts calibration; `N` is the target for `d_innerq_count` before finalize. Unset or `<= 0` leaves InnerQ off |
| `TURBO_INNERQ_STRENGTH=S` | `:174` | exponent in `pow(mean_rms/rms[i], S)`; default `0.5f`; values outside `(0,1]` are replaced by the default |

Neither variable is exported by `scripts/start_server_turbo.sh` or `scripts/run_cli.sh`, so InnerQ is **off** in the shipped launch surface.

### The identity state when inactive

Three separate mechanisms keep InnerQ a no-op when it is not active, and none of them relies on identity *values* in the device arrays:

1. `d_innerq_active` is zero-initialised and only set to `1` by `turbo_innerq_finalize()` (`:244`), so the encoder's `if (d_innerq_active)` guard (`set-rows.cu:294`) skips the multiply. The device arrays `d_innerq_scale` / `d_innerq_scale_inv` are left at their BSS zeros until then and are never read while inactive.
2. The host mirror `g_innerq_scale_inv_host` is initialised to all `1.0f` (`turbo-innerq.cu:6-11`), and the model-side tensor `turbo_innerq_scale_inv` is filled with ones both when the KV buffer is created (`llama-kv-cache.cpp:432-437`) and when it is re-created after a clear (`:541-546`). That tensor is what the rotation op reads as `src[1]`, and `k_turbo_wht_f32` multiplies by `scale_inv[t % group_size]` whenever the pointer is non-null (`turbo-wht.cu:49-53`, `:90-93`) — so on the graph side "inactive" is literally a multiplication by `1.0`.
3. `turbo_innerq_init()` returns before touching any device symbol when the env var is absent (`turbo-quant.cuh:165-168`), and `turbo_innerq_check_finalize()` returns immediately when `innerq_enabled == 0` (`:259-260`).

The tensor's pointer is only non-null for a cache whose K type is a turbo type: it is created in the same `if` that creates the rotation matrices, `if (turbo_rotation == nullptr && (type_k == TURBO3_0 || type_k == TURBO4_0 || type_k == TURBO2_0))` (`llama-kv-cache.cpp:370-379`). A turbo *V* cache with a non-turbo K type therefore gets encoder-side InnerQ gating but no tensor to carry `scale_inv` into the rotation op — the graph passes `nullptr` and the V un-rotation skips the correction (`llama-graph.cpp:2706`, `:2784`). `[INFERENCE]` on the quality consequence; the gating condition is read directly from the two sites.

## Where it lives

### Two state homes, and which one is live

This is the page's central finding. The tree contains **two** InnerQ state homes, and the older one is still the live one.

### Home A — `turbo-quant.cuh` (the live calibration and apply path)

```c
// src/ggml/src/ggml-cuda/turbo-quant.cuh:141-157
static __device__ float d_innerq_scale[INNERQ_MAX_CHANNELS];       // :147
static __device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS];   // :148
static __device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS];    // :149
static __device__ int   d_innerq_count;                            // :150
static __device__ int   d_innerq_active;                           // :151
static __device__ int   d_innerq_calibrating;                      // :152

static int  innerq_enabled       = 0;                              // :154
static int  innerq_target_tokens = 0;                              // :155
static float innerq_strength     = 0.5f;                           // :156
static bool  innerq_initialized  = false;                          // :157
```

with the four helpers `turbo_innerq_init()` `:160`, `turbo_innerq_finalize()` `:189`, `turbo_innerq_check_finalize()` `:256` and `turbo_innerq_is_active()` `:291` — all `static` (internal linkage) in the header.

- **Are they reachable?** Yes, from every translation unit that includes `turbo-quant.cuh`; each such TU gets a private copy of the six device arrays and the four host helpers. Direct includers are `set-rows.cu:3`, `convert.cu:3`, `turbo-wht.cu:1`, `triattention-score.cu:17`, `dequantize.cuh:3` (and hence `cpy.cu:2`, `getrows.cu:2`, `convert.cu:2`, `triattention-score.cu:16`) and `fattn-common.cuh:6` (and hence `fattn.cu`, `fattn-vec.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.cuh`).
- **Does any `.cu` call them?** Exactly one: `set-rows.cu` calls `turbo_innerq_check_finalize()` at `:569` (turbo3), `:914` (turbo2) and `:1129` (turbo4), and its kernels are the only code that reads or writes the `d_innerq_*` device arrays (`:288-295`, `:657-664`, `:1000-1007`). Repo-wide, `d_innerq` appears in only two files: `turbo-quant.cuh` and `set-rows.cu`. `turbo_innerq_init()` and `turbo_innerq_finalize()` are reached only from `turbo_innerq_check_finalize()`, i.e. only through that same TU. `turbo_innerq_is_active()` has **zero callers** anywhere.
- **Which copy is live on a device call?** The one instantiated in `set-rows.cu`. That TU both uploads the scales (`cudaMemcpyToSymbol` on its own device symbols, from `turbo_innerq_finalize()`) and reads them (its kernels), so calibration, finalization and application are self-consistent — and invisible to every other TU's copy, because each copy is a distinct symbol with internal linkage.

### Home B — `turbo-innerq.{cu,cuh}` (the publish mirror)

`turbo-innerq.cuh` (21 lines) declares the shared host state and the three functions that move the *result* of calibration across the TU boundary; `turbo-innerq.cu` (32 lines) defines them:

```c
bool  g_innerq_finalized = false;                                  // turbo-innerq.cu:5
float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS] = { 1, 1, … };   // :6-11
static bool g_innerq_tensor_needs_update = false;                   // :13
void turbo_innerq_publish(const float * scale_inv, int group_size); // :15-22
bool turbo_innerq_needs_tensor_update(void);                        // :24-26
void turbo_innerq_mark_tensor_updated(void);                        // :28-30
```

This module holds **no device state and no calibration logic**. Its only producer is `turbo_innerq_publish()`, called at the tail of Home A's `turbo_innerq_finalize()` (`turbo-quant.cuh:249`) — which works because a `static` function may call an `extern` one, so `set-rows.cu`'s copy of the finalizer writes into the one true `g_innerq_scale_inv_host`. Its only consumer is `llama_kv_cache_context::apply()` in `src/src/llama-kv-cache.cpp:3121-3126`, which uploads `g_innerq_scale_inv_host` into the `turbo_innerq_scale_inv` tensor and clears the flag; that TU gets the symbols through an `extern` block guarded by `GGML_USE_CUDA` (`:28-32`) with non-CUDA stubs in the `#else` branch (`:33-37`).

The header's own comment states the intended division of labour: *"The host-side state lives in turbo-innerq.cu; device-side state is per-TU in turbo-quant.cuh (only set-rows.cu needs device access)"* (`turbo-innerq.cuh:3-5`).

### Verdict, stated precisely

- The two homes are **both in use**, for different things: Home A holds the control state and device state and is the live calibration path; Home B holds only the published `scale_inv` result plus two flags.
- Home A's duplication is **latent, not currently observable**: because `set-rows.cu` is both the only producer and the only consumer of the control state, a second TU's copy can never be *observed* to disagree — but such a copy exists for every includer, and any future caller of `turbo_innerq_is_active()`/`turbo_innerq_check_finalize()` from another TU would read `innerq_enabled == 0` and see InnerQ as disabled. That is exactly the hazard [[tq-2-innerq-host-state]] tracks, and the duplicate statics are still in the header: **the defect is not fixed.**
- Home B demonstrates that the cross-TU boundary *can* be crossed cleanly; it removes none of Home A's per-TU state.

### `INNERQ_MAX_CHANNELS` and `head_dim = 256`

`#define INNERQ_MAX_CHANNELS 128` lives in `turbo-innerq.cuh:7` and bounds every array in both homes. Its four consumers are the device arrays (`turbo-quant.cuh:147-149`), the host working buffers in init/finalize (`:177`, `:191`, `:205`, `:215-216`), the publish loop (`turbo-innerq.cu:16-21`) and the model-side tensor plus its uploads (`llama-kv-cache.cpp:378`, `:434-436`, `:543-545`, `:3124`); a fourth textual copy of "128" is the `#ifndef` fallback at `llama-kv-cache.cpp:24-26`.

The scale is indexed by **position inside a WHT group**, not by absolute channel: the encoder uses `d_innerq_scale[j]` with `j = threadIdx.x ∈ [0, GROUP_SIZE)` where a block is exactly one group (`set-rows.cu:259-295`), and the rotation op uses `scale_inv[t % group_size]` (`turbo-wht.cu:50`, `:92`). Since the group size is pinned to 128 for every turbo tensor (`set-rows.cu:555`, `:900`; `turbo-wht.cu:134`; `wht_group = 128` written at `llama-kv-cache.cpp:1586-1587`, `:1637-1638`, `:1663-1664`), a 256-wide head is served by two 128-wide groups that each index the same 128-entry scale vector — equalized, not truncated. The guard that was presumably meant to catch an oversized group, `const bool multi_group_per_head = (group_size < 128);` (`turbo-quant.cuh:268-277`), only fires for groups *smaller* than 128, so it is dead as written. See [[tq-7-innerq-max-channels]] for the full argument; this page agrees with it and records that the constant is a group bound rather than a channel bound.

## Known issues

- [[tq-2-innerq-host-state]] — the four `static` host variables and four `static` helpers are still declared in `turbo-quant.cuh`; every includer gets its own copy, and the state.md claim that a *different* TU can miss the calibration is not reachable here because `set-rows.cu` is the only caller.
- [[tq-3-innerq-multigpu]] — the six `static __device__` arrays are per-TU *and* per-device, while every upload is a single unlooped `cudaMemcpyToSymbol` targeting the current device only (`turbo-quant.cuh:179-182`, `:240-244`).
- [[tq-6-innerq-race]] — `if (j == 0) atomicAdd(&d_innerq_count, 1);` (`set-rows.cu:290`, `:659`, `:1002`) assumes one group per block and one block per group; today's launches satisfy that, a future block shape would not.
- [[tq-7-innerq-max-channels]] — `INNERQ_MAX_CHANNELS = 128` is four unsynchronised literals plus a dead `multi_group_per_head` guard; with the group pinned at 128 there is no live truncation for `head_dim = 256`.
- Related: [[tq-4-wht-numerical-mismatch]] (the rotation InnerQ is applied around) and [[tq-5-tail-elements]] (the tail kernels that never accumulate or apply InnerQ).
- Not filed: since `innerq_enabled` is per-TU, running the trainer through two devices or two caches in one process would give each cache's `SET_ROWS` the same `TURBO_INNERQ` behaviour but independent counters; and the `g_innerq_finalized` flag is written by `turbo_innerq_publish()` (`turbo-innerq.cu:22`) but never read anywhere — the consumer tests `turbo_innerq_needs_tensor_update()` instead (`llama-kv-cache.cpp:3121`).

## See also

[[overview]] · [[v100-sxm2]] · [[ternary-bonsai-2-27b]] · [[kv-cache]] · [[turboquant]] · [[quantization]] · [[walsh-hadamard-transform]] · [[triattention]] · [[kv-eviction]] · [[gemm-dispatch]] · [[performance-profile]] · [[codebase-map]] · [[roadmap]]
[[source-state-md]] · [[source-readme]]
[[tq-2-innerq-host-state]] · [[tq-3-innerq-multigpu]] · [[tq-4-wht-numerical-mismatch]] · [[tq-5-tail-elements]] · [[tq-6-innerq-race]] · [[tq-7-innerq-max-channels]]
