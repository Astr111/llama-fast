---
title: Auto-fit (`--fit`)
type: entity
status: current
updated: 2026-09-29
sources: [AGENTS.md]
verified: [src/common/fit.cpp, src/common/fit.h, src/common/common.cpp, src/common/common.h, src/common/arg.cpp, src/include/llama.h, src/src/llama-model.cpp, src/ggml/src/ggml-cuda/ggml-cuda.cu]
tags: [placement, vram, cuda, operations]
---

# Auto-fit (`--fit`)

## What it is

`--fit` adjusts whatever placement and context parameters the user **left at their defaults** until the model plus its context are projected to fit in device memory. It runs once, before the real model load, inside `common_init_result` (`src/common/common.cpp:1295-1325` → `llama_model_load_from_file` at `src/common/common.cpp:1327`).

It is **on by default**: `common_params::fit_params = true` (`src/common/common.h:469`). That default is the whole reason this page exists — it is the only free-VRAM consumer on the startup path of every invocation of `llama-cli`/`llama-server`.

The header states the contract exactly (`src/common/fit.h:33-36`):

> only parameters that have the same value as in `llama_default_model_params` are modified, with the exception of the context size which is modified if and only if equal to 0

Consequences, both verified below: it **never overrides `-ngl`**, and it never overrides a context size the user set.

## How it works

### 1. The measurement primitive — no analytic cost model

Everything the fit does rests on `common_get_device_memory_data_impl` (`src/common/fit.cpp:29-152`). One call:

1. swaps the global llama log callback so messages below `log_level` are downgraded to the debug log (`src/common/fit.cpp:38-56`), restoring it at the end;
2. loads the model with `no_alloc = true` and `load_mode = LLAMA_LOAD_MODE_NONE` (`src/common/fit.cpp:58-60`) and creates a **real context** from it (`src/common/fit.cpp:67`);
3. reads `llama_get_memory_breakdown(ctx)` and attributes each buffer type's `model`/`context`/`compute` bytes to the device that owns it, with host buffer types collapsed into a trailing host entry (`src/common/fit.cpp:81-101`);
4. queries free/total memory per device — see *The VRAM probing* below;
5. returns `hp_ngl` (= `n_layer`, `+ n_layer_nextn` when `load_mtp`), `hp_n_ctx_train` and `hp_n_expert` (`src/common/fit.cpp:139-147`).

**There is no cost model for weights and no cost model for the KV cache.** The sizes come from the real allocator: the probe builds the actual backend buffers for the requested `n_gpu_layers`/`n_ctx`/KV types and reports what they occupy. The only formulaic step anywhere in the file is a secant line between two *measured* context sizes (step 3 below).

### 2. Step 1 — measure at the initial parameters, then decide whether to reduce context

`src/common/fit.cpp:262-278` measures once at the caller's parameters. Two pre-computations matter:

- `n_streams = kv_unified ? 1 : max(1, n_seq_max)` (`src/common/fit.cpp:196`) — with a non-unified KV cache the context budget is multiplied by the number of sequences;
- `n_ctx_auto = (cparams->n_ctx == 0)` (`src/common/fit.cpp:197`). If auto, the probe context is resolved to `min(hp_n_ctx_train * n_streams, UINT32_MAX)` (`src/common/fit.cpp:268-275`).

The surplus test (`src/common/fit.cpp:379-441`) then tries to close a global deficit, and it only does so when the context is *auto*: it measures a second time at the minimum context (`n_ctx_min_total`, `src/common/fit.cpp:413-414`), linearly interpolates to the target and rounds **down** to a multiple of `256 * n_streams` (`src/common/fit.cpp:430-436`). A user-set context gets an explicit no-op: `context size set by user ... -> no change` (`src/common/fit.cpp:441-442`).

A second model (draft, or an MTP context) is folded into *every* measurement (`src/common/fit.cpp:199-260`); with `shares_model` its weight term is zeroed because MTP runs on the main model's weights (`src/common/fit.cpp:257-259`).

### 3. The candidate space — layers per device, back to front

The placement search (step 3, `src/common/fit.cpp:493-724`) has one decision variable per device, a struct `ngl_t { n_layer, n_part, overflow_type }` (`src/common/fit.cpp:533-547`):

- **whole layers per device**, counting the output layer as one extra (`hp_ngl + 1`; `src/common/fit.cpp:661`), filled **back-to-front** so the last device owns the output layer, which is not allowed to be partial (`src/common/fit.cpp:655-658`, `src/common/fit.cpp:602-604`);
- **partial layers** for MoE models: the first overflowing layer of each device may push only its MoE tensors to CPU (or to the next device), via `tensor_buft_overrides` regexes built by `get_overflow_pattern` (`src/common/fit.cpp:500-530`) — `blk\.N\.ffn_(up|down|gate_up|gate)_(ch|)exps`;
- `tensor_split` is written as the per-device layer counts, and `n_gpu_layers` as their sum (`src/common/fit.cpp:559-568`).

The search is the **method of false position** (`src/common/fit.cpp:655-712`): keep a lower bound (0 layers) and an upper bound (all unassigned layers) for the device, compute the interpolation step

```
step_size = delta * (target - mem) / (mem_high - mem)
```

clamped to `[1, delta-1]`, probe it, and move whichever bound it lost to — stopping when the gap is one layer and keeping the lower bound that still fits. The comment block at `src/common/fit.cpp:408-410` states the waste heuristic the multi-device target subtracts: whole layers for dense models, `<= 1/3` of a layer per tensor for MoE, i.e. `0.5 layers/tensors per device` on average.

For a MoE model there is a step 4 (`src/common/fit.cpp:726-860`): first measure with **all** MoE tensors on the CPU (`src/common/fit.cpp:617-626`); if only the dense weights fit with a surplus, convert dense-only layers back into full layers front-to-back, and finally attempt one extra partial layer with `overflow_type` tried in the order `UP` → `GATE` → `ATTN` (`src/common/fit.cpp:790-850`).

Because `get_memory_for_layers` (`src/common/fit.cpp:588-611`) calls the full probe primitive, **every search iteration is a complete model load plus context creation**.

### 4. When it refuses to act

`common_params_fit_impl` throws `common_params_fit_exception` (caught at `src/common/fit.cpp:893-895`, logged as a warning, parameters left untouched) when the user has already made the decision:

| Guard | Line |
| :--- | :--- |
| `split_mode == LLAMA_SPLIT_MODE_TENSOR` | `src/common/fit.cpp:183-185` |
| `n_gpu_layers != llama_model_default_params().n_gpu_layers` (`-1`) | `src/common/fit.cpp:460-462` |
| any non-zero user `tensor_split` | `src/common/fit.cpp:463-472` |
| `SPLIT_MODE_ROW` with more than one device | `src/common/fit.cpp:488-490` |
| `tensor_buft_overrides` already set | `src/common/fit.cpp:491` |

So on the `-ngl`/`-c` axis it **defers, it does not fight**: `-c N` freezes the context, and `-ngl` — *any* explicit value, including `99` and `all` — aborts the placement search entirely. There is no path where the fit overrides a user's `-ngl`.

## The flag family

| Flag / env | Default | Where |
| :--- | :--- | :--- |
| `-fit` / `--fit` `[on\|off]`, env `LLAMA_ARG_FIT` | **on** (`src/common/common.h:469`) | `src/common/arg.cpp:2866-2879` |
| `-fitp` / `--fit-print` `[on\|off]`, env `LLAMA_ARG_FIT_ESTIMATE` | off (`fit_params_print = false`, `src/common/common.h:470`) | `src/common/arg.cpp:2880-2893` |
| `-fitt` / `--fit-target` `MiB0,MiB1,…`, env `LLAMA_ARG_FIT_TARGET` | 1024 MiB per device (`src/common/common.h:474`) | `src/common/arg.cpp:2894-2918` |
| `-fitc` / `--fit-ctx` `N`, env `LLAMA_ARG_FIT_CTX` | 4096 (`src/common/common.h:471`) | `src/common/arg.cpp:2919-2925` |

Two corrections to the names used elsewhere in the vault:

- **There is no `--fit-estimate` flag.** The flag is `--fit-print`/`-fitp`; `FIT_ESTIMATE` survives only as its environment name (`src/common/arg.cpp:2893`).
- `--fit-print` is gated to `LLAMA_EXAMPLE_FIT_PARAMS` (`src/common/arg.cpp:2893`), i.e. the `llama-fit-params` tool (`src/common/arg.cpp:1093`), not `llama-cli` or `llama-server`. That is why [[first-live-measurements]] could observe that the CLI prints **no memory report at all**: the flag that would print one cannot be passed to it.

`-c 0` is a third form of deference: it sets `fit_params_min_ctx = UINT32_MAX` (`src/common/arg.cpp:1656-1658`), and the fit reads that sentinel as "user has requested full context size ... no change" (`src/common/fit.cpp:427-429`).

## The VRAM probing

`ggml_backend_dev_memory(dev, &free, &total)` is the only free-memory query in the file, at three sites:

- `src/common/fit.cpp:106` — the **CPU** device (host memory; not `cudaMemGetInfo`);
- `src/common/fit.cpp:115` — each GPU device, inside the probe primitive;
- `src/common/fit.cpp:977` — inside `common_memory_breakdown_print`, which the probe primitive calls unconditionally before freeing the context (`src/common/fit.cpp:146`). Whether that print is additionally gated on a log level I did not read `[UNVERIFIED]`.

On the CUDA backend this resolves to `ggml_backend_cuda_device_get_memory` → `ggml_cuda_set_device` + `cudaMemGetInfo` (`src/ggml/src/ggml-cuda/ggml-cuda.cu:5061-5063`). Two escapes: on an integrated device (or with `GGML_CUDA_ENABLE_UNIFIED_MEMORY`) it reads `/proc/meminfo` instead (`src/ggml/src/ggml-cuda/ggml-cuda.cu:5072-5093`) — the measured GTX 1660 is discrete, so the real `cudaMemGetInfo` stands; and on failure it returns 0/0 (`src/ggml/src/ggml-cuda/ggml-cuda.cu:5064-5069`), which the fit reads as "device did not report memory; --fit will not use it" (`src/common/fit.cpp:117-127`).

A second, easy-to-miss source of the same call is the model load itself: `llama_model_load` queries free memory per device to compute the automatic `tensor_split` weights (`src/src/llama-model.cpp:1593-1603`), and the fit performs a model load per probe.

### Does that account for 13 calls and 116 ms?

The recorded run ([[first-live-measurements]], turbo configuration) was launched with **`-ngl 99`**. Trace the consequence:

1. `-ngl 99` → `mparams.n_gpu_layers = 99` (`src/common/common.cpp:1713`);
2. `llama_model_default_params().n_gpu_layers = -1` (`src/src/llama-model.cpp:2975`);
3. `99 != -1` → the guard at `src/common/fit.cpp:460-462` throws, the exception is caught at `src/common/fit.cpp:893-895`, and the fit exits.

So on exactly that command line the fit runs **one** probe (`src/common/fit.cpp:263`) before giving up: at most **2** `cudaMemGetInfo` calls from fit.cpp (`:115` and `:977`), plus 1-2 from the two model loads (probe and real; `src/src/llama-model.cpp:1594`). That is **3-4 of the 13 calls**, not 13.

**The arithmetic does not fit, and the count is refuted as stated.** For the fit to issue ~12-13 queries it would have to enter the false-position loop (~9-10 probes on a single GPU), which requires `n_gpu_layers` to be at its default, i.e. **no `-ngl` on the command line**. One of the two premises is wrong: either the profiled invocation differed from the command line recorded in [[first-live-measurements]], or the remaining ~9 calls come from model-load paths outside the fit. What can be said with confidence is the *shape*: the fit's query count is not two per run but **two per candidate per device**, and the number of candidates is data-dependent.

The **8.9 ms per call** is not explained by anything in this file either. fit.cpp only *issues* the query; between two queries the probe loads a model and creates a context (including its compute buffers), so a summary that folds device synchronisation into the API time of `cudaMemGetInfo` would be inflated — `[INFERENCE]`, the profiling mechanism was not read. What is *not* inference: "a per-request VRAM probe" is ruled out. Auto-fit is startup-only, in `common_init_result`, before the server loop ([[request-lifecycle]]).

## Interaction with this fork's memory

The KV cache is the thing being sized, and the fit handles the fork's unusual KV types correctly **by construction**:

- `cparams` is built by `common_context_params_to_llama(params)` (`src/common/common.cpp:1293`) from the user's `-ctk`/`-ctv` (`cache_type_k`/`cache_type_v`, `src/common/common.h:341-342`). The probe therefore creates a *real* context of type `turbo3`/`q8_0`/whatever, on the *real* hybrid architecture, and reads the KV buffer's actual size back out of `llama_get_memory_breakdown` (`src/common/fit.cpp:81`). There is no `ggml_row_size`-style KV formula in fit.cpp to be wrong — the quarter-of-naive cache of [[kv-accounting]], and the 16 KV-bearing layers of [[hybrid-memory]], are simply measured.
- The fit never *changes* a KV type. Its outputs are exactly `n_ctx`, `n_gpu_layers`, `tensor_split` and `tensor_buft_overrides`.

The one place an unusual KV layout can still bite is the context **slope**. When reducing an auto context the fit extrapolates from two measurements with `bytes_per_ctx = (used(n_ctx_max) - used(n_ctx_min)) / (n_ctx_max - n_ctx_min)` (`src/common/fit.cpp:435`). If the compute-buffer term is not context-independent, that secant mixes a KV slope with a compute change. [[first-live-measurements]] already suspects the turbo path shrinks the attention compute buffer on `flash_attn_ext_vec` — the same suspicion applies here, in the fit's favour or against it, but the KV term itself is measured, not modelled. The KV type is also invisible to the *placement* search, which is a layer/tensor-count search: it can only ever see the KV term through the total per-device bytes it measures.

A structural caveat: the free-memory sample is taken **while the probe's own allocations are resident** (`src/common/fit.cpp:115`), and the per-device targets are `dmds_full[id].free - margins[id]` from the first probe (`src/common/fit.cpp:652`) while later probes re-sample a device whose free memory has moved. Any other process on the GPU — including the `nvidia-smi`-visible job that may coexist with the model — makes those samples mutually inconsistent.

## Known issues

- **A cost with no benefit on this project's own command lines.** The launch scripts and the live measurement both pass `-ngl`, which makes the whole placement search unreachable (`src/common/fit.cpp:460-462`) — after the fit has already paid for one full probe load. Only a `WRN` records it (`src/common/fit.cpp:894`).
- **`LAYER_FRACTION_ATTN` and `LAYER_FRACTION_UP` compile to the identical regex** `blk\.N\.ffn_(gate|up|gate_up|down).*` (`src/common/fit.cpp:504-516`), yet the MoE step tries `UP` first and only falls back to `ATTN` when `UP` failed (`src/common/fit.cpp:790-850`). The fallback then measures a configuration that is byte-for-byte the one that just failed, and the intended attention-only overflow is never achieved. `[INFERENCE]` on the intent; the duplicate pattern and the wasted probe are read.
- The `--fit-print` flag cannot reach `llama-cli` or the server (`src/common/arg.cpp:2893`), so the estimate it would produce is unobtainable in the configurations this vault measures.
- The probe mutates global logger state and is documented as not thread-safe (`src/common/fit.h:35`).
- Multi-device correctness rests on a stated heuristic, not a measurement: `0.5 layers/tensors per device` of waste is subtracted from the target for `nd` devices (`src/common/fit.cpp:404-410`).
- `get_overflow_pattern` allocates 1000 static pattern strings per fraction (`.cpp:501`) and caps models at 1000 layers (`src/common/fit.cpp:502-503`) — a hard limit, not a degradation.

## Where it lives

- `src/common/fit.cpp` (1072 lines), header `src/common/fit.h`
- Called from `src/common/common.cpp:1295-1325`; entry point `common_fit_params` (`src/common/fit.cpp:878`)
- Flags: `src/common/arg.cpp:2866-2925`; related `-c` sentinel at `src/common/arg.cpp:1656-1658`
- Defaults: `src/common/common.h:466-474`
- Backend query: `src/ggml/src/ggml-cuda/ggml-cuda.cu:5061`; model-load query `src/src/llama-model.cpp:1593-1603`
- Companion tool `llama-fit-params` (`src/common/arg.cpp:1093`) uses the same entry point via `common_fit_print` (`src/common/fit.cpp:1047`)

## See also

[[runtime-switches]] · [[first-live-measurements]] · [[kv-accounting]] · [[kv-cache]] · [[hybrid-memory]] · [[request-lifecycle]] · [[turboquant]] · [[device-placement]] · [[open-questions]]
