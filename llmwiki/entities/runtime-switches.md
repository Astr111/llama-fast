---
title: Runtime switches (env vars and flags)
type: entity
status: current
updated: 2026-09-29
sources: [README.md, state.md]
verified: [build/cuda13.zip, src/common/arg.cpp, src/ggml/src/ggml-cuda/ggml-cuda.cu]
tags: [operations, configuration, cuda]
---

# Runtime switches

## What it is

The environment variables and flags that change this engine's behaviour, extracted from the binaries themselves rather than from the documentation. Each name below was found by `strings` over `llama-cli`, `libllama.so`, `libllama-common.so`, `libllama-cli-impl.so` and `libggml-cuda.so` from the CUDA 13 bundle (build `b10747-1773b4b1a`), so each is a reference the code actually makes — not an aspiration in a README.

This page exists because **no source document lists them**, and because two of the most consequential knobs in the vault (`GGML_CUDA_FORCE_MMQ`, `GGML_CUDA_DISABLE_GRAPHS`) appear in no page at all.

## How it works

### `GGML_CUDA_*` — the backend's own surface

| Variable | What it controls |
| :--- | :--- |
| `GGML_CUDA_GRAPH_OPT` | The graph-reuse and concurrent-stream path ([[cuda-graphs]]). Read once into a function-local `static`, so changing it mid-process has no effect |
| `GGML_CUDA_DISABLE_GRAPHS` | Turn graph capture off entirely — the ablation counterpart to the above |
| `GGML_CUDA_FORCE_MMQ` | Force the MMQ tile path (`mul_mat_q`, the kernel holding **75 % of GPU time** in the live measurement, [[first-live-measurements]]). A direct lever on [[quantized-kernel-units]] |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE` | Compute type handed to the cuBLAS fallback path ([[gemm-dispatch]]) |
| `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH` | Batch ceiling for the `PTQ1_0` (Prism ternary) MMQ path — the knob that exists because the ternary type's MMQ is gated differently from `PQ2_0`'s ([[prismml-weight-kernels]]) |
| `GGML_CUDA_FWHT_LEGACY` | Selects a legacy Walsh-Hadamard implementation. **Not mentioned anywhere in the vault before this page** — it implies the rotation has a switchable implementation, which bears directly on [[turbo-wht]] and the unification question ([[decisions-pending]], D2) |
| `GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_PDL`, `GGML_CUDA_DEVICES`, `GGML_CUDA_MAX_DEVICES`, `GGML_CUDA_P2P`, `GGML_CUDA_NCCL`, `GGML_CUDA_NO_PINNED`, `GGML_CUDA_REGISTER_HOST`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | Scheduling, multi-GPU and memory-management knobs |
| `GGML_CUDA_ALLREDUCE`, `GGML_CUDA_AR_*` | All-reduce tuning (multi-GPU path) |
| `GGML_CUDA_GB10_*` | Device-specific kernels for a GB10-class part — irrelevant to `sm_70`/`sm_75` but evidence of how many hardware-specific paths this fork carries |

### `GGML_GDN_*` and the recurrent path

`GGML_GDN_RAW_GATES_DISABLE` and `GGML_GDN_STATE_GATHER` control the gated-delta-net recurrence ([[gated-delta-net]]) — the one place where a runtime switch can change *what state the recurrent blocks keep*. Neither appears in any source document.

### `LLAMA_ARG_*` — the CLI's own variables

Every argument this build accepts has a matching `LLAMA_ARG_*` variable, and the ones that matter here are:

- **TriAttention**: `LLAMA_ARG_TRIATTENTION_STATS`, `_BUDGET`, `_WINDOW`, `_OFFSET_MAX`, `_AGG`, `_MODE`, `_TRIGGER`, `_SEED`. The presence of `_OFFSET_MAX` with a default of **0** is the switch behind [[ta-8-offset-max-zero-nan]].
- **Speculative decoding**: `LLAMA_ARG_SPEC_DRAFT_MODEL`, `_N_MAX`, `_N_MIN`, `_P_MIN`, `_P_SPLIT`, `_TYPE`, `_CACHE_TYPE_K`, `_CACHE_TYPE_V`, `_BACKEND_SAMPLING`, `_HF_REPO`, `_CPU_MOE`. This is the complete surface behind [[speculative-decoding]] — and note there is **no** `_N_DRAFT` or `_MAX_5`: the removed `--draft-max` left `LLAMA_ARG_DRAFT_MAX`/`_MIN` behind as orphaned names (see the "max 5" contradiction on that page).
- **KV cache**: `LLAMA_ARG_CACHE_TYPE_K`, `_CACHE_TYPE_V`, `LLAMA_ARG_KV_UNIFIED`, `_KV_OFFLOAD`, `LLAMA_ARG_SWA_FULL`, `LLAMA_ARG_KV_MEAN_CENTER` (the feature documented on [[source-kv-mean-center]]).
- **Memory and placement**: `LLAMA_ARG_MLOCK`, `_MMAP`, `_DIO`, `_LOAD_MODE`, `_N_GPU_LAYERS`, `_SPLIT_MODE`, `_TENSOR_SPLIT`, `_FIT`, `_FIT_CTX`.
- **RoPE**: `LLAMA_ARG_ROPE_FREQ_BASE`, `_ROPE_FREQ_SCALE`, `_ROPE_SCALING_TYPE`, and the YaRN family (`_YARN_*`). Relevant to the RoPE geometry behind [[ta-9-rope-scope-mismatch]] — but note none of them changes the *partial* rotation the model itself declares.

### The variable that does not exist

**`BONSAI_SPECULATIVE` is not in any binary.** `strings` over the five objects above returns **zero** occurrences, so no code path reads it, and setting it cannot change behaviour. It is not an undocumented feature, a hidden debug switch, or a stale name: it simply is not part of this build.

For completeness, the only `BONSAI_*` names found anywhere in the trees on this machine are `BONSAI_TOKENIZER_DIR` and `BONSAI_GGUF_PATH`, both in Python tooling of a sibling checkout rather than in the engine.

The real mechanism the name probably refers to is the flag/env surface above: `--spec-draft-model` plus `--spec-draft-n-max` (with `LLAMA_ARG_SPEC_DRAFT_N_MAX` as its environment form). If the intent was "enable speculative decoding through an env var", that is the switch to use.

## Where it lives

- Extracted from the shipped binaries: `build/cuda13.zip` → `bin/{llama-cli,libllama.so,libllama-common.so,libllama-cli-impl.so,libggml-cuda.so}`
- Argument registration and defaults: `src/common/arg.cpp`
- Backend-side variables: `src/ggml/src/ggml-cuda/ggml-cuda.cu` and the CUDA kernels

## Known issues

- **None of this is documented anywhere else.** The README lists only three variables (`GGML_CUDA_GRAPH_OPT`, `DISABLE_TRIATTENTION`, and the `CTK`/`CTV` shell variables of the launch scripts), which is a small fraction of what the binaries read.
- `GGML_CUDA_GRAPH_OPT` is captured into a `static` on first read ([[cuda-graphs]]), so an env-var-based toggle cannot be flipped at runtime — a trap for anyone trying to A/B it in one session.
- `LLAMA_ARG_DRAFT_MAX`/`_MIN` survive in the binary while the flag they belonged to was removed; an env var that sets a removed knob is a silent no-op.

## See also

[[first-live-measurements]] · [[cuda-graphs]] · [[quantized-kernel-units]] · [[gated-delta-net]] · [[triattention]] · [[speculative-decoding]] · [[turbo-wht]] · [[ta-8-offset-max-zero-nan]] · [[kv-cache]]
