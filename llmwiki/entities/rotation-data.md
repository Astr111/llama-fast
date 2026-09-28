---
title: Rotation data
type: entity
status: current
updated: 2026-09-28
sources: []
verified: [src/src/turbo-rotation-data.h, src/src/turbo-rotation-data-32.h, src/src/llama-triattention.cpp, src/src/llama-triattention.h, src/src/llama-kv-cache.cpp, src/src/llama-kv-cache.h, src/ggml/src/ggml-turbo-quant.c]
tags: [quantization, kv-cache, triattention]
---

# Rotation data — TurboQuant dense rotation constants

## What it is

Two generated C headers under `src/src/` that hold the dense square rotation tables used by TurboQuant's group rotation. They are compile-time constants (`static const float`), not runtime state and **not** part of the `.triattention` calibration file.

| Header | Size | Tables | Entry values |
| :--- | :--- | :--- | :--- |
| `src/src/turbo-rotation-data.h` | 4 103 lines / 590 143 B | `TURBO_ROTATION_RT[128*128]` at `:3`, `TURBO_ROTATION_R[128*128]` at `:2054` | `±8.83883461e-02` |
| `src/src/turbo-rotation-data-32.h` | 71 lines / 36 281 B | `TURBO_ROTATION_R_32[1024]` at `:3`, `TURBO_ROTATION_RT_32[1024]` at `:38` | `±1.76776695e-01` |

- The comment on line 1 of the 128 header is *"Pre-computed rotation matrices for TurboQuant pre-rotate-queries"*; the `-32` header says *"Pre-computed 32x32 rotation matrices for TurboQuant (group_size=32, seed=42)"*.
- `0.0883883461 ≈ 1/√128` and `0.176776695 ≈ 1/√32`, so each table is a **normalised Hadamard / Walsh–Hadamard matrix** of its stated order, not a Gaussian or learned rotation — see [[walsh-hadamard-transform]]. The 128 order matches TurboQuant's rotation group (`QK_TURBO*_GROUP = 128`, [[turboquant]]).
- The two tables of a pair are not identical: the first rows of `TURBO_ROTATION_R` (`:2055`) and `TURBO_ROTATION_RT` (`:4`) differ, consistent with `RT = Rᵀ`. Whether `RT` is exactly the transpose/inverse was **not** checked algebraically — `[UNVERIFIED]`.

**Two files, not two precisions.** Both headers store 32-bit `float`; the `-32` suffix is the *transform order / TurboQuant group size* (32 vs 128), not precision or head-dim selection. The `-32` header currently has **no includer in this checkout**: the only `#include "turbo-rotation-data*.h"` sites in the tree are `src/src/llama-kv-cache.cpp:428`, `src/src/llama-kv-cache.cpp:537` and `src/src/llama-triattention.cpp:56`, all of the 128 header; `turbo-rotation-data-32.h` appears only in the file listing of [[codebase-map]]. It reads as a dormant variant kept for a `group_size=32` configuration — possibly built in an out-of-tree checkout ([[ta-1-wht-inversion-256]] context).

## How it works

- The 128×128 tables are used as plain **dense row-major matrices**. `matvec_128()` (`src/src/llama-triattention.cpp:90-93`) computes `out[i] = Σ_j mat[i*128+j]·vec[j]`; it is the only reader of the constants.
- `TURBO_ROTATION_RT` is the **inverse** rotation: it brings dequantized K/V out of WHT-rotated space back to the original basis. Applied at `src/src/llama-triattention.cpp:620` inside `triattention_dequant_kv_head()` (`:540`), guarded by `need_wht_inv`, with the comment *"Apply inverse WHT rotation for turbo2/turbo3 / turbo4 dequant already applies R^T internally"* (`:614-616`). The caller documents the same rule at `:1015-1017`: `turbo2_0`/`turbo3_0` need the extra rotation; `turbo4_0` already applies `Rᵀ` internally; `q8_0`/`f16`/`f32` need no rotation at all.
- Both tables are also uploaded verbatim into two `F32` backend tensors, `turbo_rotation` and `turbo_rotation_inv`, created once per KV cache when the K type is a turbo type (`src/src/llama-kv-cache.cpp:370-375`), uploaded with `ggml_backend_tensor_set(..., 128*128*sizeof(float))` at `:429-430` and re-uploaded after the buffer clear at `:538-539`; accessors at `src/src/llama-kv-cache.h:182-185`. Per [[turboquant]] §3 those tensors are read by no graph node and no CUDA kernel — the upload is dead weight (see *Known issues*).

**Arithmetic on the binary cost** (from the read declarations, not measured):

- `128×128 = 16 384` floats × 4 B = **65 536 B = 64 KiB** per table, so **128 KiB** per `#include` of the 128 header.
- The header is included at three sites (two of them inside function bodies in `llama-kv-cache.cpp`), each defining fresh `static const` objects → **up to 3 × 128 KiB ≈ 384 KiB of read-only data** if the compiler/linker does not deduplicate them.
- As text the 128 header is 590 143 B for 32 768 literals ≈ 18 B per literal → **≈4.6×** the binary footprint.
- `-32` header: `1 024` floats × 4 B = **4 KiB** per table, 8 KiB for both; 36 281 B text ≈ 4.4×.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/src/turbo-rotation-data.h` | `TURBO_ROTATION_RT` (`:3`) and `TURBO_ROTATION_R` (`:2054`), 128×128 `float`, row-major |
| `src/src/turbo-rotation-data-32.h` | `TURBO_ROTATION_R_32` (`:3`), `TURBO_ROTATION_RT_32` (`:38`), 32×32 `float`; no includer in the tree |
| `src/src/llama-triattention.cpp:56`, `:90-93`, `:614-620` | `#include`, `matvec_128()`, and the only runtime read — `matvec_128(TURBO_ROTATION_RT, dequant_tmp.data()+b, final_dst+b)` in `triattention_dequant_kv_head()` |
| `src/src/llama-kv-cache.cpp:370-375`, `:427-430`, `:536-539` | tensor creation (`turbo_rotation`, `turbo_rotation_inv`) and `ggml_backend_tensor_set` upload of `TURBO_ROTATION_R` / `TURBO_ROTATION_RT` |
| `src/src/llama-kv-cache.h:182-185` | accessors and the intended semantics comment |
| `src/src/llama-triattention.h` | the `.triattention` calibration format — statistics only, no rotation constants |

## Who consumes it, and when

- **TriAttention CPU scoring path.** `triattention_dequant_kv_head()` runs on each prune round to dequantize cached K on the CPU before RoPE inversion and scoring; only `turbo2_0`/`turbo3_0` K reach the `matvec_128` call. This is the scoring path, not the calibration path.
- **GPU scoring path: no.** `triattention_init_gpu()` (`src/src/llama-triattention.cpp:885`) initialises a separate device state; the dense tables are not referenced there — the GPU path and its handling of the rotation is the subject of [[ta-1-wht-inversion-256]].
- **Backend tensors: uploaded but unused.** Populated on the KV-cache write path, read by nothing ([[turboquant]] §3).
- **Calibration: no.** The `.triattention` file carries per-`(layer, head)` `q_mean_real/imag`, `q_abs_mean` and `r_f` arrays plus `freq_count` (`src/src/llama-triattention.h`), never the rotation constants — those are compiled in ([[triattention-calibrate]]).

## How it would be regenerated

**No generator is recorded anywhere in the tree.** The 128 header's 4 103 lines contain exactly one comment (the one-line description); a search of the file for generate/script/python/provenance/seed terms returns nothing, and no `.py`/script in the repo mentions `TURBO_ROTATION` (the only `rotation` hits in Python are matplotlib label angles). The `-32` header's *"seed=42"* implies a seeded generator existed, matching the same normalised-Hadamard shape, but its source is absent from the checkout.

Also note the constants did **not** come from the runtime CPU codec: `src/ggml/src/ggml-turbo-quant.c` builds its own rotation as a Gaussian matrix orthonormalised by modified Gram–Schmidt from an LCG PRNG (`turbo_init_rotation()`, `:67-118`) — a different matrix and a different mechanism from the `±1/√n` Hadamard tables here. A 590 KB generated file with no recorded generator is a maintenance fact worth carrying: the tables cannot be re-derived or audited from this repo alone.

## Known issues

- [[ta-1-wht-inversion-256]] — the K cache is stored rotated, so any reader in the original basis must invert the rotation; the TriAttention GPU scoring kernel does not, at `head_dim = 256`. The CPU `matvec_128` path here is the contrast case.
- The `turbo_rotation` / `turbo_rotation_inv` tensors are allocated and uploaded but read by no graph node and no CUDA kernel ([[turboquant]] §3) — ~128 KiB uploaded per cache for nothing; not filed as its own issue.
- **Label contradiction:** `src/src/llama-kv-cache.h:182-184` documents `turbo_rotation = R` (forward, Q pre-rotate) and `turbo_rotation_inv = Rᵀ = R⁻¹`, but the upload code names the tensor holding `TURBO_ROTATION_R` `turbo_rotation` with the inline comment `// R^T` (`:373`) and the one holding `TURBO_ROTATION_RT` `turbo_rotation_inv` with `// R` (`:375`). The two disagree about which table is which; not filed as an issue.
- **Open question (flagged, never run):** does the dense `matvec_128(TURBO_ROTATION_RT, …)` path agree **bit-for-bit** with the butterfly implementations (the CUDA `turbo_fwht_128` family and the WHT op, [[walsh-hadamard-transform]])? The dense path sums 128 products per output element in a different order than a butterfly network, so exact agreement is not expected a priori. `[UNVERIFIED]`
- Provenance: regenerating or extending these tables (e.g. for a different group size) requires an external generator that is not in the repo.

## See also

[[triattention]] · [[turboquant]] · [[walsh-hadamard-transform]] · [[ta-1-wht-inversion-256]] · [[triattention-calibrate]] · [[quantization]] · [[kv-cache]]
