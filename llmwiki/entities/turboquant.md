---
title: TurboQuant
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified:
  - src/ggml/include/ggml.h
  - src/ggml/src/ggml-common.h
  - src/ggml/src/ggml.c
  - src/ggml/src/ggml-turbo-quant.c
  - src/ggml/src/ggml-cpu/ops.cpp
  - src/ggml/src/ggml-cuda/turbo-quant.cuh
  - src/ggml/src/ggml-cuda/turbo-wht.cu
  - src/ggml/src/ggml-cuda/turbo-wht.cuh
  - src/ggml/src/ggml-cuda/turbo-innerq.cu
  - src/ggml/src/ggml-cuda/turbo-innerq.cuh
  - src/ggml/src/ggml-cuda/set-rows.cu
  - src/ggml/src/ggml-cuda/dequantize.cuh
  - src/ggml/src/ggml-cuda/convert.cu
  - src/ggml/src/ggml-cuda/getrows.cu
  - src/ggml/src/ggml-cuda/cpy.cu
  - src/ggml/src/ggml-cuda/fattn-common.cuh
  - src/ggml/src/ggml-cuda/fattn.cu
  - src/ggml/src/ggml-cuda/fattn-vec.cuh
  - src/ggml/src/ggml-cuda/ggml-cuda.cu
  - src/ggml/src/ggml-cuda/mmq.cu
  - src/ggml/src/ggml-cuda/mmvq.cu
  - src/ggml/src/ggml-cuda/vecdotq.cuh
  - src/ggml/src/ggml-cuda/mmf.cu
  - src/ggml/src/ggml-cuda/CMakeLists.txt
  - src/src/llama-kv-cache.cpp
  - src/src/llama-kv-cache.h
  - src/src/llama-graph.cpp
  - src/src/llama-triattention.cpp
  - src/src/turbo-rotation-data.h
  - src/common/arg.cpp
  - src/common/common.h
tags: [turboquant, kv-cache, quantization, cuda, walsh-hadamard-transform]
---

# TurboQuant

## What it is

The **low-bit vector quantization applied to the KV cache**: three ggml types — `turbo2_0`, `turbo3_0`, `turbo4_0` — that store K and V as per-group normalised vectors quantized against small centroid codebooks, plus the rotation that makes that quantization well-conditioned. It is the "reduce bits per token" axis of the fork, the counterpart to [[triattention]]'s "reduce token count" axis, and it is the reason the fork can serve a 27B model with a 16 GB KV budget on [[v100-sxm2]].

The design lineage is stated in the header of the CUDA implementation: `Based on: arXiv 2504.19874 (ICLR 2026)` ([[source-readme]] cites the same paper as *TurboQuant: Online Vector-Level LLM KV-Cache Compression*), and the rotation is described there as a *Polar Walsh-Hadamard Transform* (see [[walsh-hadamard-transform]]).

The **values path** is usually a different type from the keys path. Every launch script and every README example runs `-ctk turbo3 -ctv q8_0`, which is what [[source-readme]] calls the *speed profile*; the memory-maximum profile keeps `turbo2` for V.

## How it works

### 1. Types, ids and names

| Type id | Name | Declaration | Block | Bytes/block | Bits/value |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `GGML_TYPE_TURBO3_0` = 43 | `turbo3` | `src/ggml/include/ggml.h:433` | 128 | 50 | 3.125 |
| `GGML_TYPE_TURBO4_0` = 44 | `turbo4` | `:434` | 128 | 68 | 4.25 |
| `GGML_TYPE_TURBO2_0` = 45 | `turbo2` | `:435` | 128 | 34 | 2.125 |

Type traits are ordinary quantized-type entries (`src/ggml/src/ggml.c:708-731` — `TURBO3_0` at `:708`, `TURBO4_0` at `:716`, `TURBO2_0` at `:724`): `blck_size = QK_TURBO*`, `type_size = sizeof(block_turbo*_0)`, `is_quantized = true`, `to_float = dequantize_row_turbo*_0`, `from_float_ref = quantize_row_turbo*_0_ref`. Note the names the CLI accepts are the short forms (`turbo2`/`turbo3`/`turbo4`, `src/common/arg.cpp:324-332`) and they are in the allowed KV-type list at `arg.cpp:304-316`.

### 2. Block geometry — the real constants

All three types use a **128-element block**, and that block is exactly one rotation group:

```c
#define QK_TURBO3 128          // src/ggml/src/ggml-common.h:324
#define QK_TURBO3_GROUP 128    // :325  "rotation group size = head_dim"
#define QK_TURBO4 128          // :343
#define QK_TURBO2 128          // :374
#define QK_TURBO2_GROUP 128    // :375
```

The per-block layouts are `block_turbo3_0 { ggml_half norm; uint8_t qs[QK_TURBO3/4]; uint8_t signs[QK_TURBO3/8]; }` (`:330-334`), `block_turbo2_0 { ggml_half norm; uint8_t qs[QK_TURBO2/4]; }` (`:380-383`) and — with the default `TURBO4_USE_4BIT 1` (`:341-342`) — `block_turbo4_0 { ggml_half norm; ggml_half rnorm; uint8_t qs[QK_TURBO4/2]; }` (`:347-352`); the legacy `#else` branch keeps `qs[48] + signs[16]` at the same 68 bytes (`:357-364`). `static_assert`s at `:334`, `:354`/`:365`, `:368` and `:383` pin those sizes.

So the codebook geometry *as compiled* is:

| Type | payload | real bits/value | vs fp16 | codebook |
| :--- | :--- | ---: | ---: | :--- |
| turbo3 | 2-bit index + 1-bit high bit, split across `qs`/`signs` | 3.125 | 5.12× | 8 Lloyd-Max centroids, `TURBO_CENTROIDS_3BIT` (`turbo-quant.cuh:33-36`) |
| turbo4 | nibble-packed 4-bit index | 4.25 | 3.76× | 16 centroids `TURBO_CENTROIDS_4BIT` (`:297-302`) |
| turbo2 | 2-bit index | 2.125 | 7.53× | 4 centroids `TURBO_CENTROIDS_2BIT` (`:23-25`) |

Every one of those blocks carries one `fp16` group norm, and the encoder replaces it with a **corrected** norm (`group_norm / reconstruction_norm`) so that dequantization preserves the group's magnitude: the CUDA kernels write it at `set-rows.cu:402-407` (turbo3) and the CPU reference computes the same quantity at `src/ggml/src/ggml-turbo-quant.c:305-310`. `turbo4` additionally writes `rnorm = 0` because the 4-bit mode has no QJL residual (`set-rows.cu:1093-1096`).

`NL_TURBO2`/`NL_TURBO3` and their `_VEC` twins (`ggml-common.h:327-328`, `:377-378`) are derived FA iteration counts that, repo-wide, have **no users** — the actual iteration counts come from the FA templates.

> Contradiction (2026-09-28): the block-layout comments in `ggml-common.h:318-323` and `:373-374` still describe the pre-128 layouts — "Storage block size = 32", "norm(fp16) + 2-bit indices (8 bytes) + 1-bit extra (4 bytes) = 14 bytes per 32 values = 3.5 bits/value → 4.6× compression", "= 10 bytes per 32 values = 2.5 bits/value → 6.4×". [[codebase-map]] repeats those numbers (`block_turbo3_0` 14 B, `block_turbo2_0` 10 B) while also recording `QK_TURBO* = 128`. The `static_assert`s and the struct fields are the authority: 50 B / 34 B / 68 B. The same stale "block size 32" phrase appears at `turbo-quant.cuh:5`, in the `dequantize.cuh` type comments (`:161`, `:170`; the turbo4 comment at `:152` correctly says 128), and in the scoring kernel's `dequant_head_to_smem` comments (`triattention-score.cu:128`, `:139`); the constants those files actually use are `QK_TURBO* = 128`.

### 3. The rotation, and where it is applied

The transform is `R = (1/√n)·D₂·H·D₁` with fixed ±1 sign arrays; its five implementations, live/dead status and the `R`/`Rᵀ` identities are catalogued on [[walsh-hadamard-transform]]. What matters for the codec is *where* it is applied:

| Direction | Applied to | Site |
| :--- | :--- | :--- |
| forward | **K and V at encode time**, inside the `SET_ROWS` kernels | `set-rows.cu:324-351` (turbo3), `:692-725` (turbo2), `:1039-1062` (turbo4) |
| forward | **Q**, as a graph op, whenever the K cache is a turbo type | `llama-graph.cpp:2977-2988` (FA path), `:3100-3111` (MLA path), `:3292-3303` (ISWA path) |
| inverse | **the attention output**, as a graph op, whenever the V cache is a turbo type | `llama-graph.cpp:2700-2709` (FA), `:2778-2787` (non-FA) |

The point of rotating Q with the same `R` is that the K×Q dot product is preserved (`⟨Rq, Rk⟩ = ⟨q, k⟩`); the point of the inverse on the output is that V was stored rotated. The op itself is `GGML_OP_TURBO_WHT` (`src/ggml/include/ggml.h:586`), built by `ggml_turbo_wht()` (`src/ggml/src/ggml.c:6634-6670`) with `op_params = (direction, group_size)` and the InnerQ scale as `src[1]`; the CUDA implementation is `ggml_cuda_turbo_wht` in `turbo-wht.cu:117-174`, dispatched from `ggml-cuda.cu:2092-2093`; the CPU twin is `ggml_compute_forward_turbo_wht_f32` (`src/ggml/src/ggml-cpu/ops.cpp:12253-12331`, its own copy of the sign arrays at `:12250-12251`, dispatched at `:12336-12340`).

**Group size is pinned to 128 everywhere.** The cache zero-pads each K and V head to the next multiple of 128 at allocation (`llama-kv-cache.cpp:323-346` for K, `:347-360` for V, with the `<Q,0>·<K,0>` justification in the comment), writes `wht_group = 128` into every `SET_ROWS` node's `op_params` (`:1586-1587`, `:1637-1638`, `:1663-1664`), and the kernels clamp (`set-rows.cu:555` (turbo3), `:900` (turbo2)) and assert (`turbo-wht.cu:132-134`) to `{64, 128}`. The consequence is that the "tail element" kernels of [[tq-5-tail-elements]] are unreachable while that padding is in place.

Two dense 128×128 rotation matrices are also materialised as real tensors — `turbo_rotation` and `turbo_rotation_inv` (`llama-kv-cache.cpp:370-378`, uploaded from `turbo-rotation-data.h` at `:427-430` and re-uploaded at `:536-541`). They are allocated *inside the KV buffer* (the size accounting adds "+3 for turbo rotation matrices" at `:139`), yet no graph node and no CUDA kernel reads them: the accessors `get_turbo_rotation()`/`get_turbo_rotation_inv()` (`llama-kv-cache.h:184-185`, `:440-441`) have no callers outside `llama-kv-cache.cpp`, and the only consumer of the dense data is the TriAttention **CPU** fallback, which includes the header directly (`src/src/llama-triattention.cpp:620`).

### 4. `q8_0` values and why that is the "speed profile"

`-ctv q8_0` stores V in an ordinary 8-bit type, so nothing rotates V at encode time and the graph's inverse-rotation node is never built — both inverse branches test `v->type` against the three turbo types (`llama-graph.cpp:2700`, `:2778`). The K side is unaffected: with `-ctk turbo3` the Q pre-rotation still runs, because that branch tests `k->type` (`:2977`, `:3100`, `:3292`).

The trade is therefore *quality of V* (8-bit codebook-with-scale, no rotation error) against *no per-layer inverse-transform op*. `-ctv q8_0` is compatible with the fused-attention kernel tables, which register the mixed pairs explicitly (`turbo3_0 × q8_0` and `q8_0 × turbo3_0`, `fattn.cu:342-343`).

> Contradiction (2026-09-28): [[source-readme]] describes the speed profile as eliminating "64 inverse WHT kernels per token". Structurally, the graph builds at most **one** inverse-rotation node per attention layer whose V cache is a turbo type (`llama-graph.cpp:2700-2709`, `:2778-2787`), and the target model keeps a KV cache in only 16 of its 64 blocks ([[ternary-bonsai-2-27b]], read from the GGUF). So 16 is the ceiling for this deployment, not 64; where the README's number comes from is not determinable from this tree [UNVERIFIED].

### 5. Per-tensor layout in the cache, and the read paths

K and V cache tensors are created per layer per stream as `ggml_new_tensor_3d(ctx, layer_type_k/v, n_embd_*_gqa_eff, kv_size, n_stream)` with the *padded* embedding width (`llama-kv-cache.cpp:349-350`), and the per-layer type can be overridden by the `TURBO_LAYER_ADAPTIVE` modes (`:262-321`). Rows are one cache cell; the codec walks a row as consecutive 128-element blocks, so a row is `n_embd/padded_head_dim` heads × `padded_head_dim/128` groups.

Read paths, in the order they matter:

- **Fused attention** (the hot path). `vec_dot_fattn_vec_KQ_turbo3_0` (`fattn-common.cuh:333-385`), `…_turbo2_0` (`:387-434`) and `…_turbo4_0` (`:436+`) dequantize K blocks inline and dot them with the (already rotated) Q half2/float2 pairs, and `dequantize_V_turbo3_0` / `_turbo2_0` / `_turbo4_0` (`:778`, `:838`, `:895`) do the same for V. They are selected through `get_vec_dot_KQ()` (`:954-981`) and `get_dequantize_V()` (`:982-1006`), and the allowed type pairs are registered at `fattn.cu:339-369` for turbo×{turbo,q8_0} (15 combinations when `GGML_CUDA_FA_ALL_QUANTS` is off) with the guard `ggml_cuda_fattn_kv_type_supported()` (`fattn.cu:382-399`); CMake names the 15 instantiation units at `CMakeLists.txt:125-139`, though `template-instances/` is empty in this checkout ([[codebase-map]]). `fattn-vec.cuh:87-95` is where turbo K/V are classified as "unquantized" for thread-count purposes — they are dequantized by hand, not through `vecdotq.cuh`.
- **Generic dequantize**: `dequantize.cuh:154-176` (`dequantize_turbo4_0`/`_turbo3_0`/`_turbo2_0`, one `float2` per call, using the shared `turbo*_dequant_element` helpers in `turbo-quant.cuh:348-354`, `:386-392`, `:417-421`) reached from `convert.cu:664-669` / `:735-741` / `:772-777` / `:836-841` (`to_fp16`/`to_fp32`/`to_bf16`, both contiguous and strided variants) and hence from `cpy.cu`.
- **`GET_ROWS` does not support the turbo types**: `getrows.cu` has no `TURBO` case at all and its default arm aborts (`getrows.cu:415-417`), consistent with `supports_op` admitting turbo only for `GGML_OP_SET_ROWS` (`ggml-cuda.cu:5320-5336`).

### 6. The write path

Encoding is `SET_ROWS` (per K/V element of the batch): `set_rows_cuda_turbo3` / `_turbo2` / `_turbo4` (`set-rows.cu:537`, `:884`, `:1104`) launch one block per (group, row) — `k_set_rows_turbo3<idx_t,GROUP_SIZE>` (`:237-410`, WHT `:324-351`), `k_set_rows_turbo2<…>` (`:608-770`, WHT `:692-725`), `k_set_rows_turbo4<idx_t>` (`:952-1102`, WHT `:1039-1062`). Each does: load the group → InnerQ accumulate/apply → parallel L2 norm → normalise → forward WHT → nearest-centroid index → pack → corrected norm. The tail kernels (`:422`, `:775`) are unreachable under the padding regime ([[tq-5-tail-elements]]).

### 7. There is no native tensor-core or matmul path

Turbo types appear in **no** matmul fast path. `grep -c TURBO` returns 0 for `mmq.cu`, `mmvq.cu`, `mmf.cu` and `vecdotq.cuh`; no `vec_dot_turbo*` function exists anywhere; the only CUDA turbo dot products are the fused-attention helpers listed above. So `ggml_cuda_mul_mat()` (`ggml-cuda.cu:1819-1879`) falls through MMVF → MMF → MMVQ → MMQ → `ggml_cuda_mul_mat_cublas()` (`:1878`, called again from `:1833`), and that last one dequantizes via `ggml_get_to_fp16_cuda()` and picks the `F16` compute type for any quantized `src0` on a device with fast fp16 (`:1624-1629`). That is [[tq-1-missing-gemm-kernels]] in full; [[gemm-dispatch]] holds the decision table.

> Contradiction (2026-09-28): [[source-state-md]] §1.3 and §4 TQ-1 attribute 38.81 % (157 ms) of GPU time to `magma_sgemmEx_kernel<float, __nv_bfloat16>` and describe the fallback as "cuBLAS/MAGMA". A case-insensitive search for `magma` over `src/` returns **nothing** — there is no MAGMA source, no MAGMA link, no MAGMA symbol in this tree. The in-repo fallback is cuBLAS (`ggml_cuda_mul_mat_cublas` → `cublasGemmEx`/`cublasSgemm*`), and for a quantized operand its compute type is `F16`, selected at `src/ggml/src/ggml-cuda/ggml-cuda.cu:1626-1628`, with a `GGML_PREC_F32` override afterwards. Both the vendor name and the `__nv_bfloat16` template argument are therefore unexplained by this checkout [UNVERIFIED], and the causal link from that 157 ms to a turbo `MUL_MAT` is [INFERENCE] — the standard production graph consumes turbo K/V through the fused attention kernels, not through `mul_mat`. [[tq-1-missing-gemm-kernels]] reaches the same conclusion.

### 8. The CPU reference codec is a different codec for `turbo4`

`src/ggml/src/ggml-turbo-quant.c` is the reference implementation registered in the type traits. `turbo3` and `turbo2` use the sign-array butterfly (`turbo_cpu_fwht`, `:219-240`; sign arrays `:202-215`), with the group size read from the global `turbo3_cpu_wht_group_size` (`:19`, set by the CPU `SET_ROWS` handler; read at `:249-250`, `:344-345`). `turbo4` does not: `quantize_row_turbo4_0_ref` (`:429`) and `dequantize_row_turbo4_0` (`:539`) both call `turbo_init_rotation()` (`:67-117`), which generates a **random Gaussian matrix via an LCG + Gram-Schmidt QR** (seed `TURBO_SEED_ROTATION 42`) and applies it with `matvec()` (`:456`, `:492`, `:563`, `:597`) — a different transform from the signed Hadamard butterfly the CUDA encoder applies for the same type. On top of that, `dequantize_row_turbo3_0` (`:308-320`) is a stub that returns `centroid × norm` **without** un-rotating, with the comment "Stub — Metal shader handles dequant on GPU". Both facts are latent for a CUDA-only deployment (the KV cache lives in CUDA buffers and is read by the CUDA kernels), but any CPU-side quantize/dequantize round trip of a turbo tensor will not agree with the GPU codec.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/ggml/include/ggml.h:433-435` | the three type ids |
| `src/ggml/include/ggml.h:586` | `GGML_OP_TURBO_WHT` |
| `src/ggml/src/ggml-common.h:317-383` | `QK_TURBO*`, `QK_TURBO*_GROUP`, `NL_TURBO*`, `block_turbo3_0`/`_4_0`/`_2_0` and the size `static_assert`s |
| `src/ggml/src/ggml.c:708-731` | type traits (`blck_size`, `type_size`, `to_float`, `from_float_ref`) |
| `src/ggml/src/ggml.c:6634-6670` | `ggml_turbo_wht()` op builder |
| `src/ggml/src/ggml-turbo-quant.c` (627 lines) | CPU reference codec: `turbo_rotation`/QJL tables, centroid lookup, `quantize_row_turbo{3,2,4}_0_ref`, `dequantize_row_turbo{3,2,4}_0`, `quantize_turbo{3,2,4}_0` |
| `src/ggml/src/ggml-cuda/turbo-quant.cuh` (421 lines) | centroid + midpoint tables `:23-42`, `:297-310`; WHT sign arrays `:47-79`; sequential `turbo_fwht_128`/`_64` and their dead wrappers `:88-139`; InnerQ statics `:147-157`; `quantize_f32_turbo4_0_block` `:336`, `quantize_f32_turbo3_0_block` `:370`, `quantize_f32_turbo2_0_block` `:405`; `turbo*_dequant_element` `:348-421` |
| `src/ggml/src/ggml-cuda/set-rows.cu` | encoder kernels `k_set_rows_turbo3` `:237`, `k_set_rows_turbo2` `:608`, `k_set_rows_turbo4` `:952`; tail kernels `:422`, `:775`; launchers `:537`, `:884`, `:1104`; dispatch into `ggml-cuda.cu`'s `SET_ROWS` cases |
| `src/ggml/src/ggml-cuda/dequantize.cuh:154-176` | `dequantize_turbo4_0`/`_turbo3_0`/`_turbo2_0` |
| `src/ggml/src/ggml-cuda/convert.cu:664-669, 735-741, 772-777, 836-841` | fp16/fp32/bf16 conversion dispatch for the turbo types |
| `src/ggml/src/ggml-cuda/cpy.cu`, `src/ggml/src/ggml-cuda/getrows.cu` | cpy reaches the turbo dequantizers through `dequantize.cuh`; `GET_ROWS` has no turbo case |
| `src/ggml/src/ggml-cuda/turbo-wht.cu` / `.cuh` | `k_turbo_wht_f32<direction,group_size>` `:23-96`, tail pass-through `:100-114`, `ggml_cuda_turbo_wht` `:117-174`, launches `:151-160` |
| `src/ggml/src/ggml-cuda/ggml-cuda.cu:2092-2093`, `:5320-5336` | `GGML_OP_TURBO_WHT` dispatch; `supports_op` for `SET_ROWS` |
| `src/ggml/src/ggml-cuda/fattn-common.cuh:333, 387, 436, 778, 838, 895` | the only CUDA turbo dot products and V dequantizers |
| `src/ggml/src/ggml-cuda/fattn.cu:339-369, 382-399` | registered FA type pairs and the KV-type support predicate |
| `src/ggml/src/ggml-cuda/fattn-vec.cuh:87-95` | turbo K/V treated as "unquantized" for vec-kernel thread/row geometry |
| `src/ggml/src/ggml-cuda/CMakeLists.txt:121-139` | the 15 `fattn-vec-instance-turbo*` translation units listed for the build |
| `src/src/llama-kv-cache.cpp` | per-layer type selection + `TURBO_LAYER_ADAPTIVE` `:262-321`; head-dim padding `:323-360`; rotation/scale tensors `:370-378`, `:427-430`, `:536-541`; `wht_group` in `op_params` `:1586-1587`, `:1637-1638`, `:1663-1664` |
| `src/src/llama-graph.cpp:2700-2709, 2778-2787, 2977-2988, 3100-3111, 3292-3303` | Q forward rotation, output inverse rotation |
| `src/src/turbo-rotation-data.h:3, :2054` | `TURBO_ROTATION_RT` (inverse) and `TURBO_ROTATION_R` |
| `src/common/arg.cpp:304-345`, `:2442-2465` | allowed KV types and `-ctk`/`-ctv` |

## Known issues

- [[tq-1-missing-gemm-kernels]] — **CRITICAL**: the three turbo types are absent from every matmul fast path, so a `MUL_MAT` on a turbo operand dequantizes and lands in cuBLAS with no turbo-aware tiling (see §7).
- [[tq-4-wht-numerical-mismatch]] — the same rotation exists in five places (three live), including the dead sequential pair inside `turbo-quant.cuh`; the "FP ordering" claim in [[source-state-md]] was not reproducible, so the issue is duplication.
- [[tq-5-tail-elements]] — elements past the last full group get no rotation and no InnerQ on the encode side; unreachable while the 128-padding and `wht_group = 128` are in place.
- [[innerq]] — [[tq-2-innerq-host-state]], [[tq-3-innerq-multigpu]], [[tq-6-innerq-race]], [[tq-7-innerq-max-channels]]: all four concern the per-channel equalization that runs inside the `SET_ROWS` encoder, which is the write half of this codec.
- [[ta-1-wht-inversion-256]] — the K cache is stored *rotated*, so anything that reads it in the original basis must invert the rotation; the TriAttention GPU scoring kernel does not, for `head_dim = 256`.
- Stale "block size 32" comments in four files plus [[codebase-map]]'s 14 B/10 B block sizes — see the contradiction note in §2.
- The dense `turbo_rotation`/`turbo_rotation_inv` tensors are allocated and uploaded but read by nothing on the GPU path (§3), and the CPU `turbo4` codec rotates with a Gaussian QR matrix instead of the WHT (§8). Neither is filed as an issue.

## See also

[[overview]] · [[v100-sxm2]] · [[ternary-bonsai-2-27b]] · [[kv-cache]] · [[quantization]] · [[walsh-hadamard-transform]] · [[innerq]] · [[triattention]] · [[gemm-dispatch]] · [[kv-eviction]] · [[performance-profile]] · [[codebase-map]] · [[roadmap]] · [[upstream-lineage]] · [[benchmarks]] · [[prismml-weight-kernels]] · [[cuda-graphs]]
[[source-readme]] · [[source-state-md]]
[[tq-1-missing-gemm-kernels]] · [[tq-4-wht-numerical-mismatch]] · [[tq-5-tail-elements]] · [[tq-2-innerq-host-state]] · [[tq-3-innerq-multigpu]] · [[tq-6-innerq-race]] · [[tq-7-innerq-max-channels]]
