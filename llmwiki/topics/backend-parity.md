---
title: Backend parity
type: topic
status: current
updated: 2026-09-28
sources: []
verified: [src/ggml/src/ggml-vulkan/vulkan-shaders/dequant_ptq1_0.comp, src/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_ptq1_0.comp, src/ggml/src/ggml-vulkan/ggml-vulkan.cpp, src/ggml/src/ggml-vulkan/vulkan-shaders-gen.cpp, src/ggml/src/ggml-sycl/fwht.hpp, src/ggml/src/ggml-sycl/fwht.cpp, src/ggml/src/ggml-sycl/vecdotq.hpp, src/ggml/src/ggml-sycl/mmvq.cpp, src/ggml/src/ggml-sycl/convert.cpp, src/ggml/src/ggml-sycl/cpy.cpp, src/ggml/src/ggml-sycl/getrows.cpp, src/ggml/src/ggml-sycl/dequantize.hpp, src/ggml/src/ggml-sycl/ggml-sycl.cpp, src/ggml/src/ggml-metal/ggml-metal-ops.cpp, src/ggml/src/ggml-cuda/fattn-common.cuh, src/ggml/src/ggml-cuda/fattn-vec.cuh, src/ggml/src/ggml-cuda/fattn.cu, src/ggml/src/ggml-cuda/triattention-score.cu, src/ggml/src/ggml-cuda/turbo-quant.cuh, src/ggml/src/ggml-cpu/ggml-cpu.c, src/ggml/src/ggml.c]
tags: [backend, quantization, wht]
---

# Backend parity

## Bottom line

The fork's custom surface splits into two portability classes, and the split decides which of its claims rest on a second opinion.

**The PrismML weight formats are genuinely ported.** `PTQ1_0` is implemented independently in four backends — CUDA, Vulkan, SYCL and Metal — and `PQ2_0`/`Q1_0`/`Q2_0` in most of them. Where four copies of a decoder agree, the layout is no longer one author's private convention: the copies collectively *are* the spec.

**The TurboQuant KV types and TriAttention are not ported.** `GGML_TYPE_TURBO2_0/3_0/4_0` are registered by CUDA and by a CPU reference only; no other GPU backend mentions the type at all. TriAttention is CUDA-only. So for `turboN` the sentence "the format is defined by this repo" is *literally* true in the weak sense: it is defined by one implementation plus a dequantizing CPU fallback, and there is no second implementation to arbitrate the intended numerics when the CUDA one is suspect ([[tq-1-missing-gemm-kernels]], [[turboquant]]).

There is also a naming trap. The WHT that Vulkan, SYCL and Metal implement is **not** the fork's TurboQuant WHT: it is an FWHT folded into matmul dispatch, selected by the `GGML_HINT_SRC0_IS_HADAMARD` MUL_MAT parameter. The standalone transform op is `GGML_OP_TURBO_WHT` and it is CPU+CUDA only. `GGML_OP_FWHT` does not exist anywhere in this tree.

## Evidence

### Who implements what

| Backend | Fork-specific surface | Files |
| :--- | :--- | :--- |
| CUDA | `PTQ1_0`/`PQ2_0` weight kernels; the **only** turbo-KV dot products (`vec_dot_fattn_vec_KQ_turbo{3,2,4}_0`, plus turbo dequant / V-dequant); turbo `set_rows`; TriAttention scorer; `GGML_OP_TURBO_WHT` | `src/ggml/src/ggml-cuda/fattn-common.cuh`, `fattn-vec.cuh`, `fattn.cu`, `convert.cu`, `set-rows.cu`, `triattention-score.cu`, `turbo-quant.cuh`, `ggml-cuda.cu` |
| CPU | reference `quantize_row_turboN_0_ref` and a `vec_dot` for turbo2/3/4 that dequantizes into a ≤4096-element scratch then accumulates scalars; `GGML_OP_TURBO_WHT` reference | `src/ggml/src/ggml-cpu/ggml-cpu.c` (`GGML_TYPE_TURBO3_0/2_0/4_0` traits, `ggml_vec_dot_turbo3_0_f32` at ~3537) |
| Vulkan | `dequant_ptq1_0` shader; `matmul_ptq1_0_f32` / `matmul_pq2_0_f32` (+ `mul_mat_id` variants) scalar and coopmat1; a **dedicated** integer-dot mmvq for `PTQ1_0` (`mul_mat_vecq_ptq1_0.comp`); FWHT pipelines for the hadamard-hint matmul (8 widths, f32/f16, subgroup and shared-memory forms) | `dequant_ptq1_0.comp`, `mul_mat_vecq_ptq1_0.comp`, `ggml-vulkan.cpp` (4765, 4813, 4906, 4957, 5007, 10058-10146), `vulkan-shaders-gen.cpp` (596, 774-779) |
| SYCL | `dequantize_ptq1_0`/`pq2_0`, same-quant `cpy` kernels, `get_rows`, dedicated `vec_dot_ptq1_0_q8_1` with multi-column mmvq up to `ncols_dst = 8`; FWHT fast path for the hadamard-hint matmul (sizes 64/128/256/512) | `ggml-sycl/dequantize.hpp:126-159`, `vecdotq.hpp:346-441`, `mmvq.cpp:1281-1340`, `convert.cpp:659-876`, `cpy.cpp:959-1401`, `getrows.cpp:279`, `fwht.cpp`, `fwht.hpp`, `ggml-sycl.cpp:4536-4540` |
| Metal | `PTQ1_0`/`PQ2_0`/`Q2_0` in the matvec type table, with a separate `ggml_metal_ptq1_multicol_enabled` path; `GGML_OP_FWHT`-equivalent (`ggml_metal_op_fwht`) plus a fused sign-flip + FWHT variant | `src/ggml/src/ggml-metal/ggml-metal-ops.cpp:2450-2573, 2682-2684, 2801-2803, 4075` |
| everything else (OpenCL, BLAS, …) | nothing | — |

**Turbo KV dot product outside CUDA: none.** A `ggml_type` scan for `GGML_TYPE_TURBO*` over all of `src/ggml/src/` returns hits only in `ggml.c` (type table + `from_float`), `ggml-cpu/ggml-cpu.c`, and `ggml-cuda/*`. Vulkan, SYCL and Metal do not merely implement it *worse* — they never register the type. The CPU path is not a port either: its `vec_dot` calls `to_float` into a scratch buffer (`GGML_ASSERT(n <= 4096)`) and then does a scalar accumulate, i.e. a fallback that destroys the whole point of the compressed cache.

### What the non-CUDA copies establish about the intended semantics

- **The layout, in prose.** `dequant_ptq1_0.comp` is the most readable statement of `PTQ1_0` in the tree: a block is 28 bytes = `qs[24]` (5 base-3 trits per byte) + `qh[2]` (4 trits per byte) + fp16 `d`, and the element order is *not* positional —
  `qs[j], j<16 → element 16t + j (t = 0..4)`, `qs[16+j], j<8 → element 80 + 8t + j`, `qh[h] → element 120 + 2n + h`.
  Decode is repeated `v = (v*3) & 0xFF` with the trit at `(v*3) >> 8`, value `trit - 1`, weight `(trit-1)*d`. The shader's own comment says the index maths "lives in one place rather than being reproduced per shader" and that element order "follows the CPU codec" — so the Vulkan copy declares itself a statement of the shared contract, not an independent design.
- **The numerics, in prose.** `mul_mat_vecq_ptq1_0.comp` corroborates the same order and adds the algebra: weights are `{-1, 0, +1}` times a block scale, so with `q8_1` activations the dot is `Σ (q_e - 1)·y_e = Σ_k ( d_b[k]·Σ q_e·q8_e - s_b[k] )` with `s_b = d_b·Σ q8`. The `-1` offset is never materialized; it is folded into per-sub-block sums. That is the cleanest definition of what `PTQ1_0` *means* numerically, and it is portable pseudo-code: `PTQ1_0_DWORDS = 7` (28 bytes), `Q8_1_X4_DWORDS = 36`, `CG = 3` columns per pass ("3 measured best on RDNA2").
- **A third independent copy of the order.** SYCL's `ptq1_0_trit` + `dequantize_ptq1_0` reproduce the same `e < 80` / `80 ≤ e < 120` / `e ≥ 120` branch structure in a different language family, and `vec_dot_ptq1_0_q8_1` widens bytes into 16-bit lanes "so multiply-by-three cannot carry between bytes" — the same carry hazard the Vulkan `PTQ1_0_MUL3_PK16` variant addresses.
- **The rotation is a conversion-time contract, not a runtime op.** SYCL declines float→`PTQ1_0`/`PQ2_0` copies with the comment that both types "are produced offline by the converter, which also applies the Hadamard rotation the packing assumes"; `ggml_sycl_cpy` then takes the float→quantized branch and would assert, so the pair is declined and the scheduler falls back. Only quant→same-quant copies are implemented. This matches [[prism-hadamard-weight-fold]] and the converter-side contract written down in [[source-hadamard-tied-output]] (the embedding carries the `inverse-after-lookup` record; the head's activation transform is registered at load time).
- **Known convergence limit.** Vulkan carries no `PTQ1_0` decoder in `dequant_funcs_cm2.glsl`, so `vulkan-shaders-gen` skips coopmat2 generation for `ptq1_0`/`pq2_0`; they use the scalar and coopmat1 matmul paths instead (`vulkan-shaders-gen.cpp:596`, `ggml-vulkan.cpp:4649`). `PTQ1_0` also gets its own `load_vec_quant = 8` and a dedicated mmvq shader rather than the generic `mul_mat_vecq.comp`.

### The gap, and what it means for "the format is defined by this repo"

| Surface | Copies | Consequence |
| :--- | :--- | :--- |
| `PTQ1_0` / `PQ2_0` layout | CUDA + Vulkan + SYCL + Metal (+ CPU codec the shaders cite) | the format is portable and its semantics are pinned by agreement; a disputed CUDA detail can be settled by reading the Vulkan shader |
| FWHT as a matmul fast path | Vulkan + SYCL + Metal | portable, but a *different* mechanism from the fork's transform op — see below |
| `GGML_OP_TURBO_WHT` | CPU + CUDA | not portable; the InnerQ-scaled transform exists on two backends |
| `turbo2/3/4` KV cache | CUDA (+ non-portable CPU dequant fallback) | the type is *defined* by one implementation; no second opinion on rounding, block order, or rotation handling; any other backend rejects the cache type outright |
| TriAttention | CUDA only | eviction is unavailable on every other backend; the [[triattention]] mechanism has no reference implementation at all |

For the claim "the format is defined by this repo": it holds strongly for `PTQ1_0`/`PQ2_0` (four backends agree, and the rotation/sign metadata is separately pinned by the converter contract in `hadamard_packing.json` schema 3 / `prism.hadamard.version = 2`) and weakly for `turboN`, where "this repo" means "this one CUDA kernel plus a CPU path that dequantizes". The absence of a second implementation is why [[tq-1-missing-gemm-kernels]] is a hard problem rather than a porting exercise: there is no independent statement of the intended `turbo*` GEMM numerics to port *to* MAGMA or to a new kernel.

### WHT stage structure — cross-checked, not assumed

Two different transforms live in this tree and are easy to conflate:

1. **`GGML_OP_TURBO_WHT`** — a standalone op built in `ggml.c:6652`: `src[0] = a`, `src[1] = scale` with the comment `// InnerQ scale_inv (NULL = no scaling)`. CUDA accepts it only for `F32→F32` with `ne[0] % 64 == 0`; CPU has a reference implementation. `GGML_OP_FWHT` — a name earlier notes use — does not exist here: an exhaustive grep over `src/ggml/src/` returns **zero** hits for it.
2. **The hadamard-hint FWHT** — not an op at all, but a fast path inside MUL_MAT dispatch, gated on the matmul's own parameter: Vulkan `ggml_vk_can_use_fwht` checks `ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD`; SYCL checks the same hint and calls `ggml_sycl_op_fwht(ctx, src1, dst)`; SYCL's header states `src0 is not read at all` — the Hadamard matrix is never materialized. It serves whichever matmul the hint is set on, and both candidates exist in this tree: the PrismML folded-weight path ([[prism-hadamard-weight-fold]]) and the K-cache rotation, which `src/common/kv-mean-center.cpp:43-44` describes as “a mul_mat against the `attn_inp_k_rot` input”. **Which of the two sets `GGML_HINT_SRC0_IS_HADAMARD` was not determined here** — `[UNVERIFIED]`. Either way it has nothing to do with a turbo KV cache.

So the answer to "does any non-CUDA backend have the same WHT-stage structure as [[turbo-wht]]?" is **no**, with one partial caveat: `ggml_sycl_op_fwht` takes a scalar `scale` argument threaded into its kernel, which is structurally closer to `TURBO_WHT`'s `scale_inv` input than Vulkan's or Metal's (Metal's `ggml_metal_kargs_fwht` carries only `nrows` and `n_blk`). Where that SYCL `scale` comes from was not read, so the resemblance is `[UNVERIFIED]`.

The three hadamard-hint ports also differ among themselves in ways that show they are independent code, not a shared generator: Vulkan builds 8 widths with a subgroup-vs-shared-memory split keyed on `n / subgroup_size` (and gates out Intel Windows drivers in `[32.0.101.8509, 32.0.101.8860)`), SYCL hardcodes 64/128/256/512 with `rows_per_block = 4`, Metal's supported-size list "must stay in sync with the `kernel_fwht_f32_<N>` templates in `ggml-metal.metal`" and it additionally fuses a preceding sign-flip MUL into the transform's row load (`ggml_metal_op_can_fuse_fwht_signed`, guarded by `ctx->use_fusion`).

## Open questions

- Where the SYCL FWHT `scale` comes from, and whether it is the InnerQ `scale_inv` of `GGML_OP_TURBO_WHT` or an unrelated normalization. `[UNVERIFIED]`
- Which sizes the Metal FWHT supports (`ggml_metal_fwht_supported_size`) and whether its gating matches Vulkan's `n/subgroup_size` heuristic. `[UNVERIFIED]`
- `PQ2_0` outside CUDA is only confirmed as *registered* (Vulkan pipeline tables, SYCL `dequantize_pq2_0`/`vec_dot_pq2_0_q8_1`, Metal type table); its non-CUDA layout statement was not read and no shader comment documents it the way `dequant_ptq1_0.comp` documents `PTQ1_0`. `[UNVERIFIED]`
- Is the CPU `vec_dot_turboN_0_f32` intended as the reference semantics for a future GPU port, or is it purely a correctness fallback? Its ≤4096-element scratch and scalar loop suggest the latter.
- Does any second implementation of the turbo KV layout exist outside this repo (in the TurboQuant fork named by [[upstream-lineage]])? If so the "defined by this repo" worry is weaker than it looks — the checkouts were not available here.

## See also

[[upstream-lineage]] · [[prismml-weight-kernels]] · [[turboquant]] · [[turbo-wht]] · [[gemm-dispatch]] · [[quantization]] · [[triattention]] · [[source-hadamard-tied-output]] · [[source-kv-mean-center]]
