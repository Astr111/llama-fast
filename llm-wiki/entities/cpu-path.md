---
title: CPU Path
type: entity
status: current
updated: 2026-09-29
sources: []
verified:
  - src/ggml/src/ggml-cpu/repack.cpp
  - src/ggml/src/ggml-cpu/repack.h
  - src/ggml/src/ggml-cpu/arch-fallback.h
  - src/ggml/src/ggml-cpu/CMakeLists.txt
  - src/ggml/src/ggml-cpu/ggml-cpu.cpp
  - src/ggml/src/ggml-cpu/simd-mappings.h
  - src/ggml/src/ggml-cpu/vec.h
  - src/ggml/src/ggml-cpu/quants.c
  - src/ggml/src/ggml-cpu/quants.h
  - src/common/arg.cpp
  - src/common/common.h
  - src/common/common.cpp
  - src/ggml/include/ggml.h
  - src/src/llama-model.cpp
  - src/src/llama-model-loader.cpp
  - src/build-x64-linux-gcc-debug/CMakeCache.txt
  - src/ggml/src/ggml-turbo-quant.c
  - src/ggml/src/ggml-cpu/ops.cpp
  - src/ggml/src/ggml-quants.c
  - src/ggml/src/ggml-cpu/ggml-cpu.c
  - src/ggml/src/ggml-cuda/triattention-score.cu
  - src/ggml/src/ggml-cuda/triattention-score.cuh
tags: [cpu, quantization, walsh-hadamard-transform, triattention]
---

# CPU Path

## What it is

Everything the fork's custom operations do when the work is not on the GPU: the reference codec for `turbo2`/`turbo3`/`turbo4` (`src/ggml/src/ggml-turbo-quant.c`), the CPU twins of the rotation op (`GGML_OP_TURBO_WHT`) and of the plain WHT, the turbo `vec_dot` entries registered in the ggml CPU type traits, and — outside ggml — the host-side TriAttention scoring fallback that [[ta-3-cpu-fallback-transfers]] tracks.

It matters for two reasons. The ggml-level CPU machinery is the **only readable statement of the intended encode semantics** of the turbo formats: it runs the whole pipeline (norm → normalize → signed WHT → nearest centroid → corrected norm) in plain scalar code, so a future GPU kernel can be diffed against it. And the host-side TriAttention scoring is the degraded path that stalls 15–30 s per prune round ([[ta-3-cpu-fallback-transfers]]). Both are reachable in ways this page pins down; neither is "the same thing, just slower" — `turbo4` on the CPU is a *different codec* (see the verdict below).

## How it works

### 1. Entry points

| CPU entry point | What it implements | file:line | GPU counterpart it mirrors |
| :--- | :--- | :--- | :--- |
| `quantize_row_turbo3_0_ref` / `quantize_row_turbo2_0_ref` | full encode: group L2 norm → normalize → signed WHT (`turbo_cpu_fwht`) → nearest centroid → **corrected norm** `grp_norm/recon_norm` written to every block | `ggml-turbo-quant.c:240-310` (t3; group-size read `:249-252`, corrected norm `:305-310`), `:333-395` (t2; read `:344-348`) | `k_set_rows_turbo3` / `k_set_rows_turbo2` (`set-rows.cu:237` / `:608`, WHT `:324-351` / `:692-725`); same corrected norm at `set-rows.cu:402-407` |
| `quantize_row_turbo4_0_ref` | **alternate codec**: dense Gaussian-QR rotation (`turbo_rotation`, seed 42) instead of the WHT; default `TURBO4_USE_4BIT` → 4-bit centroids, no QJL (`#else` branch keeps legacy 3-bit + QJL) | `ggml-turbo-quant.c:429` (rotation init `:67-117`) | `k_set_rows_turbo4` (`set-rows.cu:952`, WHT `:1039-1062`) — **divergent**, see §2 |
| `dequantize_row_turbo3_0` / `_turbo2_0` / `_turbo4_0` | centroid × norm unpack (t3/t2); t4 additionally inverse-rotates with `matvec(turbo_rotation_t, …)` — again Gaussian, not WHT | `ggml-turbo-quant.c`, after `:310` (t3), after the t2 quantize (`:397-410`), `:539` (t4) [bounds unverified] | `dequantize_turbo3_0`/`_turbo2_0`/`_turbo4_0` (`dequantize.cuh:154-176`) |
| `quantize_turbo3_0` / `_turbo2_0` / `_turbo4_0` | row-batched wrappers for host-side conversion (`llama-quantize`) | `ggml-turbo-quant.c` (one after each dequantize) | none — quantization is host-side by construction |
| `turbo_cpu_fwht` | signed WHT butterfly: `x·s1` → `log2(g)` butterfly → `x·(1/√g)·s2`; sign arrays `turbo_cpu_s1/s2` | `ggml-turbo-quant.c:219-240`, arrays `:202-215` | `turbo_fwht_128`/`_64` (`turbo-quant.cuh:88-139`) plus the inline WHT inside the `set-rows` kernels |
| `ggml_compute_forward_turbo_wht_f32` | `GGML_OP_TURBO_WHT` graph op: forward (`direction 0`, Q pre-rotation) and inverse (`direction 1`, attention output), `group_size` from `op_params`, InnerQ `scale_inv` applied pre-rotation (fwd) / post (inv), 64-group uses the first 64 signs, tail elements copied unchanged | `ops.cpp:12253-12331` (own copy of the sign arrays `:12250-12251`; type switch `:12336-12342`; dispatch case `ggml-cpu.c:2157-2160`) | `ggml_cuda_turbo_wht` (`turbo-wht.cu:117-174`) |
| `ggml_compute_forward_fwht` | **plain unsigned** WHT (×`1/√n`, no sign arrays, SIMD butterfly) — substituted for `MUL_MAT` when src0 is flagged `GGML_HINT_SRC0_IS_HADAMARD` (WHT-domain weights) | `ops.cpp:12066-12140` (dispatch `:12142-12160`; hook inside `ggml_compute_forward_mul_mat`, `ggml-cpu.c:1332-1337`) | none verified in this scope [UNVERIFIED] |
| `ggml_vec_dot_turbo3_0_f32` / `_turbo2_0` / `_turbo4_0` | dequantize turbo row to f32, scalar f32 dot with the operand row; registered as `.vec_dot` in the ggml CPU type traits | `ggml-cpu.c:3528` / `:3548` / `:3567`; traits table `:440-457` (`vec_dot_type = F32`, `nrows = 1`) | fused FA KQ helpers `vec_dot_fattn_vec_KQ_turbo3_0` etc. (`fattn-common.cuh:333` / `:387` / `:436`) — those dequantize inline in registers |
| CPU TriAttention scoring | **none in ggml-cpu** — `grep triattention` over `ops.cpp` + `ggml-cpu.c` returns zero; scoring kernels exist only in CUDA. The CPU *fallback* is host code: `triattention_dequant_kv_head` (`src/src/llama-triattention.cpp:540`), one `ggml_backend_tensor_get` per KV cell (`:566`/`:572`) | n/a (grep result) / per [[ta-3-cpu-fallback-transfers]] | `triattention_score_kernel` (`triattention-score.cu:167-`) |

`src/ggml/src/ggml-quants.c` contains **no** turbo entries at all (grep: zero matches) — `ggml-turbo-quant.c` is a separate translation unit and the only home of the codec.

### 2. Reference-semantics verdict

**Faithful for `turbo2`/`turbo3`; not a reference at all for `turbo4`; and never a performance model.**

- For t2/t3 the CPU path is a complete, readable statement of the format: the encode pipeline and the corrected-norm formula match the CUDA encoder point for point (`:305-310` vs `set-rows.cu:402-407`), the WHT stage is the same `(1/√g)·D₂·H·D₁` with the same seed-42 sign arrays, and the dequant is a trivial centroid × norm. A future GPU port can be checked against it.
- It is scalar and stack-buffer-bound on purpose — the codec quantizes per group into `float buf[128]`, and the dot below uses a 16 KiB f32 scratch with an `n ≤ 4096` assert — so it is ground truth for *values*, never for speed.
- **`turbo4` is a different codec on the CPU**: `turbo_init_rotation` (seed 42) synthesises a dense random Gaussian matrix and QR-orthogonalises it (`:67-117`), and both `quantize_row_turbo4_0_ref` and `dequantize_row_turbo4_0` rotate with it via `matvec` — the CUDA `turbo4` encoder uses the signed WHT (`set-rows.cu:1039-1062`). The CPU t4 was the legacy 3-bit+QJL design (its `#else` branch survives); it cannot validate the GPU codec. Known in [[turboquant]] §8.
- Two stray comments misdescribe the code: `dequantize_row_turbo3_0` says "Stub — Metal shader handles dequant on GPU" above a fully implemented body, and the t4 dequant says "TODO: add proper 4-bit centroid table to C code (currently only in Metal)" directly above a defined `CENTROIDS_4BIT` table. Both are stale; the code is complete either way.

### 3. `ggml_vec_dot_turboN_0_f32`

All three live as `static` functions in `ggml-cpu.c` (`:3528`, `:3548`, `:3567`) and are reachable only through the ggml CPU type-traits `vec_dot` slots (`:440-457`). Each asserts `nrc == 1`, dequantizes the turbo operand into `float tmp[4096]` (`GGML_ASSERT(n <= 4096)` — a 16 KiB stack scratch; the sibling report of an "int8 scratch" is the packed `qs` block payload, not the scratch, which is f32), then runs a scalar `sum += tmp[i] * y[i]` loop. Interpretation: this is a **dequantize-then-dot reference implementation** — no quantized-arithmetic tricks, no SIMD, no block-norm factoring — so it is exactly the arithmetic a GPU port must reproduce, and the 4096 cap just mirrors the maximum head dim. It is the "CPU reference dot product" the vault keeps referencing.

**Reachable in a normal GPU run — yes.** The comment at `ggml-cpu.c:3526-3527` documents the trigger: "Used by CPU flash attention for models with D not supported by CUDA FA (e.g. D=192)". When the CUDA fused-attention kernel cannot handle the head dim, the flash-attention node is scheduled to the CPU backend inside an otherwise-GPU graph, and this `vec_dot` runs there. CUDA FA never uses it (its turbo KQ dots are the fused in-register helpers). So the ggml CPU path is *not* gated on GPU-init failure.

### 4. Reachability and the TA-3 cost

Two distinct "CPU paths":

1. **ggml-level (codec, `TURBO_WHT` op, `vec_dot`) — production-reachable alongside the GPU.** CPU FA for head dims CUDA FA rejects (above); any op the CUDA backend's `supports_op` denies (turbo `GET_ROWS` aborts on CUDA and `supports_op` admits turbo only for `SET_ROWS` — [[turboquant]] §5); and all conversion/quantization, which is host-side by definition.
2. **Host-side TriAttention scoring — only on GPU-init failure.** This is the [[ta-3-cpu-fallback-transfers]] path: `triattention_dequant_kv_head` pulls every candidate KV cell to host with one synchronous `ggml_backend_tensor_get` each — ~4096 separate ~1 KiB D2H transfers at `n_decode=4096`/`head_dim=256` — costing 15–30 s per prune round (severity HIGH, [[source-state-md]]). The host scorer is also the only consumer of the dense `turbo_rotation` tensors ([[turboquant]] §3), i.e. the CPU TA fallback inverts rotation with the Gaussian matrix, not the WHT.

So the TA-3 worth on this path is dominated by the transfer pattern, not the CPU arithmetic itself; and the ggml CPU machinery exists in normal runs regardless of GPU health.

### 5. WHT-stage structure: same as `[[turbo-wht]]`

Yes for everything the fork actually rotates. The CPU encode path and the `TURBO_WHT` op both implement the signed WHT exactly as the `[[turbo-wht]]`/`[[walsh-hadamard-transform]]` structure: `D₁` signs → `log2(g)` butterfly → `(1/√g)·D₂`, with two independent copies of the identical 128-element seed-42 arrays (`turbo_cpu_s1/s2` at `ggml-turbo-quant.c:202-215`; `turbo_wht_s1/s2` at `ops.cpp:12250-12251`; sampled prefixes match in this read, and both files declare the arrays "must match" the CUDA/Metal shaders). The **third variant** is not a signed-WHT variant at all: the plain `ggml_compute_forward_fwht` (`ops.cpp:12066`, unsigned, ×1/√n) used by the `GGML_HINT_SRC0_IS_HADAMARD` matmul path, and the CPU `turbo4` codec's dense Gaussian-QR rotation, which replaces the WHT entirely.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/ggml/src/ggml-turbo-quant.c` | the codec: `turbo_cpu_fwht` `:219-240`, sign arrays `:202-215`, centroid tables, `quantize_row_turbo{3,2,4}_0_ref`, `dequantize_row_turbo{3,2,4}_0`, `quantize_turbo{3,2,4}_0` wrappers, the WHT-group global `:19` (no writers inside `src/ggml` — grep finds only the declaration and the two reads `:249`/`:344`; [[turboquant]] puts the setter in the llama-layer `SET_ROWS` path) |
| `src/ggml/src/ggml-cpu/ops.cpp:12253-12342` | `ggml_compute_forward_turbo_wht_f32` + dispatch |
| `src/ggml/src/ggml-cpu/ops.cpp:12066-12160` | plain `ggml_compute_forward_fwht` (unsigned WHT) |
| `src/ggml/src/ggml-cpu/ggml-cpu.c:440-457` | ggml CPU type traits: `.from_float`, `.vec_dot`, `.vec_dot_type`, `.nrows` for the three turbo types |
| `src/ggml/src/ggml-cpu/ggml-cpu.c:3528, 3548, 3567` | `ggml_vec_dot_turbo{3,2,4}_0_f32` |
| `src/ggml/src/ggml-cpu/ggml-cpu.c:1332-1337, 2157-2160` | the `GGML_HINT_SRC0_IS_HADAMARD` hook inside `ggml_compute_forward_mul_mat`; the `GGML_OP_TURBO_WHT` dispatch case |
| `src/ggml/src/ggml-quants.c` | nothing turbo (verified by grep) |
| `src/src/llama-triattention.cpp:540` | host-side TA scoring fallback (per [[ta-3-cpu-fallback-transfers]]; not re-read this pass) |
| `src/ggml/src/ggml-cuda/triattention-score.cu` / `.cuh` | the GPU scoring kernel, for contrast — the only scoring kernels in the tree |

## Known issues

- [[ta-3-cpu-fallback-transfers]] — per-cell synchronous D2H transfers on the host-scoring path; 15–30 s stalls at realistic decode lengths.
- CPU `turbo4` is the legacy Gaussian-QR codec, not the WHT codec the GPU encodes with ([[turboquant]] §8) — no issue filed upstream of this page.
- The CPU vec_dot caps at `n ≤ 4096` (`ggml-cpu.c:3536` etc.) and is scalar; it is a reference, not a performance path — relevant to [[tq-1-missing-gemm-kernels]] only as ground truth.
- Stale comments ("Stub — Metal…", "TODO…only in Metal") on complete CPU code (§2).
- `turbo3_cpu_wht_group_size` has no setter inside `src/ggml`; unset, group size is inferred from row 128-alignment (`ggml-turbo-quant.c:251-253`) while the GPU path pins 128 via KV padding — a divergence surface for group semantics.
- The `GGML_HINT_SRC0_IS_HADAMARD` plain-WHT path has no verified CUDA counterpart in this scope.

## The CPU kernel layer: repacking and SIMD

Everything in §1–§5 above sits on a lower layer that reorders weights **before the graph runs**. That layer is not an op and not a codec: it is a *buffer type*.

### 1. What repacking is, when it happens, and what it costs

- `ggml_backend_cpu_repack_buffer_type()` (`src/ggml/src/ggml-cpu/repack.cpp:5463-5569`) defines a buffer type named `"CPU_REPACK"`, registered as an **extra buffer type of the CPU device** (`ggml-cpu.cpp:65-66`) and exported through the device proc-address `ggml_backend_dev_get_extra_bufts` (`ggml-cpu.cpp:660-661`).
- The model loader consumes it only when `use_extra_bufts` is set: the CPU buft list is built with it first (`llama-model.cpp:1044-1057`, `:1572`), and every weight tensor is placed in the first buffer type that accepts it (`select_weight_buft`, `llama-model-loader.cpp:1065-1071`).
- The switch is `--repack` / `-nr, --no-repack`, env `LLAMA_ARG_REPACK`; the lambda sets `params.no_extra_bufts = !value` (`src/common/arg.cpp:2426-2433`), the field lives in `common.h:573` (default `false` → repacking **on**) and becomes `mparams.use_extra_bufts = !params.no_extra_bufts` (`common.cpp:1719`). It does **not** change the quantization — it only permutes already-quantized blocks. (Direct `llama_model_params` users get the opposite default: `llama.cpp:452` sets `use_extra_bufts = false`.) Note: the parent's premise that [[runtime-switches]] lists `LLAMA_ARG_REPACK` is **refuted** — that page's `LLAMA_ARG_*` inventory (`entities/runtime-switches.md:39-47`) does not mention it, and no page in the vault did before this one.

**The layout change** is row interleaving: N rows of the same block type are fused into one `xN` super-block — `using block_q4_0x8 = block<4, 8>`, `block_q8_0x16 = block<8, 16>`, etc. (`repack.h:36-51`), with `static_assert` that the struct is exactly N source blocks wide (`repack.h:36-42`). The fork's `PQ2_0` gets one too: `using block_pq2_0x4 = block<2, 4>` = 4 halves + `QK_PQ2_0` bytes (`repack.h:42`, `:51`). Each `repack_*_to_*_N_bl()` copies row-major source blocks into interleaved positions (`repack_q4_0_to_q4_0_8_bl` `repack.cpp:4051`, `repack_pq2_0_to_pq2_0_4_bl` `:4146`), driven by the template table `repack<BLOC_TYPE, INTER_SIZE, NB_COLS>` (`:4528-4633`).

**Motivation** is the paired gemv/gemm kernels, which consume N rows per pass so that one quantized activation and one int→float conversion amortize over the tile — stated outright for Q1_0 at `repack.cpp:427-429`; the kernel specializations are `:4641-4831`, and the variant is a compile-time template instance from the table at `:5227-5276`.

**When: load time, once per tensor — never per graph.** `init_tensor` stores the selected trait in `tensor->extra` (`:5463-5467`); the buffer's `set_tensor` asserts `offset == 0` and `size == ggml_nbytes(tensor)` and calls `tensor_traits_base::repack(tensor, data, size)` (`:5469-5483`), i.e. the GGUF bytes are permuted while they are copied into the buffer. At compute time the buffer type only *vouches* for ops: `extra_buffer_type::supports_op` accepts `GGML_OP_MUL_MAT` (2-D src0) and `MUL_MAT_ID` (3-D src0) when src0 lives in the repack buffer and a trait was selected (`:5511-5559`).

**Memory cost: a second, size-for-size copy — nothing frees the original.** `alloc_buffer` allocates an ordinary CPU buffer of the requested size (`:5463-5569`), `get_alloc_size` is `nullptr` so the default `ggml_nbytes` applies, and the interleaved blocks are size-identical to their sources (`repack.h:36-42`) — so repacking adds **no** bytes, but it does not remove any either: the source bytes stay in the GGUF mapping. For every tensor that lands in `CPU_REPACK`, weight memory is therefore ≈2× (file-backed pages + anonymous host copy), and the buffer is **write-only to the runtime** — `get_tensor` and `cpy_tensor` are `nullptr` (`:5463-5569`) — which is why LoRA attachment must fall back to a normal CPU buffer for tensors in a repacking extra buffer ([[loading-and-batching]]). `[UNVERIFIED]`: whether the loader drops the mmap region for repacked tensors; `llama-model.cpp:1196-1232` shows `use_mmap` interacting with buffer choice but was not read to the end.

### 2. Which types are eligible — and the fork's are (mostly) not

`ggml_repack_get_optimal_repack_type()` (`repack.cpp:5270-5444`) is the entire eligibility rule: type → required CPU feature → row-interleave constraint.

| ggml type | Repacked when | Instance |
| :--- | :--- | :--- |
| `Q4_0` | AVX2, or SVE+`matmul_int8` with `sve_cnt == QK8_0`; NEON+`matmul_int8`; NEON+dotprod; RVV 256-bit | `q4_0_8x8_q8_0`, `…_4x8`, `…_4x4` (`:5278-5304`) |
| `Q4_K` | AVX2 or NEON (`ne[1] % 8 == 0`) | `q4_K_8x8_q8_K`, `…_8x4` (`:5305-5331`) |
| `Q2_K` | **AVX-512** (or RVV 256-bit) | `q2_K_8x8_q8_K` (`:5332-5348`) |
| `Q5_K`, `Q6_K` | NEON only — **never on x86** | `q5_K_…`, `q6_K_…` (`:5349-5370`) |
| `IQ4_NL` | AVX2 / NEON+dotprod / RVV | `iq4_nl_8x8_q8_0`, `…_4x4` (`:5371-5403`) |
| `MXFP4` | AVX2 / NEON+dotprod | `mxfp4_8x8_q8_0`, `…_4x4` |
| `Q8_0` | NEON (+`matmul_int8` or dotprod) / RVV only — **not on x86** | `q8_0_4x8_q8_0`, `…_4x4` (`:5404-5425`) |
| `Q1_0` (id 41) | AVX-512+VNNI, **or AVX2**, or NEON | `q1_0_4x8_q8_0`, `…_4x4` (`:5426-5437`, instances `:5260-5261`) |
| **`PQ2_0` (id 142)** | **AVX-512 *and* AVX-512 VNNI** and `ne[1] % 4 == 0` — the last branch of the selector | `pq2_0_4x8_q8_0` (instance `:5264`, kernels `:4720`/`:4829`) |

Fork types **absent from every repack table**: `Q2_0` (42), `TURBO3_0` (43), `PTQ1_0` (143). No `repack_*_bl` function, no `tensor_traits` instance and no selector branch mentions them; searching `src/ggml/src/ggml-cpu` for them returns only the type-traits table (`ggml-cpu.c:252-262` for PQ2_0/PTQ1_0, `:440-457` for the turbo types) and the codec/host-quantizer declarations (`quants.h:17-18`). Type ids from `ggml.h:431-439`.

**Verdict, and the caveat this forces on §1–§5.** `--repack` is enabled by default and, on this machine, **does nothing for the target model**:

- the target's *weight* type `PQ2_0` **is** in the table — the parent's expectation that the fork's types are simply absent is **refuted for PQ2_0** — but its branch demands AVX-512 + AVX-512 VNNI; this workstation (`i5-14600KF`; `/proc/cpuinfo` shows `avx avx2 fma f16c` and **no** `avx512f`) fails the test at compile time, `ggml_cpu_has_avx512()` is false, the selector returns `nullptr`, and the weight stays in a plain CPU buffer;
- the target's *KV* types (`TURBO3_0`, and `Q2_0` as the ternary weight format beside it) have **no branch at all**.

So the correction to this page's opening claim: the CPU path mirrors the GPU **codec** (encode/decode arithmetic, §2), but it does **not** mirror the GPU **kernel layer** — no turbo repack exists, and the one fork type that does have a repack slot is on the wrong side of this host's ISA.

### 3. The SIMD surface — and the sources that are missing from this tree

- **ISA selection is compile-time.** `vec.h` pulls in `simd-mappings.h` (`vec.h:6`) and that header defines `GGML_SIMD` once per architecture by preprocessor alone: ARM SVE (`simd-mappings.h:172-174`), NEON+FP16 arithmetic (`:331-333`), **AVX-512F** (`:446-448`), **AVX** (`:581-583`), POWER9 (`:685-687`), wasm simd128 (`:788-790`), SSE3 (`:900-902`), RISC-V V (`:1278-1282`). Runtime feature probes (`ggml_cpu_has_avx2()`, `ggml_cpu_has_avx512()`) select a *repack variant* (`repack.cpp:5278-5444`) or gate op support — they never widen a kernel at run time.
- **The build** appends the per-arch sources for x86 (`ggml-cpu/arch/x86/quants.c`, `arch/x86/repack.cpp` — `ggml-cpu/CMakeLists.txt:242-245`) and the ISA flags (`-mavx2`, `-mavx512f`, `-mavx512vnni`, … and `ARCH_DEFINITIONS GGML_AVX2 …`, `:335-363`). This machine's in-tree debug build has `GGML_NATIVE=ON` with `GGML_AVX2:BOOL=OFF`/`GGML_AVX512:BOOL=OFF` (`src/build-x64-linux-gcc-debug/CMakeCache.txt`); under `GGML_NATIVE` the compiler's `-march=native` decides, and an i5-14600KF yields SSE/AVX2-class code with no AVX-512.
- **The SIMD sources are not in this tree.** `find src/ggml/src/ggml-cpu/arch -type f` returns **0 files**, `git ls-files | grep ggml-cpu/arch` returns only `arch-fallback.h`, and the `arch/` directory mtime is a day later than the rest of `ggml-cpu` (Sep 27 vs Sep 26) — yet `CMakeLists.txt:243-244` lists both x86 files unconditionally, and the x86_64 block of `arch-fallback.h` (`:100-129`) does **not** alias `ggml_gemv_pq2_0_4x8_q8_0_generic` / `ggml_gemm_pq2_0_4x8_q8_0_generic` (those aliases exist only for the all-generic block `:64`/`:83`, ARM `:95`/`:99`, PowerPC `:168`/`:187`), i.e. the tree expects native x86 definitions that are not present. `[INFERENCE]` no x86 SIMD quant/repack kernel can be read in this checkout and a build from this tree as-is cannot compile the sources its own CMake lists. `[UNVERIFIED]` whether the measured build predates the files' disappearance.
- **What that means for the 94 % host wall time** ([[first-live-measurements]] §CPU, where the host cost is attributed to `quantize_q8_1`/MMQ activation work): the repack layer cannot have contributed for the target's types on this host, so the fork's own CPU arithmetic in that run is the **scalar** codec/`vec_dot` of §3, while the surrounding upstream activation quantization and the Q4_0/Q4_K/IQ4_NL/MXFP4 repack paths are at best **AVX2-class** (`repack.cpp:5278-5403` requires `ggml_cpu_has_avx2()`; the CPU has AVX2/FMA/F16C). Nothing on this host is AVX-512 and nothing here is a SIMD-optimal host path; `[UNVERIFIED]` whether `--repack` was in play during that measurement.

### 4. Relation to the CUDA kernels, per fork type

| Type | CPU repack | CPU arithmetic | CUDA counterpart | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `TURBO2_0`/`TURBO3_0`/`TURBO4_0` | none | `.from_float`/`.vec_dot` slots only (`ggml-cpu.c:440-457`), dequantize-then-scalar-dot (`:3528-3580`), codec in the separate TU `ggml-turbo-quant.c` | `set-rows.cu`, `dequantize.cuh`, fused FA KQ (`fattn-common.cuh`) — §1 | **Reference codec + reachable fallback arithmetic**, never a kernel-layer mirror |
| `PQ2_0` (142) | **yes** — `repack_pq2_0_to_pq2_0_4_bl` (`repack.cpp:4146`), instance `:5264`, gemv/gemm `_4x8_q8_0` (`:4720`, `:4829`; declared `repack.h:174`, `:193`) | plain dot `ggml_vec_dot_pq2_0_q8_0` (`ggml-cpu.c:1229`, inside the Q8_K-eligibility helper `:1207-1217`), generic at `quants.c:235` | MMQ/`mmq-hopper-q1.cu` units, whose `repack_q2_dense` is a **different** repack — in-kernel dense bit-word packing, not a buffer type ([[prismml-weight-kernels]], [[quantized-kernel-units]]) | The only fork type with a CPU repack slot, and it is **unreachable on this host** (AVX-512 gate). Native x86 SIMD for its 4x8 kernels is implied by the missing fallback alias but its ISA is `[UNVERIFIED]` |
| `PTQ1_0` (143) | none | `quantize_row_ptq1_0` (`quants.c:37-39`), one generic dot (`quants.c:285`) | PrismML ternary units ([[prismml-weight-kernels]]) | **Fallback only.** `arch-fallback.h:97-98` says so in as many words: "PTQ1_0 currently has only the generic vec_dot; alias it here until a SIMD version lands" |
| `Q2_0` (42) | none | `quantize_row_q2_0` (`quants.c:29`) + `ggml_vec_dot_q2_0_q8_0_generic` (`quants.c:185`) | per [[prismml-weight-kernels]] | Scalar **even on x86**: `arch-fallback.h:104` aliases the generic for x86_64. No SIMD in this tree |

Cross-reference: §1 already lists the codec entry points and §3 the scalar `vec_dot`s — this section is the layer *below* them and adds two corrections. First, "the CPU codec is the reference for values" still holds, but the CPU **kernel layer** is not a mirror of the CUDA kernel set, and `--repack` is not a route to making it one. Second, one stale comment worth flagging: the PQ2_0 instance is introduced as `// instance for Q2_0` (`repack.cpp:5263`) while the type is `block_pq2_0` / id 142 — the same class of naming noise as the two stale CPU-codec comments in §2.

## See also

[[quantization]] · [[turboquant]] · [[turbo-wht]] · [[walsh-hadamard-transform]] · [[ta-3-cpu-fallback-transfers]] · [[backend-parity]] · [[tq-1-missing-gemm-kernels]] · [[triattention]] · [[ta-1-wht-inversion-256]]