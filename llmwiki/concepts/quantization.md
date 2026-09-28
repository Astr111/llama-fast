---
title: Quantization
type: concept
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/ggml/src/ggml-common.h, src/ggml/src/ggml.c, src/ggml/src/ggml-cuda/turbo-quant.cuh, src/ggml/src/ggml-cuda/turbo-innerq.cuh, src/ggml/src/ggml-cuda/turbo-innerq.cu, src/ggml/src/ggml-cuda/dequantize.cuh, src/ggml/src/ggml-cuda/convert.cu, src/ggml/src/ggml-cuda/vecdotq.cuh, src/ggml/src/ggml-cuda/fattn-common.cuh, src/ggml/src/ggml-cuda/set-rows.cu, src/src/llama-kv-cache.cpp, src/common/arg.cpp]
tags: [quantization, kv-cache, turboquant, innerq]
---

## Definition

Representing a tensor with fewer bits per element than its compute precision, plus a per-block scale that restores magnitude. A quantised block stores `N` integer codes and one `fp16` norm; dequantisation is `centroid[code] * norm`.

This project applies it to the **KV cache**, not to weights (weights arrive already quantised in the GGUF — the target model is `PQ2_0`). The KV cache is quantised *as it is written*, per token, by `SET_ROWS` kernels; the attention kernels then consume the compressed cache directly, and nothing writes back a decompressed copy.

Three TurboQuant families are registered as ggml types (`src/ggml/src/ggml.c:708-731`), each with `blck_size = QK_TURBO* = 128`:

| Type | CLI name | Layout (`QK_TURBO* = 128` elements/block) | Bits/value |
| :--- | :--- | :--- | :--- |
| `GGML_TYPE_TURBO3_0` | `turbo3` | `norm` fp16 + `qs[32]` (2-bit) + `signs[16]` (1-bit) = **50 B** | 3.125 |
| `GGML_TYPE_TURBO4_0` | `turbo4` | `norm` fp16 + reserved `rnorm` fp16 + `qs[64]` nibble-packed 4-bit = **68 B** | 4.25 |
| `GGML_TYPE_TURBO2_0` | `turbo2` | `norm` fp16 + `qs[32]` (2-bit) = **34 B** | 2.125 |

(Declared at `ggml-common.h:329-334`, `:349-354`, and `:379-383`; sizes are the `static_assert` sizes at `ggml-common.h:334`, `:354`, `:383`. The byte-count comments inside those structs say "14 bytes", "8 bytes" and "2.5 bits/value" — those are stale, left over from a 32-element block; the array dimensions and the `static_assert`s are the authoritative figures. Against fp16 that is 5.1× / 3.8× / 7.5× compression.)

Three properties distinguish this from `q8_0`-style block quantisation:

1. **The block is the rotation group.** `QK_TURBO3 == QK_TURBO3_GROUP == 128` (`ggml-common.h:324-325`), commented "one block per rotation group, eliminates redundant norms". So the 128 values sharing one norm are the same 128 values that went through one Walsh-Hadamard rotation before quantisation — the transform that Gaussianises the vector so a fixed scalar codebook is near-optimal ([[walsh-hadamard-transform]]).
2. **The codebook is fixed and tiny.** Values are indices into Lloyd-Max centroids for `N(0, 1/128)`: 4 levels for turbo2 (`turbo-quant.cuh:23-28`), 8 for turbo3 (`:33-39`). Dequantisation is therefore a table lookup and a multiply — `turbo3_dequant_element` (`turbo-quant.cuh:386-392`) reassembles the 3-bit index from `qs` and `signs`.
3. **The tail is not rotated.** `k_set_rows_turbo3_tail` (`set-rows.cu:422`) covers `head_dim % GROUP_SIZE != 0` and quantises those elements without WHT; the target model's `head_dim=256` is exactly `2 × GROUP_SIZE`, so it is aligned — but `head_dim=576`-class models are not ([[tq-5-tail-elements]]).

### The `q8_0` values path

Keys and values are configured independently (`-ctk` / `-ctv`, `src/common/arg.cpp:2442`). The recommended pairing is turbo3 keys with `q8_0` values, and the README's justification is explicit: `q8_0` values "eliminate 64 inverse WHT kernels per token" ([[source-readme]], *SPEED PROFILE*). Working through the mechanism: a turbo-typed K or V must be *un*-rotated (inverse WHT) before its scores or weighted sums are meaningful, so a turbo value tensor buys memory at the cost of one inverse transform per layer per token; `q8_0` values are already in the model's own basis and need no transform at all. The mixed cases are instantiated in the fused attention kernels — `FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)` and its mirror at `fattn.cu:342-343`, plus turbo2 (`:349-350`) and turbo4 (`:360-361`) variants. The layer-adaptive modes in `llama-kv-cache.cpp:264-320` then let boundary layers use `q8_0` while the bulk uses turbo2 (mode 7 is the default for turbo2-V caches, `llama-kv-cache.cpp:283-287`).

### InnerQ equalization

Per-channel rescaling of K before rotation, so that channels with larger variance do not dominate the fixed codebook. The identity it preserves is stated in the source: `<Q/s, s*K> = <Q, K>` (`turbo-quant.cuh:141-144`) — Q is divided by the same per-channel factor that K is multiplied by, so the dot product is unchanged and the equalisation costs nothing in accuracy terms. It is opt-in via the `TURBO_INNERQ=N` environment variable (`turbo-quant.cuh:160-170`) with `TURBO_INNERQ_STRENGTH` (default `0.5`), calibrates by accumulating `x[j]*x[j]` during `SET_ROWS` (`set-rows.cu:287-290`), and publishes the resulting inverse scales through the cross-TU API in `turbo-innerq.cuh`. `INNERQ_MAX_CHANNELS` is **128** (`turbo-innerq.cuh:7`), so for the target model's `head_dim=256` only half the channels can be equalised ([[tq-7-innerq-max-channels]]).

## Why it matters here

- **It is the project's reason to exist.** The KV cache is the only tensor whose size grows with the conversation; compressing it is what lets a 27B model hold long context on a 16 GB V100. The README quotes ~25 200 tokens per 1 GB VRAM for `turbo3`+`turbo2` ([[source-readme]], *MAXIMUM VRAM COMPRESSION PROFILE*).
- **Compression and throughput are separate problems.** Quantised storage is cheap; *arithmetic on quantised storage* is what costs. A type with no native dot product must be dequantised to F16 before the general GEMM can run — see [[gemm-dispatch]] for the exact fallback chain and [[tq-1-missing-gemm-kernels]] for the turbo-specific gap.
- **On Volta the fallback is unusually expensive.** The hardware has FP16 MMA but no INT8/INT4 MMA (Turing+) and no BF16 MMA (Ampere+) — `turing_mma_available` (`common.cuh:348`) and `ampere_mma_available` (`common.cuh:356`) both evaluate false at `cc = 700` ([[v100-sxm2]]). A bit-packed type like turbo3 has nothing to accelerate it except `__dp4a`-class byte math and FP16 tensors, so a missing `vec_dot` becomes a bulk dequantise plus a full-precision GEMM rather than a mildly suboptimal kernel.
- **Native dots exist only inside attention.** `vec_dot_fattn_vec_KQ_turbo{3,2,4}_0` (`fattn-common.cuh:968-974`) and the `dequantize_V_turbo*` counterparts (`fattn-common.cuh:996-1002`) consume the compressed cache in-place. `vecdotq.cuh` — the file that feeds MMVQ/MMQ — contains **zero** occurrences of `TURBO`. The same types are first-class in one code path and absent from the other.

## Tradeoffs

- **Bits vs fidelity, per channel.** turbo2 (2.125 bits) is ~1.5× smaller than turbo3 and ~2× smaller than turbo4's 4.25; each halving of the codebook costs centroid resolution that the rotation is meant to make tolerable. Which is survivable is model-dependent, which is why the type is a CLI flag.
- **Rotation cost is paid per token, twice.** A forward WHT on write, and an inverse on read for any consumer that needs un-rotated values. Choosing `q8_0` values is precisely the decision to spend memory to cancel the read-side rotation.
- **Two WHT implementations exist.** A sequential `turbo_fwht_128` in the header (`turbo-quant.cuh:88`) and a parallel shared-memory butterfly in `set-rows.cu:332-343` with `__syncthreads()`. Floating-point summation order differs between them; [[tq-4-wht-numerical-mismatch]] tracks the risk.
- **Equalisation is a calibration problem.** InnerQ needs enough tokens to estimate channel variance, its state lives in header-scope statics, and its channel ceiling is half the target model's head width — three separate defects ([[tq-2-innerq-host-state]], [[tq-3-innerq-multigpu]], [[tq-6-innerq-race]], [[tq-7-innerq-max-channels]]).
- **A quantised type is not self-describing to the backend.** `ggml_is_quantized` is `true` for turbo (`ggml.c:1399`, traits at `ggml.c:708/716/724`), which is enough to pass some dispatch gates and not enough to have a kernel — the mismatch is the subject of [[gemm-dispatch]].

## Open questions

- The byte-count comments inside `block_turbo3_0` / `block_turbo2_0` disagree with their own array dimensions (they describe a 32-element block while `QK_TURBO* = 128`). Which figure the shipped `QQ`/calibration tooling assumes is `[UNVERIFIED]`.
- Does InnerQ actually activate in a default run? Its state is header-scope `static` shared across translation units ([[tq-2-innerq-host-state]]), so activation depends on link order — `[UNVERIFIED]` without a run.
- Whether `turbo3` at 3.125 bits/value is measurably better than a hypothetical 3-bit packed layout, i.e. whether the split `qs`/`signs` encoding costs anything beyond the norm — `[UNVERIFIED]`.

## See also

[[turboquant]] · [[innerq]] · [[walsh-hadamard-transform]] · [[gemm-dispatch]] · [[kv-cache]] · [[tq-1-missing-gemm-kernels]] · [[v100-sxm2]]
