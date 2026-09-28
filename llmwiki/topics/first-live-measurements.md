---
title: First live measurements (GTX 1660, Bonsai-4B Q2_0)
type: topic
status: current
updated: 2026-09-29
sources: [README.md, state.md]
verified: [build/cuda13.zip, build/cuda124.zip, src/ggml/include/ggml.h, src/ggml/src/ggml-cuda/CMakeLists.txt]
tags: [measurement, profiling, nsys, perf, vrAM]
---

# First live measurements

## Bottom line

**This engine was run for the first time in this project's documentation history.** The prebuilt CUDA 13 bundle from `build/cuda13.zip` works on this machine's **GTX 1660** (`sm_75`), serves the cached **`Ternary-Bonsai-4B-Q2_0_g64.gguf`**, and generated correct text — so the fork's forward path, its PrismML `Q2_0` weight kernels and even its **TurboQuant KV path** are all live and measurable without a CUDA toolkit, because the bundle ships its own runtime.

Three results worth carrying:

1. **The vault's central verdict is now confirmed in a live trace, not just by reading.** With `-ctk turbo3 -ctv q8_0`, the only kernel that touches the KV cache is the fused attention kernel `flash_attn_ext_vec<(int)128,(int)1,(ggml_type)43,(ggml_type)8,(bool)0>` — type `43` = `GGML_TYPE_TURBO3_0`, type `8` = `GGML_TYPE_Q8_0` (`src/ggml/include/ggml.h:433`, `:402`). No turbo `MUL_MAT` exists anywhere in the profile, and **no cuBLAS/MAGMA kernel appears at all**. [[device-placement]] and [[tq-1-missing-gemm-kernels]] are settled empirically.
2. **TurboQuant KV costs ~11 % of generation throughput and buys 11–15 % of VRAM** on this setup (ctx 2048): 85.6 → 76.0 t/s, 1528 → 1348 MiB (`turbo3+q8_0`) or 1296 MiB (`turbo3+turbo2`). The trade is real and measured; it is *not* a speed feature at this context length.
3. **The weights do use MMQ.** `mul_mat_q<(ggml_type)42, (int)24, (bool)0>` — `42` = `GGML_TYPE_Q2_0` — is **75 % of all GPU kernel time**. The PrismML weight format is served by the MMQ tile path, not by a fallback.

## What was run

| | |
| :--- | :--- |
| Engine | `/hdd2/tools/bonsai/cuda13/bin/llama-cli`, extracted from `build/cuda13.zip` |
| Build | `version: 0.2.0-dev (build 10747, commit 1773b4b1a)`, built with GNU 16.2.1 |
| Device | `CUDA0: NVIDIA GeForce GTX 1660 (5754 MiB, 3228 MiB free)` — compute capability **7.5**, driver 615.71.09 |
| Model | `Ternary-Bonsai-4B-Q2_0_g64.gguf`, 1 137 806 656 B (1085 MiB), `ftype: Q2_0`, from the HF cache (`prism-ml/Ternary-Bonsai-4B-gguf`) |
| Profiler | Nsight Systems 2026.5.1.161, installed from `NsightSystems-linux-public-2026.5.1.161-3889610.run` |
| Command | `-m <model> -ngl 99 -c 2048 -n 128|400 --ignore-eos -p "Write one sentence about Paris." -st --no-warmup` |

Two operational findings from setting this up, both of which cost time and are worth recording:

- **`build/cuda13.zip` is LZMA-compressed; `unzip` extracts nothing from it and reports success.** It must be opened with `7z x` (or `bsdtar`). This is the same trap that made the archive's contents look empty during the earlier vault work.
- **`nsys` is not installed**, but the user's `.run` installer works without root if the archive is extracted directly: `7z x <run>` yields a POSIX tar, and `tar xf` yields `pkg/target-linux-x64/nsys`. The installer's own prompt ignores `--target` because it reads from a TTY.

## Throughput (ctx 2048, n=128, `--ignore-eos`, two runs each)

| KV configuration | Generation t/s | Wall s |
| :--- | :--- | :--- |
| `-ctk f16 -ctv f16` | **85.9 / 85.3** | 2.36 / 2.33 |
| `-ctk turbo3 -ctv q8_0` | 76.0 / 76.0 | 2.53 / 2.52 |
| `-ctk turbo3 -ctv turbo2` | 75.0 / 74.2 | 2.57 / 2.58 |

The penalty is stable across repeats (≈11 % for `turbo3`, ≈13 % for `turbo3+turbo2`). A single longer run at `n=400` gave 92.7 / 84.4 / 80.9 t/s — the same ordering, higher absolute numbers because the per-run fixed cost is amortised over more tokens. Prompt processing was 124–134 t/s in every configuration, i.e. unaffected.

## VRAM (per-process, `nvidia-smi --query-compute-apps`, ctx 2048, n=400)

| KV configuration | VRAM | Δ vs FP16 |
| :--- | ---: | ---: |
| `-ctk f16 -ctv f16` | 1528 MiB | — |
| `-ctk turbo3 -ctv q8_0` | **1348 MiB** | −180 MiB (−11.8 %) |
| `-ctk turbo3 -ctv turbo2` | **1296 MiB** | −232 MiB (−15.2 %) |

Note: the engine prints **no memory report at all** — the fork's CLI has no KV-buffer or model-size log line — so these numbers come from outside the process. The saving is larger than the K-side arithmetic suggests for a 16-layer KV (a `turbo3` K cache should save tens of MiB at this context, not 180); the difference is an open question, and it may include a smaller attention compute buffer on the `flash_attn_ext_vec` path.

## GPU profile (nsys, `-t cuda`, n=128)

Top kernels, F16 KV versus `turbo3`+`q8_0`:

| Kernel (mangled) | FP16 KV | `turbo3`+`q8_0` |
| :--- | :--- | :--- |
| `mul_mat_q<(ggml_type)42,(int)24,(bool)0>` — Q2_0 weights, MMQ tiles | 75.0 %, 119.8 ms, 249 inst | 75.0 %, 113.5 ms, 249 inst |
| `mul_mat_vec_q<(ggml_type)42,(int)2,…>` — Q2_0, MMVQ | 7.0 %, 12.4 ms, 249 inst | 8.0 %, 12.4 ms, 249 inst |
| `flash_attn_ext_f16<(int)128,(int)128,(int)8,(int)4,…>` — prefill attention | 3.0 %, 5.24 ms, 36 inst | 3.0 %, 4.84 ms, 36 inst |
| **`flash_attn_ext_vec<(int)128,(int)1,(ggml_type)43,(ggml_type)8,(bool)0>`** | **absent** | 1.0 %, 0.98 ms, 36 inst |
| `rms_norm_mul_rope_f32<(int)256,…>` | 0.31 ms, 108 inst | **0.83 ms, 216 inst** |
| `rms_norm_f32<(int)1024,…>` | 0.85 ms, 219 inst | 0.85 ms, 219 inst |
| `quantize_q8_1` | 0.79 ms, 472 inst | 0.76 ms, 472 inst |

Three observations:

- The **turbo KV type appears exactly once** on the GPU: inside the fused attention kernel, instantiated for K = `TURBO3_0` and V = `Q8_0`. That is [[device-placement]]'s verdict in a live trace.
- `rms_norm_mul_rope_f32<(int)256>` **doubles its instance count** (108 → 216) with turbo KV. This is the fused norm+RoPE op on the output side; the extra 108 invocations are the additional inverse-rotation work the quantized path requires. It is small (0.83 ms) but it is the visible per-layer cost of the rotation contract described in [[forward-pass]].
- **No `turbo_wht` kernel appears**, and neither configuration reaches a `mul_mat` on the KV cache. The rotation work is inside `set_rows` on the write side ([[turbo-wht]], [[forward-pass]]).

CUDA API summary (turbo run): `cudaStreamSynchronize` 66 % of API time (489 ms, 1106 calls), **`cudaMemGetInfo` 15 % (116 ms across just 13 calls — ≈8.9 ms each)**, `cudaMemcpyAsync` 11 % (88 ms, 668 calls), `cudaLaunchKernel` 4 % (32 ms, 4296 calls), `cudaGraphLaunch` 6.45 ms over 26 calls. Memory traffic: **H2D 1132 MB in 626 copies** (93 % of all memcpy time, including a single 109 MB transfer), memset 906 MB in 3 calls, D2H 23 MB in 38 copies.

Graphs are demonstrably in use: one `cudaGraphInstantiate`, one `cudaGraphExecUpdate`, 26 `cudaGraphLaunch` ([[cuda-graphs]]).

## CPU (`perf stat`, F16 KV, n=128)

| Counter | Value |
| :--- | ---: |
| Elapsed / user / sys | 1.99 s / 1.59 s / 0.31 s (**94 % of wall is CPU**) |
| `cpu_core` cycles | 8.46 G |
| `cpu_core` instructions | 18.32 G (**IPC ≈ 2.17**) |
| `cpu_core` branches / misses | 3.90 G / 6.97 M (**0.18 %**) |
| `cpu_core` cache refs / misses | 50.1 M / 28.3 M (**56.5 % miss rate**) |
| Page faults | 58 603 |
| Context switches, CPU migrations | **0 / 0** |

The machine is a hybrid part — `perf` reports `cpu_atom` (E-cores) and `cpu_core` (P-cores) separately, and only the `cpu_core` counters cover ~98 % of the measurement window, so the P-core figures are the meaningful ones. A 56.5 % L1/LLC cache-miss rate with a healthy IPC is the signature of a memory-hungry dequantize-and-multiply loop rather than branchy control flow — consistent with `quantize_q8_1` and the MMQ activation path dominating the host-side work.

## RAM and IO

- **Peak host RSS: 1503.3–1503.8 MiB** and *identical across all three KV configurations* — the resident set is the model plus the process, not the KV cache, at this context length. 59 082 minor faults, **0 major faults**.
- **IO during the run: 0 blocks read or written** (`ru_inblock` / `ru_oublock`), because the 1085 MiB model was already in page cache.
- Sequential read of the model file, measured directly: **949 MiB/s uncached** (`O_DIRECT`) versus **14 179 MiB/s from page cache**. The file lives on the NVMe `/home` volume, so a cold start costs ≈1.14 s of pure model-load IO — negligible against load-time compute, and invisible in every run recorded here.

## What this confirms or refutes in the vault

| Vault claim | Status now |
| :--- | :--- |
| [[device-placement]]: turbo KV is read by fused attention, no turbo `MUL_MAT` | **Confirmed live** — `flash_attn_ext_vec<…,43,8,…>`, no turbo `mul_mat` in the trace |
| [[tq-1-missing-gemm-kernels]]: the missing dispatch entry costs GPU time via cuBLAS/MAGMA | **Refuted again** — no cuBLAS or MAGMA kernel appears in either profile |
| [[quantized-kernel-units]]: `Q2_0` weight types are served by MMQ | **Confirmed** — `mul_mat_q<42,…>` is 75 % of kernel time |
| [[cuda-graphs]]: graph reuse is live on this path | **Confirmed** — instantiate/exec-update once, 26 graph launches |
| [[walsh-hadamard-transform]], [[turbo-wht]]: the rotation is not the bottleneck | **Consistent** — no WHT kernel in the top 10; the rotation's visible cost is one extra fused norm+RoPE pass |
| [[benchmarks]]: no numbers exist for this engine on any machine *this* project can access | **Superseded for the 4B model on `sm_75`** — this page is the first such measurement, and it is not the target hardware |

## Caveats — read before quoting any number here

- **The GPU is not the target.** GTX 1660 is `sm_75` (Turing, INT8 tensor cores present); the deployment target is V100 `sm_70` (no INT tensor cores). The MMQ/MMVF paths available differ, so **none of these numbers predict V100 behaviour**.
- **The model is not the target.** 4B `Q2_0` at the upstream group size 64, not the 27B `PQ2_0` at group 128.
- **Context is 2048**, not the shipped 16384–32768, which is where TurboQuant's memory advantage should matter most.
- **TriAttention was never exercised** — it needs a `.triattention` profile and none exists for the 4B model.
- **nsys runs are not comparable to plain runs**: the profiled generation rates differ from the unprofiled ones, so use the plain-run table for throughput and the nsys tables for structure only.
- The CUDA 12.4 bundle was deliberately not tried: its `libggml-cuda.so` carries only `sm_86` markers ([[v100-sxm2]]), which cannot run on `sm_75`.

## Open questions

- Why does the VRAM saving (−180 MiB for `turbo3+q8_0`) exceed the K-side arithmetic for a 16-layer cache at ctx 2048? Does the `flash_attn_ext_vec` path reduce the attention compute buffer, or is the model's KV layout different from the vault's assumption?
- What are the 13 `cudaMemGetInfo` calls costing 116 ms — a per-request VRAM probe, or startup-only? At 8.9 ms each they would be worth eliminating.
- Does the ~11 % turbo penalty hold at 16K–32K context, where the cache is large enough for bandwidth to dominate? The shipped configuration bets that it does not.
- Do these kernels exist for `sm_70`? The bundle's `sm_75` path proves the *code* works; whether the V100-targeted build emits compatible MMQ/attention instantiations is unverified ([[build-and-verify]]).
- Is the 4B model the same architecture family as the 27B? If it is not `qwen35`-hybrid, the KV-layer count behind the VRAM numbers is different, and the arithmetic above is approximate.

## See also

[[device-placement]] · [[quantized-kernel-units]] · [[turboquant]] · [[cuda-graphs]] · [[benchmarks]] · [[v100-sxm2]] · [[build-and-verify]] · [[release-artifacts]] · [[ternary-bonsai-2-27b]] · [[forward-pass]] · [[open-questions]]
