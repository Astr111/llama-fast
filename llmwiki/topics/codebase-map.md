---
title: "Codebase map — llama-fast"
type: topic
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified:
  - src/CMakeLists.txt
  - src/CMakePresets.json
  - src/ggml/CMakeLists.txt
  - src/ggml/src/CMakeLists.txt
  - src/ggml/src/ggml-cuda/CMakeLists.txt
  - src/src/CMakeLists.txt
  - src/src/llama-triattention.cpp
  - src/src/llama-triattention.h
  - src/src/turbo-rotation-data.h
  - src/src/turbo-rotation-data-32.h
  - src/src/llama-kv-cache.cpp
  - src/src/llama-graph.cpp
  - src/src/llama-context.cpp
  - src/src/llama-model-loader.cpp
  - src/src/llama-quant.cpp
  - src/ggml/include/ggml.h
  - src/ggml/src/ggml-common.h
  - src/ggml/src/ggml-quants.c
  - src/ggml/src/ggml-cpu/ggml-cpu.c
  - src/ggml/src/ggml-turbo-quant.c
  - src/ggml/src/ggml-cuda/triattention-score.cu
  - src/ggml/src/ggml-cuda/turbo-quant.cuh
  - src/ggml/src/ggml-cuda/turbo-wht.cu
  - src/ggml/src/ggml-cuda/turbo-innerq.cu
  - src/ggml/src/ggml-cuda/turbo-innerq.cuh
  - src/ggml/src/ggml-cuda/set-rows.cu
  - src/ggml/src/ggml-cuda/dequantize.cuh
  - src/ggml/src/ggml-cuda/convert.cu
  - src/ggml/src/ggml-cuda/getrows.cu
  - src/ggml/src/ggml-cuda/vecdotq.cuh
  - src/ggml/src/ggml-cuda/mmq.cu
  - src/ggml/src/ggml-cuda/mmq.cuh
  - src/ggml/src/ggml-cuda/mmvq.cu
  - src/ggml/src/ggml-cuda/mmvq.cuh
  - src/ggml/src/ggml-cuda/ggml-cuda.cu
  - src/ggml/src/ggml-cuda/fattn.cu
  - src/ggml/src/ggml-cuda/fattn-vec.cuh
  - src/ggml/src/ggml-cuda/mmq-hopper-q1.cu
  - src/ggml/src/ggml-cuda/mmq-load-tiles.cuh
  - src/ggml/src/ggml-cuda/mmq-config-pascal.cuh
  - src/ggml/src/ggml-cuda/template-instances
  - src/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_ptq1_0.comp
  - src/ggml/src/ggml-vulkan/vulkan-shaders/dequant_ptq1_0.comp
  - src/common/arg.cpp
  - src/common/speculative.cpp
  - src/tools/triattention-calibrate/CMakeLists.txt
  - src/tools/triattention-calibrate/triattention-calibrate.cpp
  - src/tests/CMakeLists.txt
  - src/docs/TRIATTENTION.md
  - src/docs/TRIATTENTION-API.md
  - src/calibration_corpus.txt
  - src/bonsai-27b.triattention
  - src/build-x64-linux-gcc-debug
  - src/build-x64-linux-gcc-debug/CMakeCache.txt
  - scripts/start_server_turbo.sh
  - scripts/start_server_baseline.sh
  - scripts/run_cli.sh
  - calibration/bonsai-27b.triattention
  - build/cuda124.zip
  - build/cuda13.zip
  - llama-fast-src.zip
  - .gitignore
tags: [layout, build, cuda, delta]
---

## Bottom line

`llama-fast` is a **llama.cpp fork**, and the fork root is the `src/` subdirectory of this repository — not the repository root. Everything upstream llama.cpp lives under `src/` (version `0.2.0`-dev, `ggml 0.21.0`), and the project's own code is a **delta of four clusters inside it**: TriAttention eviction, TurboQuant KV quantization, PrismML low-bit weight kernels, and the CUDA-graph/concurrency hook. The repository root itself holds no source: only the launch scripts, one calibration profile, two prebuilt release archives, the three prose sources, and this wiki. ([[source-readme]], [[source-state-md]])

The delta concentrates in a handful of files that did not exist upstream (`src/src/llama-triattention.*`, `src/ggml/src/ggml-cuda/triattention-score.*`, `turbo-*`, `src/ggml/src/ggml-turbo-quant.c`, `mmq-hopper-q1.cu`, `mmq-load-tiles.cuh`, `mmq-config-pascal.cuh`, `src/tools/triattention-calibrate/`, three new tests) plus existing upstream files that were extended to know about the five new `ggml_type` ids (`TURBO2_0`/`TURBO3_0`/`TURBO4_0`/`PQ2_0`/`PTQ1_0`) — 16 files mention `GGML_TYPE_TURBO2_0` and 56 mention `PQ2_0`/`PTQ1_0` (`grep -rl` counts over `src/`, excluding the build directory). The single most important asymmetry for a future agent: **the new KV types are wired into attention (flash-attention vec kernels, SET_ROWS, dequantize) but *not* into matmul dispatch**, which is exactly what [[tq-1-missing-gemm-kernels]] describes.

**The `sm_70` question.** The default CUDA arch list lives in `src/ggml/src/ggml-cuda/CMakeLists.txt:8-56` and is applied only when `CMAKE_CUDA_ARCHITECTURES` is not user-set. It has three branches:

| Condition | Result |
| :--- | :--- |
| `GGML_NATIVE` ON **and** CUDA ≥ 11.6 **and** CMake ≥ 3.24 (the normal Linux default — `GGML_NATIVE` defaults ON) | `native` → only the GPU present on the **build machine** |
| else, CUDA toolkit **< 13** | `50-virtual 61-virtual 70-virtual 75-virtual 80-virtual 86-real` (+`89-real 90-virtual` if CUDA ≥ 11.8) |
| else, CUDA toolkit **≥ 13** | the `50/61/70` trio is skipped: `75-virtual 80-virtual 86-real`, `89-real 90-virtual`, `120a-real`, `121a-real` |

So the plain answer is: **yes, a default (non-`native`) source build with a CUDA 12.x toolkit emits `sm_70` — as `70-virtual`, i.e. PTX that is JIT-compiled at first run, not embedded SASS** — and **no, it does not when either `GGML_NATIVE` resolves to a build host that is not a V100, or the toolkit is CUDA ≥ 13**, and **no, not with the build command the README publishes**, which passes `-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100;120"` and thereby suppresses the default list entirely. That published command cannot target the V100 ([[v100-sxm2]]) at all. The released `build/cuda124.zip` does not rescue the situation: measured from the artifact, its `libggml-cuda.so` carries **only** `sm_86` PTX and cubins, not the `sm_61;sm_70;…` its own README table advertises. Details and the caveat on that measurement are in *Evidence* below.

## Evidence

### Root layout — what is project payload and what is not

`ls` of the repository root (2026-09-28) finds:

| Entry | Kind | Note |
| :--- | :--- | :--- |
| `src/` | dir | **the fork root** — upstream llama.cpp with the project's additions |
| `scripts/` | dir | 3 launch scripts (`run_cli.sh`, `start_server_turbo.sh`, `start_server_baseline.sh`) |
| `calibration/` | dir | `bonsai-27b.triattention` — one TriAttention calibration profile |
| `build/` | dir | `cuda124.zip` (811,787,748 B) and `cuda13.zip` (1,663,354,910 B) — prebuilt release bundles |
| `llama-fast-src.zip` | file | 37.6 MB clean-source archive; all 3463 entries are under `src/` |
| `README.md`, `state.md` | files | the two prose sources ([[source-readme]], [[source-state-md]]) |
| `AGENTS.md`, `llmwiki.txt`, `LICENSE` | files | `AGENTS.md` is the workspace's wiki/agent contract ([[source-agents-md]]), `LICENSE` is the project licence |
| `.gitignore` | file | ignores `build/*.zip`, `*.tar.xz`, `llama-fast-src.zip` — the release binaries are deliberately untracked |
| `llmwiki/` | dir | this vault ([[source-llmwiki]]) |
| `_bmad/`, `_bmad-output/`, `.agents/`, `.claude/`, `.omp/`, `.vscode/`, `skills-lock.json` | dirs/files | agent-harness scaffolding, not project code |

> Contradiction (2026-09-28): the task brief states the root holds "only `src/`, `scripts/`, `calibration/`, `build/`, `README.md`, `AGENTS.md`, `state.md` and the archive `llama-fast-src.zip`". That is the complete **project payload** and matches what a reader of [[source-readme]] would expect, but the directory also carries the wiki vault, four agent-harness directories, `llmwiki.txt`, `skills-lock.json`, `LICENSE` and `.gitignore` — verified by directory listing.

### Where the project's own work lives

**TriAttention** — eviction policy, calibration file I/O, pruning. `src/src/llama-triattention.cpp` (1492 lines) and `src/src/llama-triattention.h` (393 lines) are compiled into `llama` unconditionally (`src/src/CMakeLists.txt:31`). The header documents the binary `.triattention` format (magic `0x54524941`, version 1) and defines the modes `TRIATTENTION_MODE_GLOBAL` / `_PER_KV_HEAD` / `_PER_LAYER_HEAD` and the trigger enum. GPU scoring lives in `src/ggml/src/ggml-cuda/triattention-score.cu` (kernel header comment: grid = `(n_cells, n_offsets_or_1, 1)`, block = `(freq_count, 1, 1)`; it includes `dequantize.cuh` and `turbo-quant.cuh`), declared in `triattention-score.cuh`. CLI surface is `--triattention-*` in `src/common/arg.cpp:4690-4828` (stats path, budget, window/divide-length, offset-max, mode, trigger, agg, seed, normalize, protect-prefill, calibrate, calibrate-out, log). The offline tool is `src/tools/triattention-calibrate/triattention-calibrate.cpp` (target `llama-triattention-calibrate`, `src/tools/triattention-calibrate/CMakeLists.txt:1`); it re-uses the common argument/context path and collects pre-RoPE query statistics per `(layer, head)` via an `cb_eval` hook. The project's own docs are `src/docs/TRIATTENTION.md` and `src/docs/TRIATTENTION-API.md`; a calibration corpus lives at `src/calibration_corpus.txt`.

**TurboQuant KV** — the three new KV types. Declared in the `enum ggml_type` at `src/ggml/include/ggml.h`: `GGML_TYPE_TURBO3_0 = 43` (line 433), `TURBO4_0 = 44` (434), `TURBO2_0 = 45` (435). Block layouts (`block_turbo3_0` 14 B, `block_turbo4_0` 68 B, `block_turbo2_0` 10 B, all `QK_TURBO* = 128`, plus `NL_TURBO*` FA iteration counts and the `QK_TURBO*_GROUP = 128` rotation group) are in `src/ggml/src/ggml-common.h:324-383`. The CPU codec/reference is `src/ggml/src/ggml-turbo-quant.c`; CUDA side is `src/ggml/src/ggml-cuda/turbo-quant.cuh` (centroid/midpoint constant tables, block codec macros `QR_TURBO* = 1`, WHT scaffolding and the InnerQ device symbols), `src/ggml/src/ggml-cuda/dequantize.cuh` (the per-type `dequantize_turbo*_0` device functions — `dequantize_turbo3_0` at line 163), `turbo-wht.{cu,cuh}` (forward/inverse WHT kernels, templated on direction and group size 128/64), `turbo-innerq.{cu,cuh}` (host-side InnerQ shared state: `g_innerq_finalized`, `g_innerq_scale_inv_host`, `turbo_innerq_publish`; see [[innerq]], [[tq-2-innerq-host-state]]) and `src/ggml/src/ggml-cuda/set-rows.cu` (the quantize-on-write kernel; type dispatch at `set-rows.cu:1241-1245`). Attention integration: `src/ggml/src/ggml-cuda/fattn.cu` (`FATTN_VEC_CASES_ALL_D_D` for every turbo/q8_0 pairing at lines 339-369; `K/V` turbo special-casing at 494/502) and `fattn-vec.cuh:87-95`. Dequantize entry points for CUDA `CPY`/`GET_ROWS` are in `convert.cu:664-669,735-740,772-777,836-837`. The rotation matrices used by the llama-graph pre-rotate step are embedded as tables in `src/src/turbo-rotation-data.h` (and `turbo-rotation-data-32.h`). KV-cache allocation and per-layer type overrides (including a `TURBO_LAYER_ADAPTIVE` env switch and boundary-layer policy) are in `src/src/llama-kv-cache.cpp:263-371`. The `-ctk/--cache-type-k` string values `turbo2|turbo3|turbo4` are parsed in `src/common/arg.cpp:324-330`.

**PrismML weight kernels (`PQ2_0`, `PTQ1_0`)** — the 2-bit / ternary weight formats the 27B model is stored in ([[prismml-weight-kernels]], [[ternary-bonsai-2-27b]]). Type ids are private high ids in `src/ggml/include/ggml.h`: `GGML_TYPE_PQ2_0 = 142`, `GGML_TYPE_PTQ1_0 = 143`, `GGML_TYPE_COUNT = 144` (with an explicit comment that ids 46-141 are unused). Block layouts: `block_pq2_0` (`QK_PQ2_0 = 128`, 2 bits/element) and `block_ptq1_0` (`QK_PTQ1_0 = 128`, 5 trits/byte + 4 trits/byte high bits) at `src/ggml/src/ggml-common.h:202-220`. CUDA kernels and their configuration:
- `src/ggml/src/ggml-cuda/vecdotq.cuh` — `vec_dot_pq2_0_q8_1` (line 978) and `vec_dot_ptq1_0_q8_1` (895, via `vec_dot_ptq1_0_q8_1_multi` at 809); vector-dot-ratio macros at 115-118.
- `src/ggml/src/ggml-cuda/mmvq.cu` — dot-product selection at 15-16, ratios at 46-47, a PTQ1_0 Turing+ path at 298, a PQ2_0 special-case on `ncols_x == 6144 && nrows_x == 2048` at 1071, dispatch cases at 1209-1216.
- `src/ggml/src/ggml-cuda/mmq.cu` — MMQ dispatch cases 17-22, PTQ1_0 gate at 375-381, PTQ1_0 batch-size cap 426-432.
- `src/ggml/src/ggml-cuda/mmq-hopper-q1.cu` — the opt-in Hopper `sm_90a` wgmma path ("dequant-in-SMEM + int8 wgmma", explicitly *not* bit-identical to standard MMQ), gated by `-DGGML_CUDA_HOPPER_Q1` + CUTLASS (`src/ggml/src/ggml-cuda/CMakeLists.txt:157-165`; option declared at `src/ggml/CMakeLists.txt:203`).
- `src/ggml/src/ggml-cuda/mmq-load-tiles.cuh` and `mmq-config-pascal.cuh` — the tile loader and the per-architecture MMQ config used by that path.
- Support matrix entries in the backend (`ggml/src`-level plumbing): `src/ggml/src/ggml-cuda/ggml-cuda.cu:5245-5287` admits both types for `MUL_MAT`/`GET_ROWS`; `getrows.cu`, `convert.cu`, `ggml-quants.c` and `ggml-cpu/ggml-cpu.c` also handle them. Non-CUDA backends carry plumbing too (`ggml-vulkan/vulkan-shaders/{mul_mat_vecq_ptq1_0.comp,dequant_ptq1_0.comp}`, plus SYCL/Metal entries).
- Tests: `src/tests/test-pq2-row-shapes.cpp`, `test-ptq1_0-cuda-dot.cpp`, `test-ptq1_0-element-map.cpp`, wired at `src/tests/CMakeLists.txt:339-341`. The first two are also shipped as binaries in both release bundles (`bin/test-pq2-row-shapes`, `bin/test-ptq1_0-cuda-dot`).

**CUDA graphs / concurrency** — `GGML_CUDA_GRAPHS` is a ggml option (default ON for llama builds: `src/CMakeLists.txt:172-173`) which adds `GGML_CUDA_USE_GRAPHS` (`src/ggml/src/ggml-cuda/CMakeLists.txt:149-150`). The documented runtime switch `GGML_CUDA_GRAPH_OPT=1` is read once in `ggml_backend_cuda_graph_optimize` (`src/ggml/src/ggml-cuda/ggml-cuda.cu:4622-4638`) and enables the stream-context path (see [[cuda-graphs]]).

`draft-dflash` speculative decoding ([[speculative-decoding]], [[qwen3-dflash-draft]]) is configured through `src/common/arg.cpp` / `src/common/speculative.cpp` (`COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH`); whether that mechanism is fork-specific or inherited from the recent upstream base is [UNVERIFIED] from this tree alone.

### What was *not* added — the dispatch gap, seen from the map

The five new types are absent from the matmul fast-path predicates that gate `mmq`/`mmvq`. `ggml_cuda_should_use_mmq` is defined at `src/ggml/src/ggml-cuda/mmq.cu:366` (declared `mmq.cuh:1723`) and its `switch (type)` lists `PTQ1_0`, `Q1_0`, `Q2_0`, `PQ2_0` alongside the upstream quant set; `ggml_cuda_should_use_mmvq` is defined at `src/ggml/src/ggml-cuda/mmvq.cu:293` (declared `mmvq.cuh:5`) and special-cases `PTQ1_0` only. Neither file contains the string `TURBO` at all (`grep -c` returns 0 for both). In `src/ggml/src/ggml-cuda/ggml-cuda.cu`, the only mention of a turbo type is the `GGML_OP_SET_ROWS` support check at line 5327. That is the *file-level* corroboration of [[tq-1-missing-gemm-kernels]]; the runtime consequence is quantified in [[performance-profile]].

### Build system

- **Presets**: `src/CMakePresets.json` declares `configurePresets` only (no build/test presets). Concrete ones: `x64-linux-gcc-debug`, `-release`, `-reldbg`, `x64-linux-gcc+static-release`, the `arm64-windows-llvm-*` and `arm64-apple-clang-*` sets, `x64-windows-llvm-*`, `x64-windows-msvc-*`, `x64-windows-sycl-*`, `x64-windows-vulkan-*`. **No preset selects CUDA** — every GPU build is a manual `-DGGML_CUDA=ON` invocation, which is why the arch question above matters.
- **Backend selection**: `option(GGML_CUDA "ggml: use CUDA" OFF)` at `src/ggml/CMakeLists.txt:200`; the CPU backend is always added (`src/ggml/src/CMakeLists.txt:484 ggml_add_backend(CPU)`), CUDA only if the option is ON (`ggml_add_backend(CUDA)` at line 591, gated by the `GGML_<NAME>` flag inside `ggml_add_backend`). CUDA-relevant options: `GGML_CUDA_GRAPHS` (line 212), `GGML_CUDA_FORCE_MMQ` (202), `GGML_CUDA_HOPPER_Q1` (203), `GGML_CUDA_FA`/`GGML_CUDA_FA_ALL_QUANTS` (210-211), `GGML_CUDA_COMPRESSION_MODE` with values `none;speed;balance;size`, default `size` (214-217; passed to nvcc as `-compress-mode=` at `src/ggml/src/ggml-cuda/CMakeLists.txt:237`).
- **Architecture list**: `src/ggml/src/ggml-cuda/CMakeLists.txt:8-56` as tabulated in *Bottom line*; the `12X → 12Xa` rewrite follows at lines 80-93.
- **Local state**: `src/build-x64-linux-gcc-debug/` is **build output, not source** — it contains `CMakeCache.txt`, `CMakeFiles/`, `Testing/` and generated `llama-config.cmake`, and its cache says `CMAKE_GENERATOR=Ninja`, `CMAKE_BUILD_TYPE=Debug`, `CMAKE_CXX_COMPILER=g++`, `GGML_NATIVE=ON`, **`GGML_CUDA=OFF`**. There is no CUDA toolkit on this machine (no `nvcc`; `/usr/local` does not exist), so no CUDA build can be reproduced locally as of 2026-09-28.
- **How the release bundles were configured**: the shipped CUDA 13 library contains exactly the seven architectures named by the README's native-build example (`75;80;86;89;90;100;120` → PTX targets `sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120a`), which is *not* the CUDA-13 default list (that one has no `sm_100`/`sm_120a`). The shipped CUDA 12.4 library contains a single architecture, `sm_86` (an RTX 3090-class target — consistent with the machine the README benchmarks were run on, and with `GGML_NATIVE`/a single-arch override, but the exact command is not recorded in the tree) [INFERENCE].

### What is inside the non-source directories

**`scripts/` (3 shell scripts, all `chmod +x`):**

| Script | Launches | Environment / flags |
| :--- | :--- | :--- |
| `start_server_turbo.sh` | `$SCRIPT_DIR/bin/llama-server` (relative to the script — i.e. it expects to sit next to an unpacked `build/cuda1*.zip`) | exports `LD_LIBRARY_PATH="$SCRIPT_DIR/bin:$SCRIPT_DIR/lib/cuda:$LD_LIBRARY_PATH"` and `GGML_CUDA_GRAPH_OPT=${GGML_CUDA_GRAPH_OPT:-1}`; defaults `CTK=turbo3`, `CTV=q8_0`, `TRI_BUDGET=4096`, `TRI_WINDOW=512`, `HOST=0.0.0.0`, `PORT=8080`, stats `$SCRIPT_DIR/../calibration/bonsai-27b.triattention`; adds `--triattention-stats/-budget/-window --triattention-protect-prefill` unless `DISABLE_TRIATTENTION=1`; runs with `-ngl 99 -c 32768 -n 8192 --reasoning-budget 4000 -np 1 -t $(nproc)` |
| `start_server_baseline.sh` | `${BIN_DIR:-$BASE_DIR/../dist-rtx3090/bin}/llama-server` | pure FP16 KV baseline: exports only `LD_LIBRARY_PATH`; `-m`/`--mmproj` default to `$HOME/Prism/llama-prism-b10743-adfffbe/*.gguf`, `-ngl 99 -c 16384`, no turbo cache types and no TriAttention flags; forwards `"$@"` |
| `run_cli.sh` | `$SCRIPT_DIR/bin/llama-cli` | same two exports and the same turbo/triattention defaults as the turbo server script, `-ngl 99 -c 16384`; `--conversation` when no prompt, otherwise `-p "$PROMPT"` |

Neither `scripts/bin/` nor `scripts/lib/cuda/` exists in this repository, and `$HOME/Prism/…` and `../dist-rtx3090/bin` do not exist either — these three scripts are the release-layout launchers (near-identical copies, 3 bytes apart, ship inside both zips as `start_server.sh`/`run_cli.sh`), so as checked out here they fail at `exec` unless a `build/*.zip` is unpacked next to them.

**`calibration/bonsai-27b.triattention`** is the offline TriAttention calibration profile for the target model: 789,571 bytes of binary, header per the format block in `src/src/llama-triattention.h`. Read from the file itself: magic `0x54524941` (bytes `41 49 52 54`, "TRIA" in little-endian), version `1`, `head_dim` 256, `num_layers` 64, `num_attn_heads` 24, `num_kv_heads` 4, `rope_theta` 1e7, `rope_style` 0, `n_sampled` 384, `freq_count` 128 (= `head_dim/2`), `name_len` 19 and the model name `Bonsai-2-27B-PQ2_0`, followed by per-head `q_mean_real`/`q_mean_imag`/`q_abs_mean`/`r_f` float blocks. `src/bonsai-27b.triattention` is a byte-identical copy (both sha256-prefix `6dff56bab1e4`), and both release zips carry the same bytes at `<root>/calibration/bonsai-27b.triattention`. The matching corpus is `src/calibration_corpus.txt` (4,353 B, also inside both zips). The tool that produces this file is `llama-triattention-calibrate -m model.gguf -f corpus.txt -o out.triattention` ([[source-readme]] Example 4; usage text in `triattention-calibrate.cpp`).

**`build/cuda124.zip` and `build/cuda13.zip`** are the release bundles described in [[source-readme]] under *Release Directory Structure*, but shipped as two archives rather than an expanded `Release/` tree (no `Release/` directory exists in the repository). Entry lists, read without extracting:

| | `cuda124.zip` (811.8 MB, 58 entries) | `cuda13.zip` (1663.4 MB, 49 entries) |
| :--- | :--- | :--- |
| Top level | `cuda12.4/` | `cuda13/` |
| `bin/` | `llama-server`, `llama-cli`, `llama-triattention-calibrate`, `test-pq2-row-shapes`, `test-ptq1_0-cuda-dot`; `libggml-base/cpu/cuda/ggml`, `libllama`, `libllama-common`, `libmtmd` + a `libllama-*-impl.so` per tool | same three executables and two tests; `libggml-*`, `libllama*`, `libmtmd*`; `*-impl.so` only for cli and server |
| `lib/cuda/` | `libcudart.so.12` → `.12.2.128`, `libcublas.so.12` → `.12.2.4.5`, `libcublasLt.so.12` → `.12.2.4.5` | `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13` (unversioned) |
| `calibration/` | `bonsai-27b.triattention`, `calibration_corpus.txt` | identical pair |
| Scripts + docs | `start_server.sh`, `run_cli.sh`, `README.md` (Russian, claims `sm_61/sm_70/sm_75/sm_80/sm_86`) | `start_server.sh`, `run_cli.sh`, `README.md` (Russian, claims `sm_75…sm_120`) |

Measured architecture coverage of `bin/libggml-cuda.so` inside those archives (by locating the embedded nvFATBIN blobs; the 12.4 library stores PTX/SASS plainly, the 13.x library stores each blob as a zstd frame):

- **cuda13**: 164 translation units × 7 targets — PTX `sm_75, sm_80, sm_86, sm_89, sm_90, sm_100, sm_120a` (PTX ISA 9.3) plus 1148 embedded cubins whose `e_flags` resolve to those same seven architectures. This matches the README's own fat-binary table.
- **cuda12.4**: 164 translation units, each carrying **one** PTX blob and **one** cubin — every PTX is `.target sm_86` (PTX ISA 8.2) and all 164 cubins share a single identical `e_flags` (`0x00560556`), with no `sm_70` (or `sm_61`/`sm_75`/`sm_80`) string anywhere in the library.

> Contradiction (2026-09-28): `build/cuda124.zip` ships a single-architecture (`sm_86`) `libggml-cuda.so`, while [[source-readme]]'s GPU matrix — and the bundle's own `README.md` — advertise `sm_61, sm_70, sm_75, sm_80, sm_86` for that folder. Taken at face value the legacy bundle would **not** run on the V100 target. Caveat: the architecture mapping was derived from PTX target strings (unambiguous) plus a repeated identical cubin `e_flags` (a single distinct value across 164 cubins, i.e. no second architecture); the exact numeric decoding of that `e_flags` field is [UNVERIFIED] because no CUDA toolkit (`cuobjdump`) is installed here.

**`llama-fast-src.zip`** (root) is the clean-source snapshot: 3,463 entries, all under a single top-level `src/`, including the custom files (`src/src/llama-triattention.*`, `src/ggml/src/ggml-cuda/triattention-score.*`, `turbo-*`, `src/tools/triattention-calibrate/`, `src/docs/TRIATTENTION*.md`).

> Finding (2026-09-28): the working `src/` tree is **not** complete relative to that archive. It holds 1,863 files against the archive's 3,463; 1,600 files present in the archive are absent from the tree (`tools/` 783, `ggml/` 728, `examples/` 84, `scripts/` 5), and `src/ggml/src/ggml-cuda/template-instances/` is an **empty directory** although CMake globs `template-instances/fattn-tile*.cu`, `fattn-mma*.cu`, `mmq*.cu`, `mmf*.cu` into the CUDA build (`src/ggml/src/ggml-cuda/CMakeLists.txt:105-112`). Consequence is [INFERENCE]: a CUDA build from this tree alone would miss those ~142 instantiation units. Whether the sparseness is intentional (publication checkout) is not recorded in the tree.

Directly in `src/ggml/src/ggml-cuda/` the tree has 71 `.cu` + 90 `.cuh` files (`fwht.{cu,cuh}` is the upstream Hadamard op; `turbo-wht.*` is the project's WHT for KV rotation).

### Where to look for X

| Task | Path |
| :--- | :--- |
| Eviction policy, budget arithmetic, protected-token counting | `src/src/llama-triattention.cpp` (+ `.h` for enums and the calibration format) |
| Scoring math / GPU scoring kernel | `src/ggml/src/ggml-cuda/triattention-score.cu`, `.cuh` |
| `.triattention` calibration file read/write | `src/src/llama-triattention.h` (format), `src/src/llama-triattention.cpp` (reader), `src/tools/triattention-calibrate/triattention-calibrate.cpp` (writer) |
| WHT forward/inverse kernels (KV rotation) | `src/ggml/src/ggml-cuda/turbo-wht.{cu,cuh}`; rotation tables in `src/src/turbo-rotation-data{,-32}.h` |
| InnerQ (per-channel equalization) | `src/ggml/src/ggml-cuda/turbo-innerq.{cu,cuh}`, used from `turbo-quant.cuh` |
| TurboQuant type ids / block layout | `src/ggml/include/ggml.h` (43-45), `src/ggml/src/ggml-common.h:324-383` |
| TurboQuant CPU codec | `src/ggml/src/ggml-turbo-quant.c` |
| TurboQuant GPU quantize (KV write) | `src/ggml/src/ggml-cuda/set-rows.cu:1241-1245` |
| TurboQuant dequantize for CUDA `CPY`/`GET_ROWS` | `src/ggml/src/ggml-cuda/convert.cu:664-837` |
| TurboQuant attention kernels | `src/ggml/src/ggml-cuda/fattn.cu:339-369,394-502`, `fattn-vec.cuh:87-95` |
| TurboQuant → matmul fast path ([[tq-1-missing-gemm-kernels]]) | `src/ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_should_use_mmq`/`mmvq`), `mmq.cuh:1723`, `mmvq.cuh:5`, then `vecdotq.cuh` |
| PrismML `PQ2_0`/`PTQ1_0` type ids and blocks | `src/ggml/include/ggml.h` (142/143), `src/ggml/src/ggml-common.h:202-220` |
| PrismML CUDA kernels | `vecdotq.cuh:809-978`, `mmvq.cu:15-16,298,1071,1209-1216`, `mmq.cu:17-22,375-432` |
| PrismML Hopper WGMMA path (opt-in) | `src/ggml/src/ggml-cuda/mmq-hopper-q1.cu`, `mmq-load-tiles.cuh`, `mmq-config-pascal.cuh`; gate `src/ggml/src/ggml-cuda/CMakeLists.txt:157-165` |
| CPU fallback quant codec | `src/ggml/src/ggml-quants.c`, `src/ggml/src/ggml-cpu/ggml-cpu.c` |
| CLI knobs | `src/common/arg.cpp` (`-ctk/-ctv` at 324-330; `--triattention-*` at 4690-4828) |
| KV-cache allocation / per-layer type overrides | `src/src/llama-kv-cache.cpp` (turbo branches at 263-371) |
| `llama-graph` / context plumbing for the new types | `src/src/llama-graph.cpp`, `src/src/llama-context.cpp`, `src/src/llama-model-loader.cpp`, `src/src/llama-quant.cpp` |
| CUDA arch list / CUDA build options | `src/ggml/src/ggml-cuda/CMakeLists.txt:1-113,149-165,237`; options in `src/ggml/CMakeLists.txt:200-217` |
| Enable/disable a backend | `src/ggml/CMakeLists.txt:200` (`GGML_CUDA`), `src/ggml/src/CMakeLists.txt:484,591` |
| Presets | `src/CMakePresets.json` (CPU/Windows/macOS/SYCL/Vulkan only) |
| Project-specific tests | `src/tests/test-pq2-row-shapes.cpp`, `test-ptq1_0-cuda-dot.cpp`, `test-ptq1_0-element-map.cpp` |
| Launch configuration actually used | `scripts/start_server_turbo.sh`, `scripts/run_cli.sh`, `scripts/start_server_baseline.sh` |
| Prebuilt binaries / bundled CUDA runtime | `build/cuda13.zip` (modern), `build/cuda124.zip` (legacy) |
| Clean source snapshot | `llama-fast-src.zip` |
| Calibration data | `calibration/bonsai-27b.triattention`, `src/calibration_corpus.txt` |
| TriAttention design notes | `src/docs/TRIATTENTION.md`, `src/docs/TRIATTENTION-API.md` |

### Repository boundaries on this machine

[[source-state-md]] §2 names four checkouts. Checked 2026-09-28: `/home/ms/llama-fast/` exists but contains no `llama.cpp/` and no `Release/`; `/home/ms/llama-cpp-turboquant/` exists as a full separate tree; `$HOME/Prism/llama-prism-b10743-adfffbe/` and `dist-rtx3090/` (referenced by the baseline script) do not exist. So **this repository is the only complete llama-fast checkout on the machine**, and the `Release/` source the [[ta-1-wht-inversion-256]] fix is said to live in is not reachable here.

## Open questions

- Is the sparse `src/` tree (1,600 files absent vs `llama-fast-src.zip`, empty `template-instances/`) deliberate publication hygiene, or an incomplete copy? If a CUDA rebuild is ever needed, unpack the zip first — with the tree as-is the CMake globs find no `template-instances/*.cu`.
- Can `sm_70` SASS ever be produced here? The default non-`native` path under CUDA 12.x gives only `70-virtual` (PTX, JIT at runtime); anyone wanting Volta SASS must write `70-real` explicitly. Which the release bundle actually ships is unresolved (see the contradiction note above).
- Was `build/cuda124.zip` really built with CUDA 12.4? Its bundled runtime is `libcudart.so.12.2.128` / `libcublas.so.12.2.4.5` and its PTX is ISA 8.2, which points at a 12.2-era toolchain [UNVERIFIED]. Its single `sm_86` arch is consistent with `GGML_NATIVE=ON` on the RTX 3090 that [[source-readme]] benchmarked on.
- Is `GGML_CUDA_COMPRESSION_MODE` (default `size`, `-compress-mode=`) a fork addition or inherited? Not determinable from this tree. It is why the CUDA 13 fatbins are zstd-compressed.
- Which files are fork-modified vs untouched upstream cannot be answered locally without an upstream reference checkout or git history — the classification above is by content (new type ids, new kernel files), not by diff.

## See also

[[overview]] · [[upstream-lineage]] · [[performance-profile]] · [[roadmap]] · [[benchmarks]] · [[build-and-verify]]
[[triattention]] · [[turboquant]] · [[innerq]] · [[prismml-weight-kernels]] · [[walsh-hadamard-transform]] · [[cuda-graphs]] · [[speculative-decoding]] · [[ternary-bonsai-2-27b]] · [[qwen3-dflash-draft]] · [[v100-sxm2]]
[[kv-cache]] · [[kv-eviction]] · [[quantization]] · [[gemm-dispatch]]
[[tq-1-missing-gemm-kernels]] · [[ta-1-wht-inversion-256]] · [[source-readme]] · [[source-state-md]]
