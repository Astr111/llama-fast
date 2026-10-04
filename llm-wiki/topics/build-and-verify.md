---
title: Build and verify
type: topic
status: current
updated: 2026-09-28
sources: [README.md, state.md]
verified: [src/CMakeLists.txt, src/ggml/CMakeLists.txt, src/ggml/src/CMakeLists.txt, src/ggml/src/ggml-cuda/CMakeLists.txt, src/tools/CMakeLists.txt, src/tools/triattention-calibrate/CMakeLists.txt, src/tests/CMakeLists.txt, src/tests/test-backend-ops.cpp, src/CMakePresets.json, src/build-x64-linux-gcc-debug/CMakeCache.txt, llama-fast-src.zip]
tags: [build, cuda, testing, verification]
---

# Build and verify

## Bottom line

The build surface is **stock llama.cpp plus one fork tool**: `src/CMakeLists.txt` is `project("llama.cpp")` version `0.2.0-dev`, which adds `ggml` (version 0.21.0) and then `src`, `common`, `tests`, `tools`. There is no fork-specific build system, no CUDA preset, and no CUDA target that a plain `cmake -B build` produces — `GGML_CUDA` is `OFF` by default. The only documented CUDA recipe is the one in README (`cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=...`), and it is the one path that cannot be expected to work from this working tree: `src/ggml/src/ggml-cuda/CMakeLists.txt` both globs and **explicitly names** files under `template-instances/`, a directory that is empty here ([[codebase-map]]) while `llama-fast-src.zip` carries 138 `.cu` files for it. As of 2026-09-28 this machine **cannot run the CUDA path at all** — no `nvcc` and no CUDA toolkit are installed — and the tree's only configure record is a CPU-only Ninja/Debug one with `GGML_CUDA=OFF` (`src/build-x64-linux-gcc-debug/CMakeCache.txt`); see *Can this machine build it today?*.

Nothing in this repository has been **built** during the construction of this wiki, and there is **no V100 measurement anywhere in it**. Correction (2026-09-29): it *has* now been **run** — the prebuilt CUDA 13 bundle executes on this machine's GTX 1660 and has served the 4B model, so the engine is measurable even without a toolkit; see [[first-live-measurements]]. Everything below about how a build fails or what a test covers is still read off the CMake files and the test source, not a build result. See *Honest status*.

> **Second missing-source hole (2026-09-29).** `src/ggml/src/ggml-cpu/arch/` — the directory that `src/ggml/src/ggml-cpu/CMakeLists.txt:243-244` lists unconditionally for x86 as `arch/x86/quants.c` and `arch/x86/repack.cpp` — contains **zero files** in this checkout. That is the same shape as the empty `template-instances/` above, and it means the x86 SIMD quant and repack kernels the CPU repack tables point at are not present here at all ([[cpu-path]]). Two independent directories that a build file references and a checkout does not contain changes the reading of the first one: the empty `template-instances/` is not a one-off mistake, it is **how this repository was published** — source subsets are withheld while the build files that require them are not. A CUDA build is blocked by the first hole; a CPU build on x86 is now suspected to be blocked by the second, which is `[INFERENCE]` until someone configures it.


## The two documented build paths

| Path | Shape | What you get |
| :--- | :--- | :--- |
| README recipe | `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=...` | CUDA backend compiled in; the arch string is left to the caller, and the README's own example omits `70` entirely ([[v100-sxm2]]) |
| Stock default | `cmake -B build && cmake --build build` | CPU-only: `option(GGML_CUDA "ggml: use CUDA" OFF)` in `src/ggml/CMakeLists.txt` |

Notes that matter for the V100 target:

- The stock CMake default, when CUDA is enabled but the architecture is not set, emits `70-virtual` — PTX only, JIT-compiled at load — and README's recipe omits `70` ([[v100-sxm2]]). The shipped `build/cuda124.zip` `libggml-cuda.so` is no better: it carries only `sm_86` markers, i.e. Ampere SASS with no Volta path at all ([[v100-sxm2]]).
- `src/CMakePresets.json` offers no accelerator preset for CUDA. `base` is Ninja with `binaryDir = ${sourceDir}/build-${presetName}`; the concrete presets are `x64-linux-gcc-{debug,release,reldbg}[+static-release]`, plus Windows LLVM/MSVC, `arm64-apple-clang`, `sycl-base`, and `vulkan` variants. So `cmake --preset x64-linux-gcc-release` is a CPU build, and there is no `CMAKE_CUDA_ARCHITECTURES` anywhere in the preset file.
- `GGML_CUDA_FA` defaults **ON** and `GGML_CUDA_FA_ALL_QUANTS` defaults **OFF** — which selects the explicit `template-instances/fattn-vec-instance-*.cu` list rather than the `GLOB` — so the default CUDA configuration is precisely the one that needs the missing directory (next section).
- The fork adds `option(GGML_CUDA_HOPPER_Q1 "ggml: opt-in sm_90a wgmma path for Q1_0/PQ2_0 prefill" OFF)` with `GGML_CUDA_CUTLASS_DIR`, and defaults `GGML_CUDA_COMPRESSION_MODE` to `size` ("requires cuda 12.8+"). None of that helps Volta; the PQ2_0 prefill path is sm_90a ([[prismml-weight-kernels]]).

## What each CMake file owns

| File | Owns |
| :--- | :--- |
| `src/CMakeLists.txt` | Project identity (`llama.cpp` `0.2.0-dev`, `LLAMA_BUILD_IS_DEV`), `CMAKE_BUILD_TYPE` defaulting to `Release`, module path `src/cmake/`, the option surface (`LLAMA_BUILD_TESTS`/`TOOLS`/`EXAMPLES`/`SERVER`/`APP`, all defaulting to `${LLAMA_STANDALONE}`), the deprecated-`LLAMA_CUBLAS`→`GGML_CUDA` shim, `add_subdirectory(ggml)`, `add_subdirectory(src)`, `include(CTest)` + `add_subdirectory(tests)` gated on `LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS`, `add_subdirectory(tools)`, and the install/package rules (`install(TARGETS llama LIBRARY PUBLIC_HEADER)`, `install(TARGETS llama-common LIBRARY)`, `llama-config.cmake`) |
| `src/ggml/CMakeLists.txt` | ggml version (`0.21.0`), git-commit/dirty stamping, and the whole backend/feature option block: `GGML_CUDA` (OFF), `GGML_CUDA_FA` / `GGML_CUDA_FA_ALL_QUANTS` / `GGML_CUDA_GRAPHS` / `GGML_CUDA_NCCL` / `GGML_CUDA_COMPRESSION_MODE` / `GGML_CUDA_HOPPER_Q1`, and the CPU instruction-set and other-backend options (`GGML_BLAS`, `GGML_VULKAN`, `GGML_SYCL`, `GGML_METAL`, …). `GGML_BUILD_TESTS`/`GGML_BUILD_EXAMPLES` default to `${GGML_STANDALONE}`. Backend subdirectories (including `ggml-cuda`) are added from the part of this file after the option block; that wiring is below line 300 and was not read for this page [UNVERIFIED] |
| `src/ggml/src/CMakeLists.txt` | The `ggml-base` target and its source list — notably the fork's own `ggml-turbo-quant.c` — plus compile flags, `_XOPEN_SOURCE`/`_GNU_SOURCE` conformance, sanitizer/CCTV flags (`GGML_SANITIZE_*`), ccache/sccache wiring, and the `GGML_BACKEND_DL` guard (`:188-190`, `FATAL_ERROR` unless `BUILD_SHARED_LIBS`) |
| `src/ggml/src/ggml-cuda/CMakeLists.txt` | The CUDA backend target and its kernel inventory: `file(GLOB GGML_SOURCES_CUDA "*.cu")` plus four globs into `template-instances/` (`:105-113`), the `GGML_CUDA_FA_ALL_QUANTS` branch that globs `template-instances/fattn-vec*.cu`, and the **default** `else()` branch (`:115+`) that appends an explicit list of `template-instances/fattn-vec-instance-*.cu` names. The `native` arch rewriting (`:94-100`) is guarded so that a ninja build with no GPU attached does not get garbage architectures |
| `src/tools/CMakeLists.txt` | The tools subtree only: `add_subdirectory(triattention-calibrate)` is unconditional; `llama-bench`, `quantize`, `imatrix`, `perplexity`, `kv-mean-center`, `tts`, `mtmd`, `results`, … plus `cli`/`server`/`ui` gated on `LLAMA_BUILD_SERVER`, `rpc` on `GGML_RPC`, and `cvector-generator`/`export-lora` only when not `GGML_BACKEND_DL` |
| `src/tools/triattention-calibrate/CMakeLists.txt` | All 8 lines of it: `set(TARGET llama-triattention-calibrate)`, `add_executable(${TARGET} triattention-calibrate.cpp)`, links `llama-common llama ${CMAKE_THREAD_LIBS_INIT}`, `cxx_std_17`, install gated on `LLAMA_TOOLS_INSTALL` |
| `src/tests/CMakeLists.txt` | Test registration, not test logic: `llama_build` / `llama_test` / `llama_test_cmd` / `llama_build_and_test` helpers wrapping `add_test` (default label `main`), plus test fixtures. Fork-specific entries: `test-kv-mean-center` (`:264`), the dspark gates `test-dspark-forward` / `test-dspark-logsnr-meta` / `test-dspark-loop` / `test-dspark-real-eval`, `test-recurrent-state-rollback`, `test-dfly-fusion`, `test-llama-archs` (fixture `test-generate-models` writes synthetic GGUF models), and `test-download-model` (pulls `stories15M-q4_0.gguf`, SHA256-pinned). `llama_build_and_test(test-backend-ops.cpp)` at `:316` |

## Targets

- **Libraries:** `llama` (created under `src/src/`, installed by the top-level file [UNVERIFIED for that subdirectory]), `llama-common`, `ggml-base`, and the enabled backends.
- **Tool binaries:** `llama-triattention-calibrate` is verified verbatim; `llama-cli` and `llama-server` come from `tools/cli` and `tools/server`, which `src/tools/CMakeLists.txt` only adds behind `LLAMA_BUILD_SERVER` [UNVERIFIED — the subdirectory files were not read, the names are the vault's and the assignment's].
- **Tests:** every `llama_build*` entry (each executable named after its source file unless `NAME` overrides), all wired into CTest because `src/CMakeLists.txt` does `include(CTest)` and adds the `tests` subdirectory when `LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS`. The fork's PrismML weight-type tests are registered here: `test-ptq1_0-element-map`, `test-ptq1_0-cuda-dot`, `test-pq2-row-shapes` (`:339-341`).
- **Artifacts it ships:** two prebuilt bundles, `build/cuda13.zip` (modern) and `build/cuda124.zip` (legacy) ([[codebase-map]]); the legacy bundle's `libggml-cuda.so` carries only `sm_86` markers ([[v100-sxm2]]). There is no build target that produces or validates either — they are inputs, not outputs.
- **Not targets:** there is nothing that builds or validates the V100 numbers, and no target that regenerates `template-instances`.

## Why a CUDA build from this tree is expected to fail

The directory `src/ggml/src/ggml-cuda/template-instances/` contains **0** files, while `src/ggml/src/ggml-cuda/CMakeLists.txt:106-116` globs it, and `llama-fast-src.zip` carries 138 `.cu` files under that path ([[codebase-map]]).

Read against the CMake text, that emptiness bites twice:

1. **Hard failure (default configuration).** With `GGML_CUDA_FA_ALL_QUANTS=OFF` — the default — the `else()` branch at `src/ggml/src/ggml-cuda/CMakeLists.txt:115+` appends individual paths such as `template-instances/fattn-vec-instance-f16-f16.cu`, `...-turbo3_0-turbo3_0.cu`, `...-turbo2_0-turbo4_0.cu` to `GGML_SOURCES_CUDA`. A path listed by name rather than globbed must exist at generate time, so CMake is expected to stop with `Cannot find source file:` on the first of them. **[INFERENCE]** — the CMake semantics are standard, the file list is text in the file, and the directory's emptiness is a verified vault fact; the failure itself was not reproduced here (no builds, see *Honest status*).
2. **Silent loss of kernels.** The four `file(GLOB ... "template-instances/…")` calls (`:105-113`, and `:116` in the all-quants branch) return an empty list for a missing directory without any diagnostic, so even with the named-list problem worked around, the backend would compile without its `fattn-tile*`, `fattn-mma*`, `mmq*` and `mmf*` template kernels. `file(GLOB)` is evaluated at configure time and has no `CONFIGURE_DEPENDS` here, so restoring files later requires re-running `cmake` before `cmake --build`.

Note what the turbo instances are doing in that list at all: `turbo2_0`, `turbo3_0` and `turbo4_0` flash-attention instantiations are part of the default CUDA source list, i.e. the missing directory is the KV-side kernel inventory for the quantized cache ([[turboquant]], [[kv-cache]]). Whether `template-instances/` is the *only* hole in the working tree's `ggml-cuda` directory is not recorded anywhere in the vault (see *Open questions*).

## Restoring the template instances

`llama-fast-src.zip` is the archival snapshot of this tree and contains the 138 `.cu` files for that directory ([[codebase-map]]). One line:

```bash
unzip -o llama-fast-src.zip '*ggml-cuda/template-instances/*.cu' -d /tmp/ti \
  && cp -n /tmp/ti/*/template-instances/*.cu src/ggml/src/ggml-cuda/template-instances/
```

The member-path wildcard is written to tolerate any prefix depth inside the archive **[INFERENCE — the zip's member layout was not listed]**; the copy is `-n` so an already-populated directory is never overwritten. After restoring, re-run CMake (the globs are configure-time, above).

## Verifying a kernel change

The test surface is CTest over the executables registered in `src/tests/CMakeLists.txt`, of which exactly one exercises backend kernels directly: **`test-backend-ops`**.

What it does (`src/tests/test-backend-ops.cpp`):

- Enumerates every device the backend registry reports (`ggml_backend_dev_count` / `ggml_backend_dev_get`, `:11275-11306`), skips CPU devices unless the mode is `grad` (`:11286-11289`), and runs each op case on the device against a **CPU reference** (`test_backend` → `test->eval(b, b_cpu, op_names_filter, …)`, `:10881`, `:10960`), printing per-op `OK`/`FAIL`/`SKIPPED`/`NOT_SUPPORTED` and exiting non-zero if any backend failed (`:11324-11328`).
- Op names come from the `GGML_OP_*` / `GGML_UNARY_OP_*` enums (`:11071-11102`); each case class reports its op via `ggml_op_name`, and `matches_filter` accepts comma-separated op names or `op(full-name)` variations (`:1286-1300`).
- Invocation (`usage`, `:11166-11168`; parsing `:11192-11248`):
  `test-backend-ops [test|perf|grad|support] [-o <op,..>] [-b <backend>] [-p <params regex>] [--output console|sql|csv] [--list-ops] [--show-coverage] [--test-file <path>] [-j <n>]` — mode `test` is the default, `perf` measures instead of comparing, `support` reports what the backend claims, `--list-ops` and `--show-coverage` dump the op set without running anything.

What it does **not** cover for this fork:

- **No turbo type appears anywhere in the file.** A case-insensitive search for `turbo` over `src/tests/test-backend-ops.cpp` returns zero matches, so `turbo2_0`/`turbo3_0`/`turbo4_0` matmul or FA paths are never compared against the CPU reference by this test, and `--show-coverage` cannot report on them beyond whatever generic op they ride on. The vault records **no turbo-type coverage in the tests** — the same conclusion a sibling reached from the page inventory: `test-backend-ops` is mentioned on exactly **one** page, [[prism-hadamard-weight-fold]], and there only for the fork's Hadamard/FWHT op cases (`test_mul_mat_hadamard`, `test_fwht_signed`).
- The fork's **weight**-side types do have registered tests — `test-ptq1_0-element-map`, `test-ptq1_0-cuda-dot`, `test-pq2-row-shapes` (`src/tests/CMakeLists.txt:339-341`) — but the **KV**-side turbo types have neither a file nor a target: no `turbo` string occurs in `src/tests/CMakeLists.txt` either, no TriAttention/eviction test is registered there, and `src/tools/triattention-calibrate/CMakeLists.txt` only builds the tool. The dspark gates in `src/tests/CMakeLists.txt` (`test-dspark-forward`, `test-dspark-loop`, `test-dspark-real-eval`) are explicit `llama_build`-only entries — the file's comments say they must be run manually, since they need a real GGUF on the command line.

So a kernel change here would be checked by hand, in roughly this order **[INFERENCE — plan, no part of it has been run]**:

1. Configure a CUDA build (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70` for the V100; see [[v100-sxm2]] for why the arch string is the whole story), restore `template-instances/` first, then `cmake --build build --target test-backend-ops`.
2. `./bin/test-backend-ops --list-ops` / `--show-coverage`, then `./bin/test-backend-ops test -b CUDA -o <OP> -p '<params regex>'` to diff the CUDA implementation against the CPU reference on the shapes you care about; `perf` mode for timing, not correctness.
3. Where that is blind — every turbo KV type, the FWHT path at `head_dim=256`, the eviction path, `gemm-dispatch` fallbacks — the only honest fallback is an end-to-end run (`llama-cli`/`llama-server` with the PQ2_0 ternary model, [[ternary-bonsai-2-27b]], plus a calibration corpus for [[innerq]]) and a `llama-bench`-style measurement on the actual V100, which is exactly what nobody has done yet ([[benchmarks]], [[performance-profile]]).

## Honest status

**Nothing in this repository has been built or run during the construction of this wiki** — no `cmake` configure, no `ctest`, no CUDA kernel execution, no `llama-bench` run appears anywhere in the record. **No V100 measurement exists**: every published number in the vault (the 1.39× and the 25 200 tok/GB figures) was measured on an RTX 3090, which has INT tensor cores and a different memory subsystem ([[source-readme]], [[benchmarks]]).

The figures on this page — 0 files in `template-instances/`, 138 `.cu` files in the archive, `sm_86`-only markers in the released `libggml-cuda.so` — are **file-level facts**, not build results. Treat the failure mode in *Why a CUDA build from this tree is expected to fail* and the procedure in *Verifying a kernel change* as a **plan with a strong paper basis**, not as a report. Anyone planning work on this tree should budget the first hour for confirming that the build behaves as predicted ([[roadmap]]).

## Evidence

- Build paths, CUDA arch caveats, and the archive/shipped-binary facts: [[codebase-map]], [[v100-sxm2]], and README's build section ([[source-readme]]).
- CMake ownership: `src/CMakeLists.txt`, `src/ggml/CMakeLists.txt`, `src/ggml/src/CMakeLists.txt`, `src/ggml/src/ggml-cuda/CMakeLists.txt:94-128`, `src/tools/CMakeLists.txt`, `src/tools/triattention-calibrate/CMakeLists.txt`, `src/tests/CMakeLists.txt`, `src/CMakePresets.json` (all read for this page).
- Test surface: `src/tests/CMakeLists.txt:316` (backend ops), `:339-341` (PrismML type tests), and `src/tests/test-backend-ops.cpp` (`:11071-11102`, `:11166-11248`, `:11275-11328`); zero `turbo` matches in either file.
- Why the backend test cannot settle the fork's open questions: [[gemm-dispatch]], [[performance-profile]], [[quantization]].

## Open questions

- Does a `GGML_CUDA=ON` configure actually fail on the named `template-instances` entries, or does something earlier (toolkit detection, `native` arch rewriting at `src/ggml/src/ggml-cuda/CMakeLists.txt:94-100`) stop it first? Nothing here has been run. **Settled for this machine** — see *Can this machine build it today?*: toolkit detection fires first, because there is no `nvcc` here at all.
- Is `template-instances/` the only missing directory in the working tree's `ggml-cuda/`, or are the top-level `*.cu` sources (the `file(GLOB GGML_SOURCES_CUDA "*.cu")` set) incomplete too? The vault records the empty `template-instances/` and nothing else. **Now measured** (see *Can this machine build it today?*): the `ggml-cuda/` diff against the archive is exactly 142 absent files — 138 `.cu` templates + `generate_cu_files.py` + `vendors/{cuda,hip,musa}.h` — and nothing else, since the top-level `*.cu` set is complete (71 = 71).
- With `template-instances/` restored, would a `sm_70` build link at all — and do the turbo fattn instantiations even compile for Volta? The instantiation list is arch-independent CMake, and the kernels' own arch guards were not read.
- Should a turbo-type case exist in `test-backend-ops`? Every other fork-specific mechanism (WHT/Hadamard has cases there; TriAttention, InnerQ, the dspark gates) currently has none, so a kernel regression in the turbo path would only surface end-to-end on hardware nobody has measured.
- Which artifacts in the tree are actually current — `build/cuda124.zip` is an `sm_86` binary from an Ampere-era build ([[v100-sxm2]]), and the vault's four checkouts ([[source-state-md]] §2) make "the build directory" ambiguous.

## Can this machine build it today?

**Verdict: no CUDA build — not "fails", *impossible*, because there is no CUDA toolkit here; and no build of any kind has been run or completed. Only a CPU-only configure is possible today.** Three things are missing: (a) `nvcc` + headers + `cuobjdump`/`nvdisasm`/`ptxas` (nothing under `/opt/cuda`, no `cuda` package installed); (b) the 142 `ggml-cuda/` files absent from the working tree, of which the 138 `template-instances/*.cu` are the ones CMake names and globs; (c) any evidence of a finished build — `src/build-x64-linux-gcc-debug/` holds configure output (`.` 20:50, `CMakeFiles/`, `DartConfiguration.tcl`, `Testing/`) but no `bin/` and no `*.so`.

### What is on the machine (probed read-only, 2026-09-28)

| Probe | Result |
| :--- | :--- |
| `which nvcc` / `nvcc --version` | nothing on `PATH`; `command not found` |
| `ls -d /opt/cuda` | `No such file or directory` |
| `ls -d /usr/local/cuda*` | no match |
| `pacman -Qs cuda` | no CUDA package installed |
| `ls /usr/bin/cuobjdump` | `No such file or directory` (nor `nvdisasm`, nor `ptxas`) |
| `ls /dev/nvidia*`, `nvidia-smi` | **present** — `NVIDIA GeForce GTX 1660, compute capability 7.5, driver 615.71.09` |

The driver and a GPU are installed — which is why `nvidia-smi` answers while the compiler does not — so `-DGGML_CUDA=ON` is expected to die at toolkit detection, i.e. *before* it ever reaches the missing `template-instances/` files **[INFERENCE: standard CMake behaviour when no `CMAKE_CUDA_COMPILER` can be found; not run]**. Note *which* GPU: `sm_75`, not the target's `sm_70` V100 ([[v100-sxm2]]). Even with a toolkit installed, a build here would produce SASS for the wrong device, so no V100/TurboQuant question is decidable on this machine.

### What the tree's own configure cache says

`src/build-x64-linux-gcc-debug/CMakeCache.txt` is the only record in the repository of how this tree was actually configured (37 693 B, mtime `2026-09-28 17:45:46 +0600` — CMake does not stamp the cache, so the mtime is the record):

| Fact | Value | Line |
| :--- | :--- | ---: |
| Generator | `Ninja` | `:974` |
| Source dir | `/home/ms/Загрузки/llama-fast/src`, generated by `/usr/bin/cmake` | `:985` |
| `GGML_CUDA` | **`OFF`** | `:418` |
| `CMAKE_CUDA_ARCHITECTURES` / `CMAKE_CUDA_COMPILER` | **absent from the whole file** — the CUDA language was never enabled, so no arch string was ever recorded | — |
| Compilers | `gcc` / `g++`, both `UNINITIALIZED` (found, not set) | `:90`, `:62` |
| `CMAKE_BUILD_TYPE` | `Debug` | `:57` |
| CUDA sub-options left at their header defaults | `GGML_CUDA_FA=ON` `:427`, `GGML_CUDA_FA_ALL_QUANTS=OFF` `:430`, `GGML_CUDA_GRAPHS=ON` `:439`, `GGML_CUDA_COMPRESSION_MODE=size` `:421`, `LLAMA_DSPARK_MARKOV_CUDA=OFF` `:740` | — |

**What it proves.** The only configuration this tree has ever had is the shape of `cmake --preset x64-linux-gcc-debug` — Ninja + gcc/g++ + Debug + `binaryDir = src/build-x64-linux-gcc-debug` reproduces that preset's four defining fields (`src/CMakePresets.json`) **[INFERENCE on the preset identity; the four fields match exactly]** — and it is CPU-only: `GGML_CUDA=OFF` means the `ggml-cuda` subdirectory was never added, so the empty `template-instances/` has never even been inside a configure path. Conversely the cache records nothing about CUDA arch, toolkit, or the missing files, because a CPU configure does not touch `ggml-cuda` at all.

### The exact restore path for the empty `template-instances/`

Listed from `llama-fast-src.zip` (every member is `src/`-rooted):

| Listing | Count |
| :--- | ---: |
| `unzip -l llama-fast-src.zip \| grep 'ggml-cuda/template-instances/.*\.cu$' \| wc -l` | **138** |
| same without `\.cu$` (all members under that path) | 139 — the 138 `.cu` plus `generate_cu_files.py` |
| `… (fattn-tile\|fattn-mma\|mmq\|mmf)*.cu` — the four default globs | 74 |
| `… fattn-vec*.cu` — globbed only when `GGML_CUDA_FA_ALL_QUANTS=ON` | 64 (= 74 + 64 = 138) |
| `grep 'template-instances/'` over the whole archive | 185 — the other 46 are `ggml-sycl/template-instances/*.cpp` |
| working tree `src/ggml/src/ggml-cuda/template-instances/` | **0 files** |

Real member prefix, first and last of the `.cu` set: `src/ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_16.cu` … `src/ggml/src/ggml-cuda/template-instances/mmq-instance-ptq1_0.cu`. Because the members are already `src/`-rooted, `-d .` from the repository root lands them exactly where `src/ggml/src/ggml-cuda/CMakeLists.txt:105-116` looks. The exact one line:

```bash
unzip -n llama-fast-src.zip 'src/ggml/src/ggml-cuda/template-instances/*' -d .
```

`-n` never overwrites, so the command is idempotent and writes only into that one empty directory (139 members). Nothing was extracted while writing this page — the directory is still 0 files. (This supersedes the placeholder wildcard in *Restoring the template instances* above, which assumed a shallow `*/template-instances/*.cu` path.) Because `file(GLOB …)` and the explicit named list are both configure-time, `cmake` must be **re-run** before any build.

**138 and 142 are both right and count different sets.** 142 is exactly *"archive files under `ggml-cuda/` that are absent from the working tree"* — the 138 `.cu` template instances plus `generate_cu_files.py` plus `vendors/{cuda,hip,musa}.h`. So the hole is not only the kernel templates: `vendors/cuda.h` is included by a CUDA build too. Not missing is the top-level kernel set — `ggml-cuda/*.cu` is 71 files in the tree and 71 in the archive (162 of the 304 `ggml-cuda` archive members are present), so those 142 files are the entire defect, not the visible tip of a more broadly stripped directory.

### First three commands of a real attempt

```bash
# from the repository root
unzip -n llama-fast-src.zip 'src/ggml/src/ggml-cuda/template-instances/*' -d .   # 1. the only repair a configure needs
cmake -S src -B src/build-x64-linux-gcc-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF   # 2. the only configure possible here
cmake --build src/build-x64-linux-gcc-release -j --target test-backend-ops llama-cli   # 3. build the one kernel test
```

Steps 2–3 are unrun. The true first command of a *CUDA* attempt is installing a toolkit (`pacman -S cuda`, outside this wiki's read-only remit); even then the arch this box wants is `75`, while the target asks for `70` ([[v100-sxm2]]).

## See also

[[codebase-map]] · [[release-artifacts]] · [[v100-sxm2]] · [[roadmap]] · [[benchmarks]] · [[turboquant]] · [[gemm-dispatch]] · [[performance-profile]] · [[quantization]] · [[prismml-weight-kernels]] · [[prism-hadamard-weight-fold]]
