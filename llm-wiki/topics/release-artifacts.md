---
title: Release artifacts
type: topic
status: current
updated: 2026-09-28
sources: [README.md]
verified: [llama-fast-src.zip, build/cuda124.zip, build/cuda13.zip, scripts/start_server_baseline.sh, llmwiki/raw/README.md]
tags: [release, artifacts, cuda, deployment]
---

# Release artifacts

## Bottom line

What this repository *actually* ships is three ZIPs and a handful of small files — and the layout the README promises does not exist. The two prebuilt CUDA bundles live in `build/` (`build/cuda124.zip`, 811 787 748 B; `build/cuda13.zip`, 1 663 354 910 B), the source snapshot lives at the repo root (`llama-fast-src.zip`, 39 379 165 B), and there is **no `Release/` directory anywhere** — `ls -d Release` returns "No such file or directory" although the README's *Release Directory Structure* section documents a full `Release/{src,calibration,build/cuda12.4,build/cuda13,llama-fast-src.zip,README.md}` tree ([[source-readme]], raw snapshot `llmwiki/raw/README.md:24-55`). The bundles are therefore published **flat**, at the paths that happen to exist here, not under the hierarchy the documentation describes.

For the V100 target the situation is narrower than "two bundles, pick one":

- The **CUDA 12.4 bundle is the only one advertised for Volta** — README's compatibility table lists `sm_61, sm_70, sm_75, sm_80, sm_86` for `build/cuda12.4/` (`llmwiki/raw/README.md:64`, [[source-readme]]) — **but its shipped `libggml-cuda.so` shows no `sm_70` marker** ([[v100-sxm2]], [[build-and-verify]]). The marker-decode itself cannot be repeated here (no CUDA toolkit on this machine), so the honest verdict is **unconfirmed for Volta** *[limit: no cuobjdump/nvdisasm available to re-decode the cubin]*.
- The **CUDA 13 bundle cannot run on `sm_70` at all**: its arch list starts at `sm_75` (README `:63`; the library itself shows `sm_75`…`sm_120a`, [[v100-sxm2]]). Volta is simply not in it.

The source snapshot is the one artifact with unambiguous value: it is the only place the 138 missing `template-instances/*.cu` kernel sources can be recovered from ([[codebase-map]], [[build-and-verify]]).

## Evidence

### Every artifact at the repo root and in `build/`

| Path | Size | What it is | Good for | Usable on the V100 target? |
| :--- | ---: | :--- | :--- | :--- |
| `llama-fast-src.zip` | 39 379 165 B | Full fork source snapshot, archive members rooted at `src/`; 138 members match `template-instances/*.cu` (`unzip -l … \| grep -c`) | Restoring the empty `src/ggml/src/ggml-cuda/template-instances/` directory so a CUDA configure can find its named sources | Only as **sources**: it is text, not a binary. Usability is decided by the build you do with it ([[build-and-verify]], [[v100-sxm2]]) |
| `build/cuda124.zip` | 811 787 748 B | Prebuilt "Legacy Release" bundle: `cuda12.4/{README.md,bin/*,calibration/*,lib/cuda/*,run_cli.sh,start_server.sh}` — 53 members, 2 641 229 175 B uncompressed | The **only advertised Volta option**; contains `llama-server`/`llama-cli` shims, `llama-triattention-calibrate`, `bonsai-27b.triattention`, a bundled CUDA 12.2 runtime (`libcudart.so.12.2.128`), cuBLAS/cuBLASLt 12.2.4.5 | **Unconfirmed for Volta** — its `libggml-cuda.so` (236 919 936 B) carries only `sm_86` markers ([[v100-sxm2]]) *[limit: cubin markers not re-decodeable here without a CUDA toolkit]* |
| `build/cuda13.zip` | 1 663 354 910 B | Prebuilt "Universal Fat Binary" bundle: `cuda13/{README.md,bin/*,calibration/*,lib/cuda/*,run_cli.sh,start_server.sh}` — 44 members, 2 264 021 709 B uncompressed; `libggml-cuda.so` is 337 985 192 B | Turing → Blackwell ([[source-readme]]); carries the fork tools and tests (`test-pq2-row-shapes`, `test-ptq1_0-cuda-dot`) | **No** — arch list begins at `sm_75`; `sm_70` is absent, so it cannot load on Volta at all ([[v100-sxm2]]) |
| `calibration/bonsai-27b.triattention` | 789 571 B | Precomputed TriAttention statistics profile for the Ternary-Bonsai-2-27B model | Passing to `--triattention-stats` for the [[ternary-bonsai-2-27b]] eviction path; identical member also sits inside both bundles (`cuda12.4/calibration/`, `cuda13/calibration/`) | Architecture-independent data — usable, but only together with a binary that runs ([[triattention-calibrate]]) |
| `scripts/run_cli.sh` | 1 540 B | Launch wrapper for `llama-cli` | Convenience wrapper | Inherits whatever binary it points at; see caveat below |
| `scripts/start_server_baseline.sh` | 1 584 B | `llama-server` launcher for FP16 KV (baseline). Defaults `MODEL_PATH=${HOME}/Prism/llama-prism-b10743-adfffbe/Ternary-Bonsai-2-27B-PQ2_0.gguf`, `MMPROJ_PATH=…/mmproj-Qwen3.8-27B-BF16.gguf`, `BIN_DIR=${BASE_DIR}/../dist-rtx3090/bin` | Baseline comparison arm | **Not runnable as-is** — see the path discrepancy below |
| `scripts/start_server_turbo.sh` | 2 438 B | `llama-server` launcher for the quantized-KV / TriAttention configuration | The optimised arm | Same caveat as the baseline script |
| `README.md` | 11 218 B | Release documentation: layout, compatibility table, launch recipes, native-build recipe | The **intent** of the release, not its shape | n/a (documentation; contradicted by the filesystem) |
| `state.md` | 10 821 B | Issue/optimisation inventory behind the vault | Context for the four optimisations | n/a |
| `AGENTS.md`, `llmwiki.txt`, `LICENSE`, `skills-lock.json` | 12 047 / 11 985 / 1 066 / 7 801 B | Repository conventions, wiki import, licence, tooling lock | Repo hygiene | n/a |

### The `Release/` discrepancy

The README's *Release Directory Structure* (`llmwiki/raw/README.md:24-55`) documents a `Release/` root containing `src/`, `calibration/`, a `build/` with `cuda13/` and `cuda12.4/` subtrees, `llama-fast-src.zip`, and its own `README.md`. **None of it is present**: there is no `Release` directory in this repository, and the actual artifacts sit directly at `build/` and the repo root. The README's own quick-start (`cd Release && ./start_server.sh …`, `:71-74`) therefore names a path that cannot be followed; the launchers that *do* ship live *inside* the zips (`cuda12.4/start_server.sh`, `cuda13/start_server.sh`), not at a `Release/` top level.

The same gap runs through the scripts. `scripts/start_server_baseline.sh` hard-defaults its inputs to paths under `${HOME}/Prism/` — `${HOME}/Prism/llama-prism-b10743-adfffbe/Ternary-Bonsai-2-27B-PQ2_0.gguf` and the matching `mmproj-Qwen3.8-27B-BF16.gguf` — and its binary directory to `${BASE_DIR}/../dist-rtx3090/bin`. On this machine **none of those exist**: `$HOME/Prism` is absent, and there is no `dist-rtx3090` directory anywhere. So the scripts document a working environment (a Prism/RTX-3090 development layout) that is **not** the checked-out tree. The model they point at is the [[ternary-bonsai-2-27b]] `PQ2_0` GGUF, which is external to the repository and must be supplied by the consumer.

### What a consumer can and cannot rely on

- **Reliable:** the ZIPs are intact and self-describing; each contains its own `README.md`, a `bin/` with the fork tools (including `llama-triattention-calibrate`), a `calibration/` with `bonsai-27b.triattention` + `calibration_corpus.txt`, and `lib/cuda/` with a bundled runtime. The source snapshot genuinely holds the missing kernel sources.
- **Not reliable, for Volta:** the CUDA 12.4 bundle is advertised for Volta (`sm_61`, `sm_70`, …) yet its `libggml-cuda.so` exposes only `sm_86` markers ([[v100-sxm2]], [[build-and-verify]]) — **unconfirmed for Volta** *[limit: the cubin/marker decode cannot be re-run here; without a CUDA toolkit (`cuobjdump`/`nvdisasm`) the presence or absence of an `sm_70` PTX fallback inside that `.so` is undecidable from this repository]*. The README's own native-build example likewise omits `70` from `CMAKE_CUDA_ARCHITECTURES` (`:236`, [[build-and-verify]], [[v100-sxm2]]).
- **Impossible, for Volta:** the CUDA 13 bundle. Its arch list starts at `sm_75`, so there is no code path for a `sm_70` device — this is a hard mismatch, not a performance question.
- **Documentation vs. reality:** the `Release/` layout and the `start_server*.sh` default paths are aspirational. Any consumer should treat the file listing as authoritative and the README's paths as intent ([[source-readme]], [[codebase-map]]).
- **No measurement of either bundle on the target exists.** The published numbers were taken on an RTX 3090, not a V100 ([[benchmarks]]); nothing in the vault records a run of either bundle on Volta.

### Two things worth extracting, and what to check first

1. **The missing kernel sources.** `unzip -o llama-fast-src.zip '*template-instances/*.cu'` recovers the 138 `.cu` files that `src/ggml/src/ggml-cuda/CMakeLists.txt` globs and names but cannot find ([[build-and-verify]], [[codebase-map]]). *Check first:* the exact member prefix inside the archive (`[UNVERIFIED]` — only the count 138 and a `src/`-rooted top level were read, not the full path of a `.cu` member), and re-run `cmake` afterwards, because the globs are configure-time ([[build-and-verify]]).
2. **A runnable binary.** The `build/cuda124.zip` `bin/` + `lib/cuda/` pair is the only candidate for Volta. *Check first:* whether its `libggml-cuda.so` really contains a `sm_70` path (`cuobjdump -lelf`/`nvdisasm` with a CUDA toolkit — unavailable here, hence "unconfirmed"), and whether the bundle's own `start_server.sh` model/calibration defaults resolve, since the repo-level scripts point at `${HOME}/Prism/…` and `dist-rtx3090` that do not exist.

## Open questions

- Does the CUDA 12.4 bundle's `libggml-cuda.so` embed an `sm_70` *PTX* fallback for JIT, or is the archive's `sm_86`-only marker set the whole story? Both bundles' intra-`.so` arch content needs a CUDA toolkit to decode; that toolkit is not on this machine.
- Why does `build/cuda13.zip` (1 663 354 910 B) exceed `build/cuda124.zip` (811 787 748 B) on disk while unpacking *less* (2 264 021 709 B vs 2 641 229 175 B)? The 12.4 zip compresses far better — plausibly LZMA vs. a different mode `[INFERENCE]`; the compression story is not documented in the vault.
- Where is the real `Release/` tree? The README describes one, the filesystem has none, and the launchers' `${HOME}/Prism/…` defaults imply it was assembled in a different environment ([[source-state-md]]'s four checkouts are the related ambiguity — [[codebase-map]]).
- Which bundle, if either, was ever exercised on a V100 — and with which of the launchers? No such run is recorded ([[benchmarks]], [[performance-profile]]).
- Are the two bundles built from the same commit as this working tree, or from the `Release/src` snapshot the README references? The ZIP timestamps (2026-09-26/27) are the only evidence, and no commit hash is recorded in the vault.

## See also

[[source-readme]] · [[codebase-map]] · [[build-and-verify]] · [[v100-sxm2]] · [[benchmarks]] · [[ternary-bonsai-2-27b]] · [[source-state-md]] · [[conversion-and-packing]] · [[prismml-weight-kernels]]
