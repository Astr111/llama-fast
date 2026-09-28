# llmwiki log

Append-only record of what happened in this vault. Newest last. Format: `## [YYYY-MM-DD] op | subject`.
`grep "^## \[" log.md | tail -5` gives the last five operations.

## [2026-09-28] setup | vault instantiated
Created `llmwiki/` as an Obsidian vault with the three layers from `raw/llmwiki.txt`: `raw/` (immutable snapshots), generated pages by type (`sources/`, `entities/`, `concepts/`, `issues/`, `topics/`), and the schema split between the *LLM Wiki* section of the root `AGENTS.md` (invariants) and `llmwiki/SCHEMA.md` (operational detail). Wrote `llmwiki/lint.sh` for the mechanical pass.

## [2026-09-28] ingest | state.md — TriAttention and TurboQuant defect inventory
Snapshot: `raw/state.md` (sha256 `a5b15a9a…`). Derived one source page, two synthesis topics, and 14 issue pages: `ta-1`…`ta-7` and `tq-1`…`tq-7`. Verified paths against the checkout — state.md's paths are relative to a different tree, so the repo-relative form is `src/src/llama-triattention.cpp` and `src/ggml/src/ggml-cuda/triattention-score.cu`.

## [2026-09-28] ingest | README.md — release surface and measurements
Snapshot: `raw/README.md` (sha256 `05dfb403…`). Derived the source page, `benchmarks`, `upstream-lineage`, and fed `overview`, `performance-profile`, `v100-sxm2`, `ternary-bonsai-2-27b`.

## [2026-09-28] ingest | llmwiki.txt — the pattern this vault implements
Snapshot: `raw/llmwiki.txt` (sha256 `dc3efe98…`). Source page records the three layers, the three operations, and the deliberate deviations this instantiation makes (snapshot raw layer with hashes; `lint` as a script).

## [2026-09-28] ingest | AGENTS.md — constitution and schema
Snapshot: `raw/AGENTS.md` (sha256 `9021c1db…`). Captured the constraints the wiki inherits, notably the PrismML CUDA kernel edit ban, which any fix routed through `[[prismml-weight-kernels]]` must respect.

## [2026-09-28] contradict | published benchmarks are Ampere, deployment target is Volta
First recorded contradiction, and it is a live one: `README.md` reports 1.39× and ~25 200 tok/GB measured on an **RTX 3090**, while `state.md` names a **Tesla V100-SXM2-16GB** as the target — no INT tensor cores. Neither source is wrong; the claim "these numbers describe the deployment target" is unsupported. Filed in `benchmarks.md` and `overview.md` as an open question, not silently reconciled. Also noted: README's native-build arch list omits `sm_70`, which its own CUDA 12.4 release ships.

## [2026-09-28] contradict | the TA-1 fix does not exist on this machine
`state.md` §3 TA-1 and §5 item 1 say the WHT inversion fix lives in `/home/ms/llama-fast/Release/`. That path does not exist here (checked during ingest). Recorded in `issues/ta-1-wht-inversion-256.md` as unverifiable rather than repeated as fact — it changes what "port the fix" (roadmap item 1) actually requires.

## [2026-09-28] lint | initial pass
`llmwiki/lint.sh` written and exercised; the script itself was corrected three times against real findings (frontmatter parse, vault-relative link resolution, repo-relative raw paths in `Provenance`). Findings at this point were only the pages still being written.

## [2026-09-28] ingest | TRIATTENTION.md + TRIATTENTION-API.md — the project's own design docs
First-party design documents under `src/docs/`, snapshotted into `raw/` with hashes (`265cb520…` and `a1e0e46e…`). Produced two source pages and `[[triattention-calibrate]]`. These are the best prose descriptions of TriAttention that exist, and they are the reason the vault could finally say what the subsystem is *meant* to do — as distinct from what the code does.

## [2026-09-28] contradict | the dominant-cost kernel cannot be found in this tree
`state.md` §1.3 attributes **38.81 % (157 ms)** of GPU time to `magma_sgemmEx_kernel<float, __nv_bfloat16>`. Three independent readers of this checkout found **no MAGMA anywhere in `src/`** — CMake exposes only `GGML_CUDA_FORCE_CUBLAS`, and the in-repo fallback is cuBLAS (`cublasGemmEx`/`cublasSgemm*`) with an **F16** compute type (`ggml-cuda.cu:1626-1628`), not bf16. The profiler symbol is either from a different build or from inside the vendor BLAS. Recorded on `gemm-dispatch.md`, `tq-1-missing-gemm-kernels.md`, `turboquant.md`, `performance-profile.md` — the attribution stays in the wiki as a claim, but it no longer reads as a fact about this code.

## [2026-09-28] contradict | the dispatch story behind TQ-1 is weaker than recorded — and stronger
Two corrections, in opposite directions. **Weaker:** `ggml_backend_cuda_device_supports_op` (`ggml-cuda.cu:5241-5269`) omits every `GGML_TYPE_TURBO*` from the `GGML_OP_MUL_MAT` allow-list, so the scheduler may never place such a node on CUDA at all — the causal chain from "type missing from `should_use_mmq`" to "38.8 % in cuBLAS" does not hold as written. **Stronger:** TurboQuant K/V in the standard graph is consumed mainly by the *fused attention* kernels (`vec_dot_fattn_vec_KQ_turbo{3,2,4}_0`, the only CUDA turbo dot products that exist), with `GGML_OP_TURBO_WHT`/set_rows for the rotation. `vecdotq.cuh` contains no `turbo` at all. Both facts recorded in `gemm-dispatch.md`, `tq-1-missing-gemm-kernels.md`, `quantization.md`, `turboquant.md`.

## [2026-09-28] contradict | the released Volta bundle is not a Volta build
Measured from `build/cuda124.zip` rather than from prose: its `libggml-cuda.so` (236.9 MB) carries **one** PTX target string across all 164 modules — `.target sm_86` — and no `sm_61`/`sm_70`/`sm_75`/`sm_80` marker anywhere, while the bundle's own README and [[source-readme]] advertise `sm_61;sm_70;sm_75;sm_80;sm_86`. The cubin `e_flags` decode is `[UNVERIFIED]` (no `cuobjdump` on this machine), so the conclusion is "unconfirmed for Volta", not "proven broken". See `v100-sxm2.md`, `codebase-map.md`.

## [2026-09-28] contradict | the working tree is missing the CUDA template instances
`src/ggml/src/ggml-cuda/template-instances/` contains **0** files in this checkout while `CMakeLists.txt:106-116` globs them and `llama-fast-src.zip` carries **138** `.cu` files under that path. A CUDA build from this working tree is expected to fail to instantiate the kernel families; the failure itself is `[INFERENCE]`, not executed. See `codebase-map.md`.

## [2026-09-28] contradict | the design doc disagrees with its own paper citation
`src/docs/TRIATTENTION.md` dates arXiv:2604.04921 to **2025** and names Ben-Nun/Zanotti/Alistarh/Liu; `README.md` cites the same id as **April 2026** and "Mao et al.". The arXiv id's `2604` prefix agrees with the README. Both logged on `source-triattention.md`; the wiki does not pick a winner.

## [2026-09-28] contradict | the design doc's CLI defaults are not the code's defaults
`TRIATTENTION.md` documents budget 2048 / window 128 / offset-max 65536 / seed 0 / normalize off; `src/common/common.h:752-766` has 0 / 0 / 0 / -1 / true, and the README matches the code. The doc is the outlier — plausible evidence that it describes an earlier revision.

## [2026-09-28] question | two unexecuted defect hypotheses in the calibration path
Recorded as `[INFERENCE]` from code reading, not as findings, because nothing was built or run: (1) `offset_max` defaults to 0 (`common.h:755`), which yields `n_offsets = 0`, which makes the mean aggregate compute `0 × (1/0)` — NaN scores; (2) the `--triattention-calibrate*` fields may be parsed but never read, which would make the README's calibration example fail. Both are on `triattention-calibrate.md` as the page's main open questions.

## [2026-09-28] note | two recorded issues are weaker than the inventory says
`TQ-4` survival: three WHT implementations exist (`turbo-quant.cuh`, `set-rows.cu`, `turbo-wht.cu`), and the sequential/butterfly pair agrees by construction — same stage order, same constants, same per-element ops — while `turbo_rotate_forward{,_64}` have no callers at all. The recorded mismatch is a latent hazard, not a defect. `TQ-5`: the tail path is unreachable in this checkout. Neither is dropped — both remain in the inventory, re-scoped.

## [2026-09-28] note | "max 5 draft tokens" does not exist in the tree
`state.md` records max 5 draft tokens. The code default is **3** (`common.h:326`), the DFlash clamp is `block_size - 1 = 7` for this drafter, the in-tree doc example is 15, and `--draft-max` was removed. The value is a prose-era configuration, not a repo fact. See `speculative-decoding.md`.

## [2026-09-28] note | InnerQ state is duplicated, not fixed
`turbo-quant.cuh` still declares file-scope `static int innerq_enabled` / `innerq_target_tokens` / `innerq_strength` / `innerq_initialized` and `static __device__` buffers (lines 147-157) with static helper definitions at 160/189/256/291 — while a newer parallel module `turbo-innerq.{cu,cuh}` holds different host state. `TQ-2`/`TQ-3` therefore stand, with the added complication that the tree now has two competing state homes. See `innerq.md`, `tq-2-innerq-host-state.md`, `tq-3-innerq-multigpu.md`.

## [2026-09-28] lint | index built, vault at 40 pages
`index.md` written as the catalog; `lint.sh` is the mechanical pass. Remaining defects at this point are the three hub entity pages (`triattention`, `turboquant`, `innerq`) still being written, which every other page links to.

