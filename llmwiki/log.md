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

## [2026-09-28] ingest | wave 2 — six gap pages from already-read material
Far cheaper than wave 1 (3–9 minutes per agent instead of 20–46): tight file lists, a tool-call ceiling, and "write first, then stop". Added `[[request-lifecycle]]`, `[[sampling]]`, `[[hybrid-memory]]`, `[[gated-delta-net]]`, `[[build-and-verify]]`, `[[conversion-and-packing]]`. Vault at 49 pages, lint clean.

## [2026-09-28] find | the GDN recurrence has its own CUDA kernel and no page mentioned it
`src/ggml/src/ggml-cuda/gated_delta_net.cu` — a dedicated GPU kernel for the 48 recurrent blocks — appeared in no vault page until `[[gated-delta-net]]` was written. The same pass found `src/src/delta-net-base.cpp` and the Metal/OpenCL counterparts. The recurrence is a first-class compute path, not a side effect of the attention code.

## [2026-09-28] note | recurrent state costs real memory, and one hazard is untracked
Per-sequence state is ~3.117 MiB per SSM layer, ~149.6 MiB for all 48 layers, ~600 MiB at 4 sequences (arithmetic from read shapes, not measured). Source comments at `src/src/llama-graph.cpp:3733-3745` describe a multi-sequence GDN state hazard that **has no issue page** — it is not in either inventory. Recorded on `[[gated-delta-net]]`, `[[hybrid-memory]]`; noted here so it is not mistaken for a documented defect.

## [2026-09-28] note | the "16 of 64 layers" rule is enforced in code, not just metadata
`hparams.has_kv`, resolved through `n_layer_kv_from_start` (`src/src/llama-hparams.cpp:274`), is the mechanism behind the layer split that `[[qwen35-architecture]]` derived from the tensor list. Metadata and code agree — which also means TriAttention's budget is an aggregate over a cache only a quarter of the layers populate (`[[triattention]]`, `[[ta-2-budget-starvation]]`).

## [2026-09-28] gap | no test covers a TurboQuant KV type
`src/tests/test-backend-ops.cpp` contains **zero** case-insensitive matches for `turbo`, as does `src/tests/CMakeLists.txt`. The registered custom tests (`test-pq2-row-shapes`, `test-ptq1_0-cuda-dot`, `test-ptq1_0-element-map`, `src/tests/CMakeLists.txt:339-341`) cover the *weight* types only. Combined with an empty `template-instances/` (see above), the TurboQuant path has neither a build nor a test behind it in this tree. See `[[build-and-verify]]`.

## [2026-09-28] contradict | two GGUF file-type fields the code cannot have written
The draft's `general.file_type = 15` maps to `GGML_FTYPE_MOSTLY_IQ2_XXS` (`ggml.h:471`) while its tensors are `Q4_K`/`Q6_K`, and the target's `general.file_type = 141` is defined by neither in-tree table (the C enum and `constants.py` both stop at 129). No in-tree converter can correct or produce either label. Recorded on `[[conversion-and-packing]]`; the loader ignores the field, so nothing depends on it — which is exactly why it sat unexplained.

## [2026-09-28] note | the Hadamard contract has a string that never appears in the file
The out-of-tree packer's manifest declares `normalized-signed-sylvester-walsh-hadamard`, while the GGUF key the same code path writes — and the loader checks — drops `signed`: `normalized-sylvester-walsh-hadamard`. Cosmetic, but it means no manifest-side string is ever the string in the artifact. See `[[conversion-and-packing]]`, `[[prism-hadamard-weight-fold]]`.

## [2026-09-28] gap | three competing entry points for the draft model's conversion
`src/conversion/dspark.py` emits arch `DSPARK`, `src/conversion/qwen.py` subclasses into arch `DFLASH`, and `src/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py` rewrites legacy `DSPARK` GGUFs into `DFLASH`. Three paths, one drafter lineage, no page previously distinguishing them. See `[[conversion-and-packing]]`, `[[qwen3-dflash-draft]]`.

## [2026-09-28] gap | the token-selection layer had zero coverage
Before `[[sampling]]`, no page in the vault mentioned `sampler` at all — the whole selection stage, including the parameters a server request can reach, was undocumented. The recorded effective-throughput trend (31.6 → 38.1+ t/s) still has no reproducible artifact: the acceptance statistics live at `src/tools/server/server-context.cpp:615-636` and no run output is checked in.

## [2026-09-28] correction | the recurrent pre-fill was stated backwards on two pages
`[[qwen35-architecture]]` and `[[hybrid-memory]]` claimed `llm_arch_is_recurrent(QWEN35) = true` filled `is_recr_impl` with 1 for every layer. The switch (`src/src/llama-arch.cpp:1068-1080`) names only MAMBA, MAMBA2, RWKV6, RWKV6QWEN2, RWKV7 and ARWKV7, and returns false by default, so `src/src/llama-model.cpp:1422` pre-fills **0**; the 48-of-64 schedule comes entirely from the interval loop, which assigns every entry. Behaviour was never in doubt — only the explanation was wrong. Found by `[[qwen35-variants]]` during wave 3, confirmed independently against the switch, and corrected in place on both pages with a dated note. (`llm_arch_is_hybrid(QWEN35) = true` was checked at the same time and is correct: `llama-arch.cpp:1096`.)

## [2026-09-28] ingest | wave 3 — eight pages, the last known gaps
Four new entities (`[[quantized-kernel-units]]`, `[[turbo-wht]]`, `[[rotation-data]]`, `[[qwen35-variants]]`), two topics (`[[release-artifacts]]`, `[[backend-parity]]`), and two source ingests (`[[source-hadamard-tied-output]]`, `[[source-kv-mean-center]]` with the matching `raw/` snapshots). Per-agent time: 1–6 minutes. Vault at 57 pages, lint clean, 8 recorded raw hashes verified against 8 files on disk.

## [2026-09-28] find | no quantized-kernel unit knows a turbo type
A case-insensitive sweep for `turbo` across `src/ggml/src/ggml-cuda/` returns **zero** hits in `mmf.cu`, `mmvf.cu`, `mmvq.cu`, `mmq.cu`, `mmq.cuh`, `vecdotq.cuh` and both `mmq-config-*.cuh`. The only CUDA turbo dot products in existence are the fused-attention ones (`vec_dot_fattn_vec_KQ_turbo{3,2,4}_0`). This is the mechanical basis of `[[tq-1-missing-gemm-kernels]]`, now stated per unit rather than globally, in `[[quantized-kernel-units]]`.

## [2026-09-28] find | `k_turbo_wht_copy_tail` is unreachable in the op as built
`ggml_turbo_wht` asserts `ne[0] % group_size == 0` (`src/ggml/src/ggml.c:6649`) and the CUDA support predicate requires `ne[0] % 64 == 0` (`ggml-cuda.cu:5493-5495`), so the tail kernel at `turbo-wht.cu:100` can never run — `tail_size` is always 0. A verified consequence of two reads, not a comment in the code. Recorded on `[[turbo-wht]]`.

## [2026-09-28] find | the WHT op params were attributed to the wrong node
`wht_group` is written onto the **`ggml_set_rows` result's** op-params (`src/src/llama-kv-cache.cpp:1586-1587, 1637-1638, 1663-1664`), not onto a `TURBO_WHT` node; the latter carries `op_params[0] = direction` and `op_params[4..7] = group_size` (`ggml.c:6657-6659`). Earlier pages implied a single op-param contract; `[[turbo-wht]]` and `[[walsh-hadamard-transform]]` now state the split. Also confirmed independently: `turbo_rotate_forward{,_64}` have no callers anywhere in `src/`.

## [2026-09-28] find | the 590 KB rotation table has no recorded generator
`src/src/turbo-rotation-data.h` is a generated 590 KB header of ±1/√n rotation constants, with no generator, script, seed or provenance note anywhere in the tree; `turbo-rotation-data-32.h` (36 KB) is **included by nothing** in this checkout. A large generated artifact with no reproduction path is a maintenance fact the sources never mention. See `[[rotation-data]]`.

## [2026-09-28] note | a fourth rotation implementation, and a hint-driven side path
`src/ggml/src/ggml-cuda/fwht.cu` exists alongside `turbo-wht.cu`, and neither has a graph op: there is **no** `GGML_OP_FWHT` in this tree, and both `fwht.cu` entry points are reached only through `GGML_HINT_SRC0_IS_HADAMARD` (`ggml-cuda.cu:1823`, `:3502`). Whether that file is upstream or a fork addition is `[UNVERIFIED]` without history. See `[[turbo-wht]]`, `[[backend-parity]]`.

## [2026-09-28] find | `--kv-mean-center` is live in this fork, and nothing documented it
The feature is wired end to end: `src/common/kv-mean-center.{h,cpp}`, the `k_cache_in` tag at `src/src/llama-graph.cpp:2958-2962`, a loader basis check (`src/src/llama-kv-cache.cpp:1702-1730`), a gate test and an F32-invariance test (`src/tests/test-kv-mean-center.cpp`), and a tool README. Neither source document mentions it. Ingested as `[[source-kv-mean-center]]`.

## [2026-09-28] verify | wave 4 — four open questions resolved to verdicts
Four pages, all verdicts rather than summaries: `[[scoring-correctness]]` (three hypotheses decided), `[[kv-accounting]]`, `[[server-layer]]`, and a new section on `[[build-and-verify]]`. Plus the capstone `[[open-questions]]`, which classifies all 84 unresolved questions by what it would take to close each one. Vault at 63 pages.

## [2026-09-28] defect | TA-8 CONFIRMED — `offset_max` defaults to 0, so every score is NaN
The H1 hypothesis is not a hypothesis. `offset_max = 0` (`src/common/common.h:755`) → `triattention_build_offsets` returns 0 (`llama-triattention.cpp:333-339`) → `n_offsets = 0` (`:683-684`) → the default `mean` aggregate multiplies by `1.0f/0.0f` on CPU (`0 × inf` = NaN) and divides `0.0f/0.0f` on GPU (`triattention-score.cu:291-296`). **Neither launch script passes `--triattention-offset-max` or `--triattention-agg`**, and the GPU path is attempted first — so this fires in the documented configuration. NaN then reaches `scores[a] > scores[b]` (`:867-870`), breaking `std::partial_sort`'s strict weak ordering. New issue page `[[ta-8-offset-max-zero-nan]]`, severity CRITICAL. Consequence for planning: this is the **third** independent mechanism corrupting *which* keys survive, alongside [[ta-1-wht-inversion-256]] and [[ta-2-budget-starvation]] — every previous measurement of eviction quality is confounded by all three.

## [2026-09-28] defect | TA-9 CONFIRMED — the scorer inverts RoPE over the wrong dimensions
The H2 hypothesis is a real defect. The model rotates `n_rot = 64` of 256 dimensions with exponent θ^(−2f/**64**) (`hparams.n_rot()` → `llama-model.cpp:1486`, applied by `ggml_rope_multi` at `models/qwen35.cpp:368-377`); the scorer builds omega with exponent θ^(−2f/**head_dim**) and inverts every pair `(f, f+128)` across all 256 dimensions with no `n_rot` or sections parameter (`llama-triattention.cpp:309-312`, `:382`, mirrored in `triattention-score.cu`). The angle error is θ^(3f/128) — it **grows with frequency**, the exact quantity the scoring formula weights. Not compensated anywhere: `turbo-rotation-data.h` contains no omega or head-dim symbol, and calibration does not cancel the mismatch because `q` is captured post-RoPE and left untouched. The "it only needs *a* basis" defence is refuted — the same map is required on both operands. New issue page `[[ta-9-rope-scope-mismatch]]`, severity HIGH.

## [2026-09-28] verify | H3 resolved — `freq_scale_sq` is dead code, and the paper's scaling is unimplemented
Not a bug and not a hidden branch: the factor is `cos²(ω·0) + sin²(ω·0)` = exactly 1.0 for finite ω (`llama-triattention.cpp:320-327`), an identity multiply on both paths, with the function's own comment marking it a placeholder for future YaRN support. The documented `1/ω²` frequency weighting (`TRIATTENTION.md:41`) exists **nowhere** in `src/` — so the paper's frequency scaling is simply not implemented, which is a fidelity gap rather than a defect. [[ta-5-freq-scale-dead-code]] re-scoped accordingly.

## [2026-09-28] environment | this machine cannot build or measure the project's target
`which nvcc` empty; no `/opt/cuda`, no `/usr/local/cuda*`, no pacman cuda package; `cuobjdump`/`nvdisasm`/`ptxas` absent. `nvidia-smi` reports an **NVIDIA GeForce GTX 1660, cc 7.5** — `sm_75`, not the V100's `sm_70`. The tree's only configure record, `src/build-x64-linux-gcc-debug/CMakeCache.txt` (2026-09-28 17:45), is Ninja + gcc + Debug from `src/` with **`GGML_CUDA=OFF`** and no `CMAKE_CUDA_COMPILER` or arch list, and produced no binary. Recorded on `[[build-and-verify]]`; this single fact makes roughly a third of the vault's open questions unreachable until the environment changes (`[[open-questions]]`).

## [2026-09-28] verify | the missing-file count, reconciled exactly
The two counts the vault carried were both right about different things: `llama-fast-src.zip` holds **138** `.cu` members under `ggml-cuda/template-instances/` (plus `generate_cu_files.py`), and the archive-minus-tree diff over the whole `ggml-cuda/` subtree is exactly **142** files — the 138, plus that script, plus `vendors/{cuda,hip,musa}.h`. The top-level `ggml-cuda/*.cu` set is complete (71 in tree, 71 in archive), so nothing else in that directory is missing. Verified restore command: `unzip -n llama-fast-src.zip 'src/ggml/src/ggml-cuda/template-instances/*' -d .` — followed by a CMake re-run, since both the globs and the named list are configure-time. Nothing was extracted; the tree stayed clean.

## [2026-09-28] refute | wave 5 — TQ-1's premise does not survive the placement path
`[[device-placement]]` follows the node from graph construction to backend assignment and reaches verdict **(c): there is no turbo `MUL_MAT` at all.** Turbo cache types *force* flash attention on (`src/src/llama-context.cpp:3882-3887` → `cparams.flash_attn` at `:320`), so `build_attn_mha` always takes the fused branch (`src/src/llama-graph.cpp:2673`, `:2699`), and the quantised blocks are dequantised **inside** `ggml_cuda_flash_attn_ext` (`fattn-vec.cuh:87-98`, `fattn.cu:338-369`), which CUDA accepts via `ggml_cuda_flash_attn_ext_supported` (`fattn.cu:647-649`, type list `:382-402`). `ggml_cuda_mul_mat` therefore **never sees a turbo tensor in decode**. The types' absence from the allow-list (`ggml-cuda.cu:5244-5270`) is real but **dead on the hot path**; if a turbo `MUL_MAT` were ever built, `ggml_backend_sched_split_graph` would push it to the **CPU** backend with D2H copies (`ggml-backend.cpp:1399-1419`) — the [[ta-3-cpu-fallback-transfers]] mechanism, never an abort.

Consequences, applied immediately: `[[tq-1-missing-gemm-kernels]]` re-scoped (the gap is real, the impact unproven) and `[[performance-profile]]`'s causal chain for the 38.81 % withdrawn — the figure now has **no verified attribution** in this vault. `[[overview]]` carries a dated correction over its "single bottleneck" paragraph, and `[[roadmap]]` item 2 is re-baselined: it must not start before [[ta-8-offset-max-zero-nan]] is fixed and a trustworthy profile exists.

## [2026-09-28] defect | TA-10 CONFIRMED — `prefix_length` is a per-context latch, and the server never resets it
One `triattention_state` per KV cache (`llama-kv-cache.h:344`); set once by the first batch containing position 0, keyed on the **batch** rather than the sequence (`llama-kv-cache.cpp:1354-1363`); reset only by `triattention_on_reset`, called from `llama_kv_cache::clear` alone (`:528`). The server never full-clears — slot recycling is `mem.seq_rm(id,-1,-1)` (`server-context.cpp:292`) — so every request after the first inherits the first request's protection boundary. A later **longer** prompt then fails `is_prefix` (`llama-triattention.cpp:1136-1137`) for its own middle and can be evicted mid-request. `-np 1` does not help: it sets the slot count (`arg.cpp:2557-2566`) and one slot still recycles the context. New issue page `[[ta-10-prefix-length-global-latch]]`, severity HIGH. The settling experiment is one run reading the `[prefix=%lld]` prune counter.

## [2026-09-28] defect | TA-11 CONFIRMED — the shipped calibration profile is in the wrong basis
Ranked #1 in `[[open-questions]]`, and the answer is a defect rather than a doubt. The collector matches the exact name `"Qcur-" + <layer>` (`triattention-calibrate.cpp:42-52`); in the `qwen35` graph exactly one tensor carries that name, `cb(Qcur, "Qcur", il)` at `models/qwen35.cpp:379`, and that is the **return of `ggml_rope_multi`** (`:367-371`) — the pre-RoPE tensors are `Qcur_full`/`Qcur_reshaped`/`Qcur_normed` (`:335`, `:340`, `:344`), none of which matches. The scorer, by design, scores **pre-RoPE** keys (`triattention_invert_rope`, `llama-triattention.cpp:382`, used at `:444-445`). The mismatch corrupts the phase term position-dependently and understates the norm (only `E[‖q_f‖]` is basis-invariant). **Independent of [[ta-9-rope-scope-mismatch]]** — one is the geometry of the inverse map, the other the basis the query was measured in — and **masked by [[ta-8-offset-max-zero-nan]]**, since every score is NaN regardless until that is fixed. New issue page `[[ta-11-calibration-post-rope-basis]]`, severity CRITICAL.

## [2026-09-28] ingest | `speculative.md`, and the CPU surface
`[[source-speculative]]` ingests the in-tree speculative-decoding doc (the tuning table, which flags still exist versus which were removed, and where the fork's `dflash` variant diverges). The "max 5" figure is sharpened but still unexplained: the doc's default for a block-8 drafter is 7 and its example is 15, so 5 is a tuning choice recorded nowhere. `[[cpu-path]]` maps the CPU implementation of the fork's ops — the readable statement of each format's intended semantics, and the mechanism behind [[ta-3-cpu-fallback-transfers]].

## [2026-09-28] note | the backlog now separates decisions from investigations
`[[decisions-pending]]` states the six choices the project owes — the repair order for the confirmed defects, the canonical WHT, whether the KV path gets test coverage, whether to restore `template-instances/`, what happens to the orphaned rotation header and dead `PTQ1_0`, and whether the README's Ampere benchmark table gets annotated — each with its options, their cost, and a recommendation. Combined with `[[open-questions]]`, the vault now says not only what is unknown but **which unknowns are waiting on a person rather than on evidence**. Vault at 68 pages.

## [2026-09-28] ingest | wave 6 — the token path, in both directions
The vault documented control flow but never data flow; the user asked for the path from vocabulary through embedding to decoded text, and it is now two pages, one per subsystem.

**`[[forward-pass]]` — ids to logits, tensor by tensor.** Input ids → the `token_embd.weight` lookup → the residual stream through the 64-block stack, with the hybrid routing deciding attention versus recurrence per layer → inside an attention block, QKV projection, the Q rotation, partial MRoPE, the KV write with its turbo quantisation and the fused `ggml_flash_attn_ext` read → inside a recurrent block, the GDN nodes → final norm → output projection → logits. The custom stacks are placed in the chain in one table (TriAttention prunes *before* the graph is built; TurboQuant acts on the KV write and read; the CUDA graph captures the per-token graph whole). Two facts it establishes that no page had: the graph applies rotation only to **Q on the way in and the output on the way out** — the K/V rotate-and-quantise must therefore happen inside the CUDA `set_rows` kernel, which makes that kernel the place to look for the KV-side WHT contract; and the output head question is settled by the model's tensor list (untied; the fold applies to `token_embd`, the one name the checkpoint's `inverse_weight_names` exempts).

**`[[tokenizer]]` — text to ids and back.** The vocabulary (248 320 entries, GPT-2-style BPE with the `qwen35` pre-tokenizer), the merge-rank lookup, special-token handling and the BOS flag that controls it; then the return path — detokenisation, the partial-UTF-8 buffering a byte-level BPE token makes necessary, and **where a streamed piece becomes the response body** (`server-context.cpp:1768`). It carries a latent-bug flag worth noting: the defensive `[UNK_BYTE_0x…]` fallback (`llama-vocab.cpp:3360-3364`) appends the *whole* piece inside the marker rather than the one bad byte — unproven and untriggered, but exactly the class of multibyte defect the page exists to make findable. Neither source document mentions any of this: the vocabulary is documented nowhere in the vault until now.

## [2026-09-29] measure | the engine runs — first live measurements, and the vault's central verdict is confirmed
The prebuilt **CUDA 13** bundle from `build/cuda13.zip` executes on this machine's **GTX 1660** (`sm_75`), because it ships its own CUDA 13 runtime and includes `sm_75` in its arch list. The cached `Ternary-Bonsai-4B-Q2_0_g64.gguf` was served end to end (`ftype: Q2_0`, build `b10747-1773b4b1a`), and the run generated coherent text. This is the **first execution of this fork recorded anywhere in the vault's history**, and it was possible despite the missing CUDA toolkit that blocked every earlier plan. Full numbers on [[first-live-measurements]]; the corrections they force are applied to [[benchmarks]], [[build-and-verify]] and the index's contradiction list.

**The decisive result:** with `-ctk turbo3 -ctv q8_0`, the only kernel that touches the KV cache is `flash_attn_ext_vec<(int)128,(int)1,(ggml_type)43,(ggml_type)8,(bool)0>` — `43` = `GGML_TYPE_TURBO3_0`, `8` = `GGML_TYPE_Q8_0` (`src/ggml/include/ggml.h:433`, `:402`). **No turbo `MUL_MAT` exists in the trace, and no cuBLAS or MAGMA kernel appears at all.** [[device-placement]]'s verdict and [[tq-1-missing-gemm-kernels]]'s refutation are now empirical, not read-off-the-code.

**Confirmed by measurement, not by reading:** `Q2_0` weights are served by the MMQ tile path — `mul_mat_q<(ggml_type)42,…>` (`42` = `GGML_TYPE_Q2_0`) is **75 % of all GPU kernel time** ([[quantized-kernel-units]]); CUDA graphs are live (one instantiate, one exec-update, 26 launches, [[cuda-graphs]]); and no WHT kernel is hot ([[turbo-wht]]).

**The trade is now measured.** TurboQuant KV costs ≈11 % of generation throughput and buys 11–15 % of VRAM at ctx 2048: F16 85.6 t/s / 1528 MiB → `turbo3+q8_0` 76.0 t/s / 1348 MiB → `turbo3+turbo2` 74.6 t/s / 1296 MiB (two runs each, `--ignore-eos`). At this context the cache is not bandwidth-bound, so the rotation overhead dominates — which is exactly the regime the vault's `benchmarks` page could never test. Host-side: 1503 MiB peak RSS *identical across all three configurations*, 0 blocks of IO (page cache), and a 1.99 s wall of which **94 % is CPU**; `perf` reports IPC 2.17 with a 56.5 % cache-miss rate, the signature of a dequantize-and-multiply loop.

**Two operational traps worth recording**, both of which had already cost the vault time: `build/cuda13.zip` is **LZMA**-compressed, so `unzip` extracts nothing from it *and reports success* (use `7z x` or `bsdtar`); and `nsys` installs without root only by extracting the archive (`7z x <run>` → tar → `pkg/target-linux-x64/nsys`), because the installer's own prompt ignores `--target`.

**Not claimed:** none of these numbers predict V100 behaviour (`sm_75` has INT8 tensor cores, `sm_70` does not), the model is the 4B at group size 64 rather than the 27B `PQ2_0`, context is 2048 rather than 16K–32K, TriAttention was never exercised for lack of a 4B profile, and nsys runs are not comparable to plain runs. Vault at 72 pages.







