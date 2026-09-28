# llmwiki — index

Catalog of the vault. Read this first, then drill into pages. Conventions and procedures: `SCHEMA.md`.
Status: `current` unless the page says otherwise. Every page cites its sources; `verified:` in the frontmatter lists the paths whose claims were checked against the code.

Maintained by the LLM. Last full pass: 2026-09-28.

---

## Start here

| Page | What it answers |
| :--- | :--- |
| [[overview]] | What this project is, where it is stuck, and what the evidence actually supports |
| [[roadmap]] | What is planned, what is unplanned but visible, and the sequencing conflicts |
| [[codebase-map]] | Where anything lives in the tree, and whether it builds |
| [[performance-profile]] | Where the GPU time goes, and which attributions survive contact with the code |
| [[request-lifecycle]] | How one request actually flows: batch → graph → kernels → sampling |
| [[open-questions]] | Every unresolved question in the vault, ranked, with what each would take |

## Sources (`raw/` snapshots)

| Page | Summary | Raw |
| :--- | :--- | :--- |
| [[source-readme]] | The release README: four optimizations, upstream lineage, GPU matrix, CLI surface, RTX 3090 measurements | `raw/README.md` |
| [[source-state-md]] | The user's working document: profiling conclusions, four checkouts, and the TA/TQ defect inventories | `raw/state.md` |
| [[source-triattention]] | The project's own TriAttention design doc: calibration, trigonometric scoring, eviction | `raw/TRIATTENTION.md` |
| [[source-triattention-api]] | The three-layer TriAttention API contract as documented | `raw/TRIATTENTION-API.md` |
| [[source-agents-md]] | The repository constitution: edit boundaries, pipeline, wiki schema | `raw/AGENTS.md` |
| [[source-llmwiki]] | The pattern this vault implements | `raw/llmwiki.txt` |
| [[source-hadamard-tied-output]] | The project doc on the tied-output Hadamard fold | `raw/hadamard-tied-output.md` |
| [[source-kv-mean-center]] | The KV mean-centering feature — live in this fork, with a test and a tool | `raw/kv-mean-center.md` |
| [[source-speculative]] | The in-tree speculative-decoding design doc, and how far it has drifted from the code | `raw/speculative.md` |

## Entities (things)

| Page | Summary |
| :--- | :--- |
| [[ternary-bonsai-2-27b]] | The target model, read from its GGUF header: `PQ2_0`, `head_dim=256`, hybrid attention/SSM |
| [[qwen35-architecture]] | The architecture the sources never name: 64 blocks, only 16 with a KV cache |
| [[qwen35-variants]] | The neighbouring models that share the family, and how they differ |
| [[hybrid-memory]] | How the engine keeps KV *and* recurrent state in one model |
| [[gated-delta-net]] | The recurrence inside the 48 SSM blocks, and its per-sequence cost |
| [[v100-sxm2]] | The deployment target and the capability gates that shape every design decision |
| [[triattention]] | Calibration-guided KV eviction: the mechanism and where it runs |
| [[turboquant]] | The KV-cache vector quantization scheme: types, geometry, rotation, fused dots |
| [[innerq]] | InnerQ equalization — and the two competing state homes it lives in |
| [[walsh-hadamard-transform]] | The rotation before quantization, and the three implementations of it |
| [[turbo-wht]] | The op that actually runs on the KV path, and what `fwht.cu` is for |
| [[rotation-data]] | The 590 KB of embedded rotation constants — and the generator nobody recorded |
| [[prism-hadamard-weight-fold]] | The *model-side* Hadamard fold baked into the checkpoint — not the KV rotation |
| [[prismml-weight-kernels]] | `PQ2_0`/`PTQ1_0` weight kernels and their arch reach (out of scope for edits) |
| [[quantized-kernel-units]] | The mmf/mmvf/mmvq/mmq units and their type tables — none of which knows a turbo type |
| [[cpu-path]] | The CPU implementation of the fork's ops, and whether it can serve as the reference semantics |
| [[cuda-graphs]] | `GGML_CUDA_GRAPH_OPT=1`: what graph reuse and concurrent streams actually buy |
| [[speculative-decoding]] | The `draft-dflash` block-proposal path |
| [[sampling]] | The token-selection chain and the parameters a request can reach |
| [[qwen3-dflash-draft]] | The draft model itself |

## Concepts (ideas)

| Page | Summary |
| :--- | :--- |
| [[kv-cache]] | Why its cost grows with context, and the three things this project does about it |
| [[kv-eviction]] | Bounding KV by a fixed cell count, and the failure mode of a fixed budget |
| [[quantization]] | Low-bit KV representation, block scales, and why a missing dot kernel is expensive here |
| [[gemm-dispatch]] | How ggml picks a matmul kernel, and what happens to a type nobody listed |

## Issues

TriAttention's inventory, from [[source-state-md]] §3 — verified against the code:

| Page | Severity | One line |
| :--- | :--- | :--- |
| [[ta-1-wht-inversion-256]] | CRITICAL | The scoring kernel skips WHT inversion at `head_dim=256`; triggered by the target model |
| [[ta-2-budget-starvation]] | HIGH | Long prefixes starve the budget to zero and eviction collapses to a sliding window |
| [[ta-3-cpu-fallback-transfers]] | HIGH | Per-cell synchronous D2H copies stall the CPU path for 15–30 s |
| [[ta-4-cooperative-fwht-race]] | MEDIUM | Shared-memory WHT assumes 64 active threads; UB on Volta |
| [[ta-5-freq-scale-dead-code]] | MEDIUM | `freq_scale_sq` computes to 1.0 always; the scaling is disabled |
| [[ta-6-overlap-double-counting]] | LOW | Latent risk for future patches, not a live defect |
| [[ta-7-config-validation]] | LOW | No guard rails on incompatible `budget`/prefix/window combinations |
| [[ta-8-offset-max-zero-nan]] | **CRITICAL (new)** | `offset_max` defaults to 0, so every eviction score is NaN in the shipped scripts |
| [[ta-9-rope-scope-mismatch]] | **HIGH (new)** | The scorer inverts RoPE over 256 dims with the wrong exponent; the model rotates 64 |
| [[ta-10-prefix-length-global-latch]] | **HIGH (new)** | `prefix_length` is latched per context; the server never resets it, so a longer prompt loses its own middle |
| [[ta-11-calibration-post-rope-basis]] | **CRITICAL (new)** | The shipped profile records post-RoPE queries; the scorer compares pre-RoPE keys against them |

TurboQuant's inventory, from [[source-state-md]] §4 — several premises corrected against the code:

| Page | Severity | One line |
| :--- | :--- | :--- |
| [[tq-1-missing-gemm-kernels]] | CRITICAL | Turbo KV types are absent from the dispatch allow-lists; the recorded `magma` symbol does not exist in the tree |
| [[tq-2-innerq-host-state]] | HIGH | Host state is file-scope `static` in a header — still present, and now duplicated by a newer module |
| [[tq-3-innerq-multigpu]] | HIGH | `static __device__` state receives `cudaMemcpyToSymbol` on one device only |
| [[tq-4-wht-numerical-mismatch]] | HIGH | Three WHT copies; the mismatch is a latent hazard, not a demonstrated defect |
| [[tq-5-tail-elements]] | MEDIUM | Tail path unreachable in this checkout |
| [[tq-6-innerq-race]] | MEDIUM | Calibration counter keyed on thread mapping |
| [[tq-7-innerq-max-channels]] | MEDIUM | `INNERQ_MAX_CHANNELS = 128` against a 256-channel head |

## Topics (synthesis)

| Page | Summary |
| :--- | :--- |
| [[overview]] | The project's situation in one screen |
| [[performance-profile]] | The recorded profiling conclusions and their confounds |
| [[benchmarks]] | The two measurement suites, neither run on the deployment hardware |
| [[codebase-map]] | Tree layout, custom-code locations, build system, and the missing template instances |
| [[build-and-verify]] | How the tree is built, what it ships, and how a kernel change would be checked |
| [[release-artifacts]] | What each shipped artifact is, and which one is usable on the target |
| [[conversion-and-packing]] | How the `PQ2_0` artifact and the Hadamard fold were produced, and where that trail leaves the repo |
| [[backend-parity]] | What the Vulkan/SYCL/Metal copies of this fork's ops reveal — and the turbo-KV gaps |
| [[scoring-correctness]] | Three verdicts on how TriAttention's scores are actually computed |
| [[kv-accounting]] | Whether the published tokens-per-GB figures describe this model's real KV structure |
| [[server-layer]] | What `llama-server` adds: slots, endpoints, and the statistics behind the numbers |
| [[open-questions]] | The whole unresolved backlog, classified by what it would take to close |
| [[device-placement]] | Whether a TurboQuant node ever runs on the GPU — and the answer refutes TQ-1's premise |
| [[decisions-pending]] | The choices the project owes, with each option's cost and a recommendation |
| [[triattention-calibrate]] | The offline calibration tool, its profile format, and where doc and code diverge |
| [[upstream-lineage]] | Three upstreams plus a paper, and what that implies for maintenance |
| [[roadmap]] | The six recorded action items, plus findings the inventory never listed |

## Open contradictions (unresolved, deliberately)

Recorded on the relevant pages and in `log.md`, never silently reconciled:

1. **Benchmarks are Ampere, the target is Volta** — the 1.39× and tok/GB figures describe an RTX 3090. See [[benchmarks]].
2. **The recorded `magma_sgemmEx_kernel` cost cannot be grounded** — no MAGMA path exists in `src/`. See [[gemm-dispatch]], [[tq-1-missing-gemm-kernels]].
3. **The TQ-1 causal story is refuted at its first step** — turbo KV types force flash attention, so no turbo `MUL_MAT` is built during decode and the cuBLAS path is never reached; the 38.81 % has no verified attribution. See [[device-placement]], [[gemm-dispatch]], [[tq-1-missing-gemm-kernels]], [[performance-profile]].
4. **The published CUDA 12.4 bundle carries only `sm_86`** while its own README claims Volta support. See [[v100-sxm2]], [[codebase-map]].
5. **`src/ggml/src/ggml-cuda/template-instances/` is empty** while CMake globs it — 138 files exist only in the archive. See [[codebase-map]].
6. **The TriAttention design doc dates its own paper to 2025 with different authors**, while the README cites the same arXiv id as April 2026. See [[source-triattention]].
7. **The design doc's CLI defaults disagree with the code and the README**, which agree with each other. See [[triattention-calibrate]].
8. **`--triattention-calibrate*` may be inert** and `offset_max=0` may yield NaN scores — `[INFERENCE]` from code, not executed. See [[triattention-calibrate]].
9. **"max 5 draft tokens" exists nowhere in the tree** — the code default is 3 and the flag was removed. See [[speculative-decoding]].
10. **`block_size=32` in the sources contradicts `QK_TURBO3 = 128` in the code.** See [[turboquant]].
11. **This machine cannot build or measure anything the project targets** — no CUDA toolkit, and the attached GPU is a GTX 1660 (`sm_75`), not the V100 (`sm_70`). The tree's only configure record is CPU-only. See [[build-and-verify]], [[open-questions]].
12. **Two confirmed defects were never in any inventory** — NaN scoring ([[ta-8-offset-max-zero-nan]]) and the RoPE scope mismatch ([[ta-9-rope-scope-mismatch]]). Both fire in the shipped configuration.
