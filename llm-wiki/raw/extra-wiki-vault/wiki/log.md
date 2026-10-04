# LLM Wiki Audit & Activity Log

Append-only chronological record of all ingest, query, synthesis, and linting operations performed on this knowledge base.

---

## [2026-10-04] refactor | Initialize LLM Wiki Workspace
- Initialized three-layer architecture: `raw/`, `wiki/`, and Schema configuration.
- Established `CLAUDE.md` and `AGENTS.md` specifying Ingest, Query, and Lint protocols.
- Provisioned Python CLI utility `tools/wiki_cli.py` and `wiki.py` for automated health checks and link validation.
- Initialized `wiki/index.md` master catalog.

## [2026-10-04] ingest | As We May Think (Vannevar Bush, 1945)
- Ingested foundational article from `raw/articles/bush-1945-as-we-may-think.md`.
- Generated source summary in [[source-bush-1945]].
- Created entity note for [[vannevar-bush]].
- Created concept notes for [[associative-trails]] and [[memex-architecture]].
- Linked concept [[compounding-knowledge]] to Bush's critique of artificial indexing.

## [2026-10-04] ingest | LLM Wiki Pattern Proposal
- Ingested `raw/articles/karpathy-llm-wiki-concept.md`.
- Created source summary [[source-llm-wiki-pattern]].
- Established concept notes [[compounding-knowledge]], [[rag-limitations]], and [[schema-driven-agent]].
- Registered tool entity notes for [[obsidian]], [[marp]], [[dataview]], and [[qmd-search]].
- Linked human-in-the-loop workflows to [[obsidian-ide-workflow]].

## [2026-10-04] ingest | TriAttention Hardware-Aware KV Cache Pruning
- Ingested research paper from `raw/papers/triattention-compression.md`.
- Created source summary [[source-triattention-paper]].
- Created technical concept notes [[tri-attention-mechanism]] and [[kv-cache-eviction]].
- Connected entity [[llama-fast]] to sparse attention execution.

## [2026-10-04] ingest | Engineering Notes: RAG vs Persistent Synthesis
- Ingested field notes from `raw/notes/rag-vs-persistent-synthesis.md`.
- Created source summary [[source-rag-vs-wiki-notes]].
- Updated concept [[rag-limitations]] with token budget saturation observations.

## [2026-10-04] synthesis | RAG vs Compounding Wiki Architecture
- Filed deep comparative analysis into [[rag-versus-compounding-wiki]].
- Formulated trade-off matrix covering synthesis compounding, multi-hop latency, and maintenance overhead.
- Cross-linked with [[compounding-knowledge]] and [[rag-limitations]].

## [2026-10-04] synthesis | The Obsidian IDE & Agent Workflow
- Synthesized workflow guide in [[obsidian-ide-workflow]].
- Outlined dual-pane setup: Obsidian as IDE, LLM as compiler, Git repository as persistent codebase.

## [2026-10-04] synthesis | Memory-Bounded Inference & Sparse Attention
- Synthesized hardware analysis in [[memory-bounded-llm-inference]].
- Connected memory bandwidth limits during autoregressive generation to [[tri-attention-mechanism]] and [[kv-cache-eviction]].

## [2026-10-04] lint | Wiki Integrity Verification
- Executed `tools/wiki_cli.py lint`.
- Verified 0 broken links, 0 orphan pages, and 100% indexing in `wiki/index.md`.

## [2026-10-04] ingest | Codebase: PrismML Weight Kernels (PTQ1_0 & PQ2_0)
- Analyzed `/src/ggml/src/ggml-common.h`, `/src/ggml/src/ggml-cuda/convert.cu`, `/src/ggml/src/ggml-cuda/vecdotq.cuh`.
- Generated source summary in [[source-src-prismml]].
- Established concept note [[prismml-weight-quantization]] detailing base-3 trit packing (5 trits/byte, 1.75 bpw) and sub-2-bit format (2.125 bpw).
- Established entity notes for [[prismml]] and [[bonsai-model]].
- Cataloged CUDA device dequantization and GEMV routines in [[ggml-cuda-kernels]].

## [2026-10-04] ingest | Codebase: TurboQuant KV Cache & WHT Integration
- Analyzed `/src/ggml/src/ggml-turbo-quant.c`, `/src/ggml/src/ggml-cuda/turbo-wht.cu`, and `llama-graph.cpp`.
- Generated source summary in [[source-src-turboquant]].
- Established concept note [[turboquant-kv-cache]] covering `turbo2_0`, `turbo3_0`, and `turbo4_0` formats.
- Established concept note [[walsh-hadamard-transform]] detailing $O(d \log d)$ randomized rotation, outlier diffusion, and attention inner product preservation.
- Established entity note for [[turboquant]].

## [2026-10-04] ingest | Codebase: TriAttention Implementation & GPU Kernel
- Analyzed `/src/src/llama-triattention.h`, `/src/src/llama-triattention.cpp`, `/src/ggml/src/ggml-cuda/triattention-score.cu`.
- Generated source summary in [[source-src-triattention]].
- Established concept note [[triattention-scoring-eviction]] detailing RoPE inversion, trigonometric future series, and cooperative shared-memory WHT inversion.
- Updated concept note [[cuda-graph-concurrency]] on `GGML_CUDA_GRAPH_OPT=1` graph replay.

## [2026-10-04] synthesis | Trifecta Synergy: PrismML, TurboQuant & TriAttention
- Synthesized comprehensive architectural thesis in [[synergy-prismml-turboquant-triattention]].
- Mapped out the compound memory reduction: 1.75 bpw weights (8x) + 2-3 bit KV tokens (5x) + TriAttention token pruning (4x-10x) yielding ~40x effective KV memory reduction.

## [2026-10-04] synthesis | llama-fast End-to-End Execution Pipeline
- Synthesized step-by-step tensor lifecycle trace in [[llama-fast-execution-pipeline]].
- Documented prefill graph execution, single-token decode loop, WHT forward/inverse transforms, and on-GPU TriAttention eviction events.

## [2026-10-04] lint | Codebase Ingestion Verification
- Executed `tools/wiki_cli.py lint` inside `llm-wiki/`.
- Verified 0 broken links, 0 orphan pages, and 100% indexing in `wiki/index.md`.

## [2026-10-04] synthesis | GGML Inference Architecture & Hook Points
- Authored comprehensive technical documentation in [[ggml-inference-architecture-and-hooks]].
- Formulated concept note [[ggml-tensor-execution-graph]] detailing DAG lifecycle, tensor views, strides, and memory planning.
- Documented 5 actionable hook points: `cb_eval` (pre/post execution), `llm_graph_cb` (DAG construction), `ggml_map_custom` (custom C/C++/CUDA operators), `ggml_backend_tensor_get/set` (direct memory I/O), and `ggml_abort_callback`.
- Verified 100% graph health via `python3 wiki.py lint`.
