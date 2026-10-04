---
title: "Wiki Catalog & Master Index"
type: "meta"
updated: 2026-10-04
status: "active"
---

# Knowledge Base Master Index

Welcome to the **LLM Wiki**. This file is the central content catalog maintained on every ingest, query, and refactor operation. LLM agents query this index first to discover relevant knowledge nodes before drilling down.

---

## 💡 Concepts
Foundational theories, computational mechanisms, and cognitive paradigms.

- [[compounding-knowledge]]: The principle of accumulating synthesized markdown structures rather than re-deriving facts on every query.
- [[associative-trails]]: Non-hierarchical associative connections between related concepts, directly descended from Bush's Memex.
- [[memex-architecture]]: Mechanical and digital personal memory supplements facilitating persistent associative lookup.
- [[rag-limitations]]: Analysis of context fragmentation, lost synthesis, and cosine-similarity failure in traditional RAG.
- [[schema-driven-agent]]: Configuration files (`CLAUDE.md`, `AGENTS.md`) enforcing disciplined wiki gardening over conversational decay.
- [[tri-attention-mechanism]]: Tri-stage attention pruning strategy (sinks, sliding window, anchor tokens) for accelerated generation.
- [[kv-cache-eviction]]: Memory-bandwidth optimization evicting redundant keys and values during long autoregressive generation.
- [[wikilinks]]: Double-bracketed hypertext syntax creating bidirectional graph edges across notes.
- [[prismml-weight-quantization]]: Ternary `PTQ1_0` (1.75 bpw base-3 packing) and sub-2-bit `PQ2_0` (2.125 bpw) kernels.
- [[turboquant-kv-cache]]: Extreme 2/3/4-bit KV cache quantization combining PolarQuant centroids, WHT rotation, and QJL projections.
- [[walsh-hadamard-transform]]: $O(d \log d)$ randomized orthogonal rotation dispersing activation outliers while preserving attention inner products.
- [[triattention-scoring-eviction]]: Mathematical formulation of RoPE key inversion and future trigonometric scoring over geometric horizons.
- [[cuda-graph-concurrency]]: Zero-overhead kernel launch replay and asynchronous stream scheduling via `GGML_CUDA_GRAPH_OPT=1`.
- [[ggml-tensor-execution-graph]]: Directed Acyclic Graph (`ggml_cgraph`) topology, topological sorting, views, and memory planning.

---

## 🏛️ Entities
Key individuals, software tools, frameworks, and inference systems.

- [[vannevar-bush]]: Engineer, inventor, and author of the 1945 Memex treatise *As We May Think*.
- [[obsidian]]: Extensible local-first Markdown knowledge base environment functioning as the IDE for the LLM Wiki.
- [[marp]]: Markdown presentation ecosystem converting wiki notes into slide decks.
- [[dataview]]: Query language plugin for Obsidian running dynamic aggregations over YAML frontmatter.
- [[qmd-search]]: Fast local markdown search CLI and MCP server supporting BM25 and vector re-ranking.
- [[llama-fast]]: High-performance C++ LLM inference engine supporting hardware-aware attention acceleration.
- [[prismml]]: AI research and systems engineering organization behind `PTQ1_0`, `PQ2_0`, and the Bonsai model family.
- [[turboquant]]: Vector quantization algorithm for 2-4 bit KV cache compression (arXiv 2504.19874, ICLR 2026).
- [[bonsai-model]]: High-parameter dense model family operating natively in ternary and sub-2-bit quantization.
- [[ggml-cuda-kernels]]: Low-level CUDA device kernels implementing PTQ1_0 GEMV, cooperative WHT, and TriAttention scoring.

---

## 📚 Ingested Sources
Curated summaries of raw materials and codebase implementations, including key claims and affected pages.

- [[source-bush-1945]]: Summary of *As We May Think* (1945), introducing associative indexing and Memex trails.
- [[source-llm-wiki-pattern]]: Canonical architectural breakdown of the three-layer LLM Wiki pattern and compounding workflows.
- [[source-triattention-paper]]: Research summary on TriAttention tri-partition KV cache compression and memory bandwidth optimization.
- [[source-rag-vs-wiki-notes]]: Empirical field notes on prompt saturation and synthesis degradation in naive RAG pipelines.
- [[source-src-prismml]]: Technical analysis of `/src` PrismML weight codecs, base-3 trit packing, and CUDA GEMV kernels.
- [[source-src-triattention]]: Codebase walkthrough of `/src` TriAttention RoPE inversion, trigonometric series, and GPU scoring.
- [[source-src-turboquant]]: Detailed trace of `/src` TurboQuant WHT rotation, PolarQuant centroids, and `llama-graph.cpp` integration.

---

## 🔬 Syntheses & Theses
Cross-cutting comparative studies, emergent hypotheses, and compounding answers filed back into the wiki.

- [[rag-versus-compounding-wiki]]: Comparative matrix evaluating latency, cost, synthesis depth, and contradiction resolution between RAG and Wiki.
- [[obsidian-ide-workflow]]: Comprehensive guide on structuring the Human-Obsidian-Agent collaborative loop.
- [[memory-bounded-llm-inference]]: Deep synthesis connecting hardware bandwidth ceilings with sparse attention mechanisms like TriAttention.
- [[synergy-prismml-turboquant-triattention]]: The complete trifecta: 1.75 bpw weights + 2-3 bit KV + trigonometric eviction = 40x memory compression.
- [[llama-fast-execution-pipeline]]: Step-by-step tensor lifecycle trace through prefill, autoregressive decode, and TriAttention pruning events.
- [[ggml-inference-architecture-and-hooks]]: Comprehensive technical reference on GGML execution pipeline and the 5 essential runtime hook points.
