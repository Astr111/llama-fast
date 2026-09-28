---
title: Upstream lineage
type: topic
status: current
updated: 2026-09-28
sources: [README.md, state.md]
verified: []
tags: [provenance, upstream]
---

# Upstream lineage

## Bottom line

`llama-fast` is a **merge of three upstreams plus a paper**, not a single fork:

| Contribution | Origin | What it brings |
| :--- | :--- | :--- |
| Base framework | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | the engine, CLI, server, ggml backends |
| Ternary / sub-2-bit weight kernels | [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) | `PQ2_0`, `PTQ1_0`; Hopper WGMMA and Ampere/Turing MMA paths |
| TurboQuant KV cache + TriAttention | [atomicmilkshake/llama-cpp-turboquant](https://github.com/atomicmilkshake/llama-cpp-turboquant) | native C++/CUDA TurboQuant KV and TriAttention kernels |
| TriAttention method | [arXiv:2604.04921](https://arxiv.org/abs/2604.04921) — Mao et al., MIT / NVIDIA / Zhejiang University, April 2026 | trigonometric series scoring, norm-based key eviction |

Knowing this matters for maintenance: a bug in the TriAttention CUDA scorer is an **upstream-fork** bug, so the fix belongs either upstream or in a clearly marked local patch, and a bug in `ggml_cuda_mul_mat` dispatch is inherited from llama.cpp proper.

## Evidence

- The lineage table and links: [[source-readme]] *Original Projects & Research Citations*.
- Four local checkouts are in play, per [[source-state-md]] §2 — the publication repo (this one), a working copy with a broken WHT, a fixed `Release/` copy, and a TurboQuant fork. None of them is inside this repository; their existence and contents are `[UNVERIFIED]` here.
- [[prismml-weight-kernels]] is the page that carries the edit boundary: `AGENTS.md` forbids modifying PrismML CUDA kernels, so any fix touching them needs the upstream route or an explicit exception.

## Open questions

- Which upstream version of llama.cpp this tree is based on — the answer determines what plumbing (backend registry, graph capture, type system) the custom kernels must match.
- Whether the local TurboQuant/TriAttention patch set is tracked as a patch series against the fork, or was merged into the tree and is now fork-of-a-fork. This decides how painful the next upstream rebase is.
- Whether `Release/` and this repo share history or only content — relevant to roadmap item 1 ([[roadmap]]), which is a port between them.

## See also

[[overview]] · [[codebase-map]] · [[prismml-weight-kernels]] · [[triattention]] · [[turboquant]] · [[source-readme]] · [[source-state-md]]
