---
title: Source — hadamard-tied-output.md
type: source
status: current
updated: 2026-09-28
sources: [hadamard-tied-output.md]
verified: []
tags: [hadamard, conversion, loader, prismml]
---

# Source — `hadamard-tied-output.md`

## Summary

A contract page for **sharing one latent token embedding between row lookup and the output projection**. For a normalized Hadamard matrix `H` and sign vector `s`, the lookup reconstructs `s * (H z)`; the output projection reuses the same stored rows after transforming its input as `H (s * h)`. The whole point of the page is the *metadata* that makes this legal: a version-2 Hadamard contract, a tied-output flag, an explicit absence of `output.weight`, and — on the conversion side — `hadamard_packing.json` schema 3. Version 1 remains valid only for models with an explicit head, and a version-1 runtime must reject a version-2 artifact before inference rather than mis-loading it.

## Key claims

- **The transform pair** *(intro)*: lookup is `s * (H z)`; the output projection transforms its input as `H (s * h)` and reuses the same stored rows.
- **Loader requirements** *(“The shared-output contract requires”)*: `prism.hadamard.version = 2`, `prism.hadamard.tied_output = true`, `token_embd.weight` present in `prism.hadamard.inverse_weight_names`, and **no** `output.weight` tensor or entry in `prism.hadamard.weight_names`.
- **Transform registration** *(paragraph after the requirement list)*: the loader registers the output tensor's activation transform from the **embedding's block size and signs**, and binds the forward and inverse transforms to the actual output and lookup tensors — which may be **separate runtime allocations** when placed on different devices. “Sharing on disk does not guarantee a single allocation across devices.”
- **Version gate** *(same paragraph)*: version 1 stays supported for explicit heads; a latent embedding with no explicit head *requires* version 2; a version-1 runtime rejects version 2 before inference. The graph verifier rejects a latent lookup tensor consumed by a matmul that has no registered forward transform.
- **Conversion contract** *(“For conversion” paragraph)*: a shared latent head requires `hadamard_packing.json` schema 3 with `tied_output=true`, a tied HF configuration, and no separate output head. The embedding keeps its `inverse-after-lookup` record; other folded tensors keep their existing records.
- **Two independent version numbers** *(same paragraph)*: manifest schema versions and GGUF contract versions are distinct; schema 3 exists so that older converters **reject** the new contract instead of silently ignoring the flag.
- **What is *not* covered** *(closing paragraph)*: plain non-Hadamard tied embeddings need no such metadata; separately trained output heads must stay explicit; consumers that only support the version-1 contract must be updated before they can accept version-2 artifacts.

## Pages derived

- [[prism-hadamard-weight-fold]] — the runtime side of the same fold: the embedding/head weights are stored rotated and the transform is applied to activations.
- [[conversion-and-packing]] — the converter side: `hadamard_packing.json` schema 3, `tied_output`, the `inverse-after-lookup` record, and why schema 3 is a *rejection* mechanism for old converters.
- [[backend-parity]] — the contract's consumer side is backend-independent: the SYCL `cpy` path states that `PTQ1_0`/`PQ2_0` are produced offline by the converter “which also applies the Hadamard rotation the packing assumes”, so no backend may re-derive the rotation at runtime.

## Provenance

- raw path: `llmwiki/raw/hadamard-tied-output.md`
- Original: `src/docs/development/hadamard-tied-output.md`
- sha256: `cba2bd0d37fac99dfa3ed458370f753c32fa5a6680c2f9dfb49dff30842283ba`
- Ingested: 2026-09-28
- Repo paths whose claims were checked against code: none (prose-only document; every claim above is a restatement of this file). The cross-reference to backend behaviour is checked in [[backend-parity]].
