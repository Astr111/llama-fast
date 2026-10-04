---
title: Source — kv-mean-center.md
type: source
status: current
updated: 2026-09-28
sources: [kv-mean-center.md]
verified: [src/common/arg.cpp, src/common/common.h, src/common/common.cpp, src/common/kv-mean-center.h, src/common/kv-mean-center.cpp, src/include/llama.h, src/src/llama-context.cpp, src/src/llama-kv-cache.cpp, src/src/llama-graph.cpp, src/tests/test-kv-mean-center.cpp, src/tools/kv-mean-center/README.md]
tags: [kv-cache, quantization, calibration]
---

# Source — `kv-mean-center.md`

## Summary

`--kv-mean-center` is an opt-in feature that subtracts a fixed, precomputed per-`(kv-head, channel)` bias `k_bar` from the K vector at the moment it is written into the KV cache, to improve quantization fidelity of a `GGML_TYPE_Q4_0` K cache. `Q4_0` is a symmetric (zero-point-free) block quantizer, so a channel whose real K activations have a nonzero mean across tokens spends dynamic range encoding that constant; centering the residual around zero removes the waste. It is exactly correctness-preserving because `q · k_bar` is a position-independent additive constant on every logit of a query row, and softmax is shift-invariant. Scope is `Q4_0` only; the bias file is a small GGUF with one F32 tensor per layer, produced by `tools/kv-mean-center` from a text calibration corpus.

**Is this fork's code or an untouched upstream feature?** It is **live in this fork and used by it** — and it is explicitly *composed with* the fork's own Hadamard K rotation. Checked against the tree, not the title: the flag is registered in `src/common/arg.cpp:2467-2475` (`--kv-mean-center FNAME`, env `LLAMA_ARG_KV_MEAN_CENTER`, description pointing at `docs/kv-mean-center.md`), surfaced as `llama_context_params::path_kv_mean_center` (`src/include/llama.h:395-397`) and `common_params::kv_mean_center_path` (`src/common/common.h:581-583`), hard-gated to `Q4_0` in `src/src/llama-context.cpp:3938-3942`, loaded by `llama_kv_cache::load_kv_mean_center()` (`src/src/llama-kv-cache.cpp:1669`), written by `src/common/kv-mean-center.{h,cpp}`, driven by the CLI tool `src/tools/kv-mean-center/`, and tested by `src/tests/test-kv-mean-center.cpp`. The interaction with `LLAMA_ATTN_ROT_DISABLE` / `attn_rot_k` (`src/src/llama-kv-cache.cpp:462-468`) is the fork-specific part. Whether the feature *originated* upstream is `[UNVERIFIED]` — no provenance note is present in the document.

## Key claims

- **What it does** *(“The idea”)*: measure a per-`(kv-head, channel)` bias `k_bar` offline, subtract it from the K vector for every token immediately before `Q4_0` quantization on cache write. Nothing else in attention changes; no decode-time cost beyond one subtract.
- **Why it is safe** *(“Why this is safe (softmax-invariance)”)*: `q · k_i = q · (k_i − k_bar) + q · k_bar`; the second term is identical for every logit in a query row, and `softmax(x + c) == softmax(x)`. Exactly correctness-preserving in infinite precision; in practice only float rounding differs. `tests/test-kv-mean-center.cpp` checks this against an unquantized F32 K cache.
- **What it buys** *(same section)*: only quantization fidelity — a smaller residual for channels with a real, consistent activation bias. The doc notes that a logit-KLD measurement at production scale is a follow-up and that **this repo ships no measured number for a specific trained model**.
- **Usage** *(“Usage”)*: generate with `./llama-kv-mean-center -m model.gguf -f calibration-data.txt -o kv-mean-center.gguf`; load with `./llama-cli -m model.gguf -ctk q4_0 --kv-mean-center kv-mean-center.gguf`. `--kv-mean-center` **requires** `--cache-type-k q4_0`; any other K type fails context creation with a clear error rather than silently doing nothing.
- **Bias file format** *(“Bias file format”)*: a small GGUF with one F32 1-D tensor per layer that has a bias, named `kv_bar.blk.<il>.k`, holding `n_embd_head_k(il) * n_head_kv(il)` values laid out `[n_embd_head_k, n_head_kv]` (channel-fastest) — the in-memory layout of the K tensor at cache-write time, so it loads as a small broadcastable per-layer bias.
- **Calibration** *(“Calibration”)*: `tools/kv-mean-center` averages the K tensor right before it would be written into the cache, captured via the `k_cache_in` tag added to `llm_graph_context::build_attn()` and read through the same backend-scheduler eval-callback mechanism `llama-imatrix` uses.
- **Layout coverage** *(“Scope and limitations”)*: every *standard*-attention KV layout — plain cache, the base/SWA pair of sliding-window models, and the attention sub-cache of hybrid (recurrent + attention) models, with or without SWA. Recurrent-only and MLA/DSA memory types are not supported.
- **Hook coverage** *(same section)*: the `k_cache_in` hook is wired only into the standard dense/GQA path (`build_attn(llm_graph_input_attn_kv *, ...)`); MLA and other specialized attention variants are not covered yet.
- **Basis interaction — the fork-specific part** *(last bullet of “Scope and limitations”)*: the bias lives in the basis the calibration run's K cache used. With this fork's optional Hadamard K rotation (automatic for quantized K caches whose head dimension is a multiple of 64, unless `LLAMA_ATTN_ROT_DISABLE=1`), a user must calibrate with the same `--cache-type-k` and rotation settings they serve with (`-ctk q4_0`); the tool records the basis as `kv_mean_center.k_rot` and the **loader rejects a mismatch**. A wrong-basis bias is still exactly safe for logits (the invariance argument is basis-independent) but degrades quantization quality instead of improving it. In the matching basis the two features compose.
- **Other types** *(same section)*: only `GGML_TYPE_Q4_0` is supported; generalizing to other quantization types is future work.

## Pages derived

- [[kv-cache]] — the write path this feature hooks, and the `Q4_0` symmetric-quantizer argument for why centering helps.
- [[quantization]] — the symmetric-block-quantizer dynamic-range argument, generalized.
- [[backend-parity]] — the `Q4_0`-only gate and the `k_cache_in` hook are graph/backend-scheduler mechanics, independent of which backend runs the matmuls; cited there as the fork's other opt-in, cache-type-gated feature.

## Provenance

- raw path: `llm-wiki/raw/kv-mean-center.md`
- Original: `src/docs/kv-mean-center.md`
- sha256: `38e3219b3382dd98928328cc5937f99b674e405e3e084e9f0236f67c6fa3cab9`
- Ingested: 2026-09-28
- Repo paths whose claims were checked against code (see `verified:`): the CLI flag, the context-param plumbing and `Q4_0` gate, `load_kv_mean_center()`, the `k_cache_in` tag at `src/src/llama-graph.cpp:2958-2962`, the `kv_mean_center.k_rot` write (`src/common/kv-mean-center.cpp:49-50`) and the loader's basis check (`src/src/llama-kv-cache.cpp:1702-1730`), the tool README's calibration-basis rule (`src/tools/kv-mean-center/README.md:60-81`), and `tests/test-kv-mean-center.cpp` (the gate test and the F32-cache softmax-invariance test).
