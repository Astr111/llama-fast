---
title: Documentation coverage
type: topic
status: current
updated: 2026-09-29
sources: [AGENTS.md]
verified: [src/src, src/common, src/ggml/src/ggml-cuda, src/tools/server]
tags: [meta, planning, coverage]
---

# Documentation coverage

## Bottom line

Of the **1 148 source files** in this tree, **1 001 are never mentioned** by any page of this vault — 87 %. That number is honest but nearly meaningless on its own, because most of those files are upstream llama.cpp baggage (other GPU backends, vendored libraries, alternative model architectures) that this project's documentation has no business covering.

The metric that matters is different: **which mechanisms on this project's own critical path are undocumented**. Measured against that, the vault is in good shape for the four stacks it exists to explain — TurboQuant, TriAttention, the PrismML weight path and the hybrid memory/GDN design — and has a **clear, ranked set of gaps** everywhere those stacks touch the layers around them: the sampler's implementation, the RoPE and SSM kernels, prompt/template assembly, and the server's internals beyond its main loop.

The most consequential single gap is **`src/src/llama-sampler.cpp` (152 KB)**: [[sampling]] documents the parameter surface and the chain order, but the implementation of every sampler stage has never been read.

## How this was measured

Basenames of every `.c/.h/.cu/.cuh/.cpp/.cc/.hpp/.py` token appearing in any page (including `verified:` frontmatter and inline `path:line` citations) were collected from the vault, then diffed against the files present under `src/`.

One methodological trap, recorded because it produced a wrong number first: a regex alternation ordered `cu|cuh` matches `common.cuh` as `common.cu`, so every `.cuh` file looks undocumented. The corrected pattern puts the longest suffix first (`cuh|cpp|cc|hpp|cu|c|h|py`), and the difference is material — `ggml-cuda` moves from 142 to **124** undocumented files, because a dozen kernel headers *are* cited.

## Coverage by area

| Area | Files | Undocumented | % | Note |
| :--- | ---: | ---: | ---: | :--- |
| `src/models` (architecture implementations) | 152 | 146 | 96 % | only the target's own `qwen35` and 5 neighbours are read |
| `ggml/ggml-cuda` | 161 | 124 | 77 % | the fork's own kernels are documented; the rest are upstream op kernels |
| `ggml/ggml-sycl` | 120 | 110 | 92 % | partially covered by [[backend-parity]] |
| `tools/mtmd` | 72 | 72 | 100 % | vision/multimodal — out of scope for every claim in the vault |
| `tools/server` | 28 | 20 | 71 % | [[server-layer]] read the main loop; the rest is undocumented |
| `ggml/ggml-cpu` | 23 | 19 | 83 % | [[cpu-path]] covers the fork's ops |
| `gguf-py/gguf` | 18 | 16 | 89 % | [[conversion-and-packing]] covers the paths that matter |
| `common/jinja` | 13 | 13 | 100 % | the chat-template engine, entirely unread |
| other backends (`virtgpu`, `metal`, `openvino`, `et`, `cann`, `webgpu`, `opencl`, RDNA configs) | ~100 | ~100 | ~100 % | deliberately out of scope |

## What is documented, and to what depth

| Documented | Pages | Depth |
| :--- | :--- | :--- |
| TurboQuant KV, InnerQ, the WHT family, rotation data | [[turboquant]], [[innerq]], [[walsh-hadamard-transform]], [[turbo-wht]], [[rotation-data]] | deep — mechanism, call sites, defects |
| TriAttention, its calibration and its scoring | [[triattention]], [[triattention-calibrate]], [[scoring-correctness]] | deep — including four defects found by reading |
| PrismML weight path and kernel dispatch | [[prismml-weight-kernels]], [[quantized-kernel-units]], [[gemm-dispatch]], [[device-placement]] | deep — dispatch tables read per unit |
| PrismML / model-side Hadamard fold | [[prism-hadamard-weight-fold]], [[conversion-and-packing]] | deep |
| Hybrid attention/SSM structure and GDN | [[qwen35-architecture]], [[hybrid-memory]], [[gated-delta-net]] | deep |
| the token path | [[forward-pass]], [[tokenizer]] | medium — data flow and vocabulary; prompt assembly absent |
| the request path | [[request-lifecycle]], [[sampling]], [[server-layer]] | medium — control flow and surface; implementations absent |
| measurement and build | [[first-live-measurements]], [[build-and-verify]], [[release-artifacts]] | deep for this machine, honest about the target |

## The gaps that matter, ranked

| # | Undocumented | Size | Why it matters | Owner page |
| --: | :--- | ---: | :--- | :--- |
| 1 | `src/src/llama-sampler.cpp` | 152 KB | the implementation of every sampler stage — [[sampling]] documents only the chain order and the flags | [[sampling]] |
| 2 | `src/ggml/src/ggml-cuda/rope.cu` | 45 KB | the RoPE kernels; [[ta-9-rope-scope-mismatch]] turns on exactly what dimensions and frequencies they rotate | [[forward-pass]], [[ta-9-rope-scope-mismatch]] |
| 3 | `src/ggml/src/ggml-cuda/ssm-scan.cu` | 41 KB | the recurrent scan for the 48 SSM blocks; [[gated-delta-net]] read the *op* and not the scan kernel | [[gated-delta-net]] |
| 4 | `src/ggml/src/ggml-cuda/norm.cu` | 37 KB | holds the fused `rms_norm_mul_rope` whose invocation count **doubles** under turbo KV — a measured cost with no documented mechanism ([[first-live-measurements]]) | [[turboquant]] |
| 5 | `src/src/llama-kv-cache-dsv4.cpp` | 78 KB | a **second KV-cache implementation** in the same tree, never mentioned by any source or page | [[kv-cache]], [[hybrid-memory]] |
| 6 | `src/common/chat.cpp`, `common/jinja/*` (13 files), `chat-peg-parser.cpp`, `chat-auto-parser-generator.cpp` | ~250 KB | messages → prompt: template application, tool-call parsing, auto-parser generation. [[tokenizer]] stops at text↔ids | new `chat-templates` |
| 7 | `src/common/json-schema-to-grammar.cpp`, `src/src/llama-grammar.cpp`, `common/peg-parser.cpp` | ~190 KB | constrained decoding and schema→GBNF — the mechanism behind structured output | [[sampling]] |
| 8 | `tools/server/server-tools.cpp`, `server-task.cpp`, `server-schema.cpp`, `server-stream.cpp`, `server-queue.cpp`, `server-mcp.cpp` | ~290 KB | the server's tool-calling, task queue, streaming and **MCP** support — the env vars for MCP were found in [[runtime-switches]], but no page describes the mechanism | [[server-layer]] |
| 9 | `src/common/fit.cpp` | 51 KB | the `--fit` automatic placement logic; plausibly where the unexplained 116 ms of `cudaMemGetInfo` comes from | [[runtime-switches]], [[first-live-measurements]] |
| 10 | `src/src/unicode.cpp`, `unicode-data.cpp` | 224 KB | byte-level BPE and UTF-8 handling — the layer where [[tokenizer]]'s `[UNK_BYTE_0x…]` latent-bug flag lives | [[tokenizer]] |
| 11 | `src/tools/llama-bench/llama-bench.cpp` | 101 KB | an in-tree benchmark harness capable of producing the target-hardware numbers the roadmap keeps asking for | [[benchmarks]] |
| 12 | `src/src/llama-batch.cpp`, `llama-chat.cpp`, `llama-mmap.cpp`, `llama-model-saver.cpp`, `llama-adapter.cpp` | ~190 KB | batching, mmap loading, model saving and LoRA adapters — plumbing around the documented path | [[request-lifecycle]] |

## Deliberately out of scope

Naming these prevents them from being re-discovered as "gaps" by the next pass. None of them affects a claim in this vault:

- **Other backends**: `ggml-sycl` (120 files), `ggml-metal`, `ggml-vulkan`, `ggml-cann`, `ggml-webgpu`, `ggml-opencl`, `ggml-openvino`, `ggml-et`, `ggml-virtgpu`, and the AMD RDNA `mmq-config-*` files. [[backend-parity]] reads the ones that mirror this fork's ops and nothing more.
- **Vision/multimodal**: `tools/mtmd` (72 files), `clip.cpp`. The README's `--mmproj` usage works through them, but no vault claim depends on their internals.
- **Vendored code**: `vendor/*` (nlohmann, cpp-httplib, stb, hash, miniaudio).
- **Tests as a body of source**: 74 files in `src/tests`; [[build-and-verify]] covers what they do and do not assert, which is the part that matters.
- **Alternative architectures**: 146 unread files under `src/src/models`. [[qwen35-variants]] classifies the ones in the target's family; the rest are upstream.

## How to close the top of the list

| Slice | Cost | Deliverable |
| :--- | :--- | :--- |
| Sampler implementation + grammar path (#1, #7) | one agent | a `sampling` page at implementation depth |
| RoPE + norm kernels (#2, #4) | one agent | the mechanism behind TA-9 and the doubling `rms_norm_mul_rope` count |
| SSM scan (#3) | one agent | `gated-delta-net` gains its kernel |
| Prompt assembly (#6) | one agent | a new `chat-templates` entity |
| Server internals (#8) | one agent | `server-layer` at implementation depth, including MCP |
| KV cache DSV4 (#5) | one agent | whether it is a fork-of-fork, a sibling design, or dead code |

Six slices, all read-only, none needing hardware. The other six entries are lower value than these.

## See also

[[open-questions]] · [[decisions-pending]] · [[overview]] · [[codebase-map]] · [[sampling]] · [[server-layer]] · [[gated-delta-net]] · [[tokenizer]] · [[first-live-measurements]] · [[build-and-verify]]
