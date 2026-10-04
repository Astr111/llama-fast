---
title: KV accounting
type: topic
status: current
updated: 2026-09-28
sources: [README.md, start_server_baseline.sh]
verified: [llmwiki/raw/README.md, scripts/start_server_baseline.sh, src/ggml/src/ggml-common.h, src/src/llama-kv-cache.cpp, src/src/llama-hparams.cpp, src/src/llama-hparams.h, src/src/llama-model.cpp, src/src/llama-arch.cpp]
tags: [kv-cache, quantization, benchmarks]
---

# KV accounting — published vs. actual

## Bottom line

The README publishes three **tokens-per-1-GB-VRAM** figures that were derived assuming a **dense 64-layer KV cache**. The actual model, `qwen35`, is a hybrid that keeps a KV cache on only **16 of its 64 blocks** — the other 48 are Gated-DeltaNet (state-space) layers whose recurrent state is fixed-size regardless of context. Two of the three published figures (baseline `~4 000` and max-mem `~25 200`) match the dense-64 arithmetic to within 1–2 %. The speed profile's `~20 000` matches **neither** the dense-64 type arithmetic nor the 16-layer arithmetic. The baseline's own launch-script banner (`"256 KB/token, 4K tok/GB VRAM"` at `scripts/start_server_baseline.sh:25`) is internally consistent with its *published* number but wrong for this model's cache size.

## Block-element geometry

All byte-per-element figures derived from the `static_assert` sizes in `src/ggml/src/ggml-common.h` (the struct comments contain stale 32-element values and are not used):

| Type | Struct | QK | Bytes / block | Bytes / element |
| :--- | :--- | :---: | :---: | :---: |
| `f16` | — | — | — | 2.0000 |
| `q8_0` | `block_q8_0` | 32 | 34 | 1.0625 |
| `turbo3_0` | `block_turbo3_0` | 128 | 50 | 0.390625 |
| `turbo2_0` | `block_turbo2_0` | 128 | 34 | 0.265625 |

The 14 B / 32-elt comment above `block_turbo3_0` and the 10 B / 32-elt comment above `block_turbo2_0` are **stale** (from when QK was 32); the `static_assert` enforcing 50 B and 34 B respectively governs the actual layout.

`src/ggml/src/ggml-common.h:282-284` (`q8_0`), `:324-333` (`turbo3_0`, `QK_TURBO3 = 128`), `:374-381` (`turbo2_0`, `QK_TURBO2 = 128`).

## Per-layer element count

Qwen3.5 uses `head_dim = 256`, `n_head_kv = 4` (GQA group 6). From `src/src/llama-hparams.cpp:131-141`:

- `n_embd_k_gqa = head_dim_k × n_head_kv = 256 × 4 = 1024`
- `n_embd_v_gqa = head_dim_v × n_head_kv = 256 × 4 = 1024`

The `is_recr()` rule for `qwen35` (interval 4, full-attention on layers 3, 7, 11, …, 63) is derived at `src/src/models/qwen35.cpp:20-29`; `is_recr(QWEN35) = false` at `src/src/llama-arch.cpp:1078-1080` means the interval rule drives every assignment. The KV cache is allocated with filter `!hparams.is_recr(il)` (`src/src/llama-memory-hybrid.cpp:44-64`, `src/src/llama-model.cpp:2787-2793`), giving **16 attention/KV layers and 48 SSM layers** — confirmed by the 16-layer filter on `layers` in `src/src/llama-kv-cache.cpp:194` and `:394`, and by the layer-count in the layer log line `:448` (`"%3d layers"`).

## Bytes per token (16 KV-bearing layers)

Per-layer KV row elements: K = 1024, V = 1024.

| Profile | K bytes/layer | V bytes/layer | Total bytes/layer | Per token (×16) | Per token |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `f16` K + `f16` V | 2048 | 2048 | 4096 | **65 536 B** | **64 KiB** |
| `turbo3` K + `q8_0` V | 400 | 1088 | 1488 | **23 808 B** | **23.25 KiB** |
| `turbo3` K + `turbo2` V | 400 | 272 | 672 | **10 752 B** | **10.5 KiB** |

Tokens per GiB = 1 073 741 824 ÷ bytes-per-token:

| Profile | 16-KV-layer tok/GiB | Dense-64-layer tok/GiB | Published tok/GiB | Ratio to 16-L | Ratio to dense-64 | Consistent with |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| Baseline FP16 | 16 384 | 4 096 | **~4 000** | 0.244 | **0.978** | dense-64 ✓ |
| Speed (t3+q8) | 45 101 | 11 275 | **~20 000** | 0.444 | **0.451** | **neither** |
| Max-Mem (t3+t2) | 99 865 | 24 966 | **~25 200** | 0.252 | **1.009** | dense-64 ✓ |

The `is_recr()` / interval rule and the 16/48 split are at `src/src/llama-model.cpp:656-663` (GQA computation keyed to `is_recr`), `:1422` (pre-fill of `is_recr_impl`), `:2772-2830` (hybrid container construction with the filter lambda). Layer-adaptive KV type selection (boundary modes 5/6/7) in `src/src/llama-kv-cache.cpp:283-318` counts `n_layer() = 64` for its boundary logic but the filter means only the 16 attention layers receive those overrides; the effective layer types for the standard `turbo3+q8` case are `layer_type_k = GGML_TYPE_TURBO3_0` and `layer_type_v = GGML_TYPE_Q8_0` for all 16 KV layers (no adaptive overrides in default mode).

## The `~20 000` discrepancy

No combination of the types, layer counts, or context lengths present in this repository produces 53 687 B/token (the bytes-per-token this figure implies). Verified candidates:

- **Dense-64, `turbo3` K + `q8_0` V:** 262 144 B / (400+1088)/2 per layer × 64 = 262 144 / 95 232 = **11 275 tok/GiB** (not 20 000).
- **16-layer model, `turbo3` K + `q8_0` V:** 23 808 B → **45 101 tok/GiB** (not 20 000).
- **Dense-64, `turbo2` K + `q8_0` V:** 262 144 / (272+1088)×32 × 64 = 262 144 / 87 040 = **12 000 tok/GiB**.
- **Dense-64, `turbo2` K + `turbo2` V:** 262 144 / 43 008 = 6095 tok/GiB.
- **Any 16-KV-layer variant** with `turbo3`+`q8_0` or any other plausible type combination maxes out around 45 101 tok/GiB.

The figure appears to be a **draft or copy-paste error** — possibly a typo for ~45 000 (the correct 16-layer `turbo3`+`q8_0` value), or an early approximation that predated the 16/48 split discovery. The measured VRAM for the speed profile is 8 122 MiB for 16K context (`llmwiki/raw/README.md:222`), which against a 23 808 B/token cache computes to ~8 500 tokens — consistent with the 16-layer cache size, not the published 20 000.

## Baseline banner internal consistency

`scripts/start_server_baseline.sh:25`:
```
echo " KV Cache:          FP16 (256 KB/token, 4K tok/GB VRAM)"
```

256 KB = 262 144 B = dense-64 ×1 (each of 64 layers × 4096 B = 262 144 B). This is exactly the dense-64 `f16` row of the table above. The banner's own arithmetic (256 KiB → 4096 tok/GiB) is internally consistent with the **published** `~4 000` figure.

The inconsistency is that the banner describes the *model's* cache as 256 KiB/token. The real model's KV cache is **64 KiB/token** (16 of 64 layers × 4096 B). So the banner is wrong about this model's memory footprint by exactly a factor of 4, while correctly describing the arithmetic that produces the published `~4 000` figure. A reader who trusts the banner to describe their actual hardware will budget 4× too much VRAM for the KV cache.

## Verdict

| Figure | Can be trusted? | Reason |
| :--- | :--- | :--- |
| **Baseline `~4 000`** | Partially | Internal arithmetic is consistent (256 KiB/token → 4096 tok/GiB, both dense-64). But the banner describes a cache 4× larger than this model's actual 16-KV-layer cache. Use 16 384 tok/GiB for capacity planning on this model. |
| **Speed `~20 000`** | **No** | Matches no arithmetic derivable from the type geometry, layer counts, or context lengths in the repository. Recorded as an open question. |
| **Max-Mem `~25 200`** | Partially | Within 1 % of dense-64 `turbo3`+`turbo2` arithmetic (24 966 tok/GiB). Correct if you are planning a **dense-64-layer** model's KV cache. On this model's 16-layer cache the correct figure is ~99 865 tok/GiB. |

The VRAM savings column (`"−3.7 GB"` and `"−3.9 GB"` at `llmwiki/raw/README.md:223`) implies a 11 800 MiB baseline, against which 8 122 and 7 914 MiB are measured. A 16-layer `f16` cache at 16K context occupies ~1 024 MiB; the measured delta (3.7–3.9 GiB) is larger than the cache savings alone, suggesting the column includes weight offloading differences between the FP16 baseline and the PQ2_0/+turboquant runs. Whether it includes the SSM recurrent state (~150 MiB, fixed, never evicted) or the TriAttention budget tensor is not determined.

## Open questions

- Where did the `~20 000` figure originate? Was it ever a measured number, or is it a typo?
- Does the VRAM savings column (`−3.7 / −3.9 GiB`) describe KV savings only, or KV + weight offload differences between the FP16 baseline and the PQ+turboquant runs?
- The baseline banner (`"256 KB/token"`) is wrong for this model's 16-KV-layer cache by exactly a factor of 4. Should it be corrected to `"64 KB/token, 16K tok/GB VRAM"`?

## See also

[[benchmarks]] · [[kv-cache]] · [[turboquant]] · [[qwen35-architecture]] · [[hybrid-memory]] · [[quantization]]
