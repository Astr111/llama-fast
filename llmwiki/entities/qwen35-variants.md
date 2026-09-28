---
title: Qwen3.5-family variants (qwen35moe, qwen3next)
type: entity
status: current
updated: 2026-09-28
sources: []
verified: [src/src/models/qwen35.cpp, src/src/models/qwen35moe.cpp, src/src/models/qwen3next.cpp, src/src/llama-arch.cpp, src/src/llama-arch.h, src/src/llama-model.cpp]
tags: [architecture, hybrid, ssm, moe, model]
---

# Qwen3.5-family variants (`qwen35moe`, `qwen3next`)

## What it is

The three architectures in this tree that share the target's hybrid attention / state-space block design. Only one of them is the model this fork serves ([[qwen35-architecture]]); the other two are the loader's neighbours:

| Architecture id | Registered | What it is |
| :--- | :--- | :--- |
| `qwen35` | `src/src/llama-arch.cpp:41` | the target — dense Qwen3.5; 27B/64-layer among its sizes |
| `qwen35moe` | `src/src/llama-arch.cpp:42` | the same generation with a Mixture-of-Experts FFN instead of the dense one |
| `qwen3next` | `src/src/llama-arch.cpp:38` | the **previous generation** whose block design the Qwen3.5 family reuses |

All three are classified hybrid at the architecture layer (`llm_arch_is_hybrid`, `src/src/llama-arch.cpp:1082-1105`: `QWEN3NEXT:1092`, `QWEN35:1096`, `QWEN35MOE:1097`), so all three route through the hybrid memory container and get the same *kind* of 16-of-64-style cache/SSM split as the target.

## How it works

### What the three genuinely share (read in all three files)

- **The same interval-4 schedule, character for character.** Each loader has the identical loop, only at a different line:

  ```cpp
  if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
      uint32_t full_attn_interval = 4;
      ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
      for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
          hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
      }
  }
  ```

  `src/src/models/qwen35.cpp:21-27`, `src/src/models/qwen35moe.cpp:24-30`, `src/src/models/qwen3next.cpp:21-27`. The key name is `%s.full_attention_interval` (`src/src/llama-arch.cpp:255`) and the override key is `%s.attention.recurrent_layers` (`:304`). **So the earlier grouping of all three under one interval-4 rule is correct as far as the layer schedule goes — including `qwen3next`.** The variant differences are elsewhere (RoPE family, SSM tensor names, recurrence shape, rollback support), see the table below. Derived from that loop: full-attention layers are the ones with `(i+1) % 4 == 0`, i.e. `floor(n_layer/4)` of them — 16 of 64 for the target, 12 of 48 for `qwen3next`, 10 of 40 / 12 of 48 / 15 of 60 for the MoE sizes.
- **The same SSM hyperparameter keys** (`ssm.conv_kernel`, `ssm.inner_size`, `ssm.state_size`, `ssm.time_step_rank`, `ssm.group_count`) and the same `SSM_A_NOSCAN` tensor name `blk.%d.ssm_a` (`src/src/llama-arch.h:517`, `src/src/llama-arch.cpp:481`) — a "no `GGML_OP_SSM_SCAN`" `ssm_a` variant used by all three (`src/src/models/qwen3next.cpp:95`, `qwen35.cpp:85`, `qwen35moe.cpp:91`).
- **The same NextN/MTP handling**: `LLM_KV_NEXTN_PREDICT_LAYERS` plus the `mtp_only` / `TENSOR_SKIP` flags (`qwen35.cpp:40-42`, `qwen35moe.cpp:43-45`, `qwen3next.cpp:42-44`).
- **The same hybrid-memory construction branch** (`src/src/llama-model.cpp:2787` lists `QWEN3NEXT || QWEN35 || QWEN35MOE` together), so per-layer KV-vs-recurrent filtering on the non-recurrent set is the same code path as the target's.

**Correction to a neighbouring page.** [[qwen35-architecture]] states that `llm_arch_is_recurrent(QWEN35) = true` fills `is_recr_impl` with 1 for every layer. Reading the function, `llm_arch_is_recurrent` (`src/src/llama-arch.cpp:1068-1081`) returns true only for `MAMBA`/`MAMBA2`/`RWKV6`/`RWKV6QWEN2`/`RWKV7`/`ARWKV7` and **false by default** — it does not list `QWEN35`, `QWEN35MOE` or `QWEN3NEXT`. So the pre-fill at `src/src/llama-model.cpp:1422` writes 0, not 1. The schedule is unaffected either way, because the interval loop above overwrites *every* entry of `is_recr_impl` when the metadata key is absent; but the "fills 1 for every layer" phrasing is not what the code does.

### How each variant differs from the target

Every row below is something read in the file named in the last column; where nothing differs in the function that was inspected, the row says so.

| | `qwen35` (target) | `qwen35moe` | `qwen3next` |
| :--- | :--- | :--- | :--- |
| **File** | `src/src/models/qwen35.cpp` | `src/src/models/qwen35moe.cpp` | `src/src/models/qwen3next.cpp` |
| **Layer → model type** | 24/32/64 → 0.8B/2B/4B/9B/27B (`:29-34`) | 40/48/60 → 35B-A3B / 122B-A10B / 397B-A17B (`:32-37`) | 48 → 80B-A3B (`:29-32`) |
| **FFN** | **dense only** — builder asserts `ffn_gate_inp == nullptr` and runs `build_ffn(ffn_up, ffn_gate, ffn_down)`; comment *"Qwen3.5 does not use MoE FFN"* (`:585-598`) | **MoE** — expert keys `expert_feed_forward_length` / `expert_shared_feed_forward_length` loaded (`:5-6`), `n_ff_exp`/`n_ff_shexp` derived (`:61-62`), builder asserts `ffn_gate_inp != nullptr` and runs `build_moe_ffn` with `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`, plus optional shared experts (`:497-519`) | **MoE, mandatory** — same expert keys, and `load_arch_tensors` **throws** `"model cannot have zero experts"` when `n_expert == 0` (`:38-40`); builder takes an MoE *or* dense branch depending on `ffn_gate_inp` (`:568-590`) |
| **Attention interval** | interval 4, default, overridable (`:21-27`) | identical (`:24-30`) | identical (`:21-27`) — **the loader rule is the same**, not a different generation's different rule |
| **RoPE** | reads `rope.dimension_sections` in the hparams loader (`:4-27`) and applies `ggml_rope_multi` (`:367`, `:373`); arch rope type `IMROPE` (`src/src/llama-model.cpp:3244-3247`) | reads `rope.dimension_sections` and applies `ggml_rope_multi` (`:327`, `:333`); arch rope type `IMROPE` (`src/src/llama-model.cpp:3245`) | **does not read `rope.dimension_sections`**; applies plain `ggml_rope_ext` (`:282`, `:287`); arch rope type `NEOX` (`src/src/llama-model.cpp:3224`) |
| **SSM β/α tensors** | separate `blk.%d.ssm_beta` and `blk.%d.ssm_alpha`, each `[n_embd, n_v_heads]` (`:86-87`, names at `src/src/llama-arch.cpp:485`, `:502`) | identical to the target (`:92-93`) | **fused** `blk.%d.ssm_ba` (`LLM_TENSOR_SSM_BETA_ALPHA`, `src/src/llama-arch.cpp:484`), one tensor of `[n_embd, ba_dim]`, read as `mixed_ba` in the graph (`:96`, `:422`) |
| **v-head broadcasting** | segmented pattern *"[k0_v0, k1_v1, k0_v2, k1_v3]"* for the fused QKV/`ssm_conv1d`/conv-cache tensors (`src/src/llama-model.cpp:583-600`) | same as the target (same branch, `:583-600`) | **different**: the default split pattern *"[k0_v0, k0_v1, k1_v2, k1_v3]"*, with an explicit comment in code that Qwen3.5 needs its own segmentation (`src/src/llama-model.cpp:594-599`) |
| **Recurrent-state rollback** | `llm_arch_supports_rs_rollback = true` (`src/src/llama-arch.cpp:1118-1132`, case at `:1120`) | `true` (`:1121`) | **not listed** → false; the speculative-decoding rollback hook ([[speculative-decoding]]) is not advertised for this arch |
| **`llm_arch_is_recurrent`** | false | false | false |

### Is the target model a MoE?

**No — the target is dense, and `qwen35moe` is simply a different architecture that this file does not use.** The evidence is the arch id itself: [[ternary-bonsai-2-27b]] records `general.architecture = qwen35`, which selects `llama_model_qwen35` (`src/src/llama-arch.cpp:41`). That loader never reads an expert hyperparameter (its whole hparams function is `qwen35.cpp:4-27`), and its FFN builder asserts there is no router (`ffn_gate_inp == nullptr`) and builds the plain `ffn_up`/`ffn_gate`/`ffn_down` path (`qwen35.cpp:585-598`). `qwen35moe` is the exact complement: it *requires* `expert_feed_forward_length` and asserts `ffn_gate_inp != nullptr` (`qwen35moe.cpp:497-499`).

This agrees with the tensor inventory quoted from that page (851 tensors, per-`ffn_*` shape `[5120, 17408]`, no expert tensors): a `qwen35moe` file would have to carry `ffn_gate_inp` plus `ffn_{gate,up,down}_exps` per MoE layer, and a flat `ffn_*` of `[5120, 17408]` is the dense-FNN shape of `n_ff = 17408`. So: **dense target, MoE variant unused here** — not "the MoE code is dead", just a different arch id selecting a different loader.

### Which of these the fork can actually run with its custom types

The custom machinery does not special-case the target's arch — it keys off hyperparameters, so what matters is whether a variant supplies the same *shape* of assumptions:

- **KV/turbo + TriAttention mount on any hybrid arch.** All three are `llm_arch_is_hybrid` and share the memory-construction branch (`src/src/llama-model.cpp:2787`), so a variant gets the same per-non-recurrent-layer KV cache that TriAttention attaches to and that the `turbo*`/`q8_0` KV types live in. The interval rule guarantees a non-empty KV-bearing layer set for any interval ≥ 2, so none of the three breaks the "a KV-bearing layer set exists" assumption.
- **The shipped TriAttention calibration does not transfer.** `calibration/bonsai-27b.triattention` is per-model (`num_layers`, `num_attn_heads`, `num_kv_heads`, `n_sampled` pairs — see [[qwen35-architecture]]); it was written for 64 layers / 24 heads / 4 KV heads. `qwen35moe` sizes are 40/48/60 layers and `qwen3next` is 48, so sampling a *variant* with this profile would place sampled layers outside the cache — the mapping code handles that by zeroing the score row (`src/src/llama-kv-cache.cpp:3015-3035`), i.e. it degrades rather than crashes, but the eviction scores would be meaningless. [[triattention-calibrate]] owns the profile format.
- **`head_dim` and the 128-element turbo block.** TurboQuant's block is 128 values (`QK_TURBO3 = 128`), and the target's `head_dim = 256` is exactly aligned, which is why [[tq-5-tail-elements]] does not bite it ([[qwen35-architecture]]). Whether the MoE/Next variants in the wild also use 256-wide heads cannot be checked here — neither variant's GGUF is present in this tree — so any variant claim about tail-element alignment is `[UNVERIFIED]`.
- **MoE weights are outside the custom-quantized set.** The turbo types cover KV-cache tensors; the expert FFN weights are ordinary ggml-quantized matmuls, so [[tq-1-missing-gemm-kernels]] and [[gemm-dispatch]] are not made worse by the MoE path — but nothing here shows the MoE path was ever profiled either. `[UNVERIFIED]`.
- **PrismML weight folding is verified for all three arches.** The loader refuses to apply the Hadamard fold for architectures it has not verified, and its whitelist explicitly contains `QWEN35`, `QWEN35MOE` and `QWEN3NEXT` alongside `LLAMA`/`QWEN3` (`src/src/llama-model.cpp:2318-2332`). So a folded variant file would load ([[prism-hadamard-weight-fold]]).
- **Speculative decoding.** Recurrent-state rollback is advertised for `qwen35`/`qwen35moe` but not `qwen3next` (`src/src/llama-arch.cpp:1118-1132`), so a `qwen3next` target would not have that hook even though its MTP block is loaded through the same `NEXTN` key.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/src/models/qwen35.cpp` | target arch: interval rule `:21-27`; type table `:29-34`; per-block tensors `:37-128`; `ggml_rope_multi` `:367`/`:373`; dense FFN builder `:585-598`; MTP graph `:600-771` |
| `src/src/models/qwen35moe.cpp` | MoE sibling: interval rule `:24-30`; type table `:32-37`; expert keys `:5-6`; expert tensor block `:58-...`; `ggml_rope_multi` `:327`/`:333`; MoE FFN builder `:497-519` |
| `src/src/models/qwen3next.cpp` | previous generation: interval rule `:21-27`; zero-expert guard `:38-40`; fused `ssm_ba` `:95-96`, used `:422`; `ggml_rope_ext` `:282`/`:287`; FFN builder `:568-590` |
| `src/src/llama-arch.cpp` | names `:38`, `:41`, `:42`; `full_attention_interval` key `:255`; `attention.recurrent_layers` key `:304`; rope-sections key `:314`; SSM tensor names `:481-502`; `llm_arch_is_recurrent` `:1068-1081`; `llm_arch_is_hybrid` `:1082-1105`; `llm_arch_supports_rs_rollback` `:1118-1132` |
| `src/src/llama-arch.h` | arch enum entries `:43`, `:46`, `:47`; the `SSM_A_NOSCAN` / `SSM_BETA_ALPHA` special-case comments `:517`, `:523` |
| `src/src/llama-model.cpp` | arch dispatch `:318-323`; v-head broadcast pattern split `:583-600`; `is_recr_impl` pre-fill `:1422`; Hadamard-fold whitelist `:2318-2332`; hybrid-memory branch `:2787`; rope type `:3224` (next → NEOX) vs `:3244-3247` (qwen35 family → IMROPE) |

## Known issues

- **Three architectures, one rule.** The interval-4 loop is genuinely identical across all three loaders, which makes it easy to generalise from one to the others — but the RoPE family, the SSM β/α tensor layout and the v-head broadcast pattern all differ between `qwen3next` and the `qwen35` pair, and the rollback hook differs between the pair and `qwen3next`. Anything that reasons about "the Qwen3.5 hybrid block" should say which arch it read.
- **The `is_recr_impl` pre-fill claim on [[qwen35-architecture]] is wrong as written** (`llm_arch_is_recurrent` is false for this arch family, `src/src/llama-arch.cpp:1068-1081`); the *behaviour* it describes (48 of 64 layers recurrent) is produced by the interval loop, not by the pre-fill.
- **No variant is exercised by the project's own artifacts.** The launch scripts, the calibration profile and every published number are for the dense `qwen35` 27B ([[ternary-bonsai-2-27b]], [[benchmarks]]); nothing in the tree measures a `qwen35moe` or `qwen3next` run ([UNVERIFIED] — no GGUF for either variant is present to check).
- The `head_dim` consequences that dominate the target's issue list ([[ta-1-wht-inversion-256]], [[tq-7-innerq-max-channels]], [[ta-3-cpu-fallback-transfers]]) are properties of head width and the KV-bearing layer set; the layer set is guaranteed for all three variants, the head width is not checkable here. `[UNVERIFIED]`.

## See also

[[qwen35-architecture]] · [[ternary-bonsai-2-27b]] · [[hybrid-memory]] · [[gated-delta-net]] · [[codebase-map]]
