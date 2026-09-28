---
title: Qwen3.8-27B-DFlash2 draft model
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: ["/hdd2/lm-studio-models/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf", src/common/speculative.cpp, src/common/arg.cpp, src/src/models/dflash.cpp, src/src/llama-arch.cpp, src/src/llama-ext.h, src/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py, scripts/start_server_turbo.sh, scripts/start_server_baseline.sh, scripts/run_cli.sh]
tags: [speculative-decoding, draft-model, dflash, model]
---

# Qwen3.8-27B-DFlash2 draft model

## What it is

The speculative-decoding **draft model** for the target: **`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`**, named in [[source-state-md]] *Models* as the companion to [[ternary-bonsai-2-27b]] and run with `draft-dflash` (the source describes it as "max 5").

It is a **DFlash2 block-diffusion drafter**, not a small copy of the target: upstream `z-lab/Qwen3.8-27B-DFlash2` — `general.finetune = DFlash2` over `Qwen/Qwen3.8-27B`, apache-2.0, `general.size_label = 1.9B` — quantized to Q4_K_M. Its header was read from the file:

| Property | Value (from the file) |
| :--- | :--- |
| GGUF version / tensors / KV pairs | 3 / 81 / 48 |
| `general.architecture` | `dflash` |
| `general.name`, `general.basename`, `general.organization`, `general.author` | `Qwen3.8-27B-DFlash2`, `Qwen3.8-27B`, `z-lab`, `Inco AI` |
| `general.base_model.0` | `Qwen3.8 27B` / `Qwen` / `https://huggingface.co/Qwen/Qwen3.8-27B` |
| `general.source.url` | `https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2` |
| `general.tags` | `dflash2`, `speculative-decoding`, `block-diffusion`, `draft-model`, `sglang`, `text-generation` |
| `general.file_type` | `15` (see *Known issues*) |
| `dflash.block_count` | 5 |
| `dflash.embedding_length` / `feed_forward_length` | 5120 / 17408 |
| `dflash.attention.head_count` / `head_count_kv` | 32 / 8 |
| `dflash.attention.key_length` / `value_length` | 128 / 128 |
| `dflash.attention.sliding_window` | 2048 (window pattern of 5 entries, all true) |
| `dflash.attention.causal` | **False** |
| **`dflash.block_size`** | **8** (the code's fallback default is 16) |
| **`dflash.selector_rank` / `selector_top_k`** | **256 / 16** (selector_top_k > 0 ⇒ this is the DFlash2 lineage) |
| `dflash.conv_kernel_size` / `conv_group_size` | 2 / 16 |
| **`dflash.target_layers`** | **[6, 20, 34, 48, 62]** |
| `dflash.context_length` | 262144 |
| Tokenizer | GPT-2-style `qwen35`, 248 320 tokens, BOS/PAD 248044, EOS 248046, **`mask_token_id` 248070** |
| Weight storage | 45 × `Q4_K`, 4 × `Q6_K`, 32 × `F32` |

Its tokenizer parameters are identical to the target's (248 320 tokens, `qwen35` pre-tokenizer, BOS/PAD 248044, EOS 248046) — as required for speculative acceptance, since the draft's tokens must be the target's tokens.

Where the payoff comes from: [[performance-profile]] records decode rising 40 → 69 tok/s and effective throughput 31.6 → 38.1+ t/s as acceptance improves over a long agent run, and attributes the growth to this model ([[source-state-md]] §1.2). See [[speculative-decoding]] for the technique itself.

## How it works

**Registration and CLI surface.** The speculative type is the string `draft-dflash` ↔ `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH` (`src/common/speculative.cpp:49-52`). Three CLI surfaces touch it:

| Flag | Meaning | Location |
| :--- | :--- | :--- |
| `--spec-draft-model`, `-md`, `--model-draft` | load this GGUF as the draft | `src/common/arg.cpp:4190-4196` |
| `--dflash` | *download* helper: also fetch the `dflash-`-prefixed sidecar from the model repo | `src/common/arg.cpp:3090-3095`; prefix match `dflash-` in `src/common/download.cpp:649-653` |
| `--spec-draft-n-max` | number of tokens to draft (the "max 5" knob) | `src/common/arg.cpp:4121-4129` |
| `--draft`, `--draft-n`, `--draft-max` | **removed** — these now raise an error telling the user to use `--spec-draft-n-max` | `src/common/arg.cpp:4336-4341` |

**Block-diffusion drafting.** The implementation is `common_speculative_impl_draft_dflash` (`src/common/speculative.cpp:1643`). A draft step builds a batch of `[last_token_id, mask × (block_size − 1)]` (`:1977-1979`) and denoises it in place. `block_size` comes from the GGUF key `dflash.block_size` with a hardcoded fallback of 16 (`:1712-1718`) — *this* file ships 8, so the fallback is not what runs. A mask token is mandatory (the comment at `:1728` notes that without one every masked slot is drafted as id −1 and nothing is ever accepted); here it is 248070.

**DFlash2 = selector lattice, not raw logits.** `selector_top_k` is read from the GGUF (`llama_model_dflash_selector_top_k()`, `src/src/llama-model.cpp:3030`, used at `speculative.cpp:1724-1725`) and `is_dflash2 = selector_top_k > 0` — true for this file (16). DFlash2 then "reads its selector lattice from h_nextn and never consumes raw logits" (`speculative.cpp:1789`): the draft's graph packs a `dflash2_lattice` tensor into `res->t_h_nextn` (`build_dflash2_selector`, `src/src/models/dflash.cpp:745+`, assigned at `:823-824`), and the runtime picks it up with `llama_get_embeddings_nextn()` (`src/src/llama-ext.h:106`; used at `speculative.cpp:2006-2008`, with `GGML_ASSERT(lattice && "DFlash2 selector produced no lattice")`).

**There are two DFlash lineages with the same arch string** — the code warns that `general.architecture = dflash` cannot distinguish them and uses the on-disk DSpark Markov head as the marker (`speculative.cpp:1695-1709`). This file carries no Markov head, so it takes the DFlash path.

**Target-layer taps.** The draft consumes hidden states harvested from the *target* model. At construction each id in `target_layers` enables capture of that layer's **input** on the target context:

```c
for (uint32_t k = 0; k < target_layer_ids_n; ++k)
    llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);   // :1786
```

and the draft step reads them back with `llama_get_embeddings_layer_inp(ctx_tgt, target_layer_ids[k])` (`:1879`) after checking `target_layer_ids_n > 0` (`GGML_ASSERT`, `:1693`). The encoder width confirms both numbers: `n_embd_enc = target_layer_ids_n * n_embd_tgt` (`:1693` → 5 × 5120) and the first real tensor is `fc.weight` with shape `(25600, 5120)` = 5 × 5120 — five taps of the target's 5120-wide hidden state (the target's `embedding_length` is 5120, verified for [[ternary-bonsai-2-27b]]).

The converter's own comment explains the indexing convention: `gguf_dspark_to_dflash.py:26` — *"input of layer k+1 is output of layer k"* — i.e. the ids are **layer inputs**, not outputs. The five ids `[6, 20, 34, 48, 62]` sit exactly one below the target's full-attention blocks `7, 21, 35, 49, 63`. [INFERENCE] That is consistent with tapping the hidden state entering five of the target's sixteen KV-bearing layers (its KV layers are `3, 7, 11, … 63`), i.e. every fourth full-attention block; the ids themselves are verified metadata, the correspondence is an interpretation.

**Attention mode.** The draft context is switched to non-causal attention and to the unmasked next-token embedding output: `llama_set_causal_attn(ctx_dft, false)` / `llama_set_embeddings_nextn(ctx_dft, true, /*masked=*/ !is_dflash2)` (`speculative.cpp:1789-1791`). The GGUF says the same thing declaratively: `dflash.attention.causal = False`, with a 2048-token sliding window over 5 blocks of 128-dim, 32-head attention.

**Loader validation.** `src/src/models/dflash.cpp` reads `conv_group_size`, `selector_rank`, `selector_top_k` and requires `target_layers` — *"DFlash model requires 'target_layers' in GGUF metadata"* (`:21-27`) — then, when the selector metadata is present, requires rank, block size, top-k, kernel size and group size all > 0 or throws *"DFlash2 model is missing conv/selector metadata"*, additionally checking `n_embd ≥ top_k × (top_k + 1)` for the lattice (`:208-219`). It then creates the three selector tensors `selector_predecessor`/`selector_successor` `[rank, n_vocab]` and `selector_hidden` `[n_embd, rank]` (`:221-223`) — which match this file's `selector_predecessor [256, 248320]`, `selector_successor [256, 248320]`, `selector_hidden [5120, 256]`.

## Where it lives

**The GGUF is not in the repository.** It was found on this machine at:

| Artifact | Path | Size |
| :--- | :--- | ---: |
| Draft model | `/hdd2/lm-studio-models/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1 143 006 816 B (1.06 GiB) |
| mmproj used by the baseline script (same Qwen3.8-27B family, different publisher) | `/home/ms/Загрузки/Ternary models/mmproj-Qwen3.8-27B-BF16.gguf` → `/hdd2/lm-studio-models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/mmproj-Qwen3.8-27B-BF16.gguf` | symlink |

**No repository script loads it.** All three launch scripts were read in full and none passes `-md` / `--spec-draft-model`, a draft path, or any `--spec-*` flag:

- `scripts/start_server_turbo.sh` and `scripts/run_cli.sh` pass only the target model, `-ctk`/`-ctv`, the four `--triattention-*` flags and standard server/CLI options.
- `scripts/start_server_baseline.sh` additionally passes `--mmproj … --no-mmproj-offload` — the Qwen3.8-27B mmproj above — and no speculative flags.

So the "max 5" drafting configuration described in [[source-state-md]] is **not** reproducible from this repo's launch surface: whichever script or command line enables it lives outside the repository. `[UNVERIFIED]`: the exact invocation, and whether `n_max` was really set to 5 (the flag is `--spec-draft-n-max`, and its default value was not read out of the code).

Implementation files:

- `src/common/speculative.cpp` — `common_speculative_impl_draft_dflash`, block-diffusion step, selector lattice consumption, target-layer capture.
- `src/src/models/dflash.cpp` — `LLM_ARCH_DFLASH` graph builder: encoder, `build_dflash2_selector`, metadata validation.
- `src/src/llama-arch.cpp:143` — architecture key `"dflash"`; `:355,:364-368` — the `target_layers`, `block_size`, `conv_kernel_size`, `conv_group_size`, `selector_rank`, `selector_top_k` keys.
- `src/src/llama-ext.h:106,112,124` — `llama_get_embeddings_nextn()`, `llama_set_embeddings_layer_inp()`, `llama_model_dflash_selector_top_k()`.
- `src/common/arg.cpp` — the three CLI surfaces in the table above.

## Known issues

This model is **not** part of the TA/TQ defect inventories — it appears in [[source-state-md]] only as the explanation for growing decode speed. The risks that are visible:

- **`general.file_type = 15`, which contradicts the tensors.** In `src/ggml/include/ggml.h:455-473`, `15` is `GGML_FTYPE_MOSTLY_IQ2_XXS`, yet the file's 81 tensors are Q4_K/Q6_K/F32 and the filename says `Q4_K_M`. The tensor types are what load; the metadata value is simply wrong or uses a different convention.
- **Acceptance is workload-dependent, and the numbers move with it.** [[performance-profile]] records 40 → 69 tok/s over a run as repetitive agent output warms acceptance up. Any benchmark of this engine that does not state its acceptance profile ([[benchmarks]]) is comparing a moving target.
- **Acceptance collapses when eviction misbehaves.** [[ta-2-budget-starvation]] explicitly notes the acceptance rate plummeting when TriAttention degrades to a sliding window, and [[ta-1-wht-inversion-256]] corrupts which keys survive — so the drafter's throughput is downstream of TriAttention correctness, not independent of it. This is the confound [[performance-profile]] flags in the decode-degradation curve.
- **Removed CLI aliases.** `--draft-max` / `--draft-n` no longer parse (`arg.cpp:4336-4341`); older launch lines (possibly including whichever out-of-repo script sets "max 5") will fail hard rather than degrade.

### Open questions

- **The `target_layers` contract is one-directional.** The draft asserts only that it has *some* target layers (`speculative.cpp:1693`); the EAGLE3 path in the same file additionally range-checks each id against the target's layer count (`:1248-1253`). Whether the DFlash path validates `[6, 20, 34, 48, 62]` against a given target model's 64 blocks was not established here — loading this draft against a different target is therefore unverified, not proven safe.
- **`block_size` mismatch between code and metadata.** The code fallback is 16 (`speculative.cpp:1713`) and this file says 8. Any deployment note or benchmark recorded with block size 16 is not this model's configuration.
- The draft's own KV cache (`key_length = 128`, 8 KV heads, 5 blocks) is never discussed in the sources; whether it is affected by the TurboQuant type dispatch of [[tq-1-missing-gemm-kernels]] depends on which `-ctk`/`-ctv` the draft context is given — unverified, since no in-repo script configures it.

## See also

[[speculative-decoding]] · [[ternary-bonsai-2-27b]] · [[performance-profile]] · [[benchmarks]] · [[source-state-md]] · [[ta-2-budget-starvation]] · [[v100-sxm2]] · [[kv-cache]]
