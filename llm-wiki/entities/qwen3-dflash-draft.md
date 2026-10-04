---
title: Qwen3.8-27B-DFlash2 draft model
type: entity
status: current
updated: 2026-09-29
sources: [state.md, README.md]
verified: ["/hdd2/lm-studio-models/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf", src/common/speculative.cpp, src/common/arg.cpp, src/src/models/dflash.cpp, src/src/models/dspark.cpp, src/src/models.h, src/src/llama-arch.cpp, src/src/llama-arch.h, src/src/llama-model.cpp, src/src/llama-model.h, src/src/llama-hparams.h, src/src/llama-ext.h, src/src/llama-context.cpp, src/src/llama-context.h, src/src/llama-graph.cpp, src/src/llama-graph.h, src/common/common.h, src/common/download.cpp, src/common/preset.cpp, src/ggml/include/ggml.h, src/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py, src/gguf-py/gguf/constants.py, src/gguf-py/gguf/tensor_mapping.py, src/gguf-py/pyproject.toml, src/conversion/qwen.py, src/conversion/deepseek.py, src/conversion/muse_glimmer.py, src/README.md, src/docs/speculative.md, scripts/start_server_turbo.sh, scripts/start_server_baseline.sh, scripts/run_cli.sh]
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

## The draft architecture: `models/dspark.cpp`

**This file is the architecture the drafter belongs to — but not the loader this page's GGUF goes through.** `src/src/models/dspark.cpp` (573 lines) implements the arch id `dspark`: registry `src/src/llama-arch.cpp:43` (`{ LLM_ARCH_DSPARK, "dspark" }`, enum at `llama-arch.h:48`) → `src/src/llama-model.cpp:338-339` (`case LLM_ARCH_DSPARK: return new llama_model_dspark(params);`, class declared at `src/src/models.h:595`). This page's GGUF declares `general.architecture = dflash`, which is a *different* enum: `llama-arch.cpp:143` (`{ LLM_ARCH_DFLASH, "dflash" }`) → `llama-model.cpp:328-329` → `llama_model_dflash` → `src/src/models/dflash.cpp:25`. **Verdict: `dflash` is the arch this fork loads for the project's drafter; `models/dspark.cpp` is a sibling on a legacy arch id that this file never enters.** Both ids are live, registered code — only the arch string in the GGUF decides which one runs.

### The competing conversion entry points, closed

| Entry point | Emits | Evidence |
| :--- | :--- | :--- |
| `qwen.py` `DFlashModel` (the `qwen35`-derived drafter family) | arch `DFLASH` | `src/conversion/qwen.py:664-665` |
| `qwen.py` DFly, and DSpark ("DFlash + a semi-autoregressive Markov head") | arch `DFLASH` | `src/conversion/qwen.py:795`, `:925` |
| `deepseek.py` `DeepseekV4DSparkModel` | arch `DFLASH` | `src/conversion/deepseek.py:922-923` |
| `muse_glimmer.py` `MuseGlimmerAssistantModel` | arch `DFLASH` | `src/conversion/muse_glimmer.py:138` |
| `gguf_dspark_to_dflash.py` (console script `gguf-dspark-to-dflash`, `src/gguf-py/pyproject.toml:27`) | `dspark` → `dflash` | docstring `:1-14`; refuses non-`dspark` input at `:178-179` |

No current converter emits `general.architecture = dspark`. The arch-`dspark` id exists only to read GGUFs written by older releases, and `src/README.md:14` says so outright: *"Drafters published for older model releases need a one-time conversion with `gguf-dspark-to-dflash` … newer releases ship ready-to-use drafters."* So of the three entry points that looked like rivals in [[conversion-and-packing]], two are the same thing (current converters, arch `dflash`) and the third is a migration tool for a deprecated arch string.

The runtime agrees with the loader: for `--spec-type draft-dspark`, the impl is chosen by re-reading the arch string — `if (std::string(arch) == "dspark")` → `common_speculative_impl_draft_dspark` (the `models/dspark.cpp` path), otherwise the DFlash impl is constructed *with* the DSpark type (`src/common/speculative.cpp:3348-3359`). Inside arch `dflash`, the Markov head is the lineage marker (`llama-ext.h:130-132`; `speculative.cpp:1695-1706`; auto-detected from the tensor `markov_w1.weight` at `:3084-3087`).

### What a DSpark drafter computes

- **Its own layers.** The trunk is the drafter's own `n_layer` dense Qwen3-style stack: `for (int il = 0; il < n_layer; ++il)` (`dspark.cpp:334`), attn RMSNorm + `build_qkv` + q/k norm + RoPE + `build_attn` over a **real persistent KV cache** (`:380-393`, with `llama_set_causal_attn(ctx_dft, false)` at `speculative.cpp:420` making the mask fully open), then FFN SiLU. RoPE family is pinned NEOX (`llama-model.cpp:3250-3251`). The target's hidden states never pass through a layer's `attn_norm`/FFN — they are projected once and re-projected per layer as attention K/V.
- **Plus taps into the target's hidden states**, but only as context K/V. The staged tap window (`dspark_ctx_feat`, width `n_embd_cap = n_dspark_target_layers * n_embd`, `dspark.cpp:178,231`) goes through `dspark_fc` (Linear) then `dspark_hidden_norm` (RMSNorm) **once per call** (`:241-244`); that tensor is concatenated with each layer's normed draft residual and passed through that layer's own q/k/v projection in a single `build_qkv` (`:351-354`). The comment at `:340-346` justifies it: `nn.Linear` has no cross-row terms, so `k_proj(concat(A,B)) == concat(k_proj(A), k_proj(B))`.
- **Tensors existing solely for the taps: exactly two** — `dspark.fc` `{n_capture*n_embd, n_embd}` and `dspark.hidden_norm` `{n_embd}` (`dspark.cpp:99-100`; GGUF names at `llama-arch.cpp:686-687`, renamed by the converter to `fc.weight` / `enc.output_norm.weight`, `gguf_dspark_to_dflash.py:19-20`). The layer ids themselves cost no tensors: the *count* is baked into `fc`'s shape, the ids are metadata only.
- **The other extras are head/conditioning, not taps:** Markov head `{markov_rank, n_vocab}` ×2 (`:118-121`, weights returned as host floats for a host-side resample, `llama-ext.h:213-225`), `confidence_head` `{n_embd (+rank), 1}` + bias (`:125-128`), `mode_embedding` bias (`:131-132`), hidden-correction `corr_gate`/`corr_up` `{2*n_embd, width}` + `corr_down` `{width, n_embd}` + two norms (`:110-112`), GIDD log-SNR `fc1 {128, n_embd}` / `fc2 {n_embd, n_embd}` (`:145-148`).
- **`h_nextn` is a first-class output of this graph** (`res->t_h_nextn = cur`, `:417-418`) — the pre-final-norm trunk state. That is the hook DFlash2 packs its selector lattice into on this page's file, and the hook the DSpark path reads a confidence head from.

The tap width here is the same arithmetic this page derived from the shipped file: `n_capture × n_embd = 5 × 5120 = 25600`, the first dimension of `fc.weight` `(25600, 5120)` (`dspark.cpp:63,99` vs the GGUF read above). **`fc.weight` in a DFlash-format file *is* `dspark.fc.weight`** — the converter renamed it; the architecture is the same.

### What `target_layers` selects

| Aspect | Key / symbol | Under `dflash` (this file) | Under legacy `dspark` |
| :--- | :--- | :--- | :--- |
| metadata key | `llama-arch.cpp:355` / `:222` | `dflash.target_layers` | `dspark.dspark.target_layers` (the "double prefix") |
| ids → which layers captured | `speculative.cpp:1785-1786`, `llama-context.cpp:1323-1328` | `llama_set_embeddings_layer_inp(ctx_tgt, id, true)` | staged through `llama_set_dspark_ctx` |
| count → projection width | `dspark.cpp:62`, `llama-model.cpp:3438` | `n_embd_inp_enc_impl = n × n_embd` (`dflash.cpp:29`) | `n_capture`, reported as `llama_dspark_meta.n_capture` |

The ids index the **target's** layer *inputs*, not this drafter's layers: `llama_set_embeddings_layer_inp` flags "the input embeddings of a specific layer" (`llama-ext.h:111-112`), and the context sizes its flag array `n_layer() + 1` precisely so `lid == n_layer()` means the last layer's output — "input of the head" (`llama-context.cpp:206-208`).

This page's reading of the converter comment holds and generalizes: the **+1 shift is applied by the current converters too** (`qwen.py:727-730`, `muse_glimmer.py:171-173`), with the same stated justification — *"`dflash.target_layers[k]` refers to the inputs going into the ith layer, which come from the (i-1)th layer's output. The transformers configuration refers to the outputs being recorded."* So the stored `[6, 20, 34, 48, 62]` are input-of indices, and the source checkpoint's recorded-output indices are `[5, 19, 33, 47, 61]`. [INFERENCE]

The trunk's tensor set is the plain dense Qwen3-style block — the same shape family as [[qwen35-architecture]] — but none of the target's variant machinery carries over: `llama-model.cpp:3248-3251` pins `LLAMA_ROPE_TYPE_NEOX` with the comment that the drafter's RoPE is *"independent of the target's RoPE family"*, so the attention/RoPE differences catalogued in [[qwen35-variants]] do not apply to the draft.

### Can the speculative loop load this draft against the target it ships beside? Yes — the check exists, one call deeper

`speculative.cpp:1693` really does assert only `target_layer_ids_n > 0`. The check this page was looking for is in the *target* context, not in the speculative loop:

```c
GGML_ASSERT(lid <= model.hparams.n_layer());   // llama-context.cpp:1326
```

with the flag array sized `n_layer() + 1` (`:207`). `GGML_ASSERT` is unconditional and hard-aborting in this tree (`src/ggml/include/ggml.h:288`), so a mismatch fails at speculative-implementation construction — not at load time, and not silently. For the pair shipped here the ids pass: `[6, 20, 34, 48, 62]` requires `n_layer_tgt ≥ 62`, and the target of [[ternary-bonsai-2-27b]] has 64 blocks. The ids are therefore constrained only as a *bound*; nothing checks that the tapped layers are semantically the right ones for a different target.

The second quantity that must line up is the tap width: the loop computes `n_embd_enc = target_layer_ids_n × n_embd_tgt` (`speculative.cpp:1710`) while the draft's own loader derives its expected encoder input from *its own* metadata (`dflash.cpp:29`); both are `5 × 5120 = 25600` here. [INFERENCE] A different target whose `n_embd` disagreed with the draft's `fc.weight` row count would surface as a shape failure at the first decode, not at load.

**A third DSpark lives under arch `dflash`, and it is not this page's model.** `hyper_connection_count > 0` (read at `dflash.cpp:70`) selects DeepSeek-V4 DSpark *stages*: `graph_dsv4` (`dflash.cpp:352-355`), MLA-style single-K cache with ISWA → `llama_kv_cache_iswa` (`llama-model.cpp:2723-2729`), RoPE `NORM` instead of `NEOX` (`:3236-3237`), and the `is_dsv4` reclassification at `:372-373`. `dsv4_hc_mult` appears nowhere in `dspark.cpp`, so the arch-`dspark` path implements only the vanilla trunk and DSV4 drafters must go through `dflash.cpp` whatever they are called.

> **Correction (2026-09-29).** The open question above — *"Whether the DFlash path validates `[6, 20, 34, 48, 62]` against a given target model's 64 blocks was not established here … loading this draft against a different target is therefore unverified, not proven safe"* — resolves to: **it is checked, as a bound.** `llama-context.cpp:1326` asserts `lid <= n_layer_tgt` for every DFlash tap, so loading this draft against a target with fewer than 63 blocks aborts at spec-impl construction. What stays unverified is the semantic match, and the tap width, which is only implicitly checked (see above). A second correction: the page's "two DFlash lineages with the same arch string" *understates* the count — under `dflash` there are at least three (vanilla/DFlash2, DSpark-with-Markov-head, DeepSeek-V4 DSpark stages), and the separate legacy arch id `dspark` adds a fourth code path, `models/dspark.cpp`, which no current converter produces.

## See also

[[speculative-decoding]] · [[ternary-bonsai-2-27b]] · [[performance-profile]] · [[benchmarks]] · [[source-state-md]] · [[ta-2-budget-starvation]] · [[v100-sxm2]] · [[kv-cache]]
