---
title: Speculative decoding (draft-dflash)
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: ["/hdd2/lm-studio-models/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf", src/common/speculative.cpp, src/common/common.h, src/common/arg.cpp, src/docs/speculative.md, src/src/llama-ext.h, src/src/llama-model.cpp, src/src/llama-context.cpp, src/tools/server/server-context.cpp, src/ggml/src/ggml-cuda/ggml-cuda.cu, scripts/start_server_turbo.sh, scripts/run_cli.sh]
tags: [speculative-decoding, dflash, throughput, block-diffusion]
---

# Speculative decoding (draft-dflash)

## What it is

The project's **throughput mechanism**: a small draft model proposes a whole *block* of tokens, the target model verifies them in one forward pass, and the accepted prefix is committed. Configured in [[source-state-md]] *Key Optimizations* as "Speculative Decoding (`draft-dflash`, max 5)", supplied by [[qwen3-dflash-draft]], and recorded in [[source-state-md]] §1.2 as the sole traced cause of a decode speed curve that **rises** during a run.

`draft-dflash` is the DFlash **block-diffusion** lineage: unlike EAGLE-3 (a single-layer autoregressive drafter emitting one token per step) the DFlash draft uses several transformer layers but emits an entire block per draft step, and it is **seeded with hidden states extracted from the target model** rather than being a small standalone LM. The in-tree description (`src/docs/speculative.md:55-63`):

> "DFlash produces an entire block of draft tokens in a single forward pass (block diffusion) and injects the target model's hidden states into the draft model's attention, instead of drafting one token at a time. This keeps the draft model small while making drafting GPU-friendly."

## How it works

### The type and its registration

| Item | Value | Where |
| :--- | :--- | :--- |
| Name string | `"draft-dflash"` | `src/common/speculative.cpp:50` |
| Enum | `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH` | `src/common/common.h:176` |
| Implementation | `struct common_speculative_impl_draft_dflash : public common_speculative_impl` | `src/common/speculative.cpp:1643` |
| Model architecture | `LLM_ARCH_DFLASH` → `new llama_model_dflash(params)` | `src/src/llama-model.cpp:328-329` |
| Requires a companion context | `if (model.arch == LLM_ARCH_EAGLE3 \|\| model.arch == LLM_ARCH_DFLASH)` demand `ctx_other` | `src/src/llama-context.cpp:236` |
| Selector enquiry (DFlash2 lineage) | `llama_model_dflash_selector_top_k()` | `src/src/llama-ext.h:124`, `src/src/llama-model.cpp:3030` |
| Sibling type from the same family | `"draft-dspark"` (Markov head) | `src/common/speculative.cpp:51` |

Construction (`:1679-1792`) reads the drafter's own metadata rather than trusting the requested type: `target_layer_ids` / `target_layer_ids_n` (which target layers feed the draft), `n_embd_dec`/`n_embd_enc`/`n_embd_tgt`, the **trained block size** from the `dflash.block_size` key with a hard fallback of `16` (`:1712-1713`), `dflash.sample_from_anchor` (`:1719-1721`), the selector (`selector_top_k > 0 ⇒ is_dflash2`, `:1724-1725`), and the tokenizer's mask id (`mask_token_id`, asserted non-null, `:1726-1731`). It also warns when the requested `--spec-type` disagrees with the on-disk lineage (`:1695-1710`).

### Drafting: one decode of a masked block

`draft()` (`:1952+`) builds, per sequence, a block of `n_draft + (is_dspark && sample_from_anchor ? 0 : 1)` tokens — the current token followed by `<mask>` filler — and decodes the **whole block in a single `llama_decode`** on `ctx_dft`:

```cpp
const int32_t n_draft = params.n_max;
const int32_t n_block_tokens = n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
...
for (int32_t i = 0; i < n_block_tokens; ++i) {
    common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, !is_dflash2);
}
...
// decode all sequence's noise block in a single batch
int ret = llama_decode(ctx_dft, batch);
```

(`:1971-1990`.) That single batched decode is the point of the design: drafting costs one forward pass regardless of block length, on a model small enough to fit beside the target. The DFlash2 variant then reads a "lattice" from `llama_get_embeddings_nextn(ctx_dft)` and walks it with an argmax over `selector_top_k` scores, carrying a `predecessor` state across block positions — a cheap in-GPU sampling chain (`:2006-2030`), consistent with `llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2)` at `:1790`.

The target's hidden states reach the draft through a separate injection batch (`batch_inject`, sized `n_embd_dec`, `:1755-1756`) populated in `process()` (`:1832+`), which hooks the target's own batch and asserts contiguity per sequence while doing so.

### Interaction with the server

The server caps each slot's draft length by context headroom, then pushes the request into the speculative engine:

```cpp
const int n_draft_max = slot.get_n_draft_max();     // n_ctx - prompt - 2, clipped by n_remaining - 1
...
common_speculative_get_draft_params(spec.get(), slot.id) = {
    /* .drafting = */ true,
    /* .n_max    = */ n_draft_max,
    /* .n_past   = */ slot.prompt.n_tokens(),
    /* .id_last  = */ slot.sampled, ...
```

(`src/tools/server/server-context.cpp:449-457` for `get_n_draft_max()`, `:2918-2944` for the handoff; `common_speculative_draft()` is then called once per round at `:2959-2964`.) Acceptance telemetry is reported per position: `n_draft_accepted / n_draft_total`, `mean len = 1 + accepted/verif_steps`, and an `acc per pos` series (`:615-636`, buffer sized from `common_speculative_n_max()` at `:3894`; the helper is `src/common/speculative.cpp:3102-3135`).

### The flags, and the "max 5"

| Flag | Default | Source |
| :--- | :--- | :--- |
| `--spec-type draft-dflash` | — (needed unless inferred from the drafter's sidecar) | `src/common/speculative.cpp:50`, `src/common/arg.cpp:559-575` |
| `--spec-draft-n-max N` (env `LLAMA_ARG_SPEC_DRAFT_N_MAX`) | **3** | `src/common/arg.cpp:4121-4130`; default in `src/common/common.h:326` |
| `--spec-draft-n-min N` | 0 | `src/common/arg.cpp:4131-4139`; `common.h:327` |
| `--spec-draft-p-min P` / `--draft-p-min` | 0.0 | `src/common/arg.cpp:4146-4151`; `common.h:331` |
| `--spec-draft-p-split P` | 0.1 | `src/common/arg.cpp:4140-4145`; `common.h:330` |
| `--dflash` (download the sidecar only) | off | `src/common/arg.cpp:3089-3094` |
| `--draft`, `--draft-n`, `--draft-max` | **removed** — hard error pointing at the new names | `src/common/arg.cpp:4336-4341` |

**There is no constant `5` anywhere in this path.** The recorded "max 5" is a *configuration choice*, not a code default, and it is not derivable from the defaults: the code default is 3 (`common.h:326`), and the DFlash clamp is `block_size - 1` — i.e. **7** for the project's drafter. That drafter's own header was read directly and declares `dflash.block_size = 8` (with `dflash.block_count = 5`, `dflash.selector_rank = 256`, `dflash.selector_top_k = 16`, the last confirming the DFlash2 lineage via `selector_top_k > 0`): `/hdd2/lm-studio-models/z-lab/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`. `5` is inside the legal range and produces exactly the shape described (one anchor + 5 masks). The clamping code (`:1739-1746`):

```cpp
// DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
// block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) { /* warn + clamp */ }
```

so any `--spec-draft-n-max` above 7 for this drafter is silently reduced with a warning, matching the documentation's "`--spec-draft-n-max` is clamped to the draft model's trained block size" (`src/docs/speculative.md:74`).

### The recorded behaviour: throughput rises as context warms

[[source-state-md]] §1.2, verbatim in substance — this is the finding the page exists to carry:

> "**Decode Speed Growth (40 → 69 tok/s):** Traced to speculative decoding (`draft-dflash`). As the agent generates repetitive CLI/code patterns, context warms up and acceptance rates grow, shifting effective throughput from **~31.6 t/s to ~38.1+ t/s**." ([[source-state-md]] §1.2)

The mechanism is the one the acceptance statistics above measure: on repetitive CLI/code output the drafter's block guesses land more often, more of each verified block is committed per target forward pass, and effective tokens-per-second rises even though the per-round cost is roughly fixed. The corollary recorded in the same source is that this curve is **fragile**: [[ta-2-budget-starvation]] states that when TriAttention's budget collapses the history on long prefixes, the "speculative decoding acceptance rate plummets" — the throughput growth depends on the evicted context still containing the patterns the drafter is predicting.

The figures `40 → 69 tok/s` and `31.6 → 38.1+ t/s` are a measurement from the user's profiling, **`[UNVERIFIED]` against this tree**; the code can neither confirm nor refute them. The behaviour *shape* (acceptance rate is the lever) is at least structurally consistent with the design: drafting cost is one batched decode (`:1980`), so the only variable that moves effective throughput is how many drafted tokens survive verification.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/common/speculative.cpp` | type table `:47-51`; stats fields incl. `n_acc_tokens_per_pos` `:156-164`; abstract `begin`/`draft` `:185-189`; `common_speculative_impl_draft_dflash` `:1643-2100+` (ctor `:1679`, metadata `:1711-1728`, clamp `:1739-1746`, batches `:1755-1758`, `begin` `:1811`, `process` `:1832`, `draft` `:1952`, block build `:1971-1978`, single decode `:1980-1990`, DFlash2 lattice `:2006-2030`); `common_speculative_n_max()` `:3102-3135`; dispatcher `:3466` |
| `src/common/common.h` | enum `:176`; `common_params_speculative_draft` `n_max=3`/`n_min=0`/`p_min`/`backend_sampling` `:325-332`; `need_n_rs_seq()` gates the RS sequence on `draft.n_max` `:389-393` |
| `src/common/arg.cpp` | `--spec-draft-n-max` `:4121`, `-n-min` `:4131`, `p-split` `:4140`, `p-min` `:4146`; sidecar inference `:559-575`; `--dflash` `:3089-3094`; removed legacy flags `:4336-4348` |
| `src/docs/speculative.md` | DFlash section `:55-80` (incl. the `--spec-draft-n-max 15` example), DFly variant `:83-100`, `--spec-draft-n-max` tuning table `:104-116`, DSpark `:123-152`, `--spec-type` reference `:388-400` |
| `src/tools/server/server-context.cpp` | `get_n_draft_max()` `:439-457`; draft-param handoff `:2918-2944`; `common_speculative_draft()` call `:2959-2964`; acceptance telemetry `:615-636`, `:3894` |
| `src/src/llama-model.cpp`, `src/src/llama-ext.h`, `src/src/llama-context.cpp` | arch dispatch `:328-329`, `:2723`; selector accessor `:3030` / `:124`; companion-context requirement `:236-244` |
| `src/ggml/src/ggml-cuda/ggml-cuda.cu` | graph key includes the batch shape, so the draft block's width is part of what decides whether a captured graph can be replayed (`ggml_cuda_graph_update_required` `:2597-2637`) — see [[cuda-graphs]] |

## Known issues

- [[ta-2-budget-starvation]] — the recorded coupling: eviction starvation on long prefixes makes acceptance "plummet", which is the failure mode of the throughput curve above. This is the only issue in the inventory that names speculative decoding in its impact.
- [[ta-1-wht-inversion-256]] — independent of the drafter, but it degrades the target's own output quality while the throughput curve looks healthy; a run can show rising acceptance and worsening generation at the same time.
- No issue page tracks the draft path itself: no defect was filed against `speculative.cpp` in either inventory ([[source-state-md]] §3, §4).
- Configuration hazard, verified: the legacy `--draft-max` family is a hard `arg_removed()` error rather than an alias (`src/common/arg.cpp:4336-4348`), while the current name is `--spec-draft-n-max`. Any launch script or habit carried over from older llama.cpp fails the argument parse. Neither `scripts/start_server_turbo.sh` nor `scripts/run_cli.sh` passes any speculative flag at all, so the shipped launch profiles run **without** the drafter described here — a discrepancy between the recorded configuration and the scripts on disk.
- Upstream's own doc warns the optimum is interior and hardware-dependent: for a block-8 drafter on an M5 Pro, `--spec-draft-n-max` 5 / 6 / 7 gave 35.7 / 42.0 / 37.5 tok/s at 57.5 % / 64.9 % / 52.7 % acceptance and 4.00 / 4.99 / 4.82 committed tokens per round (`src/docs/speculative.md:110-115`). That table is **not** this project's hardware or drafter and must not be read as evidence about the V100; it is cited because it is the in-tree statement that "sweep it rather than assuming the largest value wins" (`:116`), which is directly relevant to the recorded choice of 5.

## Open questions

- **Where does "max 5" come from?** No source in the repository states it: not a default (3), not the clamp (7), not the doc example (15). It is a tuning decision recorded in prose ([[source-state-md]] *Key Optimizations*) whose justification is unrecorded. Was it tuned on the V100, or inherited? `[UNVERIFIED]`.
- The draft block width is part of the captured CUDA graph — `ggml_cuda_graph_update_required()` compares node shapes, strides and source pointers (`src/ggml/src/ggml-cuda/ggml-cuda.cu:2597-2637`), so a change in effective `n_max` (or in the server's per-slot `get_n_draft_max()`, which shrinks as the context fills) changes the batch shape and can force a graph re-capture. Whether that happens in practice, and how often, is `[UNVERIFIED]` — it needs a run with `GGML_LOG_DEBUG` on the graph path.
- `--spec-draft-n-min` default 0 means "no minimum"; with `n_min > 0` a partially-matching block below the floor is discarded. No source states whether the project used a non-zero minimum. `[UNVERIFIED]`.
- Acceptance is reported per position by the server behind a stats flag; no archived run output from this project is present in the tree, so the 31.6 → 38.1+ curve cannot be re-derived here. `[UNVERIFIED]`.

## See also

[[qwen3-dflash-draft]] · [[ternary-bonsai-2-27b]] · [[performance-profile]] · [[benchmarks]] · [[overview]] · [[kv-cache]] · [[sampling]] · [[cuda-graphs]] · [[ta-2-budget-starvation]] · [[v100-sxm2]] · [[upstream-lineage]] · [[source-state-md]] · [[source-readme]]
