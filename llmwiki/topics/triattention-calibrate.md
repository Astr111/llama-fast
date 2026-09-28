---
title: TriAttention calibration
type: topic
status: current
updated: 2026-09-28
sources: [TRIATTENTION.md, TRIATTENTION-API.md, README.md]
verified: [src/tools/triattention-calibrate/triattention-calibrate.cpp, src/tools/triattention-calibrate/CMakeLists.txt, src/src/llama-triattention.cpp, src/src/llama-triattention.h, src/src/llama-context.cpp, src/src/llama-kv-cache.cpp, src/src/models/qwen35.cpp, src/common/arg.cpp, src/common/common.h, src/common/common.cpp, src/ggml/include/ggml-cuda.h, calibration/bonsai-27b.triattention, scripts/start_server_turbo.sh]
tags: [triattention, calibration, kv-eviction, tooling]
---

# TriAttention calibration

The offline half of TriAttention: turning a model plus a text corpus into the `.triattention` statistics file that the runtime scorer reads. The mechanism is documented in [[source-triattention]] and [[source-triattention-api]]; this page is about the tool, the artifact, and the gap between the documented workflow and the shipped one.

## Bottom line

- **Two things are called "calibration".** The design document's workflow is `python scripts/calibrate-triattention.py` against a HuggingFace model, validated by `scripts/validate-calibration.py`. **Neither script exists in this repository.** What ships is a C++ binary, `llama-triattention-calibrate` (`src/tools/triattention-calibrate/triattention-calibrate.cpp`, target defined in `src/tools/triattention-calibrate/CMakeLists.txt:1-2`, registered at `src/tools/CMakeLists.txt:21`), which loads a **GGUF** and a **plain-text corpus**. The document describes an intended pipeline; the repository has the tool.
- **The tool's job** is to run the model over the corpus and record, per (layer, attention head), the mean complex query vector and mean magnitude in the RoPE frequency domain — the `E[q_f]`, `‖E[q_f]‖`, `E[‖q_f‖]` statistics the scorer consumes. Decoding fills the KV cache (the context is built normally), but TriAttention itself is never enabled, no pruning hooks run, and the file it writes is the only output.
- **The shipped profile** is `calibration/bonsai-27b.triattention`, 789 571 B, and it parses cleanly against the documented format: `head_dim=256`, 64 layers, 24 attention heads, 4 KV heads, `rope_theta=1e7`, `rope_style=0`, `n_sampled=384`, `freq_count=128`, model name `Bonsai-2-27B-PQ2_0`. 384 = 16 KV-holding layers × 24 heads, i.e. **every full-attention layer of the hybrid model** ([[ternary-bonsai-2-27b]]) — "sampled" means "captured", not "subsampled".
- **The runtime consumes it** through `--triattention-stats` (+ `--triattention-budget`, `--triattention-window`): auto-init in `src/common/common.cpp:1403` → `llama_triattention_init` (`src/src/llama-context.cpp:4354`) → `llama_kv_cache::init_triattention` (`src/src/llama-kv-cache.cpp:2995`) → `triattention_init` (`src/src/llama-triattention.cpp:630`). Validation is `head_dim`, `n_kv_heads` and `freq_count == head_dim/2`, plus a `rope_theta` warning; the model name in the file is never checked.
- **The API contract the document promises is half real.** The parameter list of `llama_triattention_init` matches; its documented return type (`bool`) does not (`int32_t`, `0`/`-1`). The internal `triattention_prune_impl` signature in the document does not match the code at all. Full drift table in [[source-triattention-api]].
- **The tool's own "pre-RoPE" claim is false for the model it ships with.** Its collector comment says it captures the query "before RoPE" (`triattention-calibrate.cpp:60`), but for the `qwen35` architecture of Ternary-Bonsai-2-27B the `Qcur-<il>` tensor is emitted **after** `ggml_rope_multi` (`src/src/models/qwen35.cpp:367-379`). The runtime, meanwhile, inverts RoPE on keys to score in the pre-RoPE basis. That mismatch is the single most important thing on this page.

## Evidence

### CLI surface (verified from the tool's own usage block, `:150-162`)

```
llama-triattention-calibrate -m model.gguf -f corpus.txt -o model.triattention [-c 2048] [-ngl 28] [-t 6]

  -m, --model PATH        path to GGUF model
  -f, --file PATH         path to plain-text calibration corpus
  -o, --output PATH       output path for .triattention calibration file
  -c, --ctx-size N        context size (default: 2048)
  -b, --batch-size N      batch size (default: 512)
  -ngl, --n-gpu-layers N  number of GPU layers to offload
  -t, --threads N         number of CPU threads
```

Behaviour read from `main` (`:168-364`):

- Defaults are set before parsing: `out_file = "model.triattention"`, `n_ctx = 2048`, `n_batch = 512` (`:167-169`). Parsing uses `common_params_parse(..., LLAMA_EXAMPLE_CLI, ...)`, so the whole common flag set is accepted, not just the seven listed — including `-p/--prompt`, which would work as an alternative to `-f` even though the usage text does not say so.
- Both `-m` and the prompt are required: empty prompt → `LOG_ERR("no calibration text provided (use -f FNAME)")` and exit 1 (`:178-181`); empty model path → exit 1 (`:183-186`).
- The corpus is tokenized once, then processed in `ceil(n_tokens / n_ctx)` chunks with `llama_memory_clear(..., true)` between chunks (`:243-273`). So `-c` is a chunk size, **not** an upper bound on corpus size: a `-c 2048` run over a 100K-token corpus still visits all of it, in 49 passes. The document's "~2K tokens" therefore describes the *default corpus size*, not a hard limit.
- A corpus under 128 tokens is a warning, and the code recommends ≥ 1024 (`:238-241`). TriAttention's own KV-cache hooks are never enabled and the memory is cleared between chunks, so the KV type, budget and pruning policy have no effect on the output — although decoding does write the cache as usual.

### What the collector captures

- Hook: `params.cb_eval = triattention_calibrate_cb_eval` (`:191`), the graph-eval callback. Tensors are matched by **name**: `parse_qcur_layer` looks for the substring `"Qcur-"` and parses the layer index after it (`:42-52`). That name exists because the graph callback formats `"%s-%d"` with the layer index (`src/src/llama-context.cpp:2795`).
- Shape gate: `ne[1]` must equal the model's attention-head count and `ne[0]` must be even (`:61-66`); the collected tensor is asserted contiguous `F32`/`F16`/`BF16` (`:75-76`), and non-host tensors are copied back first (`:78-86`).
- Accumulation: for each token and head, for each frequency `k ∈ [0, head_dim/2)`, it reads `re = q[k]`, `im = q[k + fc]`, `mag = sqrt(re² + im²)` and accumulates `sum_real`, `sum_imag`, `sum_abs` (`:118-133`). The pairing is the **half RoPE layout** — the first half of the head is real, the second imaginary — hardcoded, with no interleaved branch.
- At write time, each sampled head's arrays are the per-head means over the accumulated token count, plus `r_f = ‖E[q]‖ / E[‖q‖]` (`:334-347`).

**The "pre-RoPE" claim.** The collector's own comment (`:60`) says "before RoPE", and the progress log repeats it ("collecting pre-RoPE Q statistics", `:246`). That is a property of the *graph*, not of the tool. For the generic builder the `Qcur` callback fires before RoPE, but for the target model's architecture the only tensor named `Qcur-<il>` is emitted after `ggml_rope_multi` (`src/src/models/qwen35.cpp:367` then `:379`; the pre-RoPE tensors are named `Qcur_full`, `Qcur_reshaped`, `Qcur_normed` and do not contain the `Qcur-` substring). So the shipped profile's `E[q_f]` is a **post-RoPE** statistic, while `triattention_invert_rope` (`src/src/llama-triattention.cpp:382`) exists precisely to bring keys back to the pre-RoPE basis for scoring. The profile itself is further evidence for this reading: its 384 sampled pairs are exactly the 16 full-attention layers × 24 heads — only the `qwen35` builder emits `Qcur-<il>` for those layers, so the file was produced through the path where the callback is post-RoPE. Whether the resulting score is merely degraded or meaningless is unresolved — see *Open questions*.

Two further properties of the collection step that the documents do not mention:

- **The model name is a constant.** `const char * model_name = "Bonsai-2-27B-PQ2_0"` (`:309`) is written for every invocation regardless of `-m`, and `name_len = strlen + 1` (`:310`). The name in a profile therefore records which constant the tool was built with, not which model was calibrated. The runtime never compares it: `triattention_init` validates `head_dim` (`:642`), `n_kv_heads` (`:651`) and warns on `rope_theta` mismatch > 1 % (`:659`) only.
- **`rope_style` is hardcoded to `0` (half)** (`:307`) even though the format and the loader support interleaved, and `head_dim` falls back to `256` if no Q tensor was ever seen (`:277`; the tool errors out earlier if none was captured).

### What a `.triattention` profile contains (verified byte-level on `calibration/bonsai-27b.triattention`)

Header, 67 B for this file (`name_len = 19`):

| Field | Type | Shipped value |
| :--- | :--- | :--- |
| `magic` | u32 | `0x54524941` ("TRIA") |
| `version` | u32 | 1 |
| `head_dim` | u32 | 256 |
| `num_layers` | u32 | 64 |
| `num_attn_heads` | u32 | 24 |
| `num_kv_heads` | u32 | 4 |
| `rope_theta` | f64 | 10 000 000.0 |
| `rope_style` | u32 | 0 (half) |
| `n_sampled` | u32 | 384 |
| `freq_count` | u32 | 128 (= head_dim/2) |
| `name_len` | u32 | 19 |
| `name` | char[19] | `Bonsai-2-27B-PQ2_0\0` |

Per sampled head, `n_sampled` times: `layer_idx` u32, `head_idx` u32, `q_mean_real[128]`, `q_mean_imag[128]`, `q_abs_mean[128]`, `r_f[128]` — all f32. Record size 8 + 4·128·4 = 2056 B; total 67 + 384·2056 = **789 571 B**, exactly the on-disk size. The layer indices present are 3, 7, 11, …, 63 with all 24 heads each — the model's 16 full-attention layers.

Notes on the format and its implementation:

- `r_f` is written but **discarded by the loader** — "validation data — not stored at runtime, just skip" (`src/src/llama-triattention.cpp:249-252`). Nothing in the shipped code ever checks it, so the field is currently inert.
- The tool's own success log computes the size as `sizeof(uint32_t)*11 + sizeof(double) + …` (`:361`) while the writer emits **10** u32 fields plus the double, so the reported KB is 4 bytes high. Cosmetic, but it means the tool's arithmetic was not derived from its writer.
- The format has no field for the corpus, token count, timestamp, or tool version. A profile's provenance is therefore not auditable from the file — only its dimensions.
- All three of `head_dim`, `n_kv_heads` and `freq_count` are enforced at load; `rope_style` silently selects the layout used by `triattention_invert_rope`; `n_sampled` sizes `score_buf` (`new float[cal->n_sampled * kv_size]`, `:701`), so an over-populated profile costs scratch memory linearly.

### What the runtime consumes, and what the document commits it to

Consumption chain, all verified:

1. `--triattention-stats PATH` (`src/common/arg.cpp:4690`); `--triattention-budget` and `--triattention-window`/`--triattention-divide-length` (`:4698`, `:4706`) set the policy the loaded statistics feed.
2. Auto-init when the stats path is non-empty **or** budget > 0 (`src/common/common.cpp:1403-1425`); if the stats file is empty the C entry point returns `-1` and the server warns and continues without eviction.
3. `llama_triattention_init` (`src/src/llama-context.cpp:4354`) casts the context memory to `llama_kv_cache` (or extracts the attention half of a hybrid memory) and copies the config.
4. `llama_kv_cache::init_triattention` (`src/src/llama-kv-cache.cpp:2995-3013`) passes `kv_size`, the model's `rope_freq_base_train`, `n_embd_head_k(0)` and `n_head_kv(0)`.
5. `triattention_init` loads, validates, and precomputes `omega` (`:309`), `freq_scale_sq` (`:320`), geometric `offsets` (`:333`), and per-head `q_mean_abs`/`extra_weight` (`:345`). Failure at any of those steps means "no eviction", not "wrong eviction".

The **API contract** the document states is that `llama_triattention_init` takes the 14 parameters, must run after context creation and before inference, and returns a boolean. In code it returns `int32_t` (`0` success, `-1` on any failure — missing stats path, no memory, memory that is not a KV cache), and the parameter is named `normalize_scores`. The commit the runtime actually makes is narrower than the document implies: **it enforces `head_dim`, `n_kv_heads` and `freq_count == head_dim/2`, warns if `rope_theta` differs by more than 1 %, and uses everything else as-is.** Nothing rejects a profile whose statistics were collected wrongly (e.g. post-RoPE, as above) or on a different corpus; nothing cross-checks `num_layers`, `num_attn_heads`, `rope_style` or `n_sampled` against the model.

Two documented behaviours the runtime does *not* implement, both from the design document and both carried by existing issues: the frequency weighting `fscale²_f` (documented as `1/ω²`, implemented as the constant `1.0` — [[ta-5-freq-scale-dead-code]]) and the "evict the lowest-scoring tokens back to the budget" promise, which collapses when the protected prefix plus recent window consume the budget ([[ta-2-budget-starvation]]). The recent window is mandatory and undocumented (`src/src/llama-triattention.cpp:1128-1146`).

### The README's calibration command does not work as written

`README.md` Example 4 invokes:

```
llama-triattention-calibrate -m model.gguf --triattention-calibrate corpus.txt \
    --triattention-calibrate-out my_model.triattention -ngl 99 -c 8192
```

`--triattention-calibrate` and `--triattention-calibrate-out` are registered in `src/common/arg.cpp:4812-4826` (CLI examples only), writing `common_params::triattention_calibrate` / `_out` (`src/common/common.h:750-751`) — **fields that no code reads anywhere in the tree**. The tool's own interface is `-f` and `-o`. With the README's flags and no `-f`, `params.prompt` stays empty and the tool exits with "no calibration text provided (use -f FNAME)" (`:178-181`). `[INFERENCE]` — read from the code, not executed.

> Contradiction (2026-09-28): `README.md` ([[source-readme]]) documents `llama-triattention-calibrate --triattention-calibrate <corpus> --triattention-calibrate-out <out>`; the tool's usage block and `main` (`src/tools/triattention-calibrate/triattention-calibrate.cpp:150-162`, `:167-186`) implement `-f`/`-o` and read neither `--triattention-calibrate*` field. Both the README and the two design documents are ingested here; the dead flags are the likeliest explanation (an interface that was declared in `common/` and never wired to the tool), but which side is intended is unsettled.

### The 40× claim, on this page

The calibration tool's product is one factor in the document's headline "**~40× effective KV memory reduction** (compression × eviction)" ([[source-triattention]], §Overview). No measurement in the wiki supports it: [[benchmarks]] holds only the README's RTX 3090 numbers (~4 000 → ~25 200 tokens/GB, ≈ 6.3×) and **no V100 run at all**. Record it as the document's claim. `[UNVERIFIED]`

## Open questions

- **Does post-RoPE capture break scoring?** The shipped profile was produced through the `qwen35` graph — its 16 sampled layers are exactly that model's full-attention layers — and in that graph the `Qcur-<il>` callback fires after `ggml_rope_multi`; the scorer, by contrast, un-rotates keys and combines them with `E[q_f]`. Either the calibrated statistics are consumed in a basis they were not measured in, or the qwen35 graph re-uses the `Qcur` name for a pre-RoPE view in some variant path this reading missed. Resolving it needs one run of the tool against the model with a print of the tensor's producer/basis, or a profile regenerated from a pre-RoPE path — the highest-value follow-up on this page. `[UNVERIFIED]`
- **Partial MRoPE vs a full-head complex layout.** Ternary-Bonsai-2-27B rotates only 64 of its 256 head dimensions (`qwen35.rope.dimension_count = 64`, sections `[11,11,10,0]`, per [[ternary-bonsai-2-27b]]), while calibration and scoring both assume the full 256-dim head splits into 128 real/imaginary frequency pairs and `omega_f = θ^(−2f/256)`. Whether those two geometries coincide for the first 32 pairs or diverge everywhere is not established here. `[UNVERIFIED]`
- **Provenance of the shipped profile.** The format carries no corpus, no token count, no timestamp and a hardcoded model name, so `calibration/bonsai-27b.triattention` cannot be audited against its inputs. Regenerating it and comparing the two files would settle whether it is current.
- **Are the dead `--triattention-calibrate*` flags the intended interface?** If the intent was for the calibrate tool to be driven by them, then `README.md` is right and the tool needs to read `common_params`; if the tool is right, they should be deleted. Either way `r_f` is written, documented as validation, and never validated.
- **Is the CLI default `offset_max = 0` reachable and harmful?** The design document lists 65536; `src/common/common.h:755` has `0`, and `scripts/start_server_turbo.sh:35` never passes it. `triattention_build_offsets` returns zero offsets for 0, which makes the mean-aggregated score `0 × (1/0)`. `[INFERENCE]`, untested; belongs with [[ta-7-config-validation]].

## See also

[[source-triattention]] · [[source-triattention-api]] · [[triattention]] · [[kv-eviction]] · [[kv-cache]] · [[ternary-bonsai-2-27b]] · [[walsh-hadamard-transform]]
[[ta-1-wht-inversion-256]] · [[ta-2-budget-starvation]] · [[ta-5-freq-scale-dead-code]] · [[ta-7-config-validation]] · [[benchmarks]] · [[codebase-map]]
