---
title: Conversion and Packing
type: topic
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/convert_hf_to_gguf.py, src/conversion/base.py, src/conversion/qwen.py, src/conversion/dspark.py, src/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py, src/gguf-py/gguf/constants.py, src/ggml/include/ggml.h]
tags: [quantization, hadamard, conversion, gguf, tools]
---

# Conversion and Packing

## Bottom line

The artifact the engine consumes is produced by **three disjoint toolchains that never meet in this repository**:

1. **An out-of-tree packer** computes the [[prism-hadamard-weight-fold]] and packs the baked weights as [[prismml-weight-kernels|`PQ2_0`]], then drops a `hadamard_packing.json` manifest beside the HF checkpoint. **This step does not live in `llama-fast`** — nothing in the tree computes a fold, and no in-tree code path can *emit* a `PQ2_0` tensor (see *the boundary* and *type ids* below). The repo can see only the packer's *contract*, never its convention.
2. **The in-repo converter** (`src/convert_hf_to_gguf.py` → `src/conversion/base.py` + ~95 per-architecture modules, incl. `qwen.py`, `dspark.py`) reads the **already-folded, already-packed** HF checkpoint and re-lays it out as GGUF: it *transfers* the manifest into `prism.hadamard.*` metadata, renames/reorders tensors, and emits whatever `raw_dtype` it is told — including passing a pre-packed `PQ2_0` tensor through verbatim. Its own `--outtype` requantizer only knows F16/BF16/Q8_0/TQ1_0/TQ2_0; it has **no `PQ2_0`/`PTQ1_0` encoder**.
3. **The draft path** is a separate translator, `src/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py`, which rewrites a *legacy* `arch=dspark` drafter GGUF into the `dflash` convention by editing metadata and tensor names only — tensor bytes are copied verbatim, never requantized.

Three consequences are worth underlining: the fold convention (left- vs right-multiplication, where the sign vector applies) **cannot be confirmed here** because the producer is out-of-tree and the in-repo consumer only validates the manifest's shape, never its arithmetic; the fork-private type ids `142/143` (= `PQ2_0`/`PTQ1_0`) and the unexplained `general.file_type = 141` bracket a gap the repo papers over by treating the packer's output as a given; and the draft/`dflash` family has grown *two* contradictory file-type stories that no converter in-tree reconciles.

## Evidence

### The in-repo converter relays, renames and (optionally) quantizes — it does not fold

- Entry point is `src/convert_hf_to_gguf.py` (existence confirmed; contents not read here).
- `ModelBase.index_tensors()` reads the checkpoint from local safetensors/pyTorch parts (or remotely), `lazy = True`, exposing name → lazy-tensor callables (`src/conversion/base.py:206-296`).
- `ModelBase.dequant_model()` (`:313-575`) first dequantizes any **HF-side** `quantization_config` scheme — gptq, bitnet, compressed-tensors NVFP4/FP8, etc. — so the GGUF writer sees float tensors. KV/embedding and 1-D or `_norm` tensors are forced to F32 (`:1095-1115`).
- `prepare_tensors()` (`:985+`) picks a per-tensor ggml type: `tensor_force_quant()` hook, then a file-type map that reads **only** `ALL_F32 / MOSTLY_F16 / MOSTLY_BF16 / MOSTLY_Q8_0 / MOSTLY_TQ1_0 / MOSTLY_TQ2_0` and raises `ValueError(f"Unknown file type: {self.ftype.name}")` otherwise (`:1148-1162`), then `gguf.quants.quantize(data, data_qtype)` (`:1165`). **There is no `PQ2_0` arm.** A grep for `PQ2_0|PTQ1_0` across all of `src/conversion/` returns nothing; the only such names in the Python tree are the type-id/block-size constants in `src/gguf-py/gguf/constants.py`. So an in-repo `--outtype` run can never *create* a packed tensor — one can only be relayed through `add_tensor(..., raw_dtype=...)`.
- `write()` (`:1224-1231`) → writes header, KV, tensors. `prepare_metadata()` calls `self.add_hadamard_metadata()` (`:1219`); `set_gguf_parameters()` terminates in `self.gguf_writer.add_file_type(self.ftype)` (`:1536-1537`). Every model module reaches the same `add_file_type(self.ftype)` call (e.g. `qwen.py`, `dspark.py`).

### The boundary: `add_hadamard_metadata()` is a validator + transcriber, not the packer

`ModelBase.add_hadamard_metadata()` (`src/conversion/base.py:635-773`) is the **only** in-tree consumer of the packer's output:

- It looks for `hadamard_packing.json` next to the HF checkpoint and **returns silently if absent** (`:637-639`); a model that *is* folded therefore *must* carry the manifest or the fold is silently dropped from the GGUF.
- It validates the contract's shape only: `schema_version ∈ {1,2,3}`, `kind == "hadamard-weight-fold"`, `status == "requires-matching-runtime"`, transform name `normalized-signed-sylvester-walsh-hadamard`, power-of-two `block_size`, `sign_mode ∈ {identity, explicit}`, sign vectors of exactly the declared width with every value in `{-1, +1}` and width a multiple of `block_size` (`:645-667`), tensor records with `axis == -1` and `role ∈ {fold-before-matmul, inverse-after-lookup}` (`:715-719`).
- It whitelists the architectures the runtime can actually consume the transform on — `{LLAMA, QWEN3, QWEN3MOE, QWEN35, QWEN35MOE, QWEN3NEXT}` (`:683-698`) — and refuses anything else, so a GGUF that "loads but skips the transform" cannot be produced.
- It then re-writes the contract into GGUF keys: `prism.hadamard.version` (1 or 2), `block_size`, `transform` (note: as the *runtime* name `normalized-sylvester-walsh-hadamard`, dropping the `signed` that the manifest carries), `axis = input-last-dimension`, `sign_mode`, `sign_widths`/`sign_values`, `weight_names`, `inverse_weight_names`, and — when `qwen.py` set it — `gdn_v_grouped` (`:746-773`).

**The boundary is `hadamard_packing.json` itself, read at `base.py:637`.** Nothing in this repository, and specifically nothing under `src/gguf-py/` (the string `hadamard` does not occur anywhere under it), ever *writes* that file or computes a weight fold. Consequently the exact fold convention — how the sign vector is applied, left- vs right-multiplication of `H` — and even the fact that the tensors actually *were* folded are **`[UNVERIFIED]` here**; only the agreed runtime contract is recoverable. The runtime side of that contract is documented on [[prism-hadamard-weight-fold]].

### What the converter does *to a folded* checkpoint (the Qwen3.5/`qwenc` path)

`_LinearAttentionVReorderBase` (`src/conversion/qwen.py:446-470`) reorders V heads from **grouped** (by key head, as trained) to **tiled** (ggml broadcast) order so `ggml_repeat` can replace an interleaved repeat. The fold gets in the way of that reorder for `ssm_out`/`linear_attn.out_proj`:

- `_hadamard_folds_tensor(name)` (`qwen.py:568-570`) matches a manifest name allowing only leading wrapper prefixes (`name.endswith(n) or n.endswith(name)`).
- For an **unfolded** `out_proj`, `qwen.py:621-629` reorders its columns (`_reorder_v_heads(..., 1, ...)`).
- For a **folded** `out_proj`, a column permutation on the rotation axis "cannot be refolded", so the converter **keeps the training (grouped) order and lets the runtime permute the activation** instead; it records that with `self._hadamard_gdn_v_grouped = True` (`:549-554`, `:621-626`), which `base.py` turns into `prism.hadamard.gdn_v_grouped = true` (`base.py:769-771`). This is the *only* place the whole pipeline changes the output because of the fold.

The Qwen3.5 archs also always write MRoPE `rope.dimension_sections` (interleaved sections `[11,11,10,0]`) so llama.cpp's required-key validation passes (`_Qwen35MRopeMixin`, `qwen.py:634-648`; [[qwen35-architecture]]).

### The `dflash`/draft path: a legacy `dspark` GGUF → `dflash` re-export

`src/gguf-py/gguf/scripts/gguf_dspark_to_dflash.py` (shipped as `gguf-dspark-to-dflash`) is a self-contained, hand-rolled GGUF reader/writer (struct-based; it does **not** use `gguf.gguf_writer`). Given `<legacy.gguf> <donor-with-tokenizer.gguf> <out.gguf>` it:

- Refuses any input with `general.architecture != "dspark"`; refuses any unmapped `dspark.*` tensor (must be added to `TENSOR_MAP`).
- Rewrites keys: `general.architecture dspark→dflash`, `dspark.<k> → dflash.<k>`, legacy double prefix `dspark.dspark.<k> → dflash.<k>`, `dspark.dspark.mask_token_id → tokenizer.ggml.mask_token_id`; every legacy `tokenizer.*` stub is dropped and replaced by the **donor (target)** tokenizer's keys (`transform_kv()`).
- Renames tensors per `TENSOR_MAP`: `confidence_head.{weight,bias} → conf_proj.*`, `fc.weight → fc.weight`, `hidden_norm.weight → enc.output_norm.weight`, `markov_head_a/b → markov_w1/w2`, `log_snr_fc1/2.* → log_snr_fc1/2.*`.
- **`<arch>.target_layers += 1` on every element** — the module docstring (`:25-28`) explains the indexing convention: the runtime taps a layer's *input* (`llama_set_embeddings_layer_inp`), so the input of layer `k+1` is the output of layer `k`, which is what these drafters were trained against; both reference pairs show `[1,16,31,46,61] → [2,17,32,47,62]`. The HF-side converter applies the identical `+1` (`DFlashModel.set_gguf_parameters`, `qwen.py:730-731`: `extract_layer_ids = [i + 1 for i in target_layer_ids]`), so the convention is consistent in both directions.
- Copies tensor **bytes verbatim** (aligned re-offset; contiguous `shutil.copyfileobj` copy of the whole data section, or per-tensor copies under `--drop-shared-tensors`, which drops `token_embd.weight`/`output.weight` — `TENSOR_NOT_REQUIRED` at runtime; a full-vocab draft borrows the target's embedding and head via `ctx_other`, ~11× smaller with a `Q4_0` repack).
- There are **two in-tree DSpark stories** besides this script: `src/conversion/dspark.py` maps a DSpark checkpoint straight to `arch = DSPARK` (with a `tokenizer.ggml.model = none` + `vocab_size` placeholder trick so batch validation passes without a real vocab, `dspark.py:16-33`), and `qwen.py:922-925` has a `DSparkModel(DFlashModel)` with `arch = DFLASH` for the `Qwen3.6-27B-DSpark` checkpoint. The script above handles the *legacy GGUF* form. See [[qwen3-dflash-draft]] for the runtime side.

### Type ids: what the converter can and cannot emit

- Fork-private tensor types are appended above upstream: `GGML_TYPE_PQ2_0 = 142`, `GGML_TYPE_PTQ1_0 = 143` (Prism-private ternary, group 128), `GGML_TYPE_COUNT = 144` with "slots above upstream types; type_traits is sized to COUNT (144) with 46..141 unused" (`src/ggml/include/ggml.h:437-441`); Python mirrors them (`src/gguf-py/gguf/constants.py:5479-5480`) and knows the block layouts — `QUANT_SIZES`: `PQ2_0 (128, 2+32)`, `PTQ1_0 (128, 2+24+2)` (`constants.py:5675-5676`). A type id known but **no encoder anywhere**: `PQ2_0|PTQ1_0` occurs in `src/gguf-py` only in `constants.py`, and not at all under `src/conversion/`.
- File-type ids: `GGML_FTYPE_MOSTLY_PQ2_0 = 128`, `MOSTLY_PTQ1_0 = 129` (`ggml.h:485-486`; `constants.py:5537-5538`). **141 is defined by no table in this tree** (both the C enum `ggml.h:456-487` and the Python `LlamaFileType` stop at 129), yet [[ternary-bonsai-2-27b]] records `general.file_type = 141` on the target GGUF. Since every in-tree converter emits `add_file_type(self.ftype)` with `ftype` drawn from `LlamaFileType`, `141` **cannot have been written by this repo's converter** — `[INFERENCE]` it came from the out-of-tree packer's own file-type vocabulary. See *Open questions*.

### Contradiction carried from the draft side

> Contradiction (2026-09-28): the draft GGUF records `general.file_type = 15`, which the runtime table maps to `GGML_FTYPE_MOSTLY_IQ2_XXS` (`ggml.h:471`), while its tensors are actually `Q4_K`/`Q6_K`. The mapping half is verified here; the tensor-type half is a sibling's reading of the draft file, `[UNVERIFIED]` from this repo. No in-tree converter explains the pairing, and `gguf_dspark_to_dflash.py` — the tool that re-exports draft GGUFs — copies tensor types through untouched, so nothing in this pipeline would ever correct it. See [[qwen3-dflash-draft]].

## Open questions

- **`general.file_type = 141` on the target GGUF.** Outside both the C and Python file-type tables (max `129`); the in-repo converter provably cannot write it (`add_file_type(self.ftype)` with `ftype ∈ LlamaFileType`). Is it a stale packer-side id, a deliberate sentinel, or the *actual* marker the runtime keyed on? See [[ternary-bonsai-2-27b]].
- **The fold convention.** Which side `H` multiplies (left vs right) and where the sign vector applies at pack time cannot be confirmed here; the only verifiable artifact is the runtime contract transcribed by `base.py:635-773` from an out-of-tree `hadamard_packing.json`. `[UNVERIFIED]` — see [[prism-hadamard-weight-fold]].
- **Who packs `PQ2_0`?** No encoder exists in `src/gguf-py` or `src/conversion`; the tensors must arrive pre-packed from the out-of-tree packer. The in-repo converter can relay them via `raw_dtype` but cannot prove their contents.
- **Why does `add_hadamard_metadata` whitelist exclude `DFLASH`/`DSPARK`?** A folded *draft* could not be produced in-repo even in principle; whether the runtime's draft head handles the transform is covered on [[qwen35-architecture]].
- **`PTQ1_0` (143) exists but is used by no artifact in this project** — no file type, no code path, no tensor in the vault's GGUF. Dead id or future path? `[UNVERIFIED]` in both directions.

## See also

[[prism-hadamard-weight-fold]] · [[prismml-weight-kernels]] · [[quantization]] · [[ternary-bonsai-2-27b]] · [[qwen3-dflash-draft]] · [[qwen35-architecture]] · [[speculative-decoding]] · [[codebase-map]]