---
title: PrismML Hadamard Weight Fold
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: ["/home/ms/Загрузки/Ternary models/Ternary-Bonsai-2-27B-PQ2_0.gguf", src/conversion/base.py, src/conversion/qwen.py, src/docs/development/hadamard-tied-output.md, src/docs/kv-mean-center.md, src/calibration_corpus.txt, src/tests/test-backend-ops.cpp, src/ggml/include/ggml.h, src/ggml/src/ggml-common.h, src/ggml/src/ggml.c, src/ggml/src/ggml-cpu/ggml-cpu.c, src/ggml/src/ggml-cpu/ggml-cpu.cpp, src/ggml/src/ggml-cpu/ops.cpp, src/ggml/src/ggml-cuda/ggml-cuda.cu, src/ggml/src/ggml-cuda/fwht.cu, src/ggml/src/ggml-cuda/fwht.cuh, src/ggml/src/ggml-blas/ggml-blas.cpp, src/ggml/src/ggml-sycl/fwht.hpp, src/ggml/src/ggml-metal/ggml-metal-ops.cpp, src/ggml/src/ggml-vulkan/ggml-vulkan.cpp, src/src/llama-impl.h, src/src/llama-model.cpp, src/src/llama-model.h, src/src/llama-graph.cpp, src/src/llama-graph.h, src/src/llama-context.cpp, src/src/llama-kv-cache.cpp, src/src/llama-kv-cache.h, src/src/models/qwen35.cpp]
tags: [quantization, hadamard, weights, model]
---

# PrismML Hadamard Weight Fold

## What it is

A **checkpoint-time, blockwise, sign-suffixed Walsh–Hadamard mixing of the model's own weight matrices**, baked into `Ternary-Bonsai-2-27B-PQ2_0.gguf` and declared by that file's `prism.hadamard.*` metadata block. The weights are *stored* in the Hadamard basis; the runtime never un-folds them, it applies the same transform to the **activation** immediately in front of each folded matmul (and, for the embedding table, immediately *after* the row lookup).

It is **not** TurboQuant's KV-cache rotation. That is a per-token rotation of cached keys and values ([[walsh-hadamard-transform]], [[turboquant]]), whereas this fold is a static property of the weights, present before the process starts, tied to no cache type, and unaffected by any CLI flag. Telling these two apart is this page's reason to exist — see *Three Walsh–Hadamard paths* below.

Neither [[source-state-md]] nor [[source-readme]] mentions the fold. It was found by reading the GGUF header, and the file labels itself: `general.basename = folded` (`general.name = Hf`, `general.size_label = 27B`). The same block is why the file loads only on the architectures listed in the loader's whitelist, and [[ternary-bonsai-2-27b]] carries the summary of it.

### The contract, as the file itself declares it

Read directly from the GGUF (all values below are file contents, not prose):

| Key | Value |
| :--- | :--- |
| `prism.hadamard.version` | `1` |
| `prism.hadamard.transform` | `normalized-sylvester-walsh-hadamard` |
| `prism.hadamard.axis` | `input-last-dimension` |
| `prism.hadamard.block_size` | `1024` |
| `prism.hadamard.sign_mode` | `explicit` |
| `prism.hadamard.sign_widths` | `[5120, 6144, 17408]` |
| `prism.hadamard.sign_values` | 28 672 × ±1 (exactly `5120 + 6144 + 17408`) |
| `prism.hadamard.weight_names` | 401 names |
| `prism.hadamard.inverse_weight_names` | `["token_embd.weight"]` |
| `prism.hadamard.gdn_v_grouped` | `true` |
| `prism.hadamard.tied_output` | *absent* |

`version = 1` plus a present `output.weight` tensor (verified: `output.weight` is a 337 715 200-byte `PQ2_0` tensor of shape `[5120, 248320]`, distinct from `token_embd.weight`) means the tied-output variant — `version 2`, `tied_output = true`, no `output.weight` (`src/docs/development/hadamard-tied-output.md:5-10`) — is *not* in use here.

### Which matrices are folded: 401 of the file's 851 tensors

Counted from `prism.hadamard.weight_names` and cross-checked against the tensor list:

| Contraction width (`ne[0]`) | Tensors | Kinds |
| :--- | ---: | :--- |
| 5120 (`n_embd`) | 273 | `output.weight` (1); `attn_q`, `attn_k`, `attn_v` (16 each); `attn_qkv`, `attn_gate` (48 each); `ffn_gate`, `ffn_up` (64 each) |
| 6144 | 64 | `ssm_out` (48, the Gated-DeltaNet blocks of [[qwen35-architecture]]); `attn_output` (16) |
| 17408 (`n_ff`) | 64 | `ffn_down` (64) |

Three facts fall out of that table and are worth stating because nothing else in the repo states them: the three `sign_widths` are *exactly* the three distinct contraction widths of the folded set; every folded tensor is stored as `PQ2_0` ([[prismml-weight-kernels]]); and the fold covers **every** weight that multiplies the residual stream or the FFN intermediate, in all 64 blocks — 192 FFN matrices, 144 SSM-block matrices, 64 attention matrices and the head.

## How it works

### The transform

`H = (1/√N)·Sylvester(N)` with `Sylvester(r,c) = (-1)^popcount(r ∧ c)`, applied **independently inside each 1024-element block** of the weight's contraction axis. The repo builds exactly this matrix in the loader (`src/src/llama-model.cpp:2054-2062`: parity of `row & col` → `±1/√block_size`, then uploaded as a real `F32` tensor named `prism.hadamard.1024`, `:2045`), and the op-level test initialises its `src0` the same way before comparing against the MUL_MAT result (`src/tests/test-backend-ops.cpp:4637-4690`, `test_mul_mat_hadamard`). Verified numerically for `N ∈ {8, 64, 1024}`: the generated matrix is **symmetric and involutive** (`H·H = I` to fp32 rounding), which is why the same matrix serves as both fold and un-fold.

`axis = input-last-dimension` means the rotation contracts each weight's **input** dimension (`ggml ne[0]`), not its output dimension. The loader enforces `weight->ne[0] % block_size == 0` per folded weight (`src/src/llama-model.cpp:2006-2008`); all three widths in this file divide by 1024 (5, 6 and 17 blocks respectively).

### Folded weights are consumed with an activation-side transform, never rewritten

The value type the loader builds per folded weight is documented as *"the activation-side transform applied immediately before the matmul: optional sign flip, then the normalized blockwise Hadamard rotation"* (`src/src/llama-graph.h:20-33`, `struct llama_hadamard_transform`). Both graph helpers consult it:

```
cur_mm = cur;
if (signs) cur_mm = ggml_mul(ctx0, cur_mm, t.signs);          // sign flip first
if (perm_rep > 1) cur_mm = <tiled → grouped permutation>;      // GDN output only
cur_mm = llama_mul_mat_hadamard(ctx0, cur_mm, t.rot);          // normalized block FWHT
res    = ggml_mul_mat(ctx0, w_folded, cur_mm);                 // the folded weight
```

— read from `src/src/llama-graph.cpp:1546-1590` (`build_lora_mm`) and `:1605-1650` (`build_lora_mm_id`). `llama_mul_mat_hadamard` (`src/src/llama-impl.h:57-75`) reshapes the activation to `[block_size, rows]`, multiplies by the rotation tensor, and tags the result with the op hint:

```c
res = ggml_mul_mat(ctx, rot, res);
ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
```

`GGML_HINT_SRC0_IS_HADAMARD` (`src/ggml/include/ggml.h:451-453`, setter `src/ggml/src/ggml.c:3349-3357`) is an instruction to the backend: **`src0` is a Hadamard matrix, do not read it — run the fast transform on `src1` instead.** Every backend that implements it does exactly that (`src/ggml/src/ggml-sycl/fwht.hpp:6-10`: *"src0 is not read at all"*; CUDA `src/ggml/src/ggml-cuda/ggml-cuda.cu:1822-1825`, CPU `src/ggml/src/ggml-cpu/ggml-cpu.c:1333-1337` → `ggml_compute_forward_fwht`, `src/ggml/src/ggml-cpu/ops.cpp:12066-12154`). On CUDA the growth is `scale = 1/sqrtf(n)` (`src/ggml/src/ggml-cuda/fwht.cu:323`) with per-block signs loaded at the transform's front end (`:36`, `:100`, `:159`, `signs + (r % n_blk) * N`), and the graph's separate sign `MUL` is fused into the same launch (`src/ggml/src/ggml-cuda/ggml-cuda.cu:3482-3496`). Backends without the FWHT simply refuse the hint and fall through to an honest matmul against the rotation matrix (`src/ggml/src/ggml-blas/ggml-blas.cpp:419-425`) — so the *result* is basis-independent even where the fast path is missing.

**The runtime therefore does not "implement the inverse fold" by de-folding anything.** It implements it as `y = W_folded · (H·s ⊙ x)`, which equals `W·x` because `H = H⁻¹`. I looked for the alternative — code that multiplies a folded weight by the rotation at load time, or a converter option that pre-folds — and there is none:

- `src/src/llama-model.cpp:1972-2139` builds only the rotation and sign tensors and a `weight → transform` map; nothing writes to the weight tensors' contents.
- The only use of the `rot` tensor as a matmul operand is the *fallback* path in backends that refuse the hint (BLAS, and the `use_ref` CPU path).
- The whole `src/` tree contains no code that computes a weight fold; see *Where it lives*.

Cost per matmul, from the geometry: the activation is transformed in 1024-element blocks, so a 5120-wide activation costs 5 blocks, a 6144-wide one 6, an FFN intermediate 17. The graph memoises the transform per `(activation tensor, rotation)` pair, so projections that share an input — `q`/`k`/`v` in the 16 attention blocks, `attn_qkv`/`attn_gate` in the 48 SSM blocks, `ffn_gate`/`ffn_up` everywhere — each pay for one transform, not one per weight (`src/src/llama-graph.cpp:1556-1573`; `src/src/llama-graph.h:1131-1133`).

### The embedding table: `inverse_weight_names`, i.e. the same fold undone on the other side of the lookup

`prism.hadamard.inverse_weight_names = ["token_embd.weight"]` does **not** mean the embedding matrix is excluded from the fold. It means the opposite: the table's rows are *stored* in the rotated basis ("latent rows"), and because a row lookup is a gather and not a matmul, the un-fold has to happen on the gathered row rather than on the matmul input. The loader says so in code (`src/src/llama-model.cpp:1328-1341`: *"tensors consumed by row lookup store latent rows and need the inverse transform applied to the lookup result instead"*), and restricts the role to exactly this tensor: any other name in `inverse_weight_names` throws, as does listing a tensor in both `weight_names` and `inverse_weight_names`.

The graph then restores the primal basis right after the lookup (`src/src/llama-graph.cpp:2398-2414`, `build_embd_rows`):

```
cur = ggml_get_rows(ctx0, tok_embd, ids);
// a Hadamard-latent embedding table stores rotated rows; restore the
// primal basis right after the lookup: h = s * (H z)
cur = llama_mul_mat_hadamard(ctx0, cur, it->second.rot);   // H first …
if (signs) cur = ggml_mul(ctx0, cur, it->second.signs);    // … then the signs
```

Note the ordering difference from the weight path (rotation-then-sign here, sign-then-rotation there): the two are the two halves of the same symmetric pair `(s, H)`, and each undoes the other. The same after-lookup transform is applied by hand in the Qwen3.5 draft head, which otherwise "would consume embeddings in the rotated basis" and be rejected by the graph verifier (`src/src/models/qwen35.cpp:640-648`).

Implication for this artifact: the file keeps a **separate** `output.weight`, also folded (it is in `weight_names`, consumed through `build_lora_mm` with the activation transform), so the embedding and the head are two distinct `PQ2_0` tensors of identical shape. A tied variant of the same checkpoint would need `version 2` + `tied_output = true`; the loader rejects the tied shape otherwise (`src/src/llama-model.cpp:1354-1356`).

### The Gated-DeltaNet order permutation (`gdn_v_grouped`)

For the 48 `ssm_out` weights the folded latent keeps the *training* grouped V order, so a column permutation applied at conversion time cannot be refolded (`src/conversion/qwen.py:549-554`). The converter records that with `_hadamard_gdn_v_grouped`, which becomes `prism.hadamard.gdn_v_grouped = true` in the file. The loader then asks the graph to permute the *activation* from tiled `[hd, nk, rep]` to grouped `[hd, rep, nk]` before the signs and the rotation (`src/src/llama-model.cpp:2123-2131`, `perm_hd/perm_nk/perm_rep`; consumed at `src/src/llama-graph.cpp:1561-1571`). With `ssm_dt_rank = 48` and `ssm_n_group = 16` ([[qwen35-architecture]]) that is `perm_hd = 128, perm_nk = 16, perm_rep = 3`. This file sets the flag; a file that folded the same tensors *without* the flag would be silently mis-ordered, and nothing cross-checks the flag against the tensor contents.

### Why fold at all

Two things are supported by the tree, one is the standard argument:

1. **The runtime can undo it in `O(d log d)` instead of `O(d²)`.** The fold is only affordable because the un-fold is an FWHT with a hint that lets every backend skip `src0` entirely (`src/ggml/src/ggml-sycl/fwht.hpp:6-10`; the `rot` tensor exists for shape/verification and for hint-refusing backends only). Operators of the form `MUL_MAT(H, x)` never actually cost a 1024×1024 matmul per 1024-block.
2. **It is an error-spreading device.** The repository's own articulation of the principle is in the TriAttention calibration corpus, about the *KV* side: *"polar Walsh-Hadamard transforms (WHT) can equalize outlier dimensions, enabling 2-bit, 3-bit, and 4-bit representations with minimal degradation in perplexity"* (`src/calibration_corpus.txt:10`). Applied to weights, folding before quantization makes every stored coefficient a balanced ±1/√N mixture of its block, so a 128-element group quantizer such as `PQ2_0` (`QK_PQ2_0 = 128`, [[prismml-weight-kernels]]) sees a flattened dynamic range instead of a few dominant channels. **[INFERENCE]** — the repo implements the mechanism but nowhere states this rationale for the *weight* fold; the quantitative claim (better perplexity at equal bits) is not measured anywhere in this repository.

## Three Walsh–Hadamard paths, and which page owns which

All three funnel into the same ggml op (`GGML_OP_MUL_MAT` + `GGML_HINT_SRC0_IS_HADAMARD`), which is precisely why they are easy to conflate. They differ in the operand, the block size, where the signs come from, and when they run:

| | **This page** — model weight fold | TurboQuant KV rotation | Fork attention K/V rotation |
| :--- | :--- | :--- | :--- |
| Operand | the 401 weight matrices + the embedding table | K and V rows as written to the cache | K and V rows as written to the cache |
| Declared by | GGUF metadata `prism.hadamard.*` | the cache type (`-ctk turbo3`, …) | `attn_rot_k` / `attn_rot_v`, gated on `ggml_is_quantized(type_k) && head_dim % 64 == 0`; env `LLAMA_ATTN_ROT_DISABLE` (`src/src/llama-kv-cache.cpp:462-486`, `src/src/llama-kv-cache.h:293-295`) |
| Block size | 1024 | 128-element "rotation group" | one matrix per size 64 … `n_embd_head_k_all` (`src/src/llama-kv-cache.cpp:493-508`) |
| Signs | from GGUF (`sign_values`, 28 672 of them) | compiled in (`TURBO_WHT_SIGNS1/2`) | host-generated |
| When | once, at pack time; un-folded per matmul input at inference | every token, per head | every token, per head |
| Needs its inverse | every folded matmul + the embedding lookup | TriAttention scoring, attention output | attention read path |
| Tracked defects | *none* | [[ta-1-wht-inversion-256]], [[ta-4-cooperative-fwht-race]], [[tq-4-wht-numerical-mismatch]], [[tq-5-tail-elements]] | none; basis coupling with the KV mean-centre bias (`src/docs/kv-mean-center.md:94-99`) |

Read the middle and right columns on [[walsh-hadamard-transform]] and [[turboquant]]; they do not apply here. In particular, **no issue in either inventory covers the weight fold**, and none of those defects can be triggered by the weight-side transform — different file (`fwht.cu` vs `turbo-wht.cu` / `set-rows.cu` / `triattention-score.cu`), different group size, different sign source, and [[triattention]] never touches a weight.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `/home/ms/Загрузки/Ternary models/Ternary-Bonsai-2-27B-PQ2_0.gguf` | the `prism.hadamard.*` block itself; `general.basename = folded`; the 401 folded `PQ2_0` tensors |
| `src/src/llama-model.cpp` | contract validation and load-time refusal `:1196-1356`; rotation + sign tensor construction and the `weight → transform` map `:1972-2139`; rotation matrix generation `:2043-2062`; sign vector chosen by `weight->ne[0]` `:2077-2083`; GDN permutation geometry `:2123-2131` |
| `src/src/llama-model.h` | per-model state `:713-720` (`hadamard_weight_blocks`, `hadamard_inverse_blocks`, `hadamard_sign_data`, `hadamard_rotations`, `hadamard_inverses`, …) |
| `src/src/llama-graph.h` / `src/src/llama-graph.cpp` | `llama_hadamard_transform` `:20-33`; activation transform in `build_lora_mm` `:1546-1590` and `build_lora_mm_id` `:1605-1650`; after-lookup inverse in `build_embd_rows` `:2398-2414`; transform memo `:1556-1573` |
| `src/src/llama-impl.h` | `llama_mul_mat_hadamard()` `:57-75` — the reshape + hinted MUL_MAT |
| `src/src/llama-context.cpp` | graph verifier `llama_verify_hadamard_graph()` `:31-98`, rotations copied into the context `:245-252`, one-time check on the first graph `:2710-2713` |
| `src/src/models/qwen35.cpp` | the draft/MTP head's explicit after-lookup inverse `:640-648` |
| `src/ggml/include/ggml.h`, `src/ggml/src/ggml.c` | `GGML_HINT_SRC0_IS_HADAMARD` `:451-453`; `ggml_mul_mat_set_hint()` `:3349-3357` |
| `src/ggml/src/ggml-cuda/fwht.cu`, `fwht.cuh` | the CUDA FWHT: warp kernel `:18-84`, shared-memory variant `:88-136`, per-row block variant `:139-212`, per-block signs `:36/:100/:159`, dispatch `fwht_launch` `:221-296` (sizes 64…8192, `FWHT_BLOCK_CASE(1024)` `:287`), `scale = 1/sqrtf(n)` `:323`, entries `ggml_cuda_op_fwht` `:331` / `ggml_cuda_op_fwht_signed` `:336` |
| `src/ggml/src/ggml-cuda/ggml-cuda.cu` | hint dispatch `:1822-1825`; sign+reshape+FWHT fusion `:3482-3496` |
| `src/ggml/src/ggml-cpu/ggml-cpu.c` `:1333-1337`, `src/ggml/src/ggml-cpu/ops.cpp` `:12066-12154` | CPU FWHT (also the reference path used by the op tests) |
| `src/ggml/src/ggml-blas/ggml-blas.cpp` | refuses the hint `:419-425` → honest matmul fallback |
| `src/ggml/src/ggml-sycl/fwht.hpp`, `src/ggml/src/ggml-metal/ggml-metal-ops.cpp`, `src/ggml/src/ggml-vulkan/ggml-vulkan.cpp` | the same op in the other backends |
| `src/tests/test-backend-ops.cpp` | `test_mul_mat_hadamard` `:4637-4690` (initialises `src0` as the normalized Sylvester matrix), `test_fwht_signed` `:4690+` (sign flip + reshape + FWHT-hint path) |
| `src/conversion/base.py` | `hadamard_folded_names()` `:620-633`, `add_hadamard_metadata()` `:635-773` — **transfers** a manifest contract into GGUF metadata |
| `src/conversion/qwen.py` | `_hadamard_folds_tensor()` `:568-570`; the folded `linear_attn.out_proj` / `ssm_out` grouped-order rule `:549-554`, `:621-626` |
| `src/docs/development/hadamard-tied-output.md` | the `version 2` / `tied_output` contract and its required metadata |

**The fold is not computed in this repository.** `add_hadamard_metadata()` reads a `hadamard_packing.json` that must sit next to the HF checkpoint (`kind = "hadamard-weight-fold"`, `status = "requires-matching-runtime"`, transform name `normalized-signed-sylvester-walsh-hadamard`, per-tensor `role ∈ {fold-before-matmul, inverse-after-lookup}`, `axis = -1`) and *only copies* it into GGUF keys; it refuses any architecture or tensor kind not on its whitelist so that a GGUF which "loads but skips the transform" cannot be produced (`src/conversion/base.py:683-746`). Searched, and found nothing else: no folding code in `src/gguf-py` (the string `hadamard` does not occur anywhere under `src/gguf-py`, and `src/gguf-py/gguf/scripts/` ships only dump/endian/hash/metadata editors), and the only file in the whole tree whose *name* mentions Hadamard is the doc above. The producer of the fold is the checkpoint packer outside this tree; the runtime side here is a *consumer* with a matching-runtime contract, which is exactly what the manifest's `status` says.

## Known issues

- **No issue page covers this subsystem.** The three WHT defects in the inventories are all KV-side and cannot be reached from folded weights: [[ta-1-wht-inversion-256]] and [[ta-4-cooperative-fwht-race]] live in `triattention-score.cu`, and [[tq-4-wht-numerical-mismatch]] and [[tq-5-tail-elements]] in `turbo-quant.cuh` / `set-rows.cu`. The weight fold uses `fwht.cu` and one hint tag; it shares no code with them.
- **The failure mode is designed to be loud, and is.** Because the transform lives in `build_lora_mm`/`build_lora_mm_id` rather than in the weights, a model could load and silently compute wrong math on any path that bypasses those helpers. Two independent guards exist: the loader refuses `prism.hadamard` on architectures not on its whitelist (`src/src/llama-model.cpp:1272-1285`, seven archs including `qwen35`), and the first graph build walks the graph and throws `"Hadamard-folded weight '%s' is consumed without its activation transform"`, or warns about an unconsumed latent lookup (`src/src/llama-context.cpp:31-98`). The verifier's own header comment names the silent-wrong-results failure it exists to prevent.
- **`gdn_v_grouped` is asserted, not verified.** The flag is trusted at face value; nothing checks it against the 48 `ssm_out` tensors' layout, and getting it wrong permutes activations silently. `[UNVERIFIED]` for this artifact (no independent way to check the byte order of a `PQ2_0` tensor without running the model).
- **No measured effect of the fold anywhere.** No benchmark, profile, or ablation in this repository isolates the weight fold from the KV stack; [[benchmarks]] measures only end-to-end runs, and the fold is not mentioned in [[source-state-md]]'s profiling conclusions at all. Whether the fold is a win on the V100 ([[v100-sxm2]]) specifically — where the un-fold FWHT is plain CUDA and the alternative fallback is a 1024² matmul — is unmeasured.

### Open questions

- **Does the weight fold interact with TurboQuant's KV rotation or InnerQ?** They act on different operands and different bases, and no TA/TQ issue or doc considers them together. `[UNVERIFIED]` in both directions: the fold is inside the weights, the KV rotation is downstream of them, so the *compositions* that exist in the served configuration (`-ctk turbo3`) are: fold (weights) → projections → per-token KV rotation (cache) → scoring. Whether the fold changes the statistics the KV quantizer and InnerQ calibration then see is not discussed anywhere in the tree.
- **Where is the packer?** The producing side (which emits `hadamard_packing.json` and the folded tensors) is outside this repository, and its exact fold convention (left- vs right-multiplication, whether the sign vector is applied to the weight or to the activation at pack time) cannot be confirmed from here — only its *runtime contract*, which is fully specified. `[UNVERIFIED]`.
- **`general.file_type = 141`** is unexplained by the in-tree type table; it is recorded on [[ternary-bonsai-2-27b]] and nothing here depends on it.
- The related `prism.hadamard.tied_output` (`version 2`) path is exercised by no artifact in this project; only the untied `version 1` path is live.

## See also

[[ternary-bonsai-2-27b]] · [[qwen35-architecture]] · [[prismml-weight-kernels]] · [[quantization]] · [[walsh-hadamard-transform]] · [[turboquant]] · [[triattention]] · [[kv-cache]] · [[gemm-dispatch]] · [[benchmarks]] · [[v100-sxm2]] · [[overview]] · [[source-state-md]] · [[ta-1-wht-inversion-256]] · [[tq-4-wht-numerical-mismatch]]
