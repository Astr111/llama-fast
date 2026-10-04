---
title: Ternary-Bonsai-2-27B (PQ2_0)
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: ["/home/ms/Загрузки/Ternary models/Ternary-Bonsai-2-27B-PQ2_0.gguf", calibration/bonsai-27b.triattention, scripts/start_server_turbo.sh, scripts/start_server_baseline.sh, scripts/run_cli.sh, src/ggml/include/ggml.h, src/ggml/src/ggml-cuda/mmq.cu, src/ggml/src/ggml-cuda/mmq.cuh, src/ggml/src/ggml-cuda/ggml-cuda.cu, src/ggml/src/ggml-cuda/triattention-score.cu, src/ggml/src/ggml-cuda/turbo-innerq.cuh]
tags: [model, quantization, kv-cache, triattention]
---

# Ternary-Bonsai-2-27B (PQ2_0)

## What it is

The single model this fork exists to serve: **`Ternary-Bonsai-2-27B-PQ2_0.gguf`** — a 27B PrismML ternary-weights release whose weight matrices are stored in PrismML's private **`PQ2_0`** format, and whose **`head_dim = 256`** is the property that stresses both custom stacks ([[source-state-md]] *Models*).

The GGUF header was read directly (not inferred from prose), which settles several things the wiki previously held only on the sources' word:

| Property | Value (from the file) |
| :--- | :--- |
| GGUF version / tensors / KV pairs | 3 / 851 / 49 |
| `general.architecture` | `qwen35` (Qwen3.5 lineage — not a bespoke architecture) |
| `general.size_label`, `general.name`, `general.basename` | `27B`, `Hf`, `folded` |
| `general.file_type` | `141` (see *Known issues*) |
| `qwen35.block_count` | 64 |
| `qwen35.embedding_length` / `feed_forward_length` | 5120 / 17408 |
| `qwen35.attention.head_count` / `head_count_kv` | 24 / 4 |
| **`qwen35.attention.key_length` / `value_length`** | **256 / 256** → `head_dim = 256` |
| `qwen35.rope.dimension_count`, `rope.dimension_sections` | 64, `[11, 11, 10, 0]` (partial/multimodal RoPE) |
| `qwen35.context_length` | 262144 |
| `qwen35.full_attention_interval` | 4 |
| SSM block params | `ssm.inner_size` 6144, `ssm.state_size` 128, `ssm.group_count` 16, `ssm.conv_kernel` 4, `ssm.time_step_rank` 48 |
| Tokenizer | GPT-2-style, `qwen35` pre-tokenizer, 248 320 tokens; BOS/PAD 248044, EOS 248046 |
| Weight storage | **402 × `PQ2_0`**, 353 × `F32` (norms, SSM scalars, biases), 96 × `BF16` (`ssm_alpha`, `ssm_beta`) |

Two structural facts follow from the tensor list that no source document states:

1. **It is a hybrid attention/SSM model, and only 16 of its 64 blocks keep a KV cache.** `attn_k.weight` / `attn_v.weight` / `attn_q.weight` / `attn_output.weight` exist for layers **3, 7, 11, … 63** (16 layers, stride 4, matching `full_attention_interval = 4`). The other 48 blocks carry `attn_qkv`, `attn_gate`, `ssm_conv1d`, `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_dt`, `ssm_norm`, `ssm_out` — i.e. Gated-DeltaNet/SSM state instead of a KV cache.
2. **The K and V projections are 1024 wide** (`[5120, 1024]` = 4 KV heads × 256), while `attn_q.weight` is `[5120, 12288]` and `ffn_*` are `[5120, 17408]`. The `1024 = head_count_kv × key_length` identity is independent confirmation of `head_dim = 256`.

The quant format itself: `GGML_TYPE_PQ2_0 = 142` is documented in-code as *"Prism-private Q2_0 at group size 128 (upstream Q2_0 is group 64)"* and `GGML_TYPE_PTQ1_0 = 143` as *"Prism-private ternary, group 128"* (`src/ggml/include/ggml.h:436-439`). The sibling ternary artifact `Ternary-Bonsai-2-27B-PTQ1_0.gguf` is a symlink into the Hugging Face cache of repo **`prism-ml/Ternary-Bonsai-2-27B-gguf`**; a `Ternary-Bonsai-4B-Q2_0_g64.gguf` from `prism-ml/Ternary-Bonsai-4B-gguf` is cached alongside it. See [[prismml-weight-kernels]].

## How it works

### `head_dim = 256` against the WHT inversion path

TurboQuant stores K/V after a Walsh-Hadamard rotation, so anything that re-reads the cache in the original basis must invert that rotation first. The TriAttention scoring kernel does this in 128-element blocks and its multi-block branch is empty:

```c
for (uint32_t b = 0; b < padded_hd; b += 128) {
    if (f < 64) {
        float * block = k_smem + b;
        // ... no rotation performed here
    }
}
if (padded_hd == 128 && f < 64) { inverse_wht_rotation_128(k_smem, f); }
```

— read verbatim from `src/ggml/src/ggml-cuda/triattention-score.cu` (the `NEED_WHT_INV` block, guard at line 225, `inverse_wht_rotation_128` defined at line 72). With `padded_hd = 256` the guard is false, no block ever calls the inverse rotation, and scoring proceeds on keys that still carry the TurboQuant rotation. That is [[ta-1-wht-inversion-256]], and a 256-dim model is precisely what triggers it. `head_dim = 128` models — the ones the comment at lines 228-230 calls the "primary case" — never touch the broken branch.

### `head_dim = 256` against the InnerQ channel ceiling

`INNERQ_MAX_CHANNELS` is hardcoded to **128** in `src/ggml/src/ggml-cuda/turbo-innerq.cuh` (line 6, `#define INNERQ_MAX_CHANNELS 128`), and the host-side scale buffer is declared as `float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS]`. A 256-channel head cannot be equalized across all its channels by this state: at most half. That is [[tq-7-innerq-max-channels]].

### Other properties that bite

- **Partial RoPE.** Only 64 of the 256 dimensions per head are rotary (`rope.dimension_count = 64`, MRoPE sections `[11, 11, 10, 0]`). The scoring kernel's step 3 applies `RoPE^{-1}` over its `f`-indexed frequencies; how a 64-of-256 rotary layout interacts with the 128-element block loop is **not** established here — open question, not a finding.
- **The weights are already Hadamard-folded, independently of TurboQuant.** The file carries a `prism.hadamard.*` metadata block — `transform = normalized-sylvester-walsh-hadamard`, `block_size = 1024`, `axis = input-last-dimension`, `sign_mode = explicit`, `sign_widths = [5120, 6144, 17408]`, `sign_values` (28 672 entries), `inverse_weight_names = ["token_embd.weight"]`, `version = 1` — and `general.basename` is literally `folded`. So there are **two** Walsh-Hadamard transforms in play: PrismML's weight fold (model-side, baked in) and TurboQuant's Polar WHT on the KV cache (runtime). The in-repo WHT issues concern the latter; the model-side fold is not covered by any TA/TQ issue.
- **Weight-kernel path on the target.** `GGML_TYPE_PQ2_0` *is* listed in `ggml_cuda_should_use_mmq()` (`src/ggml/src/ggml-cuda/mmq.cu:366+`, generic supported list) and has dp4a tile geometry and an MMQ case (`src/ggml/src/ggml-cuda/mmq.cuh`, `mmq_get_dp4a_tile_x_sizes`, `DECL_MMQ_CASE(GGML_TYPE_PQ2_0)`). The fast wgmma variant is separate and Hopper-only: `ggml_cuda_mul_mat()` tries `ggml_cuda_mul_mat_q1_hopper()` for `Q1_0`/`PQ2_0` only when the device is `sm_90` (`src/ggml/src/ggml-cuda/ggml-cuda.cu:1866-1873`). On the V100 neither the Hopper path nor the ternary `PTQ1_0` MMQ (gated on `turing_mma_available`) is reachable — see [[v100-sxm2]].
- **KV cache format in production.** All launch scripts that use TurboQuant run it with **`-ctk turbo3 -ctv q8_0`** ([[turboquant]], [[kv-cache]]). The `q8_0` value cache bypasses the inverse-WHT work entirely, which is why [[source-readme]] calls it the speed profile.

## Where it lives

**The GGUF is not in the repository.** It was found on this machine at:

| Artifact | Path | Size |
| :--- | :--- | ---: |
| Target model (PQ2_0) | `/home/ms/Загрузки/Ternary models/Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7 206 168 928 B (6.71 GiB) |
| Ternary sibling (PTQ1_0) | `/home/ms/Загрузки/Ternary models/Ternary-Bonsai-2-27B-PTQ1_0.gguf` | symlink → `~/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/blobs/5310…` |
| mmproj used by the baseline script | `/home/ms/Загрузки/Ternary models/mmproj-Qwen3.8-27B-BF16.gguf` | symlink → `/hdd2/lm-studio-models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/mmproj-Qwen3.8-27B-BF16.gguf` |

What the repo's own scripts expect is different, and one of them is broken on this machine:

- `scripts/start_server_turbo.sh` and `scripts/run_cli.sh` take the model as `$1` or `$MODEL_PATH`, defaulting to the placeholder `/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf` and exiting if it does not exist.
- `scripts/start_server_baseline.sh` hardcodes `${HOME}/Prism/llama-prism-b10743-adfffbe/Ternary-Bonsai-2-27B-PQ2_0.gguf` (and the matching `mmproj-…`). **`/home/ms/Prism` does not exist** (checked 2026-09-28), so the baseline script cannot run as shipped; its `BIN_DIR` default `${BASE_DIR}/../dist-rtx3090/bin` is likewise absent. The model it names *is* the one listed above, at a different path — the script is stale, not wrong about the model.

The TriAttention calibration profile for this model ships in the repo and is present: `calibration/bonsai-27b.triattention`, 789 571 B ([[source-readme]] describes it as "772 KB" — 771 KiB, consistent).

### What the launch scripts actually pass

| Script | Model | KV types | TriAttention flags | Other |
| :--- | :--- | :--- | :--- | :--- |
| `scripts/start_server_turbo.sh` | `$1`/`$MODEL_PATH`, placeholder default, aborts if missing | `-ctk turbo3 -ctv q8_0`, both overridable via `CTK`/`CTV` | `--triattention-stats ../calibration/bonsai-27b.triattention --triattention-budget 4096 --triattention-window 512 --triattention-protect-prefill`, added only when the stats file exists **and** `DISABLE_TRIATTENTION != 1`; budget/window overridable (`TRI_BUDGET`, `TRI_WINDOW`) | `-ngl 99`, `-c 32768`, `-n 8192`, `--reasoning-budget 4000`, `-np 1`, `-t $(nproc)`, host `0.0.0.0:8080`; exports `GGML_CUDA_GRAPH_OPT=1` |
| `scripts/run_cli.sh` | same placeholder convention | `-ctk turbo3 -ctv q8_0` (`CTK`/`CTV`) | same four flags, with budget **4096** / window **512** hardcoded (not overridable) | `-ngl 99`, `-c 16384`; interactive `--conversation` when no prompt |
| `scripts/start_server_baseline.sh` | `${HOME}/Prism/llama-prism-b10743-adfffbe/Ternary-Bonsai-2-27B-PQ2_0.gguf` | **none passed** → default FP16 KV (the script's own banner says "FP16 (256 KB/token, 4K tok/GB VRAM)") | none | `--mmproj <Qwen3.8-27B mmproj>`, `--no-mmproj-offload`, `-ngl 99`, `-c 16384`, host `127.0.0.1:8080`, forwards `"$@"` |

None of the three loads a draft model or passes any `--spec-*` flag — speculative decoding is not wired into the repo's launch surface (see [[qwen3-dflash-draft]]).

## Known issues

- [[ta-1-wht-inversion-256]] — **CRITICAL, triggered by this exact model.** `head_dim = 256` skips the WHT inversion in the scoring kernel; eviction decisions are made on rotated keys.
- [[tq-7-innerq-max-channels]] — `INNERQ_MAX_CHANNELS = 128` against 256 channels: at most half the head can be equalized.
- [[ta-2-budget-starvation]] — with budget 4096 / window 512, long agent prompts (`prefix + divide ≥ budget`) collapse eviction to a sliding window. This model is run as a coding agent with long system prompts, which is exactly the failing regime.
- [[ta-3-cpu-fallback-transfers]] — the per-cell synchronous D2H path is sized by `n_decode × head_dim`; `head_dim = 256` doubles each transfer relative to a 128-dim model.
- [[tq-1-missing-gemm-kernels]] — the KV types this model is served with (`turbo3`/`q8_0` K, optionally `turbo2` V) have no mmq/mmvq kernels, so attention lands in cuBLAS/MAGMA ([[performance-profile]]).
- [[benchmarks]] — the published 1.39× / VRAM figures were measured on an RTX 3090, not on the V100 target ([[v100-sxm2]]).

### Open questions

- `general.file_type = 141`. `src/ggml/include/ggml.h:455-488` defines `GGML_FTYPE_MOSTLY_PQ2_0 = 128` and `GGML_FTYPE_MOSTLY_PTQ1_0 = 129` and nothing at 141. The value is unexplained; the tensor types in the file are unambiguous, so nothing depends on it.
- Only 16 of 64 blocks hold KV, and each head is 256 wide with 4 KV heads over 16 layers. What that implies for the repo's per-token KV accounting (`~25 200 tokens per 1 GB`) has not been re-derived here; the published figure is a measurement, not arithmetic.
- Does the PrismML weight fold (`prism.hadamard.*`) interact with TurboQuant's KV rotation or with the InnerQ channel limits? No issue in either inventory covers the weight-side transform.

## See also

[[overview]] · [[source-state-md]] · [[v100-sxm2]] · [[triattention]] · [[turboquant]] · [[innerq]] · [[walsh-hadamard-transform]] · [[kv-cache]] · [[prismml-weight-kernels]] · [[speculative-decoding]] · [[qwen3-dflash-draft]] · [[ta-1-wht-inversion-256]] · [[tq-7-innerq-max-channels]] · [[tokenizer]]
