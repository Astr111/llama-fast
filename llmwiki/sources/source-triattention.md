---
title: Source — TRIATTENTION.md
type: source
status: current
updated: 2026-09-28
sources: [TRIATTENTION.md]
verified: [src/docs/TRIATTENTION.md, src/src/llama-triattention.h, src/src/llama-triattention.cpp, src/src/llama-kv-cache.cpp, src/src/llama-context.cpp, src/include/llama.h, src/tools/triattention-calibrate/triattention-calibrate.cpp, src/common/arg.cpp, src/common/common.h, src/common/common.cpp, src/src/models/qwen35.cpp, src/ggml/src/ggml-cuda/triattention-score.cu, calibration/bonsai-27b.triattention, scripts/start_server_turbo.sh]
tags: [triattention, kv-eviction, calibration, documentation]
---

# Source — `TRIATTENTION.md`

The first-party design document for TriAttention: the offline calibration pass, the trigonometric scoring formula, the eviction policy, the CLI surface, and the `.triattention` binary format. It is the only prose description of the mechanism, and the wiki's `ta-*` issues were derived from `state.md`, not from it — so this is where the two accounts are reconciled.

It is a **design description, not a record of the shipped build**. The scoring core, the binary format and the KV-cache hooks it describes are the ones in the tree; the defaults, file paths and calibration workflow are not (see *Discrepancies worth holding*).

## Summary

TriAttention is described as **calibration-guided KV-cache eviction**: one offline pass over ~2K tokens collects pre-RoPE query statistics per (layer, head, frequency band); at decode time every cached token is scored by asking how much attention it would receive *if future queries followed the calibration distribution*; the lowest-scoring tokens are evicted to a fixed `budget`, with prompt tokens optionally protected. Combined with TurboQuant's 2–4 bit KV compression the document claims **~40× effective KV memory reduction** (compression × eviction).

The document covers: the scoring formula and its symbol table; a `llama-server` quick start with three TriAttention flags; a 13-row CLI reference; three pruning modes (global / per-KV-head / per-layer-head) and two triggers (interval / slack); the binary format down to field order; the source-file map and integration points; performance and memory overhead figures; and a compatibility matrix (KV types, RoPE layouts, GQA).

## Key claims

| Claim | Section |
| :--- | :--- |
| TriAttention is calibration-guided trigonometric KV eviction; with TurboQuant 2–4 bit KV it gives **~40× effective KV memory reduction** | §Overview |
| Calibration is offline and short: ~2K tokens, collecting pre-RoPE query statistics per (layer, head, frequency-band) | §How it works, step 1 |
| Runtime scores *every* cached token; lowest-scoring are evicted back to the budget; prompt tokens can optionally be protected | §How it works, steps 2–3 |
| Scoring formula `S(m) = (1/D) Σ_d Σ_f amp_f·fscale²_f·cos(ω_f·δ_d + φ_f) + excess_f·fscale²_f·|k_f|` with `amp_f = ‖E[q_f]‖·|k_f|`, `φ_f = angle(E[q_f]·conj(k_f))`, `ω_f = θ^(−2f/d)`, `δ_d = current_pos − key_pos + offset_d`, `excess_f = E[‖q_f‖] − ‖E[q_f]‖` | §Scoring formula |
| Calibration is run as `python scripts/calibrate-triattention.py --model meta-llama/Llama-3.1-8B-Instruct --n-tokens 2048 --output …`, validated by `python scripts/validate-calibration.py …` | §Quick start |
| CLI defaults: `--triattention-budget` 2048, `--triattention-window` 128, `--triattention-offset-max` 65536, mode `global`, trigger `interval`, agg `mean`, seed 0, normalize off, protect-prefill on, log off | §CLI arguments |
| Three pruning modes: global (top-B by max-over-heads score), per-KV-head, per-layer-head | §Pruning modes |
| Two triggers: interval (every `window` decode tokens above budget) and slack (at `budget + window`) | §Trigger strategies |
| `.triattention` format: magic `0x54524941` "TRIA", version 1, then `head_dim, num_layers, num_attn_heads, num_kv_heads, rope_theta(f64), rope_style, n_sampled, freq_count = head_dim/2, name_len, name`; per sampled head `layer_idx, head_idx, q_mean_real, q_mean_imag, q_abs_mean, r_f` (`r_f = ‖E[q]‖/E[‖q‖]`, validation only) | §Calibration file format |
| Spec lives in `src/llama-triattention.h`; implementation in `src/llama-triattention.cpp`; scoring kernel in `ggml/src/ggml-cuda/triattention-score.cu`; calibration/validation in `scripts/*.py` | §Architecture → Source files |
| Integration: KV-cache hooks (`apply_ubatch`, `seq_rm`/`clear`, `seq_add`), 13 `--triattention-*` args in `common/arg.cpp`, `llama_triattention_init()` in `include/llama.h`, auto-init in `common/common.cpp` | §Architecture → Integration points |
| Performance: 5–10 ms per prune on CPU at 128K context, <1 ms on GPU; memory overhead ~4 B per KV position plus ~100 KB calibration data | §Performance characteristics |
| Compatibility: `TURBO2_0/TURBO3_0/TURBO4_0/Q8_0/F16/BF16/F32`; half and interleaved RoPE; full GQA; any transformer with standard RoPE attention | §Compatibility |
| Paper: arXiv:2604.04921, "TriAttention: Trigonometric KV Cache Eviction", **MIT/NVIDIA/ZJU, 2025**, Yaniv Ben-Nun, Agustín Zanotti, Dan Alistarh, Ming-Yu Liu | §Overview, §References |

## Discrepancies worth holding

### The scoring core matches the code; the frequency weighting does not

The formula's amplitude, phase, geometric offsets and aggregation are literally the implemented ones — `triattention_score_keys` (`src/src/llama-triattention.cpp:434`) computes `amp = q_mean_abs[f] * k_mag` (`:475`), `phi = atan2f(conj_im, conj_re)` of `E[q_f]·conj(k_f)` (`:480-482`), and aggregates over `offsets = {1,2,4,…,offset_max}` by mean or max (`:494-502`).

`fscale²_f`, which the document calls **"frequency importance weighting (1/ω²)"**, is a constant `1.0` in the shipped code: `triattention_build_freq_scale_sq` (`:320`) evaluates `cosf(omega[f]*0.0f)² + sinf(omega[f]*0.0f)²` (`:325-327`). The document presents a term that the code does not implement. That is [[ta-5-freq-scale-dead-code]]; the document is evidence that the term was *intended*, which sharpens the issue from "dead code" to "divergence from the stated design".

The norm term is also placed differently: the document writes it added once, outside `Σ_d`; the code adds it inside the per-offset loop (`:489-491`), so under the default `mean` aggregation it is divided by `D` like the trigonometric term.

### Eviction as promised vs eviction as implemented

The document's promise — "the lowest-scoring tokens are evicted to bring the cache back to the budget", prompt protection optional — is the *policy intent*. The shipped `triattention_prune_impl` (`:1077`) protects **two** classes it does not mention: the prefix **and** the most recent `divide_length` positions (`recent_threshold` at `:1128`, classification loop `:1134-1146`). The recent window is not optional and is not a quality choice — the comment at `:1114-1121` records that evicting the highest-position token breaks the server's position counter. The scored budget is therefore `decode_budget = (budget > n_protected) ? (budget − n_protected) : 0` (`:1150`), and with the shipped configuration (budget 4096, window 512) a system prompt above ~3500 tokens drives it to **zero**: the document's "evict the lowest-scoring tokens" degenerates into a plain sliding window, at full scoring cost. That is [[ta-2-budget-starvation]], and the document is silent on the whole regime. The document also says nothing about config validation — the condition is unguarded ([[ta-7-config-validation]]).

### Defaults: the CLI table is not the CLI

`src/common/common.h:752-766` disagrees with the document's table on five of nine rows:

| Flag | Document | Code (`src/common/common.h`) |
| :--- | ---: | ---: |
| `--triattention-budget` | 2048 | `0` (`:752`) |
| `--triattention-window` | 128 | `0` (`:753-754`) |
| `--triattention-offset-max` | 65536 | `0` (`:755`) |
| `--triattention-seed` | 0 ("0=deterministic") | `-1` ("-1 to disable") (`:759`) |
| `--triattention-normalize` | off | `true` (`:766`) |

`[[source-readme]]` agrees with the code (budget/window default 0, normalize default true), so the design document is the outlier, not the README. Two consequences are checkable in the code and were not exercised here:

- With `offset_max = 0`, `triattention_build_offsets` (`:333`) never enters its `for (d = 1; d <= offset_max; d *= 2)` loop (`:335`) and returns `n_offsets = 0`; `triattention_score_keys` then computes `inv_n_offsets = 1.0f / 0.0f` (`:450`) and multiplies a zero score by it under the default `mean` aggregation (`:501-502`). Every score becomes `NaN` and the top-B selection loses its ordering. `scripts/start_server_turbo.sh:35` does **not** pass `--triattention-offset-max`, so the shipped launch surface is in this state. `[INFERENCE]` — read from the code, not executed.
- The document's seed semantics are inverted: `seed = 0` enables the deterministic noise term (`sum += noise(rng)` at `:1319-1323`), while `-1` (the code default) is the deterministic path.

Also minor: the document's `--triattention-normalize` row implies a flag that only turns normalization on; the CLI registers `--triattention-no-normalize` as well, and normalization is on by default.

### The calibration workflow describes scripts that do not exist

The document's Quick start runs `python scripts/calibrate-triattention.py` against a HuggingFace model and validates with `python scripts/validate-calibration.py`. Neither file exists anywhere in this repository (neither `scripts/` at the repo root nor `src/scripts/`). What ships is a C++ tool, `llama-triattention-calibrate` (`src/tools/triattention-calibrate/triattention-calibrate.cpp`), registered in `src/tools/CMakeLists.txt:21`, which loads a **GGUF** and reads a plain-text corpus. Its own usage line is:

```
llama-triattention-calibrate -m model.gguf -f corpus.txt -o model.triattention [-c 2048] [-ngl 28] [-t 6]
```

So the document describes an **intended** Python/transformers pipeline, not the shipped one; the shipped one is on the menu at [[triattention-calibrate]] with its own contradictions (the README's `--triattention-calibrate`/`--triattention-calibrate-out` flags are registered in `src/common/arg.cpp:4812-4826` but read by nothing). This is a doc-drift finding, not a nuisance: it dates `TRIATTENTION.md` against the codebase.

### Paths are written against the fork's `src/` root

The document's file map reads `src/llama-triattention.h`, `src/llama-triattention.cpp`, `ggml/src/ggml-cuda/triattention-score.cu`, `common/arg.cpp`, `include/llama.h`. Repo-relative these are `src/src/llama-triattention.h`, `src/src/llama-triattention.cpp`, `src/ggml/src/ggml-cuda/triattention-score.cu`, `src/common/arg.cpp`, `src/include/llama.h` — the same off-by-one-prefix problem `state.md` has. All files except the two Python scripts exist at the corrected paths.

### Claim count and integration detail

- "All 13 `--triattention-*` arguments registered for SERVER and CLI examples" — the block at `src/common/arg.cpp:4689-4833` contains **17** `common_arg` blocks with a `--triattention-` long name (plus the aliases `--triattention-divide-length`, `--triattention-no-mlr`, `--triattention-no-trig`). Of those, `--triattention-calibrate` and `--triattention-calibrate-out` are `.set_examples({LLAMA_EXAMPLE_CLI})` only, not SERVER, and are dead (their `common_params` fields at `src/common/common.h:750-751` are written and never read). The document's table also omits `--triattention-protect-prefill`, `--triattention-no-normalize` and both calibrate flags.
- `llama_triattention_init()` in `include/llama.h` exists (`src/include/llama.h:827`), but its return type is `int32_t` (`0` success / `-1` failure), not `bool`; the parameter is named `normalize_scores`.
- Auto-init (`src/common/common.cpp:1403`) triggers on "stats path non-empty **or** budget > 0", not only on `--triattention-stats`; with budget alone it fails and prints a warning.

### The 40× headline

**Claim:** `~40× effective KV memory reduction` (compression × eviction), repeated verbatim in the code comment at `src/src/llama-triattention.h:13`. Nothing in the wiki corroborates it. [[benchmarks]] holds only the README's **Ampere** (RTX 3090) measurements — no V100 run — and those give ~4 000 → ~25 200 tokens/GB, i.e. **~6.3×**, from compression plus eviction together. The figure is recorded here as the document's claim, not as an established result. `[UNVERIFIED]`

### Performance and memory figures

- "5–10 ms per pruning event (CPU path) at 128K context; GPU <1 ms" — `[UNVERIFIED]`, no benchmark exists in the wiki. The only recorded cost figure is state.md's 0.91 % TriAttention GPU-time share ([[performance-profile]]).
- "Memory overhead ~4 bytes per KV position + ~100 KB calibration data" — `[UNVERIFIED]` and, read against the allocations, understated by orders of magnitude. `triattention_init` (`:699-702`) allocates `dequant_buf` and `unrot_buf` at `kv_size × head_dim` floats each, `score_buf` at `n_sampled × kv_size`, and `combined_buf` at `kv_size`. For the shipped configuration (`kv_size` = 32 768 from `-c 32768` in `scripts/start_server_turbo.sh`, `head_dim = 256`, `n_sampled = 384` from the shipped profile) that is ≈ 118 MB of scratch, against the document's "4 bytes per position" (≈ 128 KB). The calibration side is also larger than stated: the shipped profile is 789 571 B, not ~100 KB, and `triattention_precompute_head_derived` adds two more `freq_count` arrays per sampled head. `[INFERENCE]` — arithmetic on the allocation expressions, not measured.

### Paper provenance — contradiction with `README.md`

The document dates the paper "(MIT/NVIDIA/ZJU, **2025**)" and names Yaniv Ben-Nun, Agustín Zanotti, Dan Alistarh, Ming-Yu Liu; `llmwiki/raw/README.md` cites the **same** arXiv id (`2604.04921`) as "Mao et al. (MIT, NVIDIA, Zhejiang University, **April 2026**)". Both entries are logged below rather than silently reconciled:

> Contradiction (2026-09-28): `TRIATTENTION.md` §Overview/§References says the TriAttention paper is 2025; `README.md` ([[source-readme]], [[upstream-lineage]]) says April 2026 for arXiv:2604.04921. The author lists also differ (Ben-Nun/Zanotti/Alistarh/Liu vs "Mao et al."). The arXiv identifier's own `YYMM` prefix encodes 2026-04, which agrees with the README and contradicts the design document. The wiki holds both; `upstream-lineage` carries the README's version.

The TurboQuant reference in §References (arXiv:2504.19874, ICLR 2026) is not contradicted by any existing page, but no page records it yet either.

### What the document gets right

Worth stating plainly, because the discrepancies above are the loud part: the binary format section matches the shipped writer field-for-field (`src/tools/triattention-calibrate/triattention-calibrate.cpp:303-353` writes exactly that header and record layout, and the loader at `src/src/llama-triattention.cpp:107-285` reads it back, skipping `r_f` as validation-only at `:249`); the KV-cache hook list matches (`apply_ubatch` at `src/src/llama-kv-cache.cpp:1271` with the token/prune hooks at `:1302-1374`, `clear` at `:521`, `seq_rm` at `:551`, `seq_add` at `:741`); the supported KV-type list matches the CPU dequant switch (`:577-620`); and the CUDA data-flow claim (no full-K D2H transfer, one score array back) matches `triattention_gpu_score_head` being fed `kt->data` and the scores copied host-side once.

## Pages derived

[[triattention]] · [[triattention-calibrate]] · [[kv-eviction]] · [[kv-cache]]
[[ta-2-budget-starvation]] · [[ta-5-freq-scale-dead-code]] · [[ta-7-config-validation]] · [[ta-1-wht-inversion-256]]
[[turboquant]] · [[quantization]] · [[benchmarks]] · [[performance-profile]] · [[overview]]

## Provenance

- raw path: `llmwiki/raw/TRIATTENTION.md`
- sha256: `265cb520810c2a3d4226caac0e87c14017f335852f5919bf7d035c2c8c29b9d8`
- ingested: 2026-09-28
