---
title: First live eviction run (TriAttention on the 4B model)
type: topic
status: current
updated: 2026-09-29
sources: [state.md, README.md, TRIATTENTION.md]
verified: [src/src/llama-triattention.cpp, src/src/llama-kv-cache.cpp, src/ggml/src/ggml-cuda/triattention-score.cu, src/common/arg.cpp]
tags: [measurement, triattention, eviction, calibration]
---

# First live eviction run

## Bottom line

TriAttention was executed for the first time on real hardware: a profile was calibrated from the 700k corpus on the 4B model, then loaded into a serving run where the pruner fired **11 times in 400 tokens** and the metric log printed what it was doing. Four of the vault's claims — three of them confirmed defects — became observed behaviour rather than code readings, and two **new** facts surfaced that no page carried.

What the run shows in one line, verbatim from the log:

```
[TriAttention] Pruned: 96 → 64 tokens (32 evicted, 59 protected [prefix=27, recent=32]), 4,85 ms [GPU], pos=128
```

At `budget = 64`, **59 of the 64 retained slots are protected** — 27 prefix tokens plus a 32-token recent window — leaving **five slots for the entire history**. That is [[ta-2-budget-starvation]] happening, not inferred: the mechanism is a sliding window in everything but name, and `prefix=27` never moves ([[ta-10-prefix-length-global-latch]]).

## What was run

| | |
| :--- | :--- |
| Model | `Ternary-Bonsai-4B-Q2_0_g64.gguf` (loaded fine — the profile's mismatched identity did not block it) |
| Engine | the CUDA 13 bundle's `llama-cli` ([[first-live-measurements]]) |
| Profile | `/hdd2/tools/prof/bonsai4b-700k.triattention`, 1 188 931 B, calibrated from the 700 K corpus in **54 m 45 s** (333 chunks) |
| Command | `-ngl 99 -c 1024 -n 400 --ignore-eos -ctk turbo3 -ctv q8_0 --triattention-stats <profile> --triattention-budget 64 --triattention-window 32 --triattention-log` |
| Result | 11 prune events, ~4.8 ms each on GPU (9.5 ms for the first), generation **86.3 t/s** — indistinguishable from the no-eviction run at the same length |

**Eviction is not free but it is not expensive here:** ~53 ms of pruning across a 4.6 s generation, and the throughput matches the 86.6 t/s of the same run without eviction firing. The cost is the GPU scoring pass, and with `budget` this small it touches very few cells.

## The profile the tool actually wrote

Parsed from the file itself (`67 + 1152 × 1032` bytes, **exactly**), and from the log line the runtime prints when it loads it:

| Field | Value |
| :--- | :--- |
| Magic | `AIRT`, version 1 |
| Model name **stored in the profile** | `Bonsai-2-27B-PQ2_0` — **for a 4B calibration** |
| `layers` / `attn_heads` / `kv_heads` | 36 / 32 / 8 (the 4B's own — correct) |
| `head_dim` | **128** |
| `sampled` | 1152 (36 × 32 ✓) |
| `rope_theta` in the profile | **10 000 000** |
| Record size | 1032 B per (layer, head); the 27B's shipped profile is 2056 B — same header form, different geometry |

> **New finding (2026-09-29): the calibrator stamps a fixed model identity into every profile it writes.** The 4B calibration carries the 27B's name *and* the 27B's `rope_theta`. The runtime detects the latter and warns — `[TriAttention] WARNING: rope_theta mismatch (calibration=10000000,0, model=5000000,0)` — but **not** the former: the name is accepted silently, so a profile's provenance cannot be trusted from its own header. Combined with the fact that the shipped profile's format carries no corpus, no token count and no timestamp ([[triattention-calibrate]]), the artifact is effectively unauditable.

> **And the mismatch is not cosmetic.** The whole scoring path is driven by `omega` built from the rope base ([[scoring-correctness]]); a calibration recorded at a base the model does not use means the frequency geometry the statistic was gathered in is not the geometry it is applied in. The warning exists because the runtime knows this matters. It does not stop.

## What this confirms, live

| Claim | Status now |
| :--- | :--- |
| [[ta-8-offset-max-zero-nan]] — `offset_max = 0` makes the aggregate ill-defined | **Observed.** The log prints `offsets=0` under the shipped default, and `offsets=17` once `--triattention-offset-max 65536` is passed. The default is a zero-offset mean — the exact configuration the issue calls NaN-producing. The eviction still ran and the output stayed fluent, which is the point: the defect is **silent** |
| [[ta-10-prefix-length-global-latch]] — `prefix_length` latches per context | **Observed.** Every one of the 11 prune lines reports `prefix=27`, the first prompt's length, at positions 96 through 416 — it never grows as the request proceeds |
| [[ta-2-budget-starvation]] — eviction degenerates to a sliding window | **Observed.** 59 protected of 64 retained, i.e. 5 slots of history at `budget = 64` |
| [[device-placement]] — the scoring path is GPU-side and turbo-aware | **Observed.** `[TriAttention] GPU scoring enabled (k_type=43, heads=1152)` — type 43 is `GGML_TYPE_TURBO3_0` |
| [[ta-1-wht-inversion-256]] — the scorer needs the WHT inverse on the KV side | **Still untested.** The 4B reports `head_dim = 128`, so this run never reaches the `padded_hd == 256` branch the bug lives on. The defect remains invisible from this model and this machine |

The last row is the honest limit of the whole exercise: the one defect the project cares most about is exactly the one a 128-dimensional model cannot exhibit.

## Quality: a weak signal, deliberately not over-read

Both configurations produced fluent, on-topic text. The run with the shipped defaults (`offsets=0`) shows more self-repetition and more doubled terms — `"**Key-Value ( (KV) cache)**"`, `"**transformer-based models, **especially"` — while `--triattention-offset-max 65536` stayed cleaner through the same length. **This is not evidence.** Two single runs, no seed control, no quality metric, and `std::partial_sort` over NaN is undefined behaviour rather than a defined-but-worse ordering — so "the text looked worse" cannot be attributed.

What it does establish is that **the experiment is cheap**: two runs, 11 seconds. The measurement that would settle it — perplexity or a task score with and without eviction — has no page in this vault ([[documentation-coverage]], class A), and that absence is the reason this question is still open after a defect was confirmed twice.

## Open questions

- Does anything downstream depend on the profile's stored `rope_theta`, or is the warning purely advisory? The runtime prints it and proceeds; nothing in this run showed a consequence.
- Why does `--triattention-offset-max 65536` produce **17** offsets — is that the intended geometric set, or a clamp?
- Is the 4B's `head_dim = 128` the reason its profile record is 1032 B while the 27B's is 2056 B? The arithmetic is consistent with 128 versus 256 frequencies, but the header says `head_dim = 128` while the 27B's profile describes a 256-dimension head — so the record size and the head dimension do not relate the way the names suggest.
- With `budget` above the context length, does the pruner ever run at all? In the first attempt (budget 256, 222 cells) it did not fire once — the log was empty, which is itself a useful negative.

## See also

[[triattention-calibrate]] · [[triattention]] · [[ta-8-offset-max-zero-nan]] · [[ta-10-prefix-length-global-latch]] · [[ta-2-budget-starvation]] · [[ta-1-wht-inversion-256]] · [[device-placement]] · [[first-live-measurements]] · [[scoring-correctness]] · [[open-questions]]
