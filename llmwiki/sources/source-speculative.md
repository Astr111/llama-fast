---
title: Speculative Decoding — in-tree doc
type: source
status: current
updated: 2026-09-28
sources: [speculative.md]
verified: [src/common/speculative.cpp, src/common/common.h, src/common/arg.cpp]
tags: [speculative-decoding, dflash, draft-model]
---

# Source: Speculative Decoding (src/docs/speculative.md)

## Summary

The in-tree upstream llama.cpp reference for speculative decoding (479 lines): the implementation catalogue — Draft Model, EAGLE-3, DFlash, DFly, DSpark, and the five n-gram variants (`ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`; plus `draft-mtp` in the CLI table) — the full `--spec-*` command-line surface with its documented defaults, the per-implementation statistics format, and the SPEED-Bench benchmarking client. It is the only place in the tree with measured speculative-decoding tuning data: a `--spec-draft-n-max` sweep, recorded on an M5 Pro with a DFly drafter. It describes the `draft-dflash` block-diffusion path this fork runs; its numbers do not.

## Key claims

- **`## Implementations`** (`:9-10`): `llama-server` supports several speculative-decoding implementations; an implementation *with* a draft model can be mixed with one *without*; when mixed, the draftless decoding takes precedence (`## Command-Line Options` `:246`).
- **`### DFlash (draft-dflash)`** (`:55-79`): DFlash emits an entire block of draft tokens in a single forward pass (block diffusion) and injects the target model's hidden states into the draft model's attention, keeping the draft small and GPU-friendly. `--spec-draft-n-max` is clamped to the draft model's trained block size (`:75`); the worked example uses `--spec-draft-n-max 15` (`:72`). See `#25173` for the DFlash PR.
- **`### DFly`** (`:81-99`): a DFlash variant that runs through `draft-dflash` and is *detected from the checkpoint* rather than selected with its own `--spec-type`. Differs from DFlash in two ways: target features are fused once per draft layer instead of once for the whole draft, and a predecessor correction is applied per block position before the target head.
- **`#### Tuning --spec-draft-n-max`** (`:101-116`): the correction head makes a DFly draft round cost roughly `fixed + k * per_position`, so the optimum is interior and the default (trained block size − 1) is not always it (`:103-105`). The table (`:109-113`): 5 → 35.7 tok/s / 57.5 % acceptance / 4.00 committed tokens per round; 6 → 42.0 / 64.9 % / 4.99; 7 (default for block size 8) → 37.5 / 52.7 % / 4.82. Measured on an **M5 Pro** with the DFly pairing, greedy, interleaved rounds (`:107`). "Sweep it rather than assuming the largest value wins" (`:115`); the optimum depends on how the backend prices a multi-row verify, so it moves with hardware and target (`:115-117`).
- **`### DSpark (draft-dspark)`** (`:120-152`): block-diffusion draft plus a semi-autoregressive Markov head; `--spec-draft-conf-min P` truncates each block at the first position whose predicted acceptance falls below `P` (default 0 = disabled, `:142-143`); only Qwen3-backbone drafts supported (`:145`).
- **`### n-gram Cache` / `### n-gram Map` family / `### n-gram Mod`** (`:156-242`): model-free drafting from token-history statistics; documented defaults n-match 24, n-min 48, n-max 64 (mod) and sizes 12/48, min-hits 1 (map family).
- **`## Command-Line Options`** (`:244-442`): the full CLI reference. Documented defaults: `--spec-draft-n-max` 3 (`:274-276`), `--spec-draft-n-min` 0, `--spec-draft-p-split` 0.10, `--spec-draft-p-min` 0.00, `--spec-draft-ngl` auto (`:265-291`); backend sampling on by default, toggled by `--spec-draft-backend-sampling` / `--no-spec-draft-backend-sampling` (`:248-250`); the `--spec-type` value table incl. `draft-mtp` (`:388-400`).
- **`## Statistics`** (`:450-472`): every implementation prints a stats line — `draft acceptance rate`, and per-implementation `#calls(b,g,a)`, `#gen drafts`, `#acc drafts`, `#gen tokens`, `#acc tokens`, `dur(b,g,a)` (begin/generation/accumulation durations).
- **`## Benchmarking`** (`:476-479`): end-to-end measurement (throughput, latency, acceptance) via the SPEED-Bench client in `tools/server/bench/speed-bench/`, which compares a baseline run against a speculative-decoding run.

## Discrepancies worth holding

- The tuning table is **DFly-on-M5-Pro**, not the fork's DFlash2-on-V100: different drafter lineage (DFly's per-position correction head is what makes cost `fixed + k * per_position`, `:103-104`), different hardware, different target size (Qwen3-8B, not the 27B drafter target). The interior-optimum *lesson* transfers; the three numbers do not, and the page that cites them ([[speculative-decoding]]) says so.
- The doc's DFlash example passes `--spec-draft-n-max 15` (`:72`); this fork's drafter declares `dflash.block_size = 8`, so the clamp (`src/common/speculative.cpp:1739-1746`, `n_draft_max = block_size - 1 = 7`) would reduce 15 to 7 with a warning. The doc's own tuning text says the default for block size 8 is 7 — consistent with the code clamp.
- `--spec-draft-conf-min` is documented (`:142-143`) but **not registered in this tree**: `src/common/arg.cpp` has no such option, and a tree-wide grep finds the string only in the doc (and unrelated vendored code). Passing it in this checkout would fail argument parse. `[UNVERIFIED]` whether a newer upstream checkout registers it.
- The removed legacy family `--draft` / `--draft-n` / `--draft-max` and `--draft-min` / `--draft-n-min` is a hard `arg_removed()` error in `src/common/arg.cpp:4335-4348` ("use --spec-draft-n-max or --spec-ngram-mod-n-max"). The doc itself has moved on (it only documents `--spec-*` names), so the drift bites old launch scripts, not the doc.
- The doc's implementation prose covers Draft Model, EAGLE-3, DFlash, DFly, DSpark and the n-gram family; `draft-mtp` and `draft-simple` appear only in the `--spec-type` table (`:388-400`).
- The doc is upstream-generic: it never mentions DFlash2's selector/lattice drafting ([[qwen3-dflash-draft]]), which is the fork's actual variant, nor any V100 or TurboQuant context.

## Pages derived

[[speculative-decoding]] — carries the appended section "What the in-tree design doc adds", which links [[qwen3-dflash-draft]], [[cuda-graphs]], [[performance-profile]], [[benchmarks]].

## Provenance

- raw path: `llmwiki/raw/speculative.md`
- sha256: `edb13b37ec417102faa829d9ae3d182cc646e0bd076c716766b13f310c550c43`
