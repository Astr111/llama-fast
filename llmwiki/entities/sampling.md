---
title: Sampling
type: entity
status: current
updated: 2026-09-28
sources: [README.md]
verified: [src/common/sampling.h, src/common/sampling.cpp, src/common/common.h, src/common/arg.cpp, src/include/llama.h, src/tools/server/server-context.cpp, scripts/run_cli.sh, scripts/start_server_baseline.sh, scripts/start_server_turbo.sh, src/src/llama-sampler.h]
tags: [token-selection, decoding]
---

# Sampling

## What it is

The **token-selection stage** between the model's logits and the emitted token: a chain of filters (penalties, truncation, temperature, …) applied in a fixed order, ending in one sampler that actually picks the token. In this fork the chain is the stock llama.cpp `common_sampler`: a `llama_sampler_chain` of `llama_sampler` objects, constructed per request and reused across the request's turns. The vault previously had **zero** pages mentioning the `sampler` machinery; this page documents the surface that makes recorded decode numbers reproducible.

## How it works

### The chain, in order

The chain is a `std::vector<enum common_sampler_type>` whose **default order** is set in `src/common/common.h:261-271`:

```
penalties → dry → top-n-sigma → top-k → typical-p → top-p → min-p → xtc → temperature
```

and `common_sampler_init()` (`src/common/sampling.cpp:187`) walks that list, pushing the matching `llama_sampler_init_*` object per stage (`:353-399`): `top_k`/`top_p`/`min_p`/`xtc`/`typical`/`temp_ext`/`infill`/`penalties`/`dry`/`top_n_sigma`/`adaptive_p`, then registers them with `llama_sampler_chain_add` (`:411-413`). The chain **must end in a token selector**; outside the sampler-list case the selector is chosen by branch:

- default missing selector → `llama_sampler_init_dist(seed)` (multinomial, `:399`),
- `mirostat == 1` → `temp` + `llama_sampler_init_mirostat` (`:402-403`),
- `mirostat == 2` → `temp` + `llama_sampler_init_mirostat_v2` (`:405-406`),
- `adaptive-p` only when explicitly added by the user (`:396`).

The full inventory of available stage initializers is declared in `src/include/llama.h:1403-1563`: `greedy`, `dist`, `top_k`, `top_p`, `min_p`, `typical`, `temp`, `temp_ext`, `xtc`, `top_n_sigma`, `mirostat`/`mirostat_v2`, `grammar`, `grammar_lazy_patterns`, `penalties`, `dry`, `adaptive_p`, `logit_bias`, `infill`. The core `llama_sampler_chain_params` struct carries only `no_perf` (`src/include/llama.h:462-464`); everything else is `common_params_sampling` in the common layer.

**Fork divergence: none found.** A `grep` for `triattn|triattention|turboquant|wht|prism` across `src/common/sampling.cpp` and `src/tools/server/server-context.cpp` returns nothing — no TriAttention- or TurboQuant-aware logit filter or sampler was added. The chain, its defaults and its CLI are the upstream llama.cpp surface; the only delta is *use*: the fork wires `--reasoning-budget` into the default server launch (see [[#the-parameter-surface]]). Compared to upstream this is stated as *nothing found* rather than proven — no git diff was run.

### Grammar / structured output

`common_sampler` extends the core chain with grammar support (`src/common/sampling.h:13-20`): a grammar sampler is attached first and, when its lazy verify fails, the grammar is applied and the token resampled (`common_sampler_sample`, `:106-118`). Construction: bounded grammars via `llama_sampler_init_llg` when built with `LLAMA_USE_LLGUIDANCE` (`src/common/sampling.cpp:215-218`, abort otherwise), else `llama_sampler_init_grammar` or the lazy variant (`:266-271`). The `llguidance.md` doc exists in the docs tree per a sibling report but was not opened; no launch script enables a grammar, so this path is not exercised by any checked-in configuration.

### Who calls it

- Server per-request: `slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling))` at `src/tools/server/server-context.cpp:1720` — each request task carries its own `task.params.sampling` (`:4505`), so sampling is per-request, initialized once per slot.
- The `common_sampler_sample_and_accept_n` family (`src/common/sampling.h:77-92`) cross-checks sampled tokens against a batch of draft tokens and accepts the common prefix — that is the hook by which [[speculative-decoding]] verification and sampling share one chain.

### The parameter surface

Defaults as set by the code (`src/common/common.h:224-271`), exposed by `src/common/arg.cpp:1990-2311`:

| CLI flag | Default | Note |
| :--- | :--- | :--- |
| `--temp`/`--temperature` | 0.80 | <= 0 = greedy |
| `--top-k` | 40 | <= 0 = disabled |
| `--top-p` | 0.95 | 1.0 = disabled |
| `--min-p` | 0.05 | 0.0 = disabled |
| `--top-nsigma`/`--top-n-sigma` | -1.00 | -1.0 = disabled |
| `--typical`/`--typical-p` | 1.00 | 1.0 = disabled |
| `--xtc-probability` / `--xtc-threshold` | 0.00 / 0.10 | 0.0 = disabled |
| `--repeat-penalty` / `--repeat-last-n` | 1.00 / 64 | 1.0 / 0 = disabled |
| `--dry-multiplier` / `--dry-base` | 0.0 / 1.75 | 0.0 = disabled |
| `--samplers` / `--sampler-seq` | default chain above | `;`-separated names / char sequence |
| `-s`/`--seed` | random (`LLAMA_DEFAULT_SEED`) | |
| `--grammar` | none | GBNF / llguidance |
| `--mirostat` (+`--mirostat-tau` 5.0, `--mirostat-eta` 0.10) | 0 | 0 = disabled |
| `--adaptive-p` target/decay | -1.0 / 0.90 | negative = disabled |
| `--reasoning-budget` | -1 | -1 = unrestricted, 0 = immediate end; `src/common/arg.cpp:3706-3712` |
| `--backend-sampling` (`-bs`) | off | when on, sampling moves into `llama_decode` (`src/common/arg.cpp:2311`; disabled when grammar or reasoning-budget is active, `src/common/sampling.cpp:415-424`) |

CLI values are not the whole story — several args set a `user_sampling_config` bitfield (`src/common/arg.cpp:2007, 2013, 2019, …`), so the server can tell explicitly-set values from defaults.

**The launch scripts pass (almost) none of this.** `scripts/run_cli.sh`, `scripts/start_server_baseline.sh` and `scripts/start_server_turbo.sh` were read in full: the only sampling-adjacent flag anywhere is `--reasoning-budget 4000 --reasoning-budget-message …` in `scripts/start_server_turbo.sh`. No `--temp`, `--top-k`, `--top-p`, `--min-p`, `--seed`, `--samplers`, `--repeat-*`, `--dry-*` or grammar flag appears in any script, so every checked-in launch profile runs the **default chain with defaults** (except the turbo server's 4000-token thinking budget). The scripts set TriAttention/KV flags (`--triattention-*`, `-ctk turbo3`, `-ctv q8_0`) and `GGML_CUDA_GRAPH_OPT=1` — none of which touch token selection. Any reproduced run therefore uses stock upstream defaults unless the operator passed flags.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/common/sampling.h` | `common_sampler` API: init, accept, sample, candidate access, grammar/llguidance hook (`llama_sampler_init_llg`) |
| `src/common/sampling.cpp` | chain construction `:187+`, default-selector branches `:396-409`, reasoning-budget sampler `:311-321`, `backend_sampling` interplay `:415-424`, type↔name maps `:824-844` |
| `src/common/common.h` | `common_params_sampling` defaults `:224-271`; `common_sampler_type` enum `:115-129` |
| `src/common/arg.cpp` | CLI surface `:1990-2311`, reasoning-budget `:3706-3718` |
| `src/src/llama-sampler.{h,cpp}` | core sampler implementations; `dry_testing` extra in the header `:41` |
| `src/include/llama.h` | `llama_sampler_chain_params` `:462-464`; all `llama_sampler_init_*` declarations `:1403-1563` |
| `src/tools/server/server-context.cpp` | per-slot sampler init `:1720`, sampler-params log `:1742`, draft-acceptance telemetry `:615-636` |

## Known issues

- No issue page tracks the sampling path: neither defect inventory ([[source-state-md]] §3, §4) names a sampler defect, and `sampling.cpp` is untouched by the fork's greps.
- [[ta-2-budget-starvation]] coupling: with context evicted to starvation, the drafter's predictions miss and acceptance collapses — the throughput symptom is observed at the sampler/draft boundary (`common_sampler_sample_and_accept_n`), not inside the chain itself.

## Measured context

The acceptance statistics that make the recorded throughput curve legible are computed in the server's `print_timings()`: `draft acceptance = accepted/generated`, `mean len = 1 + accepted/verif_steps`, and an `acc per pos` series — `src/tools/server/server-context.cpp:617-636` (sibling report puts it at `:615-636`; read here at `:617-636`). No archived run output exists anywhere in the repo, so the recorded **31.6 → 38.1+ t/s** effective-throughput trend ([[source-state-md]] §1.2, carried by [[speculative-decoding]]) cannot be reproduced from anything checked in — it is `[UNVERIFIED]` against this tree, like the `40 → 69 tok/s` figure. See [[performance-profile]] and [[benchmarks]] for what is and is not measurable in repo.

## See also

[[speculative-decoding]] · [[request-lifecycle]] · [[overview]] · [[kv-cache]] · [[performance-profile]] · [[benchmarks]] · [[qwen35-architecture]]