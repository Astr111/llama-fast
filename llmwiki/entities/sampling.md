---
title: Sampling
type: entity
status: current
updated: 2026-09-29
sources: [README.md]
verified: [src/common/sampling.h, src/common/sampling.cpp, src/common/common.h, src/common/arg.cpp, src/common/json-schema-to-grammar.cpp, src/include/llama.h, src/src/llama-sampler.h, src/src/llama-sampler.cpp, src/src/llama-grammar.cpp, src/src/llama-graph.cpp, src/src/llama-context.cpp, src/tools/server/server-context.cpp, src/ggml/src/ggml-cuda/top-k.cu, src/ggml/src/ggml-cuda/argsort.cu, src/ggml/src/ggml-cuda/ggml-cuda.cu, scripts/run_cli.sh, scripts/start_server_baseline.sh, scripts/start_server_turbo.sh]
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

## Implementation depth

_Read 2026-09-29 out of `src/src/llama-sampler.cpp` (152 KB), `src/src/llama-grammar.cpp`, `src/common/sampling.cpp`, `src/common/json-schema-to-grammar.cpp`, `src/src/llama-graph.cpp`, and the CUDA `top-k`/`argsort` kernels. This closes the vault's #1 ranked gap. Everything below is from the code; the two places where a body was deliberately not read are marked._

### The chain as `common_sampler_init` actually builds it

`common_sampler_init` (`src/common/sampling.cpp:186-427`) builds a `std::vector<llama_sampler *> samplers` in this order, then adds each to the chain (`:411-413`):

1. **`logit_bias` is prepended, unconditionally, whenever the merged bias list is non-empty** (`:330-337`): `params.logit_bias` plus every token returned by `llama_vocab_get_suppress_tokens(vocab, &n)` at `-INFINITY`. No help text mentions this stage; it is the real first filter whenever a model carries suppress tokens.
2. `if (params.mirostat == 0)` guards the entire `--samplers` list (`:339`), which maps each `common_sampler_type` to an initializer (`:345-382`). One case is not positional: `COMMON_SAMPLER_TYPE_ADAPTIVE_P` only sets `use_adaptive_p = true` (`:383-395`) and is appended **after** the loop.
3. The selector is appended unconditionally: `llama_sampler_init_adaptive_p(...)` if `use_adaptive_p`, else `llama_sampler_init_dist(params.seed)` (`:396-400`). There is no assert — the "chain must end in a selector" invariant holds because the code always adds one on the `mirostat == 0` path.
4. `mirostat == 1|2` **discards `params.samplers` completely** and builds `temp` + `mirostat(n_vocab, seed, tau, eta, 100)` / `mirostat_v2` (`:401-407`).

`common_sampler` stores three things (`:427-436`): `.chain`, plus `.grmr` and `.rbudget` as **standalone `llama_sampler *`** — they are never added to the chain. `common_sampler_free` frees all three (`:444-449`).

Disabled stages stay in the chain as no-ops: an initializer whose parameter is at its "off" value returns `llama_sampler_init_empty("top-k")` etc. (e.g. `src/src/llama-sampler.cpp:1519-1534` for `top-k` with `k <= 0`), so the printed chain keeps every name (`common_sampler_print`, `sampling.cpp:765-769`) while the filter costs nothing.

With defaults and no bias list the real order is therefore

```
penalties → dry → top-n-sigma → top-k → typical → top-p → min-p → xtc → temp-ext → dist
```

in which `dry` (multiplier 0.0), `top-n-sigma` (-1.0), `typical` (1.0) and `xtc` (probability 0.0) are `empty` placeholders — the *effective* chain is `penalties → top-k(40) → top-p(0.95) → min-p(0.05) → temp-ext(0.80) → dist`.

**Where the CLI list is overridden.** `--samplers` (`src/common/arg.cpp:1990-1997`) and `--sampler-seq` (`:2006-2015`) both assign `params.sampling.samplers`, so the last one on the command line wins; unknown names are dropped with a warning rather than an error (`sampling.cpp:881-885`); `--mirostat` erases the list (§1 above); and `--backend-sampling` is silently cleared — `params` is mutated, hence the header's "note: can mutate params" — when a grammar or a reasoning budget is present (`sampling.cpp:415-425`). Per request the server builds a fresh chain from `task.params.sampling`, so an HTTP request's `sampling` object replaces the process defaults for that request (`src/tools/server/server-context.cpp:1720`).

### The algorithms, stage by stage

All stages mutate one `llama_token_data_array` (template `struct llama_sampler_i` with `.accept`/`.apply`/`.backend_*`, `src/include/llama.h:1340-1350`). Selection ends with `cur_p->selected` or `-1` on an empty candidate set.

| Stage | What the body does | Where |
| :--- | :--- | :--- |
| RNG seed | `std::mt19937`; `LLAMA_DEFAULT_SEED` → `std::random_device`, or the system clock when `entropy() == 0` | `llama-sampler.cpp:340-353` |
| temperature | `temp <= 0` → argmax with every other logit set to `-INFINITY` (greedy); otherwise divide all logits by `temp` | `:265-292` |
| softmax | `expf(logit - max_l)` normalised; sorts in place only if asked | `:293-320` |
| top-k | partial sort descending, then `cur_p->size = k`; `k <= 0` returns untouched | `:321-339` |
| sorting | `std::partial_sort` with `a.logit > b.logit` for `npartial <= 128`, else a 128-bucket histogram partial sort over `[-10, 10]` logits | `:135-215` |
| top-p | softmax, then cumulative sum until `>= p` **and** `i + 1 >= min_keep`; if unsorted and `size > 1024`, adaptively sorts 256 candidates and grows `k` until the threshold is crossed | `:1549-1601` |
| min-p | unsorted: keep `logit >= max_logit + logf(p)`; sorted: stop at the first below that bound once `i >= min_keep` | `:1749-1801` |
| typical | softmax + entropy `H`, sort by `abs(-log p - H)` ascending, keep the prefix reaching `p` | `:1910-1967` |
| temp-ext | dynamic temperature: normalised entropy → `powf(normalized_entropy, exponent)` → `min_temp..max_temp`, then re-softmax | `:2133-2202` |
| penalties | ring buffer of the last `penalty_last_n` accepted tokens + counts; repeat penalty is `*= penalty_repeat` for `logit <= 0` and `/= penalty_repeat` above (the paper's divide would make negative logits likelier); then `logit -= count*penalty_freq + (count>0)*penalty_present`; sets `.sorted = false` | `:2919-2980` |
| DRY | step 1 restart-sequence scan, step 2 reverse Z-algorithm for `dry_repeat_count` (clamped by `rep_limit`), step 3 `dry_max_token_repeat` per token, step 4 the penalty with the exponent clamped by `88.72 / log(dry_base)` to avoid `powf` overflow | `:3386-3560` |
| dist | one pass: `exp(logit - max_l)`, `sum_tgt = sum_cum * rnd`, first index whose running sum crosses it, normalising in the same pass; `size == 1` still draws once so the RNG state matches the backend path | `:1150-1220` |
| greedy | first index with a **strictly** greater logit — lowest vocabulary id wins ties | `:1053-1060` |
| top-n-sigma | reject when `n <= 0` or `size <= 1` (`:3237-3240`); the mean/σ body was not read → the exact statistic is `[UNVERIFIED]` |
| xtc / mirostat / mirostat-v2 / adaptive-p | bodies not read; mirostat softmaxes first (`llama_sampler_softmax_impl(cur_p, true)`, `:2463`, `:2576`) and keeps `mu = 2*tau` state | `:2344-2400`, `:2460-2590` |

Tie-breaking is wherever a sort or the candidate order decides it: `std::partial_sort` with a strict comparator is **not** stable (`:136-137`, `:194-195`), `dist` takes the first crossing index (`:1197-1205`), and `greedy` takes the lowest id among equals.

### What runs on the GPU, and what does not

`--backend-sampling` (`-bs`) moves sampling *into the decode graph* — but only a **contiguous prefix** of the chain:

- The server sets the chain on the context per slot: `llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(...))` (`server-context.cpp:1729-1739`; API `src/include/llama.h:1367`), guarded by `task.params.sampling.backend_sampling` and by `!need_pre_sample_logits`. Context creation then calls `llama_sampler_backend_begin` per sampler (`src/src/llama-context.cpp:2026-2028`).
- `llama_sampler_chain_backend_init` walks the chain in order: a stage runs on the backend only if its `iface->backend_init` exists **and** returns true, and the first failure sets `backend_prefix = false` for everything after it (`llama-sampler.cpp:733-768`). So a single CPU-only stage truncates the GPU prefix at that point.
- Stages that implement the backend path (each `backend_init` calls `llama_sampler_backend_support`, which checks every op of a probe graph against the backend's buffer type, `:637-663`; call sites `:1069`, `:1263`, `:1470`, `:1620`, `:1819`, `:2073`, `:2220`, `:3023`): **greedy, dist, top-k, top-p, min-p, temp, temp-ext, penalties** (plus the `empty` placeholder and the chain itself).
- Stages with **no** backend implementation: `typical`, `top-n-sigma`, `xtc`, `dry`, `mirostat`/`v2`, `grammar` (`iface` all-null at `llama-sampler.cpp:2751-2763`), `adaptive_p`, and `logit_bias` — the last declared as `llama_sampler_logit_bias : public llama_sampler_backend` (`:3886`) but with no support-check call site in the file. Since the default chain puts `logit_bias` at position 0 whenever the model's suppress-token list is non-empty, the GPU prefix can be **empty**, i.e. `-bs` still samples entirely on the host. `[INFERENCE]` from `sampling.cpp:330-337` + `llama-sampler.cpp:3886` and the absence of a `backend_init` site.
- The GPU ops: `ggml_top_k` → `ggml_cuda_op_top_k` (`src/ggml/src/ggml-cuda/ggml-cuda.cu:2363-2365`), which runs CUB `top_k_cub` **per row** (`top-k.cu:69-72`) or falls back to `argsort` + `cudaMemcpy2DAsync` of the first `k` columns (`:88-103`); `ggml_argsort` → bitonic kernel for `ncols <= 1024` else CUB (`argsort.cu:224-291`). `dist` is a graph: `soft_max → cumsum → sub(uniform) → step → sum → clamp → scale_bias/cast`, then `get_rows` back to vocabulary ids (`llama-sampler.cpp:1274-1337`), with the uniform drawn on the host and uploaded as an input (`:1338-1360`). The sampler sub-graph is built at `src/src/llama-graph.cpp:4059-4099`, which also writes the surviving `logits`/`probs`/`candidates` back into `t_sampled_logits`/`t_sampled_probs`/`t_candidates`, and special-cases a chain of exactly one `greedy` sampler to read a precomputed `t_dspark_greedy` row instead of building a sampler graph (`:4043-4049`; the producer of that tensor was not read — fork-specific origin `[UNVERIFIED]`).
- In the **default** configuration (no launch script passes `-bs`), every stage above is host code acting on the full vocabulary array: a partial sort of ~150 k logits for `top-k`, another softmax pass for `top-p`, `min-p`, penalties with hash lookups, and DRY's Z-scan over the last N tokens — once per generated token, after the logits have crossed to the host. That is the CPU-side sampler cost [[first-live-measurements]] measures indirectly. Careful reading of that page: its **94 % of wall is CPU** figure (`llmwiki/topics/first-live-measurements.md:83-95`) is a `perf stat` of an F16-KV `n=128` run whose 18.3 G instructions the page attributes to the dequantize/MMQ host path, **not** to sampling. The sampler is clocked separately (`llama_perf_sampler`, `llama-sampler.cpp:4355`; `struct llama_perf_sampler_data`, `llama.h:1622-1625`) and no recorded run reports `t_sample_ms`, so the sampler's share of that 94 % is `[UNVERIFIED]`. What the code does establish is that the token-selection stages are host-side unless `-bs` is on, for the reason above.

### Grammar-constrained sampling

This is the mechanism behind structured output, and the vault had no page for it.

**Compilation.** `common_sampler_init` turns the grammar string into a standalone `grmr` sampler: a `%llguidance` prefix selects `llama_sampler_init_llg(vocab, "lark", ...)` and `GGML_ABORT`s when the build lacks `LLAMA_USE_LLGUIDANCE` (`src/common/sampling.cpp:213-218`); otherwise `llama_sampler_init_grammar(vocab, str, "root")`, or `llama_sampler_init_grammar_lazy_patterns` with trigger patterns/tokens (`:266-271`). `llama_sampler_init_grammar_impl` (`llama-sampler.cpp:2766-2823`) delegates to `llama_grammar_init_impl` (`src/src/llama-grammar.cpp:1209-1314`), which:

1. parses the GBNF text with `llama_grammar_parser::parse` (`:689`) and returns `nullptr` on a parse error or an empty rule set;
2. requires the root symbol (`"root"`) to exist;
3. copies the rule elements into `vec_rules` and **rejects left recursion** via `llama_grammar_detect_left_recursion` (`:957-1010`);
4. expands the start rule's alternates through `llama_grammar_advance_stack` into the initial `stacks` set — the grammar state is a set of stacks, i.e. an NFA simulation over GBNF rules;
5. compiles lazy trigger patterns with `std::regex` (`:1295-1301`).

There is no AOT compilation step and no token-span caching: rule structure and stacks are rebuilt per `llama_sampler_grammar_reset` (`llama-sampler.cpp:2700-2718`).

**Where it vetoes.** The sampler's `.apply` calls `llama_grammar_apply_impl` (`llama-sampler.cpp:2680-2686`); that function (`llama-grammar.cpp:1353-1393`) is the veto:

- returns immediately while `awaiting_trigger` (lazy grammars);
- `allow_eog` is true iff some stack is empty, i.e. the root rule is complete;
- each candidate token is inspected *by its piece*: an EOG token is set to `-INFINITY` unless `allow_eog`; an empty or NUL-leading piece is set to `-INFINITY`; everything else is UTF-8 decoded and handed to `llama_grammar_reject_candidates` (`:930-955`, `:1055-1125`), whose rejects get `cur_p->data[i].logit = -INFINITY`.

**Committing.** `llama_grammar_accept_impl` (`:1396-1447`) advances the stacks from the accepted token's piece (replaying buffered tokens when a lazy trigger fires, `:1404-1440`); an EOG arriving with no empty stack is `GGML_ABORT("fatal error")` (`:1442-1447`).

**Cost and interaction.** The veto walks candidates and calls `token_to_piece` + `decode_utf8` per candidate, so applying it to the full vocabulary is the expensive case. That is why `common_sampler_sample` prefers the lazy shape described above: sample first, accept-verify, and only on failure re-run the chain with the grammar applied (`src/common/sampling.h:67-90`). Grammar is also what forbids GPU sampling: `llama_sampler_grammar_i` has no `backend_*` hooks, and `common_sampler_init` refuses the combination outright (`sampling.cpp:415-419`).

**Fork divergence: none found.** A grep for `triattn|triattention|turboquant|prism|dspark` across `llama-sampler.cpp`, `llama-grammar.cpp` and `json-schema-to-grammar.cpp` returns nothing, and every stage in the sampler file is stock llama.cpp surface. Whether the `llama_sampler_backend_*` machinery is upstream or fork-added is `[UNVERIFIED]` (no git history was consulted).

**Schema → grammar.** `json_schema_to_grammar(const common_json &, bool force_gbnf)` (`src/common/json-schema-to-grammar.cpp:1233-1247`) returns `"%llguidance {}\nstart: %json " + schema.dump()` when built with LLGuidance and `force_gbnf` is false, else builds GBNF from a template table — `PRIMITIVE_RULES` (`boolean`, integer/decimal parts, string, uuid, date-time…, `:234-247`) and `STRING_FORMAT_RULES` (`:249-256`). That string is what a structured-output or tool-call request would feed into the `--grammar` path above. No launch script in the tree enables a grammar, so none of this is exercised by a checked-in configuration.

### Corrections to this page (2026-09-29)

- **§The chain, in order** lists `penalties → dry → top-n-sigma → …` as "the chain". The code prepends a `logit_bias` stage built from the user biases plus `llama_vocab_get_suppress_tokens` (`sampling.cpp:330-337`), and appends `adaptive_p` at the end rather than at its position in the list (`:383-396`). The listed order is the `params.samplers` default, not always the built chain.
- **§Grammar / structured output** says "a grammar sampler is attached first". Refuted: `grmr` and `rbudget` are fields of the `common_sampler` struct (`sampling.cpp:427-436`) and are never passed to `llama_sampler_chain_add` — the add loop iterates only over the `samplers` vector (`:411-413`). They are applied by `common_sampler_sample`/`common_sampler_accept`, gated by `grammar_should_apply` (`:452-464`).
- **§The parameter surface, `--backend-sampling` row** says sampling "moves into `llama_decode`". True only for a contiguous prefix: the first stage without a backend implementation ends it (`llama-sampler.cpp:733-768`), and with the default chain — `dry` second, `typical` fifth, and a possible leading `logit_bias` — only `penalties` (at most) can run there. The flag is also silently cleared when a grammar or reasoning budget is active (`sampling.cpp:415-425`).
- **§The parameter surface, `mirostat` row** implies mirostat joins the chain. It replaces it: the `mirostat != 0` branches ignore `params.samplers` entirely (`sampling.cpp:339`, `:401-407`).

## See also

[[speculative-decoding]] · [[request-lifecycle]] · [[overview]] · [[kv-cache]] · [[performance-profile]] · [[benchmarks]] · [[qwen35-architecture]] · [[server-layer]]