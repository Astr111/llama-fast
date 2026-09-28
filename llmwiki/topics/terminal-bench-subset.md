---
title: Terminal-Bench 2.0 subset (10 tasks)
type: topic
status: current
updated: 2026-09-29
sources: [state.md, README.md]
verified: []
tags: [benchmark, quality, planning]
---

# Terminal-Bench 2.0 subset (10 tasks)

## Bottom line

[[benchmarks]] records that this project names two evaluation suites — Terminal-Bench 2.0 (MR Subset 39) and Harbor 0.1.x — and then reports **no results from either**. The MR subset is not a mystery: it comes from the paper repository `efficient-benchmarking-ai-agents` (the *Efficient Benchmarking of AI Agents* paper), which ships the per-task data as `terminal_bench_2_0_tasks.csv` plus a precomputed MR subset and a curated ten.

**There are two defensible 10-task subsets, and they overlap in only four tasks:**

| | Tasks | Selection | Character |
| :--- | ---: | :--- | :--- |
| **`mr10-curated`** | 10 | the repository's own hand-curated file `terminal_bench_2_0_top10.csv` | **domain-diverse** — ten distinct domains, and its rationales are written with *this* project in mind (`code-from-image` is chosen because it "directly tests mmproj-Qwen3.8", which is the multimodal projector the launch scripts pass) |
| **`mr10-irt`** | 10 | top 10 of the MR-39 by the repository's own `irt_information` column | **information-maximising** — the statistically most discriminating tasks, but concentrated in AI/ML and systems domains |

Both are valid subsets of the MR-39: the curated ten is a **strict subset** (10/10 inside it) and every task's pass rate lies in the MR band.

**Recommendation: use `mr10-curated`** unless the goal is specifically rank-fidelity measurement, in which case `mr10-irt` is the better instrument. The curated set sacrifices statistical power for the one property that matters when the thing under test has capabilities the benchmark's agents did not: **coverage of modes** — vision, security, virtualization, databases, async Python — so that a failure pattern identifies *which* capability is missing rather than reporting a scalar.

## The data behind it

All figures below are from the paper repository's own files, verified against them; the pass rates are historical agent results, not measurements of this engine.

- `data/terminal_bench_2_0_tasks.csv` — 9 167 rows: per (agent, model, task) trial counts and resolution rates.
- `data/terminal_bench_2_0_mr_subset.csv` — **39 tasks**, each with `pass_rate`, `irt_information` and `dist_from_50`, selected by the paper's Mid-Range Difficulty filter: task pass rates in **[0.30, 0.70]**.
- `data/terminal_bench_2_0_top10.csv` — the curated ten, with a `domain` and a hand-written justification column (the justifications are in Russian; the table below paraphrases them).

The MR band as it actually came out: **min 32.7 %, median 48.5 %, max 68.7 %** — the filter's intent is tasks that neither every agent solves nor every agent fails, because those are the ones that carry ranking information.

### `mr10-curated` — the ten

| Task | Domain | Pass rate | IRT info | Why it was picked |
| :--- | :--- | ---: | ---: | :--- |
| `code-from-image` | Multimodal / vision-to-code | 68.6 % | 0.882 | tests the multimodal projector directly — the agent must read an image and reconstruct clean code |
| `pytorch-model-cli` | AI & ML | 51.5 % | 0.999 | PyTorch structure plus a CLI pipeline and tensor work |
| `llm-inference-batching-scheduler` | LLM systems engineering | 43.2 % | 0.945 | continuous-batching scheduler logic — the discipline this project lives in |
| `configure-git-webserver` | DevOps & sysadmin | 49.9 % | **1.000** | the most informative task in the whole MR subset; local Git HTTP server, permissions, hooks, ports |
| `qemu-startup` | Virtualization & OS | 58.2 % | 0.918 | VM configuration, bridged networking, kernel parameters |
| `fix-code-vulnerability` | Security & auditing | 65.5 % | 0.828 | find an overflow/injection in foreign code and patch it correctly |
| `password-recovery` | Security & forensics | 42.1 % | 0.955 | extract credentials from memory dumps, decrypt local stores |
| `fix-ocaml-gc` | Systems programming | 55.8 % | 0.987 | low-level debugging inside a language runtime's garbage collector |
| `sqlite-db-truncate` | Databases | 48.5 % | 0.999 | truncate a heavy SQLite table fast while preserving referential integrity |
| `cancel-async-tasks` | Algorithms & async | 40.0 % | 0.969 | graceful cancellation of asyncio coroutines |

### `mr10-irt` — the six the curated set omits

| Task | Domain | Pass rate | IRT info |
| :--- | :--- | ---: | ---: |
| `large-scale-text-editing` | text processing | 51.9 % | 0.9986 |
| `reshard-c4-data` | AI & ML (data) | 46.9 % | 0.9962 |
| `bn-fit-modify` | numerical / ML | 54.1 % | 0.9934 |
| `mailman` | systems | 44.9 % | 0.9894 |
| `kv-store-grpc` | distributed systems | 56.0 % | 0.9854 |
| `pytorch-model-recovery` | AI & ML | 43.7 % | 0.9839 |

## Artefacts

Both subsets are written outside the repository, in three formats each, so a run does not depend on the paper checkout:

```
/hdd2/tools/benchmark/mr10-curated.{txt,csv,json}   # 10 tasks, with domains and rationales
/hdd2/tools/benchmark/mr10-irt.{txt,csv,json}       # 10 tasks, ranked by irt_information
```

## What running them would take — and what exists today

| Requirement | Status on this machine |
| :--- | :--- |
| The task definitions | **absent** — the paper repository ships data and analysis only; the tasks themselves live in the Terminal-Bench project |
| A harness to execute a task in a container | **absent** — no `tbench`, no `terminal_bench` Python module |
| A container runtime | **present** — `/usr/bin/docker`, daemon responding |
| An agent scaffold that drives a model through tool use | **absent**, but bridgeable: the engine serves an OpenAI-compatible API through `llama-server` ([[server-layer]]), which is the interface a scaffold expects |
| A model worth scoring | the 4B Q2_0 runs today ([[first-live-measurements]]); the 27B `PQ2_0` needs the target hardware |

So the gap is **the harness and the task definitions, not the hardware** — the same shape as the `llama-bench` finding in [[llama-bench]], and it is the one part of the measurement chain that is a `pip install` plus a clone rather than a purchase.

## The caveat that matters more than the subset

**These tasks were calibrated against frontier agents.** Their pass rates — 40 % to 68 % — are the historical rates of GPT-5-class models under 33 different scaffolds. A 27B ternary model served from a single V100, and certainly the 4B, is a *distribution shift* of exactly the kind the paper studies — and the paper's own finding is that **rank-order prediction survives such a shift while absolute score prediction does not**.

The practical consequence: running these ten against this engine will most likely produce a near-floor score that says little about the engine's quality and a great deal about the model's size. What the subset *can* do here is comparative and within-model: does `-ctk turbo3` score differently from `-ctk f16`? Does eviction at `budget 64` cost task success against eviction off? Those are **paired comparisons on identical tasks**, where the paper's stability result applies and a 4B's floor is a constant that cancels.

That is also the honest answer to the vault's largest remaining documentation gap: the quality question was unmeasurable not because no instrument existed, but because nobody had connected the instrument to the engine. The connection is now specified above; the install is not done.

## Open questions

- Which scaffold would drive this engine? The paper studies 33; the project's own history names Pi Agent ([[benchmarks]]). A scaffold that expects a large context and reliable tool-calling may fail on the 4B for reasons unrelated to the tasks.
- Are the ten tasks' pass rates still current, or does the leaderboard's rolling window move them out of the MR band? The CSV carries a `date` per row; a re-selection would need the latest snapshot.
- Would `mr10-irt`'s statistical advantage survive at n = 10? The paper's MR subset is 39 tasks; shrinking it to 10 is this project's choice, not the paper's recommendation, and the rank-fidelity guarantee is not stated for n = 10.

## See also

[[benchmarks]] · [[performance-profile]] · [[first-live-measurements]] · [[first-live-eviction]] · [[llama-bench]] · [[documentation-coverage]] · [[open-questions]] · [[roadmap]]
