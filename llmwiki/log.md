# llmwiki log

Append-only record of what happened in this vault. Newest last. Format: `## [YYYY-MM-DD] op | subject`.
`grep "^## \[" log.md | tail -5` gives the last five operations.

## [2026-09-28] setup | vault instantiated
Created `llmwiki/` as an Obsidian vault with the three layers from `raw/llmwiki.txt`: `raw/` (immutable snapshots), generated pages by type (`sources/`, `entities/`, `concepts/`, `issues/`, `topics/`), and the schema split between the *LLM Wiki* section of the root `AGENTS.md` (invariants) and `llmwiki/SCHEMA.md` (operational detail). Wrote `llmwiki/lint.sh` for the mechanical pass.

## [2026-09-28] ingest | state.md — TriAttention and TurboQuant defect inventory
Snapshot: `raw/state.md` (sha256 `a5b15a9a…`). Derived one source page, two synthesis topics, and 14 issue pages: `ta-1`…`ta-7` and `tq-1`…`tq-7`. Verified paths against the checkout — state.md's paths are relative to a different tree, so the repo-relative form is `src/src/llama-triattention.cpp` and `src/ggml/src/ggml-cuda/triattention-score.cu`.

## [2026-09-28] ingest | README.md — release surface and measurements
Snapshot: `raw/README.md` (sha256 `05dfb403…`). Derived the source page, `benchmarks`, `upstream-lineage`, and fed `overview`, `performance-profile`, `v100-sxm2`, `ternary-bonsai-2-27b`.

## [2026-09-28] ingest | llmwiki.txt — the pattern this vault implements
Snapshot: `raw/llmwiki.txt` (sha256 `dc3efe98…`). Source page records the three layers, the three operations, and the deliberate deviations this instantiation makes (snapshot raw layer with hashes; `lint` as a script).

## [2026-09-28] ingest | AGENTS.md — constitution and schema
Snapshot: `raw/AGENTS.md` (sha256 `9021c1db…`). Captured the constraints the wiki inherits, notably the PrismML CUDA kernel edit ban, which any fix routed through `[[prismml-weight-kernels]]` must respect.

## [2026-09-28] contradict | published benchmarks are Ampere, deployment target is Volta
First recorded contradiction, and it is a live one: `README.md` reports 1.39× and ~25 200 tok/GB measured on an **RTX 3090**, while `state.md` names a **Tesla V100-SXM2-16GB** as the target — no INT tensor cores. Neither source is wrong; the claim "these numbers describe the deployment target" is unsupported. Filed in `benchmarks.md` and `overview.md` as an open question, not silently reconciled. Also noted: README's native-build arch list omits `sm_70`, which its own CUDA 12.4 release ships.

## [2026-09-28] contradict | the TA-1 fix does not exist on this machine
`state.md` §3 TA-1 and §5 item 1 say the WHT inversion fix lives in `/home/ms/llama-fast/Release/`. That path does not exist here (checked during ingest). Recorded in `issues/ta-1-wht-inversion-256.md` as unverifiable rather than repeated as fact — it changes what "port the fix" (roadmap item 1) actually requires.

## [2026-09-28] lint | initial pass
`llmwiki/lint.sh` written and exercised; the script itself was corrected three times against real findings (frontmatter parse, vault-relative link resolution, repo-relative raw paths in `Provenance`). Findings at this point were only the pages still being written.
