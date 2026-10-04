---
title: Source — llmwiki.txt (the pattern)
type: source
status: current
updated: 2026-09-28
sources: [llmwiki.txt]
verified: []
tags: [meta, schema]
---

# Source — `llmwiki.txt`

The pattern this vault implements: "LLM Wiki — a pattern for building personal knowledge bases using LLMs". It is the origin document for [[overview]]'s structure and for the schema in `AGENTS.md`.

## Summary

The core move is to stop treating documents as RAG fodder and instead have the LLM **incrementally build and maintain a persistent wiki** between the user and the raw sources. Knowledge is compiled once and then kept current, rather than re-derived per query: cross-references already exist, contradictions are already flagged, synthesis already reflects everything read.

Three layers: **raw sources** (immutable, the LLM reads but never writes), **the wiki** (LLM-owned markdown), and **the schema** (`AGENTS.md`-style configuration that makes the LLM a disciplined maintainer). Operations are **ingest**, **query**, and **lint**. Two special files — `index.md` (content catalog, read first on every query) and `log.md` (append-only, greppable via a `## [date] op | subject` prefix) — carry navigation. Optional tooling (a search engine such as `qmd`, Marp decks, Dataview) comes later, only when scale demands it.

The human curates sources, directs analysis, and asks good questions; the LLM does the bookkeeping. Obsidian is the IDE, the LLM is the programmer, the wiki is the codebase.

## Key claims

| Claim | Where |
| :--- | :--- |
| The wiki is a persistent, compounding artifact — maintenance cost near zero is what keeps it alive where human-maintained wikis die | The core idea; Why this works |
| A single source commonly touches 10-15 wiki pages | Operations → Ingest |
| Query answers are themselves knowledge and should be **filed back** as pages rather than left in chat | Operations → Query |
| Lint looks for contradictions, stale claims, orphan pages, concepts mentioned without a page, missing cross-references, data gaps | Operations → Lint |
| `index.md` suffices to ~100 sources / hundreds of pages without embedding infrastructure | Indexing and logging |
| The whole thing is a git repo of markdown: version history, branching, collaboration for free | Tips and tricks |

## How this vault instantiates it

- Raw: `llm-wiki/raw/` — snapshots with recorded sha256, never edited after ingest.
- Wiki: `llmwiki/{sources,entities,concepts,issues,topics}/`, plus `index.md` and `log.md`.
- Schema: the *LLM Wiki* section of the root `AGENTS.md` (invariants) and `llmwiki/SCHEMA.md` (operational detail).
- Deviations from the pattern, deliberate: the raw layer is a **snapshot** (`cp -p` + hash) rather than a pointer to files the user keeps editing, so drift is detectable; and `lint` is a script (`llmwiki/lint.sh`) rather than a prose instruction.

## Pages derived

[[overview]] · [[source-agents-md]]

## Provenance

- raw path: `llm-wiki/raw/llmwiki.txt`
- sha256: `dc3efe98ae62f23dd08acad13aba2e95287beb20b6bec2f4af0423557fe37401`
- ingested: 2026-09-28
