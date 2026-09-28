---
title: Source — AGENTS.md
type: source
status: current
updated: 2026-09-28
sources: [AGENTS.md]
verified: []
tags: [governance, schema, pipeline]
---

# Source — `AGENTS.md`

The repository's agent constitution: static rules, anti-patterns, git conventions, the BMad delivery pipeline, and the LLM Wiki schema. It is both a source of constraints the wiki must respect and the schema layer the wiki runs on.

## Summary

Declares the project **brownfield** with a hard boundary: **do not edit PrismML CUDA kernels**; minimal changes elsewhere. Bans global/singleton LLM clients and DB connections in favour of explicit injection, bans `sudo`/`su`, and requires stopping the application after a test run to free VRAM/RAM/CPU. Fixes commit style (Conventional Commits 1.0.0, English) and branch naming (Conventional Branch 1.0.0, `main` as the base).

It then defines two agent-facing systems: the **BMad delivery pipeline** (`bmad-architecture` → `bmad-ticket` for epics → `bmad-ticket` for stories → one isolated `git worktree` subagent per epic running `bmad-build-auto` per ticket) and the **LLM Wiki** (`llmwiki/`, with `SCHEMA.md` as the operational contract).

## Key claims

| Claim | Where |
| :--- | :--- |
| PrismML CUDA kernels are off limits; everything else changes minimally | Project Constraints |
| Brownfield — explore the codebase independently before writing, never trust this file for structure | Role of this File |
| No singletons/globals for clients and connections; explicit dependency injection only | Anti-Patterns |
| Stop the app after a test run to free VRAM/RAM/CPU | Anti-Patterns |
| Multi-step work runs the fixed BMad pipeline; epic builders get an isolated worktree and a branch, never the shared tree | BMad Delivery Pipeline |
| `bmad-build-auto` handles exactly one ticket per invocation and HALTs on a dirty tree | BMad Delivery Pipeline → Stage 3 |
| `llmwiki/raw/` is immutable; `INDEX` navigation starts at `llmwiki/index.md`; contradictions are annotated, never overwritten | LLM Wiki |

## Constraints the wiki inherits

- Any issue page that proposes touching `src/ggml/src/ggml-cuda/` PrismML kernel files is proposing a change the constitution forbids; the fix must be routed around them. [[prismml-weight-kernels]] carries the boundary explicitly.
- The wiki is written by agents under the same rules as code: conventional commits, English, `docs/`-scoped changes.

## Pages derived

[[overview]] · [[roadmap]] · [[source-llmwiki]] · [[prismml-weight-kernels]]

## Provenance

- raw path: `llmwiki/raw/AGENTS.md`
- sha256: `9021c1db08d94289ffd4b39040b74fa12256bd7f6801149381174fd626277843`
- ingested: 2026-09-28
