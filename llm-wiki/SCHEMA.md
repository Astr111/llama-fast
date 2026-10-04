# llmwiki — Schema

Operating conventions for this knowledge base. Load before any ingest, query, or lint operation.
The invariants are in `AGENTS.md` → *LLM Wiki*; this file is the operational detail behind that trigger.

## Three layers

| Layer | Where | Who writes |
| :--- | :--- | :--- |
| Raw sources | `llmwiki/raw/` | Nobody. Immutable snapshots; the LLM only reads them |
| Wiki | `llmwiki/{sources,entities,concepts,issues,topics}/` | The LLM only |
| Schema | `AGENTS.md` (invariants) + this file (detail) | Co-evolved by user and LLM |

`llmwiki/index.md` is the content catalog. `llmwiki/log.md` is the chronological, append-only record.
The vault root is `llmwiki/` — open it directly in Obsidian; graph view is the health check for shape.

## Naming

- Filenames: kebab-case, **globally unique across the vault** — Obsidian `[[wikilinks]]` resolve by basename, so a duplicate silently links to the wrong page.
- One concept per page. A page that needs "and" in its title is two pages plus a link.
- Links: `[[page-name]]` or `[[page-name|display text]]`. Never link to a heading anchor; if a heading is worth linking, it is worth a page.
- Cross-link in both directions. A page that cites nothing and is cited by nothing is an orphan — see *Lint*.

## Frontmatter

Every generated page except `index.md` and `log.md`:

```yaml
---
title: TriAttention
type: entity | concept | issue | topic | source
status: current | draft | stale | superseded
updated: 2026-09-28        # YYYY-MM-DD, last substantive edit
sources: [state.md, README.md]   # raw/ basenames this page is derived from
verified: []               # repo paths whose claims were checked against the code itself
tags: [kv-cache, eviction]
---
```

- `status: stale` when a source it depends on has moved on and the page has not been re-read. `superseded` when a newer page replaced it (link the successor in the body).
- `verified` is not decoration: an empty list means every claim came from prose. Fill it whenever a claim was read out of the code.

## Page types and required sections

**source** (`sources/source-<raw-basename>.md`) — one per raw file.
`## Summary` · `## Key claims` (each with the section it came from) · `## Pages derived` (links) · `## Provenance` (raw path, sha256 of the snapshot, ingest date).

**entity** (`entities/<thing>.md`) — a named subsystem, model, or piece of hardware.
`## What it is` · `## How it works` (the mechanism, not the marketing) · `## Where it lives` (repo paths) · `## Known issues` (links) · `## See also`.

**concept** (`concepts/<idea>.md`) — a technique or abstraction, not tied to one file.
`## Definition` · `## Why it matters here` · `## Tradeoffs` · `## See also`.

**issue** (`issues/<ID>-<slug>.md`) — one tracked defect, ID from the source (`TA-1`, `TQ-3`). One page per issue; never a combined table.
`## Symptom` · `## Cause` · `## Impact` · `## Location` (path + symbol, verified) · `## Status` · `## Fix sketch` · `## See also`.

**topic** (`topics/<slug>.md`) — synthesis across pages.
`## Bottom line` (the current best understanding, in prose) · `## Evidence` · `## Open questions` · `## See also`.

## Evidence discipline

1. Every substantive claim cites its origin: `([[source-state-md]])` for prose, `` `path/to/file.cu:123` `` for code.
2. A path or symbol copied from prose is **unverified** until read. Read it, then record it in `verified:`. A claim that cannot be verified is marked `[UNVERIFIED]` inline rather than dropped.
3. When a new source contradicts an existing page, do not overwrite silently: leave the old claim with a `> Contradiction (YYYY-MM-DD):` note naming both sources, and record it in `log.md`. The contradiction is real knowledge until the user settles it.
4. Never invent a value to fill a field. Missing = absent, or an explicit open question.

## Operations

### Ingest

1. Drop the file into `llmwiki/raw/` (never edit it afterwards). Copy with `cp -p` so the mtime survives; record `sha256sum`.
2. Read it in full. Discuss the key takeaways with the user before writing anything — the user decides emphasis.
3. Write/refresh `sources/source-<basename>.md`.
4. Update every page the source touches — entity, concept, issue, topic. One source commonly touches 10-15 pages. Fix frontmatter `sources:`/`updated:` on each.
5. Update `index.md` (add new rows, revise summaries of touched pages).
6. Append to `log.md`.
7. Re-run *Lint* on the touched pages before declaring the ingest done.

### Query

Read `index.md` first, follow into the relevant pages, then synthesize. Cite pages inline. If the answer produced new knowledge — a comparison, an analysis, a resolved contradiction — **file it back as a page** (usually `topics/`) and log it; an answer that lives only in chat is lost work.

### Lint

Mechanical pass, run every session that touches the wiki and on request:

```bash
# pages whose frontmatter is malformed or missing a required key
# broken links: [[target]] with no matching file
# orphans: pages with no inbound link other than index.md
# stale: sources: hashes that no longer match raw/ snapshots
```

The lint is a script, not a vibe: `llmwiki/lint.sh` (or the ad-hoc equivalent when it has not been built yet) reports, the LLM fixes, then reports what it fixed. Kinds of findings: contradictions between pages, stale claims, orphan pages, missing pages for important concepts (a `[[link]]` to a nonexistent file *is* a request to create it), missing cross-references, unanswered questions worth a web search.

## log.md format

Append-only, newest last, one entry per operation:

```
## [2026-09-28] ingest | state.md — TriAttention and TurboQuant issue inventory
## [2026-09-28] query  | why is magma_sgemmEx 38% of GPU time
## [2026-09-28] lint   | 3 orphans, 1 stale source hash
```

`grep "^## \[" log.md | tail -5` gives the last five operations.

## index.md format

Grouped by page type. One row per page: link, one-line summary, source count, updated date. The LLM rewrites the touched rows on every ingest; it is a catalog, not a narrative.
