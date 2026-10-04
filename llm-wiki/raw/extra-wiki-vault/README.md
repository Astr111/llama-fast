# LLM Wiki

An autonomous, compounding personal knowledge base built according to the **LLM Wiki Pattern**, fully compatible with [Obsidian](https://obsidian.md).

## Architecture

```text
llm-wiki/
├── CLAUDE.md           # Master Agent Schema & operational protocol
├── AGENTS.md           # Universal LLM Agent instructions (Codex / Pi / Gemini)
├── wiki.py             # CLI shortcut for health checks, search, and stats
├── raw/                # Curated, immutable primary sources
│   ├── articles/       # Web clips, articles
│   ├── papers/         # Academic papers
│   └── notes/          # Field notes, interview transcripts
├── wiki/               # Compiled Markdown Knowledge Base (Obsidian Vault Root)
│   ├── index.md        # Master content catalog
│   ├── log.md          # Chronological append-only audit trail
│   ├── concepts/       # Theories, mechanisms, paradigms
│   ├── entities/       # People, tools, frameworks, systems
│   ├── sources/        # Ingested source summaries with takeaways
│   └── synthesis/      # Cross-cutting analyses and deep dives
└── tools/
    └── wiki_cli.py     # Linter, search engine, graph builder
```

## Opening in Obsidian
Point Obsidian ("Open folder as vault") directly to `llm-wiki/` or `llm-wiki/wiki/`. All notes use standard `[[wikilinks]]` and YAML frontmatter.

## CLI Commands
Run from within `llm-wiki/`:
```bash
# Verify health (broken links, orphans, unindexed files)
python3 wiki.py lint

# View knowledge graph metrics
python3 wiki.py stats

# Full-text search across all notes
python3 wiki.py search "Memex"

# Dump JSON graph topology
python3 wiki.py graph
```
