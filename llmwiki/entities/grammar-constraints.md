---
title: Grammar constraints
type: entity
status: current
updated: 2026-09-29
sources: [README.md]
verified: [src/common/arg.cpp, src/common/common.h, src/common/json-schema-to-grammar.cpp, src/common/peg-parser.cpp, src/common/peg-parser.h, src/common/sampling.cpp, src/common/sampling.h, src/common/speculative.cpp, src/common/llguidance.cpp, src/src/llama-grammar.cpp, src/src/llama-grammar.h, src/src/llama-sampler.cpp, src/include/llama.h, src/tools/server/server-context.cpp, src/tools/server/server-schema.cpp, src/tools/completion/completion.cpp, llmwiki/entities/sampling.md]
tags: [grammar, structured-output, sampling, tokenizer]
---

# Grammar constraints

## What it is

The mechanism that makes generated text **provably** match a grammar (GBNF) or a JSON schema: a pushdown-automaton walk over the GBNF rule set that vetoes tokens, plus the schema→GBNF compiler that manufactures a GBNF string from a JSON-Schema document. Two engines share the word "grammar" in this tree and they are not the same engine:

| Engine | Type | Role |
| :--- | :--- | :--- |
| `llama_grammar` (`src/src/llama-grammar.cpp`) | GBNF pushdown automaton, `LLAMA_GRETYPE_*` elements, stack sets | **constrains sampling** — the subject of this page |
| `common_peg_parser` (`src/common/peg-parser.cpp`) | PEG combinator builder + Aho–Corasick automata | **generates** lazy GBNF and **parses** model output for tool calls/reasoning — see [[chat-templates]] |

The PEG engine is not a second sampler-side implementation: it has no stacks, no `LLAMA_GRETYPE`, and never appears in `llama-sampler.cpp`; it *emits* GBNF text (`common_peg_parser_builder::build_grammar`, `peg-parser.h:331`) and calls the same `json_schema_to_grammar` front end for `p.schema(...)` (`peg-parser.h:491-493`; include at `peg-parser.cpp:4`). Its trigger rules exist precisely so that only the tool-call subtree becomes a **lazy** grammar (`peg-parser.h:495-525`).

> Refuted expectation (2026-09-29). The brief for this page listed four launch flags including `--json-schema-to-grammar`. **No such flag exists.** A tree-wide grep for the identifier returns only the internal function, the Python sibling `examples/json_schema_to_grammar.py`, and its CMake entry. The real flag set is four entries with two spellings each, below.

## How it works

### The chain: flag → compilation → filter → token

| Stage | Site | What happens |
| :--- | :--- | :--- |
| `--grammar GRAMMAR` | `arg.cpp:2272-2278` | `params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, value}` |
| `--grammar-file FNAME` | `arg.cpp:2279-2285` | same, with `read_file(value)` |
| `-j, --json-schema SCHEMA` | `arg.cpp:2286-2292` | `json_schema_to_grammar(json::parse(value))` → `COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT` |
| `-jf, --json-schema-file FILE` | `arg.cpp:2293-2309` | same, schema read from a file |
| server/API `json_schema` field | `server-schema.cpp:260-264` | converts inside `try`, sets `COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT`; conversion failure is caught here |
| template-driven tool calls | `chat.cpp:3801-3805` | `inputs.json_schema` wins over `inputs.grammar`; both land in `params.grammar` |
| build the sampler | `sampling.cpp:212-276` | `common_grammar_value()` → `%llguidance` prefix routes to the LLGuidance sampler (`:213-218`), else `llama_sampler_init_grammar[_lazy_patterns]` (`:265-271`) |
| prefill | `sampling.cpp:294-308` | for `OUTPUT_FORMAT`/`TOOL_CALLS` only (`common.h:217-221`), the generation-prompt tokens are fed to the grammar so it starts mid-document; **user grammars are deliberately not prefilled** |
| GBNF parse | `llama-grammar.cpp:1209-1314` | `llama_grammar_parser::parse`, rule-ref validation (`:1143-1152`), left-recursion detection (`:957-1012`) |
| runtime filter | `llama-sampler.cpp:2680-2686` → `llama_grammar.cpp:1353-1393` | per-candidate veto: `-INFINITY` on rejects |
| accept | `llama-sampler.cpp:2673-2678` → `llama-grammar.cpp:1396-1450` | stacks advance by the accepted token's piece |

The grammar sampler is **not** a member of the sampler chain. `common_sampler_init` puts `grmr` in a field of `common_sampler` and adds only the `params.samplers` list to the chain (`sampling.cpp:411-413`); `grmr` is applied by `common_sampler_sample`/`common_sampler_accept` through the `grammar_should_apply` gate (`sampling.cpp:452-464, 634-676`). [[sampling]] records this refutation in full.

**Where it sits relative to temperature/top-p depends on `grammar_first`, and the two orders enforce different things** (`sampling.h:64-68`):

- `grammar_first = true` (default `false`; used by the speculative paths, `speculative.cpp:1055, 1515, 2048, 2067, 2440`): the grammar is applied to the **raw full candidate array before the chain** (`sampling.cpp:634-636`), i.e. before temperature, top-k, top-p, min-p and `dist`. Grammar has the last word in the strong sense: no token the grammar rejects can be sampled, whatever the truncation settings.
- `grammar_first = false` (server `server-context.cpp:3770` and completion `completion.cpp:668` use the default): the chain samples **first** under temperature/top-p, then the single winner is tested against the grammar (`sampling.cpp:646-658`); on failure the stored logits are restored, `grmr` is applied to the whole array, and the chain is re-run once (`:660-676`). Consequence: the first draw is drawn from the unconstrained distribution, so its rejection probability is the mass the grammar removes; with a tight schema on a large vocabulary this is a redraw per generated token, and the redraw is a full second chain evaluation `[INFERENCE]`. The upside is that the common (accepted) case costs one `token_to_piece` + one code-point walk rather than a full-vocabulary pass.

For this fork the ordering has a second, non-obvious effect: **a grammar disables backend (GPU) sampling** — `llama_sampler_grammar_i` has null `backend_init/accept/apply` (`llama-sampler.cpp:2758-2760`), so the on-device prefix of the chain ends at it, and `common_sampler_init` clears `params.backend_sampling` with a warning when a grammar is present (`sampling.cpp:415-419`). The interaction with [[runtime-switches]]' `--backend-sampling` is therefore not "sampling on GPU, constrained" but "constrained ⇒ host sampling".

### The JSON-Schema subset

`json_schema_to_grammar(const common_json &, bool force_gbnf)` (`json-schema-to-grammar.cpp:1233-1247`) either returns `"%llguidance {}\nstart: %json " + schema.dump()` (only when built with `LLAMA_USE_LLGUIDANCE` and `force_gbnf == false`) or compiles to GBNF from `PRIMITIVE_RULES` (`:234-247`) and `STRING_FORMAT_RULES` (`:249-256`). **The ignored column is the dangerous one**: an ignored keyword produces a grammar that accepts a *superset* of the schema, with no diagnostic.

| Keyword | Status | Where | Behaviour |
| :--- | :--- | :--- | :--- |
| `type` (string) | supported | `:1071-1082` | dispatch to `boolean/integer/number/string/object/array/null` primitives |
| `type` (array) | supported | `:926-935` | rewritten as a union of one schema copy per listed type |
| `properties`, `required` | supported, order-free | `:945-966`, `:727-747` | required/optional split; **key order is unconstrained**, and duplicate keys are not rejected |
| `additionalProperties` (`true`/`false`/schema) | supported | `:945-947, 740-741, 964-965` | `false` emits a not-string trie rule for unknown keys (`:743-745`) |
| `enum`, `const` | supported | `:935-944` | literal alternatives |
| `$ref`, `#/…` pointers, remote `https://` bases | supported | `:697-708`, `:838-892` | `_refs` map + JSON-pointer walk; unresolvable → **error** |
| `oneOf`, `anyOf` | supported | `:918-925` | union |
| `allOf` | approximated | `:967-1014` | only `$ref`/`properties`/`enum`/`anyOf` members are merged; enum intersection requires the value in *every* member (`:1005-1009`) |
| `items` (array form), `prefixItems`, `minItems`, `maxItems` | supported | `:1016-1034` | tuple form supported |
| `pattern` | approximated | `:1036-1038`, `:391-398` | regex→GBNF; a regex feature the converter cannot express is a **warning** and the field becomes any string (`:392`) |
| `format: uuid[1-5]` | approximated | `:1039-1041, 243` | any hex UUID; version semantics dropped |
| `format: date / time / date-time` | supported | `:1042-1044`, `:249-256` | literal character classes |
| `format: email / uri / hostname / ipv4 / ipv6` | **ignored** | `:1217-1224` | recognised only by `resolves_to_string` (a type-inference helper used by the PEG path); **no GBNF rule is emitted**, so any string passes |
| `minLength`, `maxLength` | supported only with an explicit `"type": "string"` | `:1046-1050` | the guard is `schema_type == "string"`, unlike the `is_null()` guards around it; without the type the keywords fall through to `:1074-1077` and the field becomes **any JSON value** |
| `minimum`, `exclusiveMinimum`, `maximum`, `exclusiveMaximum` | supported for `"type": "integer"` only | `:1052-1070` | string arithmetic over decimal digits (`:87-152`) |
| same four on `"type": "number"` | **ignored** | `:1079-1082` | no branch matches; the plain `number` primitive is emitted (the TODO at `:1083` admits it) |
| `maxProperties`, `minProperties` | **ignored** | — | no occurrence in the file |
| `multipleOf` | **ignored** | — | no occurrence |
| `uniqueItems`, `contains` | **ignored** | — | no occurrence |
| `patternProperties`, `propertyNames` | **ignored** | — | no occurrence |
| `not` | **ignored** | — | no occurrence (`_not_strings` is an internal helper for *additionalProperties*, not the `not` keyword) |
| `if` / `then` / `else`, `dependencies`, `dependentRequired` | **ignored** | — | no occurrence |
| `title`, `description`, `default`, `examples` | ignored (annotations) | `:1074-1077` comment | an untyped schema carrying only annotations becomes the generic `value` rule |

The "ignored" rows are established by a whole-file grep for each identifier, not by reading every branch: the only hits were the internal `_not_strings` helper, the `description` mention inside the comment at `:1075`, and the `allOf`/`anyOf`/`oneOf`/`is_array` machinery. Numeric bounds on `number` are the one case where the code says so itself (`:1083`).

Approximations that can make a **schema-legal value unrepresentable**, hence a dead end at generation time:

- `integral-part` is `[0] | [1-9] [0-9]{0,15}` and `decimal-part` is `[0-9]{1,16}` (`:236-238`) — an integer with more than 16 digits, or a fraction with more than 16, has no derivation.
- strings exclude `\x7F` and `\x00-\x1F` except through escapes (`:244`) — legal-but-unusual JSON must be escaped, so a model that emits a raw control byte gets vetoed.

### Cost: no mask, candidates are tested

Representation: `llama_grammar` holds `rules` (a `vector<vector<llama_grammar_element>>`), a `stacks` set (each stack a vector of **pointers into `rules`**), `partial_utf8`, lazy/trigger state (`llama-grammar.h:100-140`). Compilation is once per grammar object, with rule-ref validation and a left-recursion DFS at init (`llama-grammar.cpp:1143-1171, 957-1012`). **But the GBNF string is re-parsed on every reset** — `llama_sampler_grammar_reset` calls `llama_grammar_init_impl` again from the stored `grammar_str` (`llama-sampler.cpp:2700-2718`), so every new sequence pays a full parse, not a copy. `llama_grammar_clone_impl` (`llama-grammar.cpp:1323-1350`) is the cheap path.

Per token there is **no token mask and no cache**: `llama_grammar_apply_impl` (`llama-grammar.cpp:1353-1393`) walks the *current candidate array*, calls `vocab->token_to_piece(id)` and `decode_utf8` for each entry, and passes the decoded candidates to `llama_grammar_reject_candidates` (`:938-955`) → `..._for_stack` (`:1055-1125`), which recurses per code point, filtering the candidate set against each stack and calling `llama_grammar_advance_stack` (`:855-935`) — the latter maintaining a `std::set` of stacks compared by pointer to deduplicate. Cost is therefore ≈ `#candidates × piece length` UTF-8 decoding, multiplied by the number of stacks, **per generated token**, with string allocation per candidate from `token_to_piece`. The measurement is still open: `// TODO: measure grammar performance` (`sampling.cpp:542`).

So the answer to "cheap or expensive" is: **it is entirely determined by `grammar_first` and by whether the first draw was rejected.** With the default order and a schema whose language covers most of the model's mass it is ~one token piece decode per step; with the veto first, or with a tight schema on a 100k+ vocabulary, it is a full-vocabulary decode + automaton walk per token, on the host, and it also costs the GPU-sampling path. [[sampling]] already records the absence of token-span caching; this page adds that the expense is linear in the *number of candidates presented*, which is why the ordering choice is the real cost knob.

## Where it lives

- `src/src/llama-grammar.cpp` — parser (`:689`), automaton (`:750-935`), candidate rejection (`:938-1125`), init/validate (`:1128-1314`), apply/accept (`:1353-1480`).
- `src/src/llama-grammar.h` — element types, `llama_grammar`, `llama_grammar_candidate`, `llama_grammar_trigger_pattern`.
- `src/src/llama-sampler.cpp:2658-2852` — the `grammar` sampler (`apply`, `accept`, `reset`, `clone`), and the three public constructors (plain / lazy-words / lazy-patterns, `llama.h:1460-1482`).
- `src/common/json-schema-to-grammar.cpp` — schema → GBNF; `common_schema_converter` with `_rules`/`_refs`/`_errors`/`_warnings`.
- `src/common/common.h:186-221` — `common_grammar_type` (`USER` / `OUTPUT_FORMAT` / `TOOL_CALLS`) and `common_grammar_needs_prefill`: it is the *type*, not the content, that decides whether the generation prompt is replayed into the grammar.
- `src/common/sampling.cpp:212-276, 452-464, 594-715` — construction, gating, the two application orders.
- `src/common/peg-parser.{h,cpp}`, `src/common/chat.cpp`, `src/common/chat-auto-parser-generator.cpp` — the second engine; see [[chat-templates]].
- `src/common/llguidance.cpp` — alternative backend when built with `-DLLAMA_LLGUIDANCE=ON`; the stub logs a warning and returns `nullptr` otherwise (`:255-258`), and `common_sampler_init` `GGML_ABORT`s if the grammar string says `%llguidance` in a build without it (`sampling.cpp:216-218`). `[UNVERIFIED]` whether the default build of this fork enables LLGuidance; nothing in the sampled files does.

## Known issues

- **Silently ignored schema keywords.** The ignored rows above yield a grammar that accepts strictly more than the schema. Only the regex path warns (`json-schema-to-grammar.cpp:392`); everything else is silent, and `check_errors()` prints `WARNING: JSON schema conversion was incomplete` **only** when `_warnings` is non-empty (`:1088-1093`).
- **`minLength`/`maxLength`/bounds are type-guarded, not type-inferred.** `{"minLength": 10}` without `"type"` compiles to "any JSON value", not to a bounded string (`:1046` vs `:1074-1077`). The same document with `"type": "string"` compiles correctly — a one-key difference between a constraint and no constraint.
- **A grammar that admits no continuation is not detected.** `apply` has no "everything rejected" branch; the all-`-INFINITY` array is handed to the chain, whose first `softmax_impl` asserts only `cur_p->size > 0` (`llama-sampler.cpp:293-294`). Nothing warns, and what the distribution stage does with an all-`-inf` array is `[UNVERIFIED]` (NaN probabilities are the obvious hazard).
- **A grammar that cannot terminate blocks EOG.** `allow_eog` is true only when some stack is empty (`llama-grammar.cpp:1364-1370`), so a dead-ended grammar cannot end the sequence: it generates to `n_predict` instead of stopping. The mirrored case — EOG accepted with no empty stack — is the `GGML_ABORT("fatal error")` in `llama_grammar_accept_impl` that [[sampling]] already records.
- **Lazy grammars can never trigger.** `awaiting_trigger` makes `apply` a no-op (`llama-grammar.cpp:1356-1359`), and `grammar_should_apply` also suppresses it while the reasoning budget is active (`sampling.cpp:452-464`). If a trigger word/pattern never arrives, the constraint silently never applies — the failure mode is unconstrained output, not an error.
- **Invalid grammar is a hard failure, early.** Parse errors and undefined rule refs return `nullptr` from `llama_grammar_init_impl` (`:1143-1152`), `llama_sampler_init_grammar_impl` then returns `nullptr` (`llama-sampler.cpp:2806-2809`), and `common_sampler_init` turns that into `throw std::runtime_error("failed to parse grammar")` (`sampling.cpp:274-276`) before any token is generated. Schema-level errors take the same route via `throw std::invalid_argument` in `check_errors()` (`json-schema-to-grammar.cpp:1088-1090`) and are caught per request in `server-schema.cpp:264`.
- **Cost is unmeasured** (`sampling.cpp:542`) and the redraw path doubles chain work on every rejected first draw — the interaction worth measuring next, together with [[sampling]]'s open question about grammar cost.

### Fork divergence

**No grammar-side fork change was found, and this could not be proven either.** A grep for `triattn|triattention|turboquant|prism|llama-fast` across the three files returns nothing, and the files carry upstream-shaped scaffolding — `#ifdef LLAMA_USE_LLGUIDANCE`, the comment `// TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero` (`json-schema-to-grammar.cpp:1083`), the upstream PR reference for lazy grammars (`llama.h:1476`, PR 9639) and for the Aho–Corasick complement (`peg-parser.cpp:1470`, PR 24839). The PEG engine, the `%llguidance` route and `common_grammar_needs_prefill` all look like recent *upstream* additions on the same evidence `[INFERENCE]`. **No git history or upstream checkout was consulted, so byte-identity with upstream is `[UNVERIFIED]`** — the honest statement is "no fork-specific identifiers, no evidence of divergence, no diff".

## See also

[[sampling]] · [[chat-templates]] · [[tokenizer]] · [[request-lifecycle]] · [[runtime-switches]] · [[jinja-engine]] · [[server-layer]] · [[overview]]
