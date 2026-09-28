---
title: Chat templates
type: entity
status: current
updated: 2026-09-29
sources: [state.md, README.md]
verified: [src/common/chat.cpp, src/common/chat-peg-parser.cpp, src/common/chat-peg-parser.h, src/common/chat-auto-parser-generator.cpp, src/common/chat-diff-analyzer.cpp, src/common/jinja/README.md, src/common/arg.cpp, src/src/llama-model.cpp, src/src/llama-arch.h, src/tools/server/server-common.cpp, src/tools/server/server-context.cpp, src/tools/server/server-task.cpp, src/tools/server/server-task.h, src/tools/server/server-schema.cpp, src/tools/server/server-chat.cpp, "/home/ms/Загрузки/Ternary models/Ternary-Bonsai-2-27B-PQ2_0.gguf"]
tags: [chat-template, jinja, tool-calling, peg, server]
---

# Chat templates

## What it is

The **layer between a JSON message list and token ids** — and, in reverse, between raw generated text and a structured tool call. [[tokenizer]] owns text ↔ ids; [[forward-pass]] owns ids → logits; neither knows what a "message", a "role" or a "tool call" is. This page is that missing middle:

- **forward**: `messages[]` → **one prompt string** (the template render), which the caller then tokenizes;
- **reverse**: **generated text** → `common_chat_msg` with `content`, `reasoning_content` and `tool_calls[]` (a PEG parse, not a JSON decode);
- **sideways**: the *output constraint* — a GBNF grammar derived either from a user schema or from the tool signatures, handed to the sampler ([[sampling]] owns what the sampler does with it).

It is pure host code in `common/` — nothing about it reaches the GPU graph. It is also the largest underead surface in the tree (~250 KB: `chat.cpp` alone is 174 KB, and all 13 `common/jinja/*` files were previously uncited).

## How it works

### Forward: messages → prompt string

1. **Template acquisition** — `common_chat_templates_init` (`src/common/chat.cpp:753`). If `--chat-template` is absent the source comes from the **model file**: `llama_model_chat_template` (`src/src/llama-model.cpp:3342`) looks up the GGUF KV `tokenizer.chat_template`, and a second lookup with name `tool_use` picks up an optional tools-specialised variant. Both are wrapped in `common_chat_template` (`chat.cpp:841-845`), which *compiles the string as Jinja*. Empty template, or the literal string `chatml`, falls back to the built-in `CHATML_TEMPLATE_SRC` (`chat.cpp:776-781`); `has_explicit_template` records whether either a model or CLI template existed. Missing `bos_token`/`eos_token` in the vocab is warned about, not fatal (`chat.cpp:815-820`).
2. **Engine selection** — `common_chat_templates_apply` (`chat.cpp:3809-3813`) branches on `inputs.use_jinja` (= `--jinja` / `--no-jinja`, `src/common/arg.cpp:3657-3661`): `common_chat_templates_apply_jinja` (`chat.cpp:3600`) or `common_chat_templates_apply_legacy` (`chat.cpp:3764`), the latter forwarding to `llama_chat_apply_template`, the ad-hoc C++ implementation of a handful of known formats. The legacy route is selected by a template *name*, not by a template body.
3. **Jinja engine** — `src/common/jinja/{lexer,parser,runtime,value,string,caps}.cpp`. Pipeline per the engine's own README: `lexer` → tokens → `parser` → `jinja::program` (AST) → `runtime` executes each statement/expression. Two properties matter to this layer: **input marking** (`jinja::string` carries an `is_input` flag so user text can never be mistaken for a special token — the prompt-injection defence), and `caps` (`common_chat_templates_get_caps`, `chat.cpp:3902-3907`), the template-feature detection that drives the workarounds below. By design the engine does no JSON → value translation itself; `common_json` is optional and only used at the boundary.
4. **Render** — `common_chat_template_direct_apply_impl` (`chat.cpp:931`) produces the full prompt into `data.prompt`; `common_chat_template_generation_prompt_impl` (`chat.cpp:1000`) produces the assistant-prefix into `data.generation_prompt`. The message JSON passed in has already been normalised by template-capability workarounds in `common_chat_templates_apply_jinja` (developer→system role, null-content fill, object-vs-string tool arguments). `--chat-template-kwargs` (and the per-request `chat_template_kwargs`) add arbitrary keys to the template's context (`params.extra_context`, `chat.cpp:3600` block; merged with the request copy at `src/tools/server/server-common.cpp:1296-1301`).
5. **Tokens** — this entity hands back *text*. Tokenization stays with the caller: the server converts the prompt in `server-common.cpp` (see [[tokenizer]], which cites `:864-881`). The only tokenization inside this layer is of the *delimiters*, `common_chat_msg_delimiters::tokenize` (`chat.cpp:146`, via `common_tokenize`), used to split a token stream back into role spans.
6. **Parser selection happens in the same pass** — after rendering, `common_chat_templates_apply_jinja` chooses how the *output* will be parsed: the pure-content bypass (`force_pure_content`, `chat.cpp:3689-3703`), then a fingerprint dispatch over specialised templates (`common_chat_try_specialized_template`, `chat.cpp:3477`), then the differential autoparser (`chat.cpp:3711-3714`). The chosen parser is serialised into `common_chat_params.parser` (`parser.save()`, `chat.cpp:3702`).

### The target model, concretely

`Ternary-Bonsai-2-27B-PQ2_0.gguf` ([[ternary-bonsai-2-27b]], arch `qwen35` per [[qwen35-architecture]]) carries **exactly one** `chat_template` entry: the substring `chat_template` occurs once in the whole 7.2 GB file, at byte offset `11 061 598` — there is **no** `tool_use` variant, so `template_tool_use` stays null and `template_default` is used even when tools are supplied.

**It is Jinja, not a built-in format**, and the file says so itself: the value begins

```
{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
```

— a `macro` definition with vision (`<|vision_start|>`) handling. Reading the value as text shows `im_start` ×9, `im_end` ×8, `<tool_call>` ×5, `<function=` ×5, `<parameter=` ×3, `<think>`/`</think>`, plus the variables `enable_thinking` and `reasoning_content`. How one *knows* this is Jinja rather than a legacy named template: (a) the source is `{%`/`{{`-delimited and defines a macro, and `common_chat_templates_init` compiles exactly this string with the in-tree engine (`chat.cpp:841-845`); (b) the legacy path is keyed on a short name like `chatml` and would not accept a 10 KB macro body; (c) `common_chat_templates_source` returns the raw string for inspection. `[UNVERIFIED]` the KV *key* spelling: the loader composes it from `LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)` and the exact composed name was not read out of `llama-arch.h`; the file's own key literally contains `chat_template`.

**Which parser the target actually gets.** Because the template source contains `<tool_call>` **and** `<function=` **and** `<parameter=`, the fingerprint arm at `chat.cpp:3589-3594` fires and dispatches to `common_chat_params_init_qwen3_coder` (`chat.cpp:1162`) — the comment there names "Qwen3-Coder, Nemotron Nano 3, Qwen3.5 and StepFun-3.5-Flash" as the same markup. So **for this checkpoint the differential autoparser never runs**; the reverse path is the hand-written PET/PEG parser for Qwen XML: role delimiters incl. `COMMON_CHAT_ROLE_TOOL` = `<|im_start|>user\n<tool_response>` (`chat.cpp:1188-1193`), reasoning ends on `</think>` or `<tool_call>` (`:1184`), a `<function=…>` opener tolerated without the enclosing `<tool_call>` (`:1220-1223`, `:1291-1294`), and arguments accepted in any order (`p.permute`, `:1277`). The vision macros in the template are inert unless an `mmproj` is loaded.

### Reverse: generated text → structured tool call

- **Wire format.** The server keeps a `common_chat_parser_params` per task (`src/tools/server/server-task.h:92`), which holds the serialised `common_peg_arena`. `common_chat_parse` (`chat.cpp:3816`) forwards to `common_chat_peg_parse` (`chat.cpp:3822`); an empty arena degrades to a content-only parser (`:3826-3828`). Streaming calls it with `is_partial = true`, which salvages whatever prefix already parsed (`:3851-3852`), and `task_result_state::update_chat_msg` (`server-task.cpp:170-172`) diffs consecutive results via `common_chat_msg_diff::compute_diffs` (`chat.cpp:266`) to emit deltas.
- **The PEG layer.** `common_peg_parser_builder` / `common_peg_arena` live in `common/peg-parser.{h,cpp}`; the chat-specific vocabulary is `common_chat_peg_builder` (`chat-peg-parser.h:60`) with `build_chat_peg_parser` (`:182`). The AST is turned into a message by a **mapper**: `common_chat_peg_mapper::from_ast` (`chat-peg-parser.cpp:284`) walking nodes in `map` (`:315`), with two format-specific subclasses for markup the generic mapper cannot express — `common_chat_peg_gemma4_mapper` (`:956`) and `common_chat_peg_minimax_m3_mapper` (`:1198`). Reusable builders cover the common shapes: JSON-object tools (`standard_json_tools`, `:905`; three key layouts, `:628`/`:707`/`:779`), constructed XML tools (`standard_constructed_tools`, `:458`), and LFM2's Python-call syntax (`python_style_tool_calls`, `:550`). `tagged_peg_parser` (`chat-peg-parser.h:201`) is not used at serving time — it is the *analysis-time* extractor.
- **The autoparser (fallback only).** When no fingerprint matches, the parser must be derived from the template. `chat-diff-analyzer.cpp` renders the very same template with synthetically varied message lists and **diffs the resulting strings**, which is why it is called differential analysis: `autoparser::analyze_template` (`:257`) drives `analyze_reasoning`, `analyze_content` and `analyze_tools`, each built on `compare_variants` (e.g. `:1066`, `:1206`, `:1505`). A table of template-source fingerprints patches known-bad inferences (`workarounds`, `:36` — Granite, Cohere, Nemotron, Laguna, Solar, Apriel, Functionary…). `chat-auto-parser-generator.cpp` then compiles that structure into a PEG parser: `peg_generator::generate_parser` (`:32`, and the pre-analysed overload `:40`), with the `analyze_*::build_parser` methods (`:117`, `:198`, `:214`, `:306`, `:377`) and rule construction reusing the same `common_chat_peg_builder`. It also fills `data.prompt` / `data.generation_prompt` (`:45-46`) and emits **lazy grammar triggers** from the inferred tool-section markers (`:102-110`).

### Grammar handshake

Two inputs reach the same slot, and they are mutually exclusive in practice:

- **User grammar**: `--grammar` (`arg.cpp:2273-2276`, `COMMON_GRAMMAR_TYPE_USER`) verbatim, or a request-level `json_schema` / `response_format` (`src/tools/server/server-common.cpp:1157-1176`), converted by `json_schema_to_grammar` inside the apply (`chat.cpp:3801-3805`). The OAI-compat server field handler does the same for its own schema field (`src/tools/server/server-schema.cpp:261`).
- **Tool-derived grammar**: when a tools array is present, a user grammar plus a non-`none` tool choice throws (`chat.cpp:3600` function body, the `Cannot specify grammar with tools` guard); instead each specialised/autoparser path builds the constraint *from the tool signatures* — `build_grammar(...)` around `parser.build_grammar(builder, data.grammar_lazy)` (`chat.cpp:1141-1154`, `:1312-1327`) — and marks it **lazy** with word triggers, so the constraint only engages after the model emits the tool-call opener (`chat.cpp:1154`, `:1325-1328`).

**Where it enters relative to the template and the sampler:** the grammar is produced *inside* template application (it is a field of `common_chat_params`, alongside `prompt`, `parser`, `grammar_lazy`, `grammar_triggers`), is passed by the server into the sampling parameters (`server-common.cpp:1335-1338`), and only then meets the sampler. How the sampler compiles and applies a lazy grammar, and what `COMMON_GRAMMAR_TYPE_*` means downstream, is [[sampling]]'s subject and is not repeated here; the server-side field plumbing is `server-schema.cpp:283-284` and its trigger validation `:375-377`.

Note for anyone grepping for a flag: **`--json-schema-to-grammar` does not exist in this tree.** That name belongs to the C++ function `json_schema_to_grammar` and to the Python helper the server README points at; the actual CLI surface is `--grammar` / `--grammar-file` and `-j/--json-schema` / `-jf/--json-schema-file`.

### Stage table

| Stage | Entry point | `path:line` | Selected / bypassed by |
| :--- | :--- | :--- | :--- |
| Template source resolution | `common_chat_templates_init` | `src/common/chat.cpp:753` | `--chat-template` (`arg.cpp:3735`), else GGUF `tokenizer.chat_template` (+ optional `tool_use`) |
| GGUF metadata lookup | `llama_model_chat_template` | `src/src/llama-model.cpp:3342` | — |
| Engine choice | `common_chat_templates_apply` | `src/common/chat.cpp:3809` | `--jinja` / `--no-jinja` (`arg.cpp:3657`) |
| Legacy render (named formats) | `common_chat_templates_apply_legacy` → `llama_chat_apply_template` | `src/common/chat.cpp:3764` | `--no-jinja` |
| Jinja default template compile | `common_chat_template` ctor | `src/common/chat.cpp:841-845` | — |
| Jinja execution | `jinja::lexer` → `jinja::parser` → `jinja::runtime` | `src/common/jinja/` (README, *Architecture*) | `--jinja` (the path that uses it) |
| Template context extension | `params.extra_context` | `src/common/chat.cpp:3600` body; merged `server-common.cpp:1296-1301` | `--chat-template-kwargs`, request `chat_template_kwargs` |
| Prompt / generation prompt | `common_chat_template_direct_apply_impl` / `_generation_prompt_impl` | `chat.cpp:931` / `:1000` | — |
| Pure-content bypass | `force_pure_content` branch (parser = content + end) | `chat.cpp:3689-3703` | `--skip-chat-parsing` (`arg.cpp:3759`) |
| Specialised parser dispatch | `common_chat_try_specialized_template` (Qwen arm) | `chat.cpp:3477`, `:3589-3594` | template fingerprint only |
| Qwen XML parser build | `common_chat_params_init_qwen3_coder` | `chat.cpp:1162` | — (the target's arm) |
| Differential analysis | `autoparser::analyze_template`, `compare_variants` | `src/common/chat-diff-analyzer.cpp:257`, `:1066` | reached only if no fingerprint matches |
| Auto-PEG generation | `peg_generator::generate_parser` | `src/common/chat-auto-parser-generator.cpp:32` (called `chat.cpp:3714`) | same |
| Arena compile / load | `parser.save()` / `arena.load` | `chat.cpp:3702` / `:3737` | — |
| Output parse | `common_chat_parse` → `common_chat_peg_parse` | `chat.cpp:3816` / `:3822` | carried in `common_chat_parser_params` (`server-task.h:92`); **bypassed** by `--skip-chat-parsing` |
| Streaming deltas | `common_chat_msg_diff::compute_diffs` | `chat.cpp:266` | per-request `stream` |
| Schema → grammar | `json_schema_to_grammar` inside apply | `chat.cpp:3801-3805` | `-j/--json-schema`, `response_format` |
| Grammar (user / tools) | `--grammar`; `build_grammar` + `parser.build_grammar` | `arg.cpp:2273`; `chat.cpp:1141-1151` | `--grammar`; tools + `tool_choice` |
| Grammar → sampler | sampling params assembly | `src/tools/server/server-common.cpp:1335-1338` | see [[sampling]] |
| Built-in tool definitions | `--tools` registration | `src/common/arg.cpp:3405` | `--tools` (creates the tools array the stages above branch on) |

### Why this matters operationally

**A wrong template or a mis-parsed tool call is invisible in a token-level trace and looks exactly like model failure.** Every measurement this vault has made — `flash_attn_ext_vec<…>` kernel tallies, 19.43 ms/token decode, 38.81 % `magma_sgemmEx` — is aggregate arithmetic over token ids, and token ids are post-template. If `template_tool_use` is null (it is, for this GGUF), if the fingerprint arm picks the Qwen3-Coder grammar for a template that renders tool calls slightly differently, if a `<function=` opener arrives without its `<tool_call>` wrapper, or if the model is fed ChatML because the template lookup missed, the symptom is a *plausible but wrong* assistant turn: fluent prose, no tool call, or an empty content field arriving as an OAI `finish_reason: tool_calls`. Nothing in the CUDA timeline, the KV accounting or the sampler statistics distinguishes that from the model being weak. The diagnostic surface is textual and separate: `common_chat_templates_source` and `common_chat_format_example` echo the actual rendered prompt (the server does the latter at startup, `server-context.cpp:1405-1406`), and `chat_format` / `grammar_triggers` / `preserved_tokens` are reported per request (`server-task.cpp:131-135`). This is why the layer deserves a page of its own before any further claim is made about target quality.

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/common/chat.cpp` | the message model, template init/apply, ~20 specialised format parsers, the PEG entry points (`:753`, `:3600`, `:3809`, `:3816`) |
| `src/common/jinja/` | the Jinja engine: `lexer.cpp`, `parser.cpp`, `runtime.cpp`, `value.cpp`, `string.cpp` (input marking), `caps.cpp` (feature detection), plus `README.md` |
| `src/common/chat-peg-parser.{h,cpp}` | chat PEG builder, mappers, JSON/tagged/Python tool shapes |
| `src/common/chat-diff-analyzer.cpp` | differential analysis: which markers a template emits, derived by rendering and diffing |
| `src/common/chat-auto-parser-generator.cpp` | turns that analysis into a PEG parser and a lazy grammar |
| `src/common/peg-parser.{h,cpp}` | the generic PEG engine/arena |
| `src/tools/server/server-common.cpp`, `server-context.cpp`, `server-task.cpp`, `server-schema.cpp` | apply, parser handoff, per-request parsing, grammar fields |
| `src/tools/server/server-chat.cpp` | does **not** apply templates — it converts between the Chat Completions and Responses API bodies (tools passthrough `:534-551`, param passthrough `:578-582`), so the template layer is shared by both endpoints |

## Known issues

- **The target's tool-call parser is selected by a substring test on the template body** (`chat.cpp:3589`). Any upstream re-wording of the Qwen3.5 template that drops `<function=` or `<parameter=` silently moves the checkpoint onto the differential autoparser, changing the parser (and the lazy grammar triggers) with no flag involved.
- **No `tool_use` variant in the target GGUF** — a tools-specialised render cannot be selected, so the tools branch runs with `template_default`; upstream models that ship `tool_use` do not have this constraint.
- **`--skip-chat-parsing` is a bypass, not a switch**: it forces the pure-content parser, so reasoning *and* tool calls arrive as `content` and are never structured (`common_chat_params_init_*` are not consulted).
- **The jinja workarounds are source-string patches** (`common_chat_templates_init` rewrites two known template snippets, `chat.cpp:783-805`; the analyzer carries a fingerprint table, `chat-diff-analyzer.cpp:36`). Both are coupled to exact upstream template text.
- `[UNVERIFIED]`: the full set of specialised arms in `common_chat_try_specialized_template` (`chat.cpp:3477-3597`) was read only where it mattered for the target (Ministral `:3479`, MiniMax-M3 `:3550`, Qwen `:3589`); the count of arms and their individual fingerprints were not enumerated.
- `[UNVERIFIED]`: the GGUF KV key spelling (`tokenizer.chat_template`) — inferred from the loader at `chat.cpp:753` + `llama-model.cpp:3342-3345`, not read from the key bytes.
- `[UNVERIFIED]`: `src/common/jinja/` was read at the capability level (README + file list) only; `caps.cpp`'s detection rules and `string.cpp`'s `is_input` propagation were not inspected.

## `llama-chat.cpp`: the in-engine template path

`src/src/llama-chat.cpp` (41 KB, ~1 000 lines) is **not** a Jinja consumer. It is the legacy, name-keyed, hand-written formatter that the engine was introduced to replace, and it is the only place in the llama core (`src/src/`) — as opposed to `common/` — where a chat template is applied without a parser. Neither a CLI chat loop nor a template-application helper in the engine's sense: a lookup table of *names* plus one renderer per known family.

- **What is in the file.** `llm_chat_template`, an enum of 55 template *names* terminating in `LLM_CHAT_TEMPLATE_UNKNOWN` (`src/src/llama-chat.h:7-63`), and `LLM_CHAT_TEMPLATES`, the name → enum map (`src/src/llama-chat.cpp:71`); roughly 900 of the file's ~1 000 lines are that map's template bodies. The three internal entry points are `llm_chat_template_from_str` (`:85` — `std::map::at`, so a miss throws `std::out_of_range`), `llm_chat_detect_template` (`:89` — catches that and yields `UNKNOWN`) and `llm_chat_apply_template` (`:244`).
- **The file says what it is**: "Simple version of `llama_apply_chat_template` that only works with strings. This function uses heuristic checks to determine commonly used template. **It is not a jinja parser.**" (`llama-chat.cpp:241-243`). The public header repeats it: "NOTE: This function does not use a jinja parser. It only support a pre-defined list of template" (`src/include/llama.h:1250-1252`).
- **C entry point and its failure signal**: `llama_chat_apply_template` (`src/src/llama.cpp:506`) detects the name, returns `-1` when unknown (`:524-526`), otherwise calls the helper. That `-1` is what `common_chat_templates_apply_legacy` forwards as "chat template is not supported" (`src/common/chat.cpp:3777-3792`).
- **Which path a run takes.** `common_chat_templates_apply` (`src/common/chat.cpp:3809-3813`) is a two-way branch — `inputs.use_jinja ? common_chat_templates_apply_jinja : common_chat_templates_apply_legacy` — and the legacy arm hands this file the template *source string*, from which it re-derives the format by name (`chat.cpp:3744-3792`). `use_jinja` defaults to **true** (`src/common/chat.h:255`, `src/common/common.h:627`) and is lowered only by `--no-jinja` / `LLAMA_ARG_JINJA` (`src/common/arg.cpp:3657-3663`); the `completion` and `mtmd` examples set it *false* unless `--jinja` is passed (`arg.cpp:1410-1414`). The server requires the engine whenever tools are present: `tools param requires --jinja flag` (`src/tools/server/server-common.cpp:1141-1144`).
  - ⇒ **server, and any CLI run with `--jinja`**: [[jinja-engine]] renders. **`llama-completion` and `llama-mtmd-cli` by default**: this file renders. `examples/simple-chat` calls `llama_chat_apply_template` directly (`simple-chat.cpp:174`) and therefore always takes this path, reporting "failed to apply the chat template" for a template it cannot name.
  - Because it is keyed on a name, the target's template body (a `{% macro %}` Qwen3.5 template — see above, and [[qwen35-architecture]] · [[ternary-bonsai-2-27b]]) is invisible to it: nothing matches, detection returns `UNKNOWN`, and `[INFERENCE]` a `--no-jinja` run on this checkpoint reports "chat template is not supported" rather than rendering anything.
- **Why the two paths cannot be merged**: the engine is an interpreter that produces, besides the prompt, the artefacts [[sampling]] later consumes — a tool-call parser and a lazy grammar. This file produces a string and nothing else, which is exactly why tools and structured output are gated on `--jinja` ([[grammar-constraints]]).
- `common_chat_verify_template` and `common_chat_format_example` take the same `use_jinja` boolean and route identically (`chat.cpp:627-642`, `:683-701`), so `--chat-template` validation tests whichever path the run will actually use.

This section also closes the two `[UNVERIFIED]` items this page carried about `src/common/jinja/`: `caps.cpp`'s detection rules and `string.cpp`'s `is_input` propagation are now described in [[jinja-engine]], which also records that the provenance flag is flattened away at `chat.cpp:982` before anything outside the engine can read it.

## See also

[[jinja-engine]] · [[tokenizer]] · [[forward-pass]] · [[sampling]] · [[server-layer]] · [[qwen35-architecture]] · [[ternary-bonsai-2-27b]] · [[conversion-and-packing]]
