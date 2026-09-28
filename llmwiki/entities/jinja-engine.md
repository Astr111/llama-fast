---
title: Jinja engine
type: entity
status: current
updated: 2026-09-29
sources: [README.md, state.md]
verified: [src/common/jinja, src/common/jinja/README.md, src/common/jinja/caps.h, src/common/jinja/caps.cpp, src/common/jinja/string.h, src/common/jinja/string.cpp, src/common/jinja/value.h, src/common/jinja/value.cpp, src/common/jinja/runtime.h, src/common/jinja/runtime.cpp, src/common/jinja/parser.cpp, src/common/jinja/utils.h, src/common/chat.cpp, src/common/chat.h, src/common/chat-auto-parser.h, src/common/common.h, src/common/arg.cpp, src/src/llama-chat.cpp, src/src/llama-chat.h, src/src/llama.cpp, src/include/llama.h, src/tools/server/server-common.cpp, src/tools/server/server-context.cpp]
tags: [jinja, chat-template, template-engine, prompt-injection]
---

# Jinja engine

## What it is

The **in-tree Jinja2 interpreter** that turns a chat template into a prompt string: 13 C++ files, ~230 KB, under `src/common/jinja/` (`value.cpp` 64.6 KB, `runtime.cpp` 39.6 KB, `value.h` 28.9 KB, `runtime.h` 24.9 KB, `parser.cpp` 22.3 KB, `caps.cpp` 18.9 KB, `lexer.cpp` 12.5 KB, the rest small). It is a C++ implementation written from scratch, "originally inspired by huggingface.js's jinja package" and introduced upstream in PR #18462 — not a binding, not vendored code, not Python Jinja2's parser translated (`src/common/jinja/README.md`).

It is the *interpreter* half of the GGUF chat template: the model file supplies the bytes of `tokenizer.chat_template`, [[chat-templates]] supplies the plumbing (which template, which messages, which workarounds), and this engine supplies the language. It never tokenizes, never touches the graph, and never sees the GPU.

## How it works

### Pipeline

`jinja::lexer` → tokens → `jinja::parser` → `jinja::program` (an AST of statements and expressions) → `jinja::runtime::execute`, which walks nodes recursively (README, *Architecture*). Unlike huggingface.js the lexer does **not** pre-process the source; the parser sees it as-is so an error can name a character offset, and the engine keeps the source in order to draw a caret under the offending text (`peak_source` / `fmt_error_with_source`, `src/common/jinja/utils.h:35-53`).

### The value model — `value.h`

Every value is a `std::shared_ptr<value_t>` (`value.h:20-21`). One base struct carries every representation plus a debug/analysis record:

```
value_t {
    int64_t val_int; double val_flt; string val_str;
    std::vector<value> val_arr;
    std::vector<std::pair<value,value>> val_obj;   // insertion-ordered
    func_handler val_func;
    struct stats_t { bool used; std::set<std::string> ops; } stats;  // only when ctx.is_get_stats
}
```

(`value.h:106-127`). Subclasses are one per Jinja type: `value_int_t` (`:214`), `value_float_t` (`:248`), `value_string_t` (`:290`), `value_bool_t` (`:326`), `value_array_t` (`:354`), `value_tuple_t` (`:463`, immutable), `value_object_t` (`:483` — the ordered vector plus an `unordered_map` for lookups; `has_builtins = false` on `context` and `loop` objects, `:485`, `runtime.h:66`, `runtime.cpp:597`), `value_none_t` (`:602`), `value_undefined_t` (`:620`, carrying a `hint` that names where the undefined came from), `value_func_t` (`:697`, name + bound receiver) and `value_kwarg_t` (`:736`). Conversion is explicit by design — `as_int()` / `as_float()` / `as_string()` / `as_bool()` throw a type error unless overridden (`value.h:130-153`); there is no operator overloading. Builtins are a per-type name → handler table returned by `get_builtins()`, built lazily and cached per type: `length` for strings (`value.cpp:650`), arrays (`:979`) and objects (`:1265`); `selectattr` / `select` / `rejectattr` / `reject` (`:1013-1016`); `items` (`:1245`).

JSON is *not* part of the value model: `global_from_json` is a template over `T_JSON` (`value.h:90-91`) and `common_json` appears only at the boundary — and in `caps.cpp`, to define probe inputs.

### The string model — `string.h` / `string.cpp`

A Jinja string is a **list of parts**, each with a provenance bit:

```cpp
struct string_part { bool is_input = false; std::string val; };
struct string      { std::vector<string_part> parts; ... };
```

(`string.h:15-40`). `is_input = true` means "this text came from the request, not from the template". `str()` concatenates the parts, `length()` sums their lengths, `hash_update` feeds them to the internal hasher (`string.cpp:39-73`).

Propagation is implemented per transform rather than inferred:
- **one-to-one** (`upper`, `lower`, `capitalize`, `title`, `strip`) — the flag survives the transform (README, *Input Marking*; the transforms themselves in `string.cpp`, section *in-place transformation*, and declared at `string.h:60-68`);
- **many-to-one** (`append`, `join`, slicing, `format`, `indent`) — the result is input only if **all** inputs were: `mark_input_based_on(other)`, commented "mark this string as input if other has ALL parts as input" (`string.h:47-48`, `string.cpp:97-103`), which is what `join` / `slice` / `indent` call (`value.cpp:690`, `:717`, `:750`, `:866`, `:913`);
- the entry point is `from_json`, which marks every string it converts when `mark_input` is true (`value.cpp:1358-1385`, driving `string::mark_input`, `string.cpp:39-43`).

The point is the README's injection scenario: a user message containing `<|im_end|>\n<|system|>…` would otherwise be indistinguishable from the template's own delimiters once both live in one `std::string`. With marking, the render is a part list where template-emitted control tokens are `is_input=false` and request text is `is_input=true`, so a consumer *could* decline to treat an injected token as a control token.

> **In this tree that consumer does not exist.** The render is flattened back to a single string before it leaves the layer — `auto parts = runtime::gather_string_parts(results); std::string result = parts->as_string().str();` (`src/common/chat.cpp:980-982`) — and a name-based search for `is_input` / `mark_input` / `gather_string_parts` across `src/common`, `src/tools/server` and `src/tools/cli` found no reader of the flags outside `common/jinja/` itself; the only setter is the `mark_input` field of the template inputs (`src/common/chat-auto-parser.h:73`, default `true`). `[INFERENCE]`: the flags are computed on every render and then discarded at `chat.cpp:982`, so the injection defence described in the engine's README is latent in this fork, not active. (Search scope as stated; a consumer reaching the parts through another accessor would have been missed.)

### `caps`: capability by execution, not by parsing

`caps` is **not** a description of Jinja language features, despite its position next to the parser. It answers a narrower question — "what can this *particular* template, as written, accept and use?" — with nine booleans (`caps.h:11-27`):

| cap | default | decided by |
| :--- | :--- | :--- |
| `supports_string_content` / `supports_typed_content` | true / **false** | content probe |
| `supports_system_role` | true | system-role probe |
| `supports_tools` / `supports_tool_calls` | true / true | tools probe |
| `supports_object_arguments` | **false** | tools probe (object arguments) |
| `supports_parallel_tool_calls` | true | parallel-calls probe |
| `supports_preserve_reasoning` | **false** | reasoning-history probe |
| `supports_reasoning_effort` | **false** | reasoning-effort probe |

`caps_get(prog)` (`caps.cpp:111`) **executes the compiled program** against synthetic message lists and reads three signals: whether execution threw, which values and operations the template touched (`ctx.is_get_stats = true`, so `value_t::stats.ops` records names such as `array_access` and `selectattr`), and whether a marker string survived into the output. `caps_try_execute` (`caps.cpp:34-72`) deliberately **swallows** the exception (`:65-68`, "ignore exceptions during capability analysis") and reports `success = false`.

The probes, in order: content typed-vs-string (`caps.cpp:~120-160` — content is passed as a plain string; `selectattr` or `array_access` recorded on it ⇒ typed content; execution failing, or the string marker missing from the render, ⇒ string content false); system role (`messages[0].content` never `used` ⇒ false); tools with **object** arguments, then a second probe with arguments as a **string** (`R"({"arg": "value"})"`) if object arguments went unused (`caps.cpp:~200-360`; `supports_tools` from whether the tool name was used, `supports_tool_calls` from `tool_calls`); parallel calls (a second `tool_calls[1].function` must be used, `:360-420`); preserve reasoning (a `reasoning_content` on a **non-final** assistant message must appear in the rendered text — checked on the output, not on stats, "because the reasoning_content may be used for `if` condition test, but not actually outputted", `:480-512`); reasoning effort (`reasoning_effort` / `reasoning_strength` are set by `caps_apply_reasoning_effort`, and the cap follows `effort->stats.used`, `:520-535`).

The inference only runs one way: a `true` default can be *demoted*, and the four `false` defaults can be *promoted*. `caps::to_map()` (`caps.cpp:87-100`) is what the server reports and branches on, preferring the `tool_use` template's caps when one exists (`src/common/chat.cpp:3905-3907`).

Caps drive **input rewrites**, never template rewrites: `caps_apply_preserve_reasoning` sets `preserve_thinking` / `clear_thinking` / `truncate_history_thinking` / `drop_thinking` in the context and `caps_apply_reasoning_effort` sets `reasoning_effort` / `reasoning_strength` (`caps.cpp:22-33`), while role/content/argument-shape fixes are applied to the message JSON before rendering — the README's "workarounds are applied to input data before entering the runtime".

### The language subset, and what a violation does

Statements are a closed set (`parser.cpp:132-214`): `set` (also in `{% set %}…{% endset %}` block form), `if` / `elif` / `else`, `macro`, `for` with optional `else`, `break`, `continue`, `call`, `filter`, and `generation` / `endgeneration`, which parse to a **`noop_statement`** for transformers compatibility (`:207-210`). Anything else — `include`, `extends`, `import`, `raw`, `with`, `block` — is not partly supported:

```cpp
} else {
    throw std::runtime_error("Unknown statement: " + name);   // parser.cpp:214
}
```

So the failure surface is:

| Construct | Behaviour |
| :--- | :--- |
| unknown `{% … %}` statement | **hard failure at compile**: `std::runtime_error("Unknown statement: …")`, `parser.cpp:214` (and `:125` for a non-identifier); the template never renders |
| unknown filter / function at render | **hard failure**: `throw std::runtime_error("Unknown (built-in) filter '" + name + "' for type " + input->type())`, `runtime.cpp:296-310`, unless the `undef_on_missing` path (which yields an undefined, for `defined` tests) is taken |
| unknown operator | **hard failure**: `throw std::runtime_error("Unknown operator …")` in `binary_expression::execute_impl`, immediately above `try_builtin_func` (`runtime.cpp:~298`) |
| a filter whose receiver is `undefined` | **silent empty** when the name is one of the ~28 entries in `value_undefined_t::get_builtins` (`value.cpp:1319-1350`), which return `empty_value_fn<…>` — `{{ missing \| upper }}`, `\| length`, `\| strip`, `\| first`, `\| join` print nothing and log nothing |
| type mismatch | **hard failure**: the base `value_t::as_*` throw (`value.h:132-137`) |
| failure *inside* `caps_get` | **swallowed**, demoted to a `false` cap |

Stated plainly: **the engine fails loudly, not silently** — with one silent pocket (the undefined-value builtin table) and one deliberate swallowing (capability probing). A template that is *accepted* is one that either rendered successfully during the probes or was never probed on that path.

Where the loud failures surface: compilation and caps happen in the `common_chat_template` constructor (`src/common/chat.h:67`, `this->caps = jinja::caps_get(prog);`); a real render is `jinja::runtime runtime(ctx); runtime.execute(tmpl.prog)` (`src/common/chat.cpp:978-979`); `common_chat_verify_template` catches and logs with the source-caret message (`chat.cpp:627-642`), so a broken `--chat-template` body is rejected at startup; errors during a request propagate into the server's chat-param construction, which wraps them (`src/tools/server/server-context.cpp:1405-1415`).

### Where it sits relative to the GGUF template

The GGUF holds a *string*; the engine holds the *language*. For the target checkpoint ([[ternary-bonsai-2-27b]], arch `qwen35` — [[qwen35-architecture]]):

1. the `tokenizer.chat_template` bytes come from the model file and are compiled once by the `common_chat_template` ctor, which probes them immediately (`src/common/chat.h:67`);
2. caps then decide the message-JSON shape and the thinking-context variables, and are reported to the client;
3. the same compiled `program` is executed per request at `src/common/chat.cpp:978-979` against the context built at `:937-975` (`jinja::global_from_json(ctx, inp, inputs.mark_input)`, `:975`);
4. the result goes to [[tokenizer]]; the output-parser selection and any lazy grammar belong to [[chat-templates]] and [[grammar-constraints]], not to this engine.

Practical consequence: a template can be perfectly valid Jinja2 and still be unusable here — one `{% raw %}` aborts compilation — while a template using only the subset can still be *misreported* by caps (see below), silently changing what the server sends.

## Where it lives

| Path | Size | Role |
| :--- | ---: | :--- |
| `src/common/jinja/lexer.{h,cpp}` | 5.2 / 12.5 KB | source → tokens; no pre-processing, so error positions are usable |
| `src/common/jinja/parser.{h,cpp}` | 0.5 / 22.3 KB | tokens → `jinja::program`; the closed statement set (`:132-214`) |
| `src/common/jinja/runtime.{h,cpp}` | 24.9 / 39.6 KB | execution, `context`/`env` (`runtime.h:65-69`), `gather_string_parts` (`runtime.h:724-766`), filter/function dispatch and its errors (`runtime.cpp:296-310`) |
| `src/common/jinja/value.{h,cpp}` | 28.9 / 64.6 KB | the value model, conversions, all builtins, `from_json` / `global_from_json` (`:1358-1464`) |
| `src/common/jinja/string.{h,cpp}` | 1.7 / 5.3 KB | `string_part::is_input` and its propagation |
| `src/common/jinja/caps.{h,cpp}` | 1.0 / 18.9 KB | executable capability probing; `caps_get` `:111`, `to_map` `:87` |
| `src/common/jinja/utils.h` | 5.0 KB | FNV-style hasher, source-peeking error formatting |
| `src/common/jinja/README.md` | 4.1 KB | design notes: architecture, input marking, caveats |
| `src/common/chat.cpp`, `chat.h` | — | compile + caps (`chat.h:67`), context build and render (`chat.cpp:937-982`), caps to the client (`:3905-3907`), verification (`:627`) |
| `src/common/arg.cpp` | — | `--jinja` / `--no-jinja` (`:3657-3663`), per-example defaults (`:1410-1414`) |
| `src/tools/server/server-*` | — | consume `caps` and the rendered prompt; tools require `--jinja` (`server-common.cpp:1141-1144`) |
| `src/src/llama-chat.{h,cpp}` | 41 KB | the **non**-Jinja path — see [[chat-templates]] |

## Known issues

- **Caps can be wrong in both directions, and the server believes them.** A template that *reads* `reasoning_content` for a decision but never prints it is reported `supports_preserve_reasoning = false` (the probe is text-based by design, `caps.cpp:~502-508`); a template that array-accesses content for a test but renders it as a string loses `supports_string_content` unless the marker still appears (`caps.cpp:~140-160`). `[INFERENCE]` from those probe implementations — the mechanisms are read, no observed false report.
- **Every probe is one synthetic conversation**, built inside `caps.cpp`: a single user message, a fixed two-tool-call history, a fixed four-message reasoning history, and `bos_token`/`eos_token` forced to `""` (`caps.cpp:44-50`). A template that branches on anything else — message count, tool names, a non-empty BOS — is measured on a history it will never see in production.
- **`caps_get` runs the template seven times** with the side effects of a normal render (`caps.cpp:111-538`); the cost is paid once per template at construction, and it is construction, not rendering, that can fail.
- **The undefined-value builtin table is a silent-empty trap**: `{{ something_typo | length }}` renders nothing, `{{ something_typo | int }}` throws, because `int` is not in that table (`value.cpp:1319-1350`, `runtime.cpp:296-310`).
- **Input marking is computed and then dropped** at `src/common/chat.cpp:982`; no reader of `is_input` was found outside `common/jinja/` in the directories searched (boxed note above).
- The README's own caveats apply at the boundary and are not enforced by this engine: a special token *constructed* from user data (`'<|' + role + '|>'`) is marked as input and will not behave as a delimiter, and a template-added leading space is a separate part and tokenizes separately.
- `[UNVERIFIED]`: `lexer.cpp` was grepped, not read — the whitespace-control rules (`{%-` / `-}}`) and the token set are not described here. Several `caps.cpp` probe line numbers are approximate (`~n`) within the ranges read; `caps.cpp` was read as `1-203`, `199-523` and `523-541`.
- `[UNVERIFIED]`: `value.cpp`'s builtin inventory is not enumerated — roughly 40 names were seen (`length`, `items`, `selectattr`, `join`, `dictsort`, `format`, `indent`, `tojson`, `safe`, `default`, …) but the complete list and the per-type split were not collected, and `parser.cpp`'s expression grammar was read only where it decides statement support. `value.cpp`, `runtime.cpp` and `parser.cpp` were read in ranges, never whole.

## See also

[[chat-templates]] · [[tokenizer]] · [[sampling]] · [[server-layer]] · [[qwen35-architecture]] · [[ternary-bonsai-2-27b]] · [[grammar-constraints]]
