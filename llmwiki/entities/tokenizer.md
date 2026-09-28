---
title: Tokenizer (qwen35 BPE)
type: entity
status: current
updated: 2026-09-29
sources: [state.md, README.md]
verified: [src/src/llama-vocab.cpp, src/src/unicode.cpp, src/src/unicode-data.cpp, src/src/unicode-data.h, src/src/unicode.h, src/src/llama-grammar.cpp, src/common/common.cpp, src/common/chat.cpp, src/tools/server/server-context.cpp, src/tools/server/server-common.cpp, src/tools/triattention-calibrate/triattention-calibrate.cpp, src/tools/completion/completion.cpp]
tags: [tokenizer, bpe, utf8, streaming]
---

# Tokenizer — the qwen35 BPE, text ↔ ids

## What it is

The bookends of the token path in this engine. Everything between them — ids → logits → sampled id — belongs to [[forward-pass]] and [[sampling]]; this page owns **text → ids** (prompt in) and **ids → text** (response out).

The target model's tokenizer (from the GGUF header, per [[ternary-bonsai-2-27b]]/[[qwen35-architecture]]): a **GPT-2-style byte-level BPE** (`tokenizer.ggml.model = gpt2`) with the **`qwen35` pre-tokenizer** (`tokenizer.ggml.pre = qwen35`), **248 320 tokens**, **BOS/PAD = 248044**, **EOS = 248046**. Two facts shape the whole engine's text handling:

- It is **byte-level**: the vocabulary stores the 256 raw bytes as single unicode-mapped characters, and merges run on those. A token can therefore span **part of one UTF-8 codepoint**, and detokenization must reassemble codepoints across token boundaries — the classic source of garbled multibyte output if any caller emits one token's piece at a time.
- The vocab is **large**: 248 320 rows against an embedding of 248 320 × 5120 — see the arithmetic under *Where it lives*.

## How it works

### Text → ids

Entry chain: `common_tokenize()` (`src/common/common.cpp:1902-1931`) → `llama_tokenize()` C API (`src/src/llama-vocab.cpp:4410-4417`) → `llama_vocab::impl::tokenize(raw_text, add_special, parse_special)` (`src/src/llama-vocab.cpp:3360-3560`).

1. **Dispatch.** `tokenizer.ggml.model = gpt2` selects `LLAMA_VOCAB_TYPE_BPE` and loads the merge table (`src/src/llama-vocab.cpp:1972-2010`); `tokenizer.ggml.pre = qwen35` selects `LLAMA_VOCAB_PRE_TYPE_QWEN35` and sets `clean_spaces = false` (`:2236-2238`). The tokenizer object is instantiated per type at `:3182-3184`.
2. **Special-token partition.** When `parse_special` is true, literal special-token strings in the prompt (e.g. `<|im_end|>`) are split out of the raw text *before* BPE, as token fragments, by `tokenizer_st_partition()` (`:3208-3330`), using `cache_special_tokens` — every id with `CONTROL | USER_DEFINED | UNKNOWN` attribute, sorted longest-first so the longest literal wins (`:2992-3007`). With `parse_special = false` the same strings are tokenized as ordinary text.
3. **Pre-tokenization.** Each raw-text fragment is split into words by `unicode_regex_split(text, tokenizer.regex_exprs, byte_encode)` (`llm_tokenizer_bpe_session::tokenize`, `:600`). The QWEN35 rule set (`:382-388`; the full regex is quoted in the code comment at `:385`, from Qwen's own `tokenizer.json`):
   - `(?i:'s|'t|'re|'ve|'m|'ll|'d)` — English contractions,
   - `[^\r\n\p{L}\p{M}]?[\p{L}\p{M}]+` — letter runs (combining marks attach),
   - `\p{N}` — one digit per word (digits do not merge across),
   - ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*` — punctuation/symbol runs with optional leading space,
   - `\s*[\r\n]+` and `\s+(?!\S)` and `\s+` — whitespace/newline runs.
   `[UNVERIFIED]` — the *active* `regex_exprs` strings at `:386-387` were read truncated; the commented original (quoted above) is what every sibling case in this file re-lists verbatim.
4. **BPE merges.** The GGUF array `tokenizer.ggml.merges` becomes `bpe_ranks[(first, second)] = i` (rank = position in the array, `:1998-2010`). Per word: split into initial byte-level symbols (each one UTF-8 char, `:630-637`), push every adjacent pair whose merge exists onto a priority queue keyed by rank (`add_new_bigram` → `vocab.find_bpe_rank`, `:725-750`), then greedily pop the lowest-rank merge, splice symbols, and re-queue the two new neighbour pairs (`:644-672`). Unmerged leftover bytes fall back to single-byte vocab entries — the byte-encoded branch looks up the byte-mapped character (`:697-710`; the mapping is `unicode_byte_to_utf8`, which is exactly how single bytes are stored in the vocab, `:3929-3930`); the alternative `<0xXX>`-format branch is for non-byte-encoded (gemma-4-style) tokenizers only. `[UNVERIFIED]` — the `byte_encode` flag value for `qwen35` itself was not read; the byte-encoded design is forced by the vocab layout at `:3929-3930`.
5. **BOS/EOS.** In the BPE branch of `impl::tokenize`: `if (add_special) session->append_bos(output)` before the fragments, `append_eos` + `check_double_bos_eos` after (`:3452-3473`). `append_bos` is a no-op unless `vocab.get_add_bos()` (`:570-575`) — the flag is read from GGUF `tokenizer.ggml.add_bos` at load (`:2594-2599`), **defaulting to false for BPE** (`:1815`). So the leading BOS is controlled by the `add_special` argument **and** the model's own metadata flag together; a prompt that already begins with the BOS string yields two BOS ids and only a warning (`:588-600`). `[UNVERIFIED]` — the value of `tokenizer.ggml.add_bos` in this model's GGUF was not re-read here.

Callers pass `add_special` per part: the server's `json_prompt_to_tokens` gives the first prompt part `add_special = params.add_special` and every later part `false` (`src/tools/server/server-common.cpp:864-868`) — so in chat only the first message gets the BOS. The TriAttention calibrator tokenizes its calibration corpus once per run with the model's own `add_bos` flag (`src/tools/triattention-calibrate/triattention-calibrate.cpp:232`).

There is **no text → ids cache** anywhere in the engine: every `llama_tokenize` call re-runs regex + merges. The "warm-up" at model load builds the *detokenization* cache instead (below).

### Ids → text

1. **Load-time warm-up cache.** At vocab load the engine precomputes the piece string for **every** token id: `cache_token_to_piece[id] = token_to_piece_for_cache(id, true)` for all 248 320 ids (`:3009-3024`; helper at `:3327-3341`), logged as `"token to piece cache size = %.4f MB"`. Per-token `token_to_piece` is then an O(1) cache hit (`:3603-3611`, `:3697-3699`).
2. **Per-token piece** — `llama_token_to_piece()` (`:4421-4427`) → `impl::token_to_piece` (`:3578-3699`):
   - `special = false` (the default callers use) makes any token with `UNKNOWN | CONTROL` attribute render as **nothing** (`if (!special && (attr & attr_special)) return 0;`, `:3580-3582`) — this is the skip-special gate. Notably **`USER_DEFINED` is not in the gate**: a user-defined special renders its literal text even with `special = false`.
   - `NORMAL` BPE tokens decode through `llama_decode_text()` (`:3349-3368`): each codepoint of the stored piece is mapped back through `unicode_utf8_to_byte`, i.e. the GPT-2 byte encoding is inverted **per codepoint**. A byte-level token that covers half a UTF-8 codepoint therefore yields a **partial byte sequence** — valid only once the following tokens' bytes arrive.
   - `BYTE`-attribute tokens yield their single byte directly (`token_to_byte`, `:3596-3599`).
3. **Whole-sequence** — `llama_detokenize()` (`:4431-4437`) → `impl::detokenize` (`:3701-3800`): concatenates the per-token pieces; with `remove_special` it drops a leading BOS and a trailing EOS first (`:3719-3731`); the leading-space trim and the ` ?`/` !`/` ,`/` 's` spacing post-passes run only when `add_space_prefix`/`clean_spaces` are set — **both false for `qwen35`** (`:1814`, `:2238`), so this model's detokenizer is a pure byte concatenation, which is exactly what reassembles split codepoints. `common_detokenize()` is the wrapper (`src/common/common.cpp:1958-1975`, whose comment notes *"the original tokenizer decodes bytes after collecting the pieces"*).

### Where the pieces become the response body (server streaming)

In `server_context::update_slots()` per sampled token (`src/tools/server/server-context.cpp`):

1. The piece is computed for the sampled id and carried as `result.text_to_send`; `slot.generated_text += token_str` with `token_str = result.text_to_send` (`:1765-1768`).
2. **Incremental UTF-8 buffering.** `slot.n_sent_text` tracks how many bytes of `generated_text` were already emitted ("handle partial UTF-8 on streaming", `:237`). Each step: `incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size()` (`:1775`); **while the tail is an incomplete UTF-8 sequence, nothing is sent** — the whole send/stop-word block is gated on `!incomplete` (`:1778`). When the tail completes (a later token supplies the missing bytes), `pos = min(n_sent_text, generated_text.size())` and `result.text_to_send = slot.generated_text.substr(pos)` is emitted, advancing `n_sent_text` (`:1779`, `:1798-1799`).
3. `send_partial_response()` ships `res->content = tkn.text_to_send` (`:1976-1998`); the final response ships the whole `slot.generated_text` in non-stream mode (`:2037`).

The naive implementation this guards against: emitting each sampled token's piece immediately. Because a byte-level BPE token can end mid-codepoint, that would stream invalid UTF-8; the server instead **withholds the unsent tail until it is complete** and never splits a codepoint. `common_token_to_piece` (default `special = false`) is the piece source, so control tokens add nothing to the body (`src/common/common.cpp:1940-1951`), and `llama_vocab_is_eog` on the sampled id is what stops generation at EOS (`server-context.cpp:1790`).

| Operation | Entry point | file:line | Who calls it |
| :--- | :--- | :--- | :--- |
| prompt → ids (whole) | `common_tokenize()` | `src/common/common.cpp:1902-1931` | CLI `completion.cpp:279`; server `server-common.cpp:864,881`; **TriAttention calibrator** `triattention-calibrate.cpp:232`; also `kv-mean-center.cpp:244`, `imatrix.cpp:804`, `perplexity.cpp:310`, `results.cpp:82`, `cvector-generator.cpp:282` |
| prompt → ids (engine) | `llama_tokenize()` → `impl::tokenize` | `src/src/llama-vocab.cpp:4410-4417` → `:3360-3560` | everything above |
| word split (pre-tokenizer) | `unicode_regex_split` in `llm_tokenizer_bpe_session::tokenize` | `src/src/llama-vocab.cpp:600` | via `impl::tokenize` |
| BPE merge application | `add_new_bigram` / merge queue loop | `src/src/llama-vocab.cpp:644-750` | via `impl::tokenize` |
| id → piece (streaming) | `common_token_to_piece()` → `llama_token_to_piece()` | `src/common/common.cpp:1940-1951` → `src/src/llama-vocab.cpp:4421-4427` → `:3578-3699` | server per sampled token (`server-context.cpp:1765-1768`); CLI verbose logs `completion.cpp:421` |
| ids → text (whole) | `common_detokenize()` → `llama_detokenize()` | `src/common/common.cpp:1958-1975` → `src/src/llama-vocab.cpp:4431-4437` → `:3701-3800` | server final response prompt echo `server-context.cpp:670,2041` |
| stream emit | `send_partial_response()` | `src/tools/server/server-context.cpp:1976-1998` | server only |

## Where it lives

| Path | What is there |
| :--- | :--- |
| `src/src/llama-vocab.cpp` | vocab load & GGUF keys `:1787-2415`; pre-tokenizer registry `:279-550` (QWEN35 `:382-388`); BPE merge table `:1972-2010`; special/attribute caches `:2992-3024`; partition `:3208-3330`; `impl::tokenize` `:3360-3560`; `token_to_piece` `:3578-3699`; `detokenize` `:3701-3800`; C API `:4410-4437` |
| `src/common/common.cpp` | `common_tokenize` `:1902-1931`; `common_token_to_piece` `:1940-1951`; `common_detokenize` `:1958-1975` |
| `src/tools/server/server-context.cpp` | `n_sent_text` `:237`; per-token piece + generated-text assembly `:1765-1799`; `send_partial_response` `:1976-1998`; final response `:2037-2041` |
| `src/tools/server/server-common.cpp` | prompt → tokens with per-part `add_special` `:864-881` |
| `src/tools/triattention-calibrate/triattention-calibrate.cpp` | corpus tokenization `:232` |
| `src/tools/completion/completion.cpp` | CLI prompt tokenization `:279`; verbose piece logs `:419-447` |

**The vocabulary's cost, as arithmetic.** `248 320 × 5120 = 1 271 398 400` embedding elements:

- f16: `1 271 398 400 × 2 B = 2 542 796 800 B ≈ 2.37 GiB` (÷ 2³⁰).
- f32: `× 4 B = 5 085 593 600 B ≈ 4.74 GiB`.
- one embedding row (one token): `5120 × 2 B = 10 240 B` f16.
- one logits row per decode position: `248 320 × 4 B = 993 280 B ≈ 0.95 MiB` f32 — sampling and output buffers therefore carry ~1 MiB per sampled index, bounded by `n_outputs_max_per_seq` ([[request-lifecycle]]).
- load-time piece cache: 248 320 strings, size reported by the loader ("token to piece cache size = … MB", `:3023`).

## Known issues

- **Partial-UTF-8 at token boundaries is the bug class to look for.** The engine is correct end-to-end: `llama_detokenize` concatenates bytes before the caller sees them, and the server withholds incomplete tails via `n_sent_text`/`validate_utf8` (`server-context.cpp:1775-1799`). Any new caller that pipes single `llama_token_to_piece` results straight into output (CLI logs, debug endpoints, tool output) will emit invalid UTF-8 whenever a token splits a codepoint. The `[UNK_BYTE_0x…]` fallback in `llama_decode_text` (`:3360-3364`) also appends the *entire piece* inside the marker instead of the single offending character — a latent double-include, reachable only if a piece contains a codepoint outside the 256-byte map.
- **`special = false` skips `UNKNOWN | CONTROL` but not `USER_DEFINED` tokens** (`:3580-3582`): a user-defined special renders its literal text even in "skip specials" mode. Correct only as long as the model's control tokens carry the `CONTROL` attribute.
- **Double BOS/EOS is warned about, never deduplicated** (`:588-600`): with `add_special` on and a prompt that literally starts with the BOS string, the batch gets two BOS ids.
- **No text → ids cache**: repeated identical prompts (chat re-send, calibrator corpus) re-run the full regex + merge pass every time; only ids → text is cached, and that cache is *all* 248 320 pieces at load (`:3009-3024`).
- The qwen35-specific `add_bos` metadata value, the `byte_encode` flag, and the exact active regex strings were not re-read from source/GGUF this pass — see the `[UNVERIFIED]` marks above.

## The unicode and UTF-8 layer

Gap #10. Four files, and only one of them has logic:

| File | Size | Role |
| :--- | :--- | :--- |
| `src/src/unicode-data.h` | 20 lines | `MAX_CODEPOINTS = 0x110000` (`:14`); `struct range_nfd { first, last, nfd }` (`:8-12`); five `extern` table declarations (`:16-20`) |
| `src/src/unicode-data.cpp` | 7 034 lines / 168 KB | the tables and nothing else — five `const std::initializer_list`/`unordered_set` aggregates |
| `src/src/unicode.cpp` | 1 408 lines / 55 KB | every operation: decode/encode, the two byte maps, category flags, and the per-model regex pre-splitters |
| `src/src/unicode.h` | 111 lines | the API surface, `:92-111` |

### What the tables are

Extents read from `unicode-data.cpp`:

- `unicode_ranges_flags` `:10-2284` — `{codepoint_start, uint16 flags}` runs (~2 040 entries), the tree's only encoding of Unicode general category plus whitespace/case/NFC properties. Consumed **eagerly**: `unicode_cpt_flags_array()` (`unicode.cpp:116-146`) walks the ranges into a `std::vector<unicode_cpt_flags>` of `MAX_CODEPOINTS = 1 114 112` entries on first use (a function-local static behind `unicode_cpt_flags_from_cpt`/`_from_utf8`, `:1147-1160`), then overlays `unicode_set_whitespace`, both case maps, and the NFD targets. `[INFERENCE]` — the vector is ~4 MiB if the flag bitfield packs into 32 bits; the allocation happens on the first tokenization, not at vocab load.
- `unicode_set_whitespace` `:2286-2312`; `unicode_map_lowercase` `:2315-3749`; `unicode_map_uppercase` `:3752-5203` (binary-searched by `unicode_tolower`, `unicode.cpp:1172-1182` — there is **no** `unicode_toupper`); `unicode_ranges_nfd` `:5205-7034`.

### What `unicode.cpp` provides

- **Byte length of a lead byte** — `unicode_len_utf8` `:16-20`, a high-nibble lookup `{1×12, 2, 2, 3, 4}`. Structural only; it does not validate.
- **Decode** — `unicode_cpt_from_utf8` `:30-60` does validate (continuation bytes must be `0b10xxxxxx`, truncation rejected) and throws `std::invalid_argument`. `unicode_cpts_from_utf8` `:1130-1144` wraps it and **swallows** the throw: it substitutes U+FFFD and advances one byte. *No malformed UTF-8 escapes this layer as an exception*; it degrades silently to replacement characters.
- **Encode** — `unicode_cpt_to_utf8` `:1088-1114`.
- **The GPT-2 byte↔unicode maps** — `unicode_byte_to_utf8_map()` `:148-170` and its inverse `unicode_utf8_to_byte_map()` `:172-193`, lazily-built function-local statics behind `unicode_byte_to_utf8` `:1162-1165` / `unicode_utf8_to_byte` `:1167-1170`; both use `map.at(...)`, so a miss **throws `std::out_of_range`**. Coverage is exactly 256 codepoints: U+0021–U+007E (94), U+00A1–U+00AC (12), U+00AE–U+00FF (82), then U+0100–U+0143 (68) for the bytes with no Latin-1 glyph. Every other codepoint — `▁` U+2581, `中`, U+FFFD, U+2192 — is a miss.
- **Pre-tokenizer split** — `unicode_regex_split` `:1216ff` (tail after `:1270` not read this pass `[UNVERIFIED]`), which collapses codepoints to one byte each when a regex uses `\p{…}` (tables at `:1218-1250`) and dispatches via `unicode_regex_split_custom` `:1050-1087` to hand-written matchers: `gpt2` `:215`, `llama3` `:333`, `qwen2` `:474`, **`qwen35` `:610`**, `kimi_k2` `:777`, `afmoe` `:948`, `newlines` `:1023`, else the STL fallback `unicode_regex_split_stl` `:739`. When the caller passes `byte_encode = true`, `unicode_byte_encoding_process` `:196-212` byte-encodes every byte of every word through `unicode_byte_to_utf8` before BPE lookup.
- Smaller surface: `unicode_cpts_to_utf8` `:22`, `unicode_cpts_normalize_nfd` `:1117`, `unicode_cpt_is_han` `:1184-1214` (`[UNVERIFIED]` — no caller outside its own declaration was found, but the grep for it was not individually exhaustive).

### Which parts the tokenizer path depends on

| Tokenizer stage | unicode symbol | file:line |
| :--- | :--- | :--- |
| word split (both BPE + SPM) | `unicode_regex_split` (+ `byte_encode`) | `src/src/llama-vocab.cpp:605`; `src/src/unicode.cpp:1216, 1050` |
| BPE symbol split into UTF-8 chars | `unicode_len_utf8` | `src/src/llama-vocab.cpp:123`, `:632` |
| byte-encode the split words (all byte-level BPE) | `unicode_byte_to_utf8` | `src/src/unicode.cpp:207` |
| BPE leftovers → single-byte vocab rows | `unicode_byte_to_utf8` | `src/src/llama-vocab.cpp:704-706` |
| piece → text (detokenization) | `unicode_cpts_from_utf8`, `unicode_cpt_to_utf8`, `unicode_utf8_to_byte` | `src/src/llama-vocab.cpp:3354-3358` |
| SPM/UGM normalizer + WPM | `unicode_cpts_from_utf8`, `unicode_cpt_to_utf8`, `unicode_cpts_normalize_nfd`, `unicode_tolower`, `unicode_len_utf8`, flag predicates | `src/src/llama-vocab.cpp:819-845`, `:987`, `:1388-1465`, `:1728-1733` |

The target model uses none of the flag predicates on the tokenize path: `qwen35` is a byte-level pre-split (`byte_encode = true`), so the codepoint *category* tables never run for it — its cost is the regex matcher plus the two byte maps.

### Verdict on the `[UNK_BYTE_0x…]` fallback: real defect; unreachable for qwen35; live for raw-UTF-8 BPE pre-types

The code (`src/src/llama-vocab.cpp:3351-3366`), quoted in full because the argument name is the whole point:

```cpp
static std::string llama_decode_text(const std::string & text) {          // :3351
    std::string decoded_text;

    const auto cpts = unicode_cpts_from_utf8(text);                        // :3354
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_cpt_to_utf8(cpt);                        // :3356
        try {
            decoded_text += unicode_utf8_to_byte(utf8);                    // :3358  map.at() -> may throw
        } catch (const std::out_of_range & /*e*/) {
            decoded_text += "[UNK_BYTE_0x";                                // :3360
            for (const auto c : utf8) {
                decoded_text += format("%02x", (uint8_t) c);
            }
            decoded_text += text + "]";                                    // :3364  <-- whole piece, not `utf8`
        }
    }
    return decoded_text;
}
```

`:3364` appends **`text`, the function's entire argument** — the token's stored piece — not the single codepoint `utf8` whose hex was printed one line earlier. The earlier slice's reading was right:

- the hex in the marker is correct (it comes from `utf8`), the payload is not;
- a piece with *k* out-of-map codepoints gets the whole piece appended *k* times;
- the decoded prefix is still concatenated, so the output is `«decoded prefix»[UNK_BYTE_0x…«raw piece»]` — the piece appears twice, once in each encoding, rather than being substituted.

**Reachability.** `llama_decode_text` has exactly one caller: `impl::token_to_piece`, BPE branch, `LLAMA_TOKEN_ATTR_NORMAL`, only when `escape_whitespaces == false` (`src/src/llama-vocab.cpp:3642-3649`). Two facts pin it down:

1. `escape_whitespaces` is false for **every** BPE model — the BPE init branch sets it (`:2131`), and the only two sites raising it again are the `gemma4` and `sarvam-moe` pre-names (`:2205`, `:2209`), i.e. exactly the two SPM-style raw-UTF-8 pre-types that would otherwise trip the fallback. That is the guard rail around this code.
2. The catch needs a piece codepoint outside the 256-entry map. `LLAMA_VOCAB_PRE_TYPE_QWEN35` has **no** case in the `byte_encode` switch (`:498-560`), so it keeps the default `byte_encode = true` (`:558`) → its NORMAL pieces are byte-encoded by construction → *every* codepoint is in the map → **the catch cannot fire for this model, for any prompt, token or correctly-built GGUF.** The page's condition ("reachable only if a piece contains a codepoint outside the 256-byte map") is correct and can be strengthened to: dead code on the qwen35 path.

Where it is live: a BPE model whose pre-type sets `byte_encode = false` **and** leaves `escape_whitespaces = false`. The only three `byte_encode = false` sites are `:520` (GEMMA4), `:528` (SARVAM_MOE) and `:543` (`LLAMA_VOCAB_PRE_TYPE_WHITESPACE`, `:538-544`, the jinaai/jina-embeddings-v2-base-zh whitespace pre-tokenizer). The first two are covered by rule 1; **WHITESPACE is not**. There the vocab keeps raw UTF-8 pieces, so any NORMAL token containing a codepoint outside the 256-set — any CJK character, any `▁` space marker, emoji — takes the catch. `[UNVERIFIED]` which `tokenizer.ggml.pre` string selects `LLAMA_VOCAB_PRE_TYPE_WHITESPACE` (the jina branch at `:2211-2213` was read only as a condition) and whether a real jina GGUF stores raw-UTF-8 pieces.

**Blast radius when it fires.** The string is produced during the *load-time* piece-cache build — `cache[id] = token_to_piece_for_cache(id, true)` for all `n_tokens` at `:3016` (`token_to_piece_for_cache` `:3331-3345`, called with `special = true`) — and `impl::token_to_piece(llama_token)` then just returns `cache_token_to_piece.at(token)` (`:3700-3702`). So the marker string *is* that token's detokenization everywhere: server body ([[server-layer]]), prompt echo, CLI. It never throws, never logs, never corrupts memory. Silent text corruption is why it survived a slice without being noticed.

> Contradiction (2026-09-29): the *Known issues* bullet frames the trigger as "a malformed sequence" and the bug as merely latent/untriggered. Read against the code: (a) malformed UTF-8 is not needed — `unicode_cpts_from_utf8` (`unicode.cpp:1130-1144`) rewrites invalid bytes to U+FFFD, and U+FFFD is itself a map miss, so garbage *does* fire it; but a perfectly well-formed non-Latin-1 codepoint fires it too, which makes it the **normal** case for a raw-UTF-8 pre-type, not an edge case; (b) it is not merely untriggered — it is unreachable for this engine's model (`byte_encode = true` at `:558`, no QWEN35 override) while live for `LLAMA_VOCAB_PRE_TYPE_WHITESPACE`-class models built by the same binary. Verdict: real defect at `:3364`, wrong `text` instead of `utf8`; not a qwen35 bug.

### Do the chat and grammar paths share this layer?

**Grammar ([[sampling]]) does not share `unicode.cpp` — it has a private, second UTF-8 decoder.** `src/src/llama-grammar.cpp` defines `decode_utf8(const char *)` `:18-32` and `decode_utf8(const std::string &, llama_partial_utf8)` `:34-92` (with `lookup` `{1×8, 0,0,0,0, 2,2,3,4}` — the 0 entries are the "this cannot start a sequence" case), plus `llama_grammar_match_partial_char` `:791-800` for codepoints split across tokens. Its only touch of this layer is *indirect and one-way*: candidate text comes from `grammar.vocab->token_to_piece(id)` (`:1376`, `:1399`), i.e. the same cached string `llama_decode_text` produced. If the marker bug fired, the grammar would see `[UNK_BYTE_0x…]`-shaped ASCII, decode it happily, and **accept it as legal content** — the constrained output would carry the marker instead of the intended characters, and the token would not be masked away, because from the grammar's viewpoint the text is well-formed. That is a different visible failure from the unconstrained path (garbage that passes a grammar, rather than garbage that is merely printed).

**The Jinja/chat path ([[chat-templates]]) touches it once.** `src/common/chat.cpp:824` calls `common_token_to_piece(vocab, token, true)` — the same `impl::token_to_piece` (with `special = true`); nothing else in that file mentions unicode/UTF-8, so the template renderer itself passes raw bytes through and never enters the codepoint layer. `[UNVERIFIED]` — the purpose of that call site (context at `:824` not read).

**The server does not use this layer at all.** The streaming partial-sequence guard uses a *separate*, server-local validator, `validate_utf8` (`src/tools/server/server-common.cpp:887`), called at `server-context.cpp:1775`, `server-task.cpp:268,286` and `server-tools.cpp:48`. So the tree carries two independent UTF-8 validators (`unicode_cpt_from_utf8` here, `validate_utf8` there) with no shared code — a divergence between them is invisible to both. `[INFERENCE]` — based on a repo-wide grep for the `unicode_*` symbols, which found no hits under `src/tools/`.

**Error propagation.** No exception from this layer reaches [[forward-pass]] or [[server-layer]]: `unicode_cpt_from_utf8`'s `std::invalid_argument` is eaten in `unicode_cpts_from_utf8`, and `unicode_utf8_to_byte`'s `std::out_of_range` is the only one that escapes `unicode.cpp` — caught one frame up inside `llama_decode_text`, and only on the BPE detokenization path. The layer's failure mode is always *wrong text*, never a crash.

## See also

[[ternary-bonsai-2-27b]] · [[forward-pass]] · [[sampling]] · [[server-layer]] · [[conversion-and-packing]] · [[qwen35-architecture]] · [[chat-templates]]
