---
title: Tokenizer (qwen35 BPE)
type: entity
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/src/llama-vocab.cpp, src/common/common.cpp, src/tools/server/server-context.cpp, src/tools/triattention-calibrate/triattention-calibrate.cpp, src/tools/completion/completion.cpp]
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

## See also

[[ternary-bonsai-2-27b]] · [[forward-pass]] · [[sampling]] · [[server-layer]] · [[conversion-and-packing]] · [[qwen35-architecture]] · [[chat-templates]]
