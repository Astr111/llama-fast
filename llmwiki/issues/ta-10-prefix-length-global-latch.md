---
title: "TA-10: prefix_length is a per-context latch the server never resets"
type: issue
status: current
updated: 2026-09-28
sources: [state.md, TRIATTENTION.md]
verified: [src/src/llama-kv-cache.cpp, src/src/llama-kv-cache.h, src/src/llama-triattention.cpp, src/src/llama-context.cpp, src/src/llama-arch.cpp, src/tools/server/server-context.cpp, src/tools/server/server.cpp, src/common/arg.cpp, scripts/start_server_turbo.sh]
tags: [triattention, kv-eviction, server, correctness]
---

# TA-10: `prefix_length` is latched per context and survives request boundaries

> **Not in either inventory.** Found 2026-09-28 while resolving the `prefix_length` question (ranked #10 in [[open-questions]]); confirmed from code, never executed.

## Symptom

On a server that reuses its context — i.e. any real deployment — the token range TriAttention treats as *protected prompt* is whatever the **first** request established. A later request with a **longer** prompt has the middle of its own prompt classified as evictable and can lose it mid-request.

## Cause

`prefix_length` is not per-sequence. It is a single value inside one `triattention_state` per KV cache (`src/src/llama-kv-cache.h:344`):

1. **Set once**, by the first batch containing position 0, keyed on the **batch** rather than on `seq_id` (`src/src/llama-kv-cache.cpp:1354-1363`).
2. **Reset only** by `triattention_on_reset`, which is called from exactly one place: `llama_kv_cache::clear` (`src/src/llama-kv-cache.cpp:528`).
3. **The server never full-clears.** Slot recycling is `mem.seq_rm(id, -1, -1)` (`src/tools/server/server-context.cpp:292`), which drops cell positions but does not call `clear` — so the latch survives every request after the first.

The two failure directions differ:

- **Later, shorter prompt** — mostly harmless. The stale boundary is a superset; the phantom range holds no cells after `seq_rm`, so the extra protection covers nothing.
- **Later, longer prompt** — the harmful case. The region between the old and the new prompt end fails `is_prefix` (`src/src/llama-triattention.cpp:1136-1137`) and is treated as ordinary history, so it is **evictable while the request is still running**. The prompt the user just sent is the least protected part of the context.

`-np 1` in `scripts/start_server_turbo.sh` does **not** protect against this: `-np` sets the number of server slots (`src/common/arg.cpp:2557-2566`), and a single slot still recycles the same context across sequential requests. It removes the multi-slot variant of the problem and nothing else.

Related paths that inherit the latch: KV-sequence forks (`seq_cp`, `src/tools/server/server-context.cpp:682`) share the parent's state. The `llm_arch_supports_rs_rollback` path (`src/src/llama-arch.cpp:1118-1122`) is about the model's recurrent state and never touches TriAttention.

## Impact

**HIGH (new).** Breaks TriAttention's central promise — prompt-token protection — for every request after the first on a reused context. Because the affected range is the *newest* prompt, the damage is exactly where a user would notice quality loss, and it is invisible in any measurement that serves a single request. Severity is bounded by the shipped configuration's other defects only in the sense that [[ta-8-offset-max-zero-nan]] already makes selection arbitrary; fixing TA-8 first is what would make this observable.

## Location

- `src/src/llama-kv-cache.h:344` — one `triattention_state` per cache
- `src/src/llama-kv-cache.cpp:1354-1363` — the single, batch-keyed assignment
- `src/src/llama-kv-cache.cpp:528` — the only reset path (`clear`)
- `src/tools/server/server-context.cpp:292` — slot recycle via `seq_rm`, not `clear`
- `src/src/llama-triattention.cpp:1136-1137` — `is_prefix`, the predicate a longer prompt fails
- `src/common/arg.cpp:2557-2566` — what `-np` actually controls

## Status

**Open, unlisted, unfixed.** Decided by reading; nothing was built or run ([[build-and-verify]]).

## Fix sketch

Make the protection boundary per-sequence, not per-context: store `prefix_length` keyed by `seq_id` (or in the per-sequence slot state) and set it when that sequence's own position 0 is seen, so a new request establishes its own boundary. Failing that, a conservative guard — treat `prefix_length` as a minimum over live sequences rather than a latch — is smaller but only moves the failure. The settling experiment is one run: serve a short prompt, then a longer one, and read the `[prefix=%lld]` prune counter (`src/src/llama-triattention.cpp:1451-1452`) — if it still reports the first request's length, the latch is live.

## See also

[[kv-eviction]] · [[triattention]] · [[request-lifecycle]] · [[server-layer]] · [[ta-2-budget-starvation]] · [[ta-7-config-validation]] · [[open-questions]]
