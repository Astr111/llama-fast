---
title: Server layer
type: topic
status: current
updated: 2026-09-28
sources: [state.md, README.md]
verified: [src/tools/server/server.cpp, src/tools/server/server-http.cpp, src/tools/server/server-context.cpp, src/tools/server/server-context.h, src/tools/server/CMakeLists.txt, src/common/arg.cpp, src/common/common.cpp, src/common/speculative.cpp, src/src/llama-context.cpp, scripts/start_server_turbo.sh]
tags: [server, scheduling, speculative-decoding]
---

# Server layer

## Bottom line

`llama-server` is **not** a thin wrapper over the engine: it owns a fixed pool of *slots*, a task queue that turns HTTP requests into slot launches, a continuous-batching step loop, and the only place in the repo where speculative-acceptance statistics are computed and printed. The engine (`llama_decode`) sees a batch assembled by the server, not one request; the request-to-slot mapping, the deferral policy when all slots are busy, and the streaming result path all live in `src/tools/server/server-context.cpp`.

Two consequences matter for this vault:

1. **The recorded throughput numbers are per-slot, per-release stdout lines, not persisted measurements.** The `eval time = … tokens per second` field and the `draft acceptance = …` line come from `server_slot::print_timings()` (`src/tools/server/server-context.cpp:588-638`), called immediately before a slot is released (`:3084-3087`, `:3801-3803`, `:3921-3924`). Nothing writes them to disk, and the launch profiles checked in here **configure no drafter at all** (see *Statistics*), so the `31.6 → 38.1+ t/s` speculative trend cannot be reproduced from this tree.
2. **`-np` is the single knob that decides both slot count and per-slot context.** `scripts/start_server_turbo.sh` pins `-np 1`, which is why every recorded number is a single-sequence number with the full `-c 32768` sequence length.

## Evidence

### The request path

| Stage | Where |
| :--- | :--- |
| HTTP routes bound | `src/tools/server/server.cpp:234-292` (`ctx_http.get/post("/completions", ex_wrapper(routes.post_completions))`, `:242`) |
| Route slots declared as `server_context` members | `src/tools/server/server-context.h:137-156`; completions funnel through `server_res_generator` / `handle_completions_impl` (`server-context.h:162-165`) |
| Request → task | Handlers build a `server_task` with a `SERVER_TASK_TYPE_*` and post it to the queue: `server-context.cpp:4814` (`/completion`), `:4826` (`/completions`), `:4842` (chat), `:4901` (responses), `:4802` (infill), `:5092` (rerank — one task per document) |
| Tokenization thread | Comment at `:2302-2303`: for HTTP-supplied prompts "no need to tokenize as it's already done inside the HTTP thread" |
| Queue → scheduler | `queue_tasks.on_new_task(process_single_task)`, `queue_tasks.on_update_slots(update_slots)` (`:1356-1360`) |
| Task → slot | `process_single_task()` (`:2289`): declines everything but `SERVER_TASK_TYPE_METRICS`/`SLOT_GET` while a decode is yielding (`:2291-2294`); `get_available_slot(task)` (`:2312`); if none free, `queue_tasks.defer(std::move(task))` (`:2318`) and the task is retried after the decode (`:2288`); otherwise `launch_slot_with_task` (`:2345`), or `launch_slots_with_parent_task` for a multimodal parent with child slots (`:2335`, `:2219-2232`) |
| Slot launch | `launch_slot_with_task` (`:1644`): LoRA/alora resolution, token validation (`:1711-1714`, `ERROR_TYPE_INVALID_REQUEST` on failure), per-request sampler `slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling))` (`:1720`), optional backend-sampling registration `llama_set_sampler(ctx_tgt, slot.id, …)` (`:1737-1741`), then `slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict` (`:1746`) |
| Slots exist | `for (int i = 0; i < params_base.n_parallel; i++) slots.emplace_back();` (`:1182-1184`), logged as `initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'` (`:1178-1179`) |
| Continuous batching | One `server_batch batch` sized `max(llama_n_batch(ctx_tgt), params_base.n_parallel)` (`:1293-1297`); `update_slots()` (`:2707`) fills it from *all* processing slots and issues one decode per step — a step is not a request |
| Step loop / result | `update_slots()` appends accepted tokens to `slot.prompt` (`:3900`); on a stop condition it does `slot.print_timings(); send_final_response(slot); slot.release();` (`:3084-3087`, `:3801-3803`, `:3920-3924`) |
| Streaming back | `if (slot.task->params.stream) send_partial_response(slot, result, false);` (`:1806-1808`) → `server_task_result_cmpl_partial` (`:1976`) posted to the results queue; the HTTP thread checks `response->is_stream()` in `process_handler_response` (`src/tools/server/server.cpp:545-547`) and chunk-writes. Progress-only partials: `:3334`, `:3722-3723`. Non-streamed requests end at `send_final_response` (`:2019`) |
| Binary target | `src/tools/server/CMakeLists.txt`: static lib `server-context` (all server logic incl. `server-context.cpp`), lib `llama-server-impl` (`server.cpp`, `server-http.cpp`, `server-models.cpp`) linking `server-context llama-ui cpp-httplib`, executable `llama-server` = `main.cpp` |

Per-slot context is derived, not set: `n_ctx_slot = llama_n_ctx_seq(ctx_tgt)` capped to `llama_model_n_ctx_train` (`:1160-1165`), and `n_ctx_seq = n_ctx / n_seq_max` (padded to 256) unless `kv_unified` (`src/src/llama-context.cpp:381-395`), with `cparams.n_seq_max = params.n_parallel` (`src/common/common.cpp:1748`). So the slot budget is `-c / -np`.

### Endpoints

Registered in single-model mode at `server.cpp:234-292`:

- **Health/ready**: `/health`, `/v1/health` — public, no API key, and the only endpoints answered while the model is loading (everything else gets 503 `Loading model`, `server-http.cpp:196-215`, `:257-271`).
- **Completions/chat**: `/completion` (legacy), `/completions`, `/v1/completions`, `/chat/completions`, `/v1/chat/completions`, `/responses` + `/v1/responses`, `/v1/messages` (Anthropic), `/infill`, `/v1/chat/completions/control` (live mid-generation control, `:2401-2403`).
- **Embeddings/rerank**: `/embedding` + `/embeddings` + `/v1/embeddings`, `/rerank`/`/reranking`/`/v1/rerank`/`/v1/reranking`. Gated: embeddings require `--embeddings` (`server-context.cpp:5294-5296`; flag `src/common/arg.cpp:3467-3470`); rerank additionally requires rank pooling (`:5051-5053`; `--reranking` also sets `params.embedding = true`, `arg.cpp:3474-3477`). `server_output_limits()` (`:41-48`) collapses the speculative output limits to `{n_batch, 1}` in these modes.
- **Introspection**: `/props` (GET+POST) reports `endpoint_slots`, `endpoint_props`, `endpoint_metrics` (`:4526-4528`) plus `total_slots = params.n_parallel` (`:4516`); `/metrics` requires `--metrics` (`:4571-4573`; `arg.cpp:3577-3581`) and runs as a `SERVER_TASK_TYPE_METRICS` task with `metrics_reset_bucket` ("gauges are averaged over the window between two scrapes", `:4596-4599`); `/slots` requires `--slots` (`:4633-4635`; `arg.cpp:3591-3595`).
- **Utilities**: `/tokenize`, `/detokenize`, `/apply-template`, `*/input_tokens` counters, `/lora-adapters` hotswap, `/slots/:id_slot`, and resumable streaming `/v1/stream` + `/v1/streams/lookup` (`:290-292`, `server-stream.h`).
- **Deliberately not exposed in single-model mode**: model management. `/models/load`, `/models/unload`, `/models/sse`, `/models` (POST/DELETE) exist **only** when `is_router_server` — i.e. launched with no `-m`, no `-hf`, no `--docker` (`server.cpp:135-137`, `:227-231`); in router mode every data route is overwritten with a proxy to a child instance (`:198-221`). There is no OpenAI files/models-retrieval surface.

### Statistics — what `server-context.cpp:615-636` records

`server_slot::print_timings()` (`:588`) runs once per slot release and prints three INFO blocks:

1. Timing, from `server_slot_stats` (`:589-613`): `prompt eval time = %10.2f ms / %5d tokens (… %8.2f tokens per second)` (`stats.n_prompt_processed`, `t_prompt_ms()`), **`eval time = %10.2f ms / %5d tokens (… %8.2f tokens per second)`** (`stats.n_gen`, `t_gen_ms()`, `stats.n_gen_tps()` — this is the `t/s` in every recorded number), `total time = … ms / … tokens`, and `graphs reused = %10d` from `llama_perf_context(ctx_tgt).n_reused` ([[cuda-graphs]]).
2. Draft acceptance — printed **only if `stats.n_draft_tokens > 0`** (`:614-618`):
   - `draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f`, where `draft_ratio = n_draft_accepted / n_draft_tokens` and `mean_acc_len = 1 + n_draft_accepted / n_draft_verif_steps` (`:619-620`, `:632-634`).
   - `acc per pos = (…)` — per-draft-position acceptance `n_accepted_per_pos[i] / n_draft_verif_steps` (`:622-636`), emitted only at trace verbosity via `SLT_TRC` (`:635`).
3. `common_speculative_print_stats(spec)` (`:638`) — per speculative impl: `dur(b,g,a) = <begin>, <draft>, <accept> ms` and `#mean acc len = …, #acc rate/pos = (…)` (`src/common/speculative.cpp:3622-3653`).

Where the counters come from: `slot.stats.n_draft_tokens += draft.size()` (`:2969`); after verification, `slot.stats.n_draft_accepted += n_accepted; slot.stats.n_draft_verif_steps += 1;` and `n_accepted_per_pos[i]++` for `i < n_accepted`, with the vector sized `common_speculative_n_max(&params_base.speculative)` (`:3888-3898`); accumulated into server-lifetime `metrics.n_draft_*` (`:4039-4044`). All of it is cleared per slot (`:351-352`).

**Why the 31.6 → 38.1+ t/s trend ([[source-state-md]] §1.2 → [[speculative-decoding]]) cannot be reproduced here.** The numbers are single-run console output of one slot's wall-clock timing plus its acceptance counters; they are never archived. And the measured configuration does not exist in the tree: `scripts/start_server_turbo.sh` (read in full) passes `-m`, `-ctk`, `-ctv`, the TriAttention flags, `-ngl 99`, `-c 32768`, `-n 8192`, `--reasoning-budget 4000`, `--reasoning-budget-message …`, `-np 1`, `-t $(nproc)`, `--host`, `--port` — **no draft-model flag (`-md`/`--model-draft`) and no `--speculative-*` flag**. With no drafter configured, `stats.n_draft_tokens` stays 0 (`common_speculative_init` is only attempted when the target context supports sequence removal, `:1186-1192`) and the whole acceptance block at `:614-636` is skipped — i.e. those statistics were produced by a command line that is not checked in.

**To reproduce it a reader must capture**: (a) the exact server command line including the draft model and speculative flags, since they decide whether the block prints at all; (b) the raw INFO stdout of each run — specifically the `eval time = … tokens per second` field and the `draft acceptance = … accepted / … generated, mean len = …` line; (c) trace-level output for `acc per pos = (…)` and `#acc rate/pos`; (d) `graphs reused = N` per release, which is how the CUDA-graph saving appears in the same log. Without (a) the acceptance line never appears; without archiving the logs elsewhere, every `t/s` figure in this vault stays `[UNVERIFIED]`.

### Launch flags at this layer

From `scripts/start_server_turbo.sh`:

| Flag | Consumed by |
| :--- | :--- |
| `-np 1` | `--parallel` (`src/common/arg.cpp:2557-2566`) → `params.n_parallel` → `cparams.n_seq_max` (`src/common/common.cpp:1748`) → `n_ctx_seq = n_ctx / n_seq_max` (`src/src/llama-context.cpp:381-395`). Effect: **one slot** (`server-context.cpp:1182-1184`) owning the entire `-c 32768` sequence; also sizes the step batch floor `max(n_batch, n_parallel)` (`:1291-1297`) and is passed to `common_speculative_init(…, params_base.n_parallel)` (`:1189`). With `-np 4` the same `-c` would give each slot 8192. |
| `-n 8192` | `-n, --predict, --n-predict` (`arg.cpp:1662-1663`) → `params.n_predict`, consumed as the per-request default: `slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict` (`server-context.cpp:1746`). Effect: **generation cap per slot/request**, not a context size; a request-level `n_predict` overrides it. |
| `-c 32768` | `params.n_ctx` → `cparams.n_ctx` (`src/common/common.cpp:1747`) → per-slot `n_ctx_seq` capped at `n_ctx_train` (`server-context.cpp:1160-1165`). |
| `--reasoning-budget 4000` | `params.sampling.reasoning_budget_tokens` (`arg.cpp:3709-3712`, env `LLAMA_ARG_THINK_BUDGET`); the server forwards it into the chat-handler/parser setup at `server-context.cpp:1433-1435`, where the sampler's reasoning-budget stage consumes it (see [[sampling]]); the live control route can force the budget mid-generation via `common_sampler_reasoning_budget_force(slot->smpl.get())` (`:2401-2403`). Because it lives under `params.sampling`, it is a per-request *default* — a request body may override it. |
| `--host` / `--port` | `params.hostname` (`arg.cpp:3312-3317`) / `params.port` (`arg.cpp:3319-3323`), bound by the HTTP context in `server.cpp` (bind call not read here). |

### What this page adds over the neighbours

[[request-lifecycle]] follows one prompt through tokenization, the graph and the decode; [[sampling]] follows the logits into the sampler chain. Neither explains that a prompt first has to *win a slot*: that requests queue in `queue_tasks` and are deferred wholesale when every slot is occupied (`:2318`), that one `llama_decode` per step carries tokens from several slots at once, that `n_ctx` is divided by `-np` before a slot ever sees it, or that the reported `t/s` and acceptance rates are per-slot console lines produced at release time rather than stored measurements. Those four facts are what make a recorded benchmark number interpretable — and what make an unreproducible one recognisable.

## Open questions

- Does the Prometheus `/metrics` output expose the `metrics.n_draft_*` counters (`:4039-4044`), i.e. could acceptance be captured by scraping instead of reading stdout? The accumulation exists; the exporter's field list was not read. `[UNVERIFIED]`
- `endpoint_props` appears in `/props` (`:4527`) but the gate that turns `--no-props` into a rejection was not located. `[UNVERIFIED]`
- Which command line produced the 31.6 → 38.1+ t/s trend, and was server `print_timings` or `llama-bench` its source? The launch scripts in this tree cannot produce it.

## See also

[[request-lifecycle]] · [[sampling]] · [[speculative-decoding]] · [[performance-profile]] · [[benchmarks]] · [[v100-sxm2]] · [[cuda-graphs]] · [[overview]]
