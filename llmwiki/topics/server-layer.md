---
title: Server layer
type: topic
status: current
updated: 2026-09-29
sources: [state.md, README.md]
verified: [src/tools/server/server.cpp, src/tools/server/server-http.cpp, src/tools/server/server-context.cpp, src/tools/server/server-context.h, src/tools/server/CMakeLists.txt, src/common/arg.cpp, src/common/common.cpp, src/common/speculative.cpp, src/src/llama-context.cpp, scripts/start_server_turbo.sh, src/tools/server/server-task.cpp, src/tools/server/server-task.h, src/tools/server/server-queue.cpp, src/tools/server/server-queue.h, src/tools/server/server-stream.cpp, src/tools/server/server-stream.h, src/tools/server/server-mcp.cpp, src/tools/server/server-mcp.h, src/tools/server/server-tools.cpp, src/tools/server/server-schema.cpp, src/tools/server/server-chat.cpp, src/common/fit.cpp, src/common/common.h, src/ggml/src/ggml-cuda/ggml-cuda.cu]
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

## Beyond the main loop

The first pass (above, [[server-layer]]) covered the slot loop, the endpoints and the acceptance counters. The subsystems that loop hands off to live in the same directory and were unread: task scheduling (`server-task.*`, `server-queue.*`), streaming (`server-stream.*`), the tool-calling surface (`server-tools.cpp`, `server-schema.cpp`) and MCP (`server-mcp.*`). This section is what they add. The per-request path from HTTP to a launched slot stays in the main loop; [[request-lifecycle]] picks it up once the slot is running.

### Task scheduling — one queue, no numeric priorities

A `server_task` (`src/tools/server/server-task.h:136`) is the queue unit, and it is **not** only an inference request. `server_task_type` (`server-task.h:15-30`) has fifteen kinds: `COMPLETION`, `EMBEDDING`, `RERANK`, `INFILL`, plus `CANCEL`, `CONTROL` (the live mid-generation control verb), `NEXT_RESPONSE`, `METRICS`, `SLOT_GET/SAVE/RESTORE/ERASE`, `GET_LORA`, `SET_LORA`. The task carries the per-request `task_params params` and the tokenized `server_tokens tokens` (`server-task.h:152-154`), an `id_target` for cancels, an `id_slot` plus a `slot_action {id_slot, filename}` payload for slot saves (`server-task.h:142-170`), and `metrics_reset_bucket` for scrapes. `params.sampling` is what `launch_slot_with_task` later turns into `slot.smpl`, which is where [[sampling]] enters.

`server-task.cpp` is **not** the scheduler — it is the task/result data layer: `task_params::to_json` (`:30`), `task_result_state` and its chat-message / partial-diff state machine (`:151-239`), `completion_token_output` (`:264`), every per-API result serializer (`to_json_oaicompat*`, `to_json_anthropic*`, `:320-1464`), the Prometheus text behind `/metrics` (`server_task_result_metrics::to_metrics`, `:1523`) and `server_prompt_cache` (`:1691-1872`). Scheduling is `server_queue` (`server-queue.cpp`) driven by `process_single_task` (`server-context.cpp:2289`).

Ordering: `queue_tasks` is FIFO, but `post(tasks, front=true)` inserts at the front and the cancel batch uses it (`server-queue.cpp:616-617`). Anything that finds no free slot is `defer`red into `queue_tasks_deferred` (`:76-79`) and comes back at the *front* of the main queue when a slot frees — `pop_deferred_task(id_slot)` prefers a deferred task already bound to that slot, otherwise takes the oldest (`:90-110`). So there are two effective priority classes (front-of-queue cancels; deferred-but-remembered requests) and **no numeric priority field** anywhere.

Cancellation is itself a task, not a flag: `SERVER_TASK_TYPE_CANCEL` with `id_target` strips the target from whichever list still holds it (`server-queue.cpp:30-34`, and the `remove_if` sweep over both `queue_tasks` and `queue_tasks_deferred` at `:375-382`). A client that disconnects or stops mid-stream makes `server_response_reader::stop()` post one cancel per outstanding task id at the front, with the comment "push to beginning of the queue, so it has highest priority" (`:603-621`).

While a decode is in flight the step loop yields: `process_single_task` declines every type except `METRICS` and `SLOT_GET` (`server-context.cpp:2291-2294`), so `/metrics` and `/slots` stay answerable during generation. The declined tasks are picked up by `server_queue::process_new_tasks(is_yielding)` (`server-queue.cpp:138`) from the worker loop (`:163`) — that is the mechanism behind the "retried after the decode" note in the main loop, and it is why a deferred request never loses its slot binding.

### Streaming — what happens before the piece becomes the response body

Producer: once a token is accepted, `send_partial_response(slot, result, false)` (`server-context.cpp:1806-1807`) builds a `server_task_result_cmpl_partial` whose `content` is `tkn.text_to_send` and whose `tokens` is the single accepted token id, then posts it to `queue_results` (`:1975-2016`). The text piece itself is produced by the token→piece step of the sampling/emit path — [[tokenizer]] owns what that string is; this layer only carries it.

Not every partial is a token: progress-only updates carry `is_progress` plus `progress{total,cache,processed,time_ms}` (`:1982-1988`, sent at `:3334` and `:3722-3723`), and the `is_begin` marker carries no content at all — `server_task_result_cmpl_partial::to_json` returns `nullptr` for it (`server-task.cpp:1026-1029`), which is the signal for the HTTP layer to write status 200 and the headers before the first token exists (`server-context.cpp:3336-3338`).

Consumer: the HTTP thread's `process_handler_response` checks `response->is_stream()` and chunk-writes the serialized result; the per-API serialization is in `server-task.cpp` (`to_json_oaicompat_chat_stream` `:462`, `to_json_oaicompat_resp_stream` `:599`, `to_json_anthropic_stream` `:797`).

Buffering and backpressure: `queue_results` is a plain `std::vector<server_task_result_ptr>` with no cap (`server-queue.h:161-162`), so the engine→HTTP hand-off never throttles generation. The backpressure point is the socket write inside the chunked-content callback: a client that stops reading blocks that HTTP thread while the step loop keeps generating, and results pile up in the queue. The reverse signal is the stop closure — `server_response_reader::next` polls the queue with a timeout and calls `should_stop` each round (`server-queue.cpp:550-563`), which is how a gone client turns into the `CANCEL` batch above.

A second, opt-in buffer exists for resumable streams: `/v1/stream` (`server-stream.cpp`). A request carrying `X-Conversation-Id` creates or replaces a `stream_session` (`:597-605`) — a capped ring buffer (`STREAM_SESSION_MAX_BYTES`) that drops from the front and tracks `prefix_dropped`, so a reader resuming behind the drop point gets `OFFSET_LOST` (`:163-172`). Sessions are GC'd on a TTL by a background thread (`:317`), `GET /v1/stream?conv_id&from=N` replays then blocks for live SSE bytes (`:454-501`), and `DELETE /v1/stream` is the explicit Stop: it cancels the producer and finalizes the session but by design does *not* interrupt the underlying generation (`:309-314`, `:561-575`). Cancelling a session also makes `server_res_spipe::conn_alive()` false (`:623-625`), so the resume path doubles as the disconnect signal for long generations.

### Tool calling — parsing and execution sit on opposite sides of the HTTP boundary

The server never executes a tool call the model emits. Generation produces text and a stop; the chat parser selected via `parse_tool_calls`/`chat_parser` (`server-schema.cpp:314-318`; the schema evaluator is `eval_llama_cmpl_schema`, `:515-548`) turns that text into `common_chat_msg` diffs inside `task_result_state::update_chat_msg` (`server-task.cpp:162-239`), and the OpenAI delta carries the fragments (`server-chat.cpp:615-632`). What the template and parser produce, and how a tool result is rendered back into the prompt, is [[chat-templates]]' subject and is not duplicated here.

The result re-enters the prompt only as a new request: the client appends a `role:"tool"` / `tool_call_id` message (`server-chat.cpp:196-215` for the Responses conversion, `:439-505` for the Anthropic one) and re-POSTs, so the tool output becomes just another message the chat template expands. The built-in tools the server ships (`read_file`, `file_glob_search`, `grep_search`, `exec_shell_command`, `write_file`, `edit_file`, `get_info`; built by `build_tools`, `server-tools.cpp:1951`, classes at `:869-1756`) are exposed exactly the same way: `GET/POST /tools`, wired in `server_tools::setup` (`:1997`) and registered in `server.cpp:340-361`. The POST body is `{"tool","params","stream"}`; the reply is `plain_text_response` (`server-tools.cpp:963`, `:1082`) that the caller pastes back as the tool message. `--server-tools` selects the set (name-validated, `"all"` accepted), the per-call headers `x-tool-cwd` and `x-tool-runtime` override working directory and I/O backend, and `x-resp-type` picks read_file's response shape (`:1997-2130`). The I/O backends are `tools_io_basic`, `tools_io_isolate` (POSIX `sh`), `tools_io_container` (docker/podman, spawn or attach) and `tools_io_ssh` (`:298-784`), chosen by the `--tools-runtime` spec and validated once at startup (`make_tools_runtime`, `:1988-1995`). Streaming tool results push `server_tool_stream_result` chunks through their own result queue (`:1758-1783`), so a long shell command can stream back to the UI. The flag surface itself is catalogued in [[runtime-switches]].

### MCP — configured child processes, not a protocol switch

`--mcp-servers-config` / `--mcp-servers-json` (env `LLAMA_ARG_MCP_SERVERS_CONFIG` / `_JSON`, `src/common/arg.cpp:3430-3441`) fill `params.mcp_servers_config` (a file path) and `params.mcp_servers_json` (inline text), parsed as Cursor-format `{"mcpServers": {...}}` JSON (`server_mcp_server_config::parse_cursor_format`, `server-mcp.cpp:139-168`; duplicate names across the two sources are rejected, `:700-706`).

What the server does at startup (`server.cpp:334`, `server_mcp::start` `server-mcp.cpp:693-748`): for each config it **spawns the configured command as a child process** over stdio pipes, sends the MCP `initialize` handshake (protocol `"2024-11-05"`, `clientInfo` `llama.cpp`/`1.0`, `:179`, `:254-270`), and performs `tools/list` under a 10 s per-server warmup deadline (`MCP_WARMUP_TIMEOUT_SECONDS`, `:657-658`, `:737-744`) to build the registry. Each discovered tool becomes a `server_mcp_tool` in the same `/tools` list as the built-ins, namespaced `<server>_<tool>` and skipped on a name collision (`server-tools.cpp:1997-2100`). A call routes through `server_mcp::call_tool` → JSON-RPC `tools/call`; the result's text parts are concatenated into `plain_text_response`, or `error` when the server sets `isError` (`server-mcp.cpp:196-212`, `:315-337`). A server that has died is re-spawned lazily on next use, with a 5 s cooldown (`MCP_COOLDOWN_SECONDS`, `:657`, `:800-810`).

**What an MCP server gets**: the child is built from `command` + `args`, with the *parent's* environment plus per-server `env` overrides (`mcp_build_env`, `server-mcp.cpp:447-483`) and the config's `cwd`, launched through the platform subprocess layer (`:462-491`). There is no sandbox, no tool allow-list, and no capability exchange beyond the empty `capabilities` object in `initialize`. An MCP server can do everything the server's own OS user can — the same reach as `exec_shell_command` with the default I/O backend. It is arbitrary code execution granted by configuration, owned by whoever can pass the flag. The server does drain the child's stderr and keep a bounded tail (`ERR_TAIL_MAX = 4096`, `:618-626`), and it ignores `SIGPIPE` process-wide because an MCP server or tools runtime can exit while the parent writes its stdin (`server.cpp:92-93`).

**Default: off.** `server_mcp::empty()` is `configs.empty()` (`server-mcp.h:143`), `params.mcp_servers_config`/`_json` default to empty (`src/common/common.h:667-669`), and both `/tools` routes are registered only when `params.server_tools` or the MCP registry is non-empty (`server.cpp:340-361`); otherwise `/tools` is a plain 403. `--agent` switches on all server tools plus the UI proxy (`arg.cpp:3448-3455`). Do not confuse the MCP connectors with `--ui-mcp-proxy` (`LLAMA_ARG_UI_MCP_PROXY`, `arg.cpp:3401-3404`) — that is the web UI's `/cors-proxy` endpoint (`server.cpp:324-327`), reported as `cors_proxy_enabled` in `/props` (`server-context.cpp:4537`). One shared side effect: enabling server tools *or* MCP rewrites the default CORS origin to `localhost` and warns (`arg.cpp:955-958`).

### The 116 ms `cudaMemGetInfo`

[[first-live-measurements]] records `cudaMemGetInfo` at 15 % of API time — **116 ms in 13 calls, ≈8.9 ms each** — and asks whether it is a per-request VRAM probe. What the code says:

- `src/common/fit.cpp` never calls `cudaMemGetInfo`; it asks the backend for free memory through `ggml_backend_dev_memory` (`src/common/fit.cpp:106`, `:115`). On CUDA that resolves to `ggml_backend_cuda_device_get_memory` → `cudaMemGetInfo` (`src/ggml/src/ggml-cuda/ggml-cuda.cu:5063`; a second call site in `ggml_backend_cuda_get_device_memory`, `:4920`). `fit.cpp` reaches it through `common_get_device_memory_data_impl`, which the parametrised fit and `common_memory_breakdown_print` share.
- It is called **per measured configuration**, not per request: `common_fit_params` re-measures after every candidate change — `fit.cpp:263` (initial), `:275` (context bumped), `:414` (min-ctx reduction), `:595` (`get_memory_for_layers`, once per layer-split candidate) — and each measurement constructs and then frees a model + context (`:75`, `:146-149`). The draft/extra model is measured again whenever its context changes (`:222-228`).
- The server has no VRAM probe in its request path. The only memory call in `server-context.cpp`/`server.cpp` is `common_memory_breakdown_print` at startup (`server.cpp:544-545`), which itself queries each device once (`src/common/common.cpp:413-415`).

[INFERENCE] `--fit` is **on by default** (`bool fit_params = true`, `src/common/common.h:469`) and the fit search measures once per candidate layer split / context size, so the 13 slow calls are startup fitting plus the breakdown print — made slow (~8.9 ms each) because the query runs right after a CUDA model/context was built or torn down and the device still has work outstanding. The two reads that force this: `src/common/common.cpp:1295-1325` (fit runs whenever `params.fit_params`) and `fit.cpp:587-595` (one measurement per candidate). It is not a per-request probe — no request-path call site exists — but splitting 116 ms into fit vs. breakdown vs. CUDA-context churn needs a `--fit off` timeline, which is a measurement, not a reading. That converts the open question in [[first-live-measurements]] from "per-request probe or startup-only?" to "how much of the 116 ms does `-fit off` remove?".

## See also

[[request-lifecycle]] · [[sampling]] · [[speculative-decoding]] · [[performance-profile]] · [[benchmarks]] · [[v100-sxm2]] · [[cuda-graphs]] · [[overview]]
