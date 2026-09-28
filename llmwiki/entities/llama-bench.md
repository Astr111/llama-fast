---
title: llama-bench
type: entity
status: current
updated: 2026-09-29
sources: []
verified: [src/tools/llama-bench/llama-bench.cpp, src/tools/llama-bench/CMakeLists.txt, src/tools/CMakeLists.txt, src/CMakeLists.txt, src/ggml/include/ggml.h, src/ggml/src/ggml.c, src/src/llama-context.cpp, src/common/sampling.cpp, build/cuda13.zip, build/cuda124.zip]
tags: [benchmarks, tooling, cuda]
---

# llama-bench

## What it is

The upstream llama.cpp throughput harness, present and wired into **this** tree: `src/tools/llama-bench/llama-bench.cpp` is 2 484 lines, split into a library `llama-bench-impl` (`src/tools/llama-bench/CMakeLists.txt:5`) plus a 20-line launcher `main.cpp` (`:19`). It is added unconditionally by `src/tools/CMakeLists.txt:22` — outside the `LLAMA_BUILD_SERVER` guard that protects `cli`/`server` — under the `LLAMA_BUILD_TOOLS` option (`src/CMakeLists.txt:133`, defaulting to `LLAMA_STANDALONE`).

This page fills gap #11 of [[documentation-coverage]]: until now no page in the vault mentioned the file.

The reason it matters here is narrow and specific. [[benchmarks]] records that every published number is Ampere-era and that no V100 measurement exists; [[roadmap]] keeps needing measurements. This is a harness built for exactly that, sitting unbuilt and unused. See *Availability* below for why it is nonetheless not runnable today.

## How it works

**Metric set.** A `test` record holds `std::vector<uint64_t> samples_ns` and derives everything from it (`llama-bench.cpp:1470-1523`):

- `avg_ns() = ::avg(samples_ns)`, `stdev_ns() = ::stdev(samples_ns)` (`:1521-1523`);
- `get_ts()` converts each sample to tokens/s as `1e9 * n_tokens / t`, where `n_tokens = n_prompt + n_gen` (`:1526-1534`), and `avg_ts()`/`stdev_ts()` average those (`:1537-1539`);
- the reported cell is `"%.2f ± %.2f"` of `avg_ts` and `stdev_ts` (`:2053`).

> **Correction (2026-09-29):** the natural assumption that llama-bench reports *medians* is wrong. It reports **mean ± standard deviation** over the `-r` repetitions (default `reps = 5`, `:400`). There is no median anywhere in the file. Mean is more outlier-sensitive than median, which is worth knowing when comparing against the hand-collected repeats in [[first-live-measurements]].

**Test naming and matrix.** Rows are named `pp<n>` for prompt-only, `tg<n>` for generation-only, `pp<n>+tg<n>` for a combined run, each suffixed ` @ d<n>` when a KV depth is set (`:2032-2044`). Every axis `cmd_params` accepts is a vector (`:327-357`), and `get_cmd_params_instances()` emits the full nested cross product over: model, fit targets, `n_gpu_layers`, `n_cpu_moe`, `split_mode`, `load_mode`, `main_gpu`, `devices`, `tensor_split`, `tensor_buft_overrides`, `no_host`, `embeddings`, `no_op_offload`, `n_batch`, `n_ubatch`, **`type_k`**, **`type_v`**, `no_kv_offload`, `flash_attn`, `n_threads`, `cpu_mask`, `cpu_strict`, `n_depth`, `poll` (`:1302-1330`). The loop order is deliberate — the comment at `:1300` says it "minimizes the number of times that each model needs to be reloaded". Each combination produces up to three instances: prompt-only, generation-only, and the `-pg pp,tg` pairs (`:1326-1420`).

**What it holds fixed.** The token stream is synthetic: `test_prompt` decodes a fixed `tokens` vector in `n_batch` chunks (`:2114-2131`) and `test_gen` decodes a single, unchanging `token` id, `n_gen` times (`:2143-2153`). Nothing is sampled and no model output is inspected — the harness measures time only. The `-d` depth is handled by first evaluating `n_depth` tokens to fill the KV cache, and the filled state is reused when consecutive tests share a depth (`cstate.depth`, `:2384-2410`); `n_ctx` is sized `n_prompt + n_gen + n_depth` so the depth fits in context (`:1281`). A warmup run precedes the measured repetitions unless `--no-warmup` is given (`:2354-2367`, `:2381-2441`).

**What it reports beyond the number.** Each row carries environment metadata: `test_time` in RFC 3339 (`:1512-1516`), model description/size/parameter count via `llama_model_desc`/`llama_model_size`/`llama_model_n_params` (`:1485-1490`), the backend list from `ggml_backend_reg_*` (`:1526-1545`), cpu/gpu info, the full parameter set including `type_k`/`type_v`, and `avg_ns`/`stddev_ns` alongside `avg_ts`/`stddev_ts` (field list at `:1564-1590`). Output is selectable as `csv | json | jsonl | md | sql` (`-o`, `:419`), markdown being the default (`:404`).

## How to drive it

**Flag surface** (`print_usage`, `:410-470`) has three groups:

| Group | Flags |
| :--- | :--- |
| Harness control | `-r/--repetitions`, `--prio`, `--delay`, `-o/--output`, `-oe/--output-err`, `--list-devices`, `-v`, `--progress`, `--no-warmup`, `--numa`, `-fitt/--fit-target`, `-fitc/--fit-ctx` |
| Test parameters | `-m/--model`, `-hf*`, `-p/--n-prompt`, `-n/--n-gen`, `-pg <pp,tg>`, `-d/--n-depth`, `-b/--batch-size`, `-ub/--ubatch-size`, **`-ctk/--cache-type-k`**, **`-ctv/--cache-type-v`**, `-t/--threads`, `-C/--cpu-mask`, `--cpu-strict`, `--poll`, `-ngl`, `-ncmoe`, `-sm`, `-mg`, `-nkvo`, `-fa`, `-dev`, `-lm/--load-mode`, `-embd`, `-ts`, `-ot`, `-nopo`, `--no-host` |

Every test parameter is a comma-separated list, i.e. an axis of the sweep.

**The invocation this project actually needs** — a like-for-like `f16` versus `turbo3`+`q8_0` KV comparison at fixed context:

```bash
llama-bench -m models/qwen35-4b-Q4_K_M.gguf \
  -p 512 -n 128 -d 0 -ngl 99 -fa on -r 5 \
  -ctk f16,turbo3 -ctv f16,q8_0 \
  -o md
```

`type_k` and `type_v` are independent axes of the cross product (`:1317-1318`), so this pair of lists yields four rows — `f16/f16` (baseline), `f16/q8_0`, `turbo3/f16` and `turbo3/q8_0` — in one table, on one machine, back to back. If only the two configurations that matter are wanted, run it twice (`-ctk f16 -ctv f16`, then `-ctk turbo3 -ctv q8_0`); the four-row form is strictly more informative because it isolates the K-side and V-side contributions. To probe depth rather than context length, add `-d 2048 -d 8192` and the rows become `tg128 @ d2048`, `tg128 @ d8192`. `--list-devices` prints what `-dev` will accept before any test runs (`:672-674`).

**What this reports that the `llama-cli` timings do not.** `llama-cli` ends a run with `llama_perf_context_print` — a single measured pass printing `prompt eval time` and `eval time` in ms, ms/token and tokens/s (`src/src/llama-context.cpp:4572-4574`; the common layer prints the same shape at `src/common/sampling.cpp:574-576`). [[first-live-measurements]] therefore had to *script repetitions by hand* and compare runs manually to get 85.6 → 76.0 t/s. llama-bench gives, in one process: warmup plus `-r` measured repetitions with mean and standard deviation; the `type_k`/`type_v` pair as an explicit output column, so the comparison is a row in a table rather than two invocations; KV-depth rows (`-d`) that `llama-cli` cannot express without filling the cache manually; device/backend selection (`-dev`, `-sm`, `-ts`, `-ot`) instead of ambient defaults; and machine-readable `csv|json|sql` output with the environment recorded alongside the number.

## Where it lives

- `src/tools/llama-bench/llama-bench.cpp` — all logic (2 484 lines)
- `src/tools/llama-bench/main.cpp` — launcher
- `src/tools/llama-bench/CMakeLists.txt` — `llama-bench-impl` library + `llama-bench` executable, linking `llama-common` and `llama` (`:9`, `:20`)
- `src/tools/CMakeLists.txt:22` — the `add_subdirectory(llama-bench)` that puts it in this tree's CMake build list

**Can it select the fork's types? Yes — both legs, with no whitelist.**

1. `-ctk`/`-ctv` are split on commas and each token resolved through `ggml_type_from_name`; an unknown name sets `invalid_param` and the run aborts (`:614-653`). The resulting `std::vector<ggml_type>` is copied verbatim into `llama_context_params.type_k` / `.type_v` (`:1284-1285`).
2. `ggml_type_from_name` resolves against the type-traits table, and the fork registers the TurboQuant names there: `"turbo3"` → `GGML_TYPE_TURBO3_0` (`src/ggml/src/ggml.c:708-710`), `"turbo4"` (`:716`), `"turbo2"` (`:724`); the enum is at `src/ggml/include/ggml.h:433-435`.
3. The harness links the same `llama` target as every other tool (`CMakeLists.txt:9`), so it exercises the fork's KV-cache path.

So the harness is **not** the blocker for the project's central question — `-ctk turbo3 -ctv q8_0` is expressible and would land in `cparams` exactly as it does for `llama-cli`. `[INFERENCE]` from the three reads above; no run was made (a GPU job was in progress).

## Availability — the real blocker is packaging

The harness logic exists in one shipped bundle and the executable in neither:

| Artifact | `libllama-bench-impl.so` | `llama-bench` executable |
| :--- | :--- | :--- |
| `build/cuda124.zip` (`cuda12.4/bin/`) | **present**, 477 368 bytes | absent |
| `build/cuda13.zip` (`cuda13/bin/`) | absent | absent |

`cuda13/bin/` ships exactly `llama-cli`, `llama-server`, `llama-triattention-calibrate`, `test-pq2-row-shapes`, `test-ptq1_0-cuda-dot` and the `.so` set; `cuda12.4/bin/` adds `libllama-batched-bench-impl.so` and the other `-impl.so` libraries but likewise ships **no `llama-bench`** and no `llama-batched-bench` executable. `find build -name "*llama-bench*"` returns nothing — there is no build tree under `build/` at all, only the two zips. The extracted deployment copy at `/hdd2/tools/bonsai/cuda13/bin` has no bench binary either.

Consequence: **the exact command to run it from the extracted bundle does not exist today.** The hypothetical form is `cuda13/bin/llama-bench -m model.gguf -ctk f16,turbo3 -ctv f16,q8_0 -p 512 -n 128` from the bundle root, but neither bundle contains that launcher, and `libllama-bench-impl.so` is an implementation library, not a runnable entry point. Getting the tool requires a build (`cmake --build <build-dir> --target llama-bench` — `[UNVERIFIED]`: no build tree exists here to confirm the target name or configure flags) or an added file to the release whitelist. The parent's premise that llama-bench is "already compiled into this tree's build list" is **half true and the useful half is false**: it is in the CMake build *list* (`src/tools/CMakeLists.txt:22`), but it is not in any build *output* that exists on this machine.

## What this changes for [[roadmap]]

The measurement gap is **tooling-present, hardware-absent** — with one packaging caveat, not a tooling-absent problem. The harness that answers "f16 versus turbo3 KV, at depth, with error bars" is in the tree, in the build list, and already type-aware; nothing has to be written to make the V100 comparison that [[benchmarks]] and [[roadmap]] keep waiting for. What is missing is two separate things, and only the second is the one the vault has been assuming: (a) the release bundles neither build nor ship the `llama-bench` executable, so it is one build step away rather than zero steps away — cheap, but real, and it is the reason the harness has gone unmentioned; and (b) a V100 to run it on ([[v100-sxm2]]). The path therefore, once hardware exists, is: add `llama-bench` to the artifact list, run the four-row `-ctk/-ctv` table plus the `-d` depth sweep, and [[kv-accounting]]'s MiB-vs-throughput trade gets a measured curve on the target rather than Ampere extrapolation. Until then it is worth recording that the [[first-live-measurements]] numbers on the GTX 1660 were produced with `llama-cli` and hand-scripted repeats — reproducible, but not what the in-tree tool would have reported.

## Known issues

- No vault-tracked defect. The packaging gap above (harness in tree, absent from both bundles) is a release-artifacts observation, not a code defect — see [[release-artifacts]].
- `-ctk`/`-ctv` accept any name `ggml_type_from_name` resolves, including types the KV cache cannot actually use; rejection then happens deeper in the model/KV path rather than at argument parsing (`[UNVERIFIED]` — not traced here).

## See also

[[benchmarks]] · [[performance-profile]] · [[first-live-measurements]] · [[roadmap]] · [[v100-sxm2]] · [[kv-accounting]] · [[turboquant]] · [[kv-cache]] · [[release-artifacts]] · [[documentation-coverage]]
