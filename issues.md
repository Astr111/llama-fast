# issues.md — consolidated defect register

Every defect, discrepancy and infrastructure gap found in the `llama-fast` repository during the
documentation campaign of 2026-09-28/29, in one place, with the evidence and what would close each.

**Detail lives in `llmwiki/`.** Each row's ID links to a page holding the full analysis. This file is
the register — for reading, for triage, and for deciding what to do first.

**Vocabulary**

| Severity | Meaning |
| :--- | :--- |
| `CRITICAL` | Corrupts results or output in the shipped configuration |
| `HIGH` | Wrong behaviour in a reachable configuration |
| `MEDIUM` | Wrong in a narrow case, or a latent hazard |
| `LOW` | Hygiene, diagnostics, or a deferred risk |
| `INFRA` | Not a code defect: the repository or the environment cannot do what it appears to |

| Status | Meaning |
| :--- | :--- |
| `OPEN` | Present in this checkout |
| `OBSERVED` | Confirmed by running the engine, not only by reading |
| `RE-SCOPED` | Real, but the recorded cause or impact was wrong |
| `REFUTED` | The recorded claim does not hold in this checkout |
| `BLOCKED` | Cannot be fixed or measured here |

---

## 1. Priority — what to do first

| # | Item | Sev | Status | Cost to fix |
| --: | :--- | :--- | :--- | :--- |
| **A1** | **TA-8** — `offset_max = 0` makes every eviction score NaN, in the shipped scripts | CRITICAL | OBSERVED | two lines (guard + a working default) |
| **A2** | **TA-11** — the shipped calibration profile is in the wrong basis | CRITICAL | OPEN | move the capture hook; regenerate the profile |
| **A3** | **TA-9** — the scorer inverts RoPE over 256 dims with the wrong exponent | HIGH | OPEN | parameterise the inverse by `n_rot` + sections |
| **A4** | **TA-10** — `prefix_length` latches per context and the server never resets it | HIGH | OBSERVED | make the boundary per-sequence |
| **A5** | **TA-1** — WHT inversion skipped at `head_dim=256`; **the fix exists in a sibling checkout that is slated for deletion** | CRITICAL | BLOCKED | port three lines (quoted in the issue page) before that checkout goes |
| **B1** | **INFRA-1** — `template-instances/` empty → no CUDA build from this tree | INFRA | OPEN | one `unzip` from `llama-fast-src.zip` |
| **B2** | **INFRA-2** — `ggml-cpu/arch/` empty → x86 SIMD sources absent | INFRA | OPEN | source restoration (not in the archive — see the page) |
| **B3** | **INFRA-3** — no CUDA toolkit on this machine; attached GPU is `sm_75`, not the target `sm_70` | INFRA | BLOCKED | different machine |
| **B4** | **INFRA-4** — no quality metric exists anywhere in the project | INFRA | OPEN | install a harness (docker is present) |
| **C1** | **HAZ-1** — `build_lora_mm()` multiplies the LoRA branch in the unrotated basis while the base product uses the rotated one | HIGH | OPEN | fold the adapter, or rotate its input |

**A1–A4 share one property: each independently corrupts *which keys survive eviction*.** Until all four
are fixed, no measurement of eviction quality means anything — the four effects are additive and
inseparable. A1 is also the gate: with every score NaN, the others cannot even be observed.

---

## 2. The pre-existing inventory (state.md §3, §4)

### TriAttention

| ID | Sev | Status | One line | Detail |
| :--- | :--- | :--- | :--- | :--- |
| TA-1 | CRITICAL | BLOCKED | The scoring kernel skips WHT inversion when `padded_hd == 256`; triggered by the target model's `head_dim = 256` | [[ta-1-wht-inversion-256]] |
| TA-2 | HIGH | OBSERVED | Long prefixes starve the budget: protection consumes `prefix + window` and eviction becomes a sliding window | [[ta-2-budget-starvation]] |
| TA-3 | HIGH | OPEN | CPU fallback does one synchronous D2H copy **per KV cell** — 15–30 s stalls | [[ta-3-cpu-fallback-transfers]] |
| TA-4 | MEDIUM | OPEN | `cooperative_fwht_128` assumes 64 active threads; UB on Volta | [[ta-4-cooperative-fwht-race]] |
| TA-5 | MEDIUM | RE-SCOPED | `freq_scale_sq` is an identity multiply — **not a bug**: the doc's `1/ω²` weighting exists nowhere in `src/` | [[ta-5-freq-scale-dead-code]] |
| TA-6 | LOW | OPEN | Overlap double-counting when `recent` and `prefix` ranges intersect | [[ta-6-overlap-double-counting]] |
| TA-7 | LOW | OPEN | No validation of incompatible `budget`/`prefix`/`window` combinations | [[ta-7-config-validation]] |

### TurboQuant

| ID | Sev | Status | One line | Detail |
| :--- | :--- | :--- | :--- | :--- |
| TQ-1 | CRITICAL | **RE-SCOPED** | The types *are* missing from every dispatch table, but **no turbo `MUL_MAT` is ever built** — fused attention consumes the cache — so the recorded 38.81 % fallback is not this mechanism, and `MAGMA` appears nowhere in the tree | [[tq-1-missing-gemm-kernels]], [[device-placement]] |
| TQ-2 | HIGH | OPEN | Host state is file-scope `static` in a header — still present, and now **duplicated** by a newer module (`turbo-innerq.{cu,cuh}`) | [[tq-2-innerq-host-state]] |
| TQ-3 | HIGH | OPEN | `static __device__` state receives `cudaMemcpyToSymbol` on one device only | [[tq-3-innerq-multigpu]] |
| TQ-4 | HIGH | **REFUTED** | Three WHT copies exist; the two in question agree by construction and `turbo_rotate_forward{,_64}` have no callers | [[tq-4-wht-numerical-mismatch]] |
| TQ-5 | MEDIUM | **REFUTED (unreachable)** | The tail path cannot run: `ggml_turbo_wht` asserts `ne[0] % group_size == 0` | [[tq-5-tail-elements]] |
| TQ-6 | MEDIUM | OPEN | Calibration counter keyed on thread mapping rather than `threadIdx.x == 0` | [[tq-6-innerq-race]] |
| TQ-7 | MEDIUM | OPEN | `INNERQ_MAX_CHANNELS = 128` against a 256-channel head — at most half equalized | [[tq-7-innerq-max-channels]] |

---

## 3. Defects found during this campaign — in no inventory

| ID | Sev | Status | One line | Detail |
| :--- | :--- | :--- | :--- | :--- |
| **TA-8** | CRITICAL | **OBSERVED** | `offset_max` defaults to 0 → `n_offsets = 0` → the mean aggregate is `0 × (1/0)` → **NaN for every key**, and NaN reaches `std::partial_sort`'s comparator (UB). Neither launch script passes the flag. Live log: `offsets=0` | [[ta-8-offset-max-zero-nan]] |
| **TA-9** | HIGH | OPEN | The scorer builds `omega` from `head_dim` and inverts every pair `(f, f+128)` across all 256 dimensions, while the model rotates `n_rot = 64` with exponent θ^(−2f/**64**). The angle error is θ^(3f/128) — **it grows with frequency**. Not compensated anywhere | [[ta-9-rope-scope-mismatch]] |
| **TA-10** | HIGH | **OBSERVED** | `prefix_length` is one latch per KV cache, set by the first batch with position 0, reset only by `clear()` — and the server never full-clears (slot recycle is `seq_rm`). A later longer prompt loses its own middle to eviction. Live log: `prefix=27` at every position | [[ta-10-prefix-length-global-latch]] |
| **TA-11** | CRITICAL | OPEN | The calibrator matches the tensor name `Qcur-<layer>`, which in the `qwen35` graph is the **post-RoPE** output, while the scorer deliberately scores **pre-RoPE** keys. The phase term is position-corrupted and the norm understated | [[ta-11-calibration-post-rope-basis]] |
| **HAZ-1** | HIGH | OPEN | `build_lora_mm()` computes the base product against the rotated activation `cur_mm` while the LoRA branch multiplies the unrotated `cur` — the wrong basis for a Hadamard-folded weight. `[INFERENCE]`, live only if an adapter was not folded | [[loading-and-batching]] |
| **HAZ-2** | MEDIUM | OPEN | The `[UNK_BYTE_0x…]` detokenisation fallback appends the **whole piece** instead of the offending codepoint, emitting it *k* times in two encodings. **Not reachable for the target model** (`qwen35` keeps `byte_encode = true`); live for `LLAMA_VOCAB_PRE_TYPE_WHITESPACE` models, and **silent** — it corrupts the load-time token cache | [[tokenizer]] |
| **HAZ-3** | MEDIUM | OPEN | The calibrator stamps a **fixed model identity** into every profile it writes: the 4B calibration carries the 27B's name and a `rope_theta` the model does not use (10 000 000 vs 5 000 000). The runtime warns about the rope base and accepts the name silently, so a `.triattention` file cannot be audited from its contents | [[first-live-eviction]] |
| **HAZ-4** | MEDIUM | OPEN | TriAttention **cannot be installed on the DSV4 cache path at all**: `llama_triattention_init` requires a `llama_kv_cache` or `llama_memory_hybrid` target, and a DSV4 context is neither — the call logs *memory is not a KV cache* and returns −1. A DeepSeek-V4 deployment gets no eviction, silently | [[kv-cache-dsv4]] |
| **HAZ-5** | LOW | OPEN | `is_uppercase`, `is_lowercase`, `is_nfd` in `unicode.cpp` have no reader anywhere; `turbo-rotation-data-32.h` (36 KB) is included by nothing; `LLAMA_ARG_DRAFT_MAX`/`_MIN` survive for a flag that was removed | [[rotation-data]], [[runtime-switches]] |

---

## 4. Repository and environment deficiencies

| ID | Sev | Status | One line | Detail |
| :--- | :--- | :--- | :--- | :--- |
| **INFRA-1** | INFRA | OPEN | `src/ggml/src/ggml-cuda/template-instances/` is **empty** while `CMakeLists.txt:106-116` globs it; 138 `.cu` files exist only in `llama-fast-src.zip`. A CUDA build from this tree cannot instantiate its kernels | [[build-and-verify]], [[codebase-map]] |
| **INFRA-2** | INFRA | OPEN | `src/ggml/src/ggml-cpu/arch/` is **empty** while `CMakeLists.txt:243-244` lists `arch/x86/quants.c` and `arch/x86/repack.cpp` for x86. A **second** empty-but-referenced directory — this is the shape of how the repo was published, not a one-off | [[build-and-verify]], [[cpu-path]] |
| **INFRA-3** | INFRA | BLOCKED | No CUDA toolkit on this machine (`nvcc`, `cuobjdump`, `ptxas` all absent) and the attached GPU is a **GTX 1660, `sm_75`** — not the V100 `sm_70` the project targets. The tree's only configure record is CPU-only | [[build-and-verify]], [[v100-sxm2]] |
| **INFRA-4** | INFRA | OPEN | **No quality metric exists in the project at all** — while `tools/perplexity`, `tools/imatrix` and `tools/tuning` sit in the tree and docker runs. Four confirmed quality-affecting defects have no instrument behind them | [[documentation-coverage]], [[terminal-bench-subset]] |
| **INFRA-5** | INFRA | OPEN | `build/cuda124.zip`'s `libggml-cuda.so` shows **only `sm_86` markers** while its own README advertises `sm_61;sm_70;sm_75;sm_80;sm_86`. The bundle labelled for Volta is the one that cannot be confirmed for Volta | [[v100-sxm2]], [[release-artifacts]] |
| **INFRA-6** | INFRA | OPEN | `build/cuda13.zip` and `build/cuda124.zip` are **LZMA-compressed**; `unzip` extracts nothing from them and **reports success**. `llama-bench` is absent from both bundles | [[release-artifacts]], [[llama-bench]] |
| **INFRA-7** | INFRA | BLOCKED | The `Release/` layout the README documents does not exist at the path `state.md` names; a sibling checkout holding the TA-1 fix exists elsewhere and is **reported as slated for deletion** | [[ta-1-wht-inversion-256]], [[release-artifacts]] |

---

## 5. Claims that did not survive verification

Not defects — places where the project's own records are wrong. Each is annotated in place rather than deleted.

| Claim | Where it came from | What the code shows |
| :--- | :--- | :--- |
| `magma_sgemmEx_kernel<float, __nv_bfloat16>` costs 38.81 % of GPU time, caused by missing turbo GEMM kernels | `state.md` §1.3, §4 TQ-1 | No `MAGMA` anywhere in `src/`; no turbo `MUL_MAT` is built during decode; no cuBLAS/MAGMA kernel in a live trace. **The 38.81 % is unattributed** |
| Benchmarks describe the target hardware | `README.md` | Measured on an **RTX 3090**; the target is a V100. The published tokens-per-GB column matches a **dense 64-layer** model, not this model's 16 KV-bearing layers |
| "max 5 draft tokens" | `state.md` | In no code path: default 3, the DFLASH clamp is 7, the doc example is 15, and `--draft-max` was removed |
| `block_size = 32` for turbo types | `state.md`, stale comments | `QK_TURBO3 = 128` |
| TQ-4: the two WHT implementations diverge numerically | `state.md` §4 | They agree by construction; three copies exist and the fourth candidate is unreachable |
| The design doc's CLI defaults | `src/docs/TRIATTENTION.md` | Disagree with both the code and the README, which agree with each other |
| The paper's date and authors | `src/docs/TRIATTENTION.md` vs `README.md` | Same arXiv id, **2025 with one author list vs April 2026 with another** |

---

## 6. What would close each class

| Class | Blocked by | Unblocks |
| :--- | :--- | :--- |
| A1–A4 (the scoring defects) | nothing but the decision to change code | trustworthy eviction |
| A5 (TA-1) | the sibling checkout being deleted | the `head_dim=256` path |
| B1, B2 (missing sources) | a restore + a decision about publication hygiene | any build at all |
| B3 (toolkit / GPU) | different hardware | V100 numbers |
| B4 (no quality metric) | a harness install + a scaffold | answering "did it get worse?" |
| §5 (wrong records) | already done in the wiki | not starting work from a false premise |

---

## 7. Provenance

- **This register**: written 2026-09-29 from the vault at `llmwiki/` (83 pages, lint clean).
- **Per-item detail**: the linked `llmwiki/issues/*.md` and `llmwiki/topics/*.md` pages, each carrying
  `verified:` frontmatter listing the files whose claims were checked against the code.
- **Live observations**: `llmwiki/topics/first-live-measurements.md` (the engine on a GTX 1660) and
  `llmwiki/topics/first-live-eviction.md` (the pruner, with log excerpts).
- **Not covered here**: upstream llama.cpp defects, other backends, and anything requiring the V100 —
  see `llmwiki/topics/documentation-coverage.md` for what is out of scope and why.
