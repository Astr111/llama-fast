# Llama-Fast 

A **llama.cpp** engine integrating the following optimizations:
1. **PrismML Weight Kernels (`PQ2_0`, `PTQ1_0`)** — CUDA kernels for ternary weights.
2. **TurboQuant KV Cache (`turbo3_0`, `turbo4_0`, `turbo2_0`)** — low-bit vector quantization with Polar Walsh-Hadamard Transform (WHT) query/value rotation.
3. **TriAttention KV Cache Pruning** — trigonometric series scoring and norm-based key eviction for bounded KV cache memory.
4. **CUDA Concurrency & Graph Reuse (`GGML_CUDA_GRAPH_OPT=1`)** — concurrent stream execution for attention projections and CUDA graph reuse.

---

## Original Projects & Research Citations

This work integrates and optimizes methods from the following upstream open-source repositories and research papers:

| Project / Research | Type | Repository / Link | Authors & Organization |
| :--- | :--- | :--- | :--- |
| **PrismML llama.cpp** | Upstream Fork | [GitHub: PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) | PrismML Team (Ternary & Sub-2-bit inference kernels) |
| **TurboQuant + TriAttention** | Upstream Fork | [GitHub: atomicmilkshake/llama-cpp-turboquant](https://github.com/atomicmilkshake/llama-cpp-turboquant) | atomicmilkshake (Native C++/CUDA TurboQuant KV & TriAttention kernels) |
| **TriAttention Paper** | Research (arXiv) | [arXiv:2604.04921](https://arxiv.org/abs/2604.04921) / [PDF](https://arxiv.org/pdf/2604.04921) | Mao et al. (MIT, NVIDIA, Zhejiang University, April 2026) |
| **llama.cpp** | Base Framework | [GitHub: ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Georgi Gerganov & GGML Community |

---

### Release Directory Structure (v1.0.1)

```text
dist/v1.0.1/
├── llama-fast-v1.0.1-bin-cuda13-lite.tar.gz   # Universal Fat Binary Lite (220 MB, relies on host CUDA 13)
├── llama-fast-v1.0.1-bin-cuda12.4-lite.tar.gz # CUDA 12.4 Lite (398 MB, relies on host CUDA 12.4)
├── llama-fast-v1.0.1-bin-cuda13.tar.gz        # Universal Fat Binary (219 MB)
├── llama-fast-v1.0.1-bin-cuda12.4.tar.gz      # Standalone Fat Binary (2.0 GB, with bundled CUDA 12.4 runtime)
├── SHA256SUMS.txt                            # Checksums
│
├── cuda13-lite/ / cuda13/                    # Unpacked CUDA 13 binaries & runners
└── cuda12.4-lite/ / cuda12.4/                # Unpacked CUDA 12.4 binaries & runners
```

---

## GPU Compatibility Matrix

| Build | Target Architectures | Supported GPUs | Min. NVIDIA Driver |
| :--- | :--- | :--- | :--- |
| **CUDA 13.x** *(Fat Binary)* | `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_120` | • **Turing**: GTX 1650/1660, RTX 2060–2080 Ti, T4<br>• **Ampere**: RTX 3050–3090, A10, A40, A100<br>• **Ada Lovelace**: RTX 4060–4090, L4, L40<br>• **Hopper**: H100, H200<br>• **Blackwell**: RTX 50xx, B100, B200 | `>= 550.x` |
| **CUDA 12.4** *(Legacy / Container)* | `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90` | • **Turing**: GTX 16xx, RTX 20xx, T4<br>• **Ampere**: RTX 30xx, A100<br>• **Ada / Hopper**: RTX 40xx, H100 | `>= 525.x` |

---

## Quick Start Guide

### 1. Launch Server via Runner Script:
```bash
./dist/v1.0.1/cuda13/run-server.sh \
  -m /path/to/Ternary-Bonsai-4B-Q2_0_g64.gguf \
  -ngl 99 \
  -c 32768 \
  -np 1 \
  -ctk q8_0 \
  -ctv turbo3 \
  --chat-template chatml \
  --no-cache-prompt \
  --triattention-stats calibration/bonsai-4b.triattention \
  --triattention-budget 2048 \
  --triattention-window 512 \
  --triattention-offset-max 4096 \
  --triattention-normalize \
  --triattention-protect-prefill \
  --reasoning-preserve \
  --reasoning-budget 4096 \
  --reasoning-budget-message "\n[Thinking limit reached. Moving to final response]\n" \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --port 8080 \
  --host 127.0.0.1
```

### 2. Launching Lite Builds (No Bundled CUDA Runtime):
If you use Lite archives, ensure the host CUDA libraries are in your `LD_LIBRARY_PATH`:
```bash
# For CUDA 13 Lite:
./dist/v1.0.1/cuda13-lite/run-server.sh -m /path/to/model.gguf ...

# For CUDA 12.4 Lite:
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
./dist/v1.0.1/cuda12.4-lite/run-server.sh -m /path/to/model.gguf ...
```

### 3. Launch on Windows via WSL2 (.bat):
If you are on Windows, you can launch the server directly using the included batch file:
```cmd
run-server-wsl.bat -m C:\Models\Ternary-Bonsai-4B-Q2_0_g64.gguf -ngl 99 -c 32768
```
*(Or double-click `run-server-wsl.bat` to run with default recommended parameters).*

### 4. Recommended Production Options Explained:
- `-ctk turbo3 -ctv turbo2` — optimal speed (75.9 tok/s) & high quality on long context with maximal VRAM savings.
- `--model-draft models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` — lightweight 1.1 GB speculative drafter yielding +21% speedup over Q8_0.
- `--chat-template chatml` — ensures clean conversational output formatting.
- `--no-cache-prompt` — avoids prompt prefix caching collisions on repeated requests.
- `--triattention-budget 2048` & `--triattention-window 768` — bounds attention decoding computation while maintaining 100% long-context accuracy.
- `--triattention-protect-prefill` — locks the system prompt & instructions in KV cache.
- `--reasoning-budget 4096` & `--reasoning-budget-message` — guarantees deterministic thought-budget capping with medium reasoning effort.

---

## Optimal Pareto-Front Configuration (Ternary-Bonsai-2-27B + DFlash2)

Multi-objective Bayesian Optimization (qLogNEHVI BoTorch, 25 iterations across 7 parameters on Tesla V100 16GB) identified the following optimal Pareto-frontier configurations balancing **Generation Speed (TPS)**, **Deep Logic / Code Accuracy**, and **Long-Context Retrieval (Semantic NIAH 32k/50k)**:

### 🏆 Best Overall Config (Max TPS & 100% Accuracy with Q4_K_M Drafter)
Achieves **~71.5 – 76.0 tokens/sec** with zero CPU offloading (100% VRAM on 16GB GPU):

```bash
./llama-server \
  -m models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --model-draft models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash \
  --spec-draft-ngl 99 \
  --spec-draft-n-max 4 \
  --spec-draft-p-min 0.445 \
  -ngl 99 \
  -ctk turbo3 \
  -ctv turbo2 \
  --triattention-stats bonsai-27b.triattention \
  --triattention-budget 2048 \
  --triattention-window 768 \
  --triattention-offset-max 2048 \
  --triattention-normalize \
  --triattention-agg max \
  --triattention-protect-prefill \
  --reasoning-preserve \
  --reasoning-budget 4096 \
  --reasoning-budget-message "\n[Thinking limit reached. Moving to final response]\n" \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --no-cache-prompt \
  --port 8080 --host 0.0.0.0
```

### Benchmark Metrics:
| Metric | Q8_0 Drafter | Q4_K_M Drafter (New) | Note |
| :--- | :---: | :---: | :--- |
| **Generation Speed** | 58.82 – 64.7 tok/s | **71.54 – 75.92 tok/s** | **+17.3% to +21.6% faster** decode speed |
| **Drafter VRAM Usage** | ~2.0 GB | **~1.1 GB** | **-900 MB VRAM savings** |
| **KV Cache VRAM Usage** | ~1.3 GB (at 60k ctx) | **~1.4 GB** (`turbo3`+`turbo2`) | Walsh-Hadamard Transform (WHT) |
| **Reasoning Accuracy** | 1.0 (100%) | **1.0 (100%)** | GSM8K, Python code gen & Zebra puzzles |
| **Semantic NIAH** | 1.0 (100%) | **1.0 (100%)** | Perfect fact retrieval at 32k and 50k context |

### Pareto Trade-off Profiles (Q4_K_M Drafter):
1. **Ultra Speed & High Quality (Top 1):** `-ctk turbo3 -ctv turbo2 --triattention-budget 2048 --triattention-window 768 --spec-draft-n-max 4 --spec-draft-p-min 0.445` (**75.92 TPS**, 100% Acc)
2. **Balanced Precision Profile:** `-ctk turbo3 -ctv turbo3 --triattention-budget 2048 --triattention-window 512 --spec-draft-n-max 4 --spec-draft-p-min 0.328` (**73.72 TPS**, 100% Acc)
3. **Deep Attention Window:** `-ctk turbo3 -ctv turbo3 --triattention-budget 3584 --triattention-window 1024 --spec-draft-n-max 3 --spec-draft-p-min 0.458` (**72.78 TPS**, 100% Acc)
4. **Conservative High-Precision:** `-ctk turbo4 -ctv turbo3 --triattention-budget 2048 --triattention-window 256 --spec-draft-n-max 4 --spec-draft-p-min 0.25` (**73.94 TPS**, 100% Acc)

---

## CLI Argument Reference (Manual)

### 1. KV Cache Quantization Arguments

| Argument | Allowed Values | Description |
| :--- | :--- | :--- |
| `-ctk, --cache-type-k TYPE` | `turbo3`, `turbo2`, `turbo4`, `q8_0`, `q4_0`, `f16`, `f32` | Format of Key cache. `turbo3` applies Polar WHT rotation and 3-bit quantization. |
| `-ctv, --cache-type-v TYPE` | `q8_0`, `turbo2`, `turbo3`, `turbo4`, `f16`, `f32` | Format of Value cache. Use `q8_0` for maximum decode speed (bypasses inverse WHT) or `turbo3`/`turbo2` for maximum memory compression. |

### 2. TriAttention Pruning Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--triattention-stats PATH` | `""` | Path to precomputed `.triattention` calibration file. Activates adaptive KV pruning. |
| `--triattention-budget N` | `0` | Target KV cache budget. When cache exceeds this budget, low-importance keys are pruned. Recommended: `1024`–`4096`. |
| `--triattention-window, --triattention-divide-length N` | `0` | Pruning check interval in generated tokens. Recommended: `512`. |
| `--triattention-protect-prefill` | `true` | Prevents eviction of initial system prompt / user instruction tokens. |
| `--triattention-no-protect-prefill` | — | Disables prompt token protection. |
| `--triattention-mode MODE` | `global` | Pruning scope: `global` (across all heads/layers), `per-kv-head`, or `per-layer-head`. |
| `--triattention-agg MODE` | `mean` | Score aggregation across query heads: `mean` or `max`. |
| `--triattention-normalize` | `true` | Z-score normalization of scores across heads before selection. |
| `--triattention-offset-max N` | `0` | Maximum RoPE offset frequency bins to consider (recommended: `4096`). |
| `--triattention-log` | `false` | Outputs detailed pruning diagnostics and token retention stats to stderr. |

### 3. Engine & Concurrency Environment Variables

| Variable | Recommended | Description |
| :--- | :--- | :--- |
| `GGML_CUDA_GRAPH_OPT=1` | `1` | Enables concurrent stream execution for attention projections and CUDA graph reuse. |
| `DISABLE_TRIATTENTION=1` | `0` | Forces engine to bypass TriAttention entirely (pure baseline execution). |

---

## Optimized Production Command Examples

### Example 1: Direct `llama-server` Command (Ternary Bonsai 4B / Qwen3)
Recommended launch configuration for chat, agent reasoning, and 32k context:

```bash
./dist/v1.0.1/cuda13/run-server.sh \
  -m /path/to/Ternary-Bonsai-4B-Q2_0_g64.gguf \
  -ngl 99 \
  -c 32768 \
  -np 1 \
  -ctk q8_0 \
  -ctv turbo3 \
  --chat-template chatml \
  --no-cache-prompt \
  --triattention-stats calibration/bonsai-4b.triattention \
  --triattention-budget 2048 \
  --triattention-window 512 \
  --triattention-offset-max 4096 \
  --triattention-normalize \
  --triattention-protect-prefill \
  --reasoning-preserve \
  --reasoning-budget 4096 \
  --reasoning-budget-message "\n[Thinking limit reached. Moving to final response]\n" \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --port 8080 \
  --host 127.0.0.1
```

### Example 2: Interactive CLI Generation (`llama-cli`)
One-shot generation or terminal testing:

```bash
./dist/v1.0.1/cuda13/run-cli.sh \
  -m /path/to/Ternary-Bonsai-4B-Q2_0_g64.gguf \
  -ngl 99 \
  -c 16384 \
  -ctk q8_0 \
  -ctv turbo3 \
  --chat-template chatml \
  --triattention-stats calibration/bonsai-4b.triattention \
  --triattention-budget 2048 \
  --triattention-window 512 \
  --triattention-protect-prefill \
  --reasoning-preserve \
  --reasoning-budget 4096 \
  --reasoning-budget-message "\n[Thinking limit reached. Moving to final response]\n" \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  -p "Explain the mathematical intuition behind Polar Walsh-Hadamard Transform in TurboQuant:"
```

### Example 4: TriAttention Offline Calibration
To generate a new `.triattention` calibration profile from an arbitrary text corpus:

```bash
Release/build/cuda13/bin/llama-triattention-calibrate \
  -m /path/to/model.gguf \
  --triattention-calibrate corpus.txt \
  --triattention-calibrate-out my_model.triattention \
  -ngl 99 \
  -c 8192
```

---


## Tests Results

Tested on **NVIDIA GeForce RTX 3090 (24GB)** with **Pi Agent** with model **Ternary-Bonsai-2-27B-PQ2_0**:

| Metric | Baseline PrismML (FP16 KV) | Turbo Fast (t3+q8, win 512) | Turbo Max-Mem (t3+t2, win 512) |
| :--- | :---: | :---: | :---: |
| **Total 10 Tasks Time** | **875.89 s** (14m 35s) | **632.07 s** (10m 32s) | **846.83 s** (14m 06s) |
| **Speedup vs Baseline** | 1.00× (Base) | **1.39× faster** | 1.03× faster |
| **VRAM Usage (16K ctx)** | ~11,800 MiB | 8,122 MiB | **7,914 MiB** |
| **VRAM Savings** | 0 (Base) | -3.7 GB VRAM | **-3.9 GB VRAM** |
| **Tokens per 1 GB VRAM** | ~4,000 | ~20,000 | **~25,200** |
| **Peak Decode Speed** | **68.68 tok/s** | **68.04 tok/s** | **67.68 tok/s** |

---

## Native Compilation from Source (`Release/src`)

To build natively from `Release/src/` or unpack `llama-fast-src.zip`:
```bash
cd Release/src
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;100;120" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-cli llama-triattention-calibrate -j$(nproc)
```

