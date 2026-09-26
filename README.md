# Llama-Fast 

A **llama.cpp** engine integrating the following optimizations:
1. **PrismML Weight Kernels (`PQ2_0`, `PTQ1_0`)** — CUDA kernels for ternary and 2-bit weights.
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

## Release Directory Structure

```text
Release/
├── src/                                  # Clean fork source code (no build artifacts, no .git)
│
├── calibration/                          # Subdirectory with calibration profiles
│   └── bonsai-27b.triattention           # TriAttention calibration profile for Ternary-Bonsai-2-27B (772 KB)
│
├── scripts/                              # Production launch & CLI scripts
│   ├── start_server_turbo.sh             # Launch server with TurboQuant + TriAttention (customizable)
│   ├── start_server_baseline.sh          # Launch server in pure baseline mode (FP16 KV)
│   └── run_cli.sh                        # Interactive CLI chat / text completion
│
├── build/                                # Precompiled production binaries
│   ├── cuda13/                           # Universal Fat Binary (CUDA 13.x: sm_75..sm_120)
│   │   ├── bin/                          # llama-server, llama-cli, llama-triattention-calibrate, tests
│   │   ├── lib/cuda/                     # Standalone CUDA 13 runtime (cudart, cublas, cublasLt)
│   │   ├── calibration/                  # Local copy of calibration file
│   │   ├── start_server.sh               # Local launch script
│   │   └── run_cli.sh                    # Local CLI chat
│   │
│   └── cuda12.4/                         # Legacy Release (CUDA 12.4: sm_61..sm_86)
│       ├── bin/                          # llama-server, llama-cli, llama-triattention-calibrate, tests
│       ├── lib/cuda/                     # Standalone CUDA 12.4 runtime (cudart, cublas, cublasLt)
│       ├── calibration/                  # Local copy of calibration file
│       ├── start_server.sh               # Local launch script
│       └── run_cli.sh                    # Local CLI chat
│ 
├── llama-fast-src.zip                    # Clean source code ZIP archive (~37.5 MB)
└── README.md                             # Release documentation and guide
```

---

## GPU Compatibility Matrix

| Build Folder | Target Architectures | Supported GPUs | Min. NVIDIA Driver |
| :--- | :--- | :--- | :--- |
| **`build/cuda13/`** *(Universal Fat Binary)* | `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90`, `sm_100`, `sm_120` | • **Turing**: GTX 1650/1660, RTX 2060–2080 Ti, T4<br>• **Ampere**: RTX 3050–3090, A10, A40, A100<br>• **Ada Lovelace**: RTX 4060–4090, L4, L40<br>• **Hopper**: H100, H200<br>• **Blackwell**: RTX 50xx, B100, B200 | `>= 550.x` |
| **`build/cuda12.4/`** *(Legacy Release)* | `sm_61`, `sm_70`, `sm_75`, `sm_80`, `sm_86` | • **Pascal**: GTX 1060–1080 Ti, Tesla P40, P100<br>• **Volta**: Titan V, Tesla V100<br>• **Turing**: GTX 16xx, RTX 20xx<br>• **Ampere**: RTX 30xx, A100 | `>= 525.x` |

---

## Quick Start Guide

### 1. Launch Server (Default Speed-Optimized Profile):
```bash
cd Release
./start_server.sh /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf
```

### 2. Switching Inference Profiles:

```bash
# SPEED PROFILE (Recommended for autonomous coding / reasoning agents)
# K=turbo3, V=q8_0: eliminates 64 inverse WHT kernels per token, 68 tok/s, concise CoT
CTK=turbo3 CTV=q8_0 ./start_server.sh /path/to/model.gguf

# MAXIMUM VRAM COMPRESSION PROFILE (For long context 32K–64K or high concurrency)
# K=turbo3, V=turbo2: ~25,200 tokens per 1 GB VRAM, 7.9 GB VRAM at 16K ctx
CTK=turbo3 CTV=turbo2 ./start_server.sh /path/to/model.gguf

# PURE BASELINE PROFILE (100% backward compatible, no TriAttention)
DISABLE_TRIATTENTION=1 ./start_server.sh /path/to/model.gguf
```

### 3. Launch from Architecture-Specific Subdirectories:
```bash
# Run with CUDA 13.x:
cd Release/build/cuda13 && ./start_server.sh /path/to/model.gguf

# Run with CUDA 12.4:
cd Release/build/cuda12.4 && ./start_server.sh /path/to/model.gguf
```

---

## CLI Argument Reference (Manual)

### 1. KV Cache Quantization Arguments

| Argument | Allowed Values | Description |
| :--- | :--- | :--- |
| `-ctk, --cache-type-k TYPE` | `turbo3`, `turbo2`, `turbo4`, `q8_0`, `q4_0`, `f16`, `f32` | Format of Key cache. `turbo3` applies Polar WHT rotation and 3-bit quantization. |
| `-ctv, --cache-type-v TYPE` | `q8_0`, `turbo2`, `turbo3`, `turbo4`, `f16`, `f32` | Format of Value cache. Use `q8_0` for maximum decode speed (bypasses inverse WHT) or `turbo2` for maximum memory compression. |

### 2. TriAttention Pruning Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--triattention-stats PATH` | `""` | Path to precomputed `.triattention` calibration file. Activates adaptive KV pruning. |
| `--triattention-budget N` | `0` | Target KV cache budget. When cache exceeds this budget, low-importance keys are pruned. Recommended: `2048`–`4096`. |
| `--triattention-window, --triattention-divide-length N` | `0` | Pruning check interval in generated tokens. Recommended: `512` (halves GPU kernel launch frequency vs `256`). |
| `--triattention-protect-prefill` | `true` | Prevents eviction of initial system prompt / user instruction tokens. |
| `--triattention-no-protect-prefill` | — | Disables prompt token protection. |
| `--triattention-mode MODE` | `global` | Pruning scope: `global` (across all heads/layers), `per-kv-head`, or `per-layer-head`. |
| `--triattention-agg MODE` | `mean` | Score aggregation across query heads: `mean` or `max`. |
| `--triattention-normalize` | `true` | Z-score normalization of scores across heads before selection. |
| `--triattention-log` | `false` | Outputs detailed pruning diagnostics and token retention stats to stderr. |

### 3. Engine & Concurrency Environment Variables

| Variable | Recommended | Description |
| :--- | :--- | :--- |
| `GGML_CUDA_GRAPH_OPT=1` | `1` | Enables concurrent stream execution for attention projections and CUDA graph reuse. |
| `DISABLE_TRIATTENTION=1` | `0` | Forces engine to bypass TriAttention entirely (pure baseline execution). |

---

## Optimized Production Command Examples

### Example 1: Direct `llama-server` Command (High-Speed Agent Profile)
Recommended for coding, terminal reasoning, and interactive agents. Uses 3-bit K, 8-bit V (0 inverse WHT overhead), window 512, budget 4096:

```bash
export LD_LIBRARY_PATH="Release/build/cuda13/bin:Release/build/cuda13/lib/cuda:$LD_LIBRARY_PATH"
export GGML_CUDA_GRAPH_OPT=1

Release/build/cuda13/bin/llama-server \
  -m /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --mmproj /path/to/mmproj-Qwen3.8-27B-BF16.gguf \
  --no-mmproj-offload \
  -ngl 99 \
  -c 16384 \
  -ctk turbo3 \
  -ctv q8_0 \
  --triattention-stats Release/calibration/bonsai-27b.triattention \
  --triattention-budget 4096 \
  --triattention-window 512 \
  --triattention-protect-prefill \
  --host 127.0.0.1 \
  --port 8080
```

### Example 2: Direct `llama-server` Command (Maximum VRAM Compression Profile)
Runs long context (32K–64K) or multiple concurrent sessions with ~25,200 tokens per 1 GB VRAM:

```bash
export LD_LIBRARY_PATH="Release/build/cuda13/bin:Release/build/cuda13/lib/cuda:$LD_LIBRARY_PATH"
export GGML_CUDA_GRAPH_OPT=1

Release/build/cuda13/bin/llama-server \
  -m /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 \
  -c 32768 \
  -ctk turbo3 \
  -ctv turbo2 \
  --triattention-stats Release/calibration/bonsai-27b.triattention \
  --triattention-budget 4096 \
  --triattention-window 512 \
  --triattention-protect-prefill \
  --host 0.0.0.0 \
  --port 8080
```

### Example 3: Interactive CLI Generation (`llama-cli`)
One-shot question or interactive conversation with speed optimizations:

```bash
export LD_LIBRARY_PATH="Release/build/cuda13/bin:Release/build/cuda13/lib/cuda:$LD_LIBRARY_PATH"
export GGML_CUDA_GRAPH_OPT=1

Release/build/cuda13/bin/llama-cli \
  -m /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 \
  -c 8192 \
  -ctk turbo3 \
  -ctv q8_0 \
  --triattention-stats Release/calibration/bonsai-27b.triattention \
  --triattention-budget 2048 \
  --triattention-window 512 \
  -p "Explain why polar Walsh-Hadamard transform improves KV cache quantization:"
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

