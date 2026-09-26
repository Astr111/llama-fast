#!/bin/bash
# ==============================================================================
# start_server.sh — Launch llama-server with NVIDIA GPU (CUDA 13 Modern)
# Supported: Turing (16xx/20xx/T4), Ampere (30xx/A100), Ada (40xx/L40),
#            Hopper (H100), Blackwell (50xx/B200)
# PrismML (PQ2_0 / PTQ1_0) + TurboQuant KV (turbo3/q8_0/turbo2) + TriAttention
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configure library paths: bundled binaries, bundled CUDA runtime, then system
export LD_LIBRARY_PATH="$SCRIPT_DIR/bin:$SCRIPT_DIR/lib/cuda:${LD_LIBRARY_PATH:-}"
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

MODEL_PATH="${1:-${MODEL_PATH:-/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf}}"
STATS_PATH="${STATS_PATH:-$SCRIPT_DIR/../calibration/bonsai-27b.triattention}"
CTK="${CTK:-turbo3}"
CTV="${CTV:-q8_0}"
TRI_BUDGET="${TRI_BUDGET:-4096}"
TRI_WINDOW="${TRI_WINDOW:-512}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "WARNING: Model file not found at: $MODEL_PATH"
    echo "Usage: $0 /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf"
    echo "Or set the path via MODEL_PATH environment variable."
    exit 1
fi

TRIATTN_ARGS=()
if [ -f "$STATS_PATH" ] && [ "${DISABLE_TRIATTENTION:-0}" != "1" ]; then
    TRIATTN_ARGS+=(--triattention-stats "$STATS_PATH" --triattention-budget "$TRI_BUDGET" --triattention-window "$TRI_WINDOW" --triattention-protect-prefill)
fi

echo "========================================================================"
echo "  Starting llama-server (CUDA 13 Universal Fat Binary)"
echo "  Model:       $MODEL_PATH"
echo "  KV Cache:    $CTK (K) + $CTV (V)"
if [ ${#TRIATTN_ARGS[@]} -gt 0 ]; then
    echo "  TriAttention: budget=$TRI_BUDGET, window=$TRI_WINDOW, stats=$STATS_PATH"
else
    echo "  TriAttention: disabled (standard baseline decode path)"
fi
echo "  Offload:     100% GPU (-ngl 99)"
echo "  Address:     http://$HOST:$PORT"
echo "========================================================================"

exec "$SCRIPT_DIR/bin/llama-server" \
  -m "$MODEL_PATH" \
  -ctk "$CTK" \
  -ctv "$CTV" \
  "${TRIATTN_ARGS[@]}" \
  -ngl 99 \
  -c 32768 \
  -n 8192 \
  --reasoning-budget 4000 \
  --reasoning-budget-message $'\n[Thinking limit reached. Moving to final response]\n' \
  -np 1 \
  -t "$(nproc)" \
  --host "$HOST" \
  --port "$PORT"
