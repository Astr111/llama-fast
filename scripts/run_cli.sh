#!/bin/bash
# ==============================================================================
# run_cli.sh — Interactive CLI chat / completion via llama-cli (CUDA 13 Universal)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$SCRIPT_DIR/bin:$SCRIPT_DIR/lib/cuda:${LD_LIBRARY_PATH:-}"
export GGML_CUDA_GRAPH_OPT="${GGML_CUDA_GRAPH_OPT:-1}"

MODEL_PATH="${1:-${MODEL_PATH:-/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf}}"
STATS_PATH="${STATS_PATH:-$SCRIPT_DIR/../calibration/bonsai-27b.triattention}"
CTK="${CTK:-turbo3}"
CTV="${CTV:-q8_0}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Usage: $0 /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf [prompt]"
    exit 1
fi

TRIATTN_ARGS=()
if [ -f "$STATS_PATH" ] && [ "${DISABLE_TRIATTENTION:-0}" != "1" ]; then
    TRIATTN_ARGS+=(--triattention-stats "$STATS_PATH" --triattention-budget 4096 --triattention-window 512 --triattention-protect-prefill)
fi

shift || true
PROMPT="$*"

if [ -z "$PROMPT" ]; then
    echo "Starting interactive chat mode..."
    exec "$SCRIPT_DIR/bin/llama-cli" \
      -m "$MODEL_PATH" \
      -ctk "$CTK" \
      -ctv "$CTV" \
      "${TRIATTN_ARGS[@]}" \
      -ngl 99 \
      -c 16384 \
      --conversation
else
    echo "Generating response for prompt: $PROMPT"
    exec "$SCRIPT_DIR/bin/llama-cli" \
      -m "$MODEL_PATH" \
      -ctk "$CTK" \
      -ctv "$CTV" \
      "${TRIATTN_ARGS[@]}" \
      -ngl 99 \
      -c 16384 \
      -p "$PROMPT"
fi
