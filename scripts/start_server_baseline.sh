#!/usr/bin/env bash
# ==============================================================================
# Baseline Launch Script: llama-server with standard FP16 KV Cache
# Model: Ternary-Bonsai-2-27B-PQ2_0 (PrismML 2-bit Weights)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN_DIR="${BIN_DIR:-${BASE_DIR}/../dist-rtx3090/bin}"
MODEL_PATH="${MODEL_PATH:-${HOME}/Prism/llama-prism-b10743-adfffbe/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
MMPROJ_PATH="${MMPROJ_PATH:-${HOME}/Prism/llama-prism-b10743-adfffbe/mmproj-Qwen3.8-27B-BF16.gguf}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-16384}"
NGL="${NGL:-99}"

echo "====================================================================="
echo " Starting llama-server [Baseline FP16 KV]"
echo "====================================================================="
echo " Model:             ${MODEL_PATH}"
echo " Multimodal:        ${MMPROJ_PATH}"
echo " KV Cache:          FP16 (256 KB/token, 4K tok/GB VRAM)"
echo " Context Window:    ${CTX_SIZE} tokens"
echo " Listening on:      http://${HOST}:${PORT}"
echo "====================================================================="

export LD_LIBRARY_PATH="${BIN_DIR}:${BIN_DIR}/../lib/cuda:${LD_LIBRARY_PATH:-}"

exec "${BIN_DIR}/llama-server" \
  -m "${MODEL_PATH}" \
  --mmproj "${MMPROJ_PATH}" \
  --no-mmproj-offload \
  -ngl "${NGL}" \
  -c "${CTX_SIZE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
