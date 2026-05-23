#!/usr/bin/env bash
# Serve Qwen3.5-4B (optionally with a LoRA adapter) on vLLM, ready for
# mini-swe-agent + Hyperagents. Kills any existing vLLM, waits for GPU free,
# launches the new server in the background, and blocks until /v1/models
# returns 200.
#
# Usage:
#   meta/scripts/serve_qwen35.sh                            # default 128K, no LoRA
#   meta/scripts/serve_qwen35.sh --ctx 32768                # short ctx
#   meta/scripts/serve_qwen35.sh --lora-dir checkpoints/phase1_v0
#   meta/scripts/serve_qwen35.sh --port 8002 --gpu-mem 0.85
#   meta/scripts/serve_qwen35.sh --kill                     # stop only

set -euo pipefail

PORT=8001
CTX=131072
GPU_MEM=0.90
MODEL_ID="Qwen/Qwen3.5-4B"
SERVED_NAME="qwen3.5-4b"
LORA_DIR=""
LORA_NAME=""
LANG_ONLY=1   # default ON — Qwen3.5-4B is a *ForConditionalGeneration model
              # whose vision tower we never use; skipping it saves VRAM.
              # Pass --no-lang-only to reproduce the pre-2026-05-23 baseline
              # exactly (gen_0 / A2 / A2' were all run with the vision tower
              # loaded but unused).
KILL_ONLY=0
LOG=/home/t-hyunlee/meta-harness-plan/logs/vllm.log

# ---- args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)      PORT="$2"; shift 2 ;;
    --ctx)       CTX="$2"; shift 2 ;;
    --gpu-mem)   GPU_MEM="$2"; shift 2 ;;
    --model)     MODEL_ID="$2"; shift 2 ;;
    --served-name) SERVED_NAME="$2"; shift 2 ;;
    --lora-dir)  LORA_DIR="$2"; shift 2 ;;
    --lora-name) LORA_NAME="$2"; shift 2 ;;
    --log)       LOG="$2"; shift 2 ;;
    --lang-only)    LANG_ONLY=1; shift ;;
    --no-lang-only) LANG_ONLY=0; shift ;;
    --kill)      KILL_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,15p' "$0"
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- kill existing ----------------------------------------------------
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null 2>&1; then
  echo "[serve_qwen35] killing existing vLLM..."
  pkill -f "vllm.entrypoints.openai.api_server" || true
  for i in $(seq 1 20); do
    pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null 2>&1 || break
    sleep 1
  done
fi

# ---- wait for GPU memory free ----------------------------------------
echo "[serve_qwen35] waiting for GPU memory..."
for i in $(seq 1 30); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${USED:-9999}" -lt 2000 ]; then
    echo "[serve_qwen35] GPU memory available (${USED} MiB used)"
    break
  fi
  sleep 2
done

if [ "$KILL_ONLY" = "1" ]; then
  echo "[serve_qwen35] --kill done"
  exit 0
fi

# ---- launch -----------------------------------------------------------
mkdir -p "$(dirname "$LOG")"
CMD=(python3.11 -m vllm.entrypoints.openai.api_server
     --model "$MODEL_ID"
     --port "$PORT"
     --host 0.0.0.0
     --max-model-len "$CTX"
     --gpu-memory-utilization "$GPU_MEM"
     --tensor-parallel-size 1
     --enable-prefix-caching
     --served-model-name "$SERVED_NAME")

if [ "$LANG_ONLY" = "1" ]; then
  CMD+=(--language-model-only)
fi

if [ -n "$LORA_DIR" ]; then
  LN="${LORA_NAME:-${SERVED_NAME}-lora}"
  CMD+=(--enable-lora --max-loras 1 --max-lora-rank 64
        --lora-modules "${LN}=${LORA_DIR}")
  echo "[serve_qwen35] LoRA enabled: ${LN} from ${LORA_DIR}"
fi

echo "[serve_qwen35] launching: ${CMD[*]}"
echo "[serve_qwen35] log:       $LOG"
nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
echo "[serve_qwen35] PID:       $PID"

# ---- wait until /v1/models 200 ---------------------------------------
echo "[serve_qwen35] waiting for /v1/models on :$PORT ..."
for i in $(seq 1 60); do
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
    echo "[serve_qwen35] READY after ${i}*5s"
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  served: {d[\"data\"][0][\"id\"]}, max_model_len: {d[\"data\"][0][\"max_model_len\"]}')" 2>/dev/null
    exit 0
  fi
  sleep 5
done

echo "[serve_qwen35] TIMEOUT — server did not respond within 5 min"
tail -30 "$LOG" >&2
exit 1
