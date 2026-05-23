#!/usr/bin/env bash
# Serve Qwen3.5-35B-A3B-GPTQ-Int4 (MoE, 3B active per forward, GPTQ Int4) on
# vLLM as the Phase H *proposer* model. Default port :8000 (distinct from the
# 4B inner-agent server on :8001).
#
# Because the 4B server occupies ~73 GB / 82 GB on the single A100-80GB, this
# script can swap it out: pass --swap-from 8001 to SIGTERM whatever vLLM holds
# that port and wait for VRAM to drop below 2 GiB before launching.
#
# Usage:
#   meta/scripts/serve_qwen35b_a3b.sh                                # plain launch
#   meta/scripts/serve_qwen35b_a3b.sh --swap-from 8001               # kill 4B server first
#   meta/scripts/serve_qwen35b_a3b.sh --ctx 32768 --gpu-mem 0.80
#   meta/scripts/serve_qwen35b_a3b.sh --quantization gptq_marlin     # fallback kernel
#   meta/scripts/serve_qwen35b_a3b.sh --kill                         # stop only

set -euo pipefail

PORT=8000
CTX=65536
GPU_MEM=0.85
MODEL_ID="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
SERVED_NAME="qwen3.5-35b-a3b"
QUANT="moe_wna16"
REASONING_PARSER="qwen3"
LANG_ONLY=1
SWAP_FROM=""
KILL_ONLY=0
LOG=/home/t-hyunlee/meta-harness-plan/logs/vllm_35b.log

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)            PORT="$2"; shift 2 ;;
    --ctx)             CTX="$2"; shift 2 ;;
    --gpu-mem)         GPU_MEM="$2"; shift 2 ;;
    --model)           MODEL_ID="$2"; shift 2 ;;
    --served-name)     SERVED_NAME="$2"; shift 2 ;;
    --quantization)    QUANT="$2"; shift 2 ;;
    --reasoning-parser) REASONING_PARSER="$2"; shift 2 ;;
    --no-lang-only)    LANG_ONLY=0; shift ;;
    --swap-from)       SWAP_FROM="$2"; shift 2 ;;
    --log)             LOG="$2"; shift 2 ;;
    --kill)            KILL_ONLY=1; shift ;;
    -h|--help)         sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

kill_vllm_holding_port() {
  local p="$1"
  local pid
  pid=$(ss -tlnp 2>/dev/null | awk -v port=":$p" '$4 ~ port { gsub(".*pid=",""); gsub(",.*",""); print; exit }')
  if [ -n "${pid:-}" ]; then
    echo "[serve_35b] killing PID $pid on :$p"
    kill "$pid" 2>/dev/null || true
    for i in $(seq 1 60); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "[serve_35b] SIGKILL"
      kill -9 "$pid" 2>/dev/null || true
      sleep 2
    fi
  else
    echo "[serve_35b] no vLLM holding :$p (already free)"
  fi
}

wait_for_vram_free() {
  local threshold="${1:-2000}"
  echo "[serve_35b] waiting for VRAM to drop below ${threshold} MiB..."
  for i in $(seq 1 60); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${USED:-99999}" -lt "$threshold" ]; then
      echo "[serve_35b] VRAM available (${USED} MiB used after $((i*2))s)"
      return 0
    fi
    sleep 2
  done
  echo "[serve_35b] WARNING: VRAM did not drop below ${threshold} MiB after 120s"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
  return 0
}

if [ -n "$SWAP_FROM" ]; then
  kill_vllm_holding_port "$SWAP_FROM"
  wait_for_vram_free 2000
fi

if pgrep -f "vllm.entrypoints.openai.api_server.*--port $PORT" > /dev/null 2>&1; then
  echo "[serve_35b] killing existing vLLM on :$PORT"
  pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" || true
  for i in $(seq 1 30); do
    pgrep -f "vllm.entrypoints.openai.api_server.*--port $PORT" > /dev/null 2>&1 || break
    sleep 1
  done
fi

if [ "$KILL_ONLY" = "1" ]; then
  echo "[serve_35b] --kill done"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
CMD=(python3.11 -m vllm.entrypoints.openai.api_server
     --model "$MODEL_ID"
     --port "$PORT"
     --host 0.0.0.0
     --max-model-len "$CTX"
     --gpu-memory-utilization "$GPU_MEM"
     --tensor-parallel-size 1
     --quantization "$QUANT"
     --reasoning-parser "$REASONING_PARSER"
     --enable-prefix-caching
     --served-model-name "$SERVED_NAME")
if [ "$LANG_ONLY" = "1" ]; then
  CMD+=(--language-model-only)
fi

echo "[serve_35b] launching: ${CMD[*]}"
echo "[serve_35b] log:       $LOG"
nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
echo "[serve_35b] PID:       $PID"

echo "[serve_35b] waiting for /v1/models on :$PORT ..."
for i in $(seq 1 120); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[serve_35b] FATAL: vLLM process died during startup. Last 40 log lines:"
    tail -40 "$LOG" >&2
    exit 1
  fi
  if curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
    echo "[serve_35b] READY after $((i*5))s"
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d['data'][0]; print(f'  served: {m[\"id\"]}  max_model_len: {m.get(\"max_model_len\",\"?\")}')" 2>/dev/null
    exit 0
  fi
  sleep 5
done

echo "[serve_35b] TIMEOUT after 10 min — last 40 log lines:"
tail -40 "$LOG" >&2
exit 1
