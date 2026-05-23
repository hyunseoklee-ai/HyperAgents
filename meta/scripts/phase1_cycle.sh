#!/usr/bin/env bash
# Phase 1 cycle: vLLM serves inference  ->  shut down  ->  LoRA SFT
#   ->  restart vLLM with --enable-lora pointing at the new adapter
#
# Encapsulates the GPU handoff. vLLM and training can't both hold the GPU,
# so we serialize: stop the server, free the memory, train, restart.
#
# Usage:
#   meta/scripts/phase1_cycle.sh                     # uses meta/training/lora_config.yaml
#   meta/scripts/phase1_cycle.sh --config xxx.yaml --output checkpoints/phase1_v1
#   meta/scripts/phase1_cycle.sh --skip-train        # only restart with existing LoRA
#   meta/scripts/phase1_cycle.sh --no-restart        # only train; leave GPU free for next thing

set -euo pipefail

CONFIG="meta/training/lora_config.yaml"
OUTPUT=""                  # default read from config
SKIP_TRAIN=0
NO_RESTART=0
RESTART_PORT=8001
RESTART_CTX=131072

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)        CONFIG="$2"; shift 2 ;;
    --output)        OUTPUT="$2"; shift 2 ;;
    --skip-train)    SKIP_TRAIN=1; shift ;;
    --no-restart)    NO_RESTART=1; shift ;;
    --port)          RESTART_PORT="$2"; shift 2 ;;
    --ctx)           RESTART_CTX="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# resolve OUTPUT from config if not given
if [ -z "$OUTPUT" ]; then
  OUTPUT=$(python3.11 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['training']['output_dir'])" "$CONFIG")
fi

echo "[phase1_cycle] config: $CONFIG"
echo "[phase1_cycle] output: $OUTPUT"

# ---- 1. shut down vLLM ----
"$SCRIPT_DIR/serve_qwen35.sh" --kill

if [ "$SKIP_TRAIN" = "0" ]; then
  # ---- 2. data prep (idempotent) ----
  echo "[phase1_cycle] data prep ..."
  python3.11 -m meta.training.data_prep --config "$CONFIG"

  # ---- 3. LoRA SFT ----
  echo "[phase1_cycle] running LoRA SFT ..."
  python3.11 -m meta.training.sft_generator --config "$CONFIG" --output "$OUTPUT"

  echo "[phase1_cycle] LoRA written to: $OUTPUT"
  ls -la "$OUTPUT"/*.safetensors 2>/dev/null | head -5 || true
else
  echo "[phase1_cycle] --skip-train: assuming existing LoRA at $OUTPUT"
fi

if [ "$NO_RESTART" = "1" ]; then
  echo "[phase1_cycle] --no-restart: GPU is free, exiting."
  exit 0
fi

# ---- 4. restart vLLM with LoRA ----
echo "[phase1_cycle] restarting vLLM with --enable-lora ..."
"$SCRIPT_DIR/serve_qwen35.sh" \
  --port "$RESTART_PORT" \
  --ctx  "$RESTART_CTX" \
  --lora-dir  "$OUTPUT" \
  --lora-name "qwen3.5-4b-phase1"

# ---- 5. sanity check the LoRA loaded ----
echo "[phase1_cycle] verifying LoRA registered ..."
curl -s "http://127.0.0.1:$RESTART_PORT/v1/models" \
  | python3 -c "import sys,json; ms=json.load(sys.stdin)['data']; print('models:', [m['id'] for m in ms])"
echo "[phase1_cycle] done"
