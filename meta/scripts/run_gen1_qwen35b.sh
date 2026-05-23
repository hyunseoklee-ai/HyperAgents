#!/usr/bin/env bash
# End-to-end orchestrator for gen_1_qwen35b:
#   1. swap 4B (:8001) -> 35B-A3B (:8000)
#   2. run meta-agent v3 (proposer = Qwen3.5-35B-A3B-Int4)
#   3. swap 35B (:8000) -> 4B (:8001)
#   4. run inner eval on 20-task slice with the new prompts.yaml
#   5. score + summarize
#
# Designed to be re-runnable. Each step has its own log; idempotent guards.
#
# Usage:
#   meta/scripts/run_gen1_qwen35b.sh                 # full run
#   meta/scripts/run_gen1_qwen35b.sh --propose-only  # skip eval
#   meta/scripts/run_gen1_qwen35b.sh --eval-only     # skip propose (re-eval existing diff)

set -euo pipefail

REPO=/home/t-hyunlee/mini-swe-agent
LOGDIR=/home/t-hyunlee/meta-harness-plan/logs
SCRIPTS=$REPO/meta/scripts
TS=$(date +%Y%m%d_%H%M%S)

PROPOSE=1
EVAL=1
OUTPUT_SUFFIX=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --propose-only)  EVAL=0; shift ;;
    --eval-only)     PROPOSE=0; shift ;;
    --output-suffix) OUTPUT_SUFFIX="$2"; shift 2 ;;
    -h|--help)       sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
OUT=/home/t-hyunlee/meta-harness-plan/results/phase_h/gen_1${OUTPUT_SUFFIX}_qwen35b

mkdir -p "$OUT" "$LOGDIR"

echo "=========================================="
echo "[$TS] gen_1_qwen35b orchestrator"
echo "  REPO=$REPO  OUT=$OUT"
echo "  PROPOSE=$PROPOSE  EVAL=$EVAL"
echo "=========================================="

if [ "$PROPOSE" = "1" ]; then
  echo
  echo "--- Step 1: snapshot 4B server state (for sanity post-swap)"
  curl -s http://127.0.0.1:8001/v1/models > "$OUT/_pre_swap_4B_models.json" 2>/dev/null \
    || echo "(no 4B server up — skipping snapshot)"

  echo
  echo "--- Step 2: swap 4B -> 35B (kill :8001, launch :8000)"
  "$SCRIPTS/serve_qwen35b_a3b.sh" --swap-from 8001 \
    > "$LOGDIR/swap_to_35b_${TS}.log" 2>&1 \
    || { echo "FATAL: 35B server launch failed"; tail -40 "$LOGDIR/swap_to_35b_${TS}.log"; exit 3; }

  echo
  echo "--- Step 3: run meta-agent v3 (proposer = Qwen3.5-35B-A3B-Int4)"
  cd "$REPO"
  if python3.11 -m meta.scripts.run_meta_agent_v3_qwen35b \
       --gen 1 --output-suffix "$OUTPUT_SUFFIX" --no-eval \
       > "$LOGDIR/meta_agent_v3_${TS}.log" 2>&1; then
    echo "meta-agent v3 done. diff:"
    ls -la "$OUT"/diff.patch 2>/dev/null || echo "(no diff.patch)"
  else
    echo "meta-agent v3 FAILED. Last 40 log lines:"
    tail -40 "$LOGDIR/meta_agent_v3_${TS}.log"
    "$SCRIPTS/serve_qwen35b_a3b.sh" --kill || true
    "$SCRIPTS/serve_qwen35.sh" || true
    exit 4
  fi

  echo
  echo "--- Step 4: swap 35B -> 4B (kill :8000, restart :8001)"
  "$SCRIPTS/serve_qwen35b_a3b.sh" --kill > "$LOGDIR/kill_35b_${TS}.log" 2>&1 || true
  "$SCRIPTS/serve_qwen35.sh" --lora-dir /home/t-hyunlee/mini-swe-agent/meta/training/checkpoints/phase1_v0 \
    > "$LOGDIR/swap_to_4b_${TS}.log" 2>&1 \
    || { echo "FATAL: 4B restart failed"; tail -40 "$LOGDIR/swap_to_4b_${TS}.log"; exit 5; }

  echo
  echo "--- Step 5: verify post-swap 4B server"
  curl -s http://127.0.0.1:8001/v1/models > "$OUT/_post_swap_4B_models.json" 2>/dev/null \
    || { echo "FATAL: post-swap 4B not responding"; exit 6; }
  if diff -q "$OUT/_pre_swap_4B_models.json" "$OUT/_post_swap_4B_models.json" > /dev/null 2>&1; then
    echo "  ✓ 4B server config byte-identical pre/post swap"
  else
    echo "  ⚠ 4B server config differs pre/post (could be ok — check $OUT/_post_swap_4B_models.json)"
    diff "$OUT/_pre_swap_4B_models.json" "$OUT/_post_swap_4B_models.json" | head -20 || true
  fi
fi

if [ "$EVAL" = "1" ]; then
  echo
  echo "--- Step 6: run inner eval (20 tasks, prompts from meta-agent's edit)"
  PROMPTS="$OUT/seed_harness/prompts.yaml"
  if [ ! -f "$PROMPTS" ]; then
    echo "FATAL: no prompts.yaml at $PROMPTS — did meta-agent produce a candidate?"
    exit 7
  fi
  cd "$REPO"
  python3.11 -m meta.phase0.run_phase0 \
    --slice 0:20 --workers 4 \
    --model-name hosted_vllm/qwen3.5-4b \
    --config "$PROMPTS" \
    --output "$OUT/eval" \
    > "$LOGDIR/eval_gen1_qwen35b_${TS}.log" 2>&1 \
    || { echo "FATAL: eval failed"; tail -40 "$LOGDIR/eval_gen1_qwen35b_${TS}.log"; exit 8; }

  echo
  echo "--- Step 7: score"
  python3.11 -m meta.phase0.score_patches \
    --preds "$OUT/eval/preds.json" \
    --output "$OUT/scores.json" \
    --only-submitted --workers 2 \
    > "$LOGDIR/score_gen1_qwen35b_${TS}.log" 2>&1 \
    || { echo "FATAL: scoring failed"; tail -40 "$LOGDIR/score_gen1_qwen35b_${TS}.log"; exit 9; }
fi

echo
echo "=========================================="
echo "[$(date +%Y%m%d_%H%M%S)] gen_1_qwen35b: DONE"
echo "  artifacts:"
echo "    $OUT/diff.patch"
echo "    $OUT/meta_trajectory.json"
echo "    $OUT/eval/"
echo "    $OUT/scores.json"
echo "=========================================="
