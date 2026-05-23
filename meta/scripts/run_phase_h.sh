#!/usr/bin/env bash
# Phase H — harness-only evolve (no weight training).
#
# Uses Qwen3.5-4B (no LoRA) as BOTH the inner agent and the meta_agent.
# At each generation the meta_agent edits the files under
#   meta/hyperagents_fork/domains/swe_gym/seed_harness/
# captures the diff, applies it in an eval container, runs the 50-task smoke,
# and records the score. Hyperagents' generate_loop.py does the archive +
# parent-selection bookkeeping.
#
# This is the A2 ablation (DGM/Hyperagents-style baseline at 4B) — see Plan §3.4.
#
# Prereqs:
#   • vLLM serving Qwen3.5-4B on :8001 (use meta/scripts/serve_qwen35.sh)
#   • Hyperagents container built (cf. meta/hyperagents_fork/Dockerfile)
#   • Hyperagents Python deps installed in python3.11
#
# Usage:
#   meta/scripts/run_phase_h.sh                                # 5 iter, default 50 smoke
#   meta/scripts/run_phase_h.sh --iters 10 --eval-samples 30
#   meta/scripts/run_phase_h.sh --proposer claude              # asymmetric baseline (A2')
#   meta/scripts/run_phase_h.sh --dry-run                      # build container only

set -euo pipefail

ITERS=5
EVAL_SAMPLES=50          # smoke set size per generation
PROPOSER="qwen3.5-4b"    # 'qwen3.5-4b' (default) | 'claude'
OUTPUT="./outputs/phase_h"
DRY_RUN=0
VLLM_BASE="http://localhost:8001/v1"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
HA_DIR="$ROOT/meta/hyperagents_fork"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iters|--iterations) ITERS="$2"; shift 2 ;;
    --eval-samples)        EVAL_SAMPLES="$2"; shift 2 ;;
    --proposer)            PROPOSER="$2"; shift 2 ;;
    --output)              OUTPUT="$2"; shift 2 ;;
    --vllm-base)           VLLM_BASE="$2"; shift 2 ;;
    --dry-run)             DRY_RUN=1; shift ;;
    -h|--help)             sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$HA_DIR"
mkdir -p "$OUTPUT"

# ---- 1. Sanity ----
echo "[phase_h] sanity: vLLM serving on $VLLM_BASE"
curl -sf --max-time 5 "$VLLM_BASE/models" > /dev/null \
  || { echo "[phase_h] vLLM not reachable. Run meta/scripts/serve_qwen35.sh first." >&2; exit 1; }

echo "[phase_h] sanity: swe_gym domain imports"
python3.11 -c "
import sys; sys.path.insert(0,'.')
from domains.swe_gym import seed_harness, task_agent
files = ['agent.py','model.py','environment.py','prompts.yaml']
total = sum(sum(1 for _ in open(f'domains/swe_gym/seed_harness/{n}')) for n in files)
print(f'  surface lines: {total} ({len(files)} files)')
print('  TaskAgent class:', task_agent.TaskAgent.__name__)
"

# ---- 2. Configure the meta_agent's model endpoint ----
case "$PROPOSER" in
  qwen3.5-4b)
    export OPENAI_API_BASE="$VLLM_BASE"
    export OPENAI_API_KEY="dummy"
    META_MODEL="qwen3.5-4b"
    ;;
  claude)
    [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "set ANTHROPIC_API_KEY for --proposer claude" >&2; exit 1; }
    META_MODEL="anthropic/claude-sonnet-4-5-20250929"
    ;;
  *)
    echo "unknown proposer: $PROPOSER (use qwen3.5-4b or claude)" >&2
    exit 2 ;;
esac
echo "[phase_h] meta_agent model: $META_MODEL  (proposer = $PROPOSER)"

if [ "$DRY_RUN" = "1" ]; then
  echo "[phase_h] --dry-run: would invoke generate_loop.py with N=$ITERS, eval_samples=$EVAL_SAMPLES"
  echo "         output_dir: $OUTPUT"
  exit 0
fi

# ---- 3. Hand off to generate_loop.py ----
# Hyperagents' outer loop owns archive, parent selection, container patching, scoring.
# Our patches to generate_loop.py (run_harness_swe_gym + dispatch) make this invocation
# self-contained for the swe_gym domain.
echo "[phase_h] launching Hyperagents generate_loop.py ..."
python3.11 -m generate_loop \
  --domains swe_gym \
  --eval_samples "$EVAL_SAMPLES" \
  --max_generation "$ITERS" \
  --output_dir "$OUTPUT" \
  --model "$META_MODEL" \
  ${PHASE_H_EXTRA_ARGS:-}

echo "[phase_h] done. Archive + per-gen scores at: $OUTPUT"
