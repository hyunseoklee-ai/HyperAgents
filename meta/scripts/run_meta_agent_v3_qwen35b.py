"""
Phase H meta-agent v3 — thin wrapper around run_meta_agent_v2 that swaps in a
stronger open-source proposer: Qwen3.5-35B-A3B-GPTQ-Int4 (MoE, 3B active, Int4).

Holds the DGM-style ~200-LOC scaffold FIXED (so the comparison with A2 isolates
proposer model strength). Adds two reasoning-aware tweaks via env-vars that
v2.py honors:
  META_MAX_TOKENS=8000        # reasoning models burn budget on hidden <think>
  META_REASONING_HINT=<text>  # appended to the system prompt

This script does not touch v2's docker invocation, edit-prompt helper, submit
gate, or empty-diff push-back. Identical inner protocol — only the proposer
endpoint and the system prompt addendum differ.

Usage:
  # assumes 35B server is already up on :8000 (use serve_qwen35b_a3b.sh)
  python3.11 -m meta.scripts.run_meta_agent_v3_qwen35b \\
      --gen 1 --no-eval                         # smoke
  python3.11 -m meta.scripts.run_meta_agent_v3_qwen35b \\
      --gen 1                                   # full: propose + eval
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/t-hyunlee/mini-swe-agent")
PHASE_H_OUT = Path("/home/t-hyunlee/meta-harness-plan/results/phase_h")

REASONING_HINT = """\
NOTE FOR REASONING MODELS:
  - Your `<think>...</think>` content is consumed by the vLLM reasoning parser
    server-side and is NEVER shown to the assistant on subsequent turns. Treat
    `<think>` as scratch-only.
  - The bash fence (```bash ... ```) MUST appear in the *visible* `content`
    field, AFTER any `<think>` block. If you only think and emit no bash
    block, the meta-agent treats this as a no-op and stops.
  - Budget for hidden reasoning is generous (max_tokens=8000) but do not
    spend more than half of it on `<think>` per turn — reserve room for the
    actual bash command.
  - One focused edit. Then exit with the sentinel.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=1,
                    help="Generation index; outputs land at gen_{N}_qwen35b/")
    ap.add_argument("--eval-samples", type=int, default=20)
    ap.add_argument("--endpoint", default="http://localhost:8000/v1",
                    help="OpenAI-compatible endpoint for the 35B proposer")
    ap.add_argument("--model", default="qwen3.5-35b-a3b",
                    help="Served model name on the proposer vLLM")
    ap.add_argument("--max-steps", type=int, default=25,
                    help="Reasoning model often needs ~25 turns vs Qwen-4B's 20")
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="Generous budget to allow <think> + bash fence")
    ap.add_argument("--no-eval", action="store_true",
                    help="Only propose; skip the 20-task inner eval")
    args = ap.parse_args()

    out_dir = PHASE_H_OUT / f"gen_{args.gen}_qwen35b"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proposer_info.txt").write_text(
        f"model: {args.model}\nendpoint: {args.endpoint}\n"
        f"max_steps: {args.max_steps}\nmax_tokens: {args.max_tokens}\n"
    )

    os.environ["META_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["META_REASONING_HINT"] = REASONING_HINT
    os.environ["PHASE_H_OUT_OVERRIDE"] = str(out_dir)

    sys.path.insert(0, str(PROJECT_ROOT))
    from meta.scripts import run_meta_agent_v2 as v2  # noqa: E402

    v2.PHASE_H_OUT = out_dir.parent
    sys.argv = [
        "run_meta_agent_v2",
        "--gen", str(args.gen),
        "--eval-samples", str(args.eval_samples),
        "--endpoint", args.endpoint,
        "--model", args.model,
        "--max-steps", str(args.max_steps),
    ]
    if args.no_eval:
        sys.argv.append("--no-eval")

    # Redirect v2's "gen_{N}" output to gen_{N}_qwen35b WITHOUT touching the
    # real gen_{N} (which holds the Qwen-4B A2 result). Use a private overlay
    # parent containing only `gen_{N}` -> our out_dir, plus a symlink to the
    # real gen_0 for archive building.
    overlay = Path("/tmp/meta_harness_phase_h_overlay_qwen35b")
    if overlay.exists():
        # clean any stale state
        for p in overlay.iterdir():
            if p.is_symlink() or p.is_file():
                p.unlink()
            else:
                import shutil
                shutil.rmtree(p)
    else:
        overlay.mkdir(parents=True, exist_ok=True)

    # gen_{N} -> gen_{N}_qwen35b (where v2 writes)
    (overlay / f"gen_{args.gen}").symlink_to(out_dir)
    # gen_0..gen_{N-1} -> real ones (read-only, for archive building)
    real_phase_h = Path("/home/t-hyunlee/meta-harness-plan/results/phase_h")
    for g in range(args.gen):
        real = real_phase_h / f"gen_{g}"
        if real.exists():
            (overlay / f"gen_{g}").symlink_to(real)

    v2.PHASE_H_OUT = overlay
    v2.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
