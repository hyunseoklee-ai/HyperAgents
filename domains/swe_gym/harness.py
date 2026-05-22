"""Batch harness entrypoint for the swe_gym domain.

Hyperagents' outer loop (`generate_loop.py`) calls a per-domain `harness()`
function. For SWE-Gym we keep this thin — delegate the actual N-task batch
execution to mini-swe-agent's existing infrastructure (via `meta.phase0.run_phase0`),
then return the output directory holding `preds.json`.

This matches Path C in the project plan: Hyperagents owns the outer loop and
archive bookkeeping; mini-swe-agent owns the inner per-task agent loop.
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path


# Project root contains mini-swe-agent + meta/. Make sure modules resolve.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def harness(
    agent_path: str | None = None,           # honored for compat; ignored when batch=True
    output_dir: str | Path = "./outputs",
    run_id: str | None = None,
    subset: str = "lite",                    # 'lite' -> SWE-Gym-Lite, 'full' -> SWE-Gym
    num_samples: int = -1,                   # not used; batch uses --slice
    num_workers: int = 4,
    slice_spec: str = "0:50",
    model_name: str = "hosted_vllm/qwen3.5-4b",
    config: str = "swebench_backticks.yaml",
    step_limit: int = 120,
    vllm_base: str = "http://localhost:8001/v1",
) -> str:
    """Run the batch and return the absolute path to the output dir holding preds.json."""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir).resolve() / run_id
    out.mkdir(parents=True, exist_ok=True)

    subset_key = "swegym_lite" if subset == "lite" else "swegym"

    cmd = [
        sys.executable, "-m", "meta.phase0.run_phase0",
        "--subset",  subset_key,
        "--split",   "train",
        "--slice",   slice_spec,
        "--workers", str(num_workers),
        "--model-name", model_name,
        "--vllm-base",  vllm_base,
        "--output",     str(out),
    ]
    env = dict(os.environ)
    # The wrapper script in meta.phase0.run_phase0 hardcodes the step_limit;
    # if you need to override it, edit that file or pass it through here later.

    print("[swe_gym/harness]", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode
    if rc != 0:
        raise RuntimeError(f"mini-swe-agent batch runner exited with rc={rc}")
    return str(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", default=None,
                   help="Compatibility flag (Hyperagents passes this; ignored in batch mode)")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--run_id",     default=None)
    p.add_argument("--subset",     default="lite", choices=["lite", "full"])
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--slice",       default="0:50", dest="slice_spec")
    p.add_argument("--model_name",  default="hosted_vllm/qwen3.5-4b")
    p.add_argument("--config",      default="swebench_backticks.yaml")
    p.add_argument("--step_limit",  type=int, default=120)
    p.add_argument("--vllm_base",   default="http://localhost:8001/v1")
    args = p.parse_args()
    print("output dir:", harness(**vars(args)))


if __name__ == "__main__":
    main()
