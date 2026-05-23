"""
Phase 0 baseline runner.

Qwen3.5-4B (served by vLLM on port 8001) + unmodified mini-swe-agent harness
on SWE-Gym-Lite, sliced subset. Records pass@1, average steps, cost, format errors.

This is a thin wrapper that:
  1. Patches mini-swe-agent's docker-image-name helper to use the xingyaoww
     SWE-Gym namespace instead of the swebench namespace.
  2. Reuses mini's batch runner (run/benchmarks/swebench.py) with vLLM endpoint
     configured via env / config overrides.

Usage:
  python3.11 -m meta.phase0.run_phase0 --slice 0:5   # smoke
  python3.11 -m meta.phase0.run_phase0 --slice 0:50  # full Phase 0
"""

import os
import sys
import argparse
from pathlib import Path

# ---- 1. Patch mini's docker-image helper for SWE-Gym/xingyaoww namespace ----
import minisweagent.run.benchmarks.swebench as sb_module

def _swe_gym_image_name(instance: dict) -> str:
    explicit = instance.get("image_name") or instance.get("docker_image")
    if explicit:
        return explicit
    iid = instance["instance_id"]
    id_compat = iid.replace("__", "_s_").lower()
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{id_compat}:latest"

sb_module.get_swebench_docker_image_name = _swe_gym_image_name

# Patch the DATASET_MAPPING to add SWE-Gym short keys
sb_module.DATASET_MAPPING["swegym_lite"] = "SWE-Gym/SWE-Gym-Lite"
sb_module.DATASET_MAPPING["swegym"]      = "SWE-Gym/SWE-Gym"


# ---- 2. Default output and config paths ----
REPO_ROOT  = Path(__file__).resolve().parents[2]
OUT_ROOT   = Path("/home/t-hyunlee/meta-harness-plan/results/phase0")
CONFIG_DIR = Path(__file__).parent / "configs"


def main():
    parser = argparse.ArgumentParser(description="Phase 0 baseline runner")
    parser.add_argument("--subset", default="swegym_lite",
                        help="Dataset short key or HF path (default: swegym_lite)")
    parser.add_argument("--split", default="train",
                        help="Dataset split (SWE-Gym-Lite only has 'train')")
    parser.add_argument("--slice", default="0:5",
                        help="Python slice over instance list (default: 0:5 smoke)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel docker workers (default 4)")
    parser.add_argument("--output", default=None,
                        help="Output directory (defaults to results/phase0/<slice>)")
    parser.add_argument("--vllm-base", default="http://localhost:8001/v1",
                        help="vLLM OpenAI-compatible base URL")
    parser.add_argument("--model-name", default="hosted_vllm/qwen3.5-4b",
                        help="litellm model name (provider prefix matters)")
    parser.add_argument("--config",
                        default=str(Path(sb_module.builtin_config_dir) / "benchmarks" / "swebench_backticks.yaml"),
                        help="mini config yaml. swebench_backticks.yaml uses text-based bash blocks (recommended for vLLM-served open models without tool-call training).")
    parser.add_argument("--model-class",
                        default="minisweagent.models.litellm_textbased_model.LitellmTextbasedModel",
                        help="model class. textbased class parses bash blocks from raw text instead of tool_calls.")
    args, extra = parser.parse_known_args()

    output = args.output or str(OUT_ROOT / args.slice.replace(":", "_"))
    Path(output).mkdir(parents=True, exist_ok=True)

    # vLLM is OpenAI-compatible; LiteLLM uses provider 'hosted_vllm/' prefix
    # See https://docs.litellm.ai/docs/providers/vllm
    os.environ["HOSTED_VLLM_API_BASE"] = args.vllm_base
    os.environ["HOSTED_VLLM_API_KEY"]  = "dummy"   # vLLM open mode

    # Reassemble argv for mini's typer app
    sys.argv = [
        "swebench",
        "--subset", args.subset,
        "--split",  args.split,
        "--slice",  args.slice,
        "--workers", str(args.workers),
        "--model",  args.model_name,
        "--model-class", args.model_class,
        "-o", output,
        "-c", args.config,
        # Override step + cost limits for cheap Phase 0; vLLM cost is irrelevant
        "-c", "agent.cost_limit=0",
        "-c", "agent.step_limit=120",
        "-c", "model.cost_tracking=ignore_errors",
        "-c", "model.model_kwargs.max_tokens=4096",
        # Qwen3.5 model-card recommended sampling for "Thinking mode, PRECISE
        # CODING tasks" — the inner agent is a precise-coding agent
        # (edit source files to pass benchmark tests). This is the per-task
        # recommendation that differs from the outer meta-agent's
        # "Thinking mode, general tasks" preset (in run_meta_agent_v2.py).
        #   temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        #   presence_penalty=0.0, repetition_penalty=1.0
        # top_k / min_p / repetition_penalty go via extra_body (vLLM-specific;
        # LiteLLM forwards them transparently for hosted_vllm provider).
        "-c", "model.model_kwargs.temperature=0.6",
        "-c", "model.model_kwargs.top_p=0.95",
        "-c", "model.model_kwargs.presence_penalty=0.0",
        "-c", 'model.model_kwargs.extra_body={"top_k":20,"min_p":0.0,"repetition_penalty":1.0}',
        *extra,
    ]
    sb_module.app()


if __name__ == "__main__":
    main()
