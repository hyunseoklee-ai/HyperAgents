"""SWE-Gym shared utilities — used by both Hyperagents' harness.py contract
and our batch runner / scorer."""

import ast
import os
from datasets import load_dataset


# Hyperagents' `domains/harness.py` looks up these module-level constants.
QUESTION_ID = "instance_id"
MODEL = os.environ.get("MSWEA_MODEL_NAME", "hosted_vllm/qwen3.5-4b")


def docker_image_for(iid: str) -> str:
    """SWE-Gym's prebuilt images live under the xingyaoww/ namespace with `_s_`
    as the instance-id separator (cf. SWE-bench's `_1776_`).

    Example: getmoto__moto-5752  ->
      docker.io/xingyaoww/sweb.eval.x86_64.getmoto_s_moto-5752:latest
    """
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{iid.replace('__','_s_').lower()}:latest"


def load_swe_gym(subset: str = "lite", split: str = "train"):
    """Load the SWE-Gym dataset from HuggingFace.

    subset='lite' loads SWE-Gym/SWE-Gym-Lite (230 instances)
    subset='full' loads SWE-Gym/SWE-Gym (~2.4k instances)
    """
    name = "SWE-Gym/SWE-Gym-Lite" if subset == "lite" else "SWE-Gym/SWE-Gym"
    return load_dataset(name, split=split)


def parse_test_list(field) -> list[str]:
    """SWE-Gym stores PASS_TO_PASS / FAIL_TO_PASS as a repr'd python list string."""
    if not field:
        return []
    if isinstance(field, list):
        return field
    try:
        return ast.literal_eval(field)
    except Exception:
        return []


def format_input_dict(row: dict) -> dict:
    """Pack a SWE-Gym row into the dict TaskAgent.forward() consumes."""
    return {
        "domain": "swe_gym",
        "instance_id":       row["instance_id"],
        "problem_statement": row["problem_statement"],
        "repo":              row["repo"],
        "base_commit":       row["base_commit"],
        "version":           row["version"],
        "image":             docker_image_for(row["instance_id"]),
        "fail_to_pass":      parse_test_list(row.get("FAIL_TO_PASS")),
        "pass_to_pass":      parse_test_list(row.get("PASS_TO_PASS")),
        "test_patch":        row.get("test_patch") or "",
    }
