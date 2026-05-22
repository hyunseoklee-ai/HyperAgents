"""TaskAgent for SWE-Gym (Hyperagents contract) that wraps mini-swe-agent.

The class name MUST be `TaskAgent` per Hyperagents' `domains/harness.load_task_agent()`.
Its `.forward(inputs)` method takes the dict produced by
`domains.swe_gym.utils.format_input_dict()` and returns
  (prediction_str, msg_history_list)
where prediction_str is the unified-diff patch (model_patch) the agent produced.

The outer loop (or batch harness) invokes this per-instance.
"""

import os
import sys
from pathlib import Path
import yaml

# Hyperagents base class (chat_history_file + log handle)
from agent.base_agent import AgentSystem

# Mutable agent surface (Phase H): we load from the LOCAL seed_harness package so
# the meta_agent's edits actually take effect. Each per-iteration patch becomes
# a diff against these files.
from .seed_harness import DefaultAgent, DockerEnvironment, LitellmTextbasedModel

_DEFAULT_CONFIG_FILE = Path(__file__).resolve().parent / "seed_harness" / "prompts.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


class TaskAgent(AgentSystem):
    """Hyperagents-facing wrapper. Per-instance docker, vLLM-backed Qwen3.5-4B,
    mini's text-based backticks parser."""

    def __init__(self, model=None, chat_history_file="./outputs/chat_history.md",
                 config_path: Path | str | None = None,
                 step_limit: int = 120, cost_limit: float = 0.0):
        super().__init__(model=model or "hosted_vllm/qwen3.5-4b",
                         chat_history_file=chat_history_file)
        self._config = _load_yaml(Path(config_path or _DEFAULT_CONFIG_FILE))
        self._step_limit = int(step_limit)
        self._cost_limit = float(cost_limit)

        # vLLM-OpenAI compatible endpoint via litellm hosted_vllm provider
        os.environ.setdefault("HOSTED_VLLM_API_BASE", "http://localhost:8001/v1")
        os.environ.setdefault("HOSTED_VLLM_API_KEY",  "dummy")

    def forward(self, inputs):
        iid     = inputs["instance_id"]
        image   = inputs["image"]
        problem = inputs["problem_statement"]

        # Build mini's runtime objects from the loaded config
        env_cfg   = dict(self._config.get("environment", {}))
        model_cfg = dict(self._config.get("model", {}))
        agent_cfg = dict(self._config.get("agent", {}))

        # Per-call overrides
        env_cfg["image"] = image
        env_cfg.setdefault("environment_class", "docker")
        agent_cfg["step_limit"] = self._step_limit
        agent_cfg["cost_limit"] = self._cost_limit

        model = LitellmTextbasedModel(model_name=self.model, **model_cfg)
        env   = DockerEnvironment(**env_cfg)
        agent = DefaultAgent(model, env, **agent_cfg)

        try:
            result = agent.run(task=problem)
        except Exception as e:
            self.log(f"TaskAgent.forward error: {type(e).__name__}: {e}")
            return "", agent.messages

        # The submission is the full git diff captured by mini's environment
        # _check_finished hook (everything after the COMPLETE_TASK sentinel line).
        patch = ""
        if isinstance(result, dict):
            patch = result.get("submission", "") or ""

        return patch, agent.messages
