# swe_gym / seed_harness

This directory is the **initial archive seed (candidate 000)** AND the
**mutable surface** (Phase H) for the SWE-Gym domain in our Hyperagents fork.

## Mutation surface

| File | Lines | Origin |
|---|---:|---|
| `agent.py` | 171 | mini-swe-agent `src/minisweagent/agents/default.py` |
| `model.py` | 45 | mini-swe-agent `src/minisweagent/models/litellm_textbased_model.py` |
| `environment.py` | 161 | mini-swe-agent `src/minisweagent/environments/docker.py` |
| `prompts.yaml` | 244 | mini-swe-agent `src/minisweagent/config/benchmarks/swebench_backticks.yaml` |
| **Total** | **621** | — |

`domains/swe_gym/task_agent.py` imports `DefaultAgent`, `LitellmTextbasedModel`,
and `DockerEnvironment` from this package's `__init__.py`, so per-iteration
meta-agent edits take effect.

The files still import stable utilities (`minisweagent.exceptions`,
`minisweagent.utils.serialize`, etc.) from the installed package — we
intentionally did NOT copy those, because they are not the interesting
mutation surface.

## Identity

| Field | Value |
|---|---|
| `id` | 000_baseline |
| `parent_id` | (none) |
| `origin` | mini-swe-agent v2.3.0 (commit at fork time) |
| `prompt_config` | `src/minisweagent/config/benchmarks/swebench_backticks.yaml` |
| `model_class` | `minisweagent.models.litellm_textbased_model.LitellmTextbasedModel` |
| `inner_model` | Qwen/Qwen3.5-4B served by vLLM 0.21 at :8001 (max_model_len 131072) |
| `step_limit` | 120 |
| `cost_limit` | 0 (vLLM is free) |
| `env_class` | `minisweagent.environments.docker.DockerEnvironment` |
| `docker_namespace` | xingyaoww (SWE-Gym pre-built images) |

## Phase 0 baseline (recorded 2026-05-22)

| Metric | v1 (32K ctx) | v2 (128K ctx) |
|---|---|---|
| Submitted | 18 / 50 | (pending) |
| Context-exceeded | 29 / 50 | (pending) |
| Step-limit | 3 / 50 | (pending) |
| **pass@1 over Submitted** | **11 / 18 (61.1%)** | (pending) |
| **pass@1 over total** | **11 / 50 (22.0%)** | (pending) |

## Notes

This seed is **not** a runtime copy of mini-swe-agent's source code; mini is
installed as an editable package and the seed identifies it by version + config
file paths. When the outer loop mutates the harness it produces a new candidate
directory (`candidates/001_*/`) that may either:
  (a) reference a different config file under the mini package, or
  (b) carry its own overlay (a diff against the seed) that is applied at startup.

That design decision will be revisited when the outer loop is first wired up
(Phase 3 of the project plan, §3.3 of the research-plan webpage).
