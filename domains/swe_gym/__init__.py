"""SWE-Gym domain for Hyperagents (Path C adapter wrapping mini-swe-agent).

This domain delegates per-task execution to mini-swe-agent's existing batch
runner and uses the xingyaoww/ docker namespace for SWE-Gym task containers.

Files:
  utils.py         — dataset loader, instance->image mapping, FAIL_TO_PASS parsing
  task_agent.py    — Hyperagents-compatible TaskAgent wrapping mini's DefaultAgent
  harness.py       — batch entrypoint (delegates to meta.phase0.run_phase0)
  report.py        — pass@1 scoring via meta.phase0.score_patches
  seed_harness/    — initial archive seed (mini-swe-agent as it ships)
"""
