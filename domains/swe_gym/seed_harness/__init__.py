"""seed_harness: the mutable surface of our SWE-Gym agent.

This is the package the meta-agent edits during Phase H (harness-only evolve).
Following Hyperagents' convention, the agent code lives here as plain Python
files so it is git-tracked and patchable. Per-iteration mutations are captured
as diffs against this directory and applied in eval containers.

Files exposed:
  agent.py       — DefaultAgent loop (171 lines)
  model.py       — LitellmTextbasedModel parser/serializer (45 lines)
  environment.py — DockerEnvironment per-action runner (161 lines)
  prompts.yaml   — system + instance + observation templates (244 lines)

Total mutable surface: 621 lines.

The four files copy mini-swe-agent v2.3.0 (commit at fork time) verbatim. They
still rely on the installed `minisweagent` package for stable utilities
(exceptions, serialize, etc.), but the LOGIC files are local and mutable.
"""

from .agent       import DefaultAgent
from .model       import LitellmTextbasedModel
from .environment import DockerEnvironment

__all__ = ["DefaultAgent", "LitellmTextbasedModel", "DockerEnvironment"]
