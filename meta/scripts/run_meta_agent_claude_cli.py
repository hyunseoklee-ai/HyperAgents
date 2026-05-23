"""
Phase H meta-agent — Claude Code CLI variant (A2' asymmetric ablation, fixed).

Why this exists:
  Direct Anthropic SDK calls (run_meta_agent_claude*.py) hit OAuth tier limits
  on sonnet/opus models — the user's claude.ai sub is gated for those routes.
  The Claude Code CLI (`claude -p ...`) uses a DIFFERENT routing pool that
  honors the Pro/Max subscription. Same OAuth token, different door.

  Claude Code is itself an agent — it already has bash + Read + Write + Edit
  tools natively. So we just feed it a prompt and let it edit
  /workspace/prompts.yaml directly, then capture the diff.

Usage:
    meta/scripts/run_meta_agent_claude_cli.py --gen 1 --model sonnet --no-eval
    meta/scripts/run_meta_agent_claude_cli.py --gen 1 --model sonnet --eval-samples 20 --budget 5
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/t-hyunlee/mini-swe-agent")
SEED         = PROJECT_ROOT / "meta/hyperagents_fork/domains/swe_gym/seed_harness"
PHASE_H_OUT  = Path("/home/t-hyunlee/meta-harness-plan/results/phase_h")
ENV_FILE     = Path("/home/t-hyunlee/meta-harness-plan/.env")

sys.path.insert(0, str(PROJECT_ROOT))
from meta.scripts.run_meta_agent_v2 import (
    copy_seed, build_archive, get_prompts_yaml_structure,
    import_check, evaluate,
)


def read_env(p: Path) -> dict[str, str]:
    if not p.exists(): return {}
    return {k.strip(): v.strip().strip('"').strip("'")
            for l in p.read_text().splitlines() if "=" in l and not l.lstrip().startswith("#")
            for k, v in [l.split("=", 1)]}


def install_helper(candidate_dir: Path):
    """Install the edit-prompt helper into <candidate>/bin/ and prepend to PATH
    via a small wrapper. Claude Code in cwd=<candidate> can then call `edit-prompt`."""
    bin_dir = candidate_dir / "bin"
    bin_dir.mkdir(exist_ok=True)
    helper = (bin_dir / "edit-prompt")
    helper.write_text(r"""#!/usr/bin/env python3
import sys
from ruamel.yaml import YAML
if len(sys.argv) < 4 or sys.argv[1] in ('-h','--help'):
    print('usage: edit-prompt KEY ACTION VALUE'); sys.exit(2)
key, action, value = sys.argv[1], sys.argv[2], sys.argv[3]
y = YAML(); y.preserve_quotes = True; y.width = 999999
prompts_path = '%s'
with open(prompts_path) as f: d = y.load(f)
parts = key.split('.')
ref = d
for p in parts[:-1]:
    if p not in ref: print(f'ERROR: {p!r} not found'); sys.exit(3)
    ref = ref[p]
leaf = parts[-1]
if leaf not in ref: print(f'ERROR: {leaf!r} not found'); sys.exit(3)
current = ref[leaf]
if not isinstance(current, str):
    print(f'ERROR: {key} not a string'); sys.exit(3)
if action == 'append':
    ref[leaf] = current.rstrip() + '\n' + value.rstrip() + '\n'
elif action == 'prepend':
    ref[leaf] = value.rstrip() + '\n' + current
elif action == 'replace_substring':
    if '|||' not in value: print('ERROR: VALUE must be "OLD|||NEW"'); sys.exit(3)
    old, new = value.split('|||', 1)
    if old not in current: print(f'ERROR: substring not found'); sys.exit(3)
    ref[leaf] = current.replace(old, new)
elif action == 'set':
    ref[leaf] = value
else:
    print(f'ERROR: unknown action'); sys.exit(3)
with open(prompts_path, 'w') as f: y.dump(d, f)
import subprocess as sp
diff = sp.run(['git', '-C', '%s', 'diff', 'prompts.yaml'],
              capture_output=True, text=True).stdout
print(f'OK: edited {key} (action={action}); diff is now {len(diff)} bytes')
""" % (str(candidate_dir / "prompts.yaml"), str(candidate_dir)))
    helper.chmod(0o755)


def run_claude_cli(candidate_dir: Path, archive_dir: Path, output_dir: Path,
                   model: str, budget_usd: float, max_turns: int,
                   auth_token: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # init git in candidate so claude's edits land as diff
    subprocess.run(["git", "init", "-q"], cwd=candidate_dir, check=False)
    subprocess.run(["git", "config", "user.email", "x@x"], cwd=candidate_dir, check=False)
    subprocess.run(["git", "config", "user.name", "x"], cwd=candidate_dir, check=False)
    subprocess.run(["git", "add", "-A"], cwd=candidate_dir, check=False)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=candidate_dir, check=False)

    install_helper(candidate_dir)

    # system prompt
    sys_prompt = textwrap.dedent(f"""\
        You are the OUTER meta-agent for a Phase H (harness-only evolve) experiment.

        Your single task: edit `prompts.yaml` in the current directory so the INNER
        agent (Qwen3.5-4B inside mini-swe-agent) solves more SWE-Gym tasks.

        Context:
          • cwd: candidate harness directory (RW).
          • {str(archive_dir)} (RO): prior generations' source + scores + trajectories.

        Edits:
          • Edit ONLY `prompts.yaml` (other files have no effect through the eval path).
          • Allowed keys (any nested YAML string):
              agent.system_template
              agent.instance_template
              model.observation_template
              model.format_error_template
          • Make at most ONE focused, hypothesis-driven edit.
          • Prefer the helper `./bin/edit-prompt KEY ACTION VALUE` (atomic
            ruamel.yaml write) over direct file rewrites.
          • You may also use your built-in Edit tool, but the helper is safer
            against YAML indentation / quoting issues.

        Workflow:
          1. Read prior generations' summary + scores + a few failing trajectories.
          2. Form ONE testable hypothesis.
          3. Apply the edit; verify with `git diff prompts.yaml`.
          4. Stop. Your final assistant message should briefly state the change
             and the hypothesis.

        Constraints:
          • Do NOT change `mswea_bash_command` or `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
            — those are protocol constants used by the inner agent.
          • Budget: ${budget_usd} USD, {max_turns} turns. Stop early when confident.
        """)

    # one-shot user prompt — Claude Code will navigate from here using its
    # built-in Read/Bash/Edit tools
    prompts_struct = get_prompts_yaml_structure(candidate_dir / "prompts.yaml")
    archive_listing = ""
    if archive_dir.exists():
        for gen in sorted(archive_dir.iterdir()):
            if gen.is_dir():
                ev = gen / "eval"
                n_traj = sum(1 for _ in ev.iterdir()) if ev.exists() else 0
                archive_listing += f"  {archive_dir.name}/{gen.name}/   ({n_traj} eval trajectories)\n"

    user_prompt = textwrap.dedent(f"""\
        Improve `prompts.yaml` to lift the inner agent's SWE-Gym pass@1.

        Read first:
          - {archive_dir}/gen_0/summary.txt
          - {archive_dir}/gen_0/scores.json
          - 2-3 failing trajectories at {archive_dir}/gen_0/eval/<iid>/<iid>.traj.json

        prompts.yaml schema (current):
        {prompts_struct}

        Apply ONE focused edit (preferably via `./bin/edit-prompt KEY ACTION VALUE`)
        to one of the four allowed keys. Then verify with `git diff prompts.yaml`.
        """)

    # invoke claude CLI
    cmd = [
        "claude", "-p", user_prompt,
        "--system-prompt", sys_prompt,
        "--model", model,
        "--output-format", "json",
        "--max-budget-usd", str(budget_usd),
        "--max-turns", str(max_turns),
        "--add-dir", str(archive_dir),
        "--permission-mode", "bypassPermissions",
        "--allow-dangerously-skip-permissions",
    ]
    env = os.environ.copy()
    env["ANTHROPIC_AUTH_TOKEN"] = auth_token

    print(f"[meta] invoking: claude -p (model={model}, budget=${budget_usd}, turns={max_turns})")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(candidate_dir), env=env,
                       capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    (output_dir / "claude_cli.stdout.log").write_text(r.stdout)
    (output_dir / "claude_cli.stderr.log").write_text(r.stderr)
    print(f"[meta] claude CLI returned in {elapsed:.0f}s (rc={r.returncode})")

    parsed = {}
    try:
        # CLI output is one JSON per line; the final 'result' object is what we want
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "result":
                    parsed = obj
                    break
            except json.JSONDecodeError:
                pass
        if not parsed:
            # fallback — try to parse whole stdout
            parsed = json.loads(r.stdout)
    except Exception as e:
        parsed = {"parse_error": str(e), "stdout_head": r.stdout[:500]}
    (output_dir / "claude_cli_result.json").write_text(json.dumps(parsed, indent=2))

    # capture diff
    diff = subprocess.run(["git", "-C", str(candidate_dir), "diff", "prompts.yaml"],
                          capture_output=True, text=True).stdout
    (output_dir / "diff.patch").write_text(diff)

    result_summary = {
        "ok": r.returncode == 0,
        "diff_bytes": len(diff),
        "elapsed_sec": round(elapsed, 1),
        "model_in_use": parsed.get("modelUsage", {}),
        "total_cost_usd": parsed.get("total_cost_usd"),
        "num_turns": parsed.get("num_turns"),
        "stop_reason": parsed.get("stop_reason"),
        "result_text": (parsed.get("result") or "")[:1500],
    }
    return result_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=1)
    ap.add_argument("--model", default="sonnet",
                    help="Claude Code model alias: sonnet | haiku | opus")
    ap.add_argument("--budget", type=float, default=3.0, help="--max-budget-usd")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--eval-samples", type=int, default=20)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--suffix", default="claude_cli")
    args = ap.parse_args()

    env = read_env(ENV_FILE)
    token = env.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not token:
        print(f"[main] no ANTHROPIC_AUTH_TOKEN in {ENV_FILE}"); sys.exit(2)

    gen_dir = PHASE_H_OUT / f"gen_{args.gen}_{args.suffix}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    candidate = gen_dir / "seed_harness"
    copy_seed(candidate)

    prior_gens = {}
    for g in range(args.gen):
        d = PHASE_H_OUT / f"gen_{g}"
        if d.exists():
            prior_gens[g] = d
    archive = gen_dir / "_archive_mount"
    build_archive(prior_gens, archive)
    print(f"[main] archive with {len(prior_gens)} prior gens")

    res = run_claude_cli(candidate, archive, gen_dir,
                         model=args.model, budget_usd=args.budget,
                         max_turns=args.max_turns, auth_token=token)
    (gen_dir / "meta_agent_result.json").write_text(json.dumps(res, indent=2))
    print(f"[meta] result: ok={res['ok']} diff_bytes={res['diff_bytes']} "
          f"turns={res['num_turns']} cost=${res['total_cost_usd']}")

    ok, msg = import_check(candidate)
    if not ok:
        print("[meta] D26 import_check FAILED")
        (gen_dir / "rejected.txt").write_text(msg[:2000]); return
    print("[meta] D26 import_check OK")

    if args.no_eval:
        print("[meta] --no-eval; stopping"); return

    print(f"=== eval {gen_dir.name} on {args.eval_samples} tasks ===")
    out = evaluate(candidate, args.eval_samples, gen_dir)
    (gen_dir / "eval_results.json").write_text(json.dumps(out, indent=2))
    print(f"[eval] {out}")


if __name__ == "__main__":
    main()
