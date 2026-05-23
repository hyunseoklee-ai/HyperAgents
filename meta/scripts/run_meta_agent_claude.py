"""
Phase H meta-agent — Claude variant (A2' asymmetric ablation).

Same architecture as run_meta_agent_v2.py:
  • Docker isolation
  • /workspace (RW) + /archive (RO) mounts
  • Pre-computed gen summary
  • edit-prompt helper (atomic ruamel.yaml writes)
  • Empty-diff gate

Differences:
  • Uses the Anthropic SDK with claude-sonnet-4-6 (or whatever --model passes)
  • Extended thinking enabled (--effort xhigh → 32K thinking budget)
  • Outputs to gen_<N>_claude/ to keep separate from the Qwen Phase H run

Usage:
    meta/scripts/run_meta_agent_claude.py --gen 1 --effort xhigh --no-eval
    meta/scripts/run_meta_agent_claude.py --gen 1 --effort xhigh --eval-samples 20
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

import ast

PROJECT_ROOT = Path("/home/t-hyunlee/mini-swe-agent")
SEED         = PROJECT_ROOT / "meta/hyperagents_fork/domains/swe_gym/seed_harness"
PHASE_H_OUT  = Path("/home/t-hyunlee/meta-harness-plan/results/phase_h")
ENV_FILE     = Path("/home/t-hyunlee/meta-harness-plan/.env")

EFFORT_BUDGET = {
    "low":    1024,
    "medium": 4096,
    "high":  16384,
    "xhigh": 32768,
}


# ---- reuse functions from v2 ----
sys.path.insert(0, str(PROJECT_ROOT))
from meta.scripts.run_meta_agent_v2 import (
    copy_seed,
    compute_gen_summary,
    build_archive,
    get_prompts_yaml_structure,
    import_check,
    evaluate,
)


def read_env(env_path: Path) -> dict[str, str]:
    out = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


META_AGENT_SCRIPT_CLAUDE = r"""
import os, re, subprocess, json, sys
from anthropic import Anthropic

api_key = os.environ.get('ANTHROPIC_API_KEY')          # sk-ant-api03-...
auth_token = os.environ.get('ANTHROPIC_AUTH_TOKEN')    # sk-ant-oat01-...
if not (api_key or auth_token):
    print('META: no ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in env'); sys.exit(2)

if auth_token and auth_token.startswith('sk-ant-oat'):
    client = Anthropic(auth_token=auth_token)
elif api_key:
    client = Anthropic(api_key=api_key)
else:
    # token of unknown shape — try auth_token first, fall through
    client = Anthropic(auth_token=auth_token)
MODEL = os.environ['META_MODEL']
MAX_STEPS = int(os.environ.get('META_MAX_STEPS','20'))
THINKING_BUDGET = int(os.environ.get('META_THINKING_BUDGET','32000'))
# max_tokens must be > thinking budget; add headroom for response
MAX_TOKENS = THINKING_BUDGET + 4000

FENCE_RE = re.compile(r'```(?:bash|sh)\s*\n(.*?)```', re.DOTALL)
# strict submit: a bash block whose effective content is ONLY `echo COMPLETE_TASK_...`
# (possibly with && or ;). Catches `echo CT...` but NOT `edit-prompt ... "...CT..."`
# where the sentinel is text inside an argument.
SUBMIT_RE = re.compile(r'(?ms)^(?:\s*echo\s+COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\s*(?:&&|;|$).*)+$')

with open('/system_prompt.txt') as f: SYS = f.read()
with open('/user_prompt.txt')   as f: USR = f.read()

messages = [{'role':'user','content':USR}]   # anthropic: system separate
trajectory = [{'role':'system','content':SYS}, {'role':'user','content':USR}]

def log_msg(label, text):
    head = text[:240].replace(chr(10),' / ')
    print(f'[meta][{label}] {head}', flush=True)

stopped_by = 'budget'
for step in range(MAX_STEPS):
    try:
        # streaming required for long-running thinking (>10 min budget)
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYS,
            thinking={'type':'enabled','budget_tokens':THINKING_BUDGET},
            messages=messages,
        ) as stream:
            for _ in stream.text_stream:
                pass
            final_msg = stream.get_final_message()
    except Exception as e:
        print(f'[meta] API error step {step}: {repr(e)[:400]}', flush=True)
        stopped_by = 'api_error'; break

    visible_text_parts = []
    thinking_chars = 0
    asst_blocks = []
    for blk in final_msg.content:
        if blk.type == 'thinking':
            thinking_chars += len(blk.thinking)
            asst_blocks.append({'type':'thinking','thinking':blk.thinking,
                                'signature':getattr(blk,'signature',None)})
        elif blk.type == 'text':
            visible_text_parts.append(blk.text)
            asst_blocks.append({'type':'text','text':blk.text})
    content_text = ''.join(visible_text_parts)
    log_msg(f'asst#{step} thinking={thinking_chars}c', content_text)

    # for next API call, must echo back the thinking+text blocks (anthropic requires it)
    messages.append({'role':'assistant','content':asst_blocks})
    trajectory.append({'role':'assistant','content':content_text,
                       'thinking_chars': thinking_chars})

    blocks = FENCE_RE.findall(content_text)
    if not blocks:
        print('[meta] no bash/sh block; stopping', flush=True)
        stopped_by = 'no_block'; break
    cmd = blocks[0].strip()
    # detect submit ONLY when the block is a clean `echo COMPLETE_TASK...` line —
    # not when the sentinel appears as text inside an edit-prompt argument
    is_submit = bool(SUBMIT_RE.match(cmd)) and 'edit-prompt' not in cmd
    if is_submit:
        cur_diff = subprocess.run(['git','-C','/workspace','diff','prompts.yaml'],
                                  capture_output=True, text=True).stdout
        if cur_diff.strip():
            print(f'[meta] agent done with diff ({len(cur_diff)} bytes)', flush=True)
            stopped_by = 'submit'; break
        retry = ('<output rc=0>(intercepted) Your git diff is empty. You have not '
                 'produced any edit yet. Apply at least one concrete edit to '
                 '/workspace/prompts.yaml using edit-prompt, verify with '
                 '`git -C /workspace diff prompts.yaml`, then exit.</output>')
        messages.append({'role':'user','content':retry})
        trajectory.append({'role':'user','content':retry})
        print('[meta] empty diff at submit; pushing back', flush=True)
        continue

    try:
        out = subprocess.run(['bash','-c',cmd], cwd='/workspace',
                             capture_output=True, text=True, timeout=120)
        obs = (out.stdout + out.stderr)[:4000]; ret = out.returncode
    except subprocess.TimeoutExpired:
        obs = '(bash command timed out after 120s)'; ret = -1
    log_msg(f'obs#{step} rc={ret}', obs)
    obs_msg = f'<output rc={ret}>\n{obs}\n</output>'
    messages.append({'role':'user','content':obs_msg})
    trajectory.append({'role':'user','content':obs_msg})

diff = subprocess.run(['git','-C','/workspace','diff'],
                      capture_output=True, text=True).stdout
print(f'[meta] final diff bytes: {len(diff)}; stopped_by: {stopped_by}', flush=True)

with open('/output/diff.patch','w') as f:    f.write(diff)
with open('/output/meta_trajectory.json','w') as f:
    f.write(json.dumps(trajectory, indent=2)[:400000])
with open('/output/meta_result.json','w') as f:
    f.write(json.dumps({'diff_bytes': len(diff), 'stopped_by': stopped_by,
                        'steps_used': step+1, 'model': MODEL,
                        'thinking_budget': THINKING_BUDGET}, indent=2))
"""


def run_meta_agent_in_docker(candidate_dir: Path, archive_dir: Path,
                             output_dir: Path,
                             model: str, api_key: str,
                             thinking_budget: int,
                             max_steps: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    sys_prompt = textwrap.dedent("""\
        You are the OUTER meta-agent.

        Your single task: edit `/workspace/prompts.yaml` so that the INNER agent
        (Qwen3.5-4B running with this prompts.yaml inside mini-swe-agent) solves
        more SWE-Gym tasks.

        # IMPORTANT — execution environment

        You are NOT in a function-calling environment.
        Anthropic tool-use tags such as `<function_calls>`, `<invoke ...>`,
        `<parameter ...>` are NOT supported here and will be SILENTLY IGNORED.
        Writing them does nothing.

        To execute a shell command you MUST write a markdown bash fence:

        ```bash
        your-command-here
        ```

        Only the FIRST such fenced block in your response is executed (cwd =
        /workspace). Output of that command comes back as the next user turn
        wrapped in <output rc=...>.

        Other hard rules:
          • NEVER write ```mswea_bash_command``` in your own response — that is
            the INNER agent's parser; producing that fence would result in
            "command not found" if it were executed.
          • Use the `edit-prompt` helper for all yaml edits (described below).
          • Stop after at most one targeted edit; verify with `git diff`.

        When you are done with a single focused edit, exit with EXACTLY this
        (a clean bash fence containing only the echo, no other commands):

        ```bash
        echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
        ```
        """)

    prompts_struct = get_prompts_yaml_structure(candidate_dir / "prompts.yaml")
    archive_listing = ""
    if archive_dir.exists():
        for gen in sorted(archive_dir.iterdir()):
            if gen.is_dir():
                ev = gen / "eval"
                n_traj = sum(1 for _ in ev.iterdir()) if ev.exists() else 0
                archive_listing += f"  /archive/{gen.name}/   ({n_traj} eval trajectories)\n"

    user_prompt = textwrap.dedent(f"""\
        # Filesystem

        /workspace/      RW    candidate harness — your edits go here
          ├── prompts.yaml          ← THE file you may edit
          ├── agent.py              ← do NOT edit (won't take effect)
          ├── model.py              ← do NOT edit
          └── environment.py        ← do NOT edit

        /archive/        RO    prior generations and their evaluation outputs
        {archive_listing}        Each generation dir contains:
          seed_harness/             prior agent code
          scores.json               {{instance_id → {{pass, f2p_*, p2p_*}}}}
          summary.txt               pre-computed pass rate + failure-class breakdown
          eval/<iid>/<iid>.traj.json     full inner-agent trajectory per task

        # prompts.yaml — what you may edit

        {prompts_struct}

        Allowed edit targets (keep yaml structure valid):
          agent.system_template
          agent.instance_template
          model.observation_template
          model.format_error_template

        # ⚠️ MUST produce a non-empty diff before exiting

        You MUST apply at least ONE concrete edit to /workspace/prompts.yaml
        and verify it shows up in `git diff` before sending the exit sentinel.
        Empty submissions are intercepted.

        # Recommended workflow

        1. `cat /archive/gen_0/summary.txt`            — orient
        2. `cat /archive/gen_0/scores.json | python3 -m json.tool | head -60`
        3. `ls /archive/gen_0/eval/ | head -20`
        4. Read 2-3 failing trajectories — e.g.
             `tail -c 8000 /archive/gen_0/eval/<failing-iid>/<failing-iid>.traj.json`
           and read the model's last assistant turns to see WHY it failed.
        5. Form ONE testable hypothesis.
        6. Apply ONE minimal edit to /workspace/prompts.yaml using edit-prompt.
        7. `git -C /workspace diff prompts.yaml`       — verify the patch
        8. Exit (see hard-rules above).

        # ✅ EDIT TOOL — use `edit-prompt` only

        Helper signature:

        ```bash
        edit-prompt KEY ACTION VALUE
        ```

        Where:
          KEY    = dot-path into prompts.yaml. Allowed:
                     agent.system_template
                     agent.instance_template
                     model.observation_template
                     model.format_error_template
          ACTION = `append` | `prepend` | `replace_substring` | `set`
          VALUE  = literal text. For `replace_substring`, write it as
                   `"OLD|||NEW"`.

        Examples:
        ```bash
        edit-prompt agent.system_template append "Submit as soon as your test reproducer passes; do not run additional verification commands."
        ```
        ```bash
        edit-prompt agent.instance_template replace_substring "3. Edit the source code to resolve the issue|||3. Edit the smallest necessary source change to resolve the issue, avoiding edits to unrelated functions"
        ```

        Verify after edit:
        ```bash
        git -C /workspace diff prompts.yaml | head -30
        ```

        # Budget
        Up to {max_steps} turns. Stop earlier if confident.
        """)

    # init git
    subprocess.run(["git","init","-q"], cwd=candidate_dir, check=False)
    subprocess.run(["git","-c","user.email=x@x","-c","user.name=x",
                    "add","-A"], cwd=candidate_dir, check=False)
    subprocess.run(["git","-c","user.email=x@x","-c","user.name=x",
                    "commit","-qm","seed"], cwd=candidate_dir, check=False)

    # write inputs
    work_meta = output_dir / "_meta_inputs"
    work_meta.mkdir(parents=True, exist_ok=True)
    (work_meta / "system_prompt.txt").write_text(sys_prompt)
    (work_meta / "user_prompt.txt").write_text(user_prompt)
    (work_meta / "meta.py").write_text(META_AGENT_SCRIPT_CLAUDE)

    out_capture = output_dir / "_meta_outputs"
    out_capture.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chmod","-R","a+rwX",str(out_capture)], check=False)

    container = f"phaseh-claude-{int(time.time())}"
    image = "python:3.11"

    subprocess.run(["docker","pull",image], capture_output=True, timeout=300)

    rc = subprocess.run([
        "docker","run","-d","--name",container,"--rm",
        "--network=host",
        "-v", f"{candidate_dir.resolve()}:/workspace",
        "-v", f"{archive_dir.resolve()}:/archive:ro",
        "-v", f"{out_capture.resolve()}:/output",
        "-v", f"{(work_meta / 'system_prompt.txt').resolve()}:/system_prompt.txt:ro",
        "-v", f"{(work_meta / 'user_prompt.txt').resolve()}:/user_prompt.txt:ro",
        "-v", f"{(work_meta / 'meta.py').resolve()}:/meta.py:ro",
        "-e", f"ANTHROPIC_API_KEY={api_key}",       # so anthropic SDK picks it up either way
        "-e", f"ANTHROPIC_AUTH_TOKEN={api_key}",
        "-e", f"META_MODEL={model}",
        "-e", f"META_MAX_STEPS={max_steps}",
        "-e", f"META_THINKING_BUDGET={thinking_budget}",
        "-w","/workspace",
        image, "sleep", "1800",
    ], capture_output=True, text=True, timeout=120)
    if rc.returncode != 0:
        return {"ok": False, "error": f"docker run failed: {rc.stderr[:400]}"}

    try:
        subprocess.run(["docker","exec",container,"bash","-c",
                        "apt-get -qq update && apt-get install -qy git jq >/dev/null 2>&1 && "
                        "pip install --quiet 'anthropic>=0.40' pyyaml ruamel.yaml"],
                       capture_output=True, text=True, timeout=300)
        subprocess.run(["docker","exec",container,"bash","-c",
                        "git config --global --add safe.directory '*' && "
                        "git config --global user.email 'meta@x.x' && "
                        "git config --global user.name 'meta'"],
                       capture_output=True, text=True, timeout=30)

        helper = r"""#!/usr/bin/env python3
import sys
from ruamel.yaml import YAML
if len(sys.argv) < 4 or sys.argv[1] in ('-h','--help'):
    print('usage: edit-prompt KEY ACTION VALUE'); sys.exit(2)
key, action, value = sys.argv[1], sys.argv[2], sys.argv[3]
y = YAML(); y.preserve_quotes = True; y.width = 999999
with open('/workspace/prompts.yaml') as f: d = y.load(f)
parts = key.split('.')
ref = d
for p in parts[:-1]:
    if p not in ref: print(f'ERROR: {p!r} not found'); sys.exit(3)
    ref = ref[p]
leaf = parts[-1]
if leaf not in ref: print(f'ERROR: {leaf!r} not found'); sys.exit(3)
current = ref[leaf]
if not isinstance(current, str):
    print(f'ERROR: {key} is not a string'); sys.exit(3)
if action == 'append':
    ref[leaf] = current.rstrip() + '\n' + value.rstrip() + '\n'
elif action == 'prepend':
    ref[leaf] = value.rstrip() + '\n' + current
elif action == 'replace_substring':
    if '|||' not in value: print('ERROR: VALUE for replace_substring must be "OLD|||NEW"'); sys.exit(3)
    old, new = value.split('|||', 1)
    if old not in current: print(f'ERROR: substring not found in {key}'); sys.exit(3)
    ref[leaf] = current.replace(old, new)
elif action == 'set':
    ref[leaf] = value
else:
    print(f'ERROR: unknown action {action!r}'); sys.exit(3)
with open('/workspace/prompts.yaml','w') as f: y.dump(d, f)
print(f'OK: edited {key} (action={action})')
import subprocess as sp
diff = sp.run(['git','-C','/workspace','diff','prompts.yaml'], capture_output=True, text=True).stdout
print(f'diff is now {len(diff)} bytes')
"""
        subprocess.run(["docker","exec",container,"bash","-c",
                        "cat > /usr/local/bin/edit-prompt <<'__HELPER_EOF__'\n" + helper +
                        "\n__HELPER_EOF__\nchmod +x /usr/local/bin/edit-prompt"],
                       capture_output=True, text=True, timeout=30)

        run = subprocess.run(["docker","exec",container,"python3","/meta.py"],
                             capture_output=True, text=True, timeout=3600)
        (output_dir / "meta_agent.stdout.log").write_text(
            run.stdout + "\n---STDERR---\n" + run.stderr)
    finally:
        subprocess.run(["docker","kill",container],
                       capture_output=True, text=True, timeout=30)

    res = {}
    for name in ("diff.patch","meta_trajectory.json","meta_result.json"):
        src = out_capture / name
        if src.exists():
            shutil.copy(src, output_dir / name)
            res[name] = str(output_dir / name)
    diff_bytes = (output_dir / "diff.patch").stat().st_size if (output_dir / "diff.patch").exists() else 0
    return {"ok": True, "diff_bytes": diff_bytes, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen",       type=int, default=1)
    ap.add_argument("--effort", default="xhigh", choices=list(EFFORT_BUDGET.keys()))
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--eval-samples", type=int, default=20)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    # load API key from .env
    env = read_env(ENV_FILE)
    api_key = (env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")
               or os.environ.get("ANTHROPIC_API_KEY")
               or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if not api_key:
        print(f"[main] no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN in {ENV_FILE} or env")
        sys.exit(2)

    gen_dir = PHASE_H_OUT / f"gen_{args.gen}_claude"
    gen_dir.mkdir(parents=True, exist_ok=True)
    candidate = gen_dir / "seed_harness"
    copy_seed(candidate)

    # archive of prior gens (Qwen gens only — Claude has its own lineage)
    prior_gens = {}
    for g in range(args.gen):
        d = PHASE_H_OUT / f"gen_{g}"
        if d.exists():
            prior_gens[g] = d
    archive = gen_dir / "_archive_mount"
    build_archive(prior_gens, archive)
    print(f"[main] archive with {len(prior_gens)} prior gens (Qwen lineage)")

    budget = EFFORT_BUDGET[args.effort]
    print(f"=== Claude meta-agent: model={args.model} effort={args.effort} thinking_budget={budget} ===")
    res = run_meta_agent_in_docker(
        candidate, archive, gen_dir,
        model=args.model, api_key=api_key,
        thinking_budget=budget, max_steps=args.max_steps,
    )
    (gen_dir / "meta_agent_result.json").write_text(json.dumps(res, indent=2))
    print(f"[meta] result: ok={res.get('ok')} diff_bytes={res.get('diff_bytes',0)}")

    ok, msg = import_check(candidate)
    if not ok:
        print("[meta] D26 import_check FAILED")
        (gen_dir / "rejected.txt").write_text(msg[:2000]); return
    print("[meta] D26 import_check OK")

    if args.no_eval:
        print("[meta] --no-eval; stopping"); return

    print(f"=== eval gen_{args.gen}_claude on {args.eval_samples} tasks ===")
    out = evaluate(candidate, args.eval_samples, gen_dir)
    (gen_dir / "eval_results.json").write_text(json.dumps(out, indent=2))
    print(f"[eval] {out}")


if __name__ == "__main__":
    main()
