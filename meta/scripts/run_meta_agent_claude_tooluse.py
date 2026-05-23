"""
Phase H meta-agent — Claude with native tool-use (A2' asymmetric ablation).

Same architecture as run_meta_agent_claude.py BUT uses Anthropic's native
tool-use API instead of markdown bash fences. Claude is trained for this
format, so it doesn't drift into hallucinated <function_calls> XML or
fabricate "submission accepted" without actually making edits.

Tools exposed:
  • bash(command)            — exec shell in /workspace
  • edit_prompt(key, action, value) — atomic yaml edit via /usr/local/bin/edit-prompt
  • complete()               — signal done (gated on non-empty diff)

Usage:
    meta/scripts/run_meta_agent_claude_tooluse.py --gen 1 --effort xhigh --no-eval
    meta/scripts/run_meta_agent_claude_tooluse.py --gen 1 --eval-samples 20
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

EFFORT_BUDGET = {"low":1024,"medium":4096,"high":16384,"xhigh":32768}

# reuse helpers
sys.path.insert(0, str(PROJECT_ROOT))
from meta.scripts.run_meta_agent_v2 import (
    copy_seed, compute_gen_summary, build_archive,
    get_prompts_yaml_structure, import_check, evaluate,
)


def read_env(p: Path) -> dict[str,str]:
    if not p.exists(): return {}
    return {k.strip(): v.strip().strip('"').strip("'")
            for l in p.read_text().splitlines() if "=" in l and not l.lstrip().startswith("#")
            for k,v in [l.split("=",1)]}


META_AGENT_SCRIPT = r"""
import os, subprocess, json, sys
from anthropic import Anthropic

api_key = os.environ.get('ANTHROPIC_API_KEY')
auth_token = os.environ.get('ANTHROPIC_AUTH_TOKEN')
if auth_token and auth_token.startswith('sk-ant-oat'):
    client = Anthropic(auth_token=auth_token)
elif api_key:
    client = Anthropic(api_key=api_key)
else:
    print('META: no creds'); sys.exit(2)

MODEL = os.environ['META_MODEL']
MAX_STEPS = int(os.environ.get('META_MAX_STEPS','20'))
THINKING_BUDGET = int(os.environ.get('META_THINKING_BUDGET','32000'))
MAX_TOKENS = THINKING_BUDGET + 6000

with open('/system_prompt.txt') as f: SYS = f.read()
with open('/user_prompt.txt')   as f: USR = f.read()

TOOLS = [
    {
        'name': 'bash',
        'description': 'Execute a shell command in /workspace. Returns stdout+stderr and returncode.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'command': {'type':'string', 'description':'Shell command (bash -c). cwd = /workspace.'}
            },
            'required': ['command']
        }
    },
    {
        'name': 'edit_prompt',
        'description': ('Atomically edit /workspace/prompts.yaml using ruamel.yaml. '
                        'Preserves YAML block style. Use this for ALL prompts.yaml edits — '
                        'do not write the file through bash.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'key':    {'type':'string',
                           'enum':['agent.system_template','agent.instance_template',
                                   'model.observation_template','model.format_error_template'],
                           'description':'Dot-path to the string field to modify.'},
                'action': {'type':'string','enum':['append','prepend','replace_substring','set']},
                'value':  {'type':'string',
                           'description':('Text to apply. For action=replace_substring, '
                                         'pass it as "OLD|||NEW" (OLD substring must appear verbatim).')}
            },
            'required':['key','action','value']
        }
    },
    {
        'name': 'complete',
        'description': ('Signal that you are done with your edit. ONLY call this after you have '
                       'used edit_prompt successfully and verified the change with `git diff`. '
                       'If git diff is empty when you call this, it will be rejected and you '
                       'will be asked to actually make an edit.'),
        'input_schema': {'type':'object','properties':{},'required':[]}
    }
]

def run_bash(command):
    try:
        out = subprocess.run(['bash','-c',command], cwd='/workspace',
                             capture_output=True, text=True, timeout=120)
        return f'<rc={out.returncode}>\n' + (out.stdout + out.stderr)[:4000]
    except subprocess.TimeoutExpired:
        return '<rc=-1>\n(timed out after 120s)'

def run_edit_prompt(key, action, value):
    cmd = ['/usr/local/bin/edit-prompt', key, action, value]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return f'<rc={out.returncode}>\n' + (out.stdout + out.stderr)[:2000]

def current_diff():
    return subprocess.run(['git','-C','/workspace','diff','prompts.yaml'],
                          capture_output=True, text=True).stdout

messages = [{'role':'user','content':USR}]
trajectory = [{'role':'system','content':SYS}, {'role':'user','content':USR}]

stopped_by = 'budget'
for step in range(MAX_STEPS):
    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYS,
            tools=TOOLS,
            thinking={'type':'enabled','budget_tokens':THINKING_BUDGET},
            messages=messages,
        ) as stream:
            for _ in stream.text_stream:
                pass
            final = stream.get_final_message()
    except Exception as e:
        print(f'[meta] API error step {step}: {repr(e)[:400]}', flush=True)
        stopped_by = 'api_error'; break

    # capture for trajectory
    text_visible = []
    thinking_chars = 0
    tool_uses = []
    for blk in final.content:
        if blk.type == 'thinking':
            thinking_chars += len(blk.thinking)
        elif blk.type == 'text':
            text_visible.append(blk.text)
        elif blk.type == 'tool_use':
            tool_uses.append({'id':blk.id,'name':blk.name,'input':blk.input})
    visible = ''.join(text_visible)
    print(f'[meta][asst#{step} thinking={thinking_chars}c stop={final.stop_reason}] {visible[:200].replace(chr(10),chr(32)+chr(47)+chr(32))}', flush=True)
    for tu in tool_uses:
        print(f'  tool_use: {tu["name"]}({json.dumps(tu["input"])[:160]})', flush=True)
    trajectory.append({
        'role':'assistant',
        'content':visible,
        'thinking_chars':thinking_chars,
        'tool_uses':tool_uses,
        'stop_reason':final.stop_reason,
    })

    # echo assistant message back (need full content blocks, incl. thinking, for next turn)
    messages.append({'role':'assistant','content':final.content})

    if final.stop_reason == 'end_turn':
        # claude is done. validate.
        diff = current_diff()
        if diff.strip():
            print(f'[meta] end_turn with diff ({len(diff)} bytes)', flush=True)
            stopped_by = 'end_turn_with_diff'; break
        # nothing to ship — push back as a synthetic user msg
        retry = [{'type':'text','text':
            ('Your git diff is empty. You did not make any actual edit yet. '
             'Use the edit_prompt tool to apply a concrete change to one of the '
             'four allowed keys, then verify with bash `git diff prompts.yaml`, '
             'then call the complete tool.')}]
        messages.append({'role':'user','content':retry})
        trajectory.append({'role':'user','content':retry[0]['text']})
        continue

    if final.stop_reason == 'tool_use':
        tool_results = []
        early_complete = False
        for blk in final.content:
            if blk.type != 'tool_use':
                continue
            if blk.name == 'bash':
                res = run_bash(blk.input.get('command',''))
            elif blk.name == 'edit_prompt':
                res = run_edit_prompt(
                    blk.input.get('key',''), blk.input.get('action',''),
                    blk.input.get('value',''))
            elif blk.name == 'complete':
                diff = current_diff()
                if diff.strip():
                    print(f'[meta] complete with diff ({len(diff)} bytes)', flush=True)
                    res = 'OK: complete accepted'
                    stopped_by = 'submit'; early_complete = True
                else:
                    res = ('REJECTED: git diff is empty. You have not made an edit yet. '
                          'Use edit_prompt, verify with bash `git diff prompts.yaml`, then '
                          'call complete again.')
            else:
                res = f'unknown tool: {blk.name}'
            tool_results.append({
                'type':'tool_result',
                'tool_use_id':blk.id,
                'content':res,
            })
            # log
            print(f'  tool_result for {blk.name}: {res[:200].replace(chr(10),chr(32))}',
                  flush=True)
            trajectory.append({
                'role':'tool_result',
                'tool_name':blk.name,
                'content':res,
            })
            if early_complete: break
        messages.append({'role':'user','content':tool_results})
        if early_complete:
            break
        continue

    # other stop reasons
    print(f'[meta] unexpected stop_reason {final.stop_reason}; stopping', flush=True)
    stopped_by = f'stop_{final.stop_reason}'
    break

diff = current_diff()
print(f'[meta] final diff bytes: {len(diff)}; stopped_by: {stopped_by}', flush=True)

with open('/output/diff.patch','w') as f:    f.write(diff)
with open('/output/meta_trajectory.json','w') as f:
    f.write(json.dumps(trajectory, indent=2)[:400000])
with open('/output/meta_result.json','w') as f:
    f.write(json.dumps({'diff_bytes': len(diff), 'stopped_by': stopped_by,
                        'steps_used': step+1, 'model': MODEL,
                        'thinking_budget': THINKING_BUDGET, 'mode':'tool_use'}, indent=2))
"""


def run_in_docker(candidate_dir: Path, archive_dir: Path, output_dir: Path,
                  model: str, api_key: str, thinking_budget: int,
                  max_steps: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    sys_prompt = textwrap.dedent("""\
        You are the OUTER meta-agent. Your job: edit /workspace/prompts.yaml so the
        INNER agent (Qwen3.5-4B inside mini-swe-agent) solves more SWE-Gym tasks.

        You have three native tools:
          • bash(command) — execute a shell command in /workspace
          • edit_prompt(key, action, value) — atomically edit /workspace/prompts.yaml
          • complete() — signal done (rejected if git diff is empty)

        Workflow:
          1. Use bash to read /archive/gen_0/summary.txt, the scores, and a few
             failing trajectories.
          2. Form ONE testable hypothesis about a prompt change.
          3. Use edit_prompt to apply it (key MUST be one of the four allowed).
          4. Use bash `git -C /workspace diff prompts.yaml` to verify.
          5. Call complete() — only after the diff is non-empty.

        Constraints:
          • Make at most ONE focused edit.
          • Never write to /workspace/prompts.yaml via bash; always use edit_prompt.
          • Do not edit agent.py / model.py / environment.py — edits there have
            no effect through the current eval path.
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

        /workspace/      RW    candidate harness — edits go here (use edit_prompt)
          ├── prompts.yaml          ← the only file your edits affect
          ├── agent.py              ← do NOT edit
          ├── model.py              ← do NOT edit
          └── environment.py        ← do NOT edit

        /archive/        RO    prior generations with their evaluation outputs
        {archive_listing}        Each generation contains:
          seed_harness/             prior code (== /workspace at the start of that gen)
          scores.json               {{instance_id → {{pass, f2p_*, p2p_*}}}}
          summary.txt               pre-computed pass rate + failure-class breakdown + IID lists
          eval/<iid>/<iid>.traj.json     full inner-agent trajectory per task

        # prompts.yaml — what you may edit

        {prompts_struct}

        Allowed `key` values for edit_prompt:
          agent.system_template
          agent.instance_template
          model.observation_template
          model.format_error_template

        # Recommended exploration

        Use the bash tool with commands like:

          cat /archive/gen_0/summary.txt
          cat /archive/gen_0/scores.json | python3 -m json.tool | head -60
          ls /archive/gen_0/eval/
          tail -c 8000 /archive/gen_0/eval/<iid>/<iid>.traj.json

        Read 2-3 failing trajectories before deciding on an edit.

        Budget: up to {max_steps} turns. Stop early when you have a confident edit.
        """)

    # init git in candidate
    subprocess.run(["git","init","-q"], cwd=candidate_dir, check=False)
    subprocess.run(["git","-c","user.email=x@x","-c","user.name=x","add","-A"],
                   cwd=candidate_dir, check=False)
    subprocess.run(["git","-c","user.email=x@x","-c","user.name=x","commit","-qm","seed"],
                   cwd=candidate_dir, check=False)

    work_meta = output_dir / "_meta_inputs"
    work_meta.mkdir(parents=True, exist_ok=True)
    (work_meta / "system_prompt.txt").write_text(sys_prompt)
    (work_meta / "user_prompt.txt").write_text(user_prompt)
    (work_meta / "meta.py").write_text(META_AGENT_SCRIPT)

    out_capture = output_dir / "_meta_outputs"
    out_capture.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chmod","-R","a+rwX",str(out_capture)], check=False)

    container = f"phaseh-claude-tu-{int(time.time())}"
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
        "-e", f"ANTHROPIC_API_KEY={api_key}",
        "-e", f"ANTHROPIC_AUTH_TOKEN={api_key}",
        "-e", f"META_MODEL={model}",
        "-e", f"META_MAX_STEPS={max_steps}",
        "-e", f"META_THINKING_BUDGET={thinking_budget}",
        "-w","/workspace",
        image, "sleep", "3600",
    ], capture_output=True, text=True, timeout=120)
    if rc.returncode != 0:
        return {"ok":False, "error": f"docker run failed: {rc.stderr[:400]}"}

    try:
        subprocess.run(["docker","exec",container,"bash","-c",
                        "apt-get -qq update && apt-get install -qy git jq >/dev/null 2>&1 && "
                        "pip install --quiet 'anthropic>=0.40' pyyaml ruamel.yaml"],
                       capture_output=True, text=True, timeout=300)
        subprocess.run(["docker","exec",container,"bash","-c",
                        "git config --global --add safe.directory '*' && "
                        "git config --global user.email meta@x.x && "
                        "git config --global user.name meta"],
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
    ap.add_argument("--gen", type=int, default=1)
    ap.add_argument("--effort", default="xhigh", choices=list(EFFORT_BUDGET.keys()))
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--eval-samples", type=int, default=20)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--suffix", default="claude_tu",
                    help="output dir suffix: gen_<N>_<suffix>")
    args = ap.parse_args()

    env = read_env(ENV_FILE)
    api_key = (env.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_AUTH_TOKEN")
               or os.environ.get("ANTHROPIC_API_KEY")
               or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if not api_key:
        print(f"[main] no creds in {ENV_FILE}"); sys.exit(2)

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

    budget = EFFORT_BUDGET[args.effort]
    print(f"=== Claude (tool-use) meta-agent: model={args.model} effort={args.effort} budget={budget} ===")
    res = run_in_docker(candidate, archive, gen_dir,
                        model=args.model, api_key=api_key,
                        thinking_budget=budget, max_steps=args.max_steps)
    (gen_dir / "meta_agent_result.json").write_text(json.dumps(res, indent=2))
    print(f"[meta] result: ok={res.get('ok')} diff_bytes={res.get('diff_bytes',0)}")

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
