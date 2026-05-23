"""
Phase H meta-agent v2 — docker-isolated, Hyperagents-style.

Key differences vs v1:
  • Runs inside docker (`python:3.11`) for safe sandboxing of model-generated bash
  • Mounts the full /archive/ of prior generations (source + traces + scores +
    summary) — Hyperagents' eval_path equivalent
  • User prompt describes prompts.yaml STRUCTURE only; the agent reads the
    file itself with `cat` (no 4000-char dump)
  • Strict bash/sh fence regex (won't get fooled by ```mswea_bash_command```)
  • 20-step budget (Hyperagents/AHE-scale exploration)

Usage:
    meta/scripts/run_meta_agent_v2.py --gen 1                        # full run
    meta/scripts/run_meta_agent_v2.py --gen 1 --no-eval              # diff only
    meta/scripts/run_meta_agent_v2.py --gen 1 --eval-samples 20      # 20-task eval
"""

import argparse
import ast
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

# ----- paths -----
PROJECT_ROOT = Path("/home/t-hyunlee/mini-swe-agent")
SEED         = PROJECT_ROOT / "meta/hyperagents_fork/domains/swe_gym/seed_harness"
PHASE_H_OUT  = Path("/home/t-hyunlee/meta-harness-plan/results/phase_h")

# ----- helpers -----

def copy_seed(dst: Path):
    if dst.exists():
        subprocess.run(["sudo","rm","-rf",str(dst)], check=False)
    shutil.copytree(SEED, dst, ignore=shutil.ignore_patterns("__pycache__",".git"))


def parse_test_list(field):
    if not field: return []
    if isinstance(field, list): return field
    try:    return ast.literal_eval(field)
    except: return []


def compute_gen_summary(gen_dir: Path) -> str:
    """Pre-compute a small summary.txt that the meta-agent can `cat` to
    orient itself: pass rate + failure-class breakdown + list of failing IIDs."""
    scores_p = gen_dir / "scores.json"
    if not scores_p.exists():
        return "(no scores.json — generation hasn't been scored yet)"
    scores = json.loads(scores_p.read_text())

    cls = Counter()
    failing_iids = []
    passing_iids = []
    for iid, r in scores.items():
        f2p_t = r.get("f2p_total", 0); f2p_p = r.get("f2p_pass", 0)
        p2p_t = r.get("p2p_total", 0); p2p_p = r.get("p2p_pass", 0)
        f2p_ok = f2p_t > 0 and f2p_p == f2p_t
        p2p_ok = p2p_t == 0 or p2p_p == p2p_t
        if f2p_ok and p2p_ok:
            cls["GOLDEN"] += 1; passing_iids.append(iid)
        elif f2p_ok and not p2p_ok:
            cls["REGRESSION"] += 1; failing_iids.append((iid, "REGRESSION"))
        elif not f2p_ok and p2p_ok:
            cls["WRONG_FIX"] += 1; failing_iids.append((iid, "WRONG_FIX"))
        else:
            cls["CATASTROPHIC"] += 1; failing_iids.append((iid, "CATASTROPHIC"))

    # exit_status breakdown from trajectories
    eval_dir = gen_dir / "eval"
    exit_status = Counter()
    if eval_dir.exists():
        for d in eval_dir.iterdir():
            if not d.is_dir(): continue
            tf = d / f"{d.name}.traj.json"
            if not tf.exists(): continue
            try:
                t = json.loads(tf.read_text())
                exit_status[t.get("info",{}).get("exit_status","?")] += 1
            except Exception:
                pass
    n_scored = len(scores); n_pass = cls["GOLDEN"]
    lines = []
    lines.append(f"# Generation summary")
    lines.append(f"")
    lines.append(f"Scored Submitted patches: {n_scored}")
    lines.append(f"  GOLDEN  (F2P pass, P2P pass): {cls['GOLDEN']}")
    lines.append(f"  REGRESSION  (F2P pass, P2P fail): {cls['REGRESSION']}")
    lines.append(f"  WRONG_FIX   (F2P fail, P2P pass): {cls['WRONG_FIX']}")
    lines.append(f"  CATASTROPHIC (F2P fail, P2P 0/all — patch broke pytest collection): {cls['CATASTROPHIC']}")
    lines.append(f"")
    lines.append(f"pass@1 over scored: {n_pass}/{n_scored} ({n_pass*100//max(n_scored,1)}%)")
    lines.append(f"")
    lines.append(f"Inner-trajectory exit_status distribution (across all attempted, including unsubmitted):")
    for s, c in exit_status.most_common():
        lines.append(f"  {s}: {c}")
    lines.append(f"")
    lines.append(f"Passing IIDs ({len(passing_iids)}):")
    for iid in passing_iids:
        lines.append(f"  ✓ {iid}")
    lines.append(f"")
    lines.append(f"Failing IIDs by class ({len(failing_iids)}):")
    for iid, c in failing_iids:
        lines.append(f"  ✗ {iid}    [{c}]")
    lines.append("")
    lines.append("Tip: open `eval/<iid>/<iid>.traj.json` for full per-task trajectory.")
    return "\n".join(lines)


def build_archive(prior_gens: dict[int, Path], dst: Path):
    """Build the /archive directory structure to be mounted RO into docker."""
    if dst.exists():
        subprocess.run(["sudo","rm","-rf",str(dst)], check=False)
    dst.mkdir(parents=True)
    for gen_id, gen_dir in prior_gens.items():
        gen_dst = dst / f"gen_{gen_id}"
        gen_dst.mkdir()
        # source code
        sh_src = gen_dir / "seed_harness"
        if sh_src.exists():
            shutil.copytree(sh_src, gen_dst / "seed_harness",
                            ignore=shutil.ignore_patterns("__pycache__",".git"))
        # scores + summary
        for fname in ("scores.json",):
            p = gen_dir / fname
            if p.exists():
                shutil.copy(p, gen_dst / fname)
        # eval trajectories — keep ONLY <iid>/<iid>.traj.json (drop *.log, *.csv)
        ev = gen_dir / "eval"
        if ev.exists():
            ev_dst = gen_dst / "eval"
            ev_dst.mkdir()
            for d in ev.iterdir():
                if not d.is_dir(): continue
                iid = d.name
                tf = d / f"{iid}.traj.json"
                if tf.exists():
                    (ev_dst / iid).mkdir()
                    shutil.copy(tf, ev_dst / iid / f"{iid}.traj.json")
        # summary
        (gen_dst / "summary.txt").write_text(compute_gen_summary(gen_dir))


def get_prompts_yaml_structure(prompts_path: Path) -> str:
    """Lightweight schema-summary of prompts.yaml — counts and key listings,
    no content."""
    import yaml
    data = yaml.safe_load(prompts_path.read_text())
    lines = []
    lines.append("Schema of /workspace/prompts.yaml (cat the file to see actual content):")
    for section in ("agent","environment","model"):
        if section not in data: continue
        lines.append(f"\n  {section}:")
        for k, v in data[section].items():
            if isinstance(v, str):
                lines.append(f"    {k}: <str, {len(v)} chars>")
            elif isinstance(v, (int,float,bool)) or v is None:
                lines.append(f"    {k}: {v!r}")
            elif isinstance(v, dict):
                inner = ", ".join(v.keys())
                lines.append(f"    {k}: <dict {{{inner}}}>")
            elif isinstance(v, list):
                lines.append(f"    {k}: <list[{len(v)}]>")
            else:
                lines.append(f"    {k}: <{type(v).__name__}>")
    lines.append("\n  Inner-agent runtime constants you must NOT change:")
    lines.append("    fence string:    `mswea_bash_command`   (the INNER agent's bash fence)")
    lines.append("    submit sentinel: `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`")
    return "\n".join(lines)


# ----- meta-agent runner (in docker) -----

META_AGENT_SCRIPT = r"""
from openai import OpenAI
import os, re, subprocess, json, sys, time
client = OpenAI(base_url=os.environ['META_ENDPOINT'], api_key='dummy')
MODEL = os.environ['META_MODEL']
MAX_STEPS = int(os.environ.get('META_MAX_STEPS','20'))

# strict bash/sh fence — does NOT match `mswea_bash_command`
FENCE_RE = re.compile(r'```(?:bash|sh)\s*\n(.*?)```', re.DOTALL)
SUBMIT_RE = re.compile(r'(?ms)^(?:\s*echo\s+COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\s*(?:&&|;|$).*)+$')

with open('/system_prompt.txt') as f:    SYS = f.read()
with open('/user_prompt.txt') as f:      USR = f.read()
# optional reasoning-aware addendum (used by reasoning models like Qwen3.5-35B-A3B)
HINT = os.environ.get('META_REASONING_HINT', '')
if HINT:
    SYS = SYS.rstrip() + '\n\n' + HINT.strip() + '\n'

MAX_TOKENS = int(os.environ.get('META_MAX_TOKENS', '3000'))

messages = [{'role':'system','content':SYS}, {'role':'user','content':USR}]
trajectory = list(messages)

def log_msg(label, text):
    head = text[:240].replace(chr(10),' / ')
    print(f'[meta][{label}] {head}')

stopped_by = 'budget'
for step in range(MAX_STEPS):
    try:
        r = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.3, max_tokens=MAX_TOKENS)
    except Exception as e:
        print(f'[meta] API error step {step}: {repr(e)[:300]}')
        stopped_by = 'api_error'
        break
    content = r.choices[0].message.content or ''
    log_msg(f'asst#{step}', content)
    messages.append({'role':'assistant','content':content})
    trajectory.append({'role':'assistant','content':content})

    blocks = FENCE_RE.findall(content)
    if not blocks:
        # fallback: catch tag-less ``` ... ``` if it looks like a real shell command
        any_block = re.findall(r'```\s*\n(.*?)```', content, re.DOTALL)
        if any_block:
            # if first line looks like a shell command (no spaces yet trying)
            first = any_block[0].splitlines()[0].strip() if any_block[0] else ''
            if first and not re.match(r'^[a-z_]+_command$', first):
                blocks = any_block
    if not blocks:
        print('[meta] no bash/sh block; stopping')
        stopped_by = 'no_block'
        break
    cmd = blocks[0].strip()
    # strict submit detection: only true `echo CT...` blocks, not edit-prompt args containing it
    is_submit = bool(SUBMIT_RE.match(cmd)) and 'edit-prompt' not in cmd
    if is_submit:
        # gate: only accept submit if there is an actual diff
        cur_diff = subprocess.run(['git','-C','/workspace','diff','prompts.yaml'],
                                  capture_output=True, text=True).stdout
        if cur_diff.strip():
            print(f'[meta] agent signaled done with diff ({len(cur_diff)} bytes)')
            stopped_by = 'submit'
            break
        # empty diff — push back and continue
        retry_msg = (
            "<output rc=0>(intercepted before exit) Your git diff is empty.\n"
            "You have NOT produced any edit yet. You MUST apply at least one\n"
            "concrete edit to /workspace/prompts.yaml using the SAFE EDIT recipe\n"
            "(string-replace or ruamel.yaml). Pick ONE focused change, apply it,\n"
            "run `git -C /workspace diff prompts.yaml` to verify a non-empty\n"
            "diff appears, and then exit.</output>"
        )
        print('[meta] empty diff at submit; pushing back')
        messages.append({'role':'user','content':retry_msg})
        trajectory.append({'role':'user','content':retry_msg})
        continue

    try:
        out = subprocess.run(['bash','-c',cmd], cwd='/workspace',
                             capture_output=True, text=True, timeout=120)
        obs = (out.stdout + out.stderr)[:4000]
        ret = out.returncode
    except subprocess.TimeoutExpired:
        obs = '(bash command timed out after 120s)'
        ret = -1
    log_msg(f'obs#{step} rc={ret}', obs)
    obs_msg = f'<output rc={ret}>\n{obs}\n</output>'
    messages.append({'role':'user','content':obs_msg})
    trajectory.append({'role':'user','content':obs_msg})

# capture diff
diff = subprocess.run(['git','-C','/workspace','diff'],
                      capture_output=True, text=True).stdout
print(f'[meta] final diff bytes: {len(diff)}; stopped_by: {stopped_by}')

with open('/output/diff.patch','w') as f:    f.write(diff)
with open('/output/meta_trajectory.json','w') as f:
    f.write(json.dumps(trajectory, indent=2)[:300000])
with open('/output/meta_result.json','w') as f:
    f.write(json.dumps({'diff_bytes': len(diff), 'stopped_by': stopped_by,
                        'steps_used': step+1, 'model': MODEL}, indent=2))
"""


def run_meta_agent_in_docker(candidate_dir: Path, archive_dir: Path,
                             output_dir: Path,
                             model: str, endpoint: str,
                             max_steps: int = 20) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # build prompts
    sys_prompt = textwrap.dedent("""\
        You are the OUTER meta-agent.

        Your single task: edit `/workspace/prompts.yaml` so that the INNER agent
        (Qwen3.5-4B running with this prompts.yaml inside mini-swe-agent) solves
        more SWE-Gym tasks.

        Hard rules:
          • You are the OUTER agent. ALWAYS use ```bash``` fences for your
            shell commands.
          • NEVER write ```mswea_bash_command``` in your own response — that is
            the INNER agent's parser; if you produce that fence here it becomes
            literal bash and will fail with "command not found".
          • One bash block per response, executed at cwd `/workspace`.
          • Stop after at most one targeted edit; verify with `git diff`.

        When you are done with a single focused edit, exit with EXACTLY this:
        ```bash
        echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
        ```
        """)

    # construct user prompt with paths + structure + prior summary
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
          ├── agent.py              ← do NOT edit (won't take effect through current eval path)
          ├── model.py              ← do NOT edit
          └── environment.py        ← do NOT edit

        /archive/        RO    prior generations and their evaluation outputs
        {archive_listing}        Each generation dir contains:
          seed_harness/             prior agent code (== /workspace at the start of that gen)
          scores.json               {{instance_id → {{pass, f2p_*, p2p_*}}}}
          summary.txt               pre-computed pass rate + failure-class breakdown + IID lists
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
        If you submit while the diff is still empty, you will be interrupted
        and asked to actually make an edit.

        # Recommended workflow

        1. `cat /archive/gen_0/summary.txt`            — orient
        2. `cat /archive/gen_0/scores.json | python3 -m json.tool | head -60`  — pass/fail map
        3. `ls /archive/gen_0/eval/ | head -20`        — list trajectories
        4. read 2-3 failing trajectories — e.g.
             `tail -c 8000 /archive/gen_0/eval/<failing-iid>/<failing-iid>.traj.json`
           and read the model's last assistant turns to see WHY it failed
        5. form ONE testable hypothesis (e.g. "instance_template doesn't tell
           the inner agent to *submit* early enough" or "observation_template
           truncation hides relevant context")
        6. apply ONE minimal edit to /workspace/prompts.yaml — see SAFE EDIT below
        7. `git -C /workspace diff prompts.yaml`       — verify the patch
        8. exit (see hard-rules above)

        # ✅ EDIT TOOL — use `edit-prompt` only

        We have installed a dedicated helper command. **Use it for all edits.**
        Do NOT use heredoc to rewrite the file — past attempts at heredoc edits
        have either truncated the file or produced empty diffs due to YAML's
        embedded backticks and tags.

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
                   `"OLD|||NEW"` (substring to find, then `|||`, then replacement).

        # Worked examples (you may copy then adapt)

        Append one rule to the system prompt:
        ```bash
        edit-prompt agent.system_template append "Submit as soon as your test reproducer passes; do not run additional verification commands."
        ```

        Substitute inside instance_template (e.g. tighten the workflow):
        ```bash
        edit-prompt agent.instance_template replace_substring "3. Edit the source code to resolve the issue|||3. Edit the smallest necessary source change to resolve the issue, avoiding edits to unrelated functions"
        ```

        Verify after edit:
        ```bash
        git -C /workspace diff prompts.yaml | head -30
        ```

        The helper writes atomically via ruamel.yaml (preserves block style)
        and prints the resulting diff size. You should see >0 bytes in
        `git diff` after a successful invocation.

        # Budget
        Up to {max_steps} turns. Stop earlier if confident.
        """)

    # init git in /workspace
    subprocess.run(["git","init","-q"], cwd=candidate_dir, check=False)
    subprocess.run(["git","-c","user.email=x@x","-c","user.name=x",
                    "add","-A"], cwd=candidate_dir, check=False)
    subprocess.run(["git","-c","user.email=x@x","-c","user.name=x",
                    "commit","-qm","seed"], cwd=candidate_dir, check=False)

    # write prompts + meta script to temp files we'll mount
    work_meta = output_dir / "_meta_inputs"
    work_meta.mkdir(parents=True, exist_ok=True)
    (work_meta / "system_prompt.txt").write_text(sys_prompt)
    (work_meta / "user_prompt.txt").write_text(user_prompt)
    (work_meta / "meta.py").write_text(META_AGENT_SCRIPT)

    # output capture dir (mounted as /output)
    out_capture = output_dir / "_meta_outputs"
    out_capture.mkdir(parents=True, exist_ok=True)

    # ensure container can write to output dir (we'll fix perms with sudo after)
    subprocess.run(["chmod","-R","a+rwX",str(out_capture)], check=False)

    container = f"phaseh-meta-{int(time.time())}"
    image = "python:3.11"

    # pull image (cached if present)
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
        "-e", f"META_ENDPOINT={endpoint}",
        "-e", f"META_MODEL={model}",
        "-e", f"META_MAX_STEPS={max_steps}",
        "-e", f"META_MAX_TOKENS={os.environ.get('META_MAX_TOKENS','3000')}",
        "-e", f"META_REASONING_HINT={os.environ.get('META_REASONING_HINT','')}",
        "-w","/workspace",
        image, "sleep", "1800",
    ], capture_output=True, text=True, timeout=120)
    if rc.returncode != 0:
        return {"ok": False, "error": f"docker run failed: {rc.stderr[:400]}"}

    try:
        # install dependencies
        subprocess.run(["docker","exec",container,"bash","-c",
                        "apt-get -qq update && apt-get install -qy git jq >/dev/null 2>&1 && "
                        "pip install --quiet 'openai>=1.0' pyyaml ruamel.yaml"],
                       capture_output=True, text=True, timeout=300)
        # fix git safe.directory across all volumes (root inside vs t-hyunlee outside)
        subprocess.run(["docker","exec",container,"bash","-c",
                        "git config --global --add safe.directory '*' && "
                        "git config --global user.email 'meta@x.x' && "
                        "git config --global user.name 'meta'"],
                       capture_output=True, text=True, timeout=30)

        # install helper command `edit-prompt KEY ACTION VALUE` for atomic YAML edits
        helper = r"""#!/usr/bin/env python3
import sys, json
from ruamel.yaml import YAML
if len(sys.argv) < 4 or sys.argv[1] in ('-h','--help'):
    print('usage: edit-prompt KEY ACTION VALUE')
    print('  KEY = e.g. agent.system_template  (dot-path inside /workspace/prompts.yaml)')
    print('  ACTION = append | prepend | replace_substring | set')
    print('  VALUE = text to apply')
    print('         for replace_substring, format VALUE as: "OLD|||NEW"')
    sys.exit(2)
key, action, value = sys.argv[1], sys.argv[2], sys.argv[3]
y = YAML(); y.preserve_quotes = True; y.width = 999999
with open('/workspace/prompts.yaml') as f:
    d = y.load(f)
parts = key.split('.')
ref = d
for p in parts[:-1]:
    if p not in ref:
        print(f'ERROR: key segment {p!r} not found in YAML'); sys.exit(3)
    ref = ref[p]
leaf = parts[-1]
if leaf not in ref:
    print(f'ERROR: key {leaf!r} not found under {".".join(parts[:-1])!r}'); sys.exit(3)
current = ref[leaf]
if not isinstance(current, str):
    print(f'ERROR: {key} is not a string (type={type(current).__name__})'); sys.exit(3)
if action == 'append':
    ref[leaf] = current.rstrip() + '\n' + value.rstrip() + '\n'
elif action == 'prepend':
    ref[leaf] = value.rstrip() + '\n' + current
elif action == 'replace_substring':
    if '|||' not in value:
        print('ERROR: VALUE for replace_substring must be \"OLD|||NEW\"'); sys.exit(3)
    old, new = value.split('|||', 1)
    if old not in current:
        print(f'ERROR: substring {old!r} not found in {key} (try a shorter unique substring)'); sys.exit(3)
    ref[leaf] = current.replace(old, new)
elif action == 'set':
    ref[leaf] = value
else:
    print(f'ERROR: unknown action {action!r}'); sys.exit(3)
with open('/workspace/prompts.yaml','w') as f:
    y.dump(d, f)
print(f'OK: edited {key} (action={action})')
import subprocess
diff = subprocess.run(['git','-C','/workspace','diff','prompts.yaml'],capture_output=True,text=True).stdout
print(f'diff is now {len(diff)} bytes')
"""
        # write helper into the container at /usr/local/bin/edit-prompt
        subprocess.run(["docker","exec",container,"bash","-c",
                        "cat > /usr/local/bin/edit-prompt <<'__HELPER_EOF__'\n" + helper +
                        "\n__HELPER_EOF__\nchmod +x /usr/local/bin/edit-prompt"],
                       capture_output=True, text=True, timeout=30)
        # run the meta-agent
        run = subprocess.run(["docker","exec",container,"python3","/meta.py"],
                             capture_output=True, text=True, timeout=1800)
        (output_dir / "meta_agent.stdout.log").write_text(
            run.stdout + "\n---STDERR---\n" + run.stderr)
    finally:
        subprocess.run(["docker","kill",container],
                       capture_output=True, text=True, timeout=30)

    # collect outputs
    res = {}
    for name in ("diff.patch","meta_trajectory.json","meta_result.json"):
        src = out_capture / name
        if src.exists():
            shutil.copy(src, output_dir / name)
            res[name] = str(output_dir / name)
    diff_bytes = (output_dir / "diff.patch").stat().st_size if (output_dir / "diff.patch").exists() else 0
    return {"ok": True, "diff_bytes": diff_bytes, **res}


# ----- D26 compliance check + eval -----

def import_check(candidate_dir: Path) -> tuple[bool, str]:
    code = (
        f"import sys; sys.path.insert(0,'{candidate_dir}'); "
        "from agent import DefaultAgent; "
        "from model import LitellmTextbasedModel; "
        "from environment import DockerEnvironment; print('OK')"
    )
    r = subprocess.run([sys.executable,"-c",code],
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0, (r.stdout + r.stderr)


def evaluate(candidate_dir: Path, n_tasks: int, out_dir: Path) -> dict:
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    candidate_prompts = candidate_dir / "prompts.yaml"
    r = subprocess.run([
        sys.executable, "-m", "meta.phase0.run_phase0",
        "--slice", f"0:{n_tasks}",
        "--workers", "4",
        "--model-name", "hosted_vllm/qwen3.5-4b",
        "--config", str(candidate_prompts),
        "--output", str(eval_dir),
    ], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=14400)
    (out_dir / "eval_runner.log").write_text(r.stdout + "\n---STDERR---\n" + r.stderr)
    preds = eval_dir / "preds.json"
    if not preds.exists():
        return {"ok": False, "error": "no preds.json"}
    scores_p = out_dir / "scores.json"
    s = subprocess.run([
        sys.executable, "-m", "meta.phase0.score_patches",
        "--preds", str(preds),
        "--output", str(scores_p),
        "--only-submitted", "--workers", "2",
    ], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    (out_dir / "scorer.log").write_text(s.stdout + "\n---STDERR---\n" + s.stderr)
    if not scores_p.exists():
        return {"ok": False, "error": "no scores.json"}
    data = json.loads(scores_p.read_text())
    n_pass = sum(1 for v in data.values() if v.get("pass"))
    return {"ok": True, "pass": n_pass, "total_scored": len(data),
            "score": n_pass / max(len(data), 1)}


# ----- main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen",       type=int, default=1)
    ap.add_argument("--eval-samples", type=int, default=20)
    ap.add_argument("--endpoint", default="http://localhost:8001/v1")
    ap.add_argument("--model",    default="qwen3.5-4b")
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--no-eval", action="store_true",
                    help="only run the meta-agent and capture the diff; skip eval")
    args = ap.parse_args()

    gen_dir = PHASE_H_OUT / f"gen_{args.gen}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    candidate = gen_dir / "seed_harness"
    copy_seed(candidate)

    # build archive of prior gens
    prior_gens = {}
    for g in range(args.gen):
        d = PHASE_H_OUT / f"gen_{g}"
        if d.exists():
            prior_gens[g] = d
    archive = gen_dir / "_archive_mount"
    build_archive(prior_gens, archive)
    print(f"[main] archive built at {archive} with {len(prior_gens)} prior gens")

    # run meta-agent in docker
    print(f"=== meta-agent v2 (docker) on gen_{args.gen} ===")
    res = run_meta_agent_in_docker(
        candidate, archive, gen_dir,
        model=args.model, endpoint=args.endpoint, max_steps=args.max_steps,
    )
    (gen_dir / "meta_agent_result.json").write_text(json.dumps(res, indent=2))
    print(f"[meta] result: ok={res.get('ok')} diff_bytes={res.get('diff_bytes',0)}")

    # D26 check
    ok, msg = import_check(candidate)
    if not ok:
        print(f"[meta] D26 import_check FAILED — rejecting candidate")
        (gen_dir / "rejected.txt").write_text(msg[:2000])
        (gen_dir / "eval_results.json").write_text(json.dumps({
            "ok": False, "rejected": "import_check_failed",
            "stderr": msg[:500]
        }))
        return
    print("[meta] D26 import_check OK")

    if args.no_eval:
        print("[meta] --no-eval; stopping")
        return

    print(f"=== eval gen_{args.gen} on {args.eval_samples} tasks ===")
    out = evaluate(candidate, args.eval_samples, gen_dir)
    (gen_dir / "eval_results.json").write_text(json.dumps(out, indent=2))
    print(f"[eval] {out}")


if __name__ == "__main__":
    main()
