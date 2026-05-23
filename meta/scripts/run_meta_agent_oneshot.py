"""
One-shot meta-agent runner — HOST-SIDE (no docker), for Phase H gen_1 retry.

Avoids the docker-in-docker complication that broke the first attempt.
Runs Qwen3.5-4B (via local vLLM at :8001) directly, gives it bash with cwd=
the candidate seed_harness/, and captures the diff.

Then optionally evaluates the candidate via mini's batch runner on a 20-task
SWE-Gym subset.

Usage:
    meta/scripts/run_meta_agent_oneshot.py --gen 1 --eval-samples 20
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

PROJECT_ROOT = Path("/home/t-hyunlee/mini-swe-agent")
SEED = PROJECT_ROOT / "meta/hyperagents_fork/domains/swe_gym/seed_harness"
PHASE_H_OUT = Path("/home/t-hyunlee/meta-harness-plan/results/phase_h")
GEN0_TRACES = PHASE_H_OUT / "gen_0" / "recent_traces"   # we'll create


def copy_seed(dst: Path):
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(SEED, dst, ignore=shutil.ignore_patterns("__pycache__"))


def gather_failure_traces(out_dir: Path, n: int = 3) -> Path:
    """Snapshot a few failed v2 trajectories the meta-agent can grep."""
    dst = out_dir / "recent_traces"
    dst.mkdir(parents=True, exist_ok=True)
    v2 = Path("/home/t-hyunlee/meta-harness-plan/results/phase0/0_50_ctx128k")
    scores = json.loads((v2.parent / "0_50_ctx128k_scores.json").read_text())
    failed = [iid for iid, r in scores.items() if not r.get("pass")][:n]
    for iid in failed:
        src = v2 / iid / f"{iid}.traj.json"
        if src.exists():
            text = src.read_text()[:60000]
            (dst / f"{iid}.traj.json").write_text(text)
    return dst


def run_meta_agent(work_dir: Path, traces_dir: Path, model: str,
                   endpoint: str, max_steps: int = 8) -> dict:
    client = OpenAI(base_url=endpoint, api_key="dummy")

    prompts = (work_dir / "prompts.yaml").read_text()
    traces_listing = subprocess.run(["ls", str(traces_dir)],
                                     capture_output=True, text=True).stdout

    system = ("You are a careful agent-harness modifier. Reason briefly, then "
              "execute exactly one bash block per turn. Stop when you have made "
              "ONE focused, minimal edit to prompts.yaml and verified it with "
              "`git diff`.")
    user = f"""Edit /workspace/prompts.yaml (a copy is at the bash cwd) to improve
SWE-Gym solve rate of the inner Qwen3.5-4B agent. Allowed keys to edit:
agent.system_template, agent.instance_template, model.observation_template,
model.format_error_template. Do NOT change `mswea_bash_command` or the
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT sentinel.

Recent failure trajectories you can inspect:
{traces_listing}

Make exactly ONE focused edit. Hypothesis-driven. After editing, run:
  git diff prompts.yaml
to verify, then output a fenced bash block with:
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
to exit.

Current prompts.yaml head (first 4000 chars):
```yaml
{prompts[:4000]}
```
"""

    # init git in work_dir to capture diffs
    subprocess.run(["git", "init", "-q"], cwd=work_dir, check=False)
    subprocess.run(["git", "-c", "user.email=x@x", "-c", "user.name=x",
                    "add", "-A"], cwd=work_dir, check=False)
    subprocess.run(["git", "-c", "user.email=x@x", "-c", "user.name=x",
                    "commit", "-qm", "seed"], cwd=work_dir, check=False)

    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user}]
    trajectory = list(messages)

    for step in range(max_steps):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=0.3, max_tokens=2500)
        except Exception as e:
            print(f"[meta] API error step {step}: {e}")
            break
        content = r.choices[0].message.content or ""
        print(f"[meta][{step}] {content[:200].replace(chr(10),' / ')}")
        trajectory.append({"role": "assistant", "content": content})
        messages.append({"role": "assistant", "content": content})

        blocks = re.findall(r"```(?:bash|sh)?\s*(.+?)```", content, re.DOTALL)
        if not blocks:
            print("[meta] no bash block; halting"); break
        cmd = blocks[0].strip()
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
            print("[meta] signaled done"); break

        # sandbox: run with cwd=work_dir, no inherited env that might leak secrets
        env = {"PATH": os.environ.get("PATH",""), "HOME": str(work_dir),
               "LANG":"C.UTF-8"}
        try:
            out = subprocess.run(["bash","-c", cmd], cwd=str(work_dir),
                                 capture_output=True, text=True, env=env, timeout=120)
            obs = (out.stdout + out.stderr)[:3000]
        except subprocess.TimeoutExpired:
            obs = "(bash command timed out after 120s)"
        print(f"[meta][{step}] obs head: {obs[:200].replace(chr(10),' / ')}")
        messages.append({"role": "user", "content": f"<output>{obs}</output>"})
        trajectory.append({"role": "user", "content": f"<output>{obs}</output>"})

    # capture diff
    diff = subprocess.run(["git", "diff"], cwd=str(work_dir),
                          capture_output=True, text=True).stdout
    return {"diff_bytes": len(diff), "diff": diff, "trajectory": trajectory}


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
    ], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=10800)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=1)
    ap.add_argument("--eval-samples", type=int, default=20)
    ap.add_argument("--endpoint", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--no-eval", action="store_true",
                    help="just run the meta-agent and save the diff; skip eval")
    args = ap.parse_args()

    gen_dir = PHASE_H_OUT / f"gen_{args.gen}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    candidate = gen_dir / "seed_harness"
    copy_seed(candidate)

    traces = gather_failure_traces(gen_dir)

    print(f"=== meta-agent on gen_{args.gen} (host-side) ===")
    res = run_meta_agent(candidate, traces, args.model, args.endpoint)
    (gen_dir / "diff.patch").write_text(res["diff"])
    (gen_dir / "meta_trajectory.json").write_text(json.dumps(res["trajectory"], indent=2)[:300000])
    print(f"[meta] diff bytes = {res['diff_bytes']}")
    if res["diff_bytes"] == 0:
        print("[meta] WARNING: meta-agent produced empty diff")

    # import_check
    chk = subprocess.run([sys.executable, "-c",
        f"import sys; sys.path.insert(0,'{candidate}'); "
        "from agent import DefaultAgent; "
        "from model import LitellmTextbasedModel; "
        "from environment import DockerEnvironment; print('OK')"],
        capture_output=True, text=True)
    if chk.returncode != 0:
        print(f"[meta] D26 import_check FAILED — rejecting candidate")
        (gen_dir / "rejected.txt").write_text(chk.stdout + chk.stderr)
        (gen_dir / "eval_results.json").write_text(json.dumps({
            "ok": False, "rejected": "import_check_failed",
            "stderr": (chk.stderr or "")[:500]}))
        return
    print("[meta] D26 import_check OK")

    if args.no_eval:
        print("[meta] --no-eval; stopping after diff capture")
        return

    print(f"=== eval gen_{args.gen} on {args.eval_samples} tasks ===")
    out = evaluate(candidate, args.eval_samples, gen_dir)
    (gen_dir / "eval_results.json").write_text(json.dumps(out, indent=2))
    print(f"[eval] {out}")


if __name__ == "__main__":
    main()
