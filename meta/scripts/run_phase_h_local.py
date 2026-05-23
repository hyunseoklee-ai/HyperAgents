"""
Phase H — harness-only evolve (LOCAL outer loop, not Hyperagents Docker stack).

Lightweight DGM-style outer loop that exercises our existing infrastructure:
  • Meta-agent: Qwen3.5-4B (no LoRA) via the local vLLM serving endpoint, called
    through mini-swe-agent's DefaultAgent on a small synthetic "modify the
    harness code" task inside a docker sandbox.
  • Mutation surface: the four files under
      meta/hyperagents_fork/domains/swe_gym/seed_harness/
    (agent.py, model.py, environment.py, prompts.yaml).
  • Per-generation eval: re-uses meta.phase0.run_phase0 + score_patches on a
    smoke set of SWE-Gym tasks with the mutated harness as the inner agent.

This is the A2 ablation (DGM-style baseline) at 4 B, scaled down to a
practical 1–3 iterations for first-pass measurement. The full Hyperagents
container-based generate_loop.py path remains available for a later, fuller
run.

Output structure:
    OUT/
      gen_0/   ← baseline (no edits) — links to existing Phase 0 v2 result
        seed_harness/      (copy)
        eval_results.json  (pass@1, exit_status_counts)
      gen_1/
        meta_agent.trajectory.json   (what the meta-agent did)
        diff_against_seed.patch
        seed_harness/                (applied)
        eval_results.json
      ...
      summary.json   (best-of-archive curve)

Usage:
    meta/scripts/run_phase_h_local.py --iters 1 --eval-samples 20
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_HARNESS_SRC = (PROJECT_ROOT
                    / "meta" / "hyperagents_fork" / "domains" / "swe_gym" / "seed_harness")


META_AGENT_TASK = """\
You are a software-engineering meta-agent. Your job is to edit the harness
prompts at /workspace/seed_harness/prompts.yaml to make the inner agent
(mini-swe-agent, Qwen3.5-4B) solve more SWE-Gym tasks. Failure trajectories
from the current baseline are at /workspace/recent_traces/.

**Scope of edits (first-pass Phase H):**
  - You may ONLY edit /workspace/seed_harness/prompts.yaml.
  - The relevant keys are: agent.system_template, agent.instance_template,
    model.observation_template, model.format_error_template.
  - Do NOT change the bash-block fence convention (`mswea_bash_command`); that
    is parsed by the runtime.
  - Do NOT change the submission sentinel (`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`).
  - DO NOT edit agent.py / model.py / environment.py in this iteration
    (those edits won't take effect through the current Phase H eval path).

Make minimal, targeted changes. Explain each change with a THOUGHT and a
hypothesis. When done, exit cleanly.
"""


def copy_seed(dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SEED_HARNESS_SRC, dst, ignore=shutil.ignore_patterns("__pycache__"))


def import_check(seed_dir: Path) -> tuple[bool, str]:
    """D26 — verify each touched module can import."""
    code = (
        "import sys; sys.path.insert(0, '%s')\n"
        "from agent import DefaultAgent\n"
        "from model import LitellmTextbasedModel\n"
        "from environment import DockerEnvironment\n"
        "print('OK')"
    ) % seed_dir
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    return p.returncode == 0, p.stdout + p.stderr


def gather_recent_traces(out_dir: Path, gen_id: int, max_chars: int = 50000) -> Path:
    """Build a small bundle of recent failure traces from prior generations
    plus Phase 0 v2 for the meta-agent to inspect."""
    bundle = out_dir / f"gen_{gen_id}" / "recent_traces"
    bundle.mkdir(parents=True, exist_ok=True)
    # Pull the 3 most recent failed trajectories from Phase 0 v2
    v2 = PROJECT_ROOT.parent / "meta-harness-plan/results/phase0/0_50_ctx128k"
    if not v2.exists():
        v2 = Path("/home/t-hyunlee/meta-harness-plan/results/phase0/0_50_ctx128k")
    scores_p = v2.parent / "0_50_ctx128k_scores.json"
    if not scores_p.exists():
        scores_p = Path("/home/t-hyunlee/meta-harness-plan/results/phase0/0_50_ctx128k_scores.json")
    if not (v2.exists() and scores_p.exists()):
        return bundle
    scores = json.loads(scores_p.read_text())
    failed = [iid for iid, r in scores.items() if not r.get("pass")][:3]
    for iid in failed:
        src = v2 / iid / f"{iid}.traj.json"
        if src.exists():
            text = src.read_text()[:max_chars]
            (bundle / f"{iid}.traj.json").write_text(text)
    return bundle


def run_meta_agent(model_endpoint: str, model_name: str, work_dir: Path,
                   gen_id: int, out_dir: Path) -> dict:
    """Have Qwen3.5-4B edit the harness. We launch a docker sandbox with
    the seed_harness mounted RW, the recent traces RO, and run mini-swe-agent
    inside it with the META_AGENT_TASK prompt."""
    # For first-pass simplicity we just call the LLM directly with file-edit
    # tools provided via the mini-swe-agent DefaultAgent. The agent has bash
    # + nothing else; it will use sed/cat/grep to edit. Future: switch to
    # Hyperagents' agent/tools for cleaner file edit semantics.

    # Build a docker container with python3.11 + git
    container = f"phaseh-meta-gen{gen_id}-{int(time.time())}"
    seed_in_container = "/workspace/seed_harness"
    traces_in_container = "/workspace/recent_traces"
    gen_dir = out_dir / f"gen_{gen_id}"
    traces_src = gather_recent_traces(out_dir, gen_id)

    # Use a lightweight image with python + git; mini's docker env defaults
    # to having git available in the swebench images, so reuse the most-recent
    # xingyaoww image. For meta-edit we don't need test infrastructure.
    image = "python:3.11-slim"
    subprocess.run(["docker", "pull", image], capture_output=True, timeout=300)
    rc = subprocess.run([
        "docker", "run", "-d", "--name", container, "--rm",
        "--network=host",                              # so localhost:8001 reaches host vLLM
        "-v", f"{work_dir.resolve()}:{seed_in_container}",
        "-v", f"{traces_src.resolve()}:{traces_in_container}:ro",
        "-w", "/workspace", image, "sleep", "1800",
    ], capture_output=True, text=True, timeout=60)
    if rc.returncode != 0:
        return {"ok": False, "error": f"docker run failed: {rc.stderr[:300]}"}

    try:
        # init git in seed_harness to capture diffs
        subprocess.run(["docker", "exec", "-w", seed_in_container, container,
                        "bash", "-c",
                        "apt-get -qq update && apt-get -qq install -y git >/dev/null 2>&1; "
                        "git init -q && git add -A && git -c user.email=x@x.com -c user.name=x commit -qm seed"],
                       capture_output=True, text=True, timeout=120)

        # Run a single LLM-driven session: ask the model to propose edits and
        # apply them with bash inside the container. We avoid full mini-swe-agent
        # to keep the first-pass simple — direct OpenAI client call + a small
        # parsing loop. Iterations capped by step limit.
        client_script = f"""
from openai import OpenAI
import os, re, subprocess, json
client = OpenAI(base_url="{model_endpoint}", api_key="dummy")
sys_yaml = open('/workspace/seed_harness/prompts.yaml').read()[:8000]
agent_py = open('/workspace/seed_harness/agent.py').read()[:6000]
prompts_dir_listing = subprocess.run(['ls','/workspace/recent_traces'],capture_output=True,text=True).stdout

context = (
  '''You are editing a small AI agent harness. The mutable file is at
  /workspace/seed_harness/prompts.yaml (244 lines). Inner agent: Qwen3.5-4B.
  Recent failure traces from baseline at /workspace/recent_traces/ -> ''' + prompts_dir_listing
  + '''
  Goal: edit prompts.yaml to improve SWE-Gym solve rate.
  Allowed edits ONLY to:  agent.system_template, agent.instance_template,
                          model.observation_template, model.format_error_template.
  Do NOT change the bash-fence string `mswea_bash_command` or the submit sentinel.

  Use bash heredocs / sed to edit. Make ONE focused edit; explain your THOUGHT.
  Then run: git -C /workspace/seed_harness diff
  Then output a fenced bash block containing exactly:
    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
  to exit.

  Current prompts.yaml (first 4000 chars):
  ''' + sys_yaml[:4000])

messages = [
  {{'role':'system','content':'You are a careful agent harness modifier. Reason briefly, then act.'}},
  {{'role':'user','content':context}},
]
print('META> sending initial msg')
for step in range(8):
  try:
    r = client.chat.completions.create(model="{model_name}", messages=messages,
                                       temperature=0.3, max_tokens=2000)
  except Exception as e:
    print('META: API error', repr(e)[:300]); break
  msg = r.choices[0].message
  content = msg.content or ''
  print('META step',step,':',content[:300].replace('\\n',' / '))
  messages.append({{'role':'assistant','content':content}})
  blocks = re.findall(r'```(?:bash|sh)?\\s*(.+?)```', content, re.DOTALL)
  if not blocks:
    print('META: no bash block, stopping'); break
  cmd = blocks[0].strip()
  if 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT' in cmd:
    print('META: agent signaled done'); break
  out = subprocess.run(['bash','-c',cmd], capture_output=True, text=True, timeout=120)
  ob = (out.stdout + out.stderr)[:3000]
  print('META obs head:', ob[:200].replace('\\n',' / '))
  messages.append({{'role':'user','content':'<output>'+ob+'</output>'}})

diff = subprocess.run(['git','-C','/workspace/seed_harness','diff'], capture_output=True, text=True).stdout
print('META> final diff bytes:', len(diff))
open('/workspace/_meta_diff.patch','w').write(diff)
open('/workspace/_meta_trajectory.json','w').write(json.dumps(messages, indent=2)[:200000])
"""
        # pip install openai client inside the container
        subprocess.run(["docker", "exec", container, "bash", "-c",
                        "pip install --quiet openai 2>&1 | tail -2"],
                       capture_output=True, text=True, timeout=180)
        # write and run the meta-agent script
        subprocess.run(["docker", "exec", container, "bash", "-c",
                        f"cat > /workspace/_meta.py <<'__MPY__'\n{client_script}\n__MPY__"],
                       capture_output=True, text=True, timeout=60)
        run = subprocess.run(["docker", "exec", container, "python3", "/workspace/_meta.py"],
                             capture_output=True, text=True, timeout=900)
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "meta_agent.stdout.log").write_text(run.stdout + "\n---STDERR---\n" + run.stderr)

        # pull the diff back
        copy_back = subprocess.run([
            "docker", "cp", f"{container}:/workspace/_meta_diff.patch", str(gen_dir / "diff.patch")],
            capture_output=True, text=True)
        subprocess.run([
            "docker", "cp", f"{container}:/workspace/_meta_trajectory.json",
            str(gen_dir / "meta_trajectory.json")], capture_output=True, text=True)

        diff_path = gen_dir / "diff.patch"
        diff_text = diff_path.read_text() if diff_path.exists() else ""
        return {"ok": True, "diff_bytes": len(diff_text), "diff_path": str(diff_path)}
    finally:
        subprocess.run(["docker", "kill", container], capture_output=True, timeout=30)


def apply_diff_to_seed(diff_path: Path, candidate_dir: Path) -> bool:
    if not diff_path.exists() or diff_path.stat().st_size == 0:
        return False
    r = subprocess.run(["git", "-C", str(candidate_dir), "apply",
                        "--whitespace=nowarn", str(diff_path)],
                       capture_output=True, text=True)
    return r.returncode == 0


def evaluate_candidate(candidate_dir: Path, n_tasks: int, out_dir: Path,
                       model_name: str) -> dict:
    """Run mini-swe-agent batch on SWE-Gym-Lite[0:n_tasks] using the CANDIDATE's
    prompts.yaml as the agent config. mini's batch runner accepts `-c <yaml>`
    and merges it over its default config; passing the mutated prompts file
    makes the candidate's edits take effect WITHOUT touching the installed
    mini package.

    For first-pass Phase H this restricts the effective mutation surface to
    prompts.yaml (system_template / instance_template / observation_template /
    format_error_template). Edits to agent.py / model.py / environment.py
    in the candidate dir are not exercised — the meta-agent's instructions
    discourage them.
    """
    eval_out = out_dir / "eval"
    eval_out.mkdir(parents=True, exist_ok=True)
    candidate_prompts = candidate_dir / "prompts.yaml"
    if not candidate_prompts.exists():
        return {"ok": False, "error": f"missing prompts.yaml in {candidate_dir}"}
    r = subprocess.run([
        sys.executable, "-m", "meta.phase0.run_phase0",
        "--slice", f"0:{n_tasks}",
        "--workers", "4",
        "--model-name", f"hosted_vllm/{model_name}",
        "--config", str(candidate_prompts),
        "--output", str(eval_out),
    ], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=7200)
    (out_dir / "eval_runner.log").write_text(r.stdout + "\n---STDERR---\n" + r.stderr)

    # Score
    preds = eval_out / "preds.json"
    if not preds.exists():
        return {"ok": False, "error": "no preds.json"}
    scores_p = out_dir / "scores.json"
    r = subprocess.run([
        sys.executable, "-m", "meta.phase0.score_patches",
        "--preds", str(preds),
        "--output", str(scores_p),
        "--only-submitted", "--workers", "2",
    ], cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)
    (out_dir / "scorer.log").write_text(r.stdout + "\n---STDERR---\n" + r.stderr)
    if not scores_p.exists():
        return {"ok": False, "error": "no scores.json"}
    data = json.loads(scores_p.read_text())
    n_pass = sum(1 for v in data.values() if v.get("pass"))
    return {"ok": True, "pass": n_pass, "total_scored": len(data),
            "score": n_pass / max(len(data), 1), "scores_path": str(scores_p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--eval-samples", type=int, default=20)
    ap.add_argument("--out", default="/home/t-hyunlee/meta-harness-plan/results/phase_h")
    ap.add_argument("--model-endpoint", default="http://localhost:8001/v1")
    ap.add_argument("--model-name", default="qwen3.5-4b")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generation 0: copy seed verbatim (no edits) as the baseline
    print("=== gen_0: baseline (no edits) ===")
    gen0_dir = out_dir / "gen_0"
    candidate0 = gen0_dir / "seed_harness"
    copy_seed(candidate0)
    ok, msg = import_check(candidate0)
    if not ok:
        print(f"[gen_0] import check FAILED: {msg[:300]}"); sys.exit(1)
    print(f"[gen_0] import OK; eval ({args.eval_samples} tasks)...")
    res0 = evaluate_candidate(candidate0, args.eval_samples, gen0_dir, args.model_name)
    (gen0_dir / "eval_results.json").write_text(json.dumps(res0, indent=2))
    archive = [{"gen": 0, "score": res0.get("score", 0), "result": res0}]
    print(f"[gen_0] pass@1 = {res0.get('pass',0)}/{res0.get('total_scored',0)}")

    for gen in range(1, args.iters + 1):
        print(f"\n=== gen_{gen}: meta-agent proposes edits ===")
        gen_dir = out_dir / f"gen_{gen}"
        # Start from the best archive member's seed_harness
        best = max(archive, key=lambda a: a["score"])
        parent_dir = out_dir / f"gen_{best['gen']}" / "seed_harness"
        candidate_dir = gen_dir / "seed_harness"
        copy_seed(candidate_dir)   # copy seed; later phases could copy parent

        meta = run_meta_agent(args.model_endpoint, args.model_name, candidate_dir, gen, out_dir)
        (gen_dir / "meta_agent_result.json").write_text(json.dumps(meta, indent=2))
        if not meta.get("ok"):
            print(f"[gen_{gen}] meta-agent FAILED: {meta.get('error','?')[:200]}")
            continue
        # diff is already applied (the meta-agent edited files directly in candidate_dir)
        ok, msg = import_check(candidate_dir)
        if not ok:
            print(f"[gen_{gen}] D26 import check FAILED — rejecting candidate")
            (gen_dir / "rejected.txt").write_text(f"import check failed:\n{msg[:1000]}")
            continue

        print(f"[gen_{gen}] eval ({args.eval_samples} tasks)...")
        res = evaluate_candidate(candidate_dir, args.eval_samples, gen_dir, args.model_name)
        (gen_dir / "eval_results.json").write_text(json.dumps(res, indent=2))
        archive.append({"gen": gen, "score": res.get("score", 0), "result": res})
        print(f"[gen_{gen}] pass@1 = {res.get('pass',0)}/{res.get('total_scored',0)}  "
              f"(best so far: {max(a['score'] for a in archive)*100:.1f}%)")

    (out_dir / "summary.json").write_text(json.dumps({
        "archive": archive,
        "best_score": max(a["score"] for a in archive),
        "best_gen":   max(archive, key=lambda a: a["score"])["gen"],
    }, indent=2))
    print(f"\n[summary] best gen = {max(archive, key=lambda a: a['score'])['gen']}, "
          f"score = {max(a['score'] for a in archive)*100:.1f}%")


if __name__ == "__main__":
    main()
