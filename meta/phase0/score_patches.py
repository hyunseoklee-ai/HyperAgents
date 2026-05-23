"""
SWE-Gym custom evaluator.

For each prediction in mini's preds.json, run the corresponding xingyaoww docker
image, apply test_patch + model_patch, run FAIL_TO_PASS and PASS_TO_PASS tests,
record pass/fail. Bypasses swebench's MAP_REPO_VERSION_TO_SPECS, which doesn't
cover SWE-Gym repos like getmoto/moto.

Pass criterion (matches SWE-bench convention):
  - All FAIL_TO_PASS tests pass
  - All PASS_TO_PASS tests still pass

Usage:
  python3.11 -m meta.phase0.score_patches \\
      --preds /home/t-hyunlee/meta-harness-plan/results/phase0/0_50/preds.json \\
      --workers 4 \\
      --output /home/t-hyunlee/meta-harness-plan/results/phase0/0_50_scores.json
"""

import argparse
import json
import subprocess
import textwrap
import shlex
import time
import sys
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datasets import load_dataset


def docker_image_for(iid: str) -> str:
    return f"docker.io/xingyaoww/sweb.eval.x86_64.{iid.replace('__','_s_').lower()}:latest"


def run_in_container(container: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
    """Run a shell command inside a running container; return (returncode, output)."""
    full = ["docker", "exec", "-w", "/testbed", container, "bash", "-lc", cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def parse_test_list(field) -> list[str]:
    """SWE-Gym stores PASS_TO_PASS / FAIL_TO_PASS as a string repr of a list."""
    if not field:
        return []
    if isinstance(field, list):
        return field
    try:
        return ast.literal_eval(field)
    except Exception:
        return []


def score_one(iid: str, model_patch: str, instance: dict, run_id: str) -> dict:
    image = docker_image_for(iid)
    container = f"phase0eval-{run_id}-{iid.replace('__','-')[:30]}-{int(time.time())}"
    result = {"instance_id": iid, "image": image, "pass": False,
              "fail_to_pass": [], "pass_to_pass": [], "error": "",
              "f2p_pass": 0, "f2p_total": 0, "p2p_pass": 0, "p2p_total": 0}

    fail_to_pass = parse_test_list(instance.get("FAIL_TO_PASS"))
    pass_to_pass = parse_test_list(instance.get("PASS_TO_PASS"))
    test_patch   = instance.get("test_patch") or ""

    result["f2p_total"] = len(fail_to_pass)
    result["p2p_total"] = len(pass_to_pass)

    # 1. Start container
    rc = subprocess.run(["docker", "run", "-d", "--name", container, "--rm",
                         "-w", "/testbed", image, "sleep", "1800"],
                        capture_output=True, text=True, timeout=120)
    if rc.returncode != 0:
        result["error"] = f"docker run failed: {rc.stderr[:300]}"
        return result

    try:
        # 2. Reset to clean state inside container (some xingyaoww images come patched already)
        run_in_container(container, "git stash -u --include-untracked >/dev/null 2>&1; git checkout -- . 2>/dev/null; true")

        # 3. Apply test_patch (writes the test files we'll evaluate against)
        if test_patch:
            cmd = "cat <<'__MSWE_TP__' | git apply --whitespace=nowarn -v - 2>&1\n" + test_patch + "\n__MSWE_TP__"
            code, out = run_in_container(container, cmd, timeout=120)
            if code != 0:
                result["error"] = f"test_patch apply failed (rc={code}): {out[:400]}"
                return result

        # 4. Apply model patch
        if not model_patch.strip():
            result["error"] = "empty model_patch"
            return result
        cmd = "cat <<'__MSWE_MP__' | git apply --whitespace=nowarn -v - 2>&1\n" + model_patch + "\n__MSWE_MP__"
        code, out = run_in_container(container, cmd, timeout=120)
        if code != 0:
            result["error"] = f"model_patch apply failed (rc={code}): {out[:400]}"
            return result

        # 5. Run FAIL_TO_PASS tests  (should pass after fix)
        def run_pytest(tests):
            if not tests:
                return [], "(no tests)"
            cmd = "python -m pytest -x --tb=short --no-header -q " + " ".join(shlex.quote(t) for t in tests)
            code, out = run_in_container(container, cmd, timeout=600)
            return code, out

        f2p_code, f2p_out = run_pytest(fail_to_pass)
        result["fail_to_pass"] = {"rc": f2p_code, "output_tail": f2p_out[-1500:]}
        result["f2p_pass"] = len(fail_to_pass) if f2p_code == 0 else 0

        # 6. Run PASS_TO_PASS tests  (should still pass after fix; regression check)
        # NOTE: SWE-bench convention only requires no NEW failures; we approximate as "all still pass"
        p2p_code, p2p_out = run_pytest(pass_to_pass)
        result["pass_to_pass"] = {"rc": p2p_code, "output_tail": p2p_out[-1500:]}
        result["p2p_pass"] = len(pass_to_pass) if p2p_code == 0 else 0

        result["pass"] = (f2p_code == 0) and (p2p_code == 0)
        return result

    finally:
        subprocess.run(["docker", "kill", container],
                       capture_output=True, text=True, timeout=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only-submitted", action="store_true",
                    help="If set, score only instances whose trajectory exited with status=Submitted")
    ap.add_argument("--traj-dir", default=None,
                    help="If --only-submitted, look here for {iid}/{iid}.traj.json")
    args = ap.parse_args()

    preds = json.loads(args.preds.read_text())
    print(f"Loaded {len(preds)} predictions")

    # Filter: only those with a non-empty diff
    candidates = {k: v for k, v in preds.items()
                  if v.get("model_patch") and "diff --git" in v["model_patch"]}
    print(f"With valid git diff: {len(candidates)}")

    if args.only_submitted:
        if not args.traj_dir:
            args.traj_dir = str(args.preds.parent)
        keep = {}
        for iid in candidates:
            tj = Path(args.traj_dir) / iid / f"{iid}.traj.json"
            if tj.exists():
                if json.loads(tj.read_text()).get("info",{}).get("exit_status") == "Submitted":
                    keep[iid] = candidates[iid]
        candidates = keep
        print(f"Filtered to Submitted-only: {len(candidates)}")

    # Load dataset for FAIL_TO_PASS / PASS_TO_PASS / test_patch
    print("Loading SWE-Gym-Lite ...")
    ds = load_dataset("SWE-Gym/SWE-Gym-Lite", split="train")
    by_iid = {row["instance_id"]: row for row in ds}

    run_id = str(int(time.time()))
    results = {}

    def task(iid):
        if iid not in by_iid:
            return iid, {"instance_id": iid, "pass": False, "error": "iid not in dataset"}
        return iid, score_one(iid, candidates[iid]["model_patch"], by_iid[iid], run_id)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, iid): iid for iid in candidates}
        for i, f in enumerate(as_completed(futs)):
            iid = futs[f]
            try:
                _, res = f.result()
            except Exception as e:
                res = {"instance_id": iid, "pass": False, "error": f"{type(e).__name__}: {e}"}
            results[iid] = res
            mark = "✓" if res.get("pass") else ("·" if res.get("f2p_pass", 0) > 0 else "✗")
            err = res.get("error","")[:80]
            print(f"  [{i+1}/{len(candidates)}] {mark} {iid:35s} f2p={res.get('f2p_pass',0)}/{res.get('f2p_total',0)} p2p={res.get('p2p_pass',0)}/{res.get('p2p_total',0)} {('('+err+')') if err else ''}")

    args.output.write_text(json.dumps(results, indent=2))
    n_pass = sum(1 for v in results.values() if v.get("pass"))
    elapsed = int(time.time() - t0)
    print(f"\nPass@1: {n_pass}/{len(results)} ({n_pass/max(len(results),1)*100:.1f}%)  in {elapsed//60}m{elapsed%60}s")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
