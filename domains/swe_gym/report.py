"""SWE-Gym reporting — pass@1 over preds.json via the custom scorer.

`generate_loop.py`'s archive bookkeeping calls each domain's report() to get a
single score it can compare across candidates. We use it directly here.
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def report(output_dir: str | Path,
           workers: int = 2,
           only_submitted: bool = True,
           **_kw) -> dict:
    """Score the preds.json in output_dir and write scores.json next to it."""
    output_dir = Path(output_dir).resolve()
    preds = output_dir / "preds.json"
    if not preds.exists():
        raise FileNotFoundError(preds)

    out = output_dir / "scores.json"

    cmd = [
        sys.executable, "-m", "meta.phase0.score_patches",
        "--preds", str(preds),
        "--output", str(out),
        "--workers", str(workers),
    ]
    if only_submitted:
        cmd.append("--only-submitted")

    print("[swe_gym/report]", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
    if rc != 0:
        raise RuntimeError(f"scorer failed rc={rc}")

    data = json.loads(out.read_text())
    n_pass = sum(1 for v in data.values() if v.get("pass"))
    n_total = len(data)
    rate = n_pass / max(n_total, 1)
    summary = {
        "score": rate,
        "pass": n_pass,
        "total_scored": n_total,
        "scores_path": str(out),
    }
    print(f"[swe_gym/report] pass@1 = {n_pass}/{n_total} ({rate*100:.1f}%)")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--all", action="store_true",
                   help="Score every prediction, not just Submitted ones")
    args = p.parse_args()
    report(args.output_dir, workers=args.workers, only_submitted=not args.all)


if __name__ == "__main__":
    main()
