"""
Catalog the v1 Phase-0 failure modes.

For each non-passing Submitted patch we load:
  - the agent-produced model_patch (from preds.json)
  - the scorer output (f2p / p2p counts + last 1500 chars of pytest stderr)

And classify into one of:
  GOLDEN          F2P pass + P2P pass  (these are the passes)
  REGRESSION      F2P pass, P2P fail   (fix works, breaks something downstream)
  WRONG_FIX       F2P fail, P2P pass   (benign but doesn't address the bug)
  CATASTROPHIC    F2P fail, P2P all-fail (patch breaks pytest collection — syntax/import)

For CATASTROPHIC patches we additionally try to extract the smoking-gun error
line from the pytest output tail so we can see WHAT Qwen3.5-4B did wrong.
"""

import argparse
import json
import re
from pathlib import Path


def classify(r: dict) -> str:
    f2p_pass, f2p_total = r.get("f2p_pass", 0), r.get("f2p_total", 0)
    p2p_pass, p2p_total = r.get("p2p_pass", 0), r.get("p2p_total", 0)
    f2p_ok = f2p_total > 0 and f2p_pass == f2p_total
    p2p_ok = p2p_total == 0 or p2p_pass == p2p_total

    if f2p_ok and p2p_ok:               return "GOLDEN"
    if f2p_ok and not p2p_ok:           return "REGRESSION"
    if not f2p_ok and p2p_ok:           return "WRONG_FIX"
    return "CATASTROPHIC"


def extract_first_error(text: str) -> str:
    """Pull the first pytest-style error or import failure line."""
    if not text:
        return ""
    patterns = [
        r"^E +.+",                   # pytest E lines
        r"SyntaxError: .+",
        r"IndentationError: .+",
        r"ImportError: .+",
        r"ModuleNotFoundError: .+",
        r"NameError: .+",
        r"AttributeError: .+",
        r"TypeError: .+",
        r"ValueError: .+",
        r"^ERRORS ",
        r"collection errors during collection",
    ]
    for line in text.splitlines():
        line = line.strip()
        for pat in patterns:
            if re.search(pat, line):
                return line[:200]
    # else: tail snippet
    return text.splitlines()[-1].strip()[:200] if text.strip() else ""


def patch_summary(patch: str) -> dict:
    if not patch:
        return {"empty": True}
    lines = patch.splitlines()
    files = [m.group(1) for m in (re.match(r"^diff --git a/(\S+) b/", l) for l in lines) if m]
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    return {"files": files, "added_lines": added, "removed_lines": removed,
            "total_lines": len(lines)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds",  type=Path,
                    default=Path("/home/t-hyunlee/meta-harness-plan/results/phase0/0_50/preds.json"))
    ap.add_argument("--scores", type=Path,
                    default=Path("/home/t-hyunlee/meta-harness-plan/results/phase0/0_50_scores.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("/home/t-hyunlee/meta-harness-plan/results/phase0/v1_failure_catalog.md"))
    args = ap.parse_args()

    preds  = json.loads(args.preds.read_text())
    scores = json.loads(args.scores.read_text())

    bucketed = {"GOLDEN": [], "REGRESSION": [], "WRONG_FIX": [], "CATASTROPHIC": []}
    rows = []
    for iid, s in scores.items():
        cls = classify(s)
        bucketed[cls].append(iid)
        rows.append({"iid": iid, "class": cls, **s,
                     "patch": (preds.get(iid) or {}).get("model_patch", "")})

    # ---------------------- write Markdown report ----------------------
    lines = []
    lines.append(f"# Phase 0 v1 — failure-mode catalog\n")
    lines.append(f"_Source: `{args.scores.name}` over `{args.preds.name}` (50 instances submitted = 18)._\n\n")

    lines.append("## Class distribution (over 18 Submitted)\n")
    lines.append("| Class | Count | Definition |")
    lines.append("|---|---:|---|")
    lines.append(f"| GOLDEN | {len(bucketed['GOLDEN'])} | F2P all pass, P2P all pass |")
    lines.append(f"| REGRESSION | {len(bucketed['REGRESSION'])} | F2P pass, P2P broken |")
    lines.append(f"| WRONG_FIX | {len(bucketed['WRONG_FIX'])} | F2P fail, P2P intact (benign-but-wrong) |")
    lines.append(f"| CATASTROPHIC | {len(bucketed['CATASTROPHIC'])} | F2P fail, P2P collapses (syntax/import) |")
    lines.append("")

    for cls in ("CATASTROPHIC", "REGRESSION", "WRONG_FIX", "GOLDEN"):
        items = [r for r in rows if r["class"] == cls]
        if not items: continue
        lines.append(f"\n## {cls}  ({len(items)} instances)\n")
        for r in items:
            iid = r["iid"]
            ps = patch_summary(r["patch"])
            files_str = ", ".join(ps.get("files", [])) or "(empty)"
            lines.append(f"### `{iid}`")
            lines.append(f"- f2p={r.get('f2p_pass',0)}/{r.get('f2p_total',0)},  "
                         f"p2p={r.get('p2p_pass',0)}/{r.get('p2p_total',0)}")
            lines.append(f"- Patch: {ps.get('added_lines',0)} added / {ps.get('removed_lines',0)} removed, files: {files_str}")
            if cls in ("CATASTROPHIC", "REGRESSION"):
                tail = r.get("pass_to_pass", {}).get("output_tail", "") if cls == "REGRESSION" \
                       else r.get("pass_to_pass", {}).get("output_tail", "") or r.get("fail_to_pass", {}).get("output_tail", "")
                err = extract_first_error(tail)
                if err:
                    lines.append(f"- Smoking gun: `{err}`")
                # also show last 30 lines of patch so we can see what was broken
                last_patch_lines = "\n".join(r["patch"].splitlines()[-30:])
                lines.append("- Final 30 lines of patch:")
                lines.append("```diff")
                lines.append(last_patch_lines)
                lines.append("```")
            lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")

    # also dump JSON for programmatic use
    js = args.output.with_suffix(".json")
    js.write_text(json.dumps({
        "buckets": bucketed,
        "rows": [{k: v for k, v in r.items() if k != "patch"} for r in rows],
    }, indent=2))
    print(f"wrote {js}")

    # console summary
    print("\nClass distribution:")
    for k, v in bucketed.items():
        print(f"  {k:14s} {len(v)} {v}")


if __name__ == "__main__":
    main()
