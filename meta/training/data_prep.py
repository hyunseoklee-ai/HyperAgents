"""
Convert successful mini-swe-agent trajectories into SFT data for Phase 1.

Pipeline:
  1. Read trajectory directories (each subdir has <iid>/<iid>.traj.json).
  2. Filter to trajectories whose paired score is pass=True.
  3. For each, walk messages: keep role in {system, user, assistant}; drop
     mini's `extra` fields and the final `exit` message; collapse adjacent
     same-role messages so the chat template behaves.
  4. Write a JSONL where each line is {"messages": [...]} ready for TRL's
     SFTTrainer with assistant_only_loss=True.

Output format (per line):
  {"instance_id": "...",
   "messages": [
     {"role": "system", "content": "..."},
     {"role": "user",   "content": "..."},
     {"role": "assistant", "content": "..."},
     ...
   ]}

Usage:
  python3.11 -m meta.training.data_prep --config meta/training/lora_config.yaml
or:
  python3.11 -m meta.training.data_prep \\
      --traj-dir RESULTS/phase0/0_50 \\
      --scores  RESULTS/phase0/0_50_scores.json \\
      --output  meta/training/data/phase1_generator.jsonl
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def _msg_text(m: dict) -> str:
    """Flatten mini's message content (list-of-dicts or plain str)."""
    c = m.get("content", "")
    if isinstance(c, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                       for part in c)
    return c or ""


def trajectory_to_messages(traj: dict) -> list[dict]:
    """Return cleaned messages suitable for SFT."""
    out = []
    for m in traj.get("messages", []):
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            # drop 'exit' and any other special markers; the final assistant turn
            # before the exit already contains the submission command.
            continue
        text = _msg_text(m).strip()
        if not text:
            continue
        # Collapse adjacent same-role turns to satisfy chat-template invariants.
        if out and out[-1]["role"] == role:
            out[-1]["content"] = out[-1]["content"] + "\n\n" + text
        else:
            out.append({"role": role, "content": text})
    return out


def gather_passing(traj_dirs: Iterable[Path],
                   scores_paths: Iterable[Path]) -> dict[str, dict]:
    """Build {instance_id: {'traj_path': Path, 'pass': bool}}."""
    pass_set: set[str] = set()
    for sp in scores_paths:
        if not sp.exists():
            continue
        data = json.loads(sp.read_text())
        pass_set.update(iid for iid, r in data.items() if r.get("pass"))

    found: dict[str, dict] = {}
    for td in traj_dirs:
        if not td.exists():
            continue
        for d in td.iterdir():
            if not d.is_dir():
                continue
            iid = d.name
            tf = d / f"{iid}.traj.json"
            if not tf.exists():
                continue
            if iid in pass_set and iid not in found:
                found[iid] = {"traj_path": tf}
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None,
                   help="If set, load trajectory_dirs / scores_paths / output_jsonl from this YAML")
    p.add_argument("--traj-dir",   action="append", type=Path, default=[])
    p.add_argument("--scores",     action="append", type=Path, default=[])
    p.add_argument("--output",     type=Path, default=Path("meta/training/data/phase1_generator.jsonl"))
    p.add_argument("--drop-empty-assistant", action="store_true", default=True)
    args = p.parse_args()

    traj_dirs    = list(args.traj_dir)
    scores_paths = list(args.scores)
    output       = args.output

    if args.config:
        if yaml is None:
            raise RuntimeError("pyyaml required for --config")
        cfg = yaml.safe_load(args.config.read_text())["data"]
        traj_dirs    = [Path(p) for p in cfg["trajectory_dirs"]] or traj_dirs
        scores_paths = [Path(p) for p in cfg["scores_paths"]]    or scores_paths
        output       = Path(cfg.get("output_jsonl", output))

    print(f"Trajectory dirs: {[str(p) for p in traj_dirs]}")
    print(f"Score files:     {[str(p) for p in scores_paths]}")
    print(f"Output JSONL:    {output}")

    passing = gather_passing(traj_dirs, scores_paths)
    print(f"Passing trajectories found: {len(passing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped_too_short = 0
    n_skipped_no_asst = 0
    with output.open("w") as fout:
        for iid, info in sorted(passing.items()):
            traj = json.loads(info["traj_path"].read_text())
            messages = trajectory_to_messages(traj)
            if len(messages) < 3:                             # at least system + user + assistant
                n_skipped_too_short += 1
                continue
            if args.drop_empty_assistant and not any(m["role"] == "assistant" for m in messages):
                n_skipped_no_asst += 1
                continue
            fout.write(json.dumps({"instance_id": iid, "messages": messages}) + "\n")
            n_written += 1

    print(f"\nWrote {n_written} examples to {output}")
    if n_skipped_too_short: print(f"  (skipped {n_skipped_too_short} for being too short)")
    if n_skipped_no_asst:   print(f"  (skipped {n_skipped_no_asst} for no assistant turn)")

    # Token-count summary (loose; uses 4 chars/token heuristic if no tokenizer handy)
    total_chars = 0
    n_msgs = 0
    with output.open() as f:
        for line in f:
            obj = json.loads(line)
            for m in obj["messages"]:
                total_chars += len(m["content"])
                n_msgs += 1
    print(f"Total messages: {n_msgs}, total chars: {total_chars:,}, approx tokens: {total_chars//4:,}")


if __name__ == "__main__":
    main()
