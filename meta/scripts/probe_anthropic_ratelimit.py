"""
Probe Anthropic OAuth rate limit clearing window.

Tries claude-sonnet-4-6 with increasing inter-call delays; logs request IDs +
any rate-limit / retry-after headers it can find.

Run: python3.11 -m meta.scripts.probe_anthropic_ratelimit
"""

import json
import os
import time
from pathlib import Path

from anthropic import Anthropic, RateLimitError

ENV = Path("/home/t-hyunlee/meta-harness-plan/.env")
env = dict(l.split("=",1) for l in ENV.read_text().strip().splitlines() if "=" in l)
token = env.get("ANTHROPIC_AUTH_TOKEN") or env.get("ANTHROPIC_API_KEY")
client = Anthropic(auth_token=token if token and token.startswith("sk-ant-oat") else None,
                   api_key=None if token and token.startswith("sk-ant-oat") else token)

MODELS = [
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-haiku-4-5",
]

# 1) snapshot: which models are accessible RIGHT NOW
print("\n=== model accessibility snapshot ===")
results = {}
for m in MODELS:
    try:
        r = client.messages.create(model=m, max_tokens=10,
            messages=[{"role":"user","content":"ok"}])
        results[m] = {"ok": True, "out": (r.content[0].text if r.content else "")[:40]}
        print(f"  {m:30s}  OK  ({r.usage.input_tokens} in, {r.usage.output_tokens} out)")
    except RateLimitError as e:
        rid = getattr(getattr(e, "response", None), "headers", {}).get("request-id", "?")
        results[m] = {"ok": False, "err": "429", "request_id": rid}
        print(f"  {m:30s}  429  request_id={rid}")
    except Exception as e:
        results[m] = {"ok": False, "err": repr(e)[:200]}
        print(f"  {m:30s}  ERR  {repr(e)[:120]}")
    time.sleep(2)

# 2) for the model that 429'd at first probe, find the clearing window
target = "claude-sonnet-4-6"
if not results.get(target, {}).get("ok"):
    print(f"\n=== {target} rate-limit clearing-window probe ===")
    # cumulative wait times (seconds)
    for cum in [10, 30, 60, 120, 300, 600, 1200, 1800, 3600]:
        wait = cum - sum(
            d for prior in [10, 30, 60, 120, 300, 600, 1200, 1800] if prior < cum
            for d in [prior]
        )
        # simpler: just sleep enough to reach cumulative `cum` since start
        print(f"  waiting until cumulative {cum}s mark ...", flush=True)
        time.sleep(max(5, cum // 4))    # ~4 sub-tries per stage isn't necessary; just one
        try:
            r = client.messages.create(model=target, max_tokens=10,
                messages=[{"role":"user","content":"ok"}])
            print(f"  CLEARED at t≈{cum}s (in={r.usage.input_tokens} out={r.usage.output_tokens})")
            results["clearing_window_sec"] = cum
            break
        except RateLimitError as e:
            rid = getattr(getattr(e,"response",None),"headers",{}).get("request-id","?")
            print(f"  still 429 at t≈{cum}s (rid={rid})")
        except Exception as e:
            print(f"  unexpected: {repr(e)[:200]}")
            break
    else:
        print(f"  did not clear within 1 hour probe window")

# 3) dump results
out = Path("/home/t-hyunlee/meta-harness-plan/results/anthropic_probe.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nwritten to {out}")
