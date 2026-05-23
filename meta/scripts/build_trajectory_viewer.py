"""
Build a static HTML viewer for inspecting agent trajectories.

Scans the results dirs and produces a navigable site under
  /home/t-hyunlee/meta-harness-plan/viewer/

Goals (per the user's request):
  • make it easy to read what Qwen3.5-4B actually did
  • show inner / outer loops clearly separated
  • highlight failure-mode signals (command-not-found, repeated commands, ...)
  • support per-task drilldown

Run:
    python3.11 -m meta.scripts.build_trajectory_viewer
"""

import html
import json
import re
import shutil
from collections import Counter
from pathlib import Path

# -------- paths --------

ROOT = Path("/home/t-hyunlee/meta-harness-plan")
VIEWER = ROOT / "viewer"

PHASE0_V2_DIR  = ROOT / "results/phase0/0_50_ctx128k"
PHASE0_V2_SCORES = ROOT / "results/phase0/0_50_ctx128k_scores.json"

PHASE_H_DIR = ROOT / "results/phase_h"

PHASE_1_DIR = ROOT / "results/phase1/eval_0_50"
PHASE_1_SCORES = ROOT / "results/phase1/eval_0_50_scores.json"


# -------- shared CSS --------

CSS = """
:root {
  --bg:#fbfaf7; --panel:#ffffff; --ink:#1a1a1a; --muted:#5b5b5b; --line:#d9d4c8;
  --accent:#8a1c1c; --good:#1b6b1b; --bad:#a01010; --soft-bad:#b06600;
  --sys-bg:#f3f2ee; --user-bg:#eaf3fb; --asst-bg:#fff7e6; --tool-bg:#f0ece4;
}
*{box-sizing:border-box}
html,body{background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;line-height:1.6;margin:0;padding:0}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.container{max-width:1100px;margin:0 auto;padding:18px 22px 80px}
header.page-head{position:sticky;top:0;background:rgba(251,250,247,.95);backdrop-filter:saturate(180%) blur(8px);z-index:10;padding:10px 22px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:18px;justify-content:space-between}
header.page-head .crumbs{font-size:14px;color:var(--muted)}
header.page-head .crumbs a{color:var(--ink);margin-right:6px}
header.page-head .crumbs strong{color:var(--ink)}
header.page-head .badges{display:flex;gap:8px;flex-wrap:wrap}
.badge{display:inline-block;font-size:11.5px;letter-spacing:.04em;text-transform:uppercase;padding:2px 8px;border-radius:999px;background:#ece4d2;color:#5b4f30;font-weight:600;font-family:"Helvetica Neue",sans-serif}
.badge.good{background:#d8e8d4;color:#1f4117}
.badge.bad{background:#f6d6d6;color:#660}
.badge.bad{background:#f6d6d6;color:#7a1313}
.badge.warn{background:#f4ecc3;color:#7a5b00}
.badge.role-system{background:var(--sys-bg);color:#3a3a3a}
.badge.role-user{background:var(--user-bg);color:#1a4868}
.badge.role-assistant{background:var(--asst-bg);color:#7a4b00}
.badge.role-tool{background:var(--tool-bg);color:#3a3a3a}
.badge.role-exit{background:#f6d6d6;color:#7a1313}
h1{font-size:24px;margin:18px 0 4px;font-weight:700}
h2{font-size:18px;margin:24px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:15px;margin:18px 0 6px;font-weight:600}
.sub{color:var(--muted);font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin:14px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;font-size:14px}
.card a{display:block}
.card .meta{color:var(--muted);font-size:12.5px;margin-top:4px}
.banner{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0;font-size:14px}
.banner.bad{background:#fff3f3;border-color:#e9b8b8;color:#7a1313}
.banner.good{background:#f3fbef;border-color:#bcd6b1;color:#1f4117}
.banner.warn{background:#fffaea;border-color:#e0b34a;color:#7a4b00}
.sync-diagram{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:14px 0;font-family:"SF Mono",Consolas,Menlo,monospace;font-size:13px;white-space:pre;overflow-x:auto}
.msg{border:1px solid var(--line);border-radius:10px;margin:14px 0;background:var(--panel);position:relative}
.msg.role-system{background:var(--sys-bg)}
.msg.role-user{background:var(--user-bg)}
.msg.role-assistant{background:var(--asst-bg)}
.msg.role-tool{background:var(--tool-bg)}
.msg.role-exit{background:#fbeaea}
.msg.has-failure{box-shadow:inset 0 0 0 2px var(--bad)}
.msg.repeated{box-shadow:inset 0 0 0 2px var(--soft-bad)}
.msg .msg-head{padding:8px 14px;border-bottom:1px solid var(--line);font-size:13px;display:flex;align-items:center;gap:10px}
.msg .msg-num{color:var(--muted);font-family:"SF Mono",Consolas,monospace;font-size:12px}
.msg .msg-body{padding:10px 14px;font-size:14px;white-space:pre-wrap;word-break:break-word;font-family:"SF Mono",Consolas,monospace;font-size:13.5px;line-height:1.5}
.msg .msg-body.body-prose{font-family:-apple-system,BlinkMacSystemFont,Helvetica,sans-serif;font-size:14px}
.msg .msg-foot{padding:4px 14px 10px;color:var(--muted);font-size:12px}
.collapsible{margin:6px 0}
.collapsible > summary{cursor:pointer;color:var(--muted);font-size:12.5px;padding:2px 0;font-weight:600;list-style:none}
.collapsible > summary::-webkit-details-marker{display:none}
.collapsible > summary::before{content:"▶ ";font-size:10px;color:var(--muted)}
.collapsible[open] > summary::before{content:"▼ "}
pre.bash{background:#1a1a1a;color:#f4f4f4;padding:10px 12px;border-radius:6px;overflow-x:auto;font-family:"SF Mono",Consolas,monospace;font-size:12.5px;line-height:1.5;margin:6px 0}
pre.bash .prompt{color:#5fc88e}
pre.bash .err{color:#ffb1b1}
pre.diff{font-family:"SF Mono",Consolas,monospace;font-size:12.5px;line-height:1.45;background:#fafafa;border:1px solid var(--line);padding:8px 10px;border-radius:6px;overflow-x:auto}
pre.diff .add{background:#e6ffec;color:#1f4117}
pre.diff .del{background:#ffeef0;color:#7a1313}
.fail-note{background:#fff3f3;border-left:3px solid var(--bad);padding:6px 10px;margin:8px 0;font-size:13px;color:#7a1313}
.repeated-note{background:#fffaea;border-left:3px solid var(--soft-bad);padding:6px 10px;margin:8px 0;font-size:13px;color:#7a4b00}
.thinking{background:#f3f0e8;border-left:3px solid #aaa;padding:6px 10px;margin:6px 0;font-size:13px;color:#3a3a3a;font-style:italic}
.toc-pad{height:60px}
.kbd{font-family:"SF Mono",Consolas,monospace;font-size:11.5px;background:#f0ece4;padding:1px 4px;border-radius:3px;color:#5b5b5b}
"""


# -------- helpers --------

def get_content_text(msg) -> str:
    c = msg.get("content", "")
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c or ""


FAILURE_PATTERNS = [
    r"command not found",
    r"No such file or directory",
    r"^Traceback ",
    r"^E +.+",
    r"SyntaxError",
    r"NameError",
    r"ImportError",
    r"ModuleNotFoundError",
    r"AttributeError",
    r"TypeError",
    r"ValueError",
    r"\bbash: line",
    r"^Error:? ",
    r"FAILED",
    r"OutOfMemory",
    r"Permission denied",
]
_FAIL_RE = re.compile("|".join(FAILURE_PATTERNS), re.MULTILINE)

THINKING_RE = re.compile(r"(<think>.*?</think>)", re.DOTALL)


def extract_bash_blocks(text: str) -> list[tuple[str, str]]:
    """Return [(fence_label, content)] for triple-backtick fences."""
    blocks = []
    for m in re.finditer(r"```(\w*)\n?(.*?)```", text, re.DOTALL):
        blocks.append((m.group(1).strip(), m.group(2)))
    return blocks


def first_bash_command(text: str) -> str | None:
    blocks = extract_bash_blocks(text)
    for label, body in blocks:
        if label in ("bash", "sh", "mswea_bash_command", ""):
            return body.strip()
    return None


def render_role(role: str) -> str:
    role = role or "?"
    return f'<span class="badge role-{html.escape(role.lower())}">{html.escape(role)}</span>'


def render_message_body(text: str, role: str) -> str:
    """Render with thinking block collapse + bash block highlighting + diff highlighting."""
    if not text:
        return '<span class="sub"><em>(empty)</em></span>'
    parts = []
    # Pull out thinking blocks
    pos = 0
    for m in THINKING_RE.finditer(text):
        if m.start() > pos:
            parts.append(("text", text[pos:m.start()]))
        parts.append(("thinking", m.group(0)))
        pos = m.end()
    if pos < len(text):
        parts.append(("text", text[pos:]))

    out = []
    for kind, chunk in parts:
        if kind == "thinking":
            # strip <think>...</think>
            inner = chunk[len("<think>"):-len("</think>")] if chunk.startswith("<think>") else chunk
            out.append(
                f'<details class="collapsible thinking">'
                f'<summary>thinking ({len(inner)} chars)</summary>'
                f'<div>{html.escape(inner)}</div></details>'
            )
            continue
        # text part — split by bash blocks
        last = 0
        for m in re.finditer(r"```(\w*)\n?(.*?)```", chunk, re.DOTALL):
            if m.start() > last:
                seg = chunk[last:m.start()]
                out.append(f'<div class="body-prose">{html.escape(seg)}</div>')
            label = m.group(1).strip() or "bash"
            body = m.group(2)
            cls = "bash" if label in ("bash", "sh", "mswea_bash_command") else "bash"
            out.append(
                f'<div style="margin:6px 0;">'
                f'<div class="sub" style="margin-bottom:2px"><span class="kbd">```{html.escape(label)}</span></div>'
                f'<pre class="{cls}">{html.escape(body)}</pre>'
                f'</div>'
            )
            last = m.end()
        if last < len(chunk):
            out.append(f'<div class="body-prose">{html.escape(chunk[last:])}</div>')

    return "\n".join(out)


def msg_failure_signals(text: str) -> list[str]:
    return list({m.group(0) for m in _FAIL_RE.finditer(text)})


def render_trajectory(messages: list[dict], *, max_msg_chars: int = 100_000,
                      detect_repeats: bool = True) -> tuple[str, dict]:
    """Returns (html, summary_stats)."""
    out = []
    prev_bash = None
    stats = {"total": len(messages), "failures": 0, "repeats": 0, "by_role": Counter()}
    for i, m in enumerate(messages):
        role = m.get("role") or m.get("type") or "?"
        stats["by_role"][role] += 1
        text = get_content_text(m)
        if len(text) > max_msg_chars:
            text = text[:max_msg_chars] + f"\n\n[truncated: {len(text)-max_msg_chars} more chars]"

        failure_signals = msg_failure_signals(text)
        is_repeat = False
        cur_bash = first_bash_command(text) if role == "assistant" else None
        if detect_repeats and cur_bash and prev_bash and cur_bash == prev_bash:
            is_repeat = True
        if role == "assistant":
            prev_bash = cur_bash or prev_bash

        classes = ["msg", f"role-{role.lower()}"]
        if failure_signals:
            classes.append("has-failure")
            stats["failures"] += 1
        if is_repeat:
            classes.append("repeated")
            stats["repeats"] += 1

        body_html = render_message_body(text, role)

        annotations = ""
        if failure_signals:
            sig = ", ".join(failure_signals[:4])
            annotations += f'<div class="fail-note"><strong>Failure signal:</strong> {html.escape(sig)}</div>'
        if is_repeat:
            annotations += '<div class="repeated-note"><strong>Repeated command:</strong> same bash as the previous assistant turn</div>'

        out.append(
            f'<article id="msg-{i}" class="{ " ".join(classes) }">'
            f'<div class="msg-head">'
            f'{render_role(role)}'
            f'<span class="msg-num">#{i}</span>'
            f'</div>'
            f'<div class="msg-body">{body_html}</div>'
            f'{annotations}'
            f'</article>'
        )
    return "\n".join(out), stats


def page_shell(title: str, breadcrumbs_html: str, body_html: str,
               extra_head: str = "", extra_badges: str = "") -> str:
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="/viewer/style.css">{extra_head}
</head><body>
<header class="page-head">
  <div class="crumbs">{breadcrumbs_html}</div>
  <div class="badges">{extra_badges}</div>
</header>
<div class="container">
{body_html}
</div></body></html>"""


# -------- Claude CLI variant: outer page from claude_cli_result.json --------

def render_claude_cli_outer_page(gen_dir: Path, *, title: str, phase_label: str,
                                 diff_text: str = "") -> str:
    """Render the meta-agent page for the Claude CLI variant.

    Claude was invoked with `claude -p --output-format json`, so we only have
    the final aggregated result (no per-turn stream). The most informative
    content here is Sonnet's stated hypothesis (`result` field), the diff,
    and the token / cost breakdown. We surface these prominently rather than
    pretending to render turn-by-turn messages we never captured.
    """
    res_path = gen_dir / "claude_cli_result.json"
    data = json.loads(res_path.read_text())
    result_text = data.get("result", "")
    usage = data.get("usage", {})
    model_usage = data.get("modelUsage", {})
    duration_s = data.get("duration_ms", 0) / 1000
    cost = data.get("total_cost_usd", 0)
    turns = data.get("num_turns", "?")
    stop_reason = data.get("stop_reason", "?")
    terminal_reason = data.get("terminal_reason", "?")
    session_id = data.get("session_id", "")

    # cost / model table
    rows = []
    for m, u in sorted(model_usage.items(), key=lambda kv: -kv[1].get("costUSD", 0)):
        rows.append(
            f"<tr><td><code>{html.escape(m)}</code></td>"
            f"<td style='text-align:right'>{u.get('inputTokens',0):,}</td>"
            f"<td style='text-align:right'>{u.get('outputTokens',0):,}</td>"
            f"<td style='text-align:right'>{u.get('cacheReadInputTokens',0):,}</td>"
            f"<td style='text-align:right'>${u.get('costUSD',0):.4f}</td></tr>"
        )
    cost_table = (
        "<table style='border-collapse:collapse;margin:8px 0;font-size:13.5px'>"
        "<thead><tr><th style='text-align:left;border-bottom:1px solid #ccc;padding:4px 12px 4px 0'>Model</th>"
        "<th style='border-bottom:1px solid #ccc;padding:4px 12px 4px 0'>Input</th>"
        "<th style='border-bottom:1px solid #ccc;padding:4px 12px 4px 0'>Output</th>"
        "<th style='border-bottom:1px solid #ccc;padding:4px 12px 4px 0'>Cache read</th>"
        "<th style='border-bottom:1px solid #ccc;padding:4px 12px 4px 0'>Cost</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    # diff
    diff_html = ""
    if diff_text:
        diff_html += '<h2>Final diff produced by Sonnet</h2>'
        diff_html += '<pre class="diff">'
        for line in diff_text.splitlines():
            cls = ""
            if line.startswith("+") and not line.startswith("+++"): cls = "add"
            elif line.startswith("-") and not line.startswith("---"): cls = "del"
            diff_html += f'<span class="{cls}">{html.escape(line)}</span>\n'
        diff_html += '</pre>'

    # sync diagram (custom for the Claude CLI case)
    sync = f"""
<div class="sync-diagram">Claude Code harness (outer)              Our DGM scaffold (inner)
─────────────────────────              ─────────────────────────
  Sonnet 4.6 reasoning           ─►    {len(diff_text)}-byte diff applied to prompts.yaml
  + Claude Code's own tools             │
  + own router (sonnet↔haiku)           ▼
  + own prompt cache                    inner Qwen3.5-4B agents (×20)
  + own system prompt
  duration: {duration_s:.0f}s    turns: {turns}
  cost:     ${cost:.3f}        stop:  {stop_reason}</div>
"""

    body_html = f"""
<h1>{html.escape(title)}</h1>
<div class="sub">{phase_label}  ·  {turns} turns  ·  ${cost:.3f}  ·  {duration_s:.0f}s  ·  stop_reason: {stop_reason}  ·  terminal_reason: {terminal_reason}</div>

<div class="banner warn">
  Per-turn trajectory is not captured for this variant because Claude was
  invoked with <code>--output-format json</code>, which returns only the final
  aggregated result. To capture per-turn messages, future runs should use
  <code>--output-format stream-json</code>. The model's own final summary and
  hypothesis (below, verbatim) are the highest-signal artifact available.
</div>

<h2>Sonnet's final summary &amp; hypothesis (verbatim)</h2>
<div class="msg role-assistant">
  <div class="msg-head">
    <span class="badge role-assistant">assistant (final)</span>
    <span class="msg-num">session {html.escape(session_id[:8])}…</span>
  </div>
  <div class="msg-body body-prose">{html.escape(result_text)}</div>
</div>

{sync}
<h2>Token &amp; cost breakdown</h2>
{cost_table}
<div class="sub">Total: {usage.get('input_tokens',0):,} in &middot; {usage.get('output_tokens',0):,} out &middot; {usage.get('cache_read_input_tokens',0):,} cache_read &middot; <strong>${cost:.4f}</strong>.</div>

{diff_html}
"""
    crumbs = f'<a href="/viewer/">viewer</a> / Phase H gen_1 (Claude CLI) / <strong>outer trajectory</strong>'
    badges = (
        f'<span class="badge">role: outer</span>'
        f'<span class="badge good">diff: {len(diff_text)} bytes</span>'
        f'<span class="badge">${cost:.2f}</span>'
        f'<span class="badge">{turns} turns</span>'
    )
    return page_shell(title, crumbs, body_html, extra_badges=badges)


# -------- main per-trajectory inner viewer --------

def render_inner_traj_page(traj_path: Path, *, phase_label: str, score: dict | None,
                           link_back: str, prev_link: str = "", next_link: str = "") -> str:
    data = json.loads(traj_path.read_text())
    iid = traj_path.stem.replace(".traj", "")
    info = data.get("info", {})
    exit_status = info.get("exit_status", "")
    api_calls = info.get("model_stats", {}).get("api_calls", "?")
    submission = info.get("submission", "") or ""
    messages = data.get("messages", [])

    body, stats = render_trajectory(messages, detect_repeats=True)

    score_badge = ""
    if score:
        f2p = f"{score.get('f2p_pass',0)}/{score.get('f2p_total',0)}"
        p2p = f"{score.get('p2p_pass',0)}/{score.get('p2p_total',0)}"
        if score.get("pass"):
            score_badge = f'<span class="badge good">PASS  F2P {f2p}  P2P {p2p}</span>'
        else:
            score_badge = f'<span class="badge bad">FAIL  F2P {f2p}  P2P {p2p}</span>'

    extra_badges = (
        f'<span class="badge">exit: {html.escape(str(exit_status))}</span>'
        f'<span class="badge">steps: {api_calls}</span>'
        + score_badge
    )

    crumbs = f'<a href="/viewer/">viewer</a> / {phase_label} / <strong>{iid}</strong>'

    nav_html = ""
    if prev_link or next_link:
        bits = []
        if prev_link: bits.append(f'<a href="{prev_link}">← prev task</a>')
        bits.append(f'<a href="{link_back}">↑ index</a>')
        if next_link: bits.append(f'<a href="{next_link}">next task →</a>')
        nav_html = f'<div class="banner">{ "  ·  ".join(bits) }</div>'

    body_html = f"""
<h1>{html.escape(iid)}</h1>
<div class="sub">{phase_label}  ·  {len(messages)} messages  ·  {stats['failures']} failure-signaled msgs  ·  {stats['repeats']} repeated bash</div>
{nav_html}
"""

    # Submission block
    if submission:
        body_html += '<h2>Final submission (git diff)</h2>'
        body_html += '<pre class="diff">'
        for line in submission.splitlines():
            cls = ""
            if line.startswith("+") and not line.startswith("+++"): cls = "add"
            elif line.startswith("-") and not line.startswith("---"): cls = "del"
            body_html += f'<span class="{cls}">{html.escape(line)}</span>\n'
        body_html += '</pre>'

    body_html += '<h2>Conversation</h2>' + body
    return page_shell(f"{iid} — {phase_label}", crumbs, body_html, extra_badges=extra_badges)


# -------- outer (meta-agent) trajectory viewer --------

def render_outer_page(messages: list[dict], *, title: str, gen_id: int,
                      diff_text: str = "", related_inner_dir: str = "",
                      phase_label: str = "Phase H gen_1 outer (meta-agent)") -> str:
    body, stats = render_trajectory(messages, detect_repeats=True)

    diff_html = ""
    if diff_text:
        diff_html += '<h2>Final diff produced by meta-agent</h2>'
        diff_html += '<pre class="diff">'
        for line in diff_text.splitlines():
            cls = ""
            if line.startswith("+") and not line.startswith("+++"): cls = "add"
            elif line.startswith("-") and not line.startswith("---"): cls = "del"
            diff_html += f'<span class="{cls}">{html.escape(line)}</span>\n'
        diff_html += '</pre>'
    else:
        diff_html += '<div class="banner bad"><strong>Empty diff.</strong> The meta-agent did not produce any patch. The downstream inner trajectories for this generation use the unmodified seed prompts.yaml.</div>'

    sync = f"""
<div class="sync-diagram">Outer (meta-agent)              Inner (per-task agents)
────────────────────             ───────────────────────
1. read seed prompts.yaml        ↓ (uses prompts.yaml at time of dispatch)
2. think → propose edit          [{stats['total']} msgs above]
3. apply edit (git diff)         →   ← {('diff: ' + str(len(diff_text)) + ' bytes') if diff_text else 'EMPTY DIFF — no sync effect'}
4. exit                          ↓
                                 inner trajectories run on UNCHANGED prompts.yaml</div>
"""

    body_html = f"""
<h1>{html.escape(title)}</h1>
<div class="sub">{phase_label}  ·  {len(messages)} msgs  ·  failure-signaled {stats['failures']}  ·  repeated bash {stats['repeats']}</div>
{sync}
{diff_html}
<h2>Meta-agent conversation</h2>
{body}
"""
    crumbs = f'<a href="/viewer/">viewer</a> / Phase H gen_{gen_id} / <strong>outer trajectory</strong>'
    badges = (
        f'<span class="badge">role: outer</span>'
        f'<span class="badge warn">diff: {len(diff_text)} bytes</span>'
    )
    return page_shell(title, crumbs, body_html, extra_badges=badges)


# -------- INDEX per-phase --------

def render_phase_index(phase_label: str, traj_root: Path, scores: dict | None,
                       page_path_template: str, sync_note: str = "",
                       extra_top_note: str = "") -> str:
    """List all inner trajectories with one-line summary + score."""
    rows = []
    pass_count = 0
    fail_count = 0
    skipped = 0
    for d in sorted(traj_root.iterdir()):
        if not d.is_dir(): continue
        tf = d / f"{d.name}.traj.json"
        if not tf.exists():
            tf = d / "traj.json"  # alt
        if not tf.exists():
            skipped += 1
            continue
        try:
            t = json.loads(tf.read_text())
        except Exception:
            skipped += 1
            continue
        iid = d.name
        info = t.get("info", {})
        exit_status = info.get("exit_status", "?")
        api_calls = info.get("model_stats", {}).get("api_calls", "?")
        sc = (scores or {}).get(iid, None)
        if sc is None:
            cls = ""
            badge_html = '<span class="badge">unscored</span>'
        elif sc.get("pass"):
            cls = "card good-border"
            badge_html = '<span class="badge good">PASS</span>'
            pass_count += 1
        else:
            badge_html = '<span class="badge bad">FAIL</span>'
            fail_count += 1
            cls = "card bad-border"
        url = page_path_template.format(iid=iid)
        rows.append(
            f'<div class="card"><a href="{url}"><strong>{html.escape(iid)}</strong></a>'
            f'<div class="meta">exit: {html.escape(str(exit_status))} · steps {api_calls}'
            f' &nbsp; {badge_html}</div></div>'
        )

    summary = f"""
<div class="banner">Trajectories: {len(rows)}  ·  passed {pass_count}  ·  failed {fail_count}{'  ·  skipped ' + str(skipped) if skipped else ''}</div>
"""
    body = f"""
<h1>{html.escape(phase_label)}</h1>
{extra_top_note}
{summary}
{sync_note}
<h2>Inner trajectories</h2>
<div class="grid">{ "".join(rows) }</div>
"""
    crumbs = f'<a href="/viewer/">viewer</a> / <strong>{phase_label}</strong>'
    return page_shell(phase_label, crumbs, body)


# -------- top-level index helpers (per-variant gen_1 cards) --------

def _phase_h_card(stats: dict, slug: str, *, title: str, proposer: str,
                  ablation: str, scaffold: str) -> str:
    if not stats.get(f"{slug}_present"):
        return (
            f'<div class="card" style="opacity:.55">'
            f'<strong>{html.escape(title)}</strong>'
            f'<div class="meta">{html.escape(ablation)} &middot; proposer: {html.escape(proposer)}<br>'
            f'<em>not run yet</em></div></div>'
        )
    total = stats.get(f"{slug}_total", 0)
    passed = stats.get(f"{slug}_pass", 0)
    scored = stats.get(f"{slug}_scored", False)
    diff_b = stats.get(f"{slug}_diff_bytes", 0)
    if scored:
        score_html = f'pass <strong>{passed} / {total}</strong>'
    elif total > 0:
        score_html = f'<strong>{total} / 20</strong> trajectories saved &middot; <em>scoring pending</em>'
    else:
        score_html = '<em>no trajectories yet</em>'
    diff_html = (f'diff: <strong>{diff_b} B</strong>' if diff_b
                 else '<span style="color:var(--bad)">empty diff</span>')
    outer_link = f'<a href="outer/{slug}.html">outer trajectory</a>' if stats.get(f"{slug}_present") else ""
    return (
        f'<div class="card">'
        f'<a href="inner/{slug}/INDEX.html"><strong>{html.escape(title)}</strong></a>'
        f'<div class="meta">'
        f'{html.escape(ablation)} &middot; {html.escape(scaffold)}<br>'
        f'proposer: {html.escape(proposer)}<br>'
        f'{score_html} &middot; {diff_html} &middot; {outer_link}'
        f'</div></div>'
    )


def _ph_gen1_card_qwen4b(stats):
    return _phase_h_card(
        stats, "phase_h_gen1",
        title="gen_1 — Qwen3.5-4B proposer",
        proposer="Qwen3.5-4B",
        ablation="A2",
        scaffold="DGM ~200-LOC scaffold",
    )


def _ph_gen1_card_claude_cli(stats):
    return _phase_h_card(
        stats, "phase_h_gen1_claude_cli",
        title="gen_1 — Claude Sonnet 4.6 proposer (via Claude Code)",
        proposer="Sonnet 4.6 (via Claude Code, $0.687, 22 turns)",
        ablation="A2' (upper-bound; harness asymmetry)",
        scaffold="Claude Code production scaffold",
    )


def _ph_gen1_card_qwen35b(stats):
    return _phase_h_card(
        stats, "phase_h_gen1_qwen35b",
        title="gen_1 — Qwen3.5-35B-A3B-Int4 proposer",
        proposer="Qwen3.5-35B-A3B-GPTQ-Int4 (MoE, 3B active)",
        ablation="A2''' (matched-scaffold strong open-source)",
        scaffold="DGM ~200-LOC scaffold (same as A2)",
    )


# -------- top-level index --------

def render_top_index(stats: dict) -> str:
    body = f"""
<h1>Trajectory viewer</h1>
<div class="sub">Read agent runs side-by-side. Inner = per-task SWE-Gym agent. Outer = meta-agent that edits prompts between generations.</div>

<div class="banner warn">
  Heuristic annotations on every page:
  red border = message contains a failure signal (<span class="kbd">command not found</span>, <span class="kbd">Traceback</span>, <span class="kbd">No such file or directory</span>, etc.);
  orange border = same bash command as the immediately preceding assistant turn (stuck-loop indicator).
</div>

<h2>Phase H — harness-only evolve (DGM-style)</h2>
<div class="sync-diagram">                                      proposer (outer)               inner (per-task, 20)
                                      ─────────────────              ───────────────────
gen_0  baseline               →       (none)                    →    Qwen3.5-4B + seed prompts.yaml
gen_1  A2  (Qwen-4B propose)  →       Qwen3.5-4B   ────edit────►     Qwen3.5-4B + mutated prompts.yaml
gen_1  A2' (Claude propose)   →       Sonnet 4.6   ────edit────►     Qwen3.5-4B + mutated prompts.yaml
gen_1  A2''' (35B propose)    →       Qwen3.5-35B  ────edit────►     Qwen3.5-4B + mutated prompts.yaml

Inner is ALWAYS Qwen3.5-4B; only the proposer varies. A2 and A2''' share
the same DGM-style ~200-LOC scaffold (matched). A2' uses Claude Code's
production scaffold (harness asymmetry — upper-bound only).</div>

<h3>gen_0 baseline (no proposer)</h3>
<div class="grid">
  <div class="card">
    <a href="inner/phase_h_gen0/INDEX.html"><strong>Inner trajectories — gen_0 ({stats.get('phase_h_gen0_total',0)})</strong></a>
    <div class="meta">baseline inner agents, pass {stats.get('phase_h_gen0_pass',0)} / {stats.get('phase_h_gen0_total',0)}</div>
  </div>
</div>

<h3>gen_1 candidates (one card per proposer)</h3>
<div class="grid">
  {_ph_gen1_card_qwen4b(stats)}
  {_ph_gen1_card_claude_cli(stats)}
  {_ph_gen1_card_qwen35b(stats)}
</div>

<h2>Phase 0 v2 — baseline reference</h2>
<div class="sub">Same harness as gen_0; full 50-task set. Useful for spot-checks of pass / fail / context-exceeded modes.</div>
<div class="grid">
  <div class="card">
    <a href="inner/phase0_v2/INDEX.html"><strong>Inner trajectories — Phase 0 v2 ({stats.get('p0_v2_total',0)})</strong></a>
    <div class="meta">pass {stats.get('p0_v2_pass',0)} / {stats.get('p0_v2_total',0)} on the full 50-task smoke (128K context)</div>
  </div>
</div>

<h2>Phase 1 LoRA evaluation (A1)</h2>
<div class="sub">Same harness as Phase 0 v2 but with the Phase 1 LoRA adapter loaded (no harness mutation).</div>
<div class="grid">
  <div class="card">
    <a href="inner/phase_1/INDEX.html"><strong>Inner trajectories — A1 ({stats.get('p1_total',0)})</strong></a>
    <div class="meta">pass {stats.get('p1_pass',0)} / {stats.get('p1_total',0)} — A1 net regression of -4pp vs Phase 0 v2</div>
  </div>
</div>

<h2>Reading tips</h2>
<ul style="font-size:14.5px;line-height:1.7">
  <li><b>Inner loop</b> = system → user (task) → assistant (THOUGHT + bash) → user (exec output) → ...  Each task gets its own page.</li>
  <li><b>Outer loop</b> = meta-agent's separate run that <em>only</em> edits <code>prompts.yaml</code> in this implementation. Its result feeds the next generation's inner runs (or, if empty-diff, leaves them on the previous prompts).</li>
  <li>The <span class="kbd">sync diagram</span> at the top of every outer / index page shows the data flow.</li>
</ul>
"""
    return page_shell("Trajectory viewer", '<a href="/">dashboard</a> / <strong>viewer</strong>', body)


# -------- main build --------

def build():
    if VIEWER.exists():
        shutil.rmtree(VIEWER)
    VIEWER.mkdir(parents=True)
    (VIEWER / "style.css").write_text(CSS)

    # ---- top stats ----
    stats = {}

    # ---- Phase 0 v2 (reference) ----
    p0v2_scores = json.loads(PHASE0_V2_SCORES.read_text()) if PHASE0_V2_SCORES.exists() else {}
    out_dir = VIEWER / "inner/phase0_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    pass_count = 0
    traj_dirs = sorted([d for d in PHASE0_V2_DIR.iterdir() if d.is_dir()])
    for d in traj_dirs:
        iid = d.name
        tf = d / f"{iid}.traj.json"
        if not tf.exists(): continue
        page_html = render_inner_traj_page(
            tf, phase_label="Phase 0 v2",
            score=p0v2_scores.get(iid),
            link_back="INDEX.html",
        )
        (out_dir / f"{iid}.html").write_text(page_html)
        if p0v2_scores.get(iid, {}).get("pass"): pass_count += 1
    (out_dir / "INDEX.html").write_text(render_phase_index(
        "Phase 0 v2 — 50 tasks with base Qwen3.5-4B (128K)",
        PHASE0_V2_DIR, p0v2_scores,
        page_path_template="{iid}.html",
    ))
    stats["p0_v2_total"] = len(traj_dirs)
    stats["p0_v2_pass"]  = pass_count

    # ---- Phase 1 (A1) ----
    if PHASE_1_DIR.exists():
        p1_scores = json.loads(PHASE_1_SCORES.read_text()) if PHASE_1_SCORES.exists() else {}
        out_dir = VIEWER / "inner/phase_1"
        out_dir.mkdir(parents=True, exist_ok=True)
        traj_dirs = sorted([d for d in PHASE_1_DIR.iterdir() if d.is_dir()])
        pass_count = 0
        for d in traj_dirs:
            iid = d.name
            tf = d / f"{iid}.traj.json"
            if not tf.exists(): continue
            page_html = render_inner_traj_page(
                tf, phase_label="Phase 1 LoRA (A1)",
                score=p1_scores.get(iid),
                link_back="INDEX.html",
            )
            (out_dir / f"{iid}.html").write_text(page_html)
            if p1_scores.get(iid, {}).get("pass"): pass_count += 1
        (out_dir / "INDEX.html").write_text(render_phase_index(
            "Phase 1 LoRA evaluation (A1) — 50 tasks with qwen3.5-4b-phase1",
            PHASE_1_DIR, p1_scores,
            page_path_template="{iid}.html",
        ))
        stats["p1_total"] = len(traj_dirs)
        stats["p1_pass"]  = pass_count

    # ---- Phase H variants (gen_0 baseline + per-proposer gen_1 candidates) ----
    # Each variant is a separate slot in the viewer; we never overwrite or
    # conflate them. `result_dir_name` is the on-disk directory under
    # PHASE_H_DIR; `slug` is the viewer-side path component.
    PH_VARIANTS = [
        {
            "result_dir_name": "gen_0",
            "slug": "phase_h_gen0",
            "label": "Phase H gen_0 — baseline (no proposer; seed harness)",
            "short_label": "gen_0 baseline",
            "proposer": None,
            "outer_traj": False,
        },
        {
            "result_dir_name": "gen_1",
            "slug": "phase_h_gen1",
            "label": "Phase H gen_1 — proposer = Qwen3.5-4B (A2)",
            "short_label": "gen_1 (Qwen3.5-4B)",
            "proposer": "Qwen3.5-4B (DGM 200-LOC scaffold)",
            "outer_traj": True,
        },
        {
            "result_dir_name": "gen_1_claude_cli",
            "slug": "phase_h_gen1_claude_cli",
            "label": "Phase H gen_1 — proposer = Claude Sonnet 4.6 via Claude Code (A2')",
            "short_label": "gen_1 (Claude Sonnet via CLI)",
            "proposer": "Claude Sonnet 4.6 (Claude Code harness, asymmetric upper-bound)",
            "outer_traj": True,
        },
        {
            "result_dir_name": "gen_1_qwen35b",
            "slug": "phase_h_gen1_qwen35b",
            "label": "Phase H gen_1 — proposer = Qwen3.5-35B-A3B-Int4 (A2''')",
            "short_label": "gen_1 (Qwen3.5-35B-A3B-Int4)",
            "proposer": "Qwen3.5-35B-A3B-GPTQ-Int4 (matched-scaffold strong open-source)",
            "outer_traj": True,
        },
    ]

    outer_dir = VIEWER / "outer"
    outer_dir.mkdir(parents=True, exist_ok=True)

    for var in PH_VARIANTS:
        gen_dir = PHASE_H_DIR / var["result_dir_name"]
        slug = var["slug"]
        eval_dir = gen_dir / "eval"
        # A variant is "present" in the viewer iff it has at least one inner
        # trajectory or a meta-agent trajectory. An empty result dir (e.g.
        # pre-created for snapshot capture) does NOT count.
        has_inner = eval_dir.exists() and any(eval_dir.iterdir())
        has_outer = (gen_dir / "meta_trajectory.json").exists()
        if not (has_inner or has_outer):
            stats[f"{slug}_total"] = 0
            stats[f"{slug}_pass"] = 0
            stats[f"{slug}_present"] = False
            stats[f"{slug}_scored"] = False
            stats[f"{slug}_diff_bytes"] = 0
            continue
        stats[f"{slug}_present"] = True

        scores_path = gen_dir / "scores.json"
        ph_scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
        out_dir = VIEWER / f"inner/{slug}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if eval_dir.exists():
            traj_dirs = sorted([d for d in eval_dir.iterdir() if d.is_dir()])
        else:
            traj_dirs = []
        pass_count = 0
        for d in traj_dirs:
            iid = d.name
            tf = d / f"{iid}.traj.json"
            if not tf.exists(): continue
            page_html = render_inner_traj_page(
                tf, phase_label=var["label"],
                score=ph_scores.get(iid),
                link_back="INDEX.html",
            )
            (out_dir / f"{iid}.html").write_text(page_html)
            if ph_scores.get(iid, {}).get("pass"): pass_count += 1

        # sync banner — describes where prompts.yaml came from for this variant
        diff_path = gen_dir / "diff.patch"
        diff_size = diff_path.stat().st_size if diff_path.exists() else 0
        if var["proposer"] is None:
            sync_note = (
                '<div class="banner">Source of prompts.yaml: '
                '<strong>seed (verbatim copy of swebench_backticks.yaml)</strong>. '
                'No outer trajectory because this is the baseline.</div>'
            )
        else:
            outer_link = f'<a href="../../outer/{slug}.html">outer trajectory</a>'
            if diff_size == 0:
                sync_note = (
                    '<div class="banner bad">'
                    f'Proposer: <strong>{html.escape(var["proposer"])}</strong>. '
                    'Meta-agent ran but produced an <strong>empty diff</strong>, '
                    f'so this variant uses the same prompts as gen_0. See the {outer_link}.'
                    '</div>'
                )
            else:
                try:
                    diff_text = diff_path.read_text()
                    diff_lines = diff_text.splitlines()[:30]
                    diff_excerpt = "\n".join(diff_lines)
                except Exception:
                    diff_excerpt = "(could not read)"
                # progress note if scores not yet computed
                progress_note = ""
                if not scores_path.exists() and traj_dirs:
                    progress_note = (
                        f'<div class="banner warn" style="margin-top:8px">'
                        f'<strong>Scoring pending.</strong> '
                        f'{len(traj_dirs)} / 20 inner trajectories saved; '
                        f'scores.json will land once <code>score_patches</code> runs.</div>'
                    )
                sync_note = (
                    '<div class="banner good">'
                    f'Proposer: <strong>{html.escape(var["proposer"])}</strong>. '
                    f'Meta-agent applied a <strong>{diff_size}-byte diff</strong> '
                    f'(viewable at {outer_link}). Edit content excerpt:'
                    f'<pre class="diff" style="margin-top:8px">{html.escape(diff_excerpt)}</pre>'
                    f'{progress_note}'
                    '</div>'
                )
        if eval_dir.exists():
            (out_dir / "INDEX.html").write_text(render_phase_index(
                f"{var['label']} — inner trajectories",
                eval_dir, ph_scores,
                page_path_template="{iid}.html",
                sync_note=sync_note,
            ))
        else:
            # outer-only: meta-agent ran but no inner eval yet
            body = f"""
<h1>{html.escape(var['label'])}</h1>
{sync_note}
<div class="banner warn">Inner eval has not been run yet for this variant.</div>
"""
            (out_dir / "INDEX.html").write_text(page_shell(
                var["label"], f'<a href="/viewer/">viewer</a> / <strong>{html.escape(var["label"])}</strong>',
                body,
            ))
        stats[f"{slug}_total"] = len(traj_dirs)
        stats[f"{slug}_pass"] = pass_count
        stats[f"{slug}_scored"] = scores_path.exists()
        stats[f"{slug}_diff_bytes"] = diff_size

        # ---- outer (meta-agent) trajectory ----
        if var["outer_traj"]:
            diff_text = diff_path.read_text() if diff_path.exists() else ""
            cli_result_path = gen_dir / "claude_cli_result.json"
            meta_path = gen_dir / "meta_trajectory.json"
            if cli_result_path.exists():
                # Claude CLI variant — render from claude_cli_result.json
                (outer_dir / f"{slug}.html").write_text(
                    render_claude_cli_outer_page(
                        gen_dir,
                        title=var["label"] + " — meta-agent (outer loop)",
                        phase_label=var["label"],
                        diff_text=diff_text,
                    )
                )
            elif meta_path.exists():
                # v2-style (turn-by-turn meta_trajectory.json)
                meta_msgs = json.loads(meta_path.read_text())
                (outer_dir / f"{slug}.html").write_text(
                    render_outer_page(
                        meta_msgs,
                        title=var["label"] + " — meta-agent (outer loop)",
                        gen_id=1,
                        diff_text=diff_text,
                        phase_label=var["label"],
                    )
                )

    # ---- top index ----
    (VIEWER / "index.html").write_text(render_top_index(stats))

    print(f"Built viewer at {VIEWER}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    build()
