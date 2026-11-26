#!/usr/bin/env python
"""Build the complete consolidated IEEE two-column HTML + SVG research paper (Task 5)
from Group 12's Task 1-4 documents and verified iteration2 JSON results."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY_PATH = HERE / "results" / "iteration2" / "proposal3_summary.json"
OUT_HTML = HERE / "TASK5_IEEE_REPORT.html"


def svg_architecture() -> str:
    return """
<svg viewBox="0 0 720 225" width="100%" xmlns="http://www.w3.org/2000/svg" style="font-family: 'Times New Roman', serif; background:#fff;">
  <defs>
    <marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
      <polygon points="0 0, 7 3.5, 0 7" fill="#222"/>
    </marker>
    <marker id="arr-blue" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
      <polygon points="0 0, 7 3.5, 0 7" fill="#1d4ed8"/>
    </marker>
  </defs>

  <!-- Ollama Box -->
  <rect x="12" y="20" width="145" height="82" rx="4" fill="#f8fafc" stroke="#334155" stroke-width="1.2"/>
  <text x="84" y="38" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#0f172a">Local LLM (Ollama)</text>
  <text x="84" y="53" text-anchor="middle" font-size="9" fill="#334155">localhost:11434/v1</text>
  <text x="84" y="67" text-anchor="middle" font-size="8.5" fill="#475569">qwen2.5:3b (100% GPU)</text>
  <text x="84" y="79" text-anchor="middle" font-size="8.5" fill="#475569">qwen2.5:7b (21/79 CPU/GPU)</text>
  <text x="84" y="91" text-anchor="middle" font-size="8.5" fill="#475569">gemma4:12b (8GB / 6GB VRAM)</text>

  <!-- Client Box -->
  <rect x="195" y="14" width="215" height="196" rx="4" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.3"/>
  <text x="302" y="32" text-anchor="middle" font-size="10.5" font-weight="bold" fill="#1e3a8a">Evaluation Client (eval/server_agent.py)</text>
  <text x="302" y="45" text-anchor="middle" font-size="8.5" fill="#1e40af">T=0.0, concurrency=1, stratified_40.json</text>

  <rect x="208" y="54" width="189" height="44" rx="3" fill="#fff" stroke="#60a5fa" stroke-width="1"/>
  <text x="302" y="68" text-anchor="middle" font-size="9" font-weight="bold" fill="#1e3a8a">Conditions (eval/conditions.py)</text>
  <text x="302" y="80" text-anchor="middle" font-size="8" fill="#334155">c0: Baseline | c1: Repeat Warn</text>
  <text x="302" y="91" text-anchor="middle" font-size="8" fill="#334155">c2: Failure Memory | c4: c2 + c3</text>

  <rect x="208" y="106" width="189" height="58" rx="3" fill="#fef3c7" stroke="#d97706" stroke-width="1.1"/>
  <text x="302" y="120" text-anchor="middle" font-size="9" font-weight="bold" fill="#92400e">c3 Targeted Verify Gate</text>
  <text x="302" y="132" text-anchor="middle" font-size="8" fill="#78350f">If commit &amp; not fired &amp; (R_env - n_env &gt;= 2):</text>
  <text x="302" y="143" text-anchor="middle" font-size="8" fill="#78350f">Hold 1st commit, inject check prompt,</text>
  <text x="302" y="154" text-anchor="middle" font-size="8" fill="#78350f">require 2nd independent env command</text>

  <rect x="208" y="172" width="189" height="28" rx="3" fill="#fff" stroke="#94a3b8" stroke-width="1"/>
  <text x="302" y="185" text-anchor="middle" font-size="8.5" font-weight="bold" fill="#0f172a">JSONL Logger &amp; Analyzers</text>
  <text x="302" y="195" text-anchor="middle" font-size="8" fill="#475569">analyze_proposal3.py (T=0 filter, McNemar)</text>

  <!-- Controller Box -->
  <rect x="448" y="45" width="120" height="72" rx="4" fill="#f8fafc" stroke="#334155" stroke-width="1.2"/>
  <text x="508" y="63" text-anchor="middle" font-size="10" font-weight="bold" fill="#0f172a">AgentRL Controller</text>
  <text x="508" y="77" text-anchor="middle" font-size="8.5" fill="#334155">localhost:5020/api</text>
  <text x="508" y="90" text-anchor="middle" font-size="8" fill="#475569">/start_sample, /interact</text>
  <text x="508" y="102" text-anchor="middle" font-size="8" fill="#475569">Session TTL &amp; Routing</text>

  <!-- Redis Box -->
  <rect x="448" y="142" width="120" height="44" rx="4" fill="#fef2f2" stroke="#dc2626" stroke-width="1"/>
  <text x="508" y="160" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#991b1b">Redis State</text>
  <text x="508" y="174" text-anchor="middle" font-size="8" fill="#7f1d1d">Container Allocation</text>

  <!-- Workers -->
  <rect x="598" y="24" width="110" height="62" rx="4" fill="#ecfdf5" stroke="#059669" stroke-width="1.2"/>
  <text x="653" y="41" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#065f46">dbbench-std</text>
  <text x="653" y="54" text-anchor="middle" font-size="8" fill="#047857">300 items (20 sampled)</text>
  <text x="653" y="66" text-anchor="middle" font-size="8" fill="#047857">MySQL Docker, R=15</text>
  <text x="653" y="78" text-anchor="middle" font-size="7.5" fill="#064e3b">execute_sql / commit</text>

  <rect x="598" y="112" width="110" height="62" rx="4" fill="#ecfdf5" stroke="#059669" stroke-width="1.2"/>
  <text x="653" y="129" text-anchor="middle" font-size="9.5" font-weight="bold" fill="#065f46">os-std</text>
  <text x="653" y="142" text-anchor="middle" font-size="8" fill="#047857">144 items (20 sampled)</text>
  <text x="653" y="154" text-anchor="middle" font-size="8" fill="#047857">Ubuntu Docker, R=8</text>
  <text x="653" y="166" text-anchor="middle" font-size="7.5" fill="#064e3b">bash_action / answer</text>

  <!-- Arrows -->
  <line x1="157" y1="55" x2="195" y2="55" stroke="#222" stroke-width="1.2" marker-end="url(#arr)"/>
  <line x1="195" y1="70" x2="157" y2="70" stroke="#222" stroke-width="1.2" marker-end="url(#arr)"/>
  <text x="176" y="49" text-anchor="middle" font-size="7.5" fill="#334155">tools</text>

  <line x1="410" y1="75" x2="448" y2="75" stroke="#1d4ed8" stroke-width="1.2" marker-end="url(#arr-blue)"/>
  <line x1="448" y1="90" x2="410" y2="90" stroke="#1d4ed8" stroke-width="1.2" marker-end="url(#arr-blue)"/>
  <text x="429" y="69" text-anchor="middle" font-size="7.5" fill="#1e40af">HTTP</text>

  <line x1="568" y1="68" x2="598" y2="55" stroke="#222" stroke-width="1.1" marker-end="url(#arr)"/>
  <line x1="568" y1="95" x2="598" y2="140" stroke="#222" stroke-width="1.1" marker-end="url(#arr)"/>
  <line x1="508" y1="117" x2="508" y2="142" stroke="#222" stroke-width="1.1" marker-end="url(#arr)"/>
</svg>
"""


def svg_success_rates(data: dict) -> str:
    models = data["models"]
    bars = []
    specs = [
        ("qwen2.5:3b", "c0", "overall", "3B c0 All", "#64748b"),
        ("qwen2.5:3b", "c3", "overall", "3B c3 All", "#2563eb"),
        ("qwen2.5:3b", "c0", "db", "3B c0 DB", "#64748b"),
        ("qwen2.5:3b", "c3", "db", "3B c3 DB", "#2563eb"),
        ("qwen2.5:3b", "c0", "os", "3B c0 OS", "#64748b"),
        ("qwen2.5:3b", "c3", "os", "3B c3 OS", "#2563eb"),
        ("qwen2.5:7b", "c0", "overall", "7B c0 All", "#0d9488"),
        ("qwen2.5:7b", "c0", "db", "7B c0 DB", "#0d9488"),
        ("qwen2.5:7b", "c3", "db", "7B c3 DB", "#0284c7"),
        ("qwen2.5:7b", "c0", "os", "7B c0 OS", "#0d9488"),
        ("gemma4:12b", "c0", "db", "12B c0 DB*", "#d97706"),
        ("gemma4:12b", "c3", "db", "12B c3 DB*", "#ea580c"),
    ]
    for m, c, sp, lbl, col in specs:
        g = models[m]["conditions"][c][sp]
        lo, hi = g["success_ci95"]
        bars.append((lbl, g["success_rate"] * 100.0, lo * 100.0, hi * 100.0, g["n"], col))

    W, H = 700, 235
    left, right, top, bottom = 46, 18, 22, 52
    pw = W - left - right
    ph = H - top - bottom
    y_max = 90.0

    def y_px(val: float) -> float:
        return top + ph * (1.0 - val / y_max)

    lines = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" style="font-family: \'Times New Roman\', serif; background:#fff;">'
    ]
    for yv in (0, 20, 40, 60, 80):
        yp = y_px(yv)
        lines.append(f'<line x1="{left}" y1="{yp:.1f}" x2="{W - right}" y2="{yp:.1f}" stroke="#e2e8f0" stroke-width="0.9"/>')
        lines.append(f'<text x="{left - 6}" y="{yp + 3:.1f}" text-anchor="end" font-size="8.5" fill="#334155">{yv}%</text>')

    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + ph}" stroke="#334155" stroke-width="1"/>')
    lines.append(f'<line x1="{left}" y1="{top + ph}" x2="{W - right}" y2="{top + ph}" stroke="#334155" stroke-width="1"/>')

    step = pw / len(bars)
    bw = step * 0.56
    for idx, (lbl, rate, lo, hi, n, col) in enumerate(bars):
        cx = left + step * (idx + 0.5)
        x0 = cx - bw / 2
        y0 = y_px(rate)
        bh = y_px(0) - y0
        lines.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{col}" opacity="0.88"/>')
        ylo, yhi = y_px(lo), y_px(hi)
        lines.append(f'<line x1="{cx:.1f}" y1="{yhi:.1f}" x2="{cx:.1f}" y2="{ylo:.1f}" stroke="#0f172a" stroke-width="1.1"/>')
        lines.append(f'<line x1="{cx - 4:.1f}" y1="{yhi:.1f}" x2="{cx + 4:.1f}" y2="{yhi:.1f}" stroke="#0f172a" stroke-width="1.1"/>')
        lines.append(f'<line x1="{cx - 4:.1f}" y1="{ylo:.1f}" x2="{cx + 4:.1f}" y2="{ylo:.1f}" stroke="#0f172a" stroke-width="1.1"/>')
        lines.append(f'<text x="{cx:.1f}" y="{yhi - 4:.1f}" text-anchor="middle" font-size="8" font-weight="bold" fill="#0f172a">{rate:.1f}%</text>')
        lines.append(f'<text x="{cx:.1f}" y="{top + ph + 13:.1f}" text-anchor="middle" font-size="8" fill="#0f172a">{lbl}</text>')
        lines.append(f'<text x="{cx:.1f}" y="{top + ph + 24:.1f}" text-anchor="middle" font-size="7.5" fill="#475569">(N={n})</text>')

    leg_y = H - 10
    lines.append(f'<rect x="140" y="{leg_y - 8}" width="10" height="8" fill="#64748b"/><text x="154" y="{leg_y}" font-size="8" fill="#334155">qwen2.5:3b c0</text>')
    lines.append(f'<rect x="240" y="{leg_y - 8}" width="10" height="8" fill="#2563eb"/><text x="254" y="{leg_y}" font-size="8" fill="#334155">qwen2.5:3b c3</text>')
    lines.append(f'<rect x="340" y="{leg_y - 8}" width="10" height="8" fill="#0d9488"/><text x="354" y="{leg_y}" font-size="8" fill="#334155">qwen2.5:7b c0</text>')
    lines.append(f'<rect x="440" y="{leg_y - 8}" width="10" height="8" fill="#d97706"/><text x="454" y="{leg_y}" font-size="8" fill="#334155">gemma4:12b c0 (*partial N=8)</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def svg_gate_impact(data: dict) -> str:
    c0 = data["models"]["qwen2.5:3b"]["conditions"]["c0"]["overall"]
    c3 = data["models"]["qwen2.5:3b"]["conditions"]["c3"]["overall"]
    mcn = data["models"]["qwen2.5:3b"]["mcnemar_vs_c0"]["c3"]

    W, H = 340, 210
    left, top = 115, 22
    bar_max_w = 185
    metrics = [
        ("Success Rate", c0["success_rate"] * 100, c3["success_rate"] * 100),
        ("<=1 Cmd Rate", c0["answered_le_1_cmd_rate"] * 100, c3["answered_le_1_cmd_rate"] * 100),
        ("P(10,0.8) Repeat", c0["rouge_l_ge_08_rate"] * 100, c3["rouge_l_ge_08_rate"] * 100),
        ("Gate Fired Rate", c0["verify_gate_fired_rate"] * 100, c3["verify_gate_fired_rate"] * 100),
        ("Ans Changed Rate", c0["verify_changed_answer_rate"] * 100, c3["verify_changed_answer_rate"] * 100),
    ]
    lines = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" style="font-family: \'Times New Roman\', serif; background:#fff;">'
    ]
    row_h = 30
    for i, (name, v0, v3) in enumerate(metrics):
        y = top + i * row_h
        lines.append(f'<text x="{left - 6}" y="{y + 13}" text-anchor="end" font-size="8.5" fill="#0f172a">{name}</text>')
        w0 = bar_max_w * (v0 / 100.0)
        w3 = bar_max_w * (v3 / 100.0)
        lines.append(f'<rect x="{left}" y="{y + 2}" width="{w0:.1f}" height="9" fill="#64748b"/>')
        lines.append(f'<text x="{left + w0 + 4:.1f}" y="{y + 9.5}" font-size="7.5" fill="#334155">{v0:.1f}%</text>')
        lines.append(f'<rect x="{left}" y="{y + 13}" width="{w3:.1f}" height="9" fill="#2563eb"/>')
        lines.append(f'<text x="{left + w3 + 4:.1f}" y="{y + 20.5}" font-size="7.5" font-weight="bold" fill="#1e3a8a">{v3:.1f}%</text>')

    ov = mcn["overall"]
    db = mcn["db"]
    os_m = mcn["os"]
    lines.append(
        f'<text x="170" y="{H - 22}" text-anchor="middle" font-size="8" fill="#334155">'
        f'Paired flips (helped/harmed): Overall +{ov["helped_wrong_to_right"]}/-{ov["harmed_right_to_wrong"]} (p={ov["mcnemar_exact_p"]:.2f}), '
        f'DB +{db["helped_wrong_to_right"]}/-{db["harmed_right_to_wrong"]}, OS +{os_m["helped_wrong_to_right"]}/-{os_m["harmed_right_to_wrong"]}</text>'
    )
    lines.append(
        f'<rect x="85" y="{H - 12}" width="9" height="7" fill="#64748b"/><text x="97" y="{H - 6}" font-size="8" fill="#334155">c0 (2.15 calls, 16.3s)</text>'
    )
    lines.append(
        f'<rect x="195" y="{H - 12}" width="9" height="7" fill="#2563eb"/><text x="207" y="{H - 6}" font-size="8" fill="#334155">c3 (3.58 calls, 28.4s)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def svg_failure_modes(data: dict) -> str:
    models = data["models"]
    rows = [
        ("qwen2.5:3b c0 (N=40)", models["qwen2.5:3b"]["conditions"]["c0"]["overall"]),
        ("qwen2.5:3b c3 (N=40)", models["qwen2.5:3b"]["conditions"]["c3"]["overall"]),
        ("qwen2.5:7b c0 (N=38)", models["qwen2.5:7b"]["conditions"]["c0"]["overall"]),
    ]
    cats = [
        ("completed", "Completed (Pass)", "#16a34a"),
        ("early_unverified_answer", "Early <=1 Cmd", "#f59e0b"),
        ("wrong_answer_after_exploration", "Wrong After Expl.", "#ef4444"),
        ("repeated_action_loop_tle", "Repeat / TLE Loop", "#7c3aed"),
        ("format_or_no_tool_error", "Format / No Tool", "#64748b"),
    ]
    W, H = 340, 165
    left, top, bar_w, bar_h = 110, 18, 210, 22
    lines = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" style="font-family: \'Times New Roman\', serif; background:#fff;">'
    ]
    for i, (lbl, grp) in enumerate(rows):
        y = top + i * 34
        n = grp["n"]
        fm = grp["failure_modes"]
        lines.append(f'<text x="{left - 6}" y="{y + 14}" text-anchor="end" font-size="8" fill="#0f172a">{lbl}</text>')
        x_cur = float(left)
        for key, _, col in cats:
            cnt = fm.get(key, 0)
            if cnt <= 0:
                continue
            w = bar_w * (cnt / n)
            lines.append(f'<rect x="{x_cur:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{col}"/>')
            if w >= 5:
                lines.append(f'<text x="{x_cur + w / 2:.1f}" y="{y + 14}" text-anchor="middle" font-size="7.2" font-weight="bold" fill="#fff">{cnt}</text>')
            x_cur += w

    for j, (_, cname, col) in enumerate(cats):
        lx = 18 + (j % 3) * 108
        ly = 128 + (j // 3) * 15
        lines.append(f'<rect x="{lx}" y="{ly - 7}" width="8" height="8" fill="{col}"/>')
        lines.append(f'<text x="{lx + 11}" y="{ly}" font-size="7.5" fill="#334155">{cname}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def build_table_rows(data: dict) -> str:
    models = data["models"]
    order = [
        ("qwen2.5:3b", "c0", "overall", "OVERALL"),
        ("qwen2.5:3b", "c0", "db", "DB"),
        ("qwen2.5:3b", "c0", "os", "OS"),
        ("qwen2.5:3b", "c3", "overall", "OVERALL"),
        ("qwen2.5:3b", "c3", "db", "DB"),
        ("qwen2.5:3b", "c3", "os", "OS"),
        ("qwen2.5:7b", "c0", "overall", "OVERALL"),
        ("qwen2.5:7b", "c0", "db", "DB"),
        ("qwen2.5:7b", "c0", "os", "OS"),
        ("qwen2.5:7b", "c3", "db", "DB"),
        ("gemma4:12b", "c0", "db", "DB*"),
        ("gemma4:12b", "c3", "db", "DB*"),
    ]
    out = []
    for m, c, sp, sp_lbl in order:
        s = models[m]["conditions"][c][sp]
        mcn = models[m].get("mcnemar_vs_c0", {}).get(c, {}).get(sp)
        paired = f"{mcn['n_paired']}: {mcn['base_success_rate']*100:.1f}%&rarr;{mcn['cond_success_rate']*100:.1f}%" if mcn else "&mdash;"
        p_val = f"{mcn['mcnemar_exact_p']:.4f}" if mcn else "&mdash;"
        hh = f"+{mcn['helped_wrong_to_right']}/-{mcn['harmed_right_to_wrong']}" if mcn else "&mdash;"
        out.append(
            f"<tr>"
            f"<td><code>{m}</code></td><td><code>{c}</code></td><td>{sp_lbl}</td>"
            f"<td class='num'>{s['n']}</td>"
            f"<td class='num'>{s['success_rate']*100:.1f}%</td>"
            f"<td class='num'>{paired}</td>"
            f"<td class='num'>{s['answered_le_1_cmd_rate']*100:.1f}%</td>"
            f"<td class='num'>{s['tle_rate']*100:.1f}%</td>"
            f"<td class='num'>{s['mean_env_calls']:.2f}</td>"
            f"<td class='num'>{s['rouge_l_ge_08_rate']*100:.1f}%</td>"
            f"<td class='num'>{s['mean_latency_s']:.1f}s</td>"
            f"<td class='num'>{p_val}</td>"
            f"<td class='num'>{hh}</td>"
            f"</tr>"
        )
    return "\n".join(out)


def main() -> None:
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Failure-Aware Tool-Use Agents and Trajectory-Aware Evaluation on AgentBench FC</title>
<style>
  @page {{
    size: letter;
    margin: 0.68in 0.58in 0.72in 0.58in;
    @bottom-center {{
      content: counter(page);
      font-family: "Times New Roman", Times, serif;
      font-size: 8.5pt;
      color: #334155;
    }}
  }}
  body {{
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 9.5pt;
    line-height: 1.32;
    color: #111;
    margin: 0;
  }}
  .title-block {{
    text-align: center;
    margin-bottom: 11pt;
    padding-bottom: 7pt;
    border-bottom: 1px solid #cbd5e1;
  }}
  h1.paper-title {{
    font-size: 17.5pt;
    font-weight: bold;
    line-height: 1.2;
    margin: 0 0 6pt 0;
  }}
  .authors {{
    font-size: 9.5pt;
    margin: 0 0 2pt 0;
  }}
  .affil {{
    font-size: 8.8pt;
    font-style: italic;
    color: #334155;
    margin: 0;
  }}
  .two-col {{
    column-count: 2;
    column-gap: 0.22in;
    text-align: justify;
  }}
  .span-all {{
    column-span: all;
    margin: 7pt 0;
    break-inside: avoid;
  }}
  .avoid-break {{
    break-inside: avoid;
  }}
  .abstract, .keywords {{
    font-size: 9pt;
    line-height: 1.28;
    margin: 0 0 7pt 0;
  }}
  h2.sec {{
    font-size: 10pt;
    font-weight: bold;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 0.3pt;
    margin: 10pt 0 4pt 0;
    break-after: avoid;
  }}
  h3.subsec {{
    font-size: 9.5pt;
    font-weight: bold;
    font-style: italic;
    margin: 7pt 0 3pt 0;
    break-after: avoid;
  }}
  p {{
    margin: 0 0 5pt 0;
    text-indent: 1em;
  }}
  p.noindent {{
    text-indent: 0;
  }}
  ul, ol {{
    margin: 3pt 0 6pt 0;
    padding-left: 15pt;
  }}
  li {{
    margin-bottom: 2.5pt;
  }}
  code {{
    font-family: "Consolas", "Courier New", monospace;
    font-size: 8.2pt;
    background: #f1f5f9;
    padding: 0 2pt;
    border-radius: 2pt;
  }}
  .eq {{
    text-align: center;
    font-style: italic;
    margin: 5pt 0;
    font-size: 9.3pt;
  }}
  .fig-box {{
    border: 0.75pt solid #cbd5e1;
    padding: 5pt;
    background: #fff;
    margin: 6pt 0;
    break-inside: avoid;
  }}
  .caption {{
    font-size: 8.4pt;
    line-height: 1.25;
    margin-top: 4pt;
    text-align: left;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 7.9pt;
    margin: 4pt 0;
  }}
  th, td {{
    border-top: 0.6pt solid #94a3b8;
    border-bottom: 0.6pt solid #94a3b8;
    padding: 2.3pt 3pt;
    text-align: left;
  }}
  th {{
    background: #f8fafc;
    font-weight: bold;
    border-top: 1.2pt solid #0f172a;
    border-bottom: 1pt solid #0f172a;
  }}
  tr:last-child td {{
    border-bottom: 1.2pt solid #0f172a;
  }}
  td.num, th.num {{
    text-align: right;
  }}
  .ref-list {{
    font-size: 8.2pt;
    line-height: 1.25;
    padding-left: 0;
    list-style: none;
  }}
  .ref-list li {{
    padding-left: 16pt;
    text-indent: -16pt;
    margin-bottom: 3pt;
  }}
</style>
</head>
<body>

<div class="title-block">
  <h1 class="paper-title">Failure-Aware Tool-Use Agents on AgentBench FC: Empirical Verify-Gate Study and IRT Scaffolding</h1>
  <p class="authors"><strong>Group 12:</strong> Abhijith S Kumar (2024A7PS0557G), Joseph Sebastian (2024A8PS0681G), Rahul Sunil (2024A7PS0050G), D A Pavan Shenoy (2024A7PS0584G), Shreyash Singh (2024ADPS0860G)</p>
  <p class="affil">Department of Computer Science &amp; Information Systems, BITS Pilani K. K. Birla Goa Campus</p>
</div>

<div class="two-col">

  <p class="abstract"><strong><em>Abstract</em>&mdash;Evaluating Large Language Models (LLMs) as autonomous tool-using agents requires multi-turn interaction in stateful environments, yet standard benchmarks rely on single greedy runs, coarse symptom-level failure buckets, and ad hoc reciprocal-weighted aggregation. In this work, we consolidate our end-to-end investigation of AgentBench Function Calling (FC) across <code>dbbench-std</code> and <code>os-std</code>. Guided by an Iteration 1 pilot (<code>Qwen3-8B</code>, <em>N</em> = 10/task) that revealed 9/10 OS failures stemmed from answering after a single unverified command rather than Task-Limit-Exceeded (TLE) loops, we evolved our design into a client-side failure-aware intervention suite (<code>c0</code>&ndash;<code>c4</code>) featuring a budget-aware targeted verify gate (<code>c3</code>), alongside implemented (but unevaluated) Trajectory-Aware Bayesian Item Response Theory (IRT) scaffolding. In Iteration 2, we evaluated local open-weight models via Ollama on a stratified 40-item subset (20 DB + 20 OS, <em>T</em> = 0.0). Across 163 valid episodes, <code>qwen2.5:3b</code> (80 episodes) showed that <code>c3</code> reliably enforced multi-step verification (reducing &le;1-command answers from 50.0% to 10.0%, gate fired on 67.5% of episodes) at ~2&times; prompt-token (4,241.4 &rarr; 8,403.9) and latency (16.30 s &rarr; 28.42 s) cost, without a statistically significant accuracy gain (27.5% &rarr; 25.0%, McNemar exact <em>p</em> = 1.0000). Paired DBBench runs on <code>qwen2.5:7b</code> (45.0% &rarr; 40.0%, <em>p</em> = 1.0000) and partial <code>gemma4:12b</code> (8 paired: 50.0% &rarr; 50.0%, <em>p</em> = 1.0000) showed the same pattern while hitting controller session-expiration timeouts under 6 GB VRAM constraints. Proposal 1 IRT analysis was skipped because the local runs did not cover a multi-model/condition grid.</strong></p>

  <p class="keywords"><strong><em>Index Terms</em>&mdash;LLM agents, tool use, AgentBench FC, verification gate, failure memory, Bayesian IRT, local inference.</strong></p>

  <h2 class="sec">I. Introduction</h2>

  <h3 class="subsec">A. Background Motivation and Problem Definition</h3>
  <p>The evaluation of Large Language Models (LLMs) has shifted from static, single-turn natural language tasks toward autonomous multi-turn agents capable of external tool invocation and environment interaction [1]. However, systematic gap analysis of AgentBench [1] reveals five structural limitations: (1) <em>lack of statistical rigor</em>, as greedy decoding (<em>T</em> = 0) with a single run per instance and uneven test splits (e.g., 20 Digital Card Game queries vs. 144 OS and 200 Web Shopping items) provides no confidence intervals or variance estimates; (2) <em>coarse, confounded failure diagnosis</em>, as Chain-of-Thought (CoT) prompting is evaluated alone with a five-bucket symptom taxonomy (CLE/IF/IA/TLE/Complete) and a single ROUGE-L repetition metric <em>P</em>(<em>n</em>, <em>t</em>) applied only to TLE cases; (3) <em>unvalidated automated judges</em> in Lateral Thinking Puzzles (Appendix F.2); (4) <em>incommensurable metric aggregation</em>, collapsing Success Rate, F1, Reward, and Game Progress via 2023-pool reciprocal weights into one Overall Score; and (5) <em>unablated 3,500-token context truncation</em> on long-horizon tasks.</p>
  <p>Furthermore, our Iteration 1 pilot on AgentBench FC (<code>Qwen3-8B</code>) uncovered a critical behavioral shift: whereas AgentBench v0.2 reported TLE repetition loops (avg. 25.5 rounds, &gt;90% with ROUGE-L &ge; 0.8) as the dominant failure, modern tool-calling 3B&ndash;8B models frequently commit an answer after &le;1 command without verifying tool outputs (9/10 OS pilot failures; 50.0% of <code>qwen2.5:3b</code> and 36.8% of <code>qwen2.5:7b</code> baseline episodes).</p>

  <h3 class="subsec">B. Contributions</h3>
  <ol>
    <li><strong>AgentRL FC Protocol Alignment:</strong> Diagnosed and resolved v0.2-to-FC incompatibilities (AgentRL controller on port <code>5020</code>, <code>session_id</code> headers, incremental <code>messages</code>, OpenAI tool schemas, LF worker scripts, and <code>aiodocker==0.24.0</code>).</li>
    <li><strong>Client-Side Failure-Aware Suite (<code>c0</code>&ndash;<code>c4</code>):</strong> Implemented modular interventions in <code>eval/conditions.py</code>&mdash;repeated-action warnings (<code>c1</code>), state-conditioned failure memory (<code>c2</code>), a budget-aware targeted verify gate (<code>c3</code>), and their combination (<code>c4</code>)&mdash;without altering environment grading.</li>
    <li><strong>Trajectory-Aware Bayesian IRT Scaffolding (Implemented, Not Evaluated):</strong> Implemented 1PL, 2PL, and trajectory-aware 2PL IRT models (<code>eval/irt.py</code>, <code>eval/analyze_proposal1.py</code>) with Gaussian priors and Laplace uncertainty, alongside temperature-filtered paired McNemar analysis (<code>eval/analyze_proposal3.py</code>); IRT evaluation was skipped due to insufficient multi-model/condition coverage.</li>
    <li><strong>Empirical Local-Hardware Evaluation &amp; Refinement Log:</strong> Executed 163 verified <em>T</em> = 0.0 episodes across <code>qwen2.5:3b</code>, <code>qwen2.5:7b</code>, and <code>gemma4:12b</code> on <code>stratified_40.json</code>, documenting exact McNemar tests, compute-latency trade-offs, session-timeout bottlenecks, and two LLM refinement rounds.</li>
  </ol>

  <h2 class="sec">II. Related Work</h2>
  <p class="noindent">Following our automated literature retrieval and manual filtering (excluding unrelated physical-dynamics papers and off-topic multimodal synergy studies [5]), prior work organizes into three areas:</p>

  <h3 class="subsec">A. Interactive LLM Agent Benchmarks</h3>
  <p>AgentBench [1] evaluates 29 LLMs across eight interactive environments using text-prompted interaction and reciprocal-weighted scoring, later updated to the October 2025 Function Calling (FC) release on AgentRL. While AgentBench establishes capability gaps between commercial and open-weight models, its single-run greedy protocol motivates budget-efficient, statistically grounded re-evaluation.</p>

  <h3 class="subsec">B. Trajectory-Aware Evaluation and Item Response Theory</h3>
  <p>Recent work on Efficient SWE Agent Benchmarking introduces PTA-IRT [2], demonstrating that execution trajectories (search paths, intermediate diffs, and logs) serve as privileged information within an Item Response Theory framework to recover full-benchmark rankings from small calibration subsets (Kendall&rsquo;s &tau; &approx; 0.89, Spearman&rsquo;s &rho; &approx; 0.97). We implement this paradigm as scaffolding for AgentBench&rsquo;s multi-environment (DB + OS) setting. Separately, the Task 4 assignment brief lists four background papers on automated scientific discovery and agentic optimization (DeepScientist <code>arXiv:2509.26603</code>, EvoScientist <code>arXiv:2603.08127</code>, AutoSOTA <code>arXiv:2604.05550</code>, and Arbor <code>arXiv:2606.11926</code>); none were read or used in our implementation.</p>

  <h3 class="subsec">C. Failure-Aware Agents, Self-Correction, and Automated Judges</h3>
  <p>Huang et al. [4] show that LLMs rarely self-correct reasoning from generic prompts without external feedback, explaining why our Iteration 1 generic verify prompt failed to flip any outcome and motivating our structured, independent-command verify gate (<code>c3</code>). Separately, mechanistic interpretability of LLM-as-a-Judge models (Themis, Prometheus) in &ldquo;Beyond Scores&rdquo; [3] reveals two-stage evaluation circuits (early-layer discrepancy routing and late-layer MLP integration), motivating our dropped Proposal 2 judge-auditing line.</p>

  <h2 class="sec">III. Problem Formulation</h2>

  <h3 class="subsec">A. Multi-Turn Episode with Client-Side Interventions</h3>
  <p>Each benchmark item <em>i</em> &isin; <em>I</em><sub>DB</sub> &cup; <em>I</em><sub>OS</sub> defines an interactive episode over turns <em>t</em> = 1, &hellip;, <em>R</em><sub>env</sub> with round budget <em>R</em><sub>DB</sub> = 15 and <em>R</em><sub>OS</sub> = 8. At turn <em>t</em>, given history <em>h<sub>t</sub></em>, policy &pi;<sub>&theta;</sub> emits tool calls <em>a<sub>t</sub></em> &isin; <em>T</em><sub>env</sub> &cup; <em>T</em><sub>commit</sub> (where <em>T</em><sub>env</sub> = {{<code>execute_sql</code>}} or {{<code>bash_action</code>}}). A client-side operator &Phi;<sub><em>c</em></sub>(<em>h<sub>t</sub></em>, <em>a<sub>t</sub></em>, <em>s<sub>t</sub></em>) maintains episode state <em>s<sub>t</sub></em> (environment call count <em>n</em><sub>env</sub>, normalized action history, and failure records) without altering server reward <em>y<sub>m,i</sub></em> &isin; {{0, 1}}:</p>
  <div class="eq">&Phi;<sub><em>c</em>3</sub> holds commit <em>a<sub>t</sub></em> &hArr; (not fired) &and; (<em>R</em><sub>env</sub> &minus; <em>n</em><sub>env</sub> &ge; 2)</div>
  <p class="noindent">When triggered, &Phi;<sub><em>c</em>3</sub> injects a synthetic tool-hold response and requires the agent to map question constraints to observed outputs and execute a second independent <em>T</em><sub>env</sub> command before re-committing.</p>

  <h3 class="subsec">B. Trajectory-Aware Bayesian IRT</h3>
  <p>Given an observed binary matrix <em>Y</em> &isin; {{0, 1}}<sup><em>M</em>&times;<em>I</em></sup> across <em>M</em> model-condition configurations and <em>I</em> items, Proposal 1 models pass probability via regularized MAP estimation with Gaussian priors and Laplace posterior standard errors:</p>
  <div class="eq"><em>P</em>(<em>y<sub>m,i</sub></em> = 1 | &theta;<sub><em>m</em></sub>, <em>a<sub>i</sub></em>, <em>b<sub>i</sub></em>) = &sigma;(<em>a<sub>i</sub></em>(&theta;<sub><em>m</em></sub> &minus; <em>b<sub>i</sub></em>))</div>
  <p class="noindent">where &theta;<sub><em>m</em></sub> is latent ability, <em>b<sub>i</sub></em> is item difficulty, <em>a<sub>i</sub></em> &gt; 0 is item discrimination (<em>a<sub>i</sub></em> = 1 in 1PL), and <code>traj_2pl</code> jointly models auxiliary trajectory features (&alpha;<sub>0</sub>, &alpha;<sub>1</sub>). Proposal 2 (mechanistic judge auditing) and the negative-results literature pipeline from Assignment 2 were scoped out because DB/OS use deterministic checkers rather than LLM judges.</p>

  <div class="fig-box span-all">
    {svg_architecture()}
    <div class="caption"><strong>Fig. 1.</strong> System architecture and client-side intervention pipeline. <code>eval/server_agent.py</code> mediates between local Ollama models and the Dockerized AgentRL FC stack (<code>dbbench-std</code>, <code>os-std</code>, Redis), applying <code>conditions.py</code> (including the <code>c3</code> verify gate) and logging trajectories for statistical analysis.</div>
  </div>

  <h2 class="sec">IV. Methodology</h2>

  <h3 class="subsec">A. Dataset and Environments</h3>
  <p>Evaluations run live on Dockerized <code>dbbench-std</code> (MySQL containers, 300 items) and <code>os-std</code> (Ubuntu bash containers, 144 items) orchestrated by <code>agentrl-controller</code> and Redis. Under consumer 6 GB VRAM constraints, we evaluate on the stratified 40-item benchmark (<code>eval/stratified_40.json</code>: 20 DB across SQL types + 20 OS across directories <code>1</code>&ndash;<code>7</code>) at greedy decoding (<em>T</em> = 0.0, concurrency 1, <code>OLLAMA_CONTEXT_LENGTH=8192</code> verified active via <code>ollama ps</code>).</p>

  <h3 class="subsec">B. Failure-Aware Agent Conditions (<code>c0</code>&ndash;<code>c4</code>)</h3>
  <p>All conditions are client-side wrappers in <code>eval/conditions.py</code>: <code>c0</code> (baseline), <code>c1</code> (repeated-action warning for exact or ROUGE-L <em>F</em><sub>1</sub> &ge; 0.85 matches), <code>c2</code> (state-conditioned failure memory for execution errors, empty non-mutating outputs, truncation, and repeats), <code>c3</code> (targeted verify gate holding the first commit when &ge;2 rounds remain), and <code>c4</code> (<code>c2</code> + <code>c3</code>).</p>

  <h3 class="subsec">C. Statistical Verification Protocol</h3>
  <p><code>eval/analyze_proposal3.py</code> excludes any <em>T</em> &gt; 0 rows via both row metadata and filename tags so <code>c0</code> remains strictly greedy. Paired comparisons use exact two-sided McNemar&rsquo;s tests on matched item IDs alongside behavioral diagnostics (&le;1-command rate, TLE rate, mean environment calls, <em>P</em>(10, 0.8) repetition, token usage, and latency).</p>

  <h2 class="sec">V. Experimentation, Results, and Discussion</h2>

  <h3 class="subsec">A. Iteration 1 Pilot &amp; Iteration 2 Local Setup</h3>
  <p>In Iteration 1 (Colab T4 + Intel Mac, <code>Qwen3-8B</code> 4-bit, first 10 indices/task), <code>c0</code> scored 7/10 on DB and 0/10 on OS (9/10 OS episodes ended after 1 command); enabling thinking mode (1/9 OS) was 28&times; slower, and a generic <code>c3</code> prompt (7/10 DB, 0/10 OS) never flipped an outcome. In Iteration 2, we evaluated three local Ollama models on Windows 11 (Ryzen 7 7840HS, 16 GB RAM, RTX 3050 6 GB): <code>qwen2.5:3b</code> (100% GPU, full <code>c0</code> + <code>c3</code>, 80 episodes), <code>qwen2.5:7b</code> (21%/79% CPU/GPU, 38 <code>c0</code> episodes), and <code>gemma4:12b</code> (8.0 GB on 6 GB VRAM, 8 DB <code>c0</code> episodes).</p>

  <div class="fig-box span-all">
    <table>
      <thead>
        <tr>
          <th>Model</th><th>Cond</th><th>Split</th><th class="num">N</th><th class="num">Success</th><th class="num">Paired (c0&rarr;c3)</th><th class="num">&le;1 Cmd</th><th class="num">TLE</th><th class="num">Env Calls</th><th class="num">P(10,0.8)</th><th class="num">Latency</th><th class="num">McNemar p</th><th class="num">Helped/Harmed</th>
        </tr>
      </thead>
      <tbody>
        {build_table_rows(data)}
      </tbody>
    </table>
    <div class="caption"><strong>TABLE I.</strong> Iteration 2 Proposal 3 summary on <code>stratified_40.json</code> (<em>T</em> = 0.0, <code>sampled_rows_excluded = 0</code>). All comparisons are against the repo&rsquo;s own <code>c0</code> baseline. *<code>gemma4:12b</code> stopped after 8 DB episodes due to session timeouts.</div>
  </div>

  <div class="fig-box span-all">
    {svg_success_rates(data)}
    <div class="caption"><strong>Fig. 2.</strong> Task success rates with 95% Wilson confidence intervals across all evaluated local models and splits (<em>T</em> = 0.0). Sample sizes <em>N</em> are annotated below each bar.</div>
  </div>

  <div class="fig-box avoid-break">
    {svg_gate_impact(data)}
    <div class="caption"><strong>Fig. 3.</strong> Behavioral and compute impact of the <code>c3</code> verify gate vs. <code>c0</code> on <code>qwen2.5:3b</code> (<em>N</em> = 40 paired items).</div>
  </div>

  <div class="fig-box avoid-break">
    {svg_failure_modes(data)}
    <div class="caption"><strong>Fig. 4.</strong> Episode outcome and failure-mode counts from <code>proposal3_summary.json</code>.</div>
  </div>

  <h3 class="subsec">B. Key Findings and Trade-Offs</h3>
  <p>First, baseline <code>c0</code> agents frequently stop after &le;1 command (50.0% on <code>qwen2.5:3b</code>, 36.8% on <code>qwen2.5:7b</code>, 50.0% on <code>gemma4:12b</code> DB). Second, <code>c3</code> reduced &le;1-command answers from 50.0% to 10.0% on <code>qwen2.5:3b</code> (gate fired on 67.5% of episodes), from 10.0% to 0.0% on <code>qwen2.5:7b</code> DB, and from 50.0% to 0.0% on <code>gemma4:12b</code> DB, which is true by construction when the gate holds the first commit. Third, paired accuracy deltas were not statistically significant on <code>qwen2.5:3b</code> (27.5% &rarr; 25.0%, +4/&minus;5, McNemar <em>p</em> = 1.0000), <code>qwen2.5:7b</code> DB (45.0% &rarr; 40.0%, +0/&minus;1, <em>p</em> = 1.0000), or <code>gemma4:12b</code> DB (8 paired: 50.0% &rarr; 50.0%, +0/&minus;0, <em>p</em> = 1.0000; unmatched 17-item <code>c3</code> = 58.8%), while increasing prompt tokens, latency, and <em>P</em>(10, 0.8) repetition.</p>

  <h3 class="subsec">C. Local Hardware Bottlenecks &amp; 2-Round Refinement Log</h3>
  <p><code>qwen2.5:7b</code> completed 20/20 DB (45.0%) and 18/20 OS (11.1%) before stopping when <code>os-std</code> index <code>76</code> hung twice (<code>session 127</code> and <code>129 expired</code>). <code>gemma4:12b</code> completed 8/20 DB (50.0%) before stalling past controller timeouts on indices <code>67</code> (<code>session 138 expired</code>) and <code>93</code> (<code>session 139</code> stalled &gt;10 min even with <code>thinking_off: true</code>). Proposal 1 (<code>analyze_proposal1.py</code>) was skipped because only <code>qwen2.5:3b</code> has paired <code>c0</code>/<code>c3</code> coverage; moreover, Proposal 1&rsquo;s subset selection is fit on the same matrix it is scored against, and 5-fold CV over (model, item) cells leaks same-item/other-condition outcomes.</p>

  <div class="fig-box span-all">
    <table>
      <thead>
        <tr><th style="width:6%;">Round</th><th style="width:32%;">Issue Found</th><th style="width:32%;">LLM Feedback &amp; Fix</th><th style="width:30%;">Effect</th></tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>R1</strong></td>
          <td>Legacy v0.2 client rejected by FC controller; <code>-T</code> <code>KeyError</code>; CRLF <code>*.sh</code>; host-net Compose.</td>
          <td>Adopt <code>server_agent.py</code> (port <code>5020</code>), LF <code>.gitattributes</code>, sanitize tool messages.</td>
          <td><code>dbbench-std</code> (300) &amp; <code>os-std</code> (144) registered; 163 <em>T</em>=0 runs logged.</td>
        </tr>
        <tr>
          <td><strong>R2</strong></td>
          <td><em>T</em>&gt;0 rows mixed into <code>c0</code>; <code>c3</code> overclaimed as &ldquo;eliminating guessing&rdquo;; unverified gaps.</td>
          <td>Add <code>filename_temperature</code> filter &amp; <code>Paired</code> McNemar; reword to &ldquo;gate fired.&rdquo;</td>
          <td>Filter added (no <em>T</em>&gt;0 rows in this run&rsquo;s data); exact <em>p</em>=1.0000 &amp; compute cost reported.</td>
        </tr>
      </tbody>
    </table>
    <div class="caption"><strong>TABLE II.</strong> Summary of the two-round LLM refinement log (see <code>eval/ITERATION2_FINDINGS.md</code> and <code>RUN_NOTES.md</code>).</div>
  </div>

  <h2 class="sec">VI. Conclusion</h2>
  <p>Client-side failure-aware interventions integrate cleanly into AgentBench FC without modifying task grading. On <code>qwen2.5:3b</code> (<em>N</em> = 40), the <code>c3</code> verify gate reliably enforces multi-step verification (&le;1-command rate 50.0% &rarr; 10.0%) at ~2&times; token and latency cost without a statistically significant accuracy gain (<em>p</em> = 1.0000). Evaluating 7B&ndash;12B models on 6 GB VRAM hardware is bottlenecked by controller session timeouts, motivating higher-VRAM multi-model runs to evaluate <code>c1</code>/<code>c2</code>/<code>c4</code> and Proposal 1 IRT calibration.</p>

  <h2 class="sec">References</h2>
  <ol class="ref-list">
    <li>[1] X. Liu <em>et al.</em>, &ldquo;AgentBench: Evaluating LLMs as Agents,&rdquo; in <em>Proc. ICLR</em>, 2024 (arXiv:2308.03688; FC AgentRL release, 2025).</li>
    <li>[2] &ldquo;Efficient SWE Agent Benchmarking via Trajectory-Aware Evaluation (PTA-IRT),&rdquo; <em>arXiv:2609.01603</em>, 2026.</li>
    <li>[3] &ldquo;Beyond Scores: Understanding LLM-as-a-Judge Mechanisms in Summarization Evaluation,&rdquo; <em>arXiv:2609.01604</em>, 2026.</li>
    <li>[4] J. Huang <em>et al.</em>, &ldquo;Large Language Models Cannot Self-Correct Reasoning Yet,&rdquo; in <em>Proc. ICLR</em>, 2024.</li>
    <li>[5] P. Wu <em>et al.</em>, &ldquo;Uncovering Understanding-Generation Synergy in Native Unified Multimodal Models,&rdquo; <em>arXiv:2609.01607</em>, 2026 (filtered out during gap mapping).</li>
    <li>[6] &ldquo;DeepScientist,&rdquo; <em>arXiv:2509.26603</em> (unread Task 4 background reference).</li>
    <li>[7] &ldquo;EvoScientist,&rdquo; <em>arXiv:2603.08127</em> (unread Task 4 background reference).</li>
    <li>[8] &ldquo;AutoSOTA,&rdquo; <em>arXiv:2604.05550</em> (unread Task 4 background reference).</li>
    <li>[9] &ldquo;Arbor,&rdquo; <em>arXiv:2606.11926</em> (unread Task 4 background reference).</li>
  </ol>

</div>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
