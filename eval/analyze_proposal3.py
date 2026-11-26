#!/usr/bin/env python
"""Proposal 3 statistical analyzer: compares c0..c4 across DB (300), OS (144), and Overall (444).

Computes:
- Success rate + 95% Wilson CI
- Paired McNemar's test (exact binomial p-value + continuity-corrected chi2 + discordant counts) vs c0
- <=1-command answer rate, TLE rate, exact repeat rate, AgentBench P(10, 0.8) repetition rate
- Verification flip breakdown (wrong->right helped vs right->wrong harmed)
- Failure taxonomy counts, token overhead, and latency

Rows sampled at temperature > 0 (the Proposal 1 variance runs, e.g. *__t0.7_k3.jsonl) are
EXCLUDED here so that the c0 baseline is a pure temperature-0 baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scipy.stats import binomtest

from trajectory_parser import TrajectoryFeatures, parse_jsonl_file

_TEMP_RE = re.compile(r"__t(\d+(?:\.\d+)?)(?=_|$)")


def filename_temperature(path: Path) -> Optional[float]:
    """Read the temperature from names like dbbench-std__c0__t0.jsonl or ...__c0__t0.7_k3.jsonl."""
    m = _TEMP_RE.search(path.stem.lower())
    return float(m.group(1)) if m else None


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_test(base_outcomes: Dict[str, int], cond_outcomes: Dict[str, int]) -> Dict[str, Any]:
    """Paired McNemar test on matched item_ids."""
    common = sorted(set(base_outcomes) & set(cond_outcomes))
    n11 = n10 = n01 = n00 = 0
    for item_id in common:
        b = base_outcomes[item_id]
        c = cond_outcomes[item_id]
        if b == 1 and c == 1:
            n11 += 1
        elif b == 1 and c == 0:
            n10 += 1  # harmed (right -> wrong)
        elif b == 0 and c == 1:
            n01 += 1  # helped (wrong -> right)
        else:
            n00 += 1

    discordant = n01 + n10
    if discordant > 0:
        p_exact = float(binomtest(min(n01, n10), discordant, 0.5, alternative="two-sided").pvalue)
        chi2_cc = ((abs(n01 - n10) - 1.0) ** 2 / discordant) if abs(n01 - n10) >= 1.0 else 0.0
    else:
        p_exact = 1.0
        chi2_cc = 0.0

    n = len(common)
    base_sr = (n11 + n10) / n if n else 0.0
    cond_sr = (n11 + n01) / n if n else 0.0
    return {
        "n_paired": n,
        "both_pass": n11,
        "both_fail": n00,
        "helped_wrong_to_right": n01,
        "harmed_right_to_wrong": n10,
        "net_gain": n01 - n10,
        "base_success_rate": round(base_sr, 4),
        "cond_success_rate": round(cond_sr, 4),
        "delta_success_rate": round(cond_sr - base_sr, 4),
        "mcnemar_exact_p": round(p_exact, 6),
        "mcnemar_chi2_cc": round(chi2_cc, 4),
    }


def infer_condition_from_filename(path: Path) -> Optional[str]:
    stem = path.stem.lower()
    for part in stem.split("__"):
        if part in ("c0", "c1", "c2", "c3", "c4") or part.startswith("c2_from_"):
            return part
    for cond in ("c0", "c1", "c2", "c3", "c4"):
        if f"-{cond}-" in stem or stem.endswith(f"-{cond}"):
            return cond
    return None


def summarize_group(feats: List[TrajectoryFeatures]) -> Dict[str, Any]:
    valid = [f for f in feats if f.status_valid]
    n = len(valid)
    if n == 0:
        return {"n": 0, "total_rows": len(feats)}

    succ = sum(f.success for f in valid)
    lo, hi = wilson_ci(succ, n)
    le1 = sum(1 for f in valid if f.answered_le_1_cmd)
    tle = sum(1 for f in valid if f.tle)
    rl_rep = sum(1 for f in valid if f.rouge_l_repeat_flag)
    vg_fired = sum(1 for f in valid if f.verify_gate_fired)
    vg_new_cmd = sum(1 for f in valid if f.verify_ran_new_command)
    vg_changed = sum(1 for f in valid if f.verify_changed_answer)
    mode_counts = dict(Counter(f.failure_mode for f in valid))

    return {
        "n": n,
        "total_rows": len(feats),
        "success_count": succ,
        "success_rate": round(succ / n, 4),
        "success_ci95": [round(lo, 4), round(hi, 4)],
        "answered_le_1_cmd_count": le1,
        "answered_le_1_cmd_rate": round(le1 / n, 4),
        "tle_count": tle,
        "tle_rate": round(tle / n, 4),
        "mean_env_calls": round(sum(f.num_env_calls for f in valid) / n, 3),
        "mean_exact_repeat_rate": round(sum(f.exact_repeat_rate for f in valid) / n, 4),
        "mean_rouge_l_p10": round(sum(f.max_rouge_l_p10 for f in valid) / n, 4),
        "rouge_l_ge_08_rate": round(rl_rep / n, 4),
        "mean_error_count": round(sum(f.error_count for f in valid) / n, 3),
        "verify_gate_fired_rate": round(vg_fired / n, 4),
        "verify_ran_new_cmd_rate": round(vg_new_cmd / n, 4),
        "verify_changed_answer_rate": round(vg_changed / n, 4),
        "mean_prompt_tokens": round(sum(f.prompt_tokens for f in valid) / n, 1),
        "mean_completion_tokens": round(sum(f.completion_tokens for f in valid) / n, 1),
        "mean_latency_s": round(sum(f.latency_s for f in valid) / n, 3),
        "failure_modes": mode_counts,
    }


def _dedup_run1(feats: List[TrajectoryFeatures]) -> Dict[str, int]:
    """Map item_id -> binary success (using run_number == 1 or first valid run)."""
    out: Dict[str, int] = {}
    for f in sorted(feats, key=lambda x: x.run_number):
        if f.status_valid and f.item_id not in out:
            out[f.item_id] = f.success
    return out


def analyze_files(paths: Iterable[Path]) -> Dict[str, Any]:
    all_feats: List[TrajectoryFeatures] = []
    sampled_excluded = 0
    for p in sorted(paths):
        cond_hint = infer_condition_from_filename(p)
        file_temp = filename_temperature(p) or 0.0
        feats = parse_jsonl_file(p)
        for f in feats:
            # Exclude temperature > 0 rows (Proposal 1 variance runs) from the c0..c4 comparison.
            row_temp = f.temperature if f.temperature > 0 else file_temp
            if row_temp > 0:
                sampled_excluded += 1
                continue
            # Prefer explicit non-default condition or filename-inferred label (e.g. c2_from_db)
            if cond_hint and (f.condition == "c0" or cond_hint.startswith("c2_from_")):
                f.condition = cond_hint
            all_feats.append(f)

    if sampled_excluded:
        print(f"[analyze_proposal3] excluded {sampled_excluded} rows sampled at temperature > 0", file=sys.stderr)

    by_model: Dict[str, Dict[str, List[TrajectoryFeatures]]] = defaultdict(lambda: defaultdict(list))
    for f in all_feats:
        by_model[f.model][f.condition].append(f)

    report: Dict[str, Any] = {"models": {}, "sampled_rows_excluded": sampled_excluded}
    for model, cond_map in sorted(by_model.items()):
        model_report: Dict[str, Any] = {"conditions": {}, "mcnemar_vs_c0": {}}
        base_feats = cond_map.get("c0", [])

        for cond, feats in sorted(cond_map.items()):
            db_feats = [f for f in feats if f.task_type == "db"]
            os_feats = [f for f in feats if f.task_type == "os"]
            model_report["conditions"][cond] = {
                "overall": summarize_group(feats),
                "db": summarize_group(db_feats),
                "os": summarize_group(os_feats),
            }

            if cond != "c0" and base_feats:
                mcn = {}
                for split_name, base_sub, cond_sub in [
                    ("overall", base_feats, feats),
                    ("db", [f for f in base_feats if f.task_type == "db"], db_feats),
                    ("os", [f for f in base_feats if f.task_type == "os"], os_feats),
                ]:
                    b_map = _dedup_run1(base_sub)
                    c_map = _dedup_run1(cond_sub)
                    if set(b_map) & set(c_map):
                        mcn[split_name] = mcnemar_test(b_map, c_map)
                model_report["mcnemar_vs_c0"][cond] = mcn

        report["models"][model] = model_report
    return report


def format_markdown_report(report: Dict[str, Any]) -> str:
    lines = ["# Proposal 3 Evaluation Report (Failure-Aware Agent Conditions)", ""]
    lines.append(
        "Temperature-0 runs only. `Paired` = matched items vs c0 (N: c0% -> condition%); "
        "use it, not the unmatched Success Rate, to compare a condition with c0."
    )
    lines.append("")
    for model, mdata in report["models"].items():
        lines.append(f"## Model: `{model}`")
        lines.append("")
        lines.append("| Condition | Split | N | Success Rate | Paired | <=1 Cmd Rate | TLE Rate | Mean Env Calls | Repeat Rate | P(10,0.8) | McNemar p vs c0 | Helped / Harmed |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cond, cdata in mdata["conditions"].items():
            mcn_all = mdata["mcnemar_vs_c0"].get(cond, {})
            for split in ("overall", "db", "os"):
                s = cdata[split]
                if s.get("n", 0) == 0:
                    continue
                mcn = mcn_all.get(split)
                p_str = f"{mcn['mcnemar_exact_p']:.4f}" if mcn else "—"
                hh_str = f"+{mcn['helped_wrong_to_right']} / -{mcn['harmed_right_to_wrong']}" if mcn else "—"
                paired_str = (
                    f"{mcn['n_paired']}: {mcn['base_success_rate']*100:.1f}% -> {mcn['cond_success_rate']*100:.1f}%"
                    if mcn else "—"
                )
                lines.append(
                    f"| `{cond}` | {split.upper()} | {s['n']} | {s['success_rate']*100:.1f}% | {paired_str} | "
                    f"{s['answered_le_1_cmd_rate']*100:.1f}% | {s['tle_rate']*100:.1f}% | "
                    f"{s['mean_env_calls']:.2f} | {s['mean_exact_repeat_rate']*100:.1f}% | "
                    f"{s['rouge_l_ge_08_rate']*100:.1f}% | {p_str} | {hh_str} |"
                )
        lines.append("")
    return "\n".join(lines)


def collect_jsonl_paths(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.jsonl")))
        elif p.exists():
            paths.append(p)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Analyze Proposal 3 experimental runs.")
    parser.add_argument("inputs", nargs="+", help="JSONL files or directories containing them")
    parser.add_argument("--out-json", default=None, help="Optional path to write JSON summary")
    parser.add_argument("--out-md", default=None, help="Optional path to write Markdown table")
    args = parser.parse_args()

    paths = collect_jsonl_paths(args.inputs)
    if not paths:
        raise SystemExit("No .jsonl files found in inputs.")

    report = analyze_files(paths)
    md = format_markdown_report(report)
    print(md)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()