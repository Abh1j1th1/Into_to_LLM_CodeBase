#!/usr/bin/env python
"""Proposal 1 evaluation: Trajectory-Aware Bayesian IRT vs baselines on AgentBench (444 items).

Evaluates:
1. Ranking stability & score error at reduced budgets (10%, 20%, 30% calibration subsets)
   via Kendall's tau, Spearman's rho, MAE/RMSE vs full-set score, and bootstrap rank intervals.
2. Subset selection strategies:
   - info_traj: Fisher item info + trajectory-process variance signal
   - info_2pl: Outcome-only 2PL Fisher item info
   - stratified: Environment + difficulty-stratified subset
   - random: Random subset averaged across seeds
3. Held-out instance-level prediction (5-fold CV over items: accuracy, log-loss, Brier, ROC-AUC).
4. Sensitivity analysis (drop DB / drop OS, graded partial credit vs binary vs trajectory-aware).
5. Run-to-run variance quantification when repeated runs (e.g. T=0.7) are present.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit
from scipy.stats import kendalltau, rankdata, spearmanr

from irt import compute_baseline_scores, fit_bayesian_irt, fit_graded_irt, item_fisher_information
from trajectory_parser import TrajectoryFeatures, parse_jsonl_file


def build_matrices(
    feats: List[TrajectoryFeatures],
) -> Tuple[List[str], List[str], List[str], np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[str, str], List[int]]]:
    """Assemble (models x items) matrices for outcomes y, process quality z, partial credit g."""
    # Group by (model_cfg, item_id); if multiple conditions exist for one model, treat model::condition
    grouped: Dict[Tuple[str, str], List[TrajectoryFeatures]] = defaultdict(list)
    multi_run_outcomes: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    item_env_map: Dict[str, str] = {}
    for f in feats:
        if not f.status_valid:
            continue
        is_sampled = float((getattr(f, "temperature", 0.0) or 0.0)) > 0.0
        base_label = f.model if f.condition == "c0" else f"{f.model} [{f.condition}]"
        if is_sampled:
            multi_run_outcomes[(base_label, f.item_id)].append(f.success)
            continue
        grouped[(base_label, f.item_id)].append(f)
        item_env_map[f.item_id] = f.task_type

    models = sorted({m for m, _ in grouped})
    items = sorted(item_env_map.keys())
    item_envs = [item_env_map[it] for it in items]

    M, I = len(models), len(items)
    y_mat = np.full((M, I), np.nan, dtype=float)
    z_mat = np.full((M, I), np.nan, dtype=float)
    g_mat = np.full((M, I), np.nan, dtype=float)

    m_pos = {m: idx for idx, m in enumerate(models)}
    i_pos = {it: idx for idx, it in enumerate(items)}

    for (m, it), run_list in grouped.items():
        run_list_sorted = sorted(run_list, key=lambda r: r.run_number)
        mi, ii = m_pos[m], i_pos[it]
        y_mat[mi, ii] = float(np.mean([r.success for r in run_list_sorted]))
        z_mat[mi, ii] = float(np.mean([r.process_quality for r in run_list_sorted]))
        g_mat[mi, ii] = float(np.mean([r.partial_credit_level for r in run_list_sorted]))
        if len(run_list_sorted) > 1 and (m, it) not in multi_run_outcomes:
            multi_run_outcomes[(m, it)] = [r.success for r in run_list_sorted]

    return models, items, item_envs, y_mat, z_mat, g_mat, multi_run_outcomes


def rank_metrics(est_scores: np.ndarray, ref_scores: np.ndarray) -> Dict[str, float]:
    tau, _ = kendalltau(est_scores, ref_scores)
    rho, _ = spearmanr(est_scores, ref_scores)
    mae = float(np.mean(np.abs(est_scores - ref_scores)))
    rmse = float(np.sqrt(np.mean((est_scores - ref_scores) ** 2)))
    return {
        "kendall_tau": 0.0 if np.isnan(tau) else round(float(tau), 4),
        "spearman_rho": 0.0 if np.isnan(rho) else round(float(rho), 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


def select_calibration_subset(
    strategy: str,
    k: int,
    y_mat: np.ndarray,
    z_mat: np.ndarray,
    item_envs: Sequence[str],
    models: Sequence[str],
    items: Sequence[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Select `k` calibration item indices using the requested strategy."""
    I = len(items)
    k = max(2, min(k, I))
    if strategy == "random":
        return np.sort(rng.choice(I, size=k, replace=False))

    if strategy == "stratified":
        # Stratify by environment + empirical item pass-rate quantile
        obs = ~np.isnan(y_mat)
        p_item = np.array([y_mat[obs[:, i], i].mean() if obs[:, i].any() else 0.5 for i in range(I)])
        buckets: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        for i in range(I):
            q = 0 if p_item[i] < 0.33 else (1 if p_item[i] < 0.67 else 2)
            buckets[(item_envs[i], q)].append(i)
        chosen: List[int] = []
        # Proportional allocation
        for key, idxs in sorted(buckets.items()):
            take = max(1, int(round(k * len(idxs) / I)))
            take = min(take, len(idxs))
            chosen.extend(rng.choice(idxs, size=take, replace=False).tolist())
        chosen = list(dict.fromkeys(chosen))
        if len(chosen) > k:
            chosen = rng.choice(chosen, size=k, replace=False).tolist()
        elif len(chosen) < k:
            rem = [i for i in range(I) if i not in set(chosen)]
            chosen.extend(rng.choice(rem, size=k - len(chosen), replace=False).tolist())
        return np.sort(np.array(chosen, dtype=int))

    # Information-based selection (using preliminary fit or pilot pool)
    use_traj = strategy == "info_traj"
    fit = fit_bayesian_irt(y_mat, models, items, z_matrix=z_mat, two_pl=True, use_trajectory=use_traj)
    info = item_fisher_information(fit)
    if use_traj:
        # Reward items with both high Fisher info and discriminative trajectory variance across models
        obs_z = ~np.isnan(z_mat)
        z_std = np.array([
            np.std(z_mat[obs_z[:, i], i]) if obs_z[:, i].sum() >= 2 else 0.0
            for i in range(I)
        ])
        score = info * (1.0 + 1.5 * z_std)
    else:
        score = info

    # Preserve environment balance proportional to DB/OS counts
    envs = sorted(set(item_envs))
    chosen_idx: List[int] = []
    for env in envs:
        env_i = [i for i, e in enumerate(item_envs) if e == env]
        take = max(1, int(round(k * len(env_i) / I)))
        env_sorted = sorted(env_i, key=lambda idx: score[idx], reverse=True)
        chosen_idx.extend(env_sorted[:take])
    chosen_idx = list(dict.fromkeys(chosen_idx))
    if len(chosen_idx) > k:
        chosen_idx = sorted(chosen_idx, key=lambda idx: score[idx], reverse=True)[:k]
    elif len(chosen_idx) < k:
        rem = sorted([i for i in range(I) if i not in set(chosen_idx)], key=lambda idx: score[idx], reverse=True)
        chosen_idx.extend(rem[: k - len(chosen_idx)])
    return np.sort(np.array(chosen_idx, dtype=int))


def bootstrap_rank_intervals(
    y_mat: np.ndarray,
    z_mat: np.ndarray,
    item_envs: Sequence[str],
    models: Sequence[str],
    items: Sequence[str],
    n_boot: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute 95% item-bootstrap rank intervals for reciprocal-weighted vs trajectory-aware IRT."""
    rng = np.random.default_rng(seed)
    M, I = y_mat.shape
    ranks_rw = np.zeros((n_boot, M), dtype=float)
    ranks_o2pl = np.zeros((n_boot, M), dtype=float)
    ranks_tirt = np.zeros((n_boot, M), dtype=float)

    for b_idx in range(n_boot):
        boot_cols = rng.integers(0, I, size=I)
        yb = y_mat[:, boot_cols]
        zb = z_mat[:, boot_cols]
        envs_b = [item_envs[c] for c in boot_cols]
        items_b = [f"boot_{j}" for j in range(I)]

        base = compute_baseline_scores(yb, envs_b)["reciprocal_weighted"]
        fit_o = fit_bayesian_irt(yb, models, items_b, two_pl=True, use_trajectory=False)
        fit_t = fit_bayesian_irt(yb, models, items_b, z_matrix=zb, two_pl=True, use_trajectory=True)

        # Rank 1 = best model (ties averaged)
        ranks_rw[b_idx] = rankdata(-base, method="average")
        ranks_o2pl[b_idx] = rankdata(-fit_o.theta, method="average")
        ranks_tirt[b_idx] = rankdata(-fit_t.theta, method="average")

    summary = {}
    for name, arr in [("reciprocal_weighted", ranks_rw), ("outcome_2pl", ranks_o2pl), ("traj_2pl", ranks_tirt)]:
        per_model = {}
        widths = []
        for m_idx, m_name in enumerate(models):
            lo, hi = np.percentile(arr[:, m_idx], [2.5, 97.5])
            widths.append(hi - lo)
            per_model[m_name] = {
                "mean_rank": round(float(arr[:, m_idx].mean()), 2),
                "ci95": [round(float(lo), 2), round(float(hi), 2)],
            }
        summary[name] = {
            "mean_rank_ci_width": round(float(np.mean(widths)), 3),
            "per_model": per_model,
        }
    return summary


def heldout_cv_prediction(
    y_mat: np.ndarray,
    z_mat: np.ndarray,
    models: Sequence[str],
    items: Sequence[str],
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate held-out instance-level prediction via k-fold CV over observed (model, item) cells."""
    rng = np.random.default_rng(seed)
    obs = ~np.isnan(y_mat)
    m_idx, i_idx = np.where(obs)
    N = len(m_idx)
    perm = rng.permutation(N)
    folds = np.array_split(perm, n_folds)

    preds = {"outcome_1pl": [], "outcome_2pl": [], "traj_2pl": []}
    trues: List[float] = []

    for test_pos in folds:
        train_mask = obs.copy()
        train_mask[m_idx[test_pos], i_idx[test_pos]] = False

        y_bin = (y_mat >= 0.5).astype(float)
        trues.extend(y_bin[m_idx[test_pos], i_idx[test_pos]].tolist())

        for key, two_pl, use_traj in [
            ("outcome_1pl", False, False),
            ("outcome_2pl", True, False),
            ("traj_2pl", True, True),
        ]:
            fit = fit_bayesian_irt(
                y_bin,
                models,
                items,
                obs_mask=train_mask,
                z_matrix=z_mat if use_traj else None,
                two_pl=two_pl,
                use_trajectory=use_traj,
            )
            eta = fit.a[i_idx[test_pos]] * (fit.theta[m_idx[test_pos]] - fit.b[i_idx[test_pos]])
            p_hat = np.clip(expit(eta), 1e-6, 1.0 - 1e-6)
            preds[key].extend(p_hat.tolist())

    y_true = np.array(trues)
    out = {}
    for key, p_list in preds.items():
        p = np.array(p_list)
        acc = float(np.mean((p >= 0.5) == (y_true >= 0.5)))
        brier = float(np.mean((p - y_true) ** 2))
        logloss = float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
        out[key] = {
            "accuracy": round(acc, 4),
            "brier": round(brier, 4),
            "log_loss": round(logloss, 4),
        }
    return out


def analyze_proposal1(
    feats: List[TrajectoryFeatures],
    budgets: Sequence[float] = (0.10, 0.20, 0.30),
    n_seeds: int = 5,
    n_boot: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    models, items, item_envs, y_mat, z_mat, g_mat, multi_run = build_matrices(feats)
    M, I = y_mat.shape
    if M < 2 or I < 5:
        raise ValueError(f"Need >=2 model configurations and >=5 items for IRT analysis; got M={M}, I={I}")

    y_bin = np.where(np.isnan(y_mat), np.nan, (y_mat >= 0.5).astype(float))
    baselines_full = compute_baseline_scores(y_bin, item_envs)
    fit_o1 = fit_bayesian_irt(y_bin, models, items, two_pl=False, use_trajectory=False)
    fit_o2 = fit_bayesian_irt(y_bin, models, items, two_pl=True, use_trajectory=False)
    fit_t2 = fit_bayesian_irt(y_bin, models, items, z_matrix=z_mat, two_pl=True, use_trajectory=True)
    fit_gr = fit_graded_irt(g_mat, models, items)

    ref_full = baselines_full["macro_mean"]

    # Budgeted subset recovery
    rng = np.random.default_rng(seed)
    budget_results: Dict[str, Any] = {}
    for frac in budgets:
        k = max(4, int(round(frac * I)))
        frac_key = f"{int(round(frac * 100))}pct_k{k}"
        strat_agg: Dict[str, List[Dict[str, float]]] = defaultdict(list)

        for strat in ("random", "stratified", "info_2pl", "info_traj"):
            reps = n_seeds if strat in ("random", "stratified") else 1
            for _ in range(reps):
                sub = select_calibration_subset(strat, k, y_bin, z_mat, item_envs, models, items, rng)
                sub_envs = [item_envs[i] for i in sub]
                sub_items = [items[i] for i in sub]
                y_sub = y_bin[:, sub]
                z_sub = z_mat[:, sub]

                rw_sub = compute_baseline_scores(y_sub, sub_envs)["reciprocal_weighted"]
                mean_sub = compute_baseline_scores(y_sub, sub_envs)["macro_mean"]
                fit_sub_o2 = fit_bayesian_irt(y_sub, models, sub_items, two_pl=True, use_trajectory=False)
                fit_sub_t2 = fit_bayesian_irt(y_sub, models, sub_items, z_matrix=z_sub, two_pl=True, use_trajectory=True)

                strat_agg[f"{strat}__reciprocal_weighted"].append(rank_metrics(rw_sub, baselines_full["reciprocal_weighted"]))
                strat_agg[f"{strat}__macro_mean"].append(rank_metrics(mean_sub, ref_full))
                strat_agg[f"{strat}__outcome_2pl"].append(rank_metrics(fit_sub_o2.predicted_full_scores, ref_full))
                strat_agg[f"{strat}__traj_2pl"].append(rank_metrics(fit_sub_t2.predicted_full_scores, ref_full))

        budget_results[frac_key] = {
            method: {
                metric: round(float(np.mean([r[metric] for r in runs])), 4)
                for metric in runs[0]
            }
            for method, runs in sorted(strat_agg.items())
        }

    # Sensitivity: drop one environment + graded vs trajectory IRT
    sensitivity = {}
    for env in sorted(set(item_envs)):
        keep = [i for i, e in enumerate(item_envs) if e != env]
        if len(keep) >= 5:
            sub_envs = [item_envs[i] for i in keep]
            sub_items = [items[i] for i in keep]
            rw_drop = compute_baseline_scores(y_bin[:, keep], sub_envs)["reciprocal_weighted"]
            t2_drop = fit_bayesian_irt(y_bin[:, keep], models, sub_items, z_matrix=z_mat[:, keep], two_pl=True, use_trajectory=True)
            sensitivity[f"drop_{env}__reciprocal"] = rank_metrics(rw_drop, baselines_full["reciprocal_weighted"])
            sensitivity[f"drop_{env}__traj_2pl"] = rank_metrics(t2_drop.theta, fit_t2.theta)
    sensitivity["graded_vs_traj_2pl"] = rank_metrics(fit_gr.theta, fit_t2.theta)

    # Variance quantification for repeated runs (e.g. T=0.7)
    variance_stats: Dict[str, Any] = {}
    if multi_run:
        per_model_vars = defaultdict(list)
        for (m, _), outs in multi_run.items():
            per_model_vars[m].append(float(np.var(outs, ddof=1)) if len(outs) > 1 else 0.0)
        variance_stats = {
            m: {
                "n_repeated_items": len(vs),
                "mean_item_variance": round(float(np.mean(vs)), 4),
                "mean_item_std": round(float(np.mean(np.sqrt(vs))), 4),
            }
            for m, vs in sorted(per_model_vars.items())
        }

    return {
        "n_models": M,
        "n_items": I,
        "env_counts": {e: sum(1 for x in item_envs if x == e) for e in sorted(set(item_envs))},
        "models": [
            {
                "model": m,
                "micro_mean": round(float(baselines_full["micro_mean"][idx]), 4),
                "macro_mean": round(float(baselines_full["macro_mean"][idx]), 4),
                "reciprocal_weighted": round(float(baselines_full["reciprocal_weighted"][idx]), 4),
                "outcome_1pl_theta": round(float(fit_o1.theta[idx]), 4),
                "outcome_2pl_theta": round(float(fit_o2.theta[idx]), 4),
                "traj_2pl_theta": round(float(fit_t2.theta[idx]), 4),
                "traj_2pl_theta_se": round(float(fit_t2.theta_se[idx]), 4),
                "traj_2pl_pred_score": round(float(fit_t2.predicted_full_scores[idx]), 4),
                "graded_theta": round(float(fit_gr.theta[idx]), 4),
            }
            for idx, m in enumerate(models)
        ],
        "heldout_cv": heldout_cv_prediction(y_bin, z_mat, models, items, seed=seed),
        "bootstrap_rank_intervals": bootstrap_rank_intervals(y_bin, z_mat, item_envs, models, items, n_boot=n_boot, seed=seed),
        "budget_recovery": budget_results,
        "sensitivity": sensitivity,
        "run_to_run_variance": variance_stats,
    }


def format_proposal1_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Proposal 1 Evaluation Report (Trajectory-Aware Bayesian IRT)",
        f"Models: {report['n_models']} | Items: {report['n_items']} ({report['env_counts']})",
        "",
        "## Full-Set Model Estimates",
        "| Model | Micro Mean | Macro Mean | Reciprocal Overall | Outcome 2PL theta | Traj-IRT theta ± SE | Traj-IRT Pred Score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(report["models"], key=lambda r: r["traj_2pl_theta"], reverse=True):
        lines.append(
            f"| `{row['model']}` | {row['micro_mean']:.3f} | {row['macro_mean']:.3f} | "
            f"{row['reciprocal_weighted']:.3f} | {row['outcome_2pl_theta']:+.3f} | "
            f"{row['traj_2pl_theta']:+.3f} ± {row['traj_2pl_theta_se']:.3f} | {row['traj_2pl_pred_score']:.3f} |"
        )
    lines.extend(["", "## Held-Out Instance Prediction (5-fold CV)", "| Model | Accuracy | Brier | Log-Loss |", "|---|---:|---:|---:|"])
    for k, v in report["heldout_cv"].items():
        lines.append(f"| `{k}` | {v['accuracy']:.4f} | {v['brier']:.4f} | {v['log_loss']:.4f} |")

    lines.extend(["", "## Calibration Subset Recovery", "| Budget | Selector + Scorer | Kendall tau | Spearman rho | MAE | RMSE |", "|---|---|---:|---:|---:|---:|"])
    for bkey, methods in report["budget_recovery"].items():
        for mkey, vals in methods.items():
            lines.append(f"| {bkey} | `{mkey}` | {vals['kendall_tau']:.3f} | {vals['spearman_rho']:.3f} | {vals['mae']:.4f} | {vals['rmse']:.4f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze Proposal 1 (Trajectory-Aware Bayesian IRT).")
    parser.add_argument("inputs", nargs="+", help="JSONL files or directories")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    paths: List[Path] = []
    for item in args.inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.jsonl")))
        elif p.exists():
            paths.append(p)
    if not paths:
        raise SystemExit("No .jsonl files found.")

    feats: List[TrajectoryFeatures] = []
    for p in paths:
        feats.extend(parse_jsonl_file(p))

    report = analyze_proposal1(feats, n_seeds=args.n_seeds, n_boot=args.n_boot)
    md = format_proposal1_markdown(report)
    print(md)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
