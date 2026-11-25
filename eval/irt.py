"""Proposal 1: Outcome-Only & Trajectory-Aware Bayesian IRT Models for AgentBench.

Designed for multi-environment (DB + OS = 444 items) evaluation with a small model
population (~5 model configurations), using Gaussian priors and Laplace posterior
uncertainty (Section 6 of Task4_Iteration1_Group12.pdf).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class IRTFitResult:
    variant: str
    models: List[str]
    items: List[str]
    theta: np.ndarray          # shape (M,)
    theta_se: np.ndarray       # shape (M,) Laplace SE
    b: np.ndarray              # shape (I,) item difficulty
    a: np.ndarray              # shape (I,) item discrimination
    alpha0: float = 0.0        # trajectory auxiliary intercept
    alpha1: float = 0.0        # trajectory auxiliary slope
    predicted_full_scores: Optional[np.ndarray] = None  # shape (M,)


def compute_baseline_scores(
    y_matrix: np.ndarray,
    item_envs: Sequence[str],
    obs_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute plain micro-mean, macro-mean, and AgentBench reciprocal-weighted Overall Score.

    Args:
        y_matrix: (M, I) binary matrix (NaN allowed if obs_mask is None).
        item_envs: length-I sequence of environment labels ('db' or 'os').
        obs_mask: optional boolean mask (M, I) of observed entries.
    """
    M, I = y_matrix.shape
    if obs_mask is None:
        obs_mask = ~np.isnan(y_matrix)
    y_clean = np.where(obs_mask, y_matrix, 0.0)

    micro_mean = np.array([
        y_clean[m, obs_mask[m]].mean() if obs_mask[m].any() else 0.0
        for m in range(M)
    ])

    env_names = sorted(set(item_envs))
    env_scores = {}
    env_has_obs = {}
    env_pool_means = {}
    for env in env_names:
        idx = np.array([e == env for e in item_envs])
        has_obs = np.array([(idx & obs_mask[m]).any() for m in range(M)])
        scores_m = np.array([
            y_clean[m, idx & obs_mask[m]].mean() if has_obs[m] else np.nan
            for m in range(M)
        ])
        env_scores[env] = scores_m
        env_has_obs[env] = has_obs
        valid_pool = scores_m[has_obs]
        pool_mean = float(valid_pool.mean()) if len(valid_pool) > 0 else 1.0
        env_pool_means[env] = max(pool_mean, 1e-3)

    macro_mean = np.array([
        np.nanmean([env_scores[e][m] for e in env_names]) if any(env_has_obs[e][m] for e in env_names) else 0.0
        for m in range(M)
    ])

    raw_weights = {e: 1.0 / env_pool_means[e] for e in env_names}
    reciprocal_weighted = np.zeros(M, dtype=float)
    for m in range(M):
        obs_envs = [e for e in env_names if env_has_obs[e][m]]
        if not obs_envs:
            continue
        w_sum_m = sum(raw_weights[e] for e in obs_envs)
        reciprocal_weighted[m] = sum((raw_weights[e] / w_sum_m) * env_scores[e][m] for e in obs_envs)

    return {
        "micro_mean": micro_mean,
        "macro_mean": macro_mean,
        "reciprocal_weighted": reciprocal_weighted,
    }


def fit_bayesian_irt(
    y_matrix: np.ndarray,
    models: Sequence[str],
    items: Sequence[str],
    obs_mask: Optional[np.ndarray] = None,
    z_matrix: Optional[np.ndarray] = None,
    two_pl: bool = True,
    use_trajectory: bool = False,
    prior_theta_sd: float = 1.0,
    prior_b_sd: float = 2.0,
    prior_log_a_sd: float = 0.5,
    traj_weight: float = 0.25,
) -> IRTFitResult:
    """Fit Bayesian 1PL, 2PL, or Trajectory-Aware 2PL IRT via regularized MAP + Laplace SE.

    When `use_trajectory=True` and `z_matrix` is provided (continuous process quality in (0, 1)),
    jointly models binary outcomes y_{mi} and logit-transformed trajectory process signals
    logit(z_{mi}) ~ N(alpha0 + alpha1 * eta_{mi}, sigma_z^2), where eta_{mi} = a_i * (theta_m - b_i).
    """
    M, I = y_matrix.shape
    if obs_mask is None:
        obs_mask = ~np.isnan(y_matrix)

    m_idx, i_idx = np.where(obs_mask)
    y_obs = y_matrix[m_idx, i_idx].astype(float)

    use_traj = bool(use_trajectory and z_matrix is not None)
    if use_traj:
        z_raw = np.clip(z_matrix[m_idx, i_idx].astype(float), 1e-3, 1.0 - 1e-3)
        logit_z = np.log(z_raw / (1.0 - z_raw))
        z_valid = ~np.isnan(logit_z)
        m_z = m_idx[z_valid]
        i_z = i_idx[z_valid]
        lz_obs = logit_z[z_valid]
    else:
        m_z = np.array([], dtype=int)
        i_z = np.array([], dtype=int)
        lz_obs = np.array([], dtype=float)

    # Parameter layout:
    # [theta (M), b (I), log_a (I if two_pl else 0), alpha0, log_alpha1 (if use_traj)]
    n_a = I if two_pl else 0
    n_traj_params = 2 if use_traj else 0
    dim = M + I + n_a + n_traj_params

    # Empirical initialization
    x0 = np.zeros(dim, dtype=float)
    for m in range(M):
        mask_m = obs_mask[m]
        p_m = np.clip(y_matrix[m, mask_m].mean() if mask_m.any() else 0.5, 0.05, 0.95)
        x0[m] = math.log(p_m / (1.0 - p_m))
    x0[:M] -= x0[:M].mean()
    for i in range(I):
        mask_i = obs_mask[:, i]
        p_i = np.clip(y_matrix[mask_i, i].mean() if mask_i.any() else 0.5, 0.05, 0.95)
        x0[M + i] = -math.log(p_i / (1.0 - p_i))
    if use_traj and len(lz_obs) > 0:
        x0[M + I + n_a] = float(np.mean(lz_obs))
        x0[M + I + n_a + 1] = 0.0  # log(alpha1=1.0)

    sigma_z2 = 2.0

    def unpack(x: np.ndarray):
        theta = x[:M]
        b = x[M : M + I]
        log_a = x[M + I : M + I + n_a] if two_pl else np.zeros(I, dtype=float)
        a = np.exp(np.clip(log_a, -2.5, 2.5))
        alpha0 = x[M + I + n_a] if use_traj else 0.0
        alpha1 = float(x[M + I + n_a + 1]) if use_traj else 0.0
        return theta, b, log_a, a, alpha0, alpha1, alpha1

    def nll_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
        theta, b, log_a, a, alpha0, log_alpha1, alpha1 = unpack(x)
        grad = np.zeros(dim, dtype=float)

        # Outcome Bernoulli log-likelihood
        eta = a[i_idx] * (theta[m_idx] - b[i_idx])
        p = expit(eta)
        eps = 1e-12
        nll = -np.sum(y_obs * np.log(p + eps) + (1.0 - y_obs) * np.log(1.0 - p + eps))

        diff = p - y_obs  # d(-logL)/d(eta)
        np.add.at(grad[:M], m_idx, diff * a[i_idx])
        np.add.at(grad[M : M + I], i_idx, -diff * a[i_idx])
        if two_pl:
            np.add.at(grad[M + I : M + I + I], i_idx, diff * eta)

        # Trajectory privileged signal likelihood
        if use_traj and len(lz_obs) > 0:
            eta_z = a[i_z] * (theta[m_z] - b[i_z])
            mu_z = alpha0 + alpha1 * eta_z
            res = mu_z - lz_obs
            nll += 0.5 * traj_weight * np.sum(res * res) / sigma_z2
            d_eta_z = traj_weight * alpha1 * res / sigma_z2
            np.add.at(grad[:M], m_z, d_eta_z * a[i_z])
            np.add.at(grad[M : M + I], i_z, -d_eta_z * a[i_z])
            if two_pl:
                np.add.at(grad[M + I : M + I + I], i_z, d_eta_z * eta_z)
            grad[M + I + n_a] += traj_weight * np.sum(res) / sigma_z2
            grad[M + I + n_a + 1] += traj_weight * np.sum(res * eta_z) / sigma_z2

        # Gaussian priors (plus weak zero-sum anchor on theta to fix scale/shift)
        nll += 0.5 * np.sum(theta * theta) / (prior_theta_sd ** 2)
        grad[:M] += theta / (prior_theta_sd ** 2)

        theta_mean = np.mean(theta)
        nll += 5.0 * (theta_mean ** 2)
        grad[:M] += 10.0 * theta_mean / M

        nll += 0.5 * np.sum(b * b) / (prior_b_sd ** 2)
        grad[M : M + I] += b / (prior_b_sd ** 2)

        if two_pl:
            nll += 0.5 * np.sum(log_a * log_a) / (prior_log_a_sd ** 2)
            grad[M + I : M + I + I] += log_a / (prior_log_a_sd ** 2)

        if use_traj:
            nll += 0.5 * (alpha0 ** 2) / 4.0 + 0.5 * (log_alpha1 ** 2) / 1.0
            grad[M + I + n_a] += alpha0 / 4.0
            grad[M + I + n_a + 1] += log_alpha1 / 1.0

        return float(nll), grad

    res = minimize(nll_and_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": 400})
    theta, b, _, a, alpha0, _, alpha1 = unpack(res.x)

    # Diagonal Laplace standard errors for theta_m
    theta_se = np.zeros(M, dtype=float)
    eta_all = a[i_idx] * (theta[m_idx] - b[i_idx])
    p_all = expit(eta_all)
    fisher_y = (a[i_idx] ** 2) * p_all * (1.0 - p_all)
    prec_theta = np.full(M, 1.0 / (prior_theta_sd ** 2) + 10.0 / (M ** 2), dtype=float)
    np.add.at(prec_theta, m_idx, fisher_y)
    if use_traj and len(lz_obs) > 0:
        np.add.at(prec_theta, m_z, traj_weight * (alpha1 ** 2) * (a[i_z] ** 2) / sigma_z2)
    theta_se = 1.0 / np.sqrt(np.maximum(prec_theta, 1e-6))

    # Predict expected score across full item bank I
    eta_full = a[None, :] * (theta[:, None] - b[None, :])
    pred_full = expit(eta_full).mean(axis=1)

    variant = "traj_2pl" if (use_traj and two_pl) else (
        "traj_1pl" if use_traj else ("outcome_2pl" if two_pl else "outcome_1pl")
    )
    return IRTFitResult(
        variant=variant,
        models=list(models),
        items=list(items),
        theta=theta,
        theta_se=theta_se,
        b=b,
        a=a,
        alpha0=float(alpha0),
        alpha1=float(alpha1),
        predicted_full_scores=pred_full,
    )


def fit_graded_irt(
    g_matrix: np.ndarray,
    models: Sequence[str],
    items: Sequence[str],
    obs_mask: Optional[np.ndarray] = None,
    n_levels: int = 4,
    prior_theta_sd: float = 1.0,
    prior_b_sd: float = 2.0,
) -> IRTFitResult:
    """Samejima Graded Response Model on ordinal partial-credit levels g in {0, 1, ..., n_levels-1}."""
    M, I = g_matrix.shape
    if obs_mask is None:
        obs_mask = ~np.isnan(g_matrix)
    m_idx, i_idx = np.where(obs_mask)
    g_obs = np.clip(g_matrix[m_idx, i_idx].astype(int), 0, n_levels - 1)

    K = n_levels - 1  # number of thresholds per item
    # Params: theta (M), b_base (I), delta_raw (I, K-1) >= 0 via softplus, log_a (I)
    dim = M + I + I * (K - 1) + I
    x0 = np.zeros(dim, dtype=float)
    for m in range(M):
        mask_m = obs_mask[m]
        avg_g = np.clip((g_matrix[m, mask_m].mean() / K) if mask_m.any() else 0.5, 0.05, 0.95)
        x0[m] = math.log(avg_g / (1.0 - avg_g))
    x0[:M] -= x0[:M].mean()

    def unpack(x: np.ndarray):
        theta = x[:M]
        b0 = x[M : M + I]
        deltas = np.log1p(np.exp(np.clip(x[M + I : M + I + I * (K - 1)].reshape(I, K - 1), -8.0, 8.0))) + 0.05
        log_a = x[M + I + I * (K - 1) :]
        a = np.exp(np.clip(log_a, -2.0, 2.0))
        thresholds = np.zeros((I, K), dtype=float)
        thresholds[:, 0] = b0 - 0.5 * deltas.sum(axis=1)
        for k in range(1, K):
            thresholds[:, k] = thresholds[:, k - 1] + deltas[:, k - 1]
        return theta, b0, log_a, a, thresholds

    def nll(x: np.ndarray) -> float:
        theta, b0, log_a, a, thresholds = unpack(x)
        # P(G >= k) for k=1..K; P(G >= 0)=1, P(G >= K+1)=0
        eta = a[i_idx, None] * (theta[m_idx, None] - thresholds[i_idx, :])
        cum_p = expit(eta)  # shape (N_obs, K)
        lower = np.where(g_obs == 0, 1.0, cum_p[np.arange(len(g_obs)), np.maximum(g_obs - 1, 0)])
        upper = np.where(g_obs == K, 0.0, cum_p[np.arange(len(g_obs)), np.minimum(g_obs, K - 1)])
        prob = np.clip(lower - upper, 1e-9, 1.0)
        loss = -np.sum(np.log(prob))
        loss += 0.5 * np.sum(theta * theta) / (prior_theta_sd ** 2)
        loss += 5.0 * (np.mean(theta) ** 2)
        loss += 0.5 * np.sum(b0 * b0) / (prior_b_sd ** 2)
        loss += 0.5 * np.sum(log_a * log_a) / 0.25
        return float(loss)

    res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 250})
    theta, _, _, a, thresholds = unpack(res.x)
    b_mid = thresholds.mean(axis=1)
    # Expected normalized partial credit score across full bank
    eta_full = a[None, :, None] * (theta[:, None, None] - thresholds[None, :, :])
    cum_full = expit(eta_full)  # (M, I, K)
    expected_g = cum_full.sum(axis=2) / K
    pred_full = expected_g.mean(axis=1)

    return IRTFitResult(
        variant="graded_partial_credit",
        models=list(models),
        items=list(items),
        theta=theta,
        theta_se=np.full(M, 0.2, dtype=float),
        b=b_mid,
        a=a,
        predicted_full_scores=pred_full,
    )


def item_fisher_information(fit: IRTFitResult, theta_grid: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute mean Fisher information per item across model abilities (or grid)."""
    thetas = fit.theta if theta_grid is None else theta_grid
    eta = fit.a[None, :] * (thetas[:, None] - fit.b[None, :])
    p = expit(eta)
    info = (fit.a[None, :] ** 2) * p * (1.0 - p)
    return info.mean(axis=0)
