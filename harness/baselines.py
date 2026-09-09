"""Econometric VaR/ES baselines (proposal §3.1). Each is a pure function of a
context window (returns, most-recent-last) -> forecast at level alpha. GARCH-t /
GJR-t live via the `arch` package in the run drivers; this module holds the
non-parametric and CAViaR baselines. All are reference-tested in
tests/test_baselines.py BEFORE entering the grid (amendment B).

Sign convention: lower-tail VaR/ES are negative numbers in return units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def historical_simulation(window: np.ndarray, alpha: float) -> tuple[float, float]:
    """HS: empirical alpha-quantile (VaR) and mean of the exceedances (ES)."""
    w = np.asarray(window, dtype=float)
    v = float(np.quantile(w, alpha))
    tail = w[w <= v]
    e = float(tail.mean()) if len(tail) else v
    return v, e


def ewma_var(window: np.ndarray, alpha: float, lam: float = 0.94) -> tuple[float, float]:
    """RiskMetrics EWMA volatility with a normal tail (VaR, ES).
    sigma^2_t = lam*sigma^2_{t-1} + (1-lam)*y^2_{t-1}, seeded at the sample var."""
    from scipy import stats
    w = np.asarray(window, dtype=float)
    s2 = np.var(w)
    for y in w:
        s2 = lam * s2 + (1 - lam) * y * y
    sigma = np.sqrt(s2)
    z = stats.norm.ppf(alpha)
    v = sigma * z
    e = -sigma * stats.norm.pdf(z) / alpha
    return float(v), float(e)


def filtered_historical_simulation(window: np.ndarray, alpha: float,
                                   lam: float = 0.94) -> tuple[float, float]:
    """FHS (McNeil-Frey flavor): standardize by EWMA vol, take the empirical
    alpha-quantile of standardized residuals, rescale by the one-step-ahead vol.
    Reduces to HS when volatility is constant."""
    w = np.asarray(window, dtype=float)
    s2 = np.var(w)
    sig = np.empty(len(w))
    for i, y in enumerate(w):
        sig[i] = np.sqrt(s2)
        s2 = lam * s2 + (1 - lam) * y * y
    sig_next = np.sqrt(s2)
    z = w / sig
    zq = float(np.quantile(z, alpha))
    v = sig_next * zq
    tail = z[z <= zq]
    e = sig_next * float(tail.mean()) if len(tail) else sig_next * zq
    return float(v), float(e)


# ---- CAViaR-SAV (Engle & Manganelli 2004) ----

def _rq_loss(y: np.ndarray, v: np.ndarray, alpha: float) -> float:
    """Regression-quantile (check) loss: sum rho_alpha(y - v)."""
    u = y - v
    return float(np.sum(u * (alpha - (u < 0).astype(float))))


def _caviar_sav_path(beta: np.ndarray, y: np.ndarray, v1: float) -> np.ndarray:
    """Symmetric-Absolute-Value recursion: v_t = b0 + b1 v_{t-1} + b2 |y_{t-1}|."""
    b0, b1, b2 = beta
    v = np.empty(len(y))
    v[0] = v1
    for t in range(1, len(y)):
        v[t] = b0 + b1 * v[t - 1] + b2 * abs(y[t - 1])
    return v


class CaviarFitError(RuntimeError):
    """All CAViaR-SAV starts failed the stability/validity conditions (v2.0
    P0-3); callers record the window as NaN plus diagnostics."""


B1_CAP = 0.999            # |b1| < 1 via b1 = B1_CAP * tanh(theta1) (SAV stability)
PATH_CAP_MULT = 100.0     # auto-fail: max|v_path| <= 100 * |unconditional alpha quantile|


def _sav_theta_to_beta(theta: np.ndarray) -> np.ndarray:
    """theta -> beta with the persistence coefficient squashed to |b1| < 1."""
    t = np.asarray(theta, dtype=float)
    return np.array([t[0], B1_CAP * np.tanh(t[1]), t[2]])


def _sav_beta_to_theta(beta: np.ndarray) -> np.ndarray:
    b = np.asarray(beta, dtype=float)
    b1 = np.clip(b[1] / B1_CAP, -0.9995, 0.9995)
    return np.array([b[0], np.arctanh(b1), b[2]])


@dataclass
class CaviarFit:
    beta: np.ndarray
    loss: float
    v1: float
    alpha: float
    n_starts: int
    loss_spread: float        # max-min objective across VALID starts
    best_start_idx: int       # index into the valid-candidate pool
    converged_starts: int     # optimizer-reported convergence count (res.success)
    valid_starts: int         # candidates passing the stability/validity conditions


def fit_caviar_sav(window: np.ndarray, alpha: float, n_starts: int = 10,
                   seed: int = 20260706, warmup: int = 300) -> CaviarFit:
    """Multi-start RQ estimation of CAViaR-SAV, CONSTRAINED (v2.0 P0-3,
    external review #9): optimization runs in theta space with
    b1 = 0.999*tanh(theta1), so the recursion is contractive by construction
    (the old unconstrained Nelder-Mead diverged on 8+ assets, producing
    million-scale and positive lower-tail VaR paths). Every candidate fit must
    pass automatic failure conditions before entering the pool:
      - in-sample VaR path entirely finite,
      - max|v_path| <= 100 * |unconditional alpha quantile of the window|,
      - one-step-ahead v_next < 0 (lower-tail sign).
    No start passing -> CaviarFitError (callers NaN the window and record
    diagnostics). converged_starts now counts optimizer-converged runs
    (res.success), not merely finite objectives (old semantics bug)."""
    y = np.asarray(window, dtype=float)
    v1 = float(np.quantile(y[:min(warmup, len(y))], alpha))
    rng = np.random.default_rng(seed)

    uncond = abs(np.quantile(y, alpha))
    # start grid (beta space): persistence-dominated + reaction-dominated + random
    base = [np.array([v1 * 0.1, 0.9, -0.1 * uncond]),
            np.array([v1 * 0.5, 0.5, -0.3 * uncond]),
            np.array([v1, 0.0, 0.0])]
    rand = [np.array([rng.uniform(-1, 0) * uncond,
                      rng.uniform(0, 0.95),
                      rng.uniform(-0.5, 0) * uncond]) for _ in range(n_starts - len(base))]
    starts = [_sav_beta_to_theta(s) for s in base + rand]

    def obj(theta):
        return _rq_loss(y, _caviar_sav_path(_sav_theta_to_beta(theta), y, v1), alpha)

    n_converged = 0
    valid = []                      # (loss, beta) passing the auto-fail conditions
    for s in starts:
        r = minimize(obj, s, method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-8})
        n_converged += int(bool(r.success))
        if not np.isfinite(r.fun):
            continue
        beta = _sav_theta_to_beta(r.x)
        path = _caviar_sav_path(beta, y, v1)
        v_next = beta[0] + beta[1] * path[-1] + beta[2] * abs(y[-1])
        if (np.all(np.isfinite(path))
                and np.max(np.abs(path)) <= PATH_CAP_MULT * uncond
                and np.isfinite(v_next) and v_next < 0):
            valid.append((float(r.fun), beta))
    if not valid:
        raise CaviarFitError(
            f"CAViaR-SAV: 0/{len(starts)} starts passed the validity conditions "
            f"(alpha={alpha}, n={len(y)}, uncond={uncond:.4g})")
    losses = np.array([f for f, _ in valid])
    best = int(np.argmin(losses))
    return CaviarFit(beta=valid[best][1], loss=float(losses[best]), v1=v1,
                     alpha=alpha, n_starts=len(starts),
                     loss_spread=float(losses.max() - losses.min()),
                     best_start_idx=best, converged_starts=n_converged,
                     valid_starts=len(valid))


def caviar_sav_forecast(window: np.ndarray, alpha: float, **kw) -> tuple[float, CaviarFit]:
    """One-step-ahead CAViaR-SAV VaR from a context window. VaR only (CAViaR is a
    quantile model; ES is not part of the standard SAV spec)."""
    fit = fit_caviar_sav(window, alpha, **kw)
    y = np.asarray(window, dtype=float)
    path = _caviar_sav_path(fit.beta, y, fit.v1)
    b0, b1, b2 = fit.beta
    v_next = b0 + b1 * path[-1] + b2 * abs(y[-1])
    return float(v_next), fit
