"""Synthetic GARCH-t and GJR-GARCH-t paths with ANALYTICALLY KNOWN one-step-ahead
conditional VaR/ES (condition 10 arbitration + oracle arm). Returns log-return-x100
scale so the TSFMs see the same units as real data.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _t_var_es(alpha: float, nu: float):
    """Standardized (unit-variance) Student-t VaR and ES at level alpha."""
    sc = np.sqrt((nu - 2.0) / nu)
    tq = stats.t.ppf(alpha, nu)
    var = tq * sc
    es = (-stats.t.pdf(tq, nu) * (nu + tq**2) / ((nu - 1.0) * alpha)) * sc
    return var, es


def garch_t_path(seed: int, n: int = 1500, n_burn: int = 500,
                 omega=0.05, a=0.09, b=0.90, nu=8.0, gjr_gamma=0.0):
    """Simulate a (GJR-)GARCH-t path. gjr_gamma>0 adds leverage:
        sigma2_t = omega + (a + gamma*1{y<0}) y_{t-1}^2 + b sigma2_{t-1}.
    Returns (y, sigma) for the kept sample and the fixed nu.
    Stationarity requires a + b + gamma/2 < 1 (symmetric-innovation E[1{y<0}]=1/2).
    """
    persistence = a + b + gjr_gamma / 2.0
    if persistence >= 1.0:
        raise ValueError(f"non-stationary: a+b+gamma/2 = {persistence:.3f} >= 1")
    rng = np.random.default_rng(seed)
    sc = np.sqrt((nu - 2.0) / nu)
    z = rng.standard_t(nu, n_burn + n) * sc
    N = n_burn + n
    sig2 = np.empty(N); y = np.empty(N)
    sig2[0] = omega / (1.0 - persistence)
    y[0] = np.sqrt(sig2[0]) * z[0]
    for t in range(1, N):
        lev = gjr_gamma if y[t - 1] < 0 else 0.0
        sig2[t] = omega + (a + lev) * y[t - 1] ** 2 + b * sig2[t - 1]
        y[t] = np.sqrt(sig2[t]) * z[t]
    return y[n_burn:], np.sqrt(sig2[n_burn:]), nu


def true_var_es(sigma_next: float, nu: float, alpha: float):
    """Known conditional VaR/ES for the next step given the recursion sigma."""
    v_std, e_std = _t_var_es(alpha, nu)
    return sigma_next * v_std, sigma_next * e_std


def known_var_es_series(sigma: np.ndarray, nu: float, alpha: float):
    v_std, e_std = _t_var_es(alpha, nu)
    return sigma * v_std, sigma * e_std
