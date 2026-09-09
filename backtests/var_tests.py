"""VaR calibration tests: Kupiec POF, Christoffersen independence/CC, Engle–Manganelli DQ.

All take y (returns) and v (VaR forecasts, lower tail, same units) as aligned arrays.
A violation is y_t <= v_t (ties counted as hits, matching rugarch/GAS).
Fixture-verified against rugarch::VaRTest and GAS::BacktestVaR on the frozen input.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _xlogy(x: float, p: float) -> float:
    """x*log(p) with the 0*log(0)=0 convention."""
    return 0.0 if x == 0 else x * np.log(p)


def hits(y: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (np.asarray(y, dtype=float) <= np.asarray(v, dtype=float)).astype(int)


def kupiec(y: np.ndarray, v: np.ndarray, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures LR test. LR ~ χ²(1) under H0."""
    h = hits(y, v)
    n, x = len(h), int(h.sum())
    pi = x / n
    ll0 = _xlogy(n - x, 1 - alpha) + _xlogy(x, alpha)
    ll1 = _xlogy(n - x, 1 - pi) + _xlogy(x, pi) if 0 < pi < 1 else 0.0 if x in (0, n) else np.nan
    if x in (0, n):
        ll1 = 0.0
    lr = -2.0 * (ll0 - ll1)
    return {"stat": float(lr), "p": float(stats.chi2.sf(lr, 1)), "n": n, "hits": x,
            "hit_rate": pi}


def christoffersen_independence(y: np.ndarray, v: np.ndarray) -> dict:
    """Christoffersen (1998) first-order Markov independence LR test. LR ~ χ²(1)."""
    h = hits(y, v)
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
           + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))
    lr = -2.0 * (ll0 - ll1)
    return {"stat": float(lr), "p": float(stats.chi2.sf(lr, 1)),
            "transitions": (n00, n01, n10, n11)}


def christoffersen_cc(y: np.ndarray, v: np.ndarray, alpha: float) -> dict:
    """Conditional coverage: LR_cc = LR_uc + LR_ind ~ χ²(2)."""
    uc = kupiec(y, v, alpha)
    ind = christoffersen_independence(y, v)
    lr = uc["stat"] + ind["stat"]
    return {"stat": float(lr), "p": float(stats.chi2.sf(lr, 2)),
            "uc": uc, "ind": ind}


def dq_test(y: np.ndarray, v: np.ndarray, alpha: float, lags: int = 4,
            include_var: bool = True) -> dict:
    """Engle–Manganelli (2004) out-of-sample dynamic quantile test.

    Demeaned hit Hit_t = 1{y_t<=v_t} - α regressed on a constant, `lags` own lags,
    and (optionally) the contemporaneous VaR forecast; DQ = Hit'X(X'X)⁻¹X'Hit / (α(1-α))
    ~ χ²(#regressors). Regressor set mirrors GAS::BacktestVaR (Lags=4).
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    hit = hits(y, v).astype(float) - alpha
    n = len(hit)
    rows = n - lags
    cols = [np.ones(rows)]
    for k in range(1, lags + 1):
        cols.append(hit[lags - k: n - k])
    if include_var:
        cols.append(v[lags:])
    X = np.column_stack(cols)
    H = hit[lags:]

    XtX_inv = np.linalg.pinv(X.T @ X)
    stat = H @ X @ XtX_inv @ X.T @ H / (alpha * (1 - alpha))
    df = X.shape[1]
    return {"stat": float(stat), "p": float(stats.chi2.sf(stat, df)), "df": df}
