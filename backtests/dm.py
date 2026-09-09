"""Diebold–Mariano test for equal predictive accuracy on a loss differential."""

from __future__ import annotations

import numpy as np
from scipy import stats


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1,
            harvey_correction: bool = True) -> dict:
    """DM test on d_t = loss_a - loss_b (positive mean → b better).

    Long-run variance via uniform-kernel HAC with h-1 lags (the classic DM
    variance for h-step forecasts; h=1 → plain sample variance). Small-sample
    Harvey–Leybourne–Newbold correction and t distribution by default.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10:
        raise ValueError("too few observations for DM")
    dbar = d.mean()
    dc = d - dbar
    gamma0 = dc @ dc / n
    lrv = gamma0
    for k in range(1, h):
        gk = dc[k:] @ dc[:-k] / n
        lrv += 2.0 * gk
    if lrv <= 0:
        lrv = gamma0
    stat = dbar / np.sqrt(lrv / n)

    if harvey_correction:
        adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        stat *= adj
        p = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    else:
        p = 2.0 * stats.norm.sf(abs(stat))
    return {"stat": float(stat), "p": float(p), "mean_diff": float(dbar), "n": n}
