"""ES calibration tests: Acerbi–Székely Z2 and exceedance-residual (ER) bootstrap.

Sign convention: lower-tail returns; v (VaR) and e (ES) negative, e < v.
"""

from __future__ import annotations

import numpy as np


def as_z2(y: np.ndarray, v: np.ndarray, e: np.ndarray, alpha: float,
          n_boot: int = 0, seed: int | None = None) -> dict:
    """Acerbi–Székely (2014) Z2 statistic:

        Z2 = (1/(n·α)) Σ_t y_t·1{y_t <= v_t} / e_t  −  1

    0 under correct (VaR, ES); > 0 when realized tail losses exceed predicted ES
    (both y and e negative in the tail, so the ratio is positive).
    Optional p-value by iid bootstrap of the per-observation contributions
    (recentered, one-sided: too-large Z2 rejects).
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    e = np.asarray(e, dtype=float)
    if np.any(e >= 0):
        raise ValueError("Z2 requires e < 0")
    n = len(y)
    contrib = np.where(y <= v, y / e, 0.0)
    z2 = contrib.sum() / (n * alpha) - 1.0
    out = {"stat": float(z2), "n_hits": int((y <= v).sum())}

    if n_boot:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        z2_b = contrib[idx].sum(axis=1) / (n * alpha) - 1.0
        z2_b -= z2_b.mean()
        out["p_onesided"] = float((z2_b >= z2).mean())
        out["p_twosided"] = float((np.abs(z2_b) >= abs(z2)).mean())
    return out


def er_backtest(y: np.ndarray, v: np.ndarray, e: np.ndarray,
                n_boot: int = 20000, seed: int = 20260704) -> dict:
    """Exceedance-residual bootstrap test (McNeil–Frey 2000 flavor), mirroring
    esback::er_backtest 'simple': on violation days (y <= v) form x = y − e,
    t0 = mean(x)/sd(x)·sqrt(m), iid-bootstrap x, recenter the bootstrap t stats.

        p_twosided = P(|t* − mean(t*)| >= |t0|)
        p_onesided = P(t* − mean(t*) <= t0)   (small p ⟺ residuals too negative
                                               ⟺ realized tail worse than ES)

    Bootstrap RNG differs from R's, so fixture comparison uses a Monte-Carlo
    tolerance, not exact equality.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    e = np.asarray(e, dtype=float)
    x = (y - e)[y <= v]
    m = len(x)
    if m < 5:
        raise ValueError("too few exceedances for ER test")

    def tstat(a: np.ndarray, axis=None):
        return a.mean(axis=axis) / a.std(axis=axis, ddof=1) * np.sqrt(m)

    t0 = float(tstat(x))
    rng = np.random.default_rng(seed)
    bx = x[rng.integers(0, m, size=(n_boot, m))]
    t = bx.mean(axis=1) / bx.std(axis=1, ddof=1) * np.sqrt(m)
    t = t[np.isfinite(t)]
    tc = t - t.mean()
    return {"stat": t0, "n_exceed": m,
            "p_twosided": float((np.abs(tc) >= abs(t0)).mean()),
            "p_onesided": float((tc <= t0).mean())}
