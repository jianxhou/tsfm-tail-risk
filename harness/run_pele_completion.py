"""Pele-protocol decile-completion self-check on the stored TimesFM-2.5 grids
(v2.0 P0-7; post-hoc, reviewer-driven — CHANGELOG 2026-07-30, NOT pre-specified).

    python -m harness.run_pele_completion

Applies the nine-decile location-scale Student-t completion (the protocol of
pele2026recalibrating as characterized in docs/related_work_verification.md §8)
to this project's own stored TimesFM-2.5 forecast parquets: per window, fit
(mu, sigma) by least squares of the nine stored deciles on the t(nu) decile
quantiles for each nu in NU_GRID, keep the min-RMSE nu, and read the completed
1% quantile mu + sigma * t_ppf(0.01, nu). Reports the per-asset 1% violation
rate (y < completed q01) over every stored OOS window, plus mean/min/max
summary rows. Cross-reference: the signer's five-asset spot check under the
same protocol gave 0.91%-2.61% (docs/external_review_9_adjudication.md P0-7).
Output: results/stage4/pele_completion.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FC = ROOT / "results" / "stage4" / "forecast"
OUT = ROOT / "results" / "stage4" / "pele_completion.csv"
DECILES = np.arange(1, 10) / 10.0
QCOLS = [f"q{k}" for k in range(1, 10)]
NU_GRID = (3, 4, 5, 6, 8, 10, 15, 20, 30)
ALPHA = 0.01


def complete_q01(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row min-RMSE location-scale t completion of the 1% quantile.

    Q is (n, 9) stored deciles. For each nu, (mu, sigma) solve the per-row OLS
    of Q on the t(nu) decile quantiles; the nu minimizing the fit RMSE is kept
    row by row. Returns (q01, nu_star)."""
    n = len(Q)
    qbar = Q.mean(axis=1)
    best_rmse = np.full(n, np.inf)
    q01 = np.full(n, np.nan)
    nu_star = np.zeros(n)
    for nu in NU_GRID:
        z = stats.t.ppf(DECILES, nu)
        zc = z - z.mean()
        sigma = (Q - qbar[:, None]) @ zc / (zc @ zc)
        mu = qbar - sigma * z.mean()
        resid = Q - (mu[:, None] + sigma[:, None] * z[None, :])
        rmse = np.sqrt((resid ** 2).mean(axis=1))
        better = rmse < best_rmse
        best_rmse = np.where(better, rmse, best_rmse)
        q01 = np.where(better, mu + sigma * stats.t.ppf(ALPHA, nu), q01)
        nu_star = np.where(better, nu, nu_star)
    return q01, nu_star


def main() -> None:
    rows = []
    for pq in sorted(FC.glob("timesfm_2_5_*.parquet")):
        asset = pq.stem.replace("timesfm_2_5_", "")
        d = pd.read_parquet(pq)
        ok = d[QCOLS].notna().all(axis=1)
        Q = d.loc[ok, QCOLS].to_numpy(dtype=float)
        y = d.loc[ok, "y"].to_numpy(dtype=float)
        q01, nu_star = complete_q01(Q)
        hits = y < q01
        rows.append({"asset": asset, "n": int(len(y)),
                     "n_dropped_nan": int((~ok).sum()),
                     "nu_median": float(np.median(nu_star)),
                     "hit01_pct": float(100.0 * hits.mean())})
        print(f"[pele] {asset}: n={len(y)} hit01={100.0*hits.mean():.2f}%", flush=True)
    df = pd.DataFrame(rows).sort_values("asset").reset_index(drop=True)
    h = df["hit01_pct"]
    summary = pd.DataFrame([
        {"asset": "ALL_MEAN", "n": int(df["n"].sum()), "n_dropped_nan": 0,
         "nu_median": np.nan, "hit01_pct": float(h.mean())},
        {"asset": "ALL_MIN", "n": np.nan, "n_dropped_nan": np.nan,
         "nu_median": np.nan, "hit01_pct": float(h.min())},
        {"asset": "ALL_MAX", "n": np.nan, "n_dropped_nan": np.nan,
         "nu_median": np.nan, "hit01_pct": float(h.max())}])
    pd.concat([df, summary], ignore_index=True).to_csv(OUT, index=False)
    print(f"pele completion: {len(df)} assets; mean {h.mean():.2f}% "
          f"range [{h.min():.2f}%, {h.max():.2f}%] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
