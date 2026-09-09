"""Pilot D3 driver: tail extraction (E0-E3, VaR+ES) + MC/S calibration + E3
production-mode variance calibration.

    nohup python -m harness.run_pilot_d3 > results/pilot_d3/run.log 2>&1 &

Part A  For each (model, asset): join stored D2 native grids with per-window
        context fits (nu MLE, GPD POT at tau_u=0.10 — computed once per asset,
        shared across models) -> q/ES at alpha in {1%, 2.5%, 5%} under E0 (native,
        where the head provides the level), E1, E2, E3.
Part B  Sampling-head S calibration (lag_llama, SPX, 16 windows): empirical
        quantiles at S in {250, 1000, 4000} vs the analytic parametric truth.
Part C  E3 estimation-noise calibration: bootstrap (B=200) the exceedance set at
        ctx in {256, 512, 2048} -> dispersion of the GPD deep quantile vs
        exceedance count. Reported in ONE table with Part B (extraction-noise
        sources), docs/pilot_d3_calibration.md (script-generated).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data.load import load_series
from erules.rules import (e1_es, e1_quantile, e2_es, e2_quantile, e3_es,
                          e3_quantile, fit_gpd, fit_nu, gpd_tail_quantile)
from harness.rolling import rolling_windows

OUT = Path(__file__).resolve().parent.parent / "results" / "pilot_d3"
D2 = OUT.parent / "pilot_d2"
CTX_LEN = 512
OOS_START = "2018-01-01"
ASSETS = ["spx", "btc", "nvda"]
ALPHAS = (0.01, 0.025, 0.05)
SEED = 20260705

DEEP_TAUS = {0.01: "q01", 0.025: "q025", 0.05: "q05", 0.1: "q1", 0.25: "q25",
             0.5: "q5", 0.75: "q75", 0.9: "q9"}
DECILE_TAUS = {round(0.1 * k, 1): "q" + f"{round(0.1*k,1):g}".replace("0.", "")
               for k in range(1, 10)}
MODELS = {"chronos_bolt": DEEP_TAUS, "chronos_2": DEEP_TAUS,
          "timesfm_2_5": DECILE_TAUS, "moirai_2_0": DEEP_TAUS}


def ctx_fits(asset: str) -> pd.DataFrame:
    """Per-window nu and GPD(tau_u=0.10) fits on the ctx — shared across models."""
    cache = OUT / f"ctxfits_{asset}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = load_series(asset)
    rs = rolling_windows(df, value_col="logret", ctx_len=CTX_LEN,
                         oos_start=OOS_START)
    rows = []
    for w in rs.windows:
        u = float(np.quantile(w.ctx, 0.10))
        exc = u - w.ctx[w.ctx < u]
        xi, beta, _ = fit_gpd(exc)           # v2.1 3-tuple; pilot cache schema kept
        rows.append({"t": w.t, "nu": fit_nu(w.ctx), "u": u, "xi": xi,
                     "beta": beta, "q50c": float(np.quantile(w.ctx, 0.5))})
    out = pd.DataFrame(rows)
    out.to_parquet(cache, index=False)
    return out


def part_a() -> None:
    for asset in ASSETS:
        fits = ctx_fits(asset).set_index("t")
        print(f"[A] ctx fits ready for {asset}", flush=True)
        df = load_series(asset)
        x = df["logret"].to_numpy()
        for model, taus in MODELS.items():
            outpq = OUT / f"extract_{model}_{asset}.parquet"
            if outpq.exists():
                continue
            d2 = pd.read_parquet(D2 / f"{model}_{asset}.parquet")
            rows = []
            for _, r in d2.iterrows():
                t = int(r["t"])
                grid = {tau: float(r[col]) for tau, col in taus.items()
                        if col in r and np.isfinite(r[col])}
                f = fits.loc[t]
                ctx = x[t - CTX_LEN: t]
                row = {"t": t, "date": r["date"], "y": r["y"]}
                for a in ALPHAS:
                    acol = f"{a:g}".replace("0.", "")
                    row[f"e0_v{acol}"] = grid.get(a, np.nan)   # native or absent
                    row[f"e1_v{acol}"] = e1_quantile(grid, a)
                    row[f"e1_s{acol}"] = e1_es(grid, a)
                    row[f"e2_v{acol}"] = e2_quantile(grid, a, f["nu"])
                    row[f"e2_s{acol}"] = e2_es(grid, a, f["nu"])
                    row[f"e3_v{acol}"] = e3_quantile(grid, ctx, a)
                    row[f"e3_s{acol}"] = e3_es(grid, ctx, a)
                row["nu"] = f["nu"]
                rows.append(row)
            pd.DataFrame(rows).to_parquet(outpq, index=False)
            print(f"[A] {model}/{asset} extracted", flush=True)


def part_b() -> pd.DataFrame:
    csv = OUT / "s_calibration.csv"
    if csv.exists():
        return pd.read_csv(csv)
    from harness.lagllama_adapter import LagLlamaAdapter
    adapter = LagLlamaAdapter(seed=SEED)
    df = load_series("spx")
    rs = rolling_windows(df, value_col="logret", ctx_len=CTX_LEN,
                         oos_start=OOS_START)
    idx = np.linspace(0, len(rs.windows) - 1, 16).astype(int)
    rows = []
    for i in idx:
        w = rs.windows[i]
        truth = adapter.predict_quantiles(w.ctx, list(ALPHAS))
        for S in (250, 1000, 4000):
            s = adapter.predict_samples(w.ctx, S)
            for a, tv in zip(ALPHAS, truth):
                rows.append({"window_t": w.t, "S": S, "alpha": a,
                             "emp": float(np.quantile(s, a)), "truth": float(tv)})
        print(f"[B] window t={w.t} done", flush=True)
    out = pd.DataFrame(rows)
    out["abs_err"] = (out.emp - out.truth).abs()
    out.to_csv(csv, index=False)
    return out


def part_c() -> pd.DataFrame:
    csv = OUT / "e3_variance.csv"
    if csv.exists():
        return pd.read_csv(csv)
    rng = np.random.default_rng(SEED)
    df = load_series("spx")
    x = df["logret"].to_numpy()
    rows = []
    for ctx_len in (256, 512, 2048):
        rs = rolling_windows(df, value_col="logret", ctx_len=ctx_len,
                             oos_start=OOS_START)
        for i in np.linspace(0, len(rs.windows) - 1, 10).astype(int):
            w = rs.windows[i]
            u = float(np.quantile(w.ctx, 0.10))
            exc = u - w.ctx[w.ctx < u]
            qs = []
            for _ in range(200):
                bs = exc[rng.integers(0, len(exc), len(exc))]
                try:
                    # pilot-era caliber kept (frozen Stage-2 D3 exhibit, not in
                    # the v2.1 rerun chain): nominal 0.10 extrapolation mass
                    xi, beta, _ = fit_gpd(bs)
                    qs.append(gpd_tail_quantile(u, xi, beta, 0.10, 0.01))
                except ValueError:
                    continue
            rows.append({"ctx_len": ctx_len, "n_exceed": len(exc),
                         "window_t": w.t, "q01_boot_std": float(np.std(qs, ddof=1))})
    out = pd.DataFrame(rows)
    out.to_csv(csv, index=False)
    return out


def summary(sb: pd.DataFrame, sc: pd.DataFrame) -> None:
    lines = ["# Pilot D3 extraction-noise calibration (script-generated; do not hand-edit)",
             "", "Sources of deep-tail extraction noise, one table (user spec, D3):", "",
             "| source | setting | tail points | MAE q01 | MAE q025 | MAE q05 |",
             "|---|---|---|---|---|---|"]
    for S in (250, 1000, 4000):
        m = [sb[(sb.S == S) & (sb.alpha == a)].abs_err.mean() for a in ALPHAS]
        lines.append(f"| sampling head MC (lag_llama, S draws) | S={S} | ~{S*0.01:.0f}/{S*0.025:.0f}/{S*0.05:.0f} "
                     f"| {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} |")
    for ctx_len in (256, 512, 2048):
        g = sc[sc.ctx_len == ctx_len]
        lines.append(f"| E3 GPD estimation (bootstrap std of q01, ctx window) | ctx={ctx_len} "
                     f"| {g.n_exceed.mean():.0f} exceedances | {g.q01_boot_std.mean():.3f} (std) | — | — |")
    lines += ["", "Units: log-return ×100. Sampling MAE = |empirical − analytic-parametric truth|, "
              "16 SPX windows. E3 std = bootstrap (B=200) dispersion of the GPD-extrapolated "
              "ctx-side q01, 10 SPX windows per ctx length.", ""]
    (Path(__file__).resolve().parent.parent / "docs" / "pilot_d3_calibration.md"
     ).write_text("\n".join(lines))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    part_a()
    sb = part_b()
    sc = part_c()
    summary(sb, sc)
    print(f"D3 complete in {(time.time()-t0)/60:.1f} min "
          f"@ {datetime.now(timezone.utc).isoformat(timespec='seconds')}", flush=True)
