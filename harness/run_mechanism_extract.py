"""Mechanism sub-panel: E0-E3 extraction + MC-calibrated backtest battery on the
mechanism forecast grids (results/mechanism/forecast/*.parquet).

    python -m harness.run_mechanism_extract

Reuses the FROZEN stage-4 battery, E-rules, and per-asset context fits (nu, GPD)
verbatim (imported from harness.run_grid_extract) so the mechanism heads are
scored on exactly the same machinery and anchors as the main panel. The two
mechanism heads are sampling-native; their forecast parquets already hold S=1000
empirical DEEP-grid quantiles, so E0 is the native deep quantile and E1-E3 build
on the same central anchors as every other DEEP-grid head. Output:
results/mechanism/backtests.csv (one row per forecaster x asset x alpha x erule).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.load import load_series
from erules.rules import (e1_es, e1_quantile, e2_es, e2_quantile,
                          e3_var_es_from_fit)
from harness.run_grid_extract import (ALPHAS, CTX_LEN, DEEP_TAUS, ERULES, acol,
                                      battery, ctx_fits)

ROOT = Path(__file__).resolve().parent.parent
FC = ROOT / "results" / "mechanism" / "forecast"
OUT = ROOT / "results" / "mechanism"
MANIFEST = ROOT / "results" / "stage4" / "run_manifest.yaml"
ASSETS12 = ["brent", "gold", "btc", "eth", "hsi", "spx",
            "aapl", "nvda", "audusd", "gbpusd", "dgs10", "dgs2"]
MODEL_TAUS = {"chronos_base": DEEP_TAUS, "moirai_1_1": DEEP_TAUS}


def main() -> None:
    mf = yaml.safe_load(MANIFEST.read_text())
    oos = mf["protocol"]["oos_start"]
    rows = []
    for asset in ASSETS12:
        if not any((FC / f"{m}_{asset}.parquet").exists() for m in MODEL_TAUS):
            print(f"[extract] {asset}: no forecast yet, skip", flush=True)
            continue
        fits = ctx_fits(asset, oos).set_index("t")   # cached stage-4 ctxfits (identical anchors)
        df = load_series(asset)
        x = df[df.attrs.get("column", "logret")].to_numpy()
        for model, taus in MODEL_TAUS.items():
            fpq = FC / f"{model}_{asset}.parquet"
            if not fpq.exists():
                continue
            d = pd.read_parquet(fpq)
            recs = []
            for _, r in d.iterrows():
                t = int(r["t"])
                grid = {tau: float(r[c]) for tau, c in taus.items()
                        if c in r and np.isfinite(r[c])}
                f = fits.loc[t]
                q50c = float(np.median(x[t - CTX_LEN:t]))
                gpd_ok = np.isfinite(f["xi"]) and np.isfinite(f["beta"])
                rec = {"t": t, "date": r["date"], "y": r["y"]}
                for a in ALPHAS:
                    c = acol(a)
                    rec[f"e0_v{c}"] = grid.get(a, np.nan)
                    rec[f"e1_v{c}"] = e1_quantile(grid, a)
                    rec[f"e1_s{c}"] = e1_es(grid, a)
                    rec[f"e2_v{c}"] = e2_quantile(grid, a, f["nu"])
                    rec[f"e2_s{c}"] = e2_es(grid, a, f["nu"])
                    if gpd_ok:
                        # v2.0 P0-1: independent VaR/ES exception paths — ES-only
                        # failure (xi>=1 mean-excess divergence) keeps the VaR.
                        # v2.1: empirical tau_mass from the shared ctxfits row.
                        rec[f"e3_v{c}"], rec[f"e3_s{c}"] = e3_var_es_from_fit(
                            grid, q50c, f["u"], f["xi"], f["beta"], a,
                            tau_mass=float(f["tau_mass"]))
                    else:
                        rec[f"e3_v{c}"] = rec[f"e3_s{c}"] = np.nan
                recs.append(rec)
            ex = pd.DataFrame(recs)
            for er in ERULES:
                for a in ALPHAS:
                    c = acol(a)
                    vcol = f"{er}_v{c}"
                    if vcol not in ex or ex[vcol].isna().all():
                        continue
                    ecol = ex.get(f"{er}_s{c}")
                    b = battery(ex["y"].to_numpy(), ex[vcol].to_numpy(),
                                ecol.to_numpy() if ecol is not None else None, a)
                    rows.append({"forecaster": model, "asset": asset, "alpha": a,
                                 "erule": er, "group": df.attrs["group"], **b})
        print(f"[extract] {asset} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "backtests.csv", index=False)
    print(f"mechanism extract complete: {len(rows)} cells", flush=True)


if __name__ == "__main__":
    main()
