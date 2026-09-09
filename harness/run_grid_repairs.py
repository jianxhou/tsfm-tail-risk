"""Stage-5 repair grid: F1-F4 + H on E3 (primary) & E2 (robustness) series for
every TSFM x asset, plus the GARCH-EVT comparator, with MC-calibrated battery.

    PYTHONUNBUFFERED=1 python -u -m harness.run_grid_repairs

Run discipline: unbuffered, per-(model,asset) progress with measured s/asset, a
45-min-per-model guard. E2 and E3 columns are produced together (never
sequentially). Output: results/stage4/repair_backtests.csv (forecaster='<model>+<arm>'
or 'garch_evt', erule in {e3,e2,param}). GARCH mu/sigma path cached per asset.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.load import load_series
from erules.rules import e2_es, e2_quantile, e3_var_es_from_fit
from fixes import arms
from fixes.garch_filter import garch_mu_sigma_path
from harness.rolling import rolling_windows
from harness.run_grid_extract import battery, load_ctxfits
from precision.diagnostics import n_alpha

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
FC = S4 / "forecast"
CTX = 512
ALPHAS = (0.01, 0.025, 0.05)
TSFM = ["chronos_bolt", "chronos_2", "timesfm_2_5", "moirai_2_0", "lag_llama"]
DEEP = {0.01: "q01", 0.025: "q025", 0.05: "q05", 0.1: "q1", 0.25: "q25",
        0.5: "q5", 0.75: "q75", 0.9: "q9"}
DECILE = {round(0.1 * k, 1): "q" + f"{round(0.1*k,1):g}".replace("0.", "")
          for k in range(1, 10)}
TAUS = {m: (DECILE if m == "timesfm_2_5" else DEEP) for m in TSFM}
SHAT_WIN = 500      # v2.0 P0-2, design §A: ŝ = median central-grid scale over the
                    # series' FIRST 500-day window, computed once, fixed (no lookahead)
_SIG_CACHE: dict = {}


def shat_first_window(al: pd.DataFrame) -> float:
    """ŝ per the frozen definition (design §A): median s_t over the first
    SHAT_WIN rows of the aligned frame (the burn-in window), fixed."""
    return float(np.median(((al["q75"] - al["q25"]) / 1.349).iloc[:SHAT_WIN]))


def acol(a):
    return f"{a:g}".replace("0.", "")


def garch_path(asset, oos):
    if asset not in _SIG_CACHE:
        df = load_series(asset)
        rs = rolling_windows(df, value_col=df.attrs.get("column", "logret"),
                             ctx_len=1000, oos_start=oos, allow_late_start=True)
        ctxs = [w.ctx for w in rs.windows]
        mu, sig = garch_mu_sigma_path(ctxs)
        ts = np.array([w.t for w in rs.windows])
        _SIG_CACHE[asset] = (ts, mu, sig)
    return _SIG_CACHE[asset]


def build_aligned(model, asset, fits, x, oos):
    """E3 and E2 aligned frames (t, y, v, e, q25, q50, q75) + GARCH mu/sigma aligned
    to the same targets. Central-grid quartiles: q25/q75 for deep grids;
    timesfm uses q20/q80 mapped to a Gaussian-equivalent IQR scale."""
    d = pd.read_parquet(FC / f"{model}_{asset}.parquet")
    taus = TAUS[model]
    ts_g, mu_g, sig_g = garch_path(asset, oos)
    gmap = dict(zip(ts_g, range(len(ts_g))))
    rows_e3, rows_e2, mu_a, sig_a = [], [], [], []
    for _, r in d.iterrows():
        t = int(r["t"])
        if t not in fits.index or t not in gmap:
            continue
        grid = {tau: float(r[c]) for tau, c in taus.items()
                if c in r and np.isfinite(r[c])}
        f = fits.loc[t]
        ctx = x[t - CTX:t]
        q50 = grid[0.5]
        if model == "timesfm_2_5":                    # decile grid -> Gaussian IQR
            from scipy.stats import norm
            s_iqr = (grid[0.8] - grid[0.2]) / (2 * norm.ppf(0.8))
            q25, q75 = q50 - 0.674 * s_iqr, q50 + 0.674 * s_iqr
        else:
            q25, q75 = grid[0.25], grid[0.75]
        base = {"t": t, "y": r["y"], "q25": q25, "q50": q50, "q75": q75}
        gpd_ok = np.isfinite(f["xi"]) and np.isfinite(f["beta"])
        q50c = float(np.median(ctx))
        r3 = dict(base); r2 = dict(base)
        for a in ALPHAS:
            c = acol(a)
            if gpd_ok:
                # v2.0 P0-1: independent VaR/ES paths — xi>=1 NaNs the ES only,
                # the VaR stays finite (mirrors extraction). v2.1: empirical
                # tau_mass from the same ctxfits cache row (tail-mass ruling).
                r3[f"v{c}"], r3[f"e{c}"] = e3_var_es_from_fit(
                    grid, q50c, f["u"], f["xi"], f["beta"], a,
                    tau_mass=float(f["tau_mass"]))
            else:
                r3[f"v{c}"] = r3[f"e{c}"] = np.nan
            try:
                r2[f"v{c}"] = e2_quantile(grid, a, f["nu"])
                r2[f"e{c}"] = e2_es(grid, a, f["nu"])
            except (ValueError, KeyError):
                r2[f"v{c}"] = r2[f"e{c}"] = np.nan
        rows_e3.append(r3); rows_e2.append(r2)
        mu_a.append(mu_g[gmap[t]]); sig_a.append(sig_g[gmap[t]])
    e3 = pd.DataFrame(rows_e3).reset_index(drop=True)
    e2 = pd.DataFrame(rows_e2).reset_index(drop=True)
    return e3, e2, np.array(mu_a), np.array(sig_a)


def _battery_rows(name, asset, group, erule, y, vv, ee):
    """vv/ee are either the EVT multi-alpha dict {alpha: (v_arr, e_arr)} or the
    column dict {'v01': arr, ...} / {'e01': arr, ...}."""
    out = []
    for a in ALPHAS:
        c = acol(a)
        if a in vv:                          # EVT multi-alpha dict
            v = vv[a][0]; e = ee[a][1]
        else:                                # column dict
            v = vv[f"v{c}"]; e = ee[f"e{c}"]
        b = battery(np.asarray(y), np.asarray(v), np.asarray(e), a)
        out.append({"forecaster": name, "asset": asset, "alpha": a,
                    "erule": erule, "group": group, **b})
    return out


def main():
    mf = yaml.safe_load((S4 / "run_manifest.yaml").read_text())
    oos = mf["protocol"]["oos_start"]
    dm = yaml.safe_load((ROOT / "data" / "data_manifest.yaml").read_text())
    assets = mf["assets"]
    rows = []

    for model in TSFM:
        m_t0 = time.time()
        for asset in assets:
          try:                          # run-discipline: one bad cell logs+skips, never kills the grid
            fpq = FC / f"{model}_{asset}.parquet"
            if not fpq.exists():
                continue
            fits = load_ctxfits(S4 / "ctxfits" / f"{asset}.parquet")  # B2-h
            df = load_series(asset); x = df[df.attrs.get("column", "logret")].to_numpy()
            group = dm[asset]["group"]
            e3, e2, mu, sig = build_aligned(model, asset, fits, x, oos)
            y = e3["y"].to_numpy()
            # F1 (substitution, extraction-independent) + H, both multi-alpha
            f1 = arms.f1_multi(e3, ALPHAS)
            h = arms.h_multi(e3, sig, ALPHAS)
            rows += _battery_rows(f"{model}+F1", asset, group, "e3", y, f1, f1)
            rows += _battery_rows(f"{model}+H", asset, group, "e3", y, h, h)
            # F2/F3/F4 on BOTH E3 and E2 (produced together). These arms take a
            # single-alpha (v, e); build a per-alpha view with v/e set from the
            # extraction's alpha column.
            for ext, al in (("e3", e3), ("e2", e2)):
                yy = al["y"].to_numpy()
                s_hat = shat_first_window(al)     # v2.0 P0-2: first-500-day ŝ
                for arm in ("F2", "F3", "F4"):
                    cols_v, cols_e = {}, {}
                    for a in ALPHAS:
                        c = acol(a)
                        alp = al[["t", "y"]].copy()
                        alp["v"] = al[f"v{c}"].to_numpy(); alp["e"] = al[f"e{c}"].to_numpy()
                        if arm == "F2":
                            v, e = arms.f2_adaptive_conformal(alp, a, s_hat)
                        elif arm == "F3":
                            v, e = arms.f3_rescale(alp, a)
                        else:
                            v, e = arms.f4_additive_fz0(alp, a, s_hat)
                        cols_v[f"v{c}"] = v; cols_e[f"e{c}"] = e
                    rows += _battery_rows(f"{model}+{arm}", asset, group, ext, yy, cols_v, cols_e)
          except AssertionError:        # B2-d: guard asserts (anchor/mass/schema)
            raise                       # must fail the RUN, never a cell skip
          except Exception as ex:
            print(f"[repairs] SKIP {model}/{asset}: {type(ex).__name__}: {ex}", flush=True)
        el = (time.time() - m_t0) / 60
        print(f"[repairs] {model} done ({el:.1f} min, {len(rows)} rows)", flush=True)
        if el > 45:
            print(f"[repairs] WARN {model} {el:.0f}min > 45min guard", flush=True)

    # GARCH-EVT comparator: per asset (no model). Use any model's aligned y (same y).
    g_t0 = time.time()
    for asset in assets:
        fpq = FC / f"chronos_2_{asset}.parquet"
        if not fpq.exists():
            continue
        fits = load_ctxfits(S4 / "ctxfits" / f"{asset}.parquet")  # B2-h
        df = load_series(asset); x = df[df.attrs.get("column", "logret")].to_numpy()
        group = dm[asset]["group"]
        e3, _, mu, sig = build_aligned("chronos_2", asset, fits, x, oos)
        y = e3["y"].to_numpy()
        ge = arms.garch_evt_multi(e3, mu, sig, ALPHAS)
        rows += _battery_rows("garch_evt", asset, group, "param", y, ge, ge)
    print(f"[repairs] garch_evt done ({(time.time()-g_t0)/60:.1f} min)", flush=True)

    pd.DataFrame(rows).to_csv(S4 / "repair_backtests.csv", index=False)
    print(f"repair grid complete: {len(rows)} cells", flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("grid_repairs")
