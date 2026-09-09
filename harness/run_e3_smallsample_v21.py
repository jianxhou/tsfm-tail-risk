"""v2.1 Phase-1 small-sample validation of the E3 tail-mass redefinition.

    python -m harness.run_e3_smallsample_v21

SPX + dgs30, all 5 CORE heads, E3 at the three alphas, OLD definition
(tau_mass = 0.10 nominal, the v2.0 convention) vs NEW definition
(tau_mass = n_kept/W empirical, docs/e3_tail_mass_ruling.md), scored on the
production battery (harness.run_grid_extract.battery — same MC nulls, same
seed). Governance:
  * READS results/stage4/{ctxfits,forecast,backtests.csv}; WRITES ONLY into
    results/stage4/e3_diagnostics/ (the v2.1 new-output directory) — the
    v2.0 result files are never touched (Phase-1 iron rule).
  * The OLD leg is a cross-check against the verification agents' recompute
    caliber: window fits must reproduce the v2.0 ctxfits cache exactly and
    the OLD-leg battery must reproduce the shipped backtests.csv E3 rows
    exactly; any mismatch is a hard error.
  * Direction heterogeneity is the reporting target (review #10 §2.2 /
    A3-T4); pass rates are NOT a success criterion.

Outputs: e3_diagnostics/<asset>.csv (review §2.5 per-window log, via the
shared e3_window_fit helper) and e3_diagnostics/smallsample_v21.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from data.load import load_series
from erules.rules import e3_var_es_from_fit
from harness.rolling import rolling_windows
from harness.run_grid_extract import (ALPHAS, CTX_LEN, DIAG, FC, MANIFEST,
                                      MODEL_TAUS, OUT, acol, battery,
                                      e3_window_fit)

ASSETS = ("spx", "dgs30")
NOMINAL = 0.10


def window_fits(asset: str, oos: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_series(asset)
    rs = rolling_windows(df, value_col=df.attrs.get("column", "logret"),
                         ctx_len=CTX_LEN, oos_start=oos, allow_late_start=True)
    fit_rows, diag_rows = [], []
    for w in rs.windows:
        fr, dr = e3_window_fit(asset, w)
        fit_rows.append({"t": w.t, **fr})
        diag_rows.append(dr)
    return pd.DataFrame(fit_rows).set_index("t"), pd.DataFrame(diag_rows)


def crosscheck_ctxfits(asset: str, fits: pd.DataFrame) -> None:
    """Fresh window fits must equal the v2.0 cache bitwise (u, xi, beta) —
    the fit itself is unchanged by the redefinition."""
    old = pd.read_parquet(OUT / "ctxfits" / f"{asset}.parquet").set_index("t")
    assert len(old) == len(fits), f"{asset}: window count {len(fits)} != cache {len(old)}"
    for col in ("u", "xi", "beta"):
        a = fits[col].to_numpy()
        b = old.loc[fits.index, col].to_numpy()
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        assert same.all(), f"{asset}: {col} mismatch on {int((~same).sum())} windows"
    print(f"[xcheck] {asset}: fresh fits == v2.0 ctxfits cache "
          f"(u/xi/beta bitwise, {len(fits)} windows)")


def main() -> None:
    DIAG.mkdir(parents=True, exist_ok=True)
    mf = yaml.safe_load(MANIFEST.read_text())
    oos = mf["protocol"]["oos_start"]
    # round_trip parsing: the default C parser is lossy in the last ulps and
    # would fail an exact bitwise comparison against recomputed doubles
    shipped = pd.read_csv(OUT / "backtests.csv", float_precision="round_trip")
    shipped = shipped[shipped.erule == "e3"].set_index(
        ["asset", "forecaster", "alpha"])

    rows, n_repro, n_cells = [], 0, 0
    for asset in ASSETS:
        fits, diag = window_fits(asset, oos)
        diag.to_csv(DIAG / f"{asset}.csv", index=False)   # review §2.5 log
        crosscheck_ctxfits(asset, fits)
        df = load_series(asset)
        x = df[df.attrs.get("column", "logret")].to_numpy()
        for model, taus in MODEL_TAUS.items():
            fpq = FC / f"{model}_{asset}.parquet"
            if not fpq.exists():
                print(f"[skip] {model}_{asset}: no forecast file")
                continue
            d = pd.read_parquet(fpq)
            v_old = {a: [] for a in ALPHAS}; e_old = {a: [] for a in ALPHAS}
            v_new = {a: [] for a in ALPHAS}; e_new = {a: [] for a in ALPHAS}
            ys = []
            for _, r in d.iterrows():
                t = int(r["t"])
                grid = {tau: float(r[c]) for tau, c in taus.items()
                        if c in r and np.isfinite(r[c])}
                f = fits.loc[t]
                ys.append(r["y"])
                q50c = float(np.median(x[t - CTX_LEN:t]))
                ok = np.isfinite(f["xi"]) and np.isfinite(f["beta"])
                for a in ALPHAS:
                    if ok:
                        vo, eo = e3_var_es_from_fit(grid, q50c, f["u"], f["xi"],
                                                    f["beta"], a, tau_mass=NOMINAL)
                        vn, en = e3_var_es_from_fit(grid, q50c, f["u"], f["xi"],
                                                    f["beta"], a,
                                                    tau_mass=float(f["tau_mass"]))
                    else:
                        vo = eo = vn = en = np.nan
                    v_old[a].append(vo); e_old[a].append(eo)
                    v_new[a].append(vn); e_new[a].append(en)
            y = np.asarray(ys, dtype=float)
            for a in ALPHAS:
                bo = battery(y, np.asarray(v_old[a]), np.asarray(e_old[a]), a)
                bn = battery(y, np.asarray(v_new[a]), np.asarray(e_new[a]), a)
                n_cells += 1
                # OLD leg must reproduce the shipped backtests.csv row exactly
                s = shipped.loc[(asset, model, a)]
                for k in ("n", "hit_rate", "kupiec_stat", "kupiec_p_mc"):
                    sv, bv = float(s[k]), float(bo[k])
                    assert (sv == bv) or (np.isnan(sv) and np.isnan(bv)), (
                        f"OLD-leg mismatch {asset}/{model}/{a} {k}: "
                        f"recomputed {bv!r} != shipped {sv!r}")
                n_repro += 1
                dv = np.asarray(v_new[a]) - np.asarray(v_old[a])
                dv = dv[np.isfinite(dv)]
                rows.append({
                    "asset": asset, "model": model, "alpha": a,
                    "n": bo["n"],
                    "hit_old": bo["hit_rate"], "hit_new": bn["hit_rate"],
                    "uc_p_old": bo["kupiec_p_mc"], "uc_p_new": bn["kupiec_p_mc"],
                    "uc_pass_old": bo["kupiec_p_mc"] > 0.05,
                    "uc_pass_new": bn["kupiec_p_mc"] > 0.05,
                    "flip": (bo["kupiec_p_mc"] > 0.05) != (bn["kupiec_p_mc"] > 0.05),
                    "fz0_old": bo.get("fz0_mean", np.nan),
                    "fz0_new": bn.get("fz0_mean", np.nan),
                    "mean_var_shift": float(dv.mean()) if len(dv) else np.nan,
                    "share_shallower": float((dv > 0).mean()) if len(dv) else np.nan,
                })
        print(f"[smallsample] {asset} done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(DIAG / "smallsample_v21.csv", index=False)
    print(f"\nOLD-leg reproduction: {n_repro}/{n_cells} cells match shipped "
          f"backtests.csv exactly (n, hit_rate, kupiec_stat, kupiec_p_mc)")
    flips = out[out.flip]
    print(f"UC(MC,5%) flips under the new definition: {len(flips)}/{len(out)}")
    for r in flips.itertuples():
        print(f"  {r.asset}/{r.model}/a={r.alpha:g}: hit {r.hit_old:.4f}->"
              f"{r.hit_new:.4f}, pass {r.uc_pass_old}->{r.uc_pass_new}")
    print("\nfull table -> results/stage4/e3_diagnostics/smallsample_v21.csv")


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("e3_smallsample")
