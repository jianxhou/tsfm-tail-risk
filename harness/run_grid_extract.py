"""Stage-4 extraction + MC-calibrated backtest battery on the grid forecasts.

    python -m harness.run_grid_extract [--assets a,b,c]

Reads results/stage4/forecast/*.parquet (native grids) -> E0-E3 VaR+ES ->
MC-calibrated battery (gate condition 2). Per-asset context fits (nu, GPD at
the fixed anchor tau_anchor=0.10 with per-window empirical extrapolation mass
tau_mass = n_kept/W — docs/e3_tail_mass_ruling.md) computed once and shared
across models; every E3 fit writes a per-window diagnostics row to
results/stage4/e3_diagnostics/<asset>.csv (review #10 §2.5). Output:
results/stage4/backtests.csv, one row per (forecaster, asset, alpha, erule);
p-values are MC-calibrated where the statistic supports it, with the
asymptotic value kept alongside for audit. Governance: only statistics marked
`enabled` in the run_manifest are computed; esr/gas stay absent until their
conditions are met (CHANGELOG).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtests.es_tests import as_z2
from backtests.scores import fz0, quantile_score
from backtests.var_tests import christoffersen_cc, dq_test, kupiec
from data.load import load_series
from erules.rules import (e1_es, e1_quantile, e2_es, e2_quantile,
                          e3_var_es_from_fit, fit_gpd_diag, fit_nu, mad)
from harness.rolling import rolling_windows
from precision.diagnostics import n_alpha, precision_floor, sigma_tail
from precision.mc_critical import MCNull

ROOT = Path(__file__).resolve().parent.parent
FC = ROOT / "results" / "stage4" / "forecast"
OUT = ROOT / "results" / "stage4"
MANIFEST = OUT / "run_manifest.yaml"
DIAG = OUT / "e3_diagnostics"
CTX_LEN = 512
TAU_ANCHOR = 0.10     # E3 splice anchor: grid lookup + ctx quantile (fixed);
                      # the extrapolation mass is the per-window empirical
                      # tau_mass = n_kept/W (docs/e3_tail_mass_ruling.md)
CTXFIT_COLS = ("t", "nu", "u", "xi", "beta", "n_kept", "tau_mass")
ALPHAS = (0.01, 0.025, 0.05)
DEEP_TAUS = {0.01: "q01", 0.025: "q025", 0.05: "q05", 0.1: "q1", 0.25: "q25",
             0.5: "q5", 0.75: "q75", 0.9: "q9"}
DECILE_TAUS = {round(0.1 * k, 1): "q" + f"{round(0.1*k,1):g}".replace("0.", "")
               for k in range(1, 10)}
MODEL_TAUS = {"chronos_bolt": DEEP_TAUS, "chronos_2": DEEP_TAUS,
              "timesfm_2_5": DECILE_TAUS, "moirai_2_0": DEEP_TAUS,
              "lag_llama": DEEP_TAUS}   # headline head (condition 4); native analytic
                                        # Student-t on the DEEP grid
ERULES = ["e0", "e1", "e2", "e3"]
_MC_CACHE: dict = {}


def acol(a: float) -> str:
    return f"{a:g}".replace("0.", "")


NU_REFIT_EVERY = 21   # CHANGELOG 2026-07-06: nu MLE every 21 windows (aligned with
                      # the econometric refit; nu is slow-varying). Invariance-checked:
                      # induced E2 deep-tail delta << the E2/E3 spread at every quantile.
                      # GPD stays per-window (E3 is a headline rule, condition 4/9).


def load_ctxfits(path) -> pd.DataFrame:
    """Read an EXISTING ctxfits parquet (no recompute), t-indexed, with the
    v2.1 schema guard (B2-h, Phase-1 re-check fixup): every driver that reads
    the cache directly must hit the same stale-schema hard failure as
    ctx_fits — a v2.0 cache (no tau_mass) must never silently feed the
    empirical-mass E3 definition."""
    fits = pd.read_parquet(path)
    assert set(CTXFIT_COLS).issubset(fits.columns), (
        f"stale v2.0 ctxfits schema at {path} (no tau_mass) — delete "
        f"results/stage4/ctxfits/ and regenerate (review #10 §2.7 step 1)")
    return fits.set_index("t")


def ctx_fits(asset: str, oos_start: str) -> pd.DataFrame:
    cache = OUT / "ctxfits" / f"{asset}.parquet"
    if cache.exists():
        cached = pd.read_parquet(cache)
        if not set(CTXFIT_COLS).issubset(cached.columns):
            raise RuntimeError(
                f"stale ctxfits cache for {asset} (pre-v2.1 schema, no tau_mass): "
                f"a v2.0 cache must not silently feed the empirical-mass E3 "
                f"definition — delete results/stage4/ctxfits/ first "
                f"(review #10 §2.7 step 1) and rerun")
        return cached
    cache.parent.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    df = load_series(asset)
    rs = rolling_windows(df, value_col=df.attrs.get("column", "logret"),
                         ctx_len=CTX_LEN, oos_start=oos_start, allow_late_start=True)
    rows, diag_rows = [], []
    nu_hold = None
    for i, w in enumerate(rs.windows):
        if i % NU_REFIT_EVERY == 0 or nu_hold is None:
            nu_hold = fit_nu(w.ctx)          # re-estimated on this window, held for 21
        fit_row, diag_row = e3_window_fit(asset, w)
        rows.append({"t": w.t, "nu": nu_hold, **fit_row})
        diag_rows.append(diag_row)
    out = pd.DataFrame(rows)
    out.to_parquet(cache, index=False)
    pd.DataFrame(diag_rows).to_csv(DIAG / f"{asset}.csv", index=False)
    return out


def e3_window_fit(asset: str, w) -> tuple[dict, dict]:
    """One window's E3 threshold + kept-population GPD fit under the v2.1
    fixed-anchor empirical-mass convention (docs/e3_tail_mass_ruling.md).
    Returns (cache_row, diagnostics_row); the diagnostics row carries the full
    review #10 §2.5 field set. Shared by ctx_fits and the v2.1 validation
    drivers so the production and validation paths cannot diverge."""
    ctx = w.ctx
    wlen = len(ctx)                          # == CTX_LEN by rolling.py I1
    u = float(np.quantile(ctx, TAU_ANCHOR))
    n_lt_u = int((ctx < u).sum())
    n_eq_u = int((ctx == u).sum())
    dg = fit_gpd_diag(u - ctx[ctx < u], scale_ref=mad(ctx))
    n_kept = dg["n_kept"]
    tau_mass = float(n_kept) / wlen if np.isfinite(n_kept) else np.nan
    fit_row = {"u": u, "xi": dg["xi"], "beta": dg["beta"],
               "n_kept": n_kept, "tau_mass": tau_mass}
    diag_row = {                             # review #10 §2.5 per-window log
        "asset": asset, "t": w.t, "date": str(w.date.date()),
        "ctx_start": w.t - wlen, "ctx_end": w.t - 1, "W": wlen,
        "u": u, "quantile_method": "linear",
        "n_lt_u": n_lt_u, "n_eq_u": n_eq_u, "n_le_u": n_lt_u + n_eq_u,
        "n_raw_exc": dg["n_raw_exc"], "tie_count": dg["tie_count"],
        "n_kept": n_kept,
        "tau_nominal": TAU_ANCHOR, "tau_strict": n_lt_u / wlen,
        "tau_fit": tau_mass,
        "scale_ref": dg["scale_ref"], "beta_floor": dg["beta_floor"],
        "fit_branch": dg["fit_branch"], "xi": dg["xi"], "beta": dg["beta"],
        "status": dg["status"],
        "var_defined": dg["status"] == "ok",
        "es_defined": dg["status"] == "ok" and dg["xi"] < 1.0,
        # arm inclusion/exclusion reasons are a repair-layer concept; the
        # extraction layer records its own exclusion reason here and the
        # repair diagnostics land with the Phase-2 rerun of the arms
        "layer": "extraction",
        "exclusion_reason": "" if dg["status"] == "ok" else dg["status"],
    }
    return fit_row, diag_row


def mc_p(statistic: str, hits_stat: float, n: int, alpha: float) -> float:
    key = (statistic, n, round(alpha, 4))
    if key not in _MC_CACHE:
        _MC_CACHE[key] = MCNull(statistic, n, alpha, B=10000, seed=20260706)
    return _MC_CACHE[key].p_value(hits_stat)


def battery(y, v, e, alpha) -> dict:
    ok = np.isfinite(v)
    y, v = y[ok], v[ok]
    e = e[ok] if e is not None else None
    n = len(y)
    uc = kupiec(y, v, alpha)
    cc = christoffersen_cc(y, v, alpha)
    dq = dq_test(y, v, alpha)
    st = sigma_tail(y, v)
    out = {"n": n, "n_alpha": n_alpha(n, alpha), "hit_rate": uc["hit_rate"],
           "kupiec_stat": uc["stat"], "kupiec_p_asym": uc["p"],
           "kupiec_p_mc": mc_p("kupiec", uc["stat"], n, alpha),
           "cc_stat": cc["stat"], "cc_p_asym": cc["p"],
           "cc_p_mc": mc_p("cc", cc["stat"], n, alpha),
           "dq_p_asym": dq["p"], "qs_mean": float(quantile_score(y, v, alpha).mean()),
           "sigma_tail": st,
           "prec_floor": precision_floor(st, n_alpha(n, alpha)) if np.isfinite(st) else np.nan}
    if e is not None:
        good = (e < v) & (e < 0) & np.isfinite(e)
        if good.mean() > 0.99:
            out["fz0_mean"] = float(fz0(y[good], v[good], e[good], alpha).mean())
            out["z2"] = as_z2(y[good], v[good], e[good], alpha)["stat"]
        else:
            out["fz0_mean"] = np.nan
            out["z2"] = np.nan
            out["es_viol"] = int((~good).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="")
    cli = ap.parse_args()
    mf = yaml.safe_load(MANIFEST.read_text())
    oos = mf["protocol"]["oos_start"]
    # asset universe from the manifest (robust to model names with underscores);
    # keep those with at least one quantile-head forecast file
    done = [a for a in mf["assets"]
            if any((FC / f"{m}_{a}.parquet").exists() for m in MODEL_TAUS)]
    assets = cli.assets.split(",") if cli.assets else done
    assets = [a for a in assets if a in done]

    rows = []
    for asset in assets:
        fits = ctx_fits(asset, oos).set_index("t")
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
                ctx = x[t - CTX_LEN: t]
                # E3 uses the CACHED GPD params (u, xi, beta) from ctx_fits; q50c is
                # a cheap median (no fit). Identical to e3_quantile's in-loop re-fit.
                q50c = float(np.median(ctx))
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
                        # v2.1: per-window empirical mass from the same cache row.
                        rec[f"e3_v{c}"], rec[f"e3_s{c}"] = e3_var_es_from_fit(
                            grid, q50c, f["u"], f["xi"], f["beta"], a,
                            tau_mass=float(f["tau_mass"]))
                    else:
                        rec[f"e3_v{c}"] = np.nan
                        rec[f"e3_s{c}"] = np.nan
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
    out = pd.DataFrame(rows)
    # review #10 §2.4 trap 1: a wiring error must not silently drop a whole
    # E-rule layer — the output row set must cover every (erule, alpha) cell.
    expected = {(er, a) for er in ERULES for a in ALPHAS}
    got = set(zip(out["erule"], out["alpha"]))
    assert got == expected, (
        f"E-rule layer(s) missing from the output row set: "
        f"{sorted(expected - got)} — whole-rule silent loss")
    out.to_csv(OUT / "backtests.csv", index=False)
    print(f"grid extract complete: {len(rows)} cells", flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("grid_extract")
