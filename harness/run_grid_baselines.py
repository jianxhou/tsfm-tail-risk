"""Full-grid econometric baseline forecasts (checklist item 7).

    nohup python -m harness.run_grid_baselines > results/stage4/baselines/run.log 2>&1 &

GARCH(1,1)-t, GJR-GARCH-t (arch pkg), EWMA(0.94), HS250, HS500, FHS, CAViaR-SAV
(harness.baselines, reference-tested) on all 32 grid-eligible assets. fit_len=1000,
refit_every=21 (A1). One parquet per (baseline, asset) with date,y,v/e at 3 alphas
(CAViaR is VaR-only). All slicing via harness.rolling. Resumable per (baseline, asset).
CAViaR convergence diagnostics (multi-start spread) logged per asset.
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from data.load import QuarantinedSeriesError, load_series
from harness.baselines import (CaviarFitError, PATH_CAP_MULT, _caviar_sav_path,
                              ewma_var, fit_caviar_sav,
                              filtered_historical_simulation, historical_simulation)
from harness.rolling import align_forecasts, audit_no_lookahead, refit_schedule, rolling_windows

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "stage4" / "run_manifest.yaml"
OUT = ROOT / "results" / "stage4" / "baselines"
ALPHAS = (0.01, 0.025, 0.05)


def acol(a):
    return f"{a:g}".replace("0.", "")


def garch_family(df, ctx_col, oos, fit_len, refit, gjr: bool):
    from arch import arch_model
    warnings.filterwarnings("ignore")
    rs = rolling_windows(df, value_col=ctx_col, ctx_len=fit_len, oos_start=oos,
                         allow_late_start=True)
    plan = refit_schedule(len(rs), refit, fit_len)
    o = 1 if gjr else 0
    params = None
    fcs = []
    for i, w in enumerate(rs.windows):
        if plan.mask[i] or params is None:
            params = arch_model(w.ctx, vol="GARCH", p=1, o=o, q=1, dist="t",
                                mean="Constant", rescale=False
                                ).fit(disp="off", show_warning=False).params
        fixed = arch_model(w.ctx, vol="GARCH", p=1, o=o, q=1, dist="t",
                           mean="Constant", rescale=False).fix(params)
        sigma = float(np.sqrt(fixed.forecast(horizon=1, reindex=False).variance.values[-1, 0]))
        mu, nu = float(params["mu"]), float(params["nu"])
        sc = np.sqrt((nu - 2.0) / nu)
        row = {}
        for a in ALPHAS:
            tq = stats.t.ppf(a, nu)
            row[f"v{acol(a)}"] = mu + sigma * tq * sc
            row[f"e{acol(a)}"] = mu + sigma * sc * (
                -stats.t.pdf(tq, nu) * (nu + tq**2) / ((nu - 1.0) * a))
        fcs.append(row)
    return rs, fcs


def nonparam(df, ctx_col, oos, fit_len, kind, refit=21):
    rs = rolling_windows(df, value_col=ctx_col, ctx_len=fit_len, oos_start=oos,
                         allow_late_start=True)
    fcs = []
    diag = {"caviar_loss_spread": [], "caviar_converged": [], "caviar_valid": [],
            "caviar_failed_fits": {a: 0 for a in ALPHAS},
            "caviar_invalid_days": {a: 0 for a in ALPHAS}}
    # CAViaR: refit params every `refit` windows (Nelder-Mead is expensive); between
    # refits roll the recursion forward with fixed beta (cheap). Per-alpha state.
    # v2.0 P0-3: a failed (all-starts-invalid) refit NaNs the window, drops the
    # stale beta and retries on the NEXT window (beta None triggers a refit);
    # per-day outputs are guarded (finite, negative, |v| within the path cap).
    caviar_beta = {a: None for a in ALPHAS}
    for i, w in enumerate(rs.windows):
        row = {}
        for a in ALPHAS:
            c = acol(a)
            if kind == "hs250":
                v, e = historical_simulation(w.ctx[-250:], a)
            elif kind == "hs500":
                v, e = historical_simulation(w.ctx[-500:], a)
            elif kind == "ewma94":
                v, e = ewma_var(w.ctx, a)
            elif kind == "fhs":
                v, e = filtered_historical_simulation(w.ctx, a)
            elif kind == "caviar_sav":
                if i % refit == 0 or caviar_beta[a] is None:
                    try:
                        fit = fit_caviar_sav(w.ctx, a, n_starts=8)
                        caviar_beta[a] = fit.beta
                        if a == 0.01:
                            diag["caviar_loss_spread"].append(fit.loss_spread)
                            diag["caviar_converged"].append(fit.converged_starts)
                            diag["caviar_valid"].append(fit.valid_starts)
                    except CaviarFitError:
                        caviar_beta[a] = None
                        diag["caviar_failed_fits"][a] += 1
                if caviar_beta[a] is None:
                    v = np.nan
                else:
                    beta = caviar_beta[a]
                    v1 = float(np.quantile(w.ctx[:min(300, len(w.ctx))], a))
                    path = _caviar_sav_path(beta, w.ctx, v1)
                    b0, b1, b2 = beta
                    v = float(b0 + b1 * path[-1] + b2 * abs(w.ctx[-1]))
                    cap = PATH_CAP_MULT * abs(np.quantile(w.ctx, a))
                    if not np.isfinite(v) or v >= 0 or abs(v) > cap:
                        diag["caviar_invalid_days"][a] += 1
                        v = np.nan
                e = np.nan
            row[f"v{c}"] = v
            row[f"e{c}"] = e
        fcs.append(row)
    return rs, fcs, diag


def main() -> None:
    mf = yaml.safe_load(MANIFEST.read_text())
    proto = mf["protocol"]
    assets = mf["assets"]
    fit_len, refit, oos = proto["fit_len"], proto["refit_every"], proto["oos_start"]
    OUT.mkdir(parents=True, exist_ok=True)
    baselines = ["garch_t", "gjr_t", "ewma94", "hs250", "hs500", "fhs", "caviar_sav"]

    for bl in baselines:
        for asset in assets:
            pq = OUT / f"{bl}_{asset}.parquet"
            if pq.exists():
                print(f"[skip] {bl}/{asset}", flush=True)
                continue
            try:
                df = load_series(asset)
            except QuarantinedSeriesError as e:
                print(f"[quarantine] {asset}", flush=True)
                continue
            ctx_col = df.attrs.get("column", "logret")
            t0 = time.time()
            diag = {}
            if bl == "garch_t":
                rs, fcs = garch_family(df, ctx_col, oos, fit_len, refit, gjr=False)
            elif bl == "gjr_t":
                rs, fcs = garch_family(df, ctx_col, oos, fit_len, refit, gjr=True)
            else:
                rs, fcs, diag = nonparam(df, ctx_col, oos, fit_len, bl)
            frame = align_forecasts(rs, fcs)
            audit_no_lookahead(df, ctx_col, rs)
            dt = time.time() - t0
            frame.to_parquet(pq, index=False)
            meta = {"baseline": bl, "asset": asset, "n_windows": rs.meta["n_windows"],
                    "runtime_s": round(dt, 1),
                    "actual_oos_start": rs.meta["actual_oos_start"],
                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if diag.get("caviar_loss_spread"):
                sp = np.array(diag["caviar_loss_spread"])
                meta["caviar_loss_spread_median"] = float(np.median(sp))
                meta["caviar_loss_spread_p95"] = float(np.percentile(sp, 95))
                meta["caviar_converged_min"] = int(min(diag["caviar_converged"]))
            if bl == "caviar_sav":               # v2.0 P0-3 diagnostics
                meta["caviar_valid_starts_min"] = (int(min(diag["caviar_valid"]))
                                                   if diag["caviar_valid"] else 0)
                meta["caviar_failed_fits"] = {str(a): n for a, n
                                              in diag["caviar_failed_fits"].items()}
                meta["caviar_invalid_days"] = {str(a): n for a, n
                                               in diag["caviar_invalid_days"].items()}
            (OUT / f"{bl}_{asset}.meta.json").write_text(json.dumps(meta, indent=1))
            extra = (f" caviar_spread~{meta.get('caviar_loss_spread_median',0):.2e}"
                     if bl == "caviar_sav" else "")
            print(f"[done] {bl}/{asset}: {rs.meta['n_windows']} win {dt/60:.1f}min{extra}",
                  flush=True)
    print("baseline grid complete", flush=True)


if __name__ == "__main__":
    main()
