"""Pilot D4 driver: econometric baselines + the full backtest battery per cell.

    nohup python -m harness.run_pilot_d4 > results/pilot_d4/run.log 2>&1 &

Baselines (proposal §9): GARCH(1,1)-t, FHS, HS-500 — fit_len=1000, refit_every=21
(A1 main setting), VaR+ES at alpha in {1%, 2.5%, 5%}, all through harness.rolling.
Battery per cell (forecaster, asset, alpha[, erule]): Kupiec, Christoffersen CC,
DQ(4), mean QS, mean FZ0 (where ES exists and e<v<0), AS-Z2, n_alpha, sigma_tail,
precision floor. Output: results/pilot_d4/backtests.csv (one row per cell).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from backtests.es_tests import as_z2
from backtests.scores import fz0, quantile_score
from backtests.var_tests import christoffersen_cc, dq_test, kupiec
from data.load import load_series
from harness.rolling import align_forecasts, refit_schedule, rolling_windows
from precision.diagnostics import n_alpha, precision_floor, sigma_tail

OUT = Path(__file__).resolve().parent.parent / "results" / "pilot_d4"
D3 = OUT.parent / "pilot_d3"
ASSETS = ["spx", "btc", "nvda"]
ALPHAS = (0.01, 0.025, 0.05)
FIT_LEN, REFIT = 1000, 21
OOS_START = "2018-01-01"
MODELS = ["chronos_bolt", "chronos_2", "timesfm_2_5", "moirai_2_0"]
ERULES = ["e0", "e1", "e2", "e3"]


def acol(a: float) -> str:
    return f"{a:g}".replace("0.", "")


def run_baselines(asset: str) -> pd.DataFrame:
    pq = OUT / f"baseline_{asset}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    from arch import arch_model
    warnings.filterwarnings("ignore")
    df = load_series(asset)
    rs = rolling_windows(df, value_col="logret", ctx_len=FIT_LEN,
                         oos_start=OOS_START)
    plan = refit_schedule(len(rs), REFIT, FIT_LEN)
    params = None
    fcs = []
    for i, w in enumerate(rs.windows):
        if plan.mask[i] or params is None:
            params = arch_model(w.ctx, vol="GARCH", p=1, q=1, dist="t",
                                mean="Constant", rescale=False
                                ).fit(disp="off", show_warning=False).params
        fixed = arch_model(w.ctx, vol="GARCH", p=1, q=1, dist="t",
                           mean="Constant", rescale=False).fix(params)
        sigma = float(np.sqrt(fixed.forecast(horizon=1, reindex=False)
                              .variance.values[-1, 0]))
        mu, nu = float(params["mu"]), float(params["nu"])
        z = np.asarray(fixed.resid) / np.asarray(fixed.conditional_volatility)
        z = z[np.isfinite(z)]
        hs = w.ctx[-500:]
        row = {}
        for a in ALPHAS:
            c = acol(a)
            tq = stats.t.ppf(a, nu)
            scale = np.sqrt((nu - 2.0) / nu)
            row[f"garch_t_v{c}"] = mu + sigma * tq * scale
            row[f"garch_t_s{c}"] = mu + sigma * scale * (
                -stats.t.pdf(tq, nu) * (nu + tq ** 2) / ((nu - 1.0) * a))
            zq = float(np.quantile(z, a))
            row[f"fhs_v{c}"] = mu + sigma * zq
            row[f"fhs_s{c}"] = mu + sigma * float(z[z <= zq].mean())
            hq = float(np.quantile(hs, a))
            row[f"hs500_v{c}"] = hq
            row[f"hs500_s{c}"] = float(hs[hs <= hq].mean())
        fcs.append(row)
    frame = align_forecasts(rs, fcs)
    frame.to_parquet(pq, index=False)
    print(f"[baseline] {asset}: {len(frame)} windows", flush=True)
    return frame


def battery(y: np.ndarray, v: np.ndarray, e: np.ndarray | None,
            alpha: float) -> dict:
    ok = np.isfinite(v)
    y, v = y[ok], v[ok]
    e = e[ok] if e is not None else None
    out = {"n": len(y), "n_alpha": n_alpha(len(y), alpha)}
    uc = kupiec(y, v, alpha)
    out.update(hit_rate=uc["hit_rate"], kupiec_p=uc["p"],
               cc_p=christoffersen_cc(y, v, alpha)["p"],
               dq_p=dq_test(y, v, alpha)["p"],
               qs_mean=float(quantile_score(y, v, alpha).mean()))
    st = sigma_tail(y, v)
    out["sigma_tail"] = st
    out["prec_floor"] = precision_floor(st, out["n_alpha"]) if np.isfinite(st) else np.nan
    if e is not None:
        good = (e < v) & (e < 0) & np.isfinite(e)
        if good.mean() > 0.99:
            out["fz0_mean"] = float(fz0(y[good], v[good], e[good], alpha).mean())
            out["z2"] = as_z2(y[good], v[good], e[good], alpha)["stat"]
        else:
            out["fz0_mean"] = np.nan
            out["z2"] = np.nan
            out["es_convention_violations"] = int((~good).sum())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for asset in ASSETS:
        base = run_baselines(asset)
        for bl in ("garch_t", "fhs", "hs500"):
            for a in ALPHAS:
                c = acol(a)
                r = battery(base["y"].to_numpy(), base[f"{bl}_v{c}"].to_numpy(),
                            base[f"{bl}_s{c}"].to_numpy(), a)
                rows.append({"forecaster": bl, "asset": asset, "alpha": a,
                             "erule": "param", **r})
        for m in MODELS:
            ex = pd.read_parquet(D3 / f"extract_{m}_{asset}.parquet")
            for er in ERULES:
                for a in ALPHAS:
                    c = acol(a)
                    vcol = f"{er}_v{c}"
                    if vcol not in ex or ex[vcol].isna().all():
                        continue
                    e = ex.get(f"{er}_s{c}")
                    r = battery(ex["y"].to_numpy(), ex[vcol].to_numpy(),
                                e.to_numpy() if e is not None else None, a)
                    rows.append({"forecaster": m, "asset": asset, "alpha": a,
                                 "erule": er, **r})
        print(f"[battery] {asset} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "backtests.csv", index=False)
    write_summary(df)
    print(f"D4 complete: {len(rows)} cells", flush=True)


def write_summary(df: pd.DataFrame) -> None:
    """Tracked decision-evidence tables in docs/ (script-generated by protocol)."""
    lines = ["# D4 backtest summary (script-generated; do not hand-edit)", "",
             "Full battery in results/pilot_d4/backtests.csv (gitignored, "
             "regenerable via `python -m harness.run_pilot_d4`).", "",
             "## GO#1 — baseline coverage, SPX (nominal in header)", "",
             "| forecaster | hit@1% | Kupiec p | hit@2.5% | p | hit@5% | p |",
             "|---|---|---|---|---|---|---|"]
    for bl in ("garch_t", "fhs", "hs500"):
        c = {a: df[(df.forecaster == bl) & (df.asset == "spx") & (df.alpha == a)]
             for a in ALPHAS}
        cells = []
        for a in ALPHAS:
            r = c[a].iloc[0]
            cells += [f"{100*r.hit_rate:.2f}%", f"{r.kupiec_p:.3f}"]
        lines.append(f"| {bl} | " + " | ".join(cells) + " |")
    lines += ["", "## GO#2 — head-type signal: E0 (native) vs E2, alpha=1%, all assets", "",
              "| model | asset | E0 hit | E0 Kupiec p | E2 hit | E2 Kupiec p | E0-E2 spread |",
              "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for asset in ASSETS:
            e0 = df[(df.forecaster == m) & (df.asset == asset) & (df.alpha == 0.01) & (df.erule == "e0")]
            e2 = df[(df.forecaster == m) & (df.asset == asset) & (df.alpha == 0.01) & (df.erule == "e2")]
            if len(e2) == 0:
                continue
            e2r = e2.iloc[0]
            if len(e0):
                e0r = e0.iloc[0]
                spread = f"{100*abs(e0r.hit_rate - e2r.hit_rate):.1f}pp"
                e0cells = f"{100*e0r.hit_rate:.2f}% | {e0r.kupiec_p:.2e}"
            else:
                spread = "— (no E0)"
                e0cells = "— | —"
            lines.append(f"| {m} | {asset} | {e0cells} | {100*e2r.hit_rate:.2f}% "
                         f"| {e2r.kupiec_p:.3f} | {spread} |")
    lines += ["", "Reading: large E0-E2 spread on a bounded-grid head (chronos_bolt) "
              "vs small spread on distinguishable heads is the C2 mechanism signal; "
              "the E0 column is each head's own native deep-tail rule.", ""]
    (Path(__file__).resolve().parent.parent / "docs" / "pilot_d4_summary.md"
     ).write_text("\n".join(lines))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {(time.time()-t0)/60:.1f} min", flush=True)
