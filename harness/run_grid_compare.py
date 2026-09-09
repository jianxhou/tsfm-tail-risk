"""Cross-forecaster comparison: Model Confidence Set + Diebold-Mariano per
(asset, alpha, E-rule) on FZ0 loss (proposal 3.4; conditions 3-5).

    python -m harness.run_grid_compare

For each E-rule, the compared set is {TSFMs under that E-rule} + {econometric
baselines} (baselines are E-rule-independent, so they appear under every rule —
the honest "TSFM-under-rule vs baseline" contrast). FZ0 loss series are aligned
on the common dates where ALL compared forecasters have a valid ES forecast
(e<v<0); MCS uses the tested backtests.mcs; per-forecaster MCS-membership rate is
aggregated. E1 is included here only as an exhibit and excluded from the headline
ranking downstream (condition 9).

Output: results/stage4/mcs_membership.csv (forecaster, alpha, erule, n_asset,
mcs_rate), results/stage4/dm_pairs.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtests.dm import dm_test
from backtests.mcs import mcs
from backtests.scores import fz0
from data.load import load_series
from erules.rules import (e1_es, e1_quantile, e2_es, e2_quantile,
                          e3_var_es_from_fit)
from harness.run_grid_extract import load_ctxfits

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
FC = S4 / "forecast"
BL = S4 / "baselines"
CTXFITS = S4 / "ctxfits"
CTX_LEN = 512
ALPHAS = (0.01, 0.025, 0.05)
DEEP_TAUS = {0.01: "q01", 0.025: "q025", 0.05: "q05", 0.1: "q1", 0.25: "q25",
             0.5: "q5", 0.75: "q75", 0.9: "q9"}
DECILE_TAUS = {round(0.1 * k, 1): "q" + f"{round(0.1*k,1):g}".replace("0.", "")
               for k in range(1, 10)}
TSFM_TAUS = {"chronos_bolt": DEEP_TAUS, "chronos_2": DEEP_TAUS,
             "timesfm_2_5": DECILE_TAUS, "moirai_2_0": DEEP_TAUS, "lag_llama": DEEP_TAUS}
BASELINES = ["garch_t", "gjr_t", "ewma94", "hs250", "hs500", "fhs"]  # ES-bearing


def tsfm_ve_series(model, asset, erule, alpha, fits, x):
    """Per-window (date, v, e) for a TSFM under an E-rule. None if unavailable."""
    pq = FC / f"{model}_{asset}.parquet"
    if not pq.exists():
        return None
    d = pd.read_parquet(pq)
    taus = TSFM_TAUS[model]
    # v2.1 P2-14: y collected per kept row (t-keyed with date/v/e), replacing
    # the positional head-slice d["y"][:len(dates)] that would misalign after
    # any interior skip. Output-identical on the frozen store (160/160 pairs,
    # zero interior skips, machine-checked 2026-08-03).
    dates, ys, vs, es = [], [], [], []
    for _, r in d.iterrows():
        t = int(r["t"])
        if t not in fits.index:
            continue
        grid = {tau: float(r[c]) for tau, c in taus.items()
                if c in r and np.isfinite(r[c])}
        f = fits.loc[t]
        ctx = x[t - CTX_LEN: t]
        try:
            if erule == "e0":
                v = grid.get(alpha, np.nan); e = np.nan
            elif erule == "e1":
                v, e = e1_quantile(grid, alpha), e1_es(grid, alpha)
            elif erule == "e2":
                v, e = e2_quantile(grid, alpha, f["nu"]), e2_es(grid, alpha, f["nu"])
            elif erule == "e3":
                if np.isfinite(f["xi"]) and np.isfinite(f["beta"]):
                    q50c = float(np.median(ctx))   # cached GPD params, no re-fit
                    # v2.0 P0-1: independent VaR/ES exception paths. ES-bearing
                    # outputs are unaffected: fz0_loss_frame drops rows without
                    # a valid ES either way. v2.1: empirical tau_mass from the
                    # same ctxfits cache row (docs/e3_tail_mass_ruling.md).
                    v, e = e3_var_es_from_fit(grid, q50c, f["u"], f["xi"],
                                              f["beta"], alpha,
                                              tau_mass=float(f["tau_mass"]))
                else:
                    v, e = np.nan, np.nan
        except (ValueError, KeyError):
            v, e = np.nan, np.nan
        dates.append(r["date"]); ys.append(float(r["y"])); vs.append(v); es.append(e)
    return pd.DataFrame({"date": dates, "y": ys, "v": vs, "e": es})


def baseline_ve_series(bl, asset, alpha):
    pq = BL / f"{bl}_{asset}.parquet"
    if not pq.exists():
        return None
    d = pd.read_parquet(pq)
    c = f"{alpha:g}".replace("0.", "")
    if f"v{c}" not in d:
        return None
    return pd.DataFrame({"date": d["date"], "y": d["y"], "v": d[f"v{c}"], "e": d[f"e{c}"]})


def fz0_loss_frame(series: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """date -> FZ0 loss, only rows with valid e<v<0."""
    s = series.dropna(subset=["v", "e"])
    good = (s["e"] < s["v"]) & (s["e"] < 0)
    s = s[good]
    if len(s) == 0:
        return pd.DataFrame(columns=["date", "loss"])
    loss = fz0(s["y"].to_numpy(), s["v"].to_numpy(), s["e"].to_numpy(), alpha)
    return pd.DataFrame({"date": s["date"].values, "loss": loss})


def main() -> None:
    dm_manifest = yaml.safe_load((ROOT / "data" / "data_manifest.yaml").read_text())
    assets = sorted(k for k, v in dm_manifest.items() if not v.get("quarantined"))
    mem_rows, dm_rows = [], []

    for erule in ("e1", "e2", "e3"):     # ES-bearing rules (E0 has no ES)
        for a in ALPHAS:
            memberships: dict[str, list] = {}
            for asset in assets:
                fpq = CTXFITS / f"{asset}.parquet"
                if not fpq.exists():
                    continue
                fits = load_ctxfits(fpq)          # B2-h: schema-guarded read
                df = load_series(asset)
                x = df[df.attrs.get("column", "logret")].to_numpy()
                losses = {}
                for m in TSFM_TAUS:
                    s = tsfm_ve_series(m, asset, erule, a, fits, x)
                    if s is not None:
                        fl = fz0_loss_frame(s, a)
                        if len(fl) > 50:
                            losses[m] = fl
                for bl in BASELINES:
                    s = baseline_ve_series(bl, asset, a)
                    if s is not None:
                        fl = fz0_loss_frame(s, a)
                        if len(fl) > 50:
                            losses[bl] = fl
                if len(losses) < 3:
                    continue
                # align on common dates
                common = None
                for fl in losses.values():
                    ds = set(fl["date"])
                    common = ds if common is None else (common & ds)
                common = sorted(common)
                if len(common) < 100:
                    continue
                names = list(losses)
                L = np.column_stack([
                    losses[n].set_index("date").loc[common, "loss"].to_numpy()
                    for n in names])
                res = mcs(L, names=names, level=0.10, n_boot=2000, seed=20260706)
                for n in names:
                    memberships.setdefault(n, []).append(n in res["mcs"])
            for n, hits in memberships.items():
                mem_rows.append({"forecaster": n, "alpha": a, "erule": erule,
                                 "n_asset": len(hits),
                                 "mcs_rate": float(np.mean(hits))})
            print(f"[compare] erule={erule} alpha={a:g} done", flush=True)

    pd.DataFrame(mem_rows).to_csv(S4 / "mcs_membership.csv", index=False)
    print(f"compare complete: {len(mem_rows)} membership rows", flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("grid_compare")
