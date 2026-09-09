"""Dual-level Model Confidence Sets (10% + 25%) from the persisted stage-4
loss inputs — v2.0 Phase-3 item 1g (registered levels: proposal §3.4
"Hansen MCS(10%/25%)"; only the 10% layer was delivered).

    python -m harness.run_mcs_levels

Both layers are recomputed from the EXISTING per-day artifacts (forecast/,
ctxfits/, baselines/, repairs/) — zero model forwards. The 10% layer is a
reproduction gate: its membership output must match the delivered
mcs_membership.csv / repair_mcs.csv cell-for-cell, proving the loss-frame
reconstruction is exact before the 25% layer is read.

Stage-4 grid: replicates run_grid_compare.main() (same skip guards, same
mcs(level, n_boot=2000, seed=20260706)), with the per-(model, asset) quantile
frame read once and all (rule, alpha) VaR/ES computed in a single row pass —
results-identical (the reproduction gate is the proof).
Repair battery: replicates run_repair_series.matched_rows()'s MCS block
(BURN=500, delivered-arm common-day mask, mcs(level, seed=20260704)) from the
persisted repairs/ per-day caches.

Output: results/stage4/mcs_membership_levels.csv (forecaster, alpha, erule,
level, n_asset, mcs_rate) and results/stage4/repair_mcs_levels.csv (model,
asset, alpha, arm, level, in_mcs, mcs_p, n).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtests.mcs import mcs
from data.load import load_series
from erules.rules import (e1_es, e1_quantile, e2_es, e2_quantile,
                          e3_var_es_from_fit)
from harness.run_grid_extract import load_ctxfits
from harness.run_grid_compare import (BASELINES, CTX_LEN, TSFM_TAUS,
                                      baseline_ve_series, fz0_loss_frame)

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
FC = S4 / "forecast"
REP = S4 / "repairs"
BL = S4 / "baselines"
CTXFITS = S4 / "ctxfits"
ALPHAS = (0.01, 0.025, 0.05)
LEVELS = (0.10, 0.25)
ERULES = ("e1", "e2", "e3")
BURN = 500                      # run_repair_series.py design §A
MCS_SET = ["F1", "H", "F2", "F3", "F4"]
ARM_PRE = {"F1": "f1", "H": "h", "F2": "f2g10_e3", "F3": "f3_e3", "F4": "f4_e3"}


def acol(a: float) -> str:
    return f"{a:g}".replace("0.", "")


def tsfm_ve_all(model: str, asset: str, fits: pd.DataFrame, x: np.ndarray):
    """All (erule, alpha) -> DataFrame(date, y, v, e) in ONE row pass.

    Row-for-row identical to run_grid_compare.tsfm_ve_series (same skip rule,
    same per-row t-keyed y collection — v2.1 P2-14, mirrored here); hoisted
    only to avoid 9 redundant parquet reads and row loops per (model, asset)."""
    pq = FC / f"{model}_{asset}.parquet"
    if not pq.exists():
        return None
    d = pd.read_parquet(pq)
    taus = TSFM_TAUS[model]
    dates, ys = [], []
    acc = {(er, a): ([], []) for er in ERULES for a in ALPHAS}
    for _, r in d.iterrows():
        t = int(r["t"])
        if t not in fits.index:
            continue
        grid = {tau: float(r[c]) for tau, c in taus.items()
                if c in r and np.isfinite(r[c])}
        f = fits.loc[t]
        ctx = x[t - CTX_LEN: t]
        q50c = float(np.median(ctx))
        for a in ALPHAS:
            for er in ERULES:
                try:
                    if er == "e1":
                        v, e = e1_quantile(grid, a), e1_es(grid, a)
                    elif er == "e2":
                        v, e = e2_quantile(grid, a, f["nu"]), e2_es(grid, a, f["nu"])
                    else:
                        if np.isfinite(f["xi"]) and np.isfinite(f["beta"]):
                            # v2.1: empirical tau_mass from the ctxfits row
                            v, e = e3_var_es_from_fit(grid, q50c, f["u"], f["xi"],
                                                      f["beta"], a,
                                                      tau_mass=float(f["tau_mass"]))
                        else:
                            v, e = np.nan, np.nan
                except (ValueError, KeyError):
                    v, e = np.nan, np.nan
                acc[(er, a)][0].append(v)
                acc[(er, a)][1].append(e)
        dates.append(r["date"]); ys.append(float(r["y"]))
    y = ys
    return {(er, a): pd.DataFrame({"date": dates, "y": y,
                                   "v": acc[(er, a)][0], "e": acc[(er, a)][1]})
            for er in ERULES for a in ALPHAS}


def stage4_levels() -> pd.DataFrame:
    manifest = yaml.safe_load((ROOT / "data" / "data_manifest.yaml").read_text())
    assets = sorted(k for k, v in manifest.items() if not v.get("quarantined"))
    # memberships[(erule, alpha, level)][forecaster] -> list of bools over assets
    memberships: dict = {(er, a, lv): {} for er in ERULES for a in ALPHAS
                         for lv in LEVELS}
    t0 = time.time()
    for i, asset in enumerate(assets):
        fpq = CTXFITS / f"{asset}.parquet"
        if not fpq.exists():
            continue
        fits = load_ctxfits(fpq)          # B2-h: schema-guarded read
        df = load_series(asset)
        x = df[df.attrs.get("column", "logret")].to_numpy()
        tsfm = {m: tsfm_ve_all(m, asset, fits, x) for m in TSFM_TAUS}
        for er in ERULES:
            for a in ALPHAS:
                losses = {}
                for m, frames in tsfm.items():
                    if frames is None:
                        continue
                    fl = fz0_loss_frame(frames[(er, a)], a)
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
                for lv in LEVELS:
                    res = mcs(L, names=names, level=lv, n_boot=2000,
                              seed=20260706)
                    for n in names:
                        memberships[(er, a, lv)].setdefault(n, []).append(
                            n in res["mcs"])
        el = time.time() - t0
        print(f"[mcs-levels stage4] {i+1}/{len(assets)} {asset} done, "
              f"{el/(i+1):.1f} s/asset, ETA {el/(i+1)*(len(assets)-i-1)/60:.0f} min",
              flush=True)
    rows = []
    for (er, a, lv), mem in memberships.items():
        for n, hits in mem.items():
            rows.append({"forecaster": n, "alpha": a, "erule": er, "level": lv,
                         "n_asset": len(hits), "mcs_rate": float(np.mean(hits))})
    return pd.DataFrame(rows)


def repair_levels() -> pd.DataFrame:
    from backtests.scores import fz0
    models = sorted(TSFM_TAUS)
    manifest = yaml.safe_load((ROOT / "data" / "data_manifest.yaml").read_text())
    assets = sorted(k for k, v in manifest.items() if not v.get("quarantined"))
    rows = []
    t0 = time.time()
    for mi, model in enumerate(models):
        for asset in assets:
            if not (REP / f"{model}_{asset}.parquet").exists():
                continue        # vol-index rows have no repair series
            pf = pd.read_parquet(REP / f"{model}_{asset}.parquet")
            gf = pd.read_parquet(REP / f"garch_evt_{asset}.parquet")
            fb = pd.read_parquet(BL / f"fhs_{asset}.parquet")
            gb = pd.read_parquet(BL / f"gjr_t_{asset}.parquet")
            fb = fb.rename(columns={f"{k}{acol(a)}": f"fhs_{k}{acol(a)}"
                                    for a in ALPHAS for k in ("v", "e")})
            gb = gb.rename(columns={f"{k}{acol(a)}": f"gjr_{k}{acol(a)}"
                                    for a in ALPHAS for k in ("v", "e")})
            bf = fb[["t"] + [c for c in fb.columns if c.startswith("fhs_")]] \
                .merge(gb[["t"] + [c for c in gb.columns if c.startswith("gjr_")]],
                       on="t")
            m = pf.merge(gf.drop(columns=["y"]), on="t").merge(bf, on="t")
            m = m.iloc[BURN:].reset_index(drop=True)
            vcols = [c for c in m.columns
                     if "_v" in c and not c.startswith(("x4_", "zl_", "cl_"))]
            mask = np.isfinite(m[vcols].to_numpy()).all(axis=1)
            m = m[mask].reset_index(drop=True)
            y = m["y"].to_numpy()
            for a in ALPHAS:
                c = acol(a)
                series = {lb: (m[f"{p}_v{c}"].to_numpy(), m[f"{p}_e{c}"].to_numpy())
                          for lb, p in ARM_PRE.items()}
                series["garch_evt"] = (m[f"ge_v{c}"].to_numpy(),
                                       m[f"ge_e{c}"].to_numpy())
                series["fhs"] = (m[f"fhs_v{c}"].to_numpy(), m[f"fhs_e{c}"].to_numpy())
                labels = MCS_SET + ["garch_evt", "fhs"]
                good = np.ones(len(y), bool)
                for lb in labels:
                    v, e = series[lb]
                    good &= np.isfinite(e) & (e < v) & (e < 0)
                ls = {lb: fz0(y[good], series[lb][0][good], series[lb][1][good], a)
                      for lb in labels}
                n_ok = int(good.sum())
                L = np.column_stack([ls[lb] for lb in labels])
                for lv in LEVELS:
                    res = mcs(L, names=labels, level=lv, seed=20260704)
                    for lb in labels:
                        rows.append({"model": model, "asset": asset, "alpha": a,
                                     "arm": lb, "level": lv,
                                     "in_mcs": lb in res["mcs"],
                                     "mcs_p": res["pvalues"].get(lb, np.nan),
                                     "n": n_ok})
        el = time.time() - t0
        print(f"[mcs-levels repair] model {mi+1}/{len(models)} {model} done, "
              f"{el/(mi+1)/60:.1f} min/model", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    # --repair-only: resume switch after a completed stage-4 pass (the
    # stage-4 CSV must already exist and is re-gated below either way)
    if "--repair-only" in sys.argv:
        s4 = pd.read_csv(S4 / "mcs_membership_levels.csv")
    else:
        s4 = stage4_levels()
        s4.to_csv(S4 / "mcs_membership_levels.csv", index=False)
    # reproduction gate: 10% layer == delivered mcs_membership.csv
    old = pd.read_csv(S4 / "mcs_membership.csv")
    new10 = s4[s4.level == 0.10].drop(columns=["level"]).reset_index(drop=True)
    key = ["forecaster", "alpha", "erule"]
    mg = old.merge(new10, on=key, suffixes=("_old", "_new"))
    if len(mg) != len(old) or len(old) != len(new10):
        sys.exit(f"REPRODUCTION FAIL stage4: row count {len(old)} vs {len(new10)}")
    bad = mg[(mg.n_asset_old != mg.n_asset_new)
             | (np.abs(mg.mcs_rate_old - mg.mcs_rate_new) > 1e-12)]
    if len(bad):
        print(bad.to_string(), flush=True)
        sys.exit(f"REPRODUCTION FAIL stage4: {len(bad)} mismatched cells")
    print("[mcs-levels] stage4 10% layer reproduces delivered membership "
          f"({len(old)} rows exact)", flush=True)

    rp = repair_levels()
    rp.to_csv(S4 / "repair_mcs_levels.csv", index=False)
    oldr = pd.read_csv(S4 / "repair_mcs.csv")
    new10 = rp[rp.level == 0.10].drop(columns=["level"]).reset_index(drop=True)
    key = ["model", "asset", "alpha", "arm"]
    mg = oldr.merge(new10, on=key, suffixes=("_old", "_new"))
    if len(mg) != len(oldr) or len(oldr) != len(new10):
        sys.exit(f"REPRODUCTION FAIL repair: row count {len(oldr)} vs {len(new10)}")
    bad = mg[(mg.in_mcs_old != mg.in_mcs_new) | (mg.n_old != mg.n_new)
             | (np.abs(mg.mcs_p_old - mg.mcs_p_new) > 1e-9)]
    if len(bad):
        print(bad.to_string(), flush=True)
        sys.exit(f"REPRODUCTION FAIL repair: {len(bad)} mismatched cells")
    print("[mcs-levels] repair 10% layer reproduces delivered repair_mcs "
          f"({len(oldr)} rows exact)", flush=True)
    print("mcs-levels complete", flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("mcs_levels")
