"""Stage-5 closeout: per-day repair series + matched-sample battery + registered
follow-up statistics (signer closeout rulings 2026-07-08).

    PYTHONUNBUFFERED=1 python -u -m harness.run_repair_series

What this adds over run_grid_repairs (which stays untouched as the delivered
artifact):
 1. PERSISTS per-day (v, e) for every arm — F1, H, F2 gamma in {0.5x, 1x, 2x},
    F3, F4 (E3 + E2), unrepaired E3/E2, GARCH-EVT — results/stage4/repairs/.
 2. MATCHED-SAMPLE battery (design §A: evaluation starts OOS day 501; the
    delivered grid evaluated arms on unmatched samples — disclosed deviation,
    remediated here): common day set per asset across all arms + FHS/GJR-t
    references, MC-calibrated UC/CC/DQ (DQ-MC closes a second deviation).
    -> repair_backtests_matched.csv
 3. F2 gamma robustness {0.5x, 2x} (pre-registered §A, previously not run).
 4. DM-on-FZ0 double bar per (model, asset, alpha): F1 vs GARCH-EVT, F1 vs F4
    (HAC h=1, registered) -> repair_dm.csv
 5. MCS membership per (model, asset, alpha) over {F1,H,F2,F3,F4,GARCH-EVT,FHS}
    FZ0 losses -> repair_mcs.csv
Continuity check: full-OOS battery of recomputed F2(gamma=1x, E3) vs the
delivered repair_backtests.csv rows (hit_rate must agree) before anything else
is written per model.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtests.dm import dm_test
from backtests.mcs import mcs
from backtests.scores import fz0
from backtests.var_tests import hits as hit_series
from data.load import load_series
from fixes import arms
from harness import run_grid_repairs as base
from harness.run_grid_extract import battery, load_ctxfits, mc_p
from precision.mc_critical import dq_stat_from_hits

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
OUT = S4 / "repairs"
BURN = 500                      # design §A: evaluation starts OOS day 501
GAMMAS = (("g05", 0.5), ("g10", 1.0), ("g20", 2.0))
ALPHAS = base.ALPHAS
TSFM = base.TSFM
MCS_SEED = 20260704             # backtests.mcs module default, as in stage-4 MCS


def acol(a):
    return base.acol(a)


def per_day_frame(model, asset, fits, x, oos):
    """All arms' per-day (v, e) for one (model, asset), aligned on t."""
    e3, e2, mu, sig = base.build_aligned(model, asset, fits, x, oos)
    fc = pd.read_parquet(base.FC / f"{model}_{asset}.parquet")[["t", "date"]]
    dates = e3[["t"]].merge(fc, on="t")["date"].to_numpy()
    cols = {"t": e3["t"].to_numpy(), "date": dates, "y": e3["y"].to_numpy()}
    for ext, al in (("e3", e3), ("e2", e2)):
        for a in ALPHAS:
            c = acol(a)
            cols[f"u_{ext}_v{c}"] = al[f"v{c}"].to_numpy()
            cols[f"u_{ext}_e{c}"] = al[f"e{c}"].to_numpy()
    f1 = arms.f1_multi(e3, ALPHAS)
    h = arms.h_multi(e3, sig, ALPHAS)
    x4 = arms.x4_multi(e3, mu, ALPHAS)      # v2.0 control arm (review #9 §9.1)
    for name, d in (("f1", f1), ("h", h), ("x4", x4)):
        for a in ALPHAS:
            c = acol(a)
            cols[f"{name}_v{c}"] = d[a][0]
            cols[f"{name}_e{c}"] = d[a][1]
    for ext, al in (("e3", e3), ("e2", e2)):
        s_hat = base.shat_first_window(al)        # v2.0 P0-2: first-500-day ŝ
        for a in ALPHAS:
            c = acol(a)
            alp = al[["t", "y"]].copy()
            alp["v"] = al[f"v{c}"].to_numpy()
            alp["e"] = al[f"e{c}"].to_numpy()
            for gname, gm in GAMMAS:
                v, e = arms.f2_adaptive_conformal(alp, a, s_hat, gamma_mult=gm)
                cols[f"f2{gname}_{ext}_v{c}"] = v
                cols[f"f2{gname}_{ext}_e{c}"] = e
            v, e = arms.f3_rescale(alp, a)
            cols[f"f3_{ext}_v{c}"] = v
            cols[f"f3_{ext}_e{c}"] = e
            v, e = arms.f4_additive_fz0(alp, a, s_hat)
            cols[f"f4_{ext}_v{c}"] = v
            cols[f"f4_{ext}_e{c}"] = e
    return pd.DataFrame(cols)


def garch_evt_frame(asset, fits, x, oos):
    """GARCH-EVT + the two v2.0 location-control arms (zero / constant location,
    GARCH scale; review #9 §9.1) — model-independent, computed once per asset on
    the chronos_2-aligned frame and merged into each matched frame by t."""
    e3, _, mu, sig = base.build_aligned("chronos_2", asset, fits, x, oos)
    ge = arms.garch_evt_multi(e3, mu, sig, ALPHAS)
    zl = arms.zero_loc_multi(e3, sig, ALPHAS)
    cl = arms.const_loc_multi(e3, sig, ALPHAS)
    cols = {"t": e3["t"].to_numpy(), "y": e3["y"].to_numpy()}
    for pre, d in (("ge", ge), ("zl", zl), ("cl", cl)):
        for a in ALPHAS:
            c = acol(a)
            cols[f"{pre}_v{c}"] = d[a][0]
            cols[f"{pre}_e{c}"] = d[a][1]
    return pd.DataFrame(cols)


def baseline_frame(asset):
    """FHS + GJR-t per-day references from the persisted stage-4 baselines."""
    f = pd.read_parquet(S4 / "baselines" / f"fhs_{asset}.parquet")
    g = pd.read_parquet(S4 / "baselines" / f"gjr_t_{asset}.parquet")
    f = f.rename(columns={f"{k}{acol(a)}": f"fhs_{k}{acol(a)}"
                          for a in ALPHAS for k in ("v", "e")})
    g = g.rename(columns={f"{k}{acol(a)}": f"gjr_{k}{acol(a)}"
                          for a in ALPHAS for k in ("v", "e")})
    keep_f = ["t"] + [c for c in f.columns if c.startswith("fhs_")]
    keep_g = ["t"] + [c for c in g.columns if c.startswith("gjr_")]
    return f[keep_f].merge(g[keep_g], on="t")


def battery_m(y, v, e, alpha):
    """battery() + MC-calibrated DQ (design §A: MCNull 'dq'; observed stat is the
    hits-only DQ so it matches the MCNull null construction)."""
    b = battery(y, v, e, alpha)
    ok = np.isfinite(v)
    hh = hit_series(y[ok], v[ok])
    b["dq_p_mc"] = mc_p("dq", dq_stat_from_hits(hh, alpha), int(ok.sum()), alpha)
    return b


ARM_COLS = {          # forecaster label -> per-day column prefix
    "E3": "u_e3", "E2": "u_e2", "F1": "f1", "H": "h",
    "X4": "x4",   # v2.0 fourth-cell control arm (GARCH loc + TSFM scale)
    "F2g05": "f2g05_e3", "F2": "f2g10_e3", "F2g20": "f2g20_e3",
    "F3": "f3_e3", "F4": "f4_e3",
    "F2g05-e2": "f2g05_e2", "F2-e2": "f2g10_e2", "F2g20-e2": "f2g20_e2",
    "F3-e2": "f3_e2", "F4-e2": "f4_e2",
}
MCS_SET = ["F1", "H", "F2", "F3", "F4"]          # + garch_evt, fhs appended
# v2.0: the three control arms (X4/zero_loc/const_loc) enter the matched battery
# CSV ONLY — the MCS membership set and DM pairs stay as delivered (adding arms
# would change every member's MCS p-value; review arms are CSV-only in Phase 1).


def matched_rows(model, asset, group, pf, gf, bf):
    """Matched-sample battery rows + DM + MCS for one (model, asset)."""
    m = pf.merge(gf.drop(columns=["y"]), on="t").merge(bf, on="t")
    m = m.iloc[BURN:].reset_index(drop=True)
    # v2.0: the common-day mask is defined over the DELIVERED arm set only — the
    # three review-suggested control arms (x4/zl/cl) are evaluated ON this day
    # set (own NaNs drop per-cell inside battery) and must not shrink the
    # matched sample the delivered numbers are defined on (design §A).
    vcols = [c for c in m.columns
             if "_v" in c and not c.startswith(("x4_", "zl_", "cl_"))]
    mask = np.isfinite(m[vcols].to_numpy()).all(axis=1)
    dropped = int((~mask).sum())
    m = m[mask].reset_index(drop=True)
    y = m["y"].to_numpy()
    rows, dm_rows, mcs_rows = [], [], []
    for a in ALPHAS:
        c = acol(a)
        series = {}
        for label, pre in ARM_COLS.items():
            erule = "e2" if (label.endswith("-e2") or label == "E2") else "e3"
            name = f"{model}+{label.replace('-e2', '')}" if label not in ("E3", "E2") \
                else f"{model}+unrepaired"
            v, e = m[f"{pre}_v{c}"].to_numpy(), m[f"{pre}_e{c}"].to_numpy()
            series[label] = (v, e)
            rows.append({"forecaster": name, "asset": asset, "alpha": a,
                         "erule": erule, "group": group,
                         **battery_m(y, v, e, a)})
        for name, pre in (("garch_evt", "ge"), ("zero_loc", "zl"),
                          ("const_loc", "cl"), ("fhs", "fhs"), ("gjr_t", "gjr")):
            v, e = m[f"{pre}_v{c}"].to_numpy(), m[f"{pre}_e{c}"].to_numpy()
            series[name] = (v, e)
            rows.append({"forecaster": name, "asset": asset, "alpha": a,
                         "erule": "param", "group": group, "context_model": model,
                         **battery_m(y, v, e, a)})
        # per-day FZ0 losses on the common good-ES mask of each compared set
        def losses(labels):
            good = np.ones(len(y), bool)
            for lb in labels:
                v, e = series[lb]
                good &= np.isfinite(e) & (e < v) & (e < 0)
            return {lb: fz0(y[good], series[lb][0][good], series[lb][1][good], a)
                    for lb in labels}, int(good.sum())
        for pair in (("F1", "garch_evt"), ("F1", "F4")):
            ls, n_ok = losses(pair)
            d = dm_test(ls[pair[0]], ls[pair[1]], h=1)
            dm_rows.append({"model": model, "asset": asset, "alpha": a,
                            "pair": f"{pair[0]}_vs_{pair[1]}", "n": n_ok, **d})
        labels = MCS_SET + ["garch_evt", "fhs"]
        ls, n_ok = losses(labels)
        L = np.column_stack([ls[lb] for lb in labels])
        res = mcs(L, names=labels, level=0.10, seed=MCS_SEED)
        for lb in labels:
            mcs_rows.append({"model": model, "asset": asset, "alpha": a, "arm": lb,
                             "in_mcs": lb in res["mcs"],
                             "mcs_p": res["pvalues"].get(lb, np.nan), "n": n_ok})
    return rows, dm_rows, mcs_rows, dropped


def continuity_check(model, asset, pf, delivered):
    """Recomputed F2(1x, E3) full-OOS battery must reproduce the delivered row."""
    y = pf["y"].to_numpy()
    diffs = []
    for a in ALPHAS:
        c = acol(a)
        b = battery(y, pf[f"f2g10_e3_v{c}"].to_numpy(),
                    pf[f"f2g10_e3_e{c}"].to_numpy(), a)
        ref = delivered[(delivered.forecaster == f"{model}+F2")
                        & (delivered.asset == asset) & (delivered.alpha == a)
                        & (delivered.erule == "e3")]
        if len(ref):
            diffs.append(abs(b["hit_rate"] - float(ref.hit_rate.iloc[0])))
    return max(diffs) if diffs else np.nan


def _append(path: Path, rows):
    """Incremental per-asset persistence (crash/kill-resumable runs).

    v1.6 item D22 fix: when appending to an existing file, reindex to the
    file's header order — per-chunk DataFrames can carry a different
    first-seen key order (param rows put context_model before the battery
    keys), which previously shifted 279 param rows' values one column left
    (model name landed in es_viol)."""
    if rows:
        df = pd.DataFrame(rows)
        if path.exists():
            cols = pd.read_csv(path, nrows=0).columns
            df = df.reindex(columns=cols.union(df.columns, sort=False))
        df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    # v2.0 orchestration CLI (results-invariant: outputs are seeded and cached
    # per (model, asset); sharding only changes wall-clock): --models runs a
    # subset, --no-finalize skips the final concat (one finalize call after all
    # shards), --prep-assets precomputes the per-asset garch_evt/baseline
    # frames for a shard of assets then exits (avoids parquet write races).
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--no-finalize", action="store_true")
    ap.add_argument("--prep-assets", default="")
    cli = ap.parse_args()
    tsfm_run = [m for m in cli.models.split(",") if m in TSFM] if cli.models else TSFM

    OUT.mkdir(exist_ok=True)
    mf = yaml.safe_load((S4 / "run_manifest.yaml").read_text())
    oos = mf["protocol"]["oos_start"]
    dm_manifest = yaml.safe_load((ROOT / "data" / "data_manifest.yaml").read_text())
    assets = mf["assets"]

    if cli.prep_assets:
        for asset in [a for a in cli.prep_assets.split(",") if a in assets]:
            gp = OUT / f"garch_evt_{asset}.parquet"
            if gp.exists():
                print(f"[prep] {asset} cached", flush=True)
                continue
            fits = load_ctxfits(S4 / "ctxfits" / f"{asset}.parquet")  # B2-h
            df = load_series(asset)
            x = df[df.attrs.get("column", "logret")].to_numpy()
            garch_evt_frame(asset, fits, x, oos).to_parquet(gp)
            print(f"[prep] {asset} done", flush=True)
        print("prep shard complete", flush=True)
        return

    delivered = pd.read_csv(S4 / "repair_backtests.csv")

    ge_cache, bl_cache = {}, {}
    for asset in assets:
        gp = OUT / f"garch_evt_{asset}.parquet"
        if gp.exists():
            gf = pd.read_parquet(gp)
        else:
            fits = load_ctxfits(S4 / "ctxfits" / f"{asset}.parquet")  # B2-h
            df = load_series(asset)
            x = df[df.attrs.get("column", "logret")].to_numpy()
            gf = garch_evt_frame(asset, fits, x, oos)
            gf.to_parquet(gp)
        ge_cache[asset] = gf
        bl_cache[asset] = baseline_frame(asset)
    print("[series] garch_evt + baseline frames ready (32 assets)", flush=True)

    for model in tsfm_run:
        t0 = time.time()
        cont = []
        part_b = OUT / f"_battery_{model}.csv"
        part_d = OUT / f"_dm_{model}.csv"
        part_m = OUT / f"_mcs_{model}.csv"
        done = (set(pd.read_csv(part_b).asset.unique()) if part_b.exists()
                else set())
        for k, asset in enumerate(assets):
            if asset in done:
                continue
            try:
                fpq = base.FC / f"{model}_{asset}.parquet"
                if not fpq.exists():
                    print(f"[series] SKIP {model}/{asset}: no forecast", flush=True)
                    continue
                group = dm_manifest[asset]["group"]
                pq = OUT / f"{model}_{asset}.parquet"
                if pq.exists():
                    pf = pd.read_parquet(pq)
                else:
                    fits = load_ctxfits(S4 / "ctxfits" / f"{asset}.parquet")  # B2-h
                    df = load_series(asset)
                    x = df[df.attrs.get("column", "logret")].to_numpy()
                    pf = per_day_frame(model, asset, fits, x, oos)
                    pf.to_parquet(pq)
                cont.append(continuity_check(model, asset, pf, delivered))
                r, d, mrows, dropped = matched_rows(model, asset, group, pf,
                                                    ge_cache[asset], bl_cache[asset])
                _append(part_b, r)
                _append(part_d, d)
                _append(part_m, mrows)
                el = time.time() - t0
                print(f"[series] {model} {asset} ({k+1}/{len(assets)}) "
                      f"{el/(k+1):.1f}s/asset dropped={dropped}", flush=True)
            except AssertionError:      # B2-d: guard asserts (anchor/mass/schema)
                raise                   # must fail the RUN, never a cell skip
            except Exception as ex:
                print(f"[series] SKIP {model}/{asset}: {type(ex).__name__}: {ex}",
                      flush=True)
        cmax = np.nanmax(cont) if cont else np.nan
        print(f"[series] {model} done ({(time.time()-t0)/60:.1f} min, "
              f"continuity max|dHit|={cmax:.2e})", flush=True)

    if cli.no_finalize:
        print("shard complete (no finalize)", flush=True)
        return
    bat = pd.concat([pd.read_csv(OUT / f"_battery_{m}.csv") for m in TSFM])
    dmr = pd.concat([pd.read_csv(OUT / f"_dm_{m}.csv") for m in TSFM])
    mcr = pd.concat([pd.read_csv(OUT / f"_mcs_{m}.csv") for m in TSFM])
    bat.to_csv(S4 / "repair_backtests_matched.csv", index=False)
    dmr.to_csv(S4 / "repair_dm.csv", index=False)
    mcr.to_csv(S4 / "repair_mcs.csv", index=False)
    meta = {"burn_in": BURN, "gammas": [g for _, g in GAMMAS],
            "mc_seed": 20260706, "mcs_seed": MCS_SEED, "f4_seed": arms.SEED,
            "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
            "n_battery_rows": len(bat), "n_dm_rows": len(dmr),
            "n_mcs_rows": len(mcr)}
    (OUT / "meta.yaml").write_text(yaml.safe_dump(meta))
    print(f"repair series complete: {len(bat)} battery rows, "
          f"{len(dmr)} DM rows, {len(mcr)} MCS rows", flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("repair_series")
