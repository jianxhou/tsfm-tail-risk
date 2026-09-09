"""v1.5 robustness analyses (ZERO new predictions).

NOTE (replication scope): this script reads the intermediate repair-series
layer results/stage4/repairs/*.parquet, which is NOT distributed in the
public replication package for size reasons; the package ships the two
output CSVs (v15_decomp_robustness.csv, v15_dm_nw.csv) instead.

    python -m harness.run_v15_robustness

 10a. Decomposition-CI robustness on the SAME per-asset paired FZ0 series
      behind the v1.2 section-A analysis (scale = F1-H, location = H-GE at
      alpha=1%, E3, frozen matched CSV): (i) leave-one-asset-class-out means
      over the 6 classes; (ii) UNclustered asset-level iid bootstrap 95% CIs
      (B=2000, seed 20260722).
 10b. DM HAC sensitivity: daily FZ0 loss differentials are rebuilt from the
      stored repair-series parquets (results/stage4/repairs/*.parquet) - a
      derivation from existing series, no model run - and every DM row in
      repairs/_dm_*.csv is recomputed with a Newey-West (Bartlett) HAC
      variance at the automatic lag L = floor(4*(n/100)^(2/9)).
      FIDELITY GATE: the plain h=1 recomputation must reproduce the frozen
      stat to 1e-6 for every row BEFORE the NW variant is reported; any
      mismatch aborts (zero-result-change discipline).

Outputs: results/stage4/v15_decomp_robustness.csv,
results/stage4/v15_dm_nw.csv, docs/v15_robustness.md (script-generated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from backtests.scores import fz0
from backtests.dm import dm_test
from harness.run_prescription import A, ALL_HEADS, S4, ROOT
from harness.run_v12_sensitivity import per_asset_fz0

B = 2000
SEED = 20260722
ALPHAS = {0.01: "01", 0.025: "025", 0.05: "05"}


def loo_and_unclustered(rb):
    ge_fz, _ = per_asset_fz0(rb, "garch_evt")
    rows = []
    for head in ALL_HEADS:
        f1_fz, grp = per_asset_fz0(rb, f"{head}+F1", "e3")
        h_fz, _ = per_asset_fz0(rb, f"{head}+H", "e3")
        d = pd.DataFrame({"f1": f1_fz, "h": h_fz, "ge": ge_fz,
                          "group": grp}).dropna()
        d["scale"] = d["f1"] - d["h"]
        d["loc"] = d["h"] - d["ge"]
        classes = sorted(d.group.unique())
        loo_s = [d[d.group != c]["scale"].mean() for c in classes]
        loo_l = [d[d.group != c]["loc"].mean() for c in classes]
        rng = np.random.default_rng(SEED)
        n = len(d)
        idx = rng.integers(0, n, size=(B, n))
        sv, lv = d["scale"].to_numpy(), d["loc"].to_numpy()
        bs_s = sv[idx].mean(axis=1)
        bs_l = lv[idx].mean(axis=1)
        ci = lambda v: (float(np.percentile(v, 2.5)),
                        float(np.percentile(v, 97.5)))
        s_lo, s_hi = ci(bs_s)
        l_lo, l_hi = ci(bs_l)
        rows.append(dict(
            head=head, n_assets=n,
            scale_mean=sv.mean(), loc_mean=lv.mean(),
            loo_scale_min=min(loo_s), loo_scale_max=max(loo_s),
            loo_loc_min=min(loo_l), loo_loc_max=max(loo_l),
            unclustered_scale_lo=s_lo, unclustered_scale_hi=s_hi,
            unclustered_loc_lo=l_lo, unclustered_loc_hi=l_hi))
    return pd.DataFrame(rows)


def matched_frame(model: str, asset: str):
    """Reproduce matched_rows()' alignment VERBATIM from the stored series:
    merge model parquet + garch_evt frame + baselines frame on t, skip the
    500-day burn-in, then keep rows where ALL *_v columns are finite."""
    from harness.run_repair_series import BURN, baseline_frame
    pf = pd.read_parquet(S4 / "repairs" / f"{model}_{asset}.parquet")
    gf = pd.read_parquet(S4 / "repairs" / f"garch_evt_{asset}.parquet")
    bf = baseline_frame(asset)
    m = pf.merge(gf.drop(columns=["y"]), on="t").merge(bf, on="t")
    m = m.iloc[BURN:].reset_index(drop=True)
    vcols = [c for c in m.columns if "_v" in c]
    mask = np.isfinite(m[vcols].to_numpy()).all(axis=1)
    return m[mask].reset_index(drop=True)


def pair_losses(m: pd.DataFrame, a: float, pre_a: str, pre_b: str):
    """Per-day FZ0 losses on the common good-ES mask (verbatim logic)."""
    c = ALPHAS[a]
    y = m["y"].to_numpy()
    good = np.ones(len(y), bool)
    out = []
    for pre in (pre_a, pre_b):
        v, e = m[f"{pre}_v{c}"].to_numpy(), m[f"{pre}_e{c}"].to_numpy()
        good &= np.isfinite(e) & (e < v) & (e < 0)
    for pre in (pre_a, pre_b):
        v, e = m[f"{pre}_v{c}"].to_numpy(), m[f"{pre}_e{c}"].to_numpy()
        out.append(fz0(y[good], v[good], e[good], a))
    return out[0], out[1], int(good.sum())


def nw_dm(diff: np.ndarray):
    """DM statistic with Newey-West (Bartlett) HAC variance, automatic lag."""
    n = len(diff)
    dbar = diff.mean()
    dc = diff - dbar
    L = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    g0 = float((dc ** 2).mean())
    s = g0
    for l in range(1, L + 1):
        gl = float((dc[l:] * dc[:-l]).mean())
        s += 2.0 * (1.0 - l / (L + 1.0)) * gl
    stat = dbar / np.sqrt(max(s, 1e-300) / n)
    return stat, 2.0 * (1.0 - sps.norm.cdf(abs(stat))), L


PAIR_PRE = {"F1_vs_garch_evt": ("f1", "ge"), "F1_vs_F4": ("f1", "f4_e3")}


def dm_hac(rb_models):
    rows = []
    worst = 0.0
    for model in rb_models:
        frozen = pd.read_csv(S4 / "repairs" / f"_dm_{model}.csv")
        frames = {}
        for _, r in frozen.iterrows():
            a = float(r.alpha)
            if r.asset not in frames:
                frames[r.asset] = matched_frame(model, r.asset)
            da, db, n_ok = pair_losses(frames[r.asset], a, *PAIR_PRE[r.pair])
            assert n_ok == int(r.n), ("ALIGNMENT", model, r.asset, a, r.pair,
                                      n_ok, r.n)
            rep = dm_test(da, db, h=1)
            worst = max(worst, abs(rep["stat"] - r.stat))
            assert abs(rep["stat"] - r.stat) < 1e-6, (
                "FIDELITY GATE FAILED", model, r.asset, a, r.pair,
                rep["stat"], r.stat)
            stat_nw, p_nw, L = nw_dm(np.asarray(da) - np.asarray(db))
            rows.append(dict(model=model, asset=r.asset, alpha=a, pair=r.pair,
                             n=int(r.n), stat_frozen=r.stat, p_frozen=r.p,
                             stat_nw=stat_nw, p_nw=p_nw, nw_lag=L,
                             mean_diff=r.mean_diff))
    print(f"[10b] fidelity gate passed: max |stat_recomputed - stat_frozen| "
          f"= {worst:.2e} over {len(rows)} rows")
    return pd.DataFrame(rows)


def main():
    rb = pd.read_csv(S4 / "repair_backtests_matched.csv")

    dec = loo_and_unclustered(rb)
    dec.to_csv(S4 / "v15_decomp_robustness.csv", index=False)

    dm = dm_hac(ALL_HEADS)
    dm.to_csv(S4 / "v15_dm_nw.csv", index=False)

    sig_f = dm.p_frozen < 0.05
    sig_n = dm.p_nw < 0.05
    flips = dm[sig_f != sig_n]
    fav_f1_sig_nw = dm[(dm.p_nw < 0.05) & (dm.mean_diff < 0) &
                       (dm.pair == "F1_vs_garch_evt")]

    L = ["# v1.5 robustness analyses (script-generated by "
         "harness/run_v15_robustness.py; seed %d, B=%d)" % (SEED, B), "",
         "## 10a. Decomposition-CI robustness (alpha=1%, E3; same frozen "
         "per-asset series as the v1.2 section-A analysis)", "",
         "| head | n | scale mean | LOO scale range | unclustered scale CI | "
         "loc mean | LOO loc range | unclustered loc CI |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in dec.iterrows():
        L.append(
            f"| {r['head']} | {int(r.n_assets)} | {r.scale_mean:+.3f} | "
            f"[{r.loo_scale_min:+.3f}, {r.loo_scale_max:+.3f}] | "
            f"[{r.unclustered_scale_lo:+.3f}, {r.unclustered_scale_hi:+.3f}] | "
            f"{r.loc_mean:+.3f} | "
            f"[{r.loo_loc_min:+.3f}, {r.loo_loc_max:+.3f}] | "
            f"[{r.unclustered_loc_lo:+.3f}, {r.unclustered_loc_hi:+.3f}] |")
    zero_excl = [(r["head"], r.unclustered_scale_lo > 0) for _, r in dec.iterrows()]
    L += ["", "Unclustered scale CI excludes zero: " +
          ", ".join(f"{h}={'yes' if z else 'NO'}" for h, z in zero_excl), "",
          "LOO scale means all positive: " +
          str(bool((dec.loo_scale_min > 0).all())), ""]

    L += ["## 10b. DM Newey-West HAC sensitivity (automatic Bartlett lag; "
          "fidelity gate: frozen h=1 stats reproduced to 1e-6 on all rows)",
          "",
          f"- rows recomputed: {len(dm)}",
          f"- significant at 5% (frozen): {int(sig_f.sum())}; (NW): "
          f"{int(sig_n.sum())}; decision flips: {len(flips)}",
          f"- flips by direction: to-significant {int(((~sig_f) & sig_n).sum())}, "
          f"to-insignificant {int((sig_f & (~sig_n)).sum())}",
          f"- NW-significant F1-vs-GARCH-EVT rows favoring F1 (mean_diff<0): "
          f"{len(fav_f1_sig_nw)}", ""]
    if len(flips):
        L += ["| model | asset | alpha | pair | p frozen | p NW |", "|---|---|---|---|---|---|"]
        for _, r in flips.iterrows():
            L.append(f"| {r.model} | {r.asset} | {r.alpha} | {r.pair} | "
                     f"{r.p_frozen:.4f} | {r.p_nw:.4f} |")
    out = ROOT / "docs" / "v15_robustness.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} + 2 CSVs")


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("v15_robustness")
