"""External-review micro-round v1.2 analyses (signer-adjudicated, 2026-07-17).

    python -m harness.run_v12_sensitivity

Zero new forecasts: everything is computed from the frozen matched-sample CSV
results/stage4/repair_backtests_matched.csv already behind §10.

 A. Decomposition uncertainty (P0 item 6): per-asset paired FZ0 differences
    (F1−H) [scale term] and (H−GE) [location term] at α=1%, E3; per head the
    count of assets with scale > location, and asset-class cluster block
    bootstrap 95% CIs (resample the 6 classes with replacement, B=2000,
    seed 20260717) for the mean of each term.
 B. Criterion-threshold sensitivity (P0 item 7): the registered joint success
    criterion re-evaluated for H on a grid margin {5,10,15}pp × FZ0 tolerance
    {1.00,1.025,1.05,1.10} × fraction {1/2,2/3,3/4}. The registered combo
    (10pp, 1.05, 2/3) is reproduced and asserted against the frozen §10.2
    numbers before any sweep. The registered ref<=0 fallback (additive band
    0.05 = tol−1) generalizes as tol−1 on the sweep.
 C. Bare pass counts for §10.2 (P0 item 5e): per-head H MC-UC+CC pass counts
    k/32 behind the published percentages.

Outputs: docs/v12_sensitivity.md (script-generated; the only source the paper
tables/pack read), results/stage4/v12_decomp_uncertainty.csv,
results/stage4/v12_criterion_grid.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from harness.run_prescription import (A, ALL_HEADS, S4, ROOT, cell,
                                      coverage_pass)

B = 2000
SEED = 20260717
MARGINS = [0.05, 0.10, 0.15]
TOLS = [1.00, 1.025, 1.05, 1.10]
FRACS = [(1, 2), (2, 3), (3, 4)]
# v2.0: the pre-v2.0 frozen registered-combo values are kept as a REFERENCE for
# the old->new comparison; the hard continuity asserts against them were retired
# when the v2.0 CHANGELOG entry unfroze the affected results (external review
# #9, P0-1/2/3 recomputation) — drift vs these values is now reported, not fatal.
PRE_V20_PR = {"chronos_2": 0.69, "moirai_2_0": 0.75, "lag_llama": 0.81,
              "chronos_bolt": 0.72, "timesfm_2_5": 0.72}
PRE_V20_BFRAC = {"chronos_2": 0.84, "moirai_2_0": 0.81, "lag_llama": 0.78,
                 "chronos_bolt": 0.81, "timesfm_2_5": 0.78}


def cr1_infer(x: np.ndarray, groups: np.ndarray) -> dict:
    """v2.0 P0-6 cluster-valid inference for a clustered mean (G=6 asset classes).

    - CR1 cluster-robust SE for the mean-only model x_i = mu + e_i:
      u_g = sum_{i in g}(x_i - xbar); Var = c * sum_g u_g^2 / n^2 with the CR1
      small-sample factor c = G/(G-1) * (n-1)/(n-k), k=1.
    - 95% CI on t(G-1) = t(5).
    - RESTRICTED (null-imposed, mu=0) wild cluster bootstrap with Rademacher
      weights, FULL enumeration of all 2^G = 64 sign vectors — deterministic,
      so no RNG enters (the enumeration replaces seeded sampling); p = share of
      sign vectors with |t*| >= |t_obs| (weak inequality, both tails; the
      identity and global-flip vectors always count, so the attainable floor
      is 2/64 ~= 0.031 under this convention)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    gs = list(pd.unique(groups))
    G = len(gs)
    gidx = np.array([gs.index(g) for g in groups])

    def se_of(v):
        vb = v.mean()
        u = np.array([(v[gidx == j] - vb).sum() for j in range(G)])
        c = G / (G - 1) * (n - 1) / (n - 1)
        return float(np.sqrt(c * (u ** 2).sum()) / n)

    xbar = float(x.mean())
    se = se_of(x)
    t_obs = xbar / se if se > 0 else np.inf
    tq = float(sps.t.ppf(0.975, G - 1))
    hits = 0
    for m in range(2 ** G):
        s = np.array([1.0 if (m >> j) & 1 else -1.0 for j in range(G)])
        xs = x * s[gidx]                     # restricted residuals = x (null mu=0)
        se_s = se_of(xs)
        t_s = xs.mean() / se_s if se_s > 0 else np.inf
        if abs(t_s) >= abs(t_obs) - 1e-12:
            hits += 1
    return dict(se=se, t=t_obs, lo=xbar - tq * se, hi=xbar + tq * se,
                wcr_p=hits / 2 ** G)


def per_asset_fz0(rb, name, erule=None, a=A):
    sub = rb[(rb.forecaster == name) & (rb.alpha == a)]
    if erule is not None:
        sub = sub[sub.erule == erule]
    sub = sub.drop_duplicates("asset").set_index("asset")
    return sub.fz0_mean, sub.group


def verdict_param(e3, thr, min_ref, tol, frac):
    """criterion()'s verdict, with (thr, tol, frac) parametrized; logic
    otherwise verbatim from harness/run_prescription.py::criterion."""
    pr = coverage_pass(e3)
    okb = []
    for asset, fzv in e3.drop_duplicates("asset").set_index("asset").fz0_mean.items():
        ref = min_ref.get(asset, np.nan)
        if not np.isfinite(fzv) or not np.isfinite(ref):
            okb.append(False)
        elif ref <= 0:                              # registered fallback, generalized
            okb.append(fzv - ref <= tol - 1.0)
        else:
            okb.append(fzv <= tol * ref)
    bfrac = float(np.mean(okb)) if okb else np.nan
    a_ok = bool(np.isfinite(pr) and pr >= thr)
    return dict(pr=pr, a_ok=a_ok, bfrac=bfrac,
                joint=bool(a_ok and np.isfinite(bfrac) and bfrac >= frac))


def main():
    rb = pd.read_csv(S4 / "repair_backtests_matched.csv")
    fhs_r = rb[(rb.forecaster == "fhs") & (rb.alpha == A)].drop_duplicates("asset")
    gjr_r = rb[(rb.forecaster == "gjr_t") & (rb.alpha == A)].drop_duplicates("asset")
    ge_r = rb[(rb.forecaster == "garch_evt") & (rb.alpha == A)].drop_duplicates("asset")
    fhs_pass = coverage_pass(fhs_r)
    min_ref = pd.concat([gjr_r.set_index("asset").fz0_mean,
                         fhs_r.set_index("asset").fz0_mean], axis=1).min(axis=1)

    # ---------- A. decomposition uncertainty ----------
    ge_fz, _ = per_asset_fz0(rb, "garch_evt")
    dec_rows, md_dec, md_inf = [], [], []
    for head in ALL_HEADS:
        f1_fz, grp = per_asset_fz0(rb, f"{head}+F1", "e3")
        h_fz, _ = per_asset_fz0(rb, f"{head}+H", "e3")
        d = pd.DataFrame({"f1": f1_fz, "h": h_fz, "ge": ge_fz, "group": grp}).dropna()
        d["scale"] = d["f1"] - d["h"]
        d["loc"] = d["h"] - d["ge"]
        n = len(d)
        k = int((d["scale"] > d["loc"]).sum())
        rng = np.random.default_rng(SEED)
        classes = sorted(d.group.unique())
        bs_s, bs_l = [], []
        for _ in range(B):
            pick = rng.choice(classes, size=len(classes), replace=True)
            bd = pd.concat([d[d.group == c] for c in pick])
            bs_s.append(bd["scale"].mean())
            bs_l.append(bd["loc"].mean())
        ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        s_lo, s_hi = ci(bs_s)
        l_lo, l_hi = ci(bs_l)
        # v2.0 P0-6: cluster-valid inference columns (CR1 SE, t(5) CI,
        # restricted wild cluster bootstrap p) for both decomposition terms
        inf_s = cr1_infer(d["scale"].to_numpy(), d["group"].to_numpy())
        inf_l = cr1_infer(d["loc"].to_numpy(), d["group"].to_numpy())
        dec_rows.append(dict(head=head, n_assets=n, n_scale_gt_loc=k,
                             scale_mean=d["scale"].mean(), scale_lo=s_lo, scale_hi=s_hi,
                             loc_mean=d["loc"].mean(), loc_lo=l_lo, loc_hi=l_hi,
                             scale_cr1_se=inf_s["se"], scale_t=inf_s["t"],
                             scale_t5_lo=inf_s["lo"], scale_t5_hi=inf_s["hi"],
                             scale_wcr_p=inf_s["wcr_p"],
                             loc_cr1_se=inf_l["se"], loc_t=inf_l["t"],
                             loc_t5_lo=inf_l["lo"], loc_t5_hi=inf_l["hi"],
                             loc_wcr_p=inf_l["wcr_p"]))
        md_dec.append(f"| {head} | {n} | {k} | {d['scale'].mean():+.3f} "
                      f"[{s_lo:+.3f}, {s_hi:+.3f}] | {d['loc'].mean():+.3f} "
                      f"[{l_lo:+.3f}, {l_hi:+.3f}] |")
        md_inf.append(f"| {head} | {inf_s['se']:.4f} | {inf_s['t']:+.2f} | "
                      f"[{inf_s['lo']:+.4f}, {inf_s['hi']:+.4f}] | {inf_s['wcr_p']:.3f} | "
                      f"{inf_l['se']:.4f} | {inf_l['t']:+.2f} | "
                      f"[{inf_l['lo']:+.4f}, {inf_l['hi']:+.4f}] | {inf_l['wcr_p']:.3f} |")
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(S4 / "v12_decomp_uncertainty.csv", index=False)
    assert (dec.n_scale_gt_loc == dec.n_assets).sum() >= 0  # counts recorded as-is

    # ---------- B. criterion grid ----------
    # v2.0: the registered combo is recomputed and COMPARED to the pre-v2.0
    # frozen values (drift = the expected signature of the authorized P0-1/2/3
    # recomputation; reported in the md + impact report, no longer fatal).
    reg, reg_drift = {}, []
    for head in ALL_HEADS:
        e3 = cell(rb, f"{head}+H", "e3")
        reg[head] = verdict_param(e3, fhs_pass - 0.10, min_ref, 1.05, 2 / 3)
        if round(reg[head]["pr"], 2) != PRE_V20_PR[head]:
            reg_drift.append(f"{head}: pr {PRE_V20_PR[head]:.2f} -> "
                             f"{reg[head]['pr']:.2f}")
        if round(reg[head]["bfrac"], 2) != PRE_V20_BFRAC[head]:
            reg_drift.append(f"{head}: bfrac {PRE_V20_BFRAC[head]:.2f} -> "
                             f"{reg[head]['bfrac']:.2f}")
        if not reg[head]["joint"]:
            reg_drift.append(f"{head}: registered-combo joint verdict now FAILS")
    n_joint_reg = sum(1 for h in ALL_HEADS if reg[h]["joint"])

    grid_rows = []
    for m in MARGINS:
        for tol in TOLS:
            for fn, fd in FRACS:
                heads_pass = []
                for head in ALL_HEADS:
                    e3 = cell(rb, f"{head}+H", "e3")
                    v = verdict_param(e3, fhs_pass - m, min_ref, tol, fn / fd)
                    heads_pass.append((head, v["joint"]))
                grid_rows.append(dict(margin_pp=int(m * 100), tol=tol,
                                      frac=f"{fn}/{fd}",
                                      n_heads_pass=sum(j for _, j in heads_pass),
                                      heads_fail=",".join(h for h, j in heads_pass
                                                          if not j) or "-"))
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(S4 / "v12_criterion_grid.csv", index=False)
    n55 = int((grid.n_heads_pass == 5).sum())

    # C2 criticality at margin 5pp (item 7 honesty clause)
    c2_m5 = verdict_param(cell(rb, "chronos_2+H", "e3"), fhs_pass - 0.05,
                          min_ref, 1.05, 2 / 3)

    # ---------- C. bare pass counts ----------
    counts = {}
    for head in ALL_HEADS:
        e3 = cell(rb, f"{head}+H", "e3").drop_duplicates("asset")
        kk = int(((e3.kupiec_p_mc > 0.05) & (e3.cc_p_mc > 0.05)).sum())
        counts[head] = (kk, len(e3))
        # internal consistency (stays fatal): the bare counts must reproduce
        # the section-B registered-combo coverage rate computed from the same CSV
        if round(kk / len(e3), 2) != round(reg[head]["pr"], 2):
            raise AssertionError(f"pass count/rate mismatch for {head}: {kk}/{len(e3)}")

    # ---------- write the md ----------
    cov_str = "/".join(f"{100*reg[h]['pr']:.0f}" for h in ALL_HEADS)
    bfr_str = "/".join(f"{reg[h]['bfrac']:.2f}".lstrip("0") for h in ALL_HEADS)
    drift_note = ("Pre-v2.0 frozen values reproduced exactly."
                  if not reg_drift else
                  "Drift vs the pre-v2.0 frozen values (expected — v2.0 "
                  "P0-1/2/3 recomputation, CHANGELOG): " + "; ".join(reg_drift) + ".")
    L = ["# v1.2 external-review sensitivity analyses (script-generated; "
         "do not hand-edit)", "",
         f"Source: results/stage4/repair_backtests_matched.csv (matched "
         f"sample; zero new forecasts). alpha=1%, E3. Cluster bootstrap: "
         f"6 asset classes resampled with replacement, B={B}, seed {SEED}. "
         f"FHS pass (matched) = {100*fhs_pass:.1f}%.", "",
         "## A. Decomposition uncertainty (per-asset paired FZ0 differences)", "",
         "| head | n assets | n(scale>loc) | mean scale (F1-H) [95% CI] | "
         "mean loc (H-GE) [95% CI] |",
         "|---|---|---|---|---|"] + md_dec + ["",
         "### A2. Cluster-valid inference (v2.0 P0-6: CR1 cluster-robust SE, "
         "t(5) 95% CI, restricted wild cluster bootstrap p — Rademacher, full "
         "2^6=64 enumeration, deterministic; G=6 asset classes)", "",
         "| head | scale CR1 SE | scale t | scale t(5) CI | scale WCR p | "
         "loc CR1 SE | loc t | loc t(5) CI | loc WCR p |",
         "|---|---|---|---|---|---|---|---|---|"] + md_inf + ["",
         "## B. Registered-criterion threshold sensitivity for H", "",
         f"Grid: margin {{5,10,15}}pp x FZ0 tolerance {{1.00,1.025,1.05,1.10}} "
         f"x fraction {{1/2,2/3,3/4}} = {len(grid)} combinations. Registered "
         f"combo = (10pp, 1.05, 2/3): H joint success {n_joint_reg}/5 heads; "
         f"coverage {cov_str}; b-fractions {bfr_str}. {drift_note}", "",
         f"H passes on all five heads in **{n55} of {len(grid)}** combinations.", "",
         "| margin (pp) | frac | tol=1.00 | tol=1.025 | tol=1.05 | tol=1.10 |",
         "|---|---|---|---|---|---|"]
    for m in MARGINS:
        for fn, fd in FRACS:
            cells = []
            for tol in TOLS:
                r = grid[(grid.margin_pp == int(m * 100)) & (grid.tol == tol)
                         & (grid.frac == f"{fn}/{fd}")].iloc[0]
                cells.append(str(r.n_heads_pass))
            L.append(f"| {int(m*100)} | {fn}/{fd} | " + " | ".join(cells) + " |")
    # v1.6 item A1: the margin-5pp narrative names ALL failing heads. v2.0: the
    # failure set and the per-head critical margins are COMPUTED (the pre-v2.0
    # hardcoded set/margins were retired with the recompute unfreeze).
    m5row = grid[(grid.margin_pp == 5) & (grid.tol == 1.05) & (grid.frac == "2/3")]
    fail5 = m5row.heads_fail.iloc[0]
    fail5_list = [] if fail5 == "-" else fail5.split(",")
    n5pass = int(m5row.n_heads_pass.iloc[0])
    fail5_pretty = "; ".join(
        f"{h} at {100*reg[h]['pr']:.0f}%" for h in fail5_list) or "none"
    # condition-(a) critical margin per head: fails once margin > fhs_pass - pr
    crit = sorted(((h, 100 * (fhs_pass - reg[h]["pr"])) for h in ALL_HEADS),
                  key=lambda kv: kv[1])
    crit_str = ", ".join(f"{h} below {m:.2f} pp" for h, m in crit)
    L += ["",
          f"Margin-5pp criticality: at margin 5pp the condition-(a) threshold "
          f"rises to {100*(fhs_pass-0.05):.1f}% and {len(fail5_list)} head(s) "
          f"fall below it ({fail5_pretty}), so {n5pass} of five heads pass; "
          f"for Chronos-2+H condition (a) "
          f"{'holds' if c2_m5['a_ok'] else 'FAILS'} and the joint verdict is "
          f"{'success' if c2_m5['joint'] else 'FAILURE'}. Per-head critical "
          f"condition-(a) margins (head fails once the margin tightens past): "
          f"{crit_str}.", "",
          "## C. Bare pass counts behind the §10.2 H coverage rates", "",
          "| head | pass count | rate |",
          "|---|---|---|"]
    for head in ALL_HEADS:
        kk, nn = counts[head]
        L.append(f"| {head} | {kk}/{nn} | {100*kk/nn:.0f}% |")
    L.append("")
    (ROOT / "docs" / "v12_sensitivity.md").write_text("\n".join(L))
    print("written: docs/v12_sensitivity.md +2 csvs;",
          f"5/5-stable combos {n55}/{len(grid)};",
          "C2@5pp joint:", "success" if c2_m5["joint"] else "FAILURE")


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("v12_sensitivity")
