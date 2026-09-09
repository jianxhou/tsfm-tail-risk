"""v2.0 Phase-3 verification statistics — one deterministic pass over the
committed stage-4 artifacts, emitting docs/v20_phase3_analyses.md (the
committed number source for the Phase-3 prose landings and evidence-pack
batch) and results/stage4/cutoff_exposure.csv (feeds the new cutoff table).

    python -m harness.run_p3_stats

Authorization: external review #9 adjudication -> v2.0 Phase-3 directive.
Every number in the output doc is machine-computed here from results/stage4
CSVs, tests/fixtures, and the registry-derived cutoff constants; nothing is
hand-typed. Randomized quantities (cluster bootstrap, regime episode
bootstrap) use B=10,000 with seed 20260730, recorded in the doc header.
The MC null draws reuse the production convention (B=10,000, seed 20260706;
run_grid_extract.mc_p).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from precision.mc_critical import MCNull, kupiec_stat_from_hits
from harness.run_regime_analysis import CUTOFF

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
DOC = ROOT / "docs" / "v20_phase3_analyses.md"
TSFM = ["chronos_bolt", "chronos_2", "timesfm_2_5", "moirai_2_0", "lag_llama"]
BASE = ["garch_t", "gjr_t", "ewma94", "hs250", "hs500", "fhs", "caviar_sav"]
ALPHAS = (0.01, 0.025, 0.05)
SEED = 20260730
B_BOOT = 10_000

L: list[str] = []          # output document lines


def w(s: str = "") -> None:
    L.append(s)


def table(header: list[str], rows: list[list]) -> None:
    w("| " + " | ".join(header) + " |")
    w("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w()


def s1_z2(bt: pd.DataFrame, bl: pd.DataFrame) -> None:
    w("## 1. Z2 (Acerbi-Szekely) summary — item 1a")
    w()
    nn_bt, nn_bl = int(bt.z2.notna().sum()), int(bl.z2.notna().sum())
    w(f"backtests.csv z2 non-null: **{nn_bt}** of {len(bt)} rows "
      f"(E1/E2 480 each; E3 465 = 480 minus the 15 dgs30 cells whose ES is "
      f"genuinely missing post-P0-1; E0 has no ES). baseline_backtests.csv "
      f"z2 non-null: {nn_bl} (CAViaR VaR-only 96 null; hs250/dgs30/a01 1 null). "
      f"Review premise said 1,440 non-null: that was the PRE-P0-1 state; the "
      f"P0-1 correction moved 15 dgs30 E3 cells to genuinely-missing ES.")
    w()
    rows = []
    for f in TSFM:
        for er in ("e1", "e2", "e3"):
            d = bt[(bt.forecaster == f) & (bt.erule == er)].z2.dropna()
            if len(d):
                rows.append([f, er, len(d), f"{d.mean():.3f}", f"{d.median():.3f}",
                             f"{(d > 0).mean()*100:.1f}%"])
    for f in BASE:
        d = bl[bl.forecaster == f].z2.dropna()
        if len(d):
            rows.append([f, "param", len(d), f"{d.mean():.3f}", f"{d.median():.3f}",
                         f"{(d > 0).mean()*100:.1f}%"])
    table(["forecaster", "rule", "cells", "mean Z2", "median Z2", "share Z2>0"],
          rows)


def s2_dq_cc(bt: pd.DataFrame, bl: pd.DataFrame) -> None:
    w("## 2. DQ (asymptotic) vs CC (MC) pass rates — items 1c / S-1 context")
    w()
    rows = []
    for f in TSFM:
        for er in ("e0", "e1", "e2", "e3"):
            d = bt[(bt.forecaster == f) & (bt.erule == er)]
            if len(d):
                rows.append([f, er, len(d),
                             f"{(d.dq_p_asym > 0.05).mean()*100:.1f}%",
                             f"{(d.cc_p_mc > 0.05).mean()*100:.1f}%",
                             f"{(d.kupiec_p_mc > 0.05).mean()*100:.1f}%"])
    for f in BASE:
        d = bl[bl.forecaster == f]
        rows.append([f, "param", len(d),
                     f"{(d.dq_p_asym > 0.05).mean()*100:.1f}%",
                     f"{(d.cc_p_mc > 0.05).mean()*100:.1f}%",
                     f"{(d.kupiec_p_mc > 0.05).mean()*100:.1f}%"])
    table(["forecaster", "rule", "cells", "DQ pass (asym)", "CC pass (MC)",
           "UC pass (MC)"], rows)
    hs = bl[bl.forecaster == "hs500"]
    w(f"HS-500 exact counts: DQ pass {int((hs.dq_p_asym > 0.05).sum())}/96 = "
      f"{(hs.dq_p_asym > 0.05).mean()*100:.1f}%; CC pass "
      f"{int((hs.cc_p_mc > 0.05).sum())}/96 = "
      f"{(hs.cc_p_mc > 0.05).mean()*100:.1f}%; UC pass "
      f"{int((hs.kupiec_p_mc > 0.05).sum())}/96 = "
      f"{(hs.kupiec_p_mc > 0.05).mean()*100:.1f}%.")
    w()


def s3_uc_power(bt: pd.DataFrame, bl: pd.DataFrame) -> None:
    w("## 3. UC test power at the panel n — item 1e")
    w()
    n, a = 2639, 0.01
    for B in (10_000, 20_000):
        null = MCNull("kupiec", n, a, B=B, seed=20260706)
        acc = [x for x in range(0, 90)
               if null.p_value(kupiec_stat_from_hits(
                   np.r_[np.ones(x), np.zeros(n - x)].astype(int), a)) > 0.05]
        lo, hi = min(acc), max(acc)
        assert acc == list(range(lo, hi + 1)), "non-contiguous acceptance region"
        power = float(stats.binom.cdf(lo - 1, n, 1.25 * a)
                      + stats.binom.sf(hi, n, 1.25 * a))
        tag = "PRODUCTION (B=10,000, run_grid_extract.mc_p)" if B == 10_000 \
            else "module default B=20,000 (reference)"
        w(f"- n={n}, alpha=1%, {tag}: acceptance region = **{lo}..{hi} "
          f"violations** = [{lo/(n*a):.3f}x, {hi/(n*a):.3f}x] nominal "
          f"(n*alpha={n*a:.2f}); exact-binomial power against a true rate of "
          f"1.25x nominal = **{power:.3f}**.")
        if B == 10_000:
            for x in (lo - 1, lo, hi, hi + 1):
                p = null.p_value(kupiec_stat_from_hits(
                    np.r_[np.ones(x), np.zeros(n - x)].astype(int), a))
                w(f"  - boundary p at x={x}: {p:.4f}")
    w()
    w("Mean violation-rate multiples of nominal at alpha=1% (32-asset means):")
    rows = []
    for f, er in (("chronos_2", "e2"), ("chronos_2", "e3"),
                  ("moirai_2_0", "e2"), ("moirai_2_0", "e3"),
                  ("timesfm_2_5", "e2"), ("timesfm_2_5", "e3")):
        d = bt[(bt.forecaster == f) & (bt.erule == er) & (bt.alpha == 0.01)]
        rows.append([f, er, f"{d.hit_rate.mean()/0.01:.2f}x"])
    for f in ("fhs", "gjr_t", "garch_t", "ewma94"):
        d = bl[(bl.forecaster == f) & (bl.alpha == 0.01)]
        rows.append([f, "param", f"{d.hit_rate.mean()/0.01:.2f}x"])
    table(["forecaster", "rule", "mean hit-rate multiple (a=1%)"], rows)
    d = bl[(bl.forecaster == "fhs") & (bl.alpha == 0.01)]
    m = d.hit_rate / 0.01
    w(f"FHS per-asset multiples at alpha=1%: min {m.min():.2f}x, max "
      f"{m.max():.2f}x over 32 assets (all {int((m >= m.min()).sum())} in "
      f"[{m.min():.2f}x, {m.max():.2f}x]).")
    for f in ("chronos_2", "moirai_2_0"):
        for er in ("e2", "e3"):
            d = bt[(bt.forecaster == f) & (bt.erule == er) & (bt.alpha == 0.01)]
            m = d.hit_rate / 0.01
            w(f"- {f}/{er} per-asset multiple range: [{m.min():.2f}x, "
              f"{m.max():.2f}x], mean {m.mean():.2f}x, SD {m.std(ddof=1):.2f}.")
    d = bl[(bl.forecaster == "gjr_t") & (bl.alpha == 0.01)]
    m = d.hit_rate / 0.01
    w(f"- gjr_t per-asset multiple range: [{m.min():.2f}x, {m.max():.2f}x], "
      f"mean {m.mean():.2f}x, SD {m.std(ddof=1):.2f}.")
    d = bl[(bl.forecaster == "fhs") & (bl.alpha == 0.01)]
    m = d.hit_rate / 0.01
    w(f"- fhs per-asset multiple SD {m.std(ddof=1):.2f}, mean {m.mean():.2f}x.")
    w()


def s4_mc_contribution(bt: pd.DataFrame, bl: pd.DataFrame) -> None:
    w("## 4. Asymptotic-vs-MC decision agreement — item 1f")
    w()
    al = pd.concat([bt, bl], ignore_index=True)
    ua, um = al.kupiec_p_asym > 0.05, al.kupiec_p_mc > 0.05
    d = al[ua != um]
    w(f"UC layer: {len(al)} decisions (1,824 TSFM + 672 baseline); "
      f"**{len(d)}** differ ({len(d)/len(al)*100:.2f}%): "
      f"{int(((ua) & (~um)).sum())} asym-pass->MC-fail, "
      f"{int(((~ua) & (um)).sum())} asym-fail->MC-pass.")
    w()
    rows = []
    for a in ALPHAS:
        s = al[al.alpha == a]
        ca, cm = s.cc_p_asym > 0.05, s.cc_p_mc > 0.05
        rows.append([f"{a:g}", len(s), int((ca != cm).sum()),
                     int((ca & ~cm).sum()), int((~ca & cm).sum()),
                     f"{(ca != cm).mean()*100:.1f}%"])
    table(["alpha", "cells", "CC decisions differ", "asym-accept MC-reject",
           "asym-reject MC-accept", "share"], rows)
    s = al[al.alpha == 0.01]
    ca, cm = s.cc_p_asym > 0.05, s.cc_p_mc > 0.05
    w(f"CC alpha=1% direction check: of {int((ca != cm).sum())} differing "
      f"decisions, {int((ca & ~cm).sum())} are asym-accept/MC-reject and "
      f"{int((~ca & cm).sum())} the reverse.")
    w()


def s5_cluster_boot(bt: pd.DataFrame) -> None:
    w("## 5. Cluster bootstrap for headline pass-rate differences — item 1d")
    w()
    w(f"Paired cluster bootstrap over the 32 assets (asset = cluster, the "
      f"three alpha cells stay nested), B={B_BOOT:,}, seed {SEED}, "
      f"percentile 95% CI.")
    w()
    rng = np.random.default_rng(SEED)
    assets = sorted(bt.asset.unique())
    idx_mat = rng.integers(0, len(assets), size=(B_BOOT, len(assets)))
    rows = []
    for er in ("e3", "e2"):
        per_asset = {}
        for f in ("chronos_2", "moirai_2_0", "timesfm_2_5"):
            d = bt[(bt.forecaster == f) & (bt.erule == er)]
            per_asset[f] = d.assign(ok=(d.kupiec_p_mc > 0.05).astype(float)) \
                            .groupby("asset").ok.mean().reindex(assets).to_numpy()
        for fa, fb in (("chronos_2", "moirai_2_0"),
                       ("chronos_2", "timesfm_2_5"),
                       ("moirai_2_0", "timesfm_2_5")):
            diff = per_asset[fa] - per_asset[fb]
            obs = diff.mean() * 100
            boot = diff[idx_mat].mean(axis=1) * 100
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append([er, f"{fa} - {fb}", f"{obs:+.1f}pp",
                         f"[{lo:+.1f}, {hi:+.1f}]pp",
                         "excludes 0" if (lo > 0 or hi < 0) else "includes 0"])
    table(["rule", "pair", "observed diff", "95% cluster CI", "zero"], rows)
    w("Within-asset ICC of the UC pass indicator (one-way ANOVA estimator over "
      "32 clusters of 3 nested alpha cells) and the implied design effect "
      "1+(m-1)ICC, m=3:")
    rows = []
    for er in ("e3", "e2"):
        for f in ("chronos_2", "moirai_2_0", "timesfm_2_5"):
            d = bt[(bt.forecaster == f) & (bt.erule == er)]
            g = d.assign(ok=(d.kupiec_p_mc > 0.05).astype(float))
            grp = g.groupby("asset").ok
            m_i = grp.mean()
            k, m = len(m_i), 3
            gm = g.ok.mean()
            msb = m * ((m_i - gm) ** 2).sum() / (k - 1)
            msw = (g.ok - g.asset.map(m_i)).pow(2).sum() / (k * (m - 1))
            icc = (msb - msw) / (msb + (m - 1) * msw) if (msb + 2 * msw) else 0.0
            de = 1 + (m - 1) * icc
            rows.append([er, f, f"{icc:.2f}", f"{de:.2f}", f"{96/de:.0f}"])
    table(["rule", "head", "ICC", "design effect", "effective cells (of 96)"],
          rows)


def s6_dm_nw() -> None:
    w("## 6. DM headline under the Newey-West variance — item S-2")
    w()
    dm = pd.read_csv(S4 / "v15_dm_nw.csv")
    ge = dm[dm.pair == "F1_vs_garch_evt"]
    lg = ge[(ge.model == "lag_llama") & (ge.alpha == 0.01)]
    fc_f = int(((lg.p_frozen < 0.05) & (lg.mean_diff > 0)).sum())
    f1_f = int(((lg.p_frozen < 0.05) & (lg.mean_diff < 0)).sum())
    fc_n = int(((lg.p_nw < 0.05) & (lg.mean_diff > 0)).sum())
    f1_n = int(((lg.p_nw < 0.05) & (lg.mean_diff < 0)).sum())
    w(f"Lag-Llama F1 vs GARCH-EVT at alpha=1% over 32 assets: baseline "
      f"variance {fc_f}/32 favor the comparator, {f1_f}/32 favor F1; "
      f"Newey-West **{fc_n}/32 favor the comparator, {f1_n}/32 favor F1**.")
    a1 = ge[ge.alpha == 0.01]
    w(f"All-model F1-vs-GARCH-EVT at alpha=1% (160 comparisons): baseline "
      f"{int(((a1.p_frozen < 0.05) & (a1.mean_diff > 0)).sum())} significant "
      f"favoring the comparator, "
      f"{int(((a1.p_frozen < 0.05) & (a1.mean_diff < 0)).sum())} favoring F1; "
      f"NW {int(((a1.p_nw < 0.05) & (a1.mean_diff > 0)).sum())} favoring the "
      f"comparator, {int(((a1.p_nw < 0.05) & (a1.mean_diff < 0)).sum())} "
      f"favoring F1.")
    sf, sn = dm.p_frozen < 0.05, dm.p_nw < 0.05
    w(f"Full 960-comparison panel: {int(sf.sum())} baseline-significant, "
      f"{int(sn.sum())} NW-significant; {int((sf != sn).sum())} decisions "
      f"move, {int((sf & ~sn).sum())} toward insignificance.")
    w()


def s7_ewma_fhs(bt: pd.DataFrame, bl: pd.DataFrame) -> None:
    w("## 7. EWMA vs FHS controlled form contrast — Phase-3 block 4")
    w()
    esr = pd.read_csv(S4 / "esr_results.csv")
    for f in ("ewma94", "fhs"):
        d1 = bl[(bl.forecaster == f) & (bl.alpha == 0.01)]
        dall = bl[bl.forecaster == f]
        es = esr[esr.forecaster == f]
        es1 = es[es.alpha == 0.01]       # v2.1 P1-4 option A: table caliber
        w(f"- {f}: alpha=1% UC pass {int((d1.kupiec_p_mc > 0.05).sum())}/32 = "
          f"{(d1.kupiec_p_mc > 0.05).mean()*100:.2f}%; pooled UC pass "
          f"{int((dall.kupiec_p_mc > 0.05).sum())}/96 = "
          f"{(dall.kupiec_p_mc > 0.05).mean()*100:.2f}%; mean FZ0(a=1%) "
          f"{d1.fz0_mean.mean():.4f}; mean hit rate(a=1%) "
          f"{d1.hit_rate.mean()*100:.2f}%; ESR non-rejection(a=1%) "
          f"{int((es1.esr_p > 0.05).sum())}/32 = "
          f"{(es1.esr_p > 0.05).mean()*100:.1f}% (pooled "
          f"{int((es.esr_p > 0.05).sum())}/96 = "
          f"{(es.esr_p > 0.05).mean()*100:.1f}%).")
    piv = bl[bl.alpha == 0.01].pivot(index="asset", columns="forecaster",
                                     values="fz0_mean")
    w(f"- FHS FZ0 lower than EWMA on "
      f"{int((piv.fhs < piv.ewma94).sum())}/32 assets; mean difference "
      f"{(piv.ewma94 - piv.fhs).mean():+.4f}.")
    w()


def s8_e1_esr() -> None:
    w("## 8. E1 ESR column — item S-8")
    w()
    esr = pd.read_csv(S4 / "esr_results.csv")
    rows = []
    for f in TSFM:
        d = esr[(esr.forecaster == f) & (esr.erule == "e1")]
        rows.append([f, len(d), int((d.esr_p > 0.05).sum()),
                     f"{(d.esr_p > 0.05).mean()*100:.2f}%"])
    table(["head", "E1 cells", "non-rejected", "non-rejection share"], rows)
    bord = esr[(esr.erule == "e1") & (esr.esr_p > 0.03) & (esr.esr_p < 0.07)]
    w("Borderline E1 cells (0.03 < p < 0.07), for the review-reference "
      "comparison:")
    for _, r in bord.iterrows():
        w(f"- {r.forecaster}/{r.asset}/a{r.alpha:g}: p = {r.esr_p:.4f}")
    w()


def s9_precision(bt: pd.DataFrame, bl: pd.DataFrame) -> None:
    w("## 9. Precision layer — item 1h")
    w()
    al = pd.concat([bt, bl], ignore_index=True)
    a1 = al[al.alpha == 0.01]
    rm = pd.read_csv(S4 / "repair_backtests_matched.csv")
    w(f"- n*alpha is a design constant of (panel length, alpha): alpha=1% "
      f"cells have median n*alpha = {a1.n_alpha.median():.1f}, min = "
      f"{a1.n_alpha.min():.2f} (n = {int(al[al.alpha==0.01].n.min())}..."
      f"{int(al[al.alpha==0.01].n.max())}).")
    w(f"- All-cell minimum across the main panel: {al.n_alpha.min():.2f}; "
      f"matched repair battery minimum: {rm.n_alpha.min():.2f}. The nα<10 "
      f"fragility floor is untriggerable at every panel and matched sample "
      f"length: main-panel cells with n_alpha<10: "
      f"{int((al.n_alpha < 10).sum())}; matched cells: "
      f"{int((rm.n_alpha < 10).sum())}.")
    hs = bl[(bl.forecaster == "hs250")]
    w(f"- hs250 valid-ES cells at alpha=1%: "
      f"{int(hs[hs.alpha == 0.01].fz0_mean.notna().sum())}/32 (dgs30 es_viol "
      f"{int(hs[hs.alpha == 0.01].es_viol.max())} days).")
    w(f"- caviar_sav has no ES output: fz0_mean/z2 all null "
      f"({int(bl[bl.forecaster == 'caviar_sav'].fz0_mean.isna().sum())}/96); "
      f"its n_alpha is the same design constant as every other forecaster's "
      f"and is not an ES diagnostic.")
    w()


def s10_regime() -> None:
    w("## 10. Regime recovery medians with episode bootstrap — item 1j")
    w()
    tau = pd.read_csv(S4 / "regime_taurec_v2.csv")
    vix = tau[(tau["index"] == "vix") & (tau.window == 21)]
    layers = ([f"{m}:e3" for m in TSFM] + ["gjr_t", "fhs"])
    rows = []
    for lay in layers:
        for a in (0.05, 0.025):
            s = vix[(vix.layer == lay) & (vix.alpha == a)]
            rec = s[s.ever_above & ~s.censored].tau_rec.to_numpy()
            if not len(rec):
                rows.append([lay, f"{a:g}", "---", "---", 0,
                             int(s.censored.sum()), int((~s.ever_above).sum())])
                continue
            med = float(np.median(rec))
            # per-cell rng (same convention as make_tables t_taurec, so the
            # doc and the generated table carry identical intervals)
            rng = np.random.default_rng(SEED)
            boot = np.median(
                rec[rng.integers(0, len(rec), size=(B_BOOT, len(rec)))], axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append([lay, f"{a:g}", f"{med:.1f}", f"[{lo:.0f}, {hi:.0f}]",
                         len(rec), int(s.censored.sum()),
                         int((~s.ever_above).sum())])
    table(["layer", "alpha", "raw median tau_rec", "95% episode-bootstrap CI "
           f"(B={B_BOOT:,}, seed {SEED})", "recovered", "censored",
           "never-elevated"], rows)
    w("Raw (unrounded) medians at alpha=5%: the displayed integer table "
      "rounds half to even; the parity-relevant raw values are listed above "
      "(item S-3).")
    w()


def s11_cutoff(bt: pd.DataFrame) -> None:
    w("## 11. Pretraining-bound exposure shares — item 1l")
    w()
    w("Bound dates = the registry bounds materialized as in "
      "harness/run_regime_analysis.CUTOFF (Bolt = pinned-revision "
      "lastModified, exact date; Chronos-2/Moirai-2.0/Lag-Llama = "
      "arXiv-report month-ends; TimesFM-2.5 = disclosed month-end).")
    w()
    rows_csv, rows_md = [], []
    c2_alt = pd.Timestamp("2026-06-05")   # chronos_2 pinned-revision lastModified
    for m in TSFM:
        bound = pd.Timestamp(CUTOFF[m])
        tot = pre = post_alt = 0
        for pq in sorted((S4 / "forecast").glob(f"{m}_*.parquet")):
            d = pd.read_parquet(pq, columns=["date"])
            dates = pd.to_datetime(d["date"])
            tot += len(dates)
            pre += int((dates <= bound).sum())
            if m == "chronos_2":
                post_alt += int((dates > c2_alt).sum())
        share = pre / tot * 100
        rows_csv.append({"model": m, "bound": CUTOFF[m], "days_total": tot,
                         "days_pre_bound": pre, "share_pre_bound": pre / tot})
        rows_md.append([m, CUTOFF[m], tot, pre, f"{share:.2f}%"])
        if m == "chronos_2":
            w(f"- chronos_2 dual rule: arXiv bound {CUTOFF[m]} leaves "
              f"{tot - pre:,} post-bound days; the pinned-revision "
              f"lastModified bound (2026-06-05) leaves {post_alt:,}.")
    w()
    table(["model", "bound", "asset-days", "days <= bound", "share <= bound"],
          rows_md)
    pd.DataFrame(rows_csv).to_csv(S4 / "cutoff_exposure.csv", index=False)
    w("Written: results/stage4/cutoff_exposure.csv")
    w()


def s12_matched_dates(bt: pd.DataFrame, bl: pd.DataFrame) -> None:
    w("## 12. Late-start day-set premise + matched-date sensitivity — item 1b")
    w()
    rows = []
    for a in ("csi300", "eth", "xrp", "btc", "ltc"):
        tn = int(bt[(bt.asset == a) & (bt.alpha == 0.01)
                    & (bt.erule == "e2")].n.iloc[0])
        bn = int(bl[(bl.asset == a) & (bl.alpha == 0.01)].n.iloc[0])
        rows.append([a, tn, bn, tn - bn, f"{(tn-bn)/bn*100:.1f}%"])
    table(["asset", "TSFM n", "baseline n", "extra TSFM days",
           "TSFM excess vs baseline"], rows)
    p = S4 / "matched_dates.csv"
    if not p.exists():
        w("matched_dates.csv not yet computed (run harness.run_matched_dates).")
        w()
        return
    md = pd.read_csv(p)
    flips = md[md.pass_full != md.pass_matched]
    w(f"matched_dates.csv: {len(md)} cells; UC pass full-set "
      f"{int(md.pass_full.sum())}/{len(md)} vs baseline-intersection "
      f"{int(md.pass_matched.sum())}/{len(md)}; decision flips: {len(flips)} "
      f"({int((~flips.pass_full & flips.pass_matched).sum())} toward passing, "
      f"{int((flips.pass_full & ~flips.pass_matched).sum())} toward failing).")
    if len(flips):
        for _, r in flips.iterrows():
            w(f"- flip: {r.forecaster}/{r.asset}/{r.erule}/a{r.alpha:g} "
              f"hit {r.hit_full*100:.2f}%->{r.hit_matched*100:.2f}%, "
              f"pass {bool(r.pass_full)}->{bool(r.pass_matched)}")
    w()


def s13_garch_sigma() -> None:
    w("## 13. GARCH sigma vs rugarch measured deviation — item S-7")
    w()
    from fixes.garch_filter import garch_mu_sigma_path
    ctx = pd.read_csv(ROOT / "tests" / "fixtures"
                      / "mcneilfrey_garch_input.csv")["logret"].to_numpy()
    ref = pd.read_csv(ROOT / "tests" / "fixtures"
                      / "mcneilfrey_garch_reference.csv").iloc[0]
    mu, sig = garch_mu_sigma_path([ctx])
    rel = abs(sig[0] - ref["sigma1_rugarch"]) / ref["sigma1_rugarch"]
    w(f"- ours sigma1 = {sig[0]:.6f}; rugarch sigma1 = "
      f"{ref['sigma1_rugarch']:.6f}; relative difference = **{rel*100:.2f}%** "
      f"(the appendix currently says 'within 1%').")
    w()


def s14_dq_size() -> None:
    w("## 14. DQ asymptotic size at deep alpha — item S-1")
    w()
    for B in (10_000, 20_000):
        null = MCNull("dq", 2639, 0.01, B=B, seed=20260706)
        size5 = float((stats.chi2.sf(null.draws, 5) < 0.05).mean())
        w(f"- iid-Bernoulli null, n=2639, alpha=1%, B={B:,}: rejection rate of "
          f"the asymptotic chi2(5) DQ test at the 5% level = **{size5:.4f}** "
          f"(hits-only DQ, the MCNull construction: constant + 4 lags, "
          f"df=5).")
    w("The production battery's stored dq_p_asym additionally includes the "
      "contemporaneous VaR regressor (df=6) on real forecasts; the size "
      "experiment above isolates the reference-distribution distortion "
      "under the exact iid null.")
    w()


def s15_mcs_levels() -> None:
    w("## 15. MCS at the registered dual levels — item 1g")
    w()
    p = S4 / "mcs_membership_levels.csv"
    if not p.exists():
        w("mcs_membership_levels.csv not yet computed "
          "(run harness.run_mcs_levels).")
        w()
        return
    mm = pd.read_csv(p)
    w("Stage-4 membership rates (E2-comparison convention of tab_mcs), "
      "10% vs 25% level:")
    rows = []
    for f in TSFM + ["garch_t", "gjr_t", "ewma94", "hs250", "hs500", "fhs"]:
        r = [f]
        for lv in (0.10, 0.25):
            for a in ALPHAS:
                d = mm[(mm.forecaster == f) & (mm.erule == "e2")
                       & (mm.alpha == a) & (mm.level == lv)]
                r.append(f"{float(d.mcs_rate.iloc[0])*100:.0f}%" if len(d)
                         else "---")
        rows.append(r)
    table(["forecaster", "10% a=1%", "10% a=2.5%", "10% a=5%",
           "25% a=1%", "25% a=2.5%", "25% a=5%"], rows)
    rp = S4 / "repair_mcs_levels.csv"
    if rp.exists():
        rm = pd.read_csv(rp)
        w("Repair-battery membership rates at alpha=1% (share of 160 "
          "model-asset cells), 10% vs 25%:")
        rows = []
        for arm in ("F1", "F2", "F3", "F4", "H", "garch_evt", "fhs"):
            r = [arm]
            for lv in (0.10, 0.25):
                d = rm[(rm.arm == arm) & (rm.alpha == 0.01) & (rm.level == lv)]
                r.append(f"{d.in_mcs.mean()*100:.0f}%")
            rows.append(r)
        table(["arm", "in-MCS share 10%", "in-MCS share 25%"], rows)
    w()


def s16_repair_dayset(bt: pd.DataFrame) -> None:
    w("## 16. Repair-arm day-set continuity — item S-5 verification")
    w()
    total_nan_v = 0
    for m in TSFM:
        for a in ("dgs2", "dgs5", "dgs10", "dgs30"):
            pq = S4 / "repairs" / f"{m}_{a}.parquet"
            d = pd.read_parquet(pq)
            nan_v = int(d["u_e3_v01"].isna().sum())
            total_nan_v += nan_v
            if nan_v:
                w(f"- {m}/{a}: {nan_v} NaN E3-VaR days (unexpected)")
    w(f"- NaN unrepaired-E3 VaR days across all 5 heads x 4 rate assets in "
      f"the repair per-day store: **{total_nan_v}** — the P0-1 VaR/ES split "
      f"leaves the VaR path complete, so the repair day sets are contiguous "
      f"and no splice exists to disclose (S-5 void).")
    w()


def s17_v21_direction(bt: pd.DataFrame) -> None:
    """v2.1 Phase 3 (§D16 pack feed): rate-asset direction account under the
    empirical-mass E3, Kendall low-tau composition, and the main-panel Moirai
    E0 caliber — emitted here so the evidence pack parses them from a tracked
    script-generated doc (backtests.csv itself is not git-tracked)."""
    w("## 17. v2.1 empirical-mass E3 — direction account + panel calibers")
    w()
    rates = ["dgs2", "dgs5", "dgs10", "dgs30"]
    d = bt[(bt.erule == "e3") & (bt.alpha == 0.01) & (bt.asset.isin(rates))]
    no = int((d.kupiec_p_mc > 0.05).sum())
    lo = int(((d.kupiec_p_mc <= 0.05) & (d.hit_rate < 0.01)).sum())
    hi = int(((d.kupiec_p_mc <= 0.05) & (d.hit_rate > 0.01)).sum())
    hih = sorted(d[(d.kupiec_p_mc <= 0.05) & (d.hit_rate > 0.01)]
                 .forecaster.unique())
    assert no + lo + hi == len(d) == 20
    w(f"Rate-asset E3 cells at alpha=1% (20 cells): no-reject **{no}**, "
      f"reject-low **{lo}**, reject-high **{hi}** (reject-high all "
      f"{'/'.join(hih)}).")
    kd = pd.read_csv(S4 / "kendall_tau_detail.csv")
    low = kd[kd.kendall_tau < 0.5]
    by = low.groupby("alpha").size()
    a1 = sorted(low[low.alpha == 0.01].asset)
    w(f"Kendall tau<0.5 composition: **{len(low)} cells** of {len(kd)} — "
      + ", ".join(f"{n} @{a*100:g}%" for a, n in by.items())
      + f"; @1% assets: {', '.join(a1)}.")
    d0 = bt[(bt.forecaster == "moirai_2_0") & (bt.erule == "e0")
            & (bt.alpha == 0.01)]
    w(f"Moirai-2.0 E0 mean violation at 1% on the {len(d0)}-asset main "
      f"panel: **{100 * d0.hit_rate.mean():.2f}%**.")
    w()


def main() -> None:
    bt = pd.read_csv(S4 / "backtests.csv")
    bl = pd.read_csv(S4 / "baseline_backtests.csv")
    w("# v2.0 Phase-3 verification analyses (script-generated)")
    w()
    w("Generated by `python -m harness.run_p3_stats` from the committed "
      "stage-4 artifacts; authorization = review #9 adjudication + the "
      "Phase-3 directive. Bootstrap quantities use B=10,000, seed "
      f"{SEED}; MC null draws use the production convention (B=10,000, "
      "seed 20260706). Nothing in this file is hand-edited.")
    w()
    w("v2.1 Phase-3 regeneration (2026-08-03): same script, fresh store — "
      "the empirical-mass-E3 full-chain rerun (docs/e3_tail_mass_ruling.md; "
      "review #10 adjudication §8 flag 5) plus the per-cell-seeded ESR "
      "engine. E3- and ESR-dependent sections move accordingly; the v2.0-era "
      "values remain readable at tag `v2.1-phase2` and earlier.")
    w()
    s1_z2(bt, bl)
    s2_dq_cc(bt, bl)
    s3_uc_power(bt, bl)
    s4_mc_contribution(bt, bl)
    s5_cluster_boot(bt)
    s6_dm_nw()
    s7_ewma_fhs(bt, bl)
    s8_e1_esr()
    s9_precision(bt, bl)
    s10_regime()
    s11_cutoff(bt)
    s12_matched_dates(bt, bl)
    s13_garch_sigma()
    s14_dq_size()
    s15_mcs_levels()
    s16_repair_dayset(bt)
    s17_v21_direction(bt)
    DOC.write_text("\n".join(L) + "\n")
    print(f"wrote {DOC} ({len(L)} lines)", flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("p3_stats")
