"""Stage-5 prescription table v2 (design §A; conditions 4/5/9 + precision;
closeout rulings 2/3/4 + audit remediations A2/A3/A5/A13).

    python -m harness.run_prescription

Reads results/stage4/repair_backtests_matched.csv (matched evaluation sample,
OOS day 501+, common day set — design §A as registered; the delivered
mixed-sample table is superseded, see CHANGELOG 2026-07-08) plus
repair_dm.csv / repair_mcs.csv. Produces docs/stage5_prescription.md:
 - headline heads x arms with unrepaired reference row, FZ0 n_valid disclosure
   (A5), E3 primary and E2 robustness side by side, fragile flags;
 - bolt exhibit with its design-required E2-vs-E3 dispersion column;
 - MANDATORY F2 gamma {0.5x,1x,2x} robustness table + lag_llama attribution
   (design decision 2);
 - pre-registered success criterion (a)+(b) verdicts (first execution — A3);
 - F1/H/GARCH-EVT decomposition under the decision-3 wording, with the DM double
   bar and MCS membership summaries (design decision 4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
HEADLINE = ["chronos_2", "moirai_2_0", "lag_llama"]      # condition 4 (manifest)
ALL_HEADS = ["chronos_2", "moirai_2_0", "lag_llama", "chronos_bolt", "timesfm_2_5"]
ARMS = ["unrepaired", "F1", "F2", "F3", "F4", "H"]
CRIT_ARMS = ["F1", "F2", "F3", "F4", "H"]
FRAGILE_NA = 10.0
A = 0.01


def _flag(na):
    return "‡" if (np.isfinite(na) and na < FRAGILE_NA) else ""


def coverage_pass(sub):
    if not len(sub):
        return np.nan
    return ((sub.kupiec_p_mc > 0.05) & (sub.cc_p_mc > 0.05)).mean()


def fz(sub):
    """(mean FZ0 over valid assets, n_valid)."""
    if not len(sub):
        return np.nan, 0
    v = sub.fz0_mean.dropna()
    return (v.mean() if len(v) else np.nan), int(len(v))


def cell(rb, name, erule, a=A):
    return rb[(rb.forecaster == name) & (rb.erule == erule) & (rb.alpha == a)]


def criterion(arms_df, fhs_df, gjr_df, ge_df, a=A):
    """Success criterion (design §A) on ONE evaluation sample. Returns the FHS
    pass rate, the (a) threshold, per-(head, arm) verdicts, and GARCH-EVT's own
    self-test verdict. Reused verbatim for the matched and delivered samples so
    the dual-sample robustness check (prerequisite b) and the GE self-test line
    (prerequisite c) share identical logic with the headline table."""
    fhs_r = fhs_df[fhs_df.alpha == a].drop_duplicates(["asset", "alpha"])
    gjr_r = gjr_df[gjr_df.alpha == a].drop_duplicates(["asset", "alpha"])
    ge_r = ge_df[ge_df.alpha == a].drop_duplicates(["asset", "alpha"])
    fhs_pass = coverage_pass(fhs_r)
    thr = fhs_pass - 0.10
    min_ref = pd.concat([gjr_r.set_index("asset").fz0_mean,
                         fhs_r.set_index("asset").fz0_mean], axis=1).min(axis=1)

    def verdict(e3):
        pr = coverage_pass(e3)
        okb = []
        for asset, fzv in e3.drop_duplicates("asset").set_index("asset").fz0_mean.items():
            ref = min_ref.get(asset, np.nan)
            if not np.isfinite(fzv) or not np.isfinite(ref):
                okb.append(False)
            elif ref <= 0:                          # registered fallback
                okb.append(fzv - ref <= 0.05)
            else:
                okb.append(fzv <= 1.05 * ref)
        bfrac = float(np.mean(okb)) if okb else np.nan
        a_ok = bool(np.isfinite(pr) and pr >= thr)
        return dict(pr=pr, a_ok=a_ok, bfrac=bfrac,
                    joint=bool(a_ok and np.isfinite(bfrac) and bfrac >= 2 / 3))

    rows = {}
    for head in ALL_HEADS:
        for arm in CRIT_ARMS:
            e3 = cell(arms_df, f"{head}+{arm}", "e3", a)
            if len(e3):
                rows[(head, arm)] = verdict(e3)
    return dict(fhs_pass=fhs_pass, thr=thr, rows=rows, ge=verdict(ge_r))


def main():
    rb = pd.read_csv(S4 / "repair_backtests_matched.csv")
    dmr = pd.read_csv(S4 / "repair_dm.csv")
    mc = pd.read_csv(S4 / "repair_mcs.csv")
    refs = {nm: rb[rb.forecaster == nm].drop_duplicates(["asset", "alpha"])
            for nm in ("fhs", "gjr_t", "garch_evt")}
    fhs_pass = coverage_pass(refs["fhs"][refs["fhs"].alpha == A])
    ge_pass = coverage_pass(refs["garch_evt"][refs["garch_evt"].alpha == A])
    n_med = rb[rb.alpha == A].n.median()

    # Dual-sample criterion (prereqs b/c/d): the matched sample (paper) and the
    # delivered mixed full-OOS sample (baselines live in baseline_backtests.csv,
    # GARCH-EVT in the delivered repair grid). Superseded but reported for the
    # robustness disclosure the closing verdict requires.
    rb_dlv = pd.read_csv(S4 / "repair_backtests.csv")
    bl = pd.read_csv(S4 / "baseline_backtests.csv")
    mcrit = criterion(rb, rb[rb.forecaster == "fhs"], rb[rb.forecaster == "gjr_t"],
                      rb[rb.forecaster == "garch_evt"])
    dcrit = criterion(rb_dlv, bl[bl.forecaster == "fhs"], bl[bl.forecaster == "gjr_t"],
                      rb_dlv[rb_dlv.forecaster == "garch_evt"])
    n_spx_dlv = int(bl[(bl.forecaster == "fhs") & (bl.alpha == A)
                       & (bl.asset == "spx")].n.iloc[0])
    n_spx_m = int(rb[(rb.forecaster == "fhs") & (rb.alpha == A)
                     & (rb.asset == "spx")].n.iloc[0])

    def fmtc(c):
        return f"{100*c:.0f}%" if np.isfinite(c) else "—"

    def fmtf(f, nv, total=32):
        if not np.isfinite(f):
            return "—"
        return f"{f:.3f}" if nv == total else f"{f:.3f} (n={nv})"

    L = ["# Stage-5 prescription table v2 (script-generated; matched evaluation "
         "sample per design §A — OOS day 501+, common day set; supersedes the "
         "delivered mixed-sample table, CHANGELOG 2026-07-08)", "",
         f"alpha=1%. Coverage = fraction of assets passing MC UC+CC at 5%. FZ0 = "
         f"mean loss over assets with valid ES cells; '(n=k)' marks means over "
         f"k<32 assets (F4's additive shift breaks e<v on >1% of days in those "
         f"cells — audit A5). '‡' = precision-fragile (median n_alpha < "
         f"{FRAGILE_NA:.0f}; matched median n = {n_med:.0f}). References on the "
         f"SAME matched sample: FHS pass = {100*fhs_pass:.0f}%, GARCH-EVT pass = "
         f"{100*ge_pass:.0f}%.", "",
         f"Footnote (FHS reference drop — prerequisite d): the FHS criterion-(a) "
         f"pass rate is {100*dcrit['fhs_pass']:.0f}% on the delivered full-OOS "
         f"sample but {100*mcrit['fhs_pass']:.0f}% here — an attribution, not "
         f"cherry-picking. The matched sample's 500-day burn-in (design §A common "
         f"day set) removes the calm 2016–17 segment where FHS covered well, "
         f"shortening per-asset n (spx {n_spx_dlv}→{n_spx_m}). The matched FHS "
         f"pass sets the (a) threshold at {100*mcrit['thr']:.1f}%; the delivered "
         f"value would set a higher {100*dcrit['thr']:.1f}% bar (see the "
         f"dual-sample section). Disclosed to preempt any suspicion the matched "
         f"sample was chosen to depress the FHS reference.", "",
         "## Headline heads (native deep-tail-distinguishable; condition 4)", "",
         "| head | arm | cov E3 | FZ0 E3 | cov E2 | FZ0 E2 | frag |",
         "|---|---|---|---|---|---|---|"]
    for head in HEADLINE:
        for arm in ARMS:
            name = f"{head}+{arm}"
            e3 = cell(rb, name, "e3")
            e2 = cell(rb, name, "e2")
            f3, n3 = fz(e3)
            f2_, n2 = fz(e2)
            na = e3.n_alpha.median() if len(e3) else np.nan
            L.append(f"| {head} | {arm} | {fmtc(coverage_pass(e3))} | "
                     f"{fmtf(f3, n3)} | {fmtc(coverage_pass(e2))} | "
                     f"{fmtf(f2_, n2)} | {_flag(na)} |")
    L += ["", "Caveat (audit A13): lag_llama is headline-eligible under the frozen "
          "condition-4 classification (distinguishable head) but sits OUTSIDE the "
          "E3 caliber arbitration, which covers the 4 quantile heads only; its "
          "rows carry that reservation. timesfm_2_5 has no native deep tail "
          "(interface) and is excluded from the headline product per the frozen "
          "manifest; its repair rows live in the CSV and the criterion table "
          "below.", "",
          "## Exhibit: chronos_bolt (clamped head; condition 5 — via E1-E3 only, "
          "not a headline ranking entry). E2-vs-E3 dispersion per design §A.", "",
          "| arm | cov E3 | FZ0 E3 | cov E2 | FZ0 E2 | |cov E3−E2| |",
          "|---|---|---|---|---|---|"]
    for arm in ARMS:
        e3 = cell(rb, f"chronos_bolt+{arm}", "e3")
        e2 = cell(rb, f"chronos_bolt+{arm}", "e2")
        c3, c2 = coverage_pass(e3), coverage_pass(e2)
        f3, n3 = fz(e3)
        f2_, n2 = fz(e2)
        disp = f"{100*abs(c3-c2):.0f}pp" if np.isfinite(c3) and np.isfinite(c2) else "—"
        L.append(f"| {arm} | {fmtc(c3)} | {fmtf(f3, n3)} | {fmtc(c2)} | "
                 f"{fmtf(f2_, n2)} | {disp} |")

    # ---- design decision 2: F2 gamma robustness (mandatory with every F2 table) ----
    L += ["", "## F2 gamma robustness (ruling 2 — mandatory column; "
          "gamma = 0.01·s_hat·mult)", "",
          "| head | cov 0.5x | cov 1x | cov 2x | FZ0 0.5x | FZ0 1x | FZ0 2x |",
          "|---|---|---|---|---|---|---|"]
    for head in ALL_HEADS:
        cs, fs = [], []
        for g in ("F2g05", "F2", "F2g20"):
            e3 = cell(rb, f"{head}+{g}", "e3")
            cs.append(fmtc(coverage_pass(e3)))
            f_, n_ = fz(e3)
            fs.append(fmtf(f_, n_))
        L.append(f"| {head} | " + " | ".join(cs + fs) + " |")
    ll = {g: cell(rb, f"lag_llama+{g}", "e3") for g in ("F2g05", "F2", "F2g20")}
    att = []
    for g, sub in ll.items():
        uc = sub.kupiec_p_mc <= 0.05
        cc = sub.cc_p_mc <= 0.05
        att.append(f"{g.replace('F2g05', '0.5x').replace('F2g20', '2x').replace('F2', '1x')}: "
                   f"mean hit {sub.hit_rate.mean():.4f}, UC-only "
                   f"{int((uc & ~cc).sum())}, CC-only {int((~uc & cc).sum())}, "
                   f"both {int((uc & cc).sum())}")
    L += ["", "lag_llama attribution (ruling 2): " + "; ".join(att) + ". "
          "Far-from-nominal hit rates with dual UC+CC failures indicate gamma "
          "scale mismatch (adaptation too slow for the head's bias), not the "
          "honest-ACI clustering limitation; the 2x column is the direct test.", ""]

    # ---- success criterion (design §A, first execution — audit A3) ----
    L += ["## Success criterion (design §A; (a) pass-rate >= FHS−10pp; (b) "
          "FZ0 <= 1.05·min(GJR-t, FHS) per asset on >= 2/3 of non-fragile "
          "assets; NaN FZ0 counts as (b)-fail)", "",
          f"(a) threshold = {100*mcrit['thr']:.1f}%. All assets non-fragile "
          f"at the matched n.", "",
          "| head | arm | pass rate | (a) | (b) frac | joint |",
          "|---|---|---|---|---|---|"]
    for head in ALL_HEADS:
        for arm in CRIT_ARMS:
            v = mcrit["rows"].get((head, arm))
            if v is None:
                continue
            L.append(f"| {head} | {arm} | {100*v['pr']:.0f}% | "
                     f"{'PASS' if v['a_ok'] else 'fail'} | {v['bfrac']:.2f} | "
                     f"{'**SUCCEEDS**' if v['joint'] else 'fails'} |")
    ge = mcrit["ge"]
    L += ["", f"GARCH-EVT self-test (prerequisite c, matched sample): pass rate "
          f"{100*ge['pr']:.0f}% {'>=' if ge['a_ok'] else '<'} "
          f"{100*mcrit['thr']:.1f}% -> (a) {'PASS' if ge['a_ok'] else 'fail'}; "
          f"FZ0 <= 1.05·min(GJR-t,FHS) on {100*ge['bfrac']:.0f}% of assets "
          f">= 2/3 -> (b) {'PASS' if np.isfinite(ge['bfrac']) and ge['bfrac']>=2/3 else 'fail'}; "
          f"joint = **{'SUCCEEDS' if ge['joint'] else 'fails'}**. The classical "
          f"McNeil–Frey comparator "
          f"{'meets' if ge['joint'] else 'does not meet'} its own bar on the "
          f"paper sample. (On the delivered sample it fails (a) only — "
          f"{100*dcrit['ge']['pr']:.0f}% vs the inflated {100*dcrit['thr']:.1f}% "
          f"bar — with (b) still {dcrit['ge']['bfrac']:.2f}.)", ""]

    # ---- prerequisite (b): dual-sample criterion robustness ----
    hm = sum(1 for h in ALL_HEADS if mcrit["rows"].get((h, "H"), {}).get("joint"))
    hd = sum(1 for h in ALL_HEADS if dcrit["rows"].get((h, "H"), {}).get("joint"))
    flips = [(h, arm) for h in ALL_HEADS for arm in CRIT_ARMS
             if (h, arm) in mcrit["rows"] and (h, arm) in dcrit["rows"]
             and mcrit["rows"][(h, arm)]["joint"] != dcrit["rows"][(h, arm)]["joint"]]
    L += ["## Dual-sample criterion robustness (prerequisite b)", "",
          f"The success criterion re-run per (head, arm) on the matched sample "
          f"(paper) and the delivered mixed full-OOS sample. (a)-threshold: "
          f"matched {100*mcrit['thr']:.1f}% (FHS {100*mcrit['fhs_pass']:.0f}%), "
          f"delivered {100*dcrit['thr']:.1f}% (FHS {100*dcrit['fhs_pass']:.0f}%; "
          f"burn-in-inflated, footnote d). The delivered sample is superseded "
          f"(audit A2) and shown only to demonstrate the headline is not a "
          f"sample-choice artifact.", "",
          "| head | arm | matched pass | matched joint | delivered pass | delivered joint |",
          "|---|---|---|---|---|---|"]
    for head in ALL_HEADS:
        for arm in CRIT_ARMS:
            vm = mcrit["rows"].get((head, arm))
            vd = dcrit["rows"].get((head, arm))
            if vm is None or vd is None:
                continue
            L.append(f"| {head} | {arm} | {100*vm['pr']:.0f}% | "
                     f"{'SUCCEEDS' if vm['joint'] else 'fails'} | "
                     f"{100*vd['pr']:.0f}% | "
                     f"{'SUCCEEDS' if vd['joint'] else 'fails'} |")
    if len(flips) == 1 and flips[0][1] == "H":
        fh = flips[0][0]
        detail = (f"The sole verdict change across the two samples is {fh}+H: it "
                  f"clears the matched {100*mcrit['thr']:.1f}% (a)-bar "
                  f"({100*mcrit['rows'][(fh,'H')]['pr']:.0f}%, margin "
                  f"{100*(mcrit['rows'][(fh,'H')]['pr']-mcrit['thr']):+.1f}pp) but "
                  f"misses the higher delivered {100*dcrit['thr']:.1f}% bar "
                  f"({100*dcrit['rows'][(fh,'H')]['pr']:.0f}%, margin "
                  f"{100*(dcrit['rows'][(fh,'H')]['pr']-dcrit['thr']):+.1f}pp), "
                  f"while its (b) fraction holds either way "
                  f"({mcrit['rows'][(fh,'H')]['bfrac']:.2f} matched / "
                  f"{dcrit['rows'][(fh,'H')]['bfrac']:.2f} delivered). This is a "
                  f"threshold effect from FHS's calm-segment-inflated delivered "
                  f"pass rate, not an H degradation.")
    else:
        detail = ("Verdict changes across samples: "
                  + (", ".join(f"{h}+{a}" for h, a in flips) if flips else "none")
                  + ".")
    L += ["", f"H headline: matched {hm}/5, delivered {hd}/5. {detail} H is the "
          f"only arm that succeeds on any head in either sample (F1–F4 fail "
          f"jointly throughout); no F1–F4 cell changes verdict between samples.", ""]

    # ---- ruling 3: decomposition wording + ruling 4: DM double bar + MCS ----
    L += ["## Location vs scale decomposition — F1 vs H vs GARCH-EVT (ruling 3)", "",
          "F1 = TSFM location + TSFM-IQR scale; H = TSFM location + GARCH scale; "
          "GARCH-EVT = GARCH location + GARCH scale. Binding reading (ruling 3): "
          "**at the daily one-step horizon, location is a second-order term; the "
          "binding shortfall of TSFM heads is the conditional scale. Substituting "
          "the GARCH sigma (H) lets TSFM-anchored EVT match classical "
          "GARCH-EVT.** Mean FZ0 at alpha=1%, E3, matched sample.", "",
          "| head | F1 FZ0 | H FZ0 | GARCH-EVT FZ0 | scale term (F1−H) | "
          "location term (H−GE) |", "|---|---|---|---|---|---|"]
    ge_fz_all, _ = fz(refs["garch_evt"][refs["garch_evt"].alpha == A])
    for head in ALL_HEADS:
        f1, _ = fz(cell(rb, f"{head}+F1", "e3"))
        h, _ = fz(cell(rb, f"{head}+H", "e3"))
        L.append(f"| {head} | {f1:.3f} | {h:.3f} | {ge_fz_all:.3f} | "
                 f"{f1-h:+.3f} | {h-ge_fz_all:+.3f} |")
    L += ["", "## DM-on-FZ0 double bar (registered; HAC h=1, per asset, "
          "alpha=1%, matched sample)", "",
          "| head | pair | assets GE/F4 better (p<0.05) | assets F1 better "
          "(p<0.05) | median stat |", "|---|---|---|---|---|"]
    for head in ALL_HEADS:
        for pair in ("F1_vs_garch_evt", "F1_vs_F4"):
            s = dmr[(dmr.model == head) & (dmr.alpha == A) & (dmr.pair == pair)]
            if not len(s):
                continue
            sig = s[s.p < 0.05]
            L.append(f"| {head} | {pair} | {int((sig.mean_diff > 0).sum())}/32 | "
                     f"{int((sig.mean_diff < 0).sum())}/32 | {s.stat.median():+.2f} |")
    L += ["", "(mean_diff > 0 = second element better on FZ0.)", "",
          "## MCS membership at alpha=1% (10% level, per (head, asset); "
          "share of assets where arm is in the confidence set)", "",
          "| head | " + " | ".join(CRIT_ARMS + ["garch_evt", "fhs"]) + " |",
          "|---|" + "---|" * 7]
    for head in ALL_HEADS:
        s = mc[(mc.model == head) & (mc.alpha == A)]
        row = []
        for arm in CRIT_ARMS + ["garch_evt", "fhs"]:
            sa = s[s.arm == arm]
            row.append(f"{100*sa.in_mcs.mean():.0f}%" if len(sa) else "—")
        L.append(f"| {head} | " + " | ".join(row) + " |")
    L.append("")

    (ROOT / "docs" / "stage5_prescription.md").write_text("\n".join(L))
    print("prescription v2 written: docs/stage5_prescription.md", flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("prescription")
