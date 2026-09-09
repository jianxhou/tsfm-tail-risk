"""Emit the paper's figures into paper/draft/figures/*.pdf.

Sources (hard rule 5): results/stage4/{backtests.csv, regime_curves.csv,
vol_strata.csv, repair_backtests_matched.csv}. Aggregates that also appear in
paper/evidence_pack.md are asserted before the figure is drawn.

v2.1 assert overhaul (IMPLEMENTED Phase 3; review #10 §0 principle 3 + §6
fix 3; adjudication §8):
  * ASSERTIONS MUST NOT PIN FROZEN INTERMEDIATE CSVs. The v2.0 strata asserts
    (assert_pct('chronos_2 Q1', ..., 88) etc.) checked values against the
    SAME frozen vol_strata.csv the figure was drawn from — a stale input
    passed its own assert and "figure asserts passed" had zero detection
    power (the P1-0 stale-product family shipped exactly this way).
  * fig_strata: every paper-consumed stratum aggregate is RE-DERIVED from the
    per-day upstream products (repairs/*.parquet + the data layer's RV
    series, kupiec + MC null) and compared to the figure's CSV value.
  * fig_repair: (1) upstream-fingerprint check — the matched CSV must
    postdate every repairs/fhs/gjr per-day parquet it aggregates; (2) matched
    n + per-asset H fz0_mean + the FHS reference are RE-DERIVED from the
    per-day parquets (BURN + delivered common-day mask + battery good-ES
    rule) and compared to the CSV; (3) protective invariant (P1-2):
    H_mean >= GE_mean per head — H matches, does not exceed, GARCH-EVT.
    The GARCH-EVT per-day series is not persisted (re-deriving it means
    refitting EVT per window — too heavy for the build loop), so GE rides
    legs (1) and (3), the sanctioned fallback.
  No assert target is hand-anchored to a frozen number anymore.

Style: Okabe-Ito CVD-safe palette, fixed model->color map everywhere (color
follows the entity); econometric baselines in gray/black dashed (class by
line style = secondary encoding); single y-axis per panel; recessive grid.

Run:  python paper/figures/make_figures.py

Distribution copies (v2.1 Phase 4): the re-derivation legs need the per-day
upstream layer (results/stage4/repairs/*.parquet, ~376 MB, containing the
return series) and the raw data store — neither is distributed in the
replication packages. `--skip-upstream-rederive` skips EXACTLY those legs,
loudly, and refuses to run when the upstream layer IS present (so the flag
cannot weaken the repo build); in the packages, integrity of the aggregated
CSVs is carried by MANIFEST.csv sha256 pins plus the repo-side re-derivation
record. The CSV-internal protective invariant (H_mean >= GE_mean) always runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import COLOR, FIG, NAME, ROOT, S4, TSFM, assert_val, check

sys.path.insert(0, str(ROOT))   # backtests/data/harness imports (re-derivation)

# v2.1 Phase 4 (package mode; module docstring): legitimate ONLY where the
# per-day upstream layer is absent — the guard below hard-refuses otherwise.
SKIP_REDERIVE = "--skip-upstream-rederive" in sys.argv
if SKIP_REDERIVE and any((S4 / "repairs").glob("*.parquet")):
    raise SystemExit(
        "--skip-upstream-rederive REFUSED: results/stage4/repairs/*.parquet "
        "is present, so the full re-derivation battery must run (the flag "
        "exists only for distribution copies that do not carry the per-day "
        "upstream layer)")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

W = 5.5  # text-width inches for elsarticle 12pt preprint


# ------------------------------------------------ F1: E-rule sensitivity @1%
# v1.6 item C11(b): deterministic PDF metadata so regenerated figures are
# byte-identical (CreationDate pinned; content streams unchanged).
PDF_META = {"CreationDate": None}

# v1.9.1 C3: global model->marker map (fig_erule / fig_regime / fig_strata);
# fig_repair keeps its marker=repair-arm semantics. Colors unchanged
# (annotation/marker layer only; no data, axes, palette, or size change).
MODEL_MARK = {"chronos_bolt": "o", "chronos_2": "s", "timesfm_2_5": "^",
              "moirai_2_0": "D", "lag_llama": "v"}


def fig_erule():
    bt = pd.read_csv(S4 / "backtests.csv")
    d = bt[bt.alpha == 0.01]
    hit = d.groupby(["forecaster", "erule"]).hit_rate.mean()

    assert_val("bolt E0 hit", hit[("chronos_bolt", "e0")], 0.127, tol=5e-4)
    assert_val("lag_llama E0 hit", hit[("lag_llama", "e0")], 0.038, tol=5e-4)
    assert_val("chronos_2 E0 hit", hit[("chronos_2", "e0")], 0.012, tol=5e-4)
    assert_val("moirai E0 hit", hit[("moirai_2_0", "e0")], 0.014, tol=5e-4)

    fig, ax = plt.subplots(figsize=(W, 2.9))
    erules = ["e0", "e1", "e2", "e3"]
    xs = np.arange(len(erules))
    for m in TSFM:
        ys = [hit.get((m, e), np.nan) * 100 for e in erules]
        # v2.0 Phase-3 P2-7: E1 is the deliberately misspecified Gaussian
        # control — it stays OFF the connecting line (isolated open marker),
        # so the polyline runs E0 -> E2 -> E3 only (NaN E0 drops its segment).
        ax.plot([0, 2, 3], [ys[0], ys[2], ys[3]], marker=MODEL_MARK[m], ms=4,
                lw=1.4, color=COLOR[m], label=NAME[m])
        ax.plot([1], [ys[1]], marker=MODEL_MARK[m], ms=4, color=COLOR[m],
                mfc="none", ls="none")
    ax.axhline(1.0, color="#888888", lw=0.9, ls=":")
    ax.annotate("nominal 1%", xy=(0.05, 0.62), fontsize=7, color="#666666")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10])
    ax.set_yticklabels(["0.5", "1", "2", "5", "10"])
    ax.set_xticks(xs)
    ax.set_xticklabels(["E0 (native)", "E1 (normal)", "E2 ($t$)", "E3 (GPD)"])
    ax.set_ylabel("mean violation rate (%), $\\alpha=1\\%$, log scale")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_erule.pdf", metadata=PDF_META)
    plt.close(fig)
    print("wrote fig_erule.pdf")


# ------------------------------------- F2: event-time regime recovery curves
def fig_regime():
    cv = pd.read_csv(S4 / "regime_curves.csv")
    eps = [("vix_ep17", "2020 COVID onset (2020-02-24)"),
           ("vix_ep23", "2025-04 episode (2025-04-03)")]
    layers = [f"{m}:e3" for m in TSFM] + ["gjr_t", "fhs"]

    d17 = cv[(cv.episode == "vix_ep17") & (cv.layer == "lag_llama:e3")
             & (cv.alpha == 0.05) & (cv.t == 21)]
    assert_val("lag_llama COVID t=21", float(d17.trail21.iloc[0]), 0.395, tol=5e-4)

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), sharey=True)
    for ax, (ep, title) in zip(axes, eps):
        sub = cv[(cv.episode == ep) & (cv.alpha == 0.05)]
        # v2.1.3 (review #12 P1): the calm-stratum band is LAYER-SPECIFIC.
        # The old code shaded 0-to-band21.iloc[0] — a single layer's
        # threshold (chronos_bolt:e3, 0.07788) presented as a common null
        # zone, so a curve could dip into the shading while still above its
        # own threshold (Moirai-2.0's COVID path never crosses ITS band and
        # tab:taurec censors it, yet the plot showed a day-46 "recovery").
        # The gray strip now spans the per-layer thresholds of the seven
        # plotted layers; recovery classification uses each layer's own band.
        plotted_bands = {}
        for lay in layers:
            s = sub[sub.layer == lay].sort_values("t")
            plotted_bands[lay] = float(s.band21.dropna().iloc[0])
            m = lay.split(":")[0]
            # v1.9.1 C2: sparse phase-offset markers on the five TSFM curves
            # (markevery ~7, size 4); baselines stay dashed, marker-free.
            style = dict(color=COLOR[m], lw=1.3, marker=MODEL_MARK[m], ms=4,
                         markevery=(TSFM.index(m) % 7, 7)) \
                if lay.endswith(":e3") else \
                dict(color=COLOR[m], lw=1.1, ls="--")
            ax.plot(s.t, s.trail21, label=NAME[m], **style)
        bmin, bmax = min(plotted_bands.values()), max(plotted_bands.values())
        # assert: strip endpoints == an independent re-derivation from
        # regime_curves.csv (fresh read, groupby over the same seven layers)
        ref = pd.read_csv(S4 / "regime_curves.csv")
        rb = ref[(ref.episode == ep) & (ref.alpha == 0.05)
                 & ref.layer.isin(layers)].groupby("layer").band21.first()
        check(f"fig_regime strip endpoints {ep}",
              len(rb) == len(layers)
              and abs(bmin - float(rb.min())) < 1e-12
              and abs(bmax - float(rb.max())) < 1e-12,
              f"plotted [{bmin:.5f}, {bmax:.5f}] != rederived "
              f"[{float(rb.min()):.5f}, {float(rb.max()):.5f}] over {len(rb)} layers")
        print(f"[strip] {ep}: band21 min {bmin:.5f} "
              f"({min(plotted_bands, key=plotted_bands.get)}) / max {bmax:.5f} "
              f"({max(plotted_bands, key=plotted_bands.get)})")
        ax.axhspan(bmin, bmax, color="#e4e4e4", zorder=0)
        ax.axhline(0.05, color="#888888", lw=0.8, ls=":")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("event day $t$")
        ax.set_xlim(21, 60)
        ax.set_ylim(0, 0.42)
    axes[0].set_ylabel("trailing-21 violation rate, $\\alpha=5\\%$")
    # v1.9.1 C2 label zone, re-measured v2.1.3: left panel t in [22,32] has
    # all seven curves >= 0.0873, above the strip top 0.07788; the label sits
    # below the 0.05 nominal line in the curve-free zone (the right panel is
    # crossed by curves at strip height everywhere).
    axes[0].annotate("calm-stratum thresholds", xy=(22.5, 0.030), fontsize=7,
                     color="#777777",
                     bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "fig_regime.pdf", bbox_inches="tight", metadata=PDF_META)
    plt.close(fig)
    print("wrote fig_regime.pdf")


# ------------------------------------------- v2.1 Phase-3 re-derivation legs
def _manifest_assets():
    import yaml
    return yaml.safe_load((S4 / "run_manifest.yaml").read_text())["assets"]


def _strata_rederive(need):
    """{(layer, 'Qk'): MC-UC pass rate over assets} re-derived from the
    per-day upstream products (repairs/*.parquet + data-layer RV), mirroring
    harness/run_vol_strata.py's registered rules (alpha=5%, E3; trailing-21d
    RV ending t-1; full-OOS quartile cutoffs; n>=60 cells; series>=400)."""
    from backtests.var_tests import kupiec
    from data.load import load_series
    from harness.run_grid_extract import mc_p
    assets = _manifest_assets()
    rv_cache = {}

    def rv_of(a):
        if a not in rv_cache:
            df = load_series(a)
            x = df[df.attrs.get("column", "logret")].to_numpy()
            rv_cache[a] = pd.Series(x).rolling(21).std().shift(1).to_numpy()
        return rv_cache[a]

    by_layer = {}
    for lay, q in need:
        by_layer.setdefault(lay, set()).add(q)
    out = {}
    for lay, qs in by_layer.items():
        model = lay.split(":")[0]
        acc = {q: [] for q in qs}
        for a in assets:
            p = S4 / "repairs" / f"{model}_{a}.parquet"
            if not p.exists():
                continue
            d = pd.read_parquet(p, columns=["t", "y", "u_e3_v05", "u_e3_e05"])
            d = d[np.isfinite(d["u_e3_v05"])].reset_index(drop=True)
            if len(d) < 400:
                continue
            y, v = d["y"].to_numpy(), d["u_e3_v05"].to_numpy()
            rv = rv_of(a)[d["t"].to_numpy().astype(int)]
            okrv = np.isfinite(rv)
            cut = np.nanquantile(rv[okrv], [0.25, 0.5, 0.75])
            stratum = np.digitize(rv, cut)
            for q in qs:
                m = okrv & (stratum == int(q[1]) - 1)
                if int(m.sum()) < 60:
                    continue
                uc = kupiec(y[m], v[m], 0.05)
                acc[q].append(mc_p("kupiec", uc["stat"], int(m.sum()), 0.05) > 0.05)
        for q in qs:
            out[(lay, q)] = float(np.mean(acc[q]))
    return out


def _repair_rederive(d1):
    """Re-derive the matched battery's +H and FHS aggregates from the per-day
    parquets: BURN=500, common-day mask over the delivered arm columns
    (x4/zl/cl excluded, run_repair_series.matched_rows), battery good-ES rule.
    Returns (h_means, fhs_mean, n_mismatches) and cross-checks every
    (model, asset) matched n against the CSV."""
    from backtests.scores import fz0 as fz0_daily
    assets = _manifest_assets()
    h_means, fhs_vals, n_bad = {}, [], 0
    for model in TSFM:
        vals = []
        for a in assets:
            pf = pd.read_parquet(S4 / "repairs" / f"{model}_{a}.parquet")
            fb = pd.read_parquet(S4 / "baselines" / f"fhs_{a}.parquet",
                                 columns=["t", "v01", "e01"]).rename(
                columns={"v01": "fhs_v01", "e01": "fhs_e01"})
            gb = pd.read_parquet(S4 / "baselines" / f"gjr_t_{a}.parquet",
                                 columns=["t", "v01", "e01"]).rename(
                columns={"v01": "gjr_v01", "e01": "gjr_e01"})
            m = pf.merge(fb, on="t").merge(gb, on="t").iloc[500:]
            vcols = [c for c in m.columns
                     if "_v" in c and not c.startswith(("x4_", "zl_", "cl_"))]
            m = m[np.isfinite(m[vcols].to_numpy()).all(axis=1)]
            row = d1[(d1.forecaster == f"{model}+H") & (d1.asset == a)
                     & (d1.erule == "e3")]
            if not len(row) or int(row.n.iloc[0]) != len(m):
                n_bad += 1
                continue
            y = m["y"].to_numpy()
            for pre, sink in (("h", vals), ("fhs", fhs_vals)):
                v, e = m[f"{pre}_v01"].to_numpy(), m[f"{pre}_e01"].to_numpy()
                ok = np.isfinite(v)
                ys, vs_, es = y[ok], v[ok], e[ok]
                good = (es < vs_) & (es < 0) & np.isfinite(es)
                if good.mean() > 0.99:
                    sink.append(float(fz0_daily(ys[good], vs_[good], es[good],
                                                0.01).mean()))
        h_means[model] = float(np.mean(vals))
    return h_means, float(np.mean(fhs_vals)), n_bad


# --------------------------------------------------- F3: volatility strata
def fig_strata():
    vs = pd.read_csv(S4 / "vol_strata.csv")
    layers = [f"{m}:e3" for m in TSFM] + ["gjr_t", "fhs"]
    pas = vs.groupby(["layer", "stratum"]).uc_pass.mean()

    # v2.1 Phase-3 assert overhaul: every paper-consumed stratum aggregate is
    # compared to its re-derivation from the per-day upstream series — a
    # stale vol_strata.csv can no longer pass its own assert.
    need = {("chronos_2:e3", "Q1"), ("chronos_2:e3", "Q4"),
            ("chronos_bolt:e3", "Q1"), ("moirai_2_0:e3", "Q2"),
            ("moirai_2_0:e3", "Q3"), ("lag_llama:e3", "Q1"),
            ("lag_llama:e3", "Q4"), ("timesfm_2_5:e3", "Q4")}
    if SKIP_REDERIVE:
        print("[package mode] strata upstream re-derivation SKIPPED — the "
              "per-day repairs layer + raw data store are not distributed; "
              "integrity here = MANIFEST.csv sha256 pins + the repo-side "
              "re-derivation record (v2.1 Phase-3/4 execution reports)")
    else:
        fresh = _strata_rederive(need)
        for (lay, q), val in sorted(fresh.items()):
            check(f"strata {lay} {q} vs upstream re-derivation",
                  abs(float(pas[(lay, q)]) - val) < 1e-9,
                  f"csv {float(pas[(lay, q)]):.6f} != rederived {val:.6f}")
            print(f"[rederive] strata {lay} {q}: {val:.4f}")

    fig, ax = plt.subplots(figsize=(W, 2.9))
    qs = ["Q1", "Q2", "Q3", "Q4"]
    xs = np.arange(4)
    for lay in layers:
        m = lay.split(":")[0]
        ys = [pas[(lay, q)] * 100 for q in qs]
        style = dict(color=COLOR[m], lw=1.4, marker=MODEL_MARK[m], ms=4) \
            if lay.endswith(":e3") else \
            dict(color=COLOR[m], lw=1.2, marker="s", ms=3.5, ls="--")
        ax.plot(xs, ys, label=NAME[m], **style)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Q1 (low vol)", "Q2", "Q3", "Q4 (high vol)"])
    ax.set_ylabel("MC-UC pass rate over assets (%), $\\alpha=5\\%$, E3")
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "fig_strata.pdf", bbox_inches="tight", metadata=PDF_META)
    plt.close(fig)
    print("wrote fig_strata.pdf")


# ------------------------------------------- F4: repair FZ0 (matched sample)
def fig_repair():
    rb = pd.read_csv(S4 / "repair_backtests_matched.csv")
    d = rb[(rb.alpha == 0.01)]
    arms = ["unrepaired", "F1", "F2", "F3", "F4", "H"]

    def fz(forecaster, erule):
        s = d[(d.forecaster == forecaster) & (d.erule == erule)].fz0_mean.dropna()
        return s.mean()

    ge = fz("garch_evt", d[d.forecaster == "garch_evt"].erule.iloc[0])
    fhs = fz("fhs", d[d.forecaster == "fhs"].erule.iloc[0])

    # v2.1 Phase-3 assert overhaul (module docstring): fingerprint leg,
    # re-derivation leg, protective-invariant leg — no pinned frozen values.
    # Phase 4: in package mode only the CSV-internal invariant leg runs.
    if SKIP_REDERIVE:
        print("[package mode] repair upstream re-derivation SKIPPED — the "
              "per-day repairs layer is not distributed; integrity here = "
              "MANIFEST.csv sha256 pins + the repo-side re-derivation record")
    else:
        up = (list((S4 / "repairs").glob("*.parquet"))
              + [p for p in (S4 / "baselines").glob("*.parquet")
                 if p.name.startswith(("fhs_", "gjr_t_"))])
        newest = max(p.stat().st_mtime for p in up)
        check("matched CSV postdates its per-day upstream parquets",
              (S4 / "repair_backtests_matched.csv").stat().st_mtime >= newest,
              "stale repair_backtests_matched.csv — rerun the §2.7 chain in order")
        h_red, fhs_red, n_bad = _repair_rederive(d)
        check("matched n re-derivation", n_bad == 0,
              f"{n_bad} (model,asset) matched-n mismatches vs per-day parquets")
        check("FHS FZ0 vs upstream re-derivation", abs(fhs - fhs_red) < 1e-7,
              f"csv {fhs:.6f} != rederived {fhs_red:.6f}")
        for m in TSFM:
            hm = fz(f"{m}+H", "e3")
            check(f"H FZ0 {m} vs upstream re-derivation",
                  abs(hm - h_red[m]) < 1e-7,
                  f"csv {hm:.6f} != rederived {h_red[m]:.6f}")
            print(f"[rederive] H FZ0 {m}: {h_red[m]:.4f}")
        print(f"[rederive] GE FZ0 matched {ge:.4f}; FHS {fhs_red:.4f}")
    for m in TSFM:
        hm = fz(f"{m}+H", "e3")
        check(f"H FZ0 {m} >= GARCH-EVT (matches, does not exceed)",
              hm >= ge, f"H {hm:.6f} < GE {ge:.6f}")

    # grouped dot plot (signer directive 2026-07-17): one row per model, one
    # marker per arm, vertical GARCH-EVT / FHS reference lines. Replaces the
    # 30-bar chart; values and asserts unchanged.
    fig, ax = plt.subplots(figsize=(W, 2.9))
    ys = np.arange(len(TSFM))[::-1]
    MARK = {"unrepaired": ("o", "none"), "F1": ("s", "full"), "F2": ("^", "full"),
            "F3": ("v", "full"), "F4": ("D", "full"), "H": ("*", "full")}
    SIZE = {"unrepaired": 5.5, "F1": 5.5, "F2": 5.5, "F3": 5.5, "F4": 5.0, "H": 10.0}
    for arm in arms:
        vals = [fz(f"{m}+{arm}", "e3") for m in TSFM]
        mk, fill = MARK[arm]
        first = True
        for y, v, m in zip(ys, vals, TSFM):
            ax.plot(v, y, marker=mk, markersize=SIZE[arm],
                    color=COLOR[m], fillstyle=fill,
                    markeredgecolor=COLOR[m], linestyle="none",
                    label=arm if first else None, zorder=3)
            first = False
    for y in ys:
        ax.axhline(y, color="#DDDDDD", lw=0.5, zorder=1)
    ax.axvline(ge, color="#000000", lw=1.0, ls="--", zorder=2)
    # v1.9.1 C1: white-backed reference-line labels (no dash-through).
    ax.annotate(f"GARCH-EVT {ge:.3f}", xy=(ge, ys[0] + 0.55), fontsize=7,
                color="#333333", ha="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.axvline(fhs, color="#666666", lw=0.9, ls=":", zorder=2)
    ax.annotate(f"FHS {fhs:.3f}", xy=(fhs, ys[-1] - 0.55), fontsize=7,
                color="#666666", ha="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_yticks(ys)
    ax.set_yticklabels([NAME[m] for m in TSFM])
    ax.set_ylim(ys[-1] - 0.9, ys[0] + 0.9)
    ax.set_xlim(1.80, 2.35)
    ax.set_xlabel("mean FZ0, $\\alpha=1\\%$, E3, matched sample")
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, frameon=False, ncol=6, loc="upper center",
                    bbox_to_anchor=(0.5, 1.24), title="arm (marker)",
                    title_fontsize=7, handletextpad=0.2, columnspacing=0.9)
    for h in leg.legend_handles:
        h.set_color("#444444")
        h.set_markeredgecolor("#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "fig_repair.pdf", bbox_inches="tight", metadata=PDF_META)
    plt.close(fig)
    print("wrote fig_repair.pdf")


if __name__ == "__main__":
    fig_erule()
    fig_regime()
    fig_strata()
    fig_repair()
    print("all figure asserts passed")
    # G6 freshness stamp (v2.1.1, review #11): loud no-op inside package
    # copies (no .git; package mode cannot see the per-day upstream layer)
    from harness.freshness import stamp_safe
    stamp_safe("make_figures")
