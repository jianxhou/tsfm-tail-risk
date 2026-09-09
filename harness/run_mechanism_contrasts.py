"""Step-5 contrast readings (signer directive 2026-07-17): all contrasts on the
SAME 12-asset mechanism subset. Main-panel heads (chronos_bolt, chronos_2,
moirai_2_0) are RE-SLICED from results/stage4/backtests.csv to the 12 assets —
32-asset means are never compared against 12-asset means. Mechanism heads come
from results/mechanism/backtests.csv (identical battery + anchors).

    python -m harness.run_mechanism_contrasts

Calibers replicate the main panel exactly:
- UC pooled pass = (kupiec_p_mc > 0.05).mean() over asset x alpha cells
  (make_tables.py t_var_audit caliber), here 12 x 3 = 36 cells per (head, rule).
- E0@1% mean hit / E0-E2 spread = mean hit_rate over the 12 assets at alpha=1%
  (docs/stage4_final_table.md §4 spread caliber).
Output: docs/mechanism_contrasts.md (script-generated; feeds the evidence pack).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ASSETS12 = ["brent", "gold", "btc", "eth", "hsi", "spx",
            "aapl", "nvda", "audusd", "gbpusd", "dgs10", "dgs2"]
CHRONOS = ["chronos_bolt", "chronos_base", "chronos_2"]
MOIRAI = ["moirai_1_1", "moirai_2_0"]
ERULES = ["e0", "e1", "e2", "e3"]
ALPHAS = (0.01, 0.025, 0.05)


def load() -> pd.DataFrame:
    a = pd.read_csv(ROOT / "results/stage4/backtests.csv")
    b = pd.read_csv(ROOT / "results/mechanism/backtests.csv")
    d = pd.concat([a, b], ignore_index=True)
    d = d[d.asset.isin(ASSETS12)]
    return d


def uc_pool(d, m, e):
    s = d[(d.forecaster == m) & (d.erule == e)]
    if not len(s):
        return np.nan, 0
    return float((s.kupiec_p_mc > 0.05).mean()), len(s)


def cc_pool(d, m, e):
    s = d[(d.forecaster == m) & (d.erule == e)]
    return float((s.cc_p_mc > 0.05).mean()) if len(s) else np.nan


def hit1(d, m, e):
    s = d[(d.forecaster == m) & (d.erule == e) & (d.alpha == 0.01)]
    return float(s.hit_rate.mean()) if len(s) else np.nan


def fz0_e2_1(d, m):
    s = d[(d.forecaster == m) & (d.erule == "e2") & (d.alpha == 0.01)]
    v = s.fz0_mean.dropna()
    return (float(v.mean()), int(len(v)))


def qs5_e0(d, m):
    s = d[(d.forecaster == m) & (d.erule == "e0") & (d.alpha == 0.05)]
    v = s.qs_mean.dropna()
    return float(v.mean()) if len(v) else np.nan


def main() -> None:
    d = load()
    for m in CHRONOS + MOIRAI:
        n = d[d.forecaster == m].asset.nunique()
        assert n == 12, f"{m}: {n} assets on the subset (expected 12)"

    L = ["# Mechanism contrast readings (script-generated; do not hand-edit)",
         "",
         "All numbers on the SAME 12-asset subset (brent gold btc eth hsi spx aapl "
         "nvda audusd gbpusd dgs10 dgs2). Main-panel heads re-sliced from "
         "results/stage4/backtests.csv; mechanism heads from results/mechanism/"
         "backtests.csv (identical frozen battery + per-asset anchors). "
         "UC/CC pooled pass = share of 36 asset x alpha cells with MC-calibrated "
         "p > 0.05; hit rates at alpha=1% are means over the 12 assets.",
         "", "## 1. Chronos lineage three-way (same vendor + data lineage, three output forms)",
         "", "| head | form | E0@1% mean hit | E0-E2 spread @1% (pp) | UC pooled E0 | E1 | E2 | E3 |",
         "|---|---|---|---|---|---|---|---|"]
    FORM = {"chronos_bolt": "bounded quantile grid (clamp)",
            "chronos_base": "token-quantized sampling",
            "chronos_2": "deep-quantile head",
            "moirai_1_1": "mixture (sampled)", "moirai_2_0": "quantile head"}
    for m in CHRONOS:
        h0, h2 = hit1(d, m, "e0"), hit1(d, m, "e2")
        cells = [f"{uc_pool(d, m, e)[0]*100:.0f}%" for e in ERULES]
        L.append(f"| {m} | {FORM[m]} | {h0*100:.1f}% | {(h0-h2)*100:.1f} | "
                 + " | ".join(cells) + " |")
    L += ["", "## 2. Moirai migration (mixture 1.1 -> quantile 2.0, same family)",
          "", "| metric | moirai_1_1 | moirai_2_0 | direction |", "|---|---|---|---|"]
    rows = []
    for lab, fn, fmt, better_low in (
            ("E0@1% mean hit (nominal 1%)", lambda m: hit1(d, m, "e0"), "{:.2%}", None),
            ("UC pooled pass E0", lambda m: uc_pool(d, m, "e0")[0], "{:.0%}", False),
            ("UC pooled pass E2", lambda m: uc_pool(d, m, "e2")[0], "{:.0%}", False),
            ("UC pooled pass E3", lambda m: uc_pool(d, m, "e3")[0], "{:.0%}", False),
            ("CC pooled pass E3", lambda m: cc_pool(d, m, "e3"), "{:.0%}", False),
            ("mean FZ0 (E2, 1%)", lambda m: fz0_e2_1(d, m)[0], "{:.3f}", True),
            ("mean qs (E0, 5%)", lambda m: qs5_e0(d, m), "{:.4f}", True)):
        v1, v2 = fn("moirai_1_1"), fn("moirai_2_0")
        if lab.startswith("E0@1%"):
            direction = "closer to nominal: " + ("2.0" if abs(v2-0.01) < abs(v1-0.01) else "1.1")
        elif better_low is None:
            direction = "-"
        else:
            b = (v1 < v2) if better_low else (v1 > v2)
            direction = "1.1 better" if b else "2.0 better"
            if np.isclose(v1, v2, rtol=0.005):
                direction = "~equal"
        rows.append((lab, fmt.format(v1), fmt.format(v2), direction))
        L.append(f"| {lab} | {fmt.format(v1)} | {fmt.format(v2)} | {direction} |")
    n1 = fz0_e2_1(d, "moirai_1_1")[1]
    L += ["", f"(FZ0 cells with valid ES: moirai_1_1 {n1}/12, "
          f"moirai_2_0 {fz0_e2_1(d, 'moirai_2_0')[1]}/12 assets at E2/1%.)"]
    # per-alpha E0 hits for the migration paragraph
    L += ["", "## 3. Per-alpha E0 mean hit rates (12 assets)", "",
          "| head | 1% | 2.5% | 5% |", "|---|---|---|---|"]
    for m in CHRONOS + MOIRAI:
        vals = [d[(d.forecaster == m) & (d.erule == "e0") & (d.alpha == a)].hit_rate.mean()
                for a in ALPHAS]
        L.append(f"| {m} | " + " | ".join(f"{v*100:.1f}%" for v in vals) + " |")
    L += ["", "## 4. Chronos-base total failure count (all rules)",
          "", f"- chronos_base UC pooled pass across ALL rules: "
          f"{int((d[(d.forecaster=='chronos_base')].kupiec_p_mc > 0.05).sum())}"
          f"/{len(d[d.forecaster=='chronos_base'])} cells (0/36 per rule).",
          f"- moirai_1_1 UC pooled pass across ALL rules: "
          f"{int((d[(d.forecaster=='moirai_1_1')].kupiec_p_mc > 0.05).sum())}"
          f"/{len(d[d.forecaster=='moirai_1_1'])} cells.", ""]
    (ROOT / "docs" / "mechanism_contrasts.md").write_text("\n".join(L))
    print("wrote docs/mechanism_contrasts.md")


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("mechanism_contrasts")
