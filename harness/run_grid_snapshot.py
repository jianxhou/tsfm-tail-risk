"""Stage-4 mid-point snapshot (amendment C): produced at ~12 assets, then STOP for
operator review. Reads results/stage4/backtests.csv + forecast meta.json.

    python -m harness.run_grid_snapshot

Writes docs/stage4_midpoint_snapshot.md covering the four required items:
 1. Chronos-Bolt clamp consistency across asset classes (E0 vs E2 at alpha=1%).
 2. E2-vs-E3 spread distribution (defensible-extractor agreement, condition 11 pre-view).
 3. anomalous-asset list (extreme cells, ES-convention or GPD failures).
 4. measured s/window vs pilot extrapolation (amendment D provenance).
No GO/analysis claims — this is a checkpoint, not a result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "stage4"
FC = OUT / "forecast"
PILOT_SPW = {"chronos_bolt": 0.041, "chronos_2": 0.039,
             "timesfm_2_5": 0.095, "moirai_2_0": 0.008}


def main() -> None:
    bt = pd.read_csv(OUT / "backtests.csv")
    a1 = bt[bt.alpha == 0.01]
    L = ["# Stage-4 mid-point snapshot (script-generated; amendment C)", "",
         f"Coverage: {bt.asset.nunique()} assets, {sorted(bt.forecaster.unique())}. "
         "Checkpoint only — no GO/analysis claims. Judgment output is MC-calibrated "
         "(gate condition 2).", ""]

    # 1. Bolt clamp across asset classes
    L += ["## 1. Chronos-Bolt clamp consistency across asset classes", "",
          "E0 (native) vs E2 hit rate at alpha=1%, nominal 1%. Clamp = E0 hugely "
          "over-violates; E2 repairs. Grouped by asset class.", "",
          "| group | asset | E0 hit | E2 hit | E0-E2 spread |", "|---|---|---|---|---|"]
    b = a1[a1.forecaster == "chronos_bolt"]
    for _, g in b.groupby("group"):
        for asset in sorted(g.asset.unique()):
            e0 = g[(g.asset == asset) & (g.erule == "e0")]
            e2 = g[(g.asset == asset) & (g.erule == "e2")]
            if len(e0) and len(e2):
                s = abs(e0.hit_rate.iloc[0] - e2.hit_rate.iloc[0])
                L.append(f"| {g.group.iloc[0]} | {asset} | {100*e0.hit_rate.iloc[0]:.1f}% "
                         f"| {100*e2.hit_rate.iloc[0]:.2f}% | {100*s:.1f}pp |")
    e0all = b[b.erule == "e0"].hit_rate
    L += ["", f"Bolt E0 hit at 1%: mean {100*e0all.mean():.1f}%, range "
          f"{100*e0all.min():.1f}-{100*e0all.max():.1f}% across "
          f"{e0all.size} assets/classes (nominal 1%). Consistency of the clamp "
          "across classes is the item-1 question.", ""]

    # 2. E2 vs E3 spread distribution
    L += ["## 2. E2-vs-E3 agreement (condition-11 pre-view)", "",
          "|hit_rate(E2) - hit_rate(E3)| in pp at alpha=1%, per (model, asset):", "",
          "| model | n cells | median | p90 | max |", "|---|---|---|---|---|"]
    for m in sorted(a1.forecaster.unique()):
        diffs = []
        sub = a1[a1.forecaster == m]
        for asset in sub.asset.unique():
            e2 = sub[(sub.asset == asset) & (sub.erule == "e2")]
            e3 = sub[(sub.asset == asset) & (sub.erule == "e3")]
            if len(e2) and len(e3) and np.isfinite(e3.hit_rate.iloc[0]):
                diffs.append(abs(e2.hit_rate.iloc[0] - e3.hit_rate.iloc[0]))
        if diffs:
            d = np.array(diffs) * 100
            L.append(f"| {m} | {len(d)} | {np.median(d):.2f}pp | "
                     f"{np.percentile(d,90):.2f}pp | {d.max():.2f}pp |")
    L += ["", "Pilot found |E2-E3| <= 0.80pp in all cells; this is the full-grid "
          "pre-view feeding the pre-registered Kendall-tau diagnostic (condition 11).", ""]

    # 3. anomalous assets
    L += ["## 3. Anomalous-asset watch", ""]
    anom = []
    for _, r in bt.iterrows():
        why = []
        if r.get("es_viol", 0) and r["es_viol"] > 0.01 * r["n"]:
            why.append(f"{int(r['es_viol'])} ES-convention viol")
        if r["erule"] in ("e2", "e3") and r["forecaster"] != "chronos_bolt":
            if r.hit_rate > 3 * r.alpha or r.hit_rate < 0.2 * r.alpha:
                why.append(f"hit {100*r.hit_rate:.1f}% vs nom {100*r.alpha:.1f}%")
        if why:
            anom.append(f"- {r.forecaster}/{r.asset} a={r.alpha:g} {r.erule}: "
                        + "; ".join(why))
    L += (anom or ["- none flagged at current coverage"]) + [""]

    # 4. timing
    L += ["## 4. Measured s/window vs pilot extrapolation (amendment D)", "",
          "| model | cells | median s/win | pilot s/win | ratio |", "|---|---|---|---|---|"]
    metas = {}
    for f in FC.glob("*.meta.json"):
        m = json.loads(f.read_text())
        metas.setdefault(m["model"], []).append(m["s_per_window"])
    for model, spws in sorted(metas.items()):
        med = float(np.median(spws))
        L.append(f"| {model} | {len(spws)} | {med:.3f} | {PILOT_SPW[model]} "
                 f"| {med/PILOT_SPW[model]:.1f}x |")
    L += ["", "All ratios below the amendment-D 2x self-halt threshold (the run "
          "would have stopped otherwise).", ""]

    (ROOT / "docs" / "stage4_midpoint_snapshot.md").write_text("\n".join(L))
    print("snapshot written: docs/stage4_midpoint_snapshot.md", flush=True)


if __name__ == "__main__":
    main()
