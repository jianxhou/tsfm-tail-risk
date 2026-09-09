"""Pilot D5 driver: tail-PIT coverage curves, violation-clustering timelines
(COVID + 2025-04), and the E0-E3 sensitivity table at alpha=1% by head type.

    python -m harness.run_pilot_d5

Figures -> results/figures/ (regenerable); sensitivity table ->
docs/pilot_d5_sensitivity.md (script-generated, committed).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "results" / "pilot_d3"
FIG = ROOT / "results" / "figures"
ASSETS = ["spx", "btc", "nvda"]
MODELS = ["chronos_bolt", "chronos_2", "timesfm_2_5", "moirai_2_0"]
ERULES = ["e0", "e1", "e2", "e3"]
HEAD = {m: yaml.safe_load((ROOT / "harness" / "registry.yaml").read_text())[m]["head_type"]
        for m in MODELS}


def tail_pit_curves() -> None:
    taus = [0.01, 0.025, 0.05, 0.1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, asset in zip(axes, ASSETS):
        for m in MODELS:
            ex = pd.read_parquet(D3 / f"extract_{m}_{asset}.parquet")
            cov = []
            for a in taus:
                col = f"e0_v{f'{a:g}'.replace('0.', '')}"
                if col in ex and ex[col].notna().any():
                    cov.append((ex["y"] <= ex[col]).mean())
                else:
                    cov.append(np.nan)
            ax.plot(taus, cov, marker="o", label=m)
        ax.plot(taus, taus, "k--", lw=1, label="nominal")
        ax.set(title=asset, xlabel="nominal tail level", xscale="log")
    axes[0].set_ylabel("empirical coverage (E0/native)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "d5_tail_pit_e0.png", dpi=150)
    plt.close(fig)


def violation_timelines() -> None:
    windows = {"covid": ("2020-01-01", "2020-07-31"),
               "tariff_2025": ("2025-02-01", "2025-07-31")}
    for wname, (lo, hi) in windows.items():
        fig, ax = plt.subplots(figsize=(11, 3.2))
        for k, m in enumerate(MODELS):
            ex = pd.read_parquet(D3 / f"extract_{m}_spx.parquet")
            ex["date"] = pd.to_datetime(ex["date"])
            sub = ex[(ex.date >= lo) & (ex.date <= hi)]
            hits = sub[sub["y"] <= sub["e2_v01"]]
            ax.scatter(hits["date"], np.full(len(hits), k), s=14, label=m)
        ax.set_yticks(range(len(MODELS)), MODELS)
        ax.set_title(f"SPX alpha=1% (E2) violation clustering — {wname}")
        fig.tight_layout()
        fig.savefig(FIG / f"d5_violations_{wname}.png", dpi=150)
        plt.close(fig)


def sensitivity_table() -> None:
    lines = ["# D5: E0-E3 sensitivity at alpha=1% (script-generated)",
             "", "Empirical hit rates (nominal 1%); '—' = rule unavailable "
             "(timesfm has no E0). Grouped by head type.", "",
             "| model (head) | asset | E0 | E1 | E2 | E3 | max-min spread |",
             "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for asset in ASSETS:
            ex = pd.read_parquet(D3 / f"extract_{m}_{asset}.parquet")
            rates, cells = [], []
            for er in ERULES:
                col = f"{er}_v01"
                if col in ex and ex[col].notna().any():
                    r = float((ex["y"] <= ex[col]).mean())
                    rates.append(r)
                    cells.append(f"{100*r:.2f}%")
                else:
                    cells.append("—")
            spread = f"{100*(max(rates)-min(rates)):.2f}pp" if len(rates) > 1 else "—"
            lines.append(f"| {m} ({HEAD[m]}) | {asset} | " + " | ".join(cells)
                         + f" | {spread} |")
    lines += ["", "Reading discipline: spreads larger than the model-vs-model spread "
              "at the same alpha feed C2 (structural underdetermination); no "
              "calibration verdicts from this table alone.", ""]
    (ROOT / "docs" / "pilot_d5_sensitivity.md").write_text("\n".join(lines))


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    tail_pit_curves()
    violation_timelines()
    sensitivity_table()
    print("D5 complete: 3 figures + sensitivity table", flush=True)
