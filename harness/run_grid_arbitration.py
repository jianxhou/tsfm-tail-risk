"""Ranking-stability (condition 11) + BTC contamination DiD (condition 7).

    python -m harness.run_grid_arbitration

Reads results/stage4/backtests.csv (the extraction battery). Writes:
  docs/stage4_kendall_tau.md   E2- vs E3-conditioned model rankings, Kendall tau
                               per (asset, alpha); chronos_bolt on its own row
                               (per design condition — never smoothed).
  docs/stage4_btc_did.md       exposed {chronos_2,moirai_2_0,timesfm_2_5} vs
                               unexposed {chronos_bolt,lag_llama} difference-in-
                               differences around the 2021-07 corpus breakpoint.
No headline claims — these are the pre-registered diagnostics; interpretation is
for the final table review.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "stage4"
HEADLINE_HEADS = ["chronos_2", "moirai_2_0", "lag_llama"]   # condition 4
ALPHAS = (0.01, 0.025, 0.05)


def kendall_tau_table(bt: pd.DataFrame) -> str:
    """Per (asset, alpha): rank forecasters by mean FZ0 under E2 and under E3;
    Kendall tau between the two rankings. Rank only cells with finite FZ0."""
    L = ["# Stage-4 E2-vs-E3 ranking stability (Kendall tau; condition 11)", "",
         "Pre-registered diagnostic. Per (asset, alpha): forecasters ranked by mean "
         "FZ0 loss under E2 and under E3; Kendall tau between rankings. tau=1 => "
         "identical order (defensible extractors agree); low/negative tau => "
         "ranking is extraction-dependent and is PROMOTED to a headline finding "
         "(gate ruling), not smoothed.", "",
         "chronos_bolt is listed but flagged: per condition 5 its deep tail enters "
         "only via E1-E3, and it is not a headline-ranking head (condition 4).", "",
         "| alpha | assets | mean tau | min tau | # asset with tau<0.5 |",
         "|---|---|---|---|---|"]
    for a in ALPHAS:
        taus = []
        for asset in sorted(bt.asset.unique()):
            e2 = bt[(bt.asset == asset) & (bt.alpha == a) & (bt.erule == "e2")]
            e3 = bt[(bt.asset == asset) & (bt.alpha == a) & (bt.erule == "e3")]
            common = set(e2.forecaster) & set(e3.forecaster)
            r2 = e2[e2.forecaster.isin(common)].dropna(subset=["fz0_mean"])
            r3 = e3[e3.forecaster.isin(common)].dropna(subset=["fz0_mean"])
            shared = sorted(set(r2.forecaster) & set(r3.forecaster))
            if len(shared) < 3:
                continue
            o2 = r2.set_index("forecaster").loc[shared, "fz0_mean"].rank()
            o3 = r3.set_index("forecaster").loc[shared, "fz0_mean"].rank()
            tau = kendalltau(o2, o3).statistic
            if np.isfinite(tau):
                taus.append((asset, tau))
        if taus:
            tv = np.array([t for _, t in taus])
            L.append(f"| {a:g} | {len(taus)} | {tv.mean():.3f} | {tv.min():.3f} "
                     f"| {(tv < 0.5).sum()} |")
    L += ["", "Per-asset detail is in results/stage4/kendall_tau_detail.csv.", ""]
    return "\n".join(L)


def kendall_detail(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a in ALPHAS:
        for asset in sorted(bt.asset.unique()):
            e2 = bt[(bt.asset == asset) & (bt.alpha == a) & (bt.erule == "e2")].dropna(subset=["fz0_mean"])
            e3 = bt[(bt.asset == asset) & (bt.alpha == a) & (bt.erule == "e3")].dropna(subset=["fz0_mean"])
            shared = sorted(set(e2.forecaster) & set(e3.forecaster))
            if len(shared) < 3:
                continue
            o2 = e2.set_index("forecaster").loc[shared, "fz0_mean"].rank()
            o3 = e3.set_index("forecaster").loc[shared, "fz0_mean"].rank()
            rows.append({"asset": asset, "alpha": a, "n_forecasters": len(shared),
                         "kendall_tau": kendalltau(o2, o3).statistic})
    return pd.DataFrame(rows)


def btc_did(bt: pd.DataFrame) -> str:
    """DiD is computed from per-window series (needs forecasts), so here we only
    report the pre/post hit-rate levels available in the battery; the full
    window-level DiD lands with the final table (needs the forecast parquets)."""
    L = ["# Stage-4 BTC contamination DiD (condition 7) — battery-level preview", "",
         "Full window-level difference-in-differences (breakpoint 2021-07,",
         "exposed {chronos_2, moirai_2_0, timesfm_2_5} vs unexposed {chronos_bolt,",
         "lag_llama}) requires the per-window forecast series and lands with the",
         "final table. This preview shows BTC E2 alpha=5% hit rate by forecaster",
         "from the battery (whole-OOS, not yet split at the breakpoint).", "",
         "| forecaster | BTC E2 hit@5% | exposure (disclosed) |", "|---|---|---|"]
    exposure = {"chronos_2": "exposed", "moirai_2_0": "exposed",
                "timesfm_2_5": "exposed", "chronos_bolt": "unexposed",
                "lag_llama": "unexposed"}
    sub = bt[(bt.asset == "btc") & (bt.alpha == 0.05) & (bt.erule == "e2")]
    for _, r in sub.iterrows():
        L.append(f"| {r.forecaster} | {100*r.hit_rate:.2f}% "
                 f"| {exposure.get(r.forecaster, '?')} |")
    L += ["", "NOTE: level-only; DiD needs the pre/post split (final table).", ""]
    return "\n".join(L)


def main() -> None:
    bt = pd.read_csv(OUT / "backtests.csv")
    (ROOT / "docs" / "stage4_kendall_tau.md").write_text(kendall_tau_table(bt))
    kendall_detail(bt).to_csv(OUT / "kendall_tau_detail.csv", index=False)
    (ROOT / "docs" / "stage4_btc_did.md").write_text(btc_did(bt))
    print("arbitration: kendall_tau + btc_did written", flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("grid_arbitration")
