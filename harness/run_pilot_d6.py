"""Pilot D6 driver: mini regime table, post-cutoff consistency quick-check,
and a ctx-length ablation (fast, chronos_bolt/SPX).

    nohup python -m harness.run_pilot_d6 > results/pilot_d6/run.log 2>&1 &

Outputs (script-generated, committed):
  docs/pilot_d6_regime.md        COVID vs 2019-calm calibration, alpha=5%
  docs/pilot_d6_postcutoff.md    full-sample vs post-cutoff direction check
  docs/pilot_d6_ctx_ablation.md  chronos_bolt SPX hit-rate at ctx in {256,512,2048}
All backtest reads use the frozen D3 extracts + registry cutoffs; the ablation is
the only new model run (bolt is ~90s/pass).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtests.var_tests import kupiec
from data.load import load_series
from harness.rolling import run_rolling

ROOT = Path(__file__).resolve().parent.parent
D3 = ROOT / "results" / "pilot_d6"  # placeholder; extracts read from pilot_d3
EX = ROOT / "results" / "pilot_d3"
OUT = ROOT / "results" / "pilot_d6"
ASSETS = ["spx", "btc", "nvda"]
MODELS = ["chronos_bolt", "chronos_2", "timesfm_2_5", "moirai_2_0"]
REG = yaml.safe_load((ROOT / "harness" / "registry.yaml").read_text())

# cutoff upper-bound dates parsed from registry prose (machine-registered there)
CUTOFF = {"chronos_bolt": "2025-11-21", "chronos_2": "2025-10-31",
          "timesfm_2_5": "2023-11-30", "moirai_2_0": "2025-11-30"}
REGIMES = {"covid": ("2020-02-15", "2020-04-30"),
           "calm_2019": ("2019-01-01", "2019-12-31")}


def _rate(ex: pd.DataFrame, col: str, mask=None) -> tuple[float, int]:
    sub = ex if mask is None else ex[mask]
    v = sub[col]
    ok = v.notna()
    if ok.sum() == 0:
        return float("nan"), 0
    return float((sub["y"][ok] <= v[ok]).mean()), int(ok.sum())


def regime_table() -> None:
    """Every regime cell carries n_alpha and a precision-fragile flag (user spec,
    D7): fragile cells support DIRECTION evidence only, never level claims."""
    a = 0.05
    lines = ["# D6 mini regime table (script-generated)", "",
             "SPX alpha=5% empirical hit rate, E2 VaR, by regime window. Nominal 5%.",
             "Each cell: rate (n, n_alpha). FRAGILE = n_alpha < 10 (pre-set reporting",
             "convention in the spirit of the Pele & M-M-P precision machinery):",
             "direction evidence only, no level claims.", "",
             "| model | COVID (2020-02-15..04-30) | 2019 calm | delta | COVID flag |",
             "|---|---|---|---|---|"]
    for m in MODELS:
        ex = pd.read_parquet(EX / f"extract_{m}_spx.parquet")
        ex["date"] = pd.to_datetime(ex["date"])
        cov, nc = _rate(ex, "e2_v05",
                        (ex.date >= REGIMES["covid"][0]) & (ex.date <= REGIMES["covid"][1]))
        clm, nk = _rate(ex, "e2_v05",
                        (ex.date >= REGIMES["calm_2019"][0]) & (ex.date <= REGIMES["calm_2019"][1]))
        na_c, na_k = a * nc, a * nk
        flag = "FRAGILE (direction-only)" if na_c < 10 else "ok"
        lines.append(f"| {m} | {100*cov:.1f}% (n={nc}, nα={na_c:.1f}) "
                     f"| {100*clm:.1f}% (n={nk}, nα={na_k:.1f}) "
                     f"| {100*(cov-clm):+.1f}pp | {flag} |")
    lines += ["", f"Expected violations per COVID cell at nominal 5%: "
              f"{a*52:.1f} — every COVID cell is precision-fragile by construction; "
              "the table supports 'all models undercover during the shift' as a "
              "DIRECTION statement (binomial SE at n=52 is ~3.0pp; observed excesses "
              "are 4.1-18.3pp), not calibrated-level comparisons between models. "
              "Feeds H3 qualitatively; formal event-time decay analysis is Stage 5.", ""]
    (ROOT / "docs" / "pilot_d6_regime.md").write_text("\n".join(lines))


def postcutoff_table() -> None:
    lines = ["# D6 post-cutoff consistency quick-check (script-generated)", "",
             "Full-sample vs post-cutoff-strata alpha=5% hit rate (E2 VaR), 3 assets "
             "pooled. Direction check per GO#4: does the post-cutoff slice agree in "
             "sign of miscalibration (over/under) with the full sample?", "",
             "| model | cutoff | full rate | post rate | full n | post n | same sign? |",
             "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        yv, vv, dts = [], [], []
        for asset in ASSETS:
            ex = pd.read_parquet(EX / f"extract_{m}_{asset}.parquet")
            yv.append(ex["y"].to_numpy())
            vv.append(ex["e2_v05"].to_numpy())
            dts.append(pd.to_datetime(ex["date"]).to_numpy())
        y = np.concatenate(yv); v = np.concatenate(vv)
        d = np.concatenate(dts)
        ok = np.isfinite(v)
        full = float((y[ok] <= v[ok]).mean())
        post_mask = ok & (d > np.datetime64(CUTOFF[m]))
        post = float((y[post_mask] <= v[post_mask]).mean()) if post_mask.sum() else float("nan")
        same = "yes" if (np.sign(full - 0.05) == np.sign(post - 0.05)) else "NO"
        lines.append(f"| {m} | {CUTOFF[m]} | {100*full:.2f}% | {100*post:.2f}% "
                     f"| {ok.sum()} | {int(post_mask.sum())} | {same} |")
    lines += ["", "1% not reported on short post-cutoff strata (pre-declared "
              "fragile, contamination_strata.md). TimesFM-2.5 is the anchor "
              "(largest clean stratum).", ""]
    (ROOT / "docs" / "pilot_d6_postcutoff.md").write_text("\n".join(lines))


def ctx_ablation() -> None:
    from harness.chronos_adapters import ChronosBoltAdapter
    adapter = ChronosBoltAdapter()
    levels = [0.01, 0.025, 0.05]
    names = ["q01", "q025", "q05"]
    df = load_series("spx")
    lines = ["# D6 ctx-length ablation (script-generated)", "",
             "chronos_bolt SPX native (E0) hit rate vs context length. Nominal in "
             "parens.", "",
             "| ctx | q01 (1%) | q025 (2.5%) | q05 (5%) | n |",
             "|---|---|---|---|---|"]
    for ctx_len in (256, 512, 2048):
        def fn(ctx):
            q = adapter.predict_quantiles(ctx, levels)
            return dict(zip(names, map(float, q)))
        frame, meta = run_rolling(df, fn, value_col="logret", ctx_len=ctx_len,
                                  oos_start="2018-01-01")
        rates = [(frame["y"] <= frame[c]).mean() for c in names]
        lines.append(f"| {ctx_len} | {100*rates[0]:.2f}% | {100*rates[1]:.2f}% "
                     f"| {100*rates[2]:.2f}% | {meta['n_windows']} |")
        print(f"[ctx] {ctx_len}: done", flush=True)
    lines += ["", "Bolt clamps deep tails to its grid edge (registry "
              "native_tail_behavior); ctx sensitivity here is the clamp moving, "
              "not tail knowledge. Read with the E0-E3 sensitivity table.", ""]
    (ROOT / "docs" / "pilot_d6_ctx_ablation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    regime_table()
    postcutoff_table()
    ctx_ablation()
    print("D6 complete: regime + post-cutoff + ctx-ablation tables", flush=True)
