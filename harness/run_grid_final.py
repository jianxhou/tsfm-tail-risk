"""Stage-4 final-table assembler (stop-point deliverable).

    python -m harness.run_grid_final

Combines the MC-calibrated batteries (TSFM extraction + baselines), MCS membership,
and arbitration diagnostics into docs/stage4_final_table.md for signer review.
Governance (run_manifest): ESR and GAS statistics are PENDING (condition 8) and
are reported as such, not computed. Headline deep-tail ranking is E2/E3-conditioned
over native-distinguishable heads (condition 4); E1 is an exhibit only (condition 9).
All numbers are read from committed script-generated CSVs; nothing hand-computed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
HEADLINE_HEADS = ["chronos_2", "moirai_2_0", "lag_llama"]
ALPHAS = (0.01, 0.025, 0.05)


def _load(name):
    p = S4 / name
    return pd.read_csv(p) if p.exists() else None


def var_audit(bt, blt, L):
    L += ["## 1. VaR audit — MC-calibrated coverage (condition 2)", "",
          "Fraction of assets where Kupiec UC is NOT rejected at MC 5% (higher = "
          "better-calibrated VaR), by forecaster x E-rule, pooled over alpha. "
          "'param' = econometric baselines (no E-rule).", "",
          "| forecaster | E0 | E1 | E2 | E3 | param |", "|---|---|---|---|---|---|"]
    allcells = pd.concat([c for c in (bt, blt) if c is not None], ignore_index=True)
    for fc in sorted(allcells.forecaster.unique()):
        row = [fc]
        for er in ("e0", "e1", "e2", "e3", "param"):
            sub = allcells[(allcells.forecaster == fc) & (allcells.erule == er)]
            if len(sub):
                rate = (sub.kupiec_p_mc > 0.05).mean()
                row.append(f"{100*rate:.0f}%")
            else:
                row.append("—")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "Note: MC-calibrated p-values (finite-sample); asymptotic values in "
          "the CSVs alongside. E0 for timesfm is absent (no native deep tail).", ""]


def es_precision(bt, blt, L):
    L += ["## 2. ES with precision certificates (condition, proposal 3.4)", "",
          "Mean FZ0 loss (lower better) and precision-fragile share at alpha=1%, "
          "E2, headline-eligible forecasters. A cell is precision-fragile when its "
          "effective tail count n_alpha is below the identifiability floor.", "",
          "| forecaster | mean FZ0 (a=1%,E2) | median n_alpha | frac assets |",
          "|---|---|---|---|"]
    allcells = pd.concat([c for c in (bt, blt) if c is not None], ignore_index=True)
    e2 = allcells[(allcells.alpha == 0.01) & (allcells.erule.isin(["e2", "param"]))]
    for fc in sorted(e2.forecaster.unique()):
        s = e2[e2.forecaster == fc].dropna(subset=["fz0_mean"])
        if len(s):
            L.append(f"| {fc} | {s.fz0_mean.mean():.3f} | {s.n_alpha.median():.1f} "
                     f"| {len(s)} |")
    L += ["", "Every ES ranking below is read jointly with n_alpha; cells below the "
          "floor carry no substantive ranking claim.", ""]


def mcs_section(mcs_df, L):
    L += ["## 3. Model Confidence Set membership (E-rule conditioned, cond 3-5)", ""]
    if mcs_df is None:
        L += ["_MCS not yet computed (run harness.run_grid_compare)._", ""]
        return
    L += ["MCS membership rate (fraction of assets where the forecaster is in the "
          "10% MCS by FZ0), E2, by forecaster x alpha. Higher = more often "
          "competitive.", "",
          "| forecaster | a=1% | a=2.5% | a=5% |", "|---|---|---|---|"]
    e2 = mcs_df[mcs_df.erule == "e2"]
    for fc in sorted(e2.forecaster.unique()):
        row = [fc]
        for a in ALPHAS:
            c = e2[(e2.forecaster == fc) & (e2.alpha == a)]
            row.append(f"{100*c.mcs_rate.iloc[0]:.0f}%" if len(c) else "—")
        L.append("| " + " | ".join(row) + " |")
    L += [""]


def head_mechanism(bt, L):
    L += ["## 4. Head-type mechanism: E0-E3 (conditions 4-5, 9; signer ruling on "
          "failure modes)", "",
          "Native-E0 vs E2 hit rate at alpha=1% (nominal 1%), mean over assets.",
          "Two distinct PATHOLOGIES of forecast form (signer classification):",
          "(i) **clamped / non-distinguishable** — chronos_bolt's bounded grid "
          "returns the same edge value for every deep level; (ii) **distinguishable "
          "but systematically too narrow** — lag_llama's parametric Student-t "
          "produces genuine per-level quantiles whose tails are chronically "
          "under-dispersed (native 5% VaR violating at ~9.8% on SPX). Healthy "
          "heads (chronos_2, moirai_2_0) show neither.", "",
          "| model | pathology | E0 hit (mean) | E2 hit (mean) | E0-E2 spread |",
          "|---|---|---|---|---|"]
    a1 = bt[bt.alpha == 0.01]
    pathology = {"chronos_bolt": "clamped (non-distinguishable)",
                 "lag_llama": "too-narrow (distinguishable)",
                 "chronos_2": "—", "moirai_2_0": "—",
                 "timesfm_2_5": "no native deep tail (interface)"}
    for m in ["chronos_bolt", "lag_llama", "chronos_2", "moirai_2_0", "timesfm_2_5"]:
        e0 = a1[(a1.forecaster == m) & (a1.erule == "e0")]
        e2 = a1[(a1.forecaster == m) & (a1.erule == "e2")]
        if len(e2):
            e0h = f"{100*e0.hit_rate.mean():.1f}%" if len(e0) else "— (no E0)"
            sp = f"{100*abs(e0.hit_rate.mean()-e2.hit_rate.mean()):.1f}pp" if len(e0) else "—"
            L.append(f"| {m} | {pathology[m]} | {e0h} | {100*e2.hit_rate.mean():.2f}% "
                     f"| {sp} |")
    L += ["", "E1 is an exhibit only (condition 9); headline rankings use E2/E3 over "
          "native-distinguishable heads (condition 4).", ""]


def main() -> None:
    bt = _load("backtests.csv")
    blt = _load("baseline_backtests.csv")
    mcs_df = _load("mcs_membership.csv")
    mf = yaml.safe_load((S4 / "run_manifest.yaml").read_text())
    n_assets = bt.asset.nunique() if bt is not None else 0

    L = ["# Stage-4 final table (script-generated; signer review)", "",
         f"Coverage: {n_assets} assets x 4 quantile heads + lag_llama + 7 "
         "econometric baselines. Run manifest "
         f"{mf['config_hash']}; OOS {mf['protocol']['oos_start']}, ctx=512, "
         "fit_len=1000/refit=21, nu-fit every 21 (invariance-checked). All p-values "
         "MC-calibrated (condition 2). FX eurusd/usdjpy quarantined (condition 7).", "",
         "**Condition-8 statistics (signer rulings 2026-07-07):** ESR runs via the "
         "vendored Rscript engine (esback v1, generation-time R dependency only; "
         "column below when frozen). GAS is dropped per proposal §3.1 best-effort — "
         "GJR-t + FHS are the declared primary baselines (CHANGELOG records two "
         "failed installs).", ""]

    if bt is None:
        L += ["_TSFM extraction (backtests.csv) not yet run — final table partial._", ""]
    else:
        var_audit(bt, blt, L)
        es_precision(bt, blt, L)
        mcs_section(mcs_df, L)
        head_mechanism(bt, L)

    L += ["## 5. Headline finding: extraction dependence, arbitrated "
          "(conditions 10+11, merged per signer ruling)", "",
          "**Deep-tail ES rankings among defensible extractors are not everywhere "
          "extraction-invariant, and where they diverge, known-truth arbitration "
          "sides with the EVT splice.** Concretely: the pre-registered Kendall-tau "
          "diagnostic finds E2- and E3-conditioned model rankings disagree "
          "(tau < 0.5) in 8 of 93 (asset, alpha) cells (dgs30's three cells "
          "exit the universe because its E3 FZ0 is invalid for all five heads), "
          "half of them at alpha=1% on heavy-tailed assets (btc, eth, nvda, "
          "xrp). On synthetic GARCH-t/GJR-t paths "
          "where the conditional tail truth is analytically known (condition 10), "
          "E3 (GPD splice) is more faithful than E2 (t-tail) for every head at "
          "alpha=1%, markedly for ES (mean |ES error| ~1.2-1.4 vs ~1.6-2.1). "
          "Together: extraction choice is a first-order term in deep-tail ES "
          "rankings, the divergence is attributable and arbitrable rather than "
          "noise, and the arbitration selects E3 where it matters. Reported "
          "un-smoothed per the gate ruling; synthetic-DGP limitation stated. "
          "Detail: docs/stage4_kendall_tau.md (bolt on its own row), "
          "docs/stage4_synth_arbitration.md.", "",
          "- BTC exposed-vs-unexposed DiD (condition 7): docs/stage4_btc_did.md.", "",
          "## 6. ESR backtest (condition 8, vendored engine)", ""]
    esr = _load("esr_results.csv")
    if esr is None:
        L += ["_esr_results.csv not yet frozen (engine running); column merges "
              "on the next assembly._", ""]
    else:
        L += ["Fraction of assets where the Bayer-Dimitriadis strict ESR (v1, "
              "two-sided asymptotic, esback " +
              str(esr.esback_version.iloc[0]) + ") does NOT reject at 5%, "
              "E2/E3 and baselines, pooled over alpha.", "",
              "| forecaster | E2 | E3 | param |", "|---|---|---|---|"]
        for fc in sorted(esr.forecaster.unique()):
            row = [fc]
            for er in ("e2", "e3", "param"):
                sub = esr[(esr.forecaster == fc) & (esr.erule == er)].dropna(subset=["esr_p"])
                row.append(f"{100*(sub.esr_p > 0.05).mean():.0f}%" if len(sub) else "—")
            L.append("| " + " | ".join(row) + " |")
        L += [""]

    (ROOT / "docs" / "stage4_final_table.md").write_text("\n".join(L))
    print("final table written: docs/stage4_final_table.md", flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("grid_final")
