"""tab_w512 — post-hoc information-set arm (review #13 item 6,
author-approved 2026-08-25): GARCH-t, GJR-t, and FHS re-estimated on
512-observation rolling windows (the TSFM context length) against the
registered 1,000-observation setting; identical OOS days, refit cadence,
and assets (frozen run manifest; harness/run_w512_arm.py). Not
pre-specified; the registered arm remains 1,000."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import NAME, ROOT, S4, write_tex

W512 = ROOT / "results" / "v31_w512" / "baseline_backtests_w512.csv"
ARMS = ("garch_t", "gjr_t", "fhs")


def main() -> None:
    w5 = pd.read_csv(W512)
    w1 = pd.read_csv(S4 / "baseline_backtests.csv")
    w1 = w1[w1.forecaster.isin(ARMS)]
    uc = lambda d: (d.kupiec_p_mc >= 0.05).mean()

    body = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{Post-hoc information-set arm: the two GARCH variants and",
        r"FHS re-estimated on 512-observation rolling windows, matching the",
        r"TSFM context length, against the registered 1{,}000-observation",
        r"setting. Same out-of-sample days, refit cadence, and 32-asset",
        r"panel; UC pass is the share of assets passing MC-calibrated",
        r"unconditional coverage at the 5\% test level; FZ0 is the mean",
        r"joint VaR--ES loss at $\alpha=1\%$. This arm was added after the",
        r"main results (not pre-specified); the registered setting remains",
        r"1{,}000.}",
        r"\label{tab:w512}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lcccccccc@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{UC pass, $\alpha=1\%$}"
        r" & \multicolumn{2}{c}{UC pass, $\alpha=2.5\%$}"
        r" & \multicolumn{2}{c}{UC pass, $\alpha=5\%$}"
        r" & \multicolumn{2}{c}{FZ0, $\alpha=1\%$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"window & 512 & 1{,}000 & 512 & 1{,}000 & 512 & 1{,}000 & 512 & 1{,}000 \\",
        r"\midrule",
    ]
    for b in ARMS:
        cells = []
        for a in (0.01, 0.025, 0.05):
            d5 = w5[(w5.forecaster == b) & (w5.alpha == a)]
            d1 = w1[(w1.forecaster == b) & (w1.alpha == a)]
            cells += [f"{uc(d5)*100:.0f}\\%", f"{uc(d1)*100:.0f}\\%"]
        d5 = w5[(w5.forecaster == b) & (w5.alpha == 0.01)]
        d1 = w1[(w1.forecaster == b) & (w1.alpha == 0.01)]
        cells += [f"{d5.fz0_mean.mean():.3f}", f"{d1.fz0_mean.mean():.3f}"]
        body.append(NAME[b] + " & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_tex("tab_w512.tex", "\n".join(body))


if __name__ == "__main__":
    main()
