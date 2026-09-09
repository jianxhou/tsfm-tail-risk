"""Condition-10 synthetic extractor arbitration: run all quantile-head TSFMs on
synthetic GARCH-t / GJR-t paths where the conditional tail truth is analytically
known, and measure E2-vs-E3 fidelity per head.

    nohup python -m synth.run_arbitration > results/stage4/synth_arb.log 2>&1 &

Scope (documented; modest by design — this characterizes fidelity, not a full
backtest): 20 GARCH-t + 20 GJR-t paths, n=1500, ctx=512, SAMPLE windows evenly
spaced (t>=512). For each sampled window, each TSFM's native grid -> E2 and E3
VaR/ES; error = extraction - known truth (from the path's conditional sigma).
Output: results/stage4/synth_arbitration.csv, and docs/stage4_synth_arbitration.md
(mean |error| by head x E-rule x alpha; which rule is closer to truth per head).
LIMITATION stated wherever cited: synthetic GARCH-t/GJR-t != real returns; this
arbitrates extractor fidelity under a KNOWN DGP only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from erules.rules import (e2_es, e2_quantile, e3_var_es_from_fit, fit_gpd,
                          fit_nu, mad)
from synth.paths import garch_t_path, true_var_es

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "stage4"
CTX = 512
N = 1500
N_PATHS = 20
N_SAMPLE = 60
ALPHAS = (0.01, 0.025, 0.05)
DEEP = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
DECILES = [round(0.1 * k, 1) for k in range(1, 10)]
ADAPTERS = {
    "chronos_bolt": ("harness.chronos_adapters", "ChronosBoltAdapter", DEEP),
    "chronos_2": ("harness.chronos_adapters", "Chronos2Adapter", DEEP),
    "timesfm_2_5": ("harness.timesfm_adapter", "TimesFM25Adapter", DECILES),
    "moirai_2_0": ("harness.moirai_adapter", "Moirai2Adapter", DEEP),
}
DGPS = [("garch_t", dict(a=0.09, b=0.90, gjr_gamma=0.0)),
        ("gjr_t", dict(a=0.03, b=0.90, gjr_gamma=0.08))]


def colname(lv):
    return "q" + f"{lv:g}".replace("0.", "")


def main() -> None:
    import warnings, logging
    warnings.filterwarnings("ignore")
    for n in ("chronos", "transformers", "gluonts"):
        logging.getLogger(n).setLevel(logging.ERROR)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    import time as _time
    for model, (mod_name, cls_name, levels) in ADAPTERS.items():
        m_t0 = _time.time()
        mod = __import__(mod_name, fromlist=[cls_name])
        adapter = getattr(mod, cls_name)()
        names = [colname(lv) for lv in levels]
        for dgp, params in DGPS:
            for p in range(N_PATHS):
                y, sig, nu_true = garch_t_path(20260706 + p, n=N, **params)
                idx = np.linspace(CTX, N - 1, N_SAMPLE).astype(int)
                for t in idx:
                    ctx = y[t - CTX:t].astype(np.float32)
                    # grid MUST be keyed by float level (e2/e3 index grid[0.5], grid[0.10]),
                    # not by column name — matches run_grid_extract's construction.
                    grid = dict(zip(levels, map(float, adapter.predict_quantiles(ctx, levels))))
                    nu_ctx = fit_nu(ctx)
                    u = float(np.quantile(ctx, 0.10))
                    q50c = float(np.median(ctx))
                    try:                                       # v2.0 P0-1 caliber
                        xi, beta, n_kept = fit_gpd(u - ctx[ctx < u], scale_ref=mad(ctx))
                        tau_mass = n_kept / len(ctx)   # v2.1 empirical mass (ruling)
                        gpd_ok = True
                    except ValueError:
                        gpd_ok = False
                    for a in ALPHAS:
                        vt, et = true_var_es(sig[t], nu_true, a)
                        try:
                            v2, e2 = e2_quantile(grid, a, nu_ctx), e2_es(grid, a, nu_ctx)
                        except (ValueError, KeyError):
                            v2 = e2 = np.nan
                        if gpd_ok:
                            # v2.0 P0-1: independent VaR/ES exception paths
                            v3, e3 = e3_var_es_from_fit(grid, q50c, u, xi, beta, a,
                                                        tau_mass=tau_mass)
                        else:
                            v3 = e3 = np.nan
                        rows.append({"model": model, "dgp": dgp, "path": p, "t": int(t),
                                     "alpha": a, "true_v": vt, "true_e": et,
                                     "e2_v": v2, "e2_e": e2, "e3_v": v3, "e3_e": e3})
                el = _time.time() - m_t0
                print(f"[synth-arb] {model}/{dgp} path {p}/{N_PATHS-1} "
                      f"(model elapsed {el/60:.1f}min)", flush=True)
        # run-discipline guard (print-level): flag a model that runs long
        if _time.time() - m_t0 > 45 * 60:
            print(f"[synth-arb] WARN {model} took {(_time.time()-m_t0)/60:.0f}min "
                  f"(>45min guard) — check for pathology", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synth_arbitration.csv", index=False)
    _summary(df)
    print(f"synth arbitration complete: {len(df)} rows", flush=True)


def _summary(df: pd.DataFrame) -> None:
    L = ["# Condition-10 synthetic extractor arbitration (script-generated)", "",
         f"{N_PATHS} GARCH-t + {N_PATHS} GJR-t paths, {N_SAMPLE} sampled windows "
         "each, ctx=512. Mean |extraction - known truth| for VaR, by head x E-rule "
         "x alpha (return units x100). LIMITATION: synthetic DGP, not real returns "
         "- arbitrates extractor fidelity under KNOWN truth only.", "",
         "| model | alpha | mean|E2 VaR err| | mean|E3 VaR err| | closer |",
         "|---|---|---|---|---|"]
    for m in df.model.unique():
        for a in ALPHAS:
            s = df[(df.model == m) & (df.alpha == a)]
            e2 = (s.e2_v - s.true_v).abs().mean()
            e3 = (s.e3_v - s.true_v).abs().mean()
            closer = "E2" if e2 < e3 else "E3"
            L.append(f"| {m} | {a:g} | {e2:.3f} | {e3:.3f} | {closer} |")
    L += ["", "ES fidelity + per-DGP breakdown in results/stage4/synth_arbitration.csv.", ""]
    (ROOT / "docs" / "stage4_synth_arbitration.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("synth_arbitration")
