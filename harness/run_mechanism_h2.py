"""H2 test (token quantization limits tail resolution) on the synthetic oracle
arm. Convention frozen in docs/mechanism_h2_convention.md BEFORE this runs.

    python -m harness.run_mechanism_h2

Same 40 GARCH-t/GJR-t paths + 60 windows + analytic truth as
synth/run_arbitration.py. Measures, per the pre-committed convention:
  (a) Chronos-base fidelity to known truth (E0/E2/E3, VaR+ES);
  (b) discreteness of its exact one-step categorical tail support;
  (c) head contrast vs Chronos-Bolt / Chronos-2 on identical windows via a fine
      deep-probability RESOLUTION SWEEP (distinct achievable VaR values).
Writes results/mechanism/h2_synth.csv (per-window) + h2_resolution.csv (sweep)
and the script-generated docs/mechanism_h2_results.md. Reported as-is.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
for _n in ("chronos", "transformers", "gluonts"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from erules.rules import (e2_es, e2_quantile, e3_var_es_from_fit, fit_gpd,
                          fit_nu, mad)
from synth.paths import garch_t_path, true_var_es

OUT = Path(__file__).resolve().parent.parent / "results" / "mechanism"
DOCS = Path(__file__).resolve().parent.parent / "docs"
CTX, N, N_PATHS, N_SAMPLE = 512, 1500, 20, 60
ALPHAS = (0.01, 0.025, 0.05)
DEEP = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
DGPS = [("garch_t", dict(a=0.09, b=0.90, gjr_gamma=0.0)),
        ("gjr_t", dict(a=0.03, b=0.90, gjr_gamma=0.08))]
PSWEEP = np.round(np.linspace(0.002, 0.05, 40), 5)   # fine deep-tail probability grid
SEED = 20260706


def _cat_quantiles(vals_sorted, cum, ps):
    return np.array([vals_sorted[min(np.searchsorted(cum, p), len(vals_sorted) - 1)]
                     for p in ps])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from harness.chronos_base_adapter import ChronosBaseAdapter
    from harness.chronos_adapters import Chronos2Adapter, ChronosBoltAdapter
    cb = ChronosBaseAdapter(seed=SEED)
    bolt = ChronosBoltAdapter()
    c2 = Chronos2Adapter()

    fid, res = [], []
    for dgp, params in DGPS:
        for p in range(N_PATHS):
            y, sig, nu_true = garch_t_path(SEED + p, n=N, **params)
            idx = np.linspace(CTX, N - 1, N_SAMPLE).astype(int)
            for t in idx:
                ctx = y[t - CTX:t].astype(np.float32)
                nu_ctx = fit_nu(ctx)
                u = float(np.quantile(ctx, 0.10)); q50c = float(np.median(ctx))
                try:                     # v2.0 P0-1 caller convention: explicit scale_ref
                    xi, beta, n_kept = fit_gpd(u - ctx[ctx < u], scale_ref=mad(ctx))
                    tau_mass = n_kept / len(ctx)   # v2.1 empirical mass (ruling)
                    gpd_ok = True
                except ValueError:
                    gpd_ok = False
                    tau_mass = float("nan")   # B2-i: defined on every path

                # ---- Chronos-base: exact categorical + S=1000 empirical grid ----
                vals, probs = cb.categorical(ctx)
                keep = probs > 1e-9
                order = np.argsort(vals[keep])
                cv = vals[keep][order]; cp = np.cumsum(probs[keep][order])
                s = cb.predict_samples(ctx, 1000)
                grid_cb = dict(zip(DEEP, np.quantile(s, DEEP)))
                e0_q05 = float(np.quantile(s, 0.05))
                achiev = np.unique(cv)
                tail = achiev[achiev < e0_q05]
                tail_gap = float(np.median(np.diff(tail))) if len(tail) >= 2 else np.nan
                bin_gap = float(np.min(np.diff(achiev))) if len(achiev) >= 2 else np.nan
                # resolution sweep (distinct achievable VaR over the fine p grid)
                cb_sweep = _cat_quantiles(cv, cp, PSWEEP)
                res.append({"dgp": dgp, "path": p, "t": int(t), "head": "chronos_base",
                            "support_k": int(keep.sum()),
                            "tail_support": int(len(tail)), "tail_nn_gap": tail_gap,
                            "bin_gap": bin_gap,
                            "n_distinct_sweep": int(len(np.unique(np.round(cb_sweep, 6))))})

                # ---- Chronos-Bolt / Chronos-2: native grids + sweep ----
                for name, ad in (("chronos_bolt", bolt), ("chronos_2", c2)):
                    g = dict(zip(DEEP, map(float, ad.predict_quantiles(ctx, DEEP))))
                    sweep = ad.predict_quantiles(ctx, list(PSWEEP))
                    res.append({"dgp": dgp, "path": p, "t": int(t), "head": name,
                                "support_k": np.nan, "tail_support": np.nan,
                                "tail_nn_gap": np.nan, "bin_gap": np.nan,
                                "n_distinct_sweep": int(len(np.unique(np.round(sweep, 6))))})
                    _fidelity(fid, name, dgp, p, t, g, sig[t], nu_true, nu_ctx,
                              q50c, u, xi, beta, tau_mass, gpd_ok)

                _fidelity(fid, "chronos_base", dgp, p, t, grid_cb, sig[t], nu_true,
                          nu_ctx, q50c, u, xi, beta, tau_mass, gpd_ok)
            print(f"[h2] {dgp} path {p}/{N_PATHS-1} done", flush=True)

    fd = pd.DataFrame(fid); rd = pd.DataFrame(res)
    fd.to_csv(OUT / "h2_synth.csv", index=False)
    rd.to_csv(OUT / "h2_resolution.csv", index=False)
    _summary(fd, rd)
    print(f"H2 complete: {len(fd)} fidelity rows, {len(rd)} resolution rows", flush=True)


def _fidelity(acc, head, dgp, p, t, grid, sig_t, nu_true, nu_ctx, q50c, u, xi,
              beta, tau_mass, gpd_ok):
    # B2-i (Phase-1 re-check fixup): tau_mass threaded through the signature —
    # the Phase-1 landing referenced it here without passing it (NameError on
    # any gpd_ok window; erratum in the Phase-1 report).
    for a in ALPHAS:
        vt, et = true_var_es(sig_t, nu_true, a)
        v0 = grid.get(a, np.nan)               # native/E0 deep quantile (absent -> nan)
        try:
            v2, e2 = e2_quantile(grid, a, nu_ctx), e2_es(grid, a, nu_ctx)
        except (ValueError, KeyError):
            v2 = e2 = np.nan
        if gpd_ok:
            # v2.0 P0-1: independent VaR/ES exception paths — ES-only failure
            # (xi>=1 mean-excess divergence) keeps the VaR.
            v3, e3 = e3_var_es_from_fit(grid, q50c, u, xi, beta, a,
                                        tau_mass=tau_mass)
        else:
            v3 = e3 = np.nan
        acc.append({"head": head, "dgp": dgp, "path": p, "t": int(t), "alpha": a,
                    "true_v": vt, "true_e": et, "e0_v": v0, "e2_v": v2, "e2_e": e2,
                    "e3_v": v3, "e3_e": e3})


def _summary(fd: pd.DataFrame, rd: pd.DataFrame) -> None:
    L = ["# H2 results — token quantization and tail resolution (script-generated)",
         "", "Convention: docs/mechanism_h2_convention.md (frozen before this run). "
         f"{N_PATHS} GARCH-t + {N_PATHS} GJR-t paths, {N_SAMPLE} windows each, ctx=512, "
         "S=1000. Synthetic DGP, not real returns — resolution/fidelity under KNOWN "
         "truth only.", "",
         "## (a) Chronos-base VaR fidelity vs analytic truth (mean|err|, return units)",
         "", "| E-rule | alpha=1% | 2.5% | 5% |", "|---|---|---|---|"]
    cb = fd[fd["head"] == "chronos_base"]   # NB: fd.head is the DataFrame method, not the column
    for er, col in (("E0", "e0_v"), ("E2", "e2_v"), ("E3", "e3_v")):
        vals = [f"{(cb[cb['alpha'] == a][col] - cb[cb['alpha'] == a].true_v).abs().mean():.3f}"
                for a in ALPHAS]
        L.append(f"| {er} | {vals[0]} | {vals[1]} | {vals[2]} |")
    L += ["", "## (b) Chronos-base categorical tail-support discreteness (per window)",
          "", "| metric | median | p10 | p90 |", "|---|---|---|---|"]
    g = rd[rd["head"] == "chronos_base"]
    for m, lab in (("support_k", "top_k support size"),
                   ("tail_support", "distinct token values below q05"),
                   ("tail_nn_gap", "tail nearest-nbr gap (ret units)"),
                   ("bin_gap", "token grid step (ret units)")):
        s = g[m].dropna()
        L.append(f"| {lab} | {s.median():.4g} | {s.quantile(0.1):.4g} | {s.quantile(0.9):.4g} |")
    L += ["", "## (c) Deep-tail resolution contrast (distinct achievable VaR over a "
          f"{len(PSWEEP)}-point p-sweep in [0.002,0.05])", "",
          "| head | median distinct | p10 | p90 |", "|---|---|---|---|"]
    med = {}
    for h in ("chronos_base", "chronos_bolt", "chronos_2"):
        s = rd[rd["head"] == h].n_distinct_sweep
        med[h] = s.median()
        L.append(f"| {h} | {s.median():.0f} | {s.quantile(0.1):.0f} | {s.quantile(0.9):.0f} |")
    ratio = med["chronos_2"] / max(med["chronos_base"], 1e-9)
    # pre-committed verdict
    ts = g["tail_support"].median(); tg = g["tail_nn_gap"].dropna().median()
    c1 = ts <= 10; c2c = tg >= 0.10; c3 = ratio >= 10
    verdict = "SUPPORTED" if (c1 and c2c and c3) else "NOT SUPPORTED"
    L += ["", "## Pre-committed verdict (frozen criteria)", "",
          f"- (1) median tail_support = {ts:.1f}  (<= 10?  {'YES' if c1 else 'NO'})",
          f"- (2) median tail_nn_gap = {tg:.3f}  (>= 0.10?  {'YES' if c2c else 'NO'})",
          f"- (3) Chronos-2 / Chronos-base distinct-resolution ratio = {ratio:.1f}x  "
          f"(>= 10x?  {'YES' if c3 else 'NO'})",
          "", f"**H2: {verdict}** on the synthetic arm (all three criteria required). "
          "Reported as-is; the mechanism/cause reading (bin grid vs top_k vs mass "
          "concentration) is deferred to the contrast step per the convention.", ""]
    (DOCS / "mechanism_h2_results.md").write_text("\n".join(L))


if __name__ == "__main__":
    import sys
    if "--summary-only" in sys.argv:      # regenerate the doc from the intact CSVs
        _summary(pd.read_csv(OUT / "h2_synth.csv"),
                 pd.read_csv(OUT / "h2_resolution.csv"))
        print("H2 summary regenerated from CSVs", flush=True)
    else:
        main()
    # G6 freshness stamp (v2.1.1, review #11): both modes end with current
    # outputs on disk; loud no-op inside package copies (no .git)
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("mechanism_h2")
