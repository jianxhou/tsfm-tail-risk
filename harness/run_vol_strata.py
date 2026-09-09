"""Stage-5 vol strata + Markov-switching robustness (design §B / §C.2;
executed at closeout — see CHANGELOG 2026-07-08, rulings §4).

    PYTHONUNBUFFERED=1 python -u -m harness.run_vol_strata

Primary: per RV-quartile stratum — trailing 21-day realized vol ENDING at t-1
(strictly prior, conditioning-only; quartile cutoffs from full-OOS per asset,
disclosed) — MC-UC pass rate over assets and mean FZ0 (mean over assets of the
per-asset stratum mean, valid-ES days only), per layer at alpha=5%, E3.
Layers: unrepaired E3 per TSFM + GJR-t + FHS baselines, per-day series from
results/stage4/{repairs,baselines}/ (full aligned OOS sample; this is §B
regime conditioning, not the §A matched repair evaluation).

Robustness (committed, §C.2): statsmodels MarkovRegression 2-state on squared
OOS returns (switching variance), smoothed-probability argmax states, state 1
relabeled = high variance; same stats per state. Non-converged assets logged
and skipped.

Outputs: results/stage4/vol_strata.csv, markov_strata.csv, docs/stage5_strata.md.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from backtests.scores import fz0
from backtests.var_tests import kupiec
from data.load import load_series
from harness.run_grid_extract import mc_p

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
ALPHA = 0.05
RV_WIN = 21
TSFM = ["chronos_bolt", "chronos_2", "timesfm_2_5", "moirai_2_0", "lag_llama"]
LAYERS = [f"{m}:e3" for m in TSFM] + ["gjr_t", "fhs"]


def layer_frame(layer, asset):
    """(date-indexed) t, y, v, e at alpha=5% for one layer."""
    if layer in ("gjr_t", "fhs"):
        p = S4 / "baselines" / f"{layer}_{asset}.parquet"
        if not p.exists():
            return None
        d = pd.read_parquet(p)
        out = d[["t", "date", "y"]].copy()
        out["v"], out["e"] = d["v05"], d["e05"]
    else:
        model = layer.split(":")[0]
        p = S4 / "repairs" / f"{model}_{asset}.parquet"
        if not p.exists():
            return None
        d = pd.read_parquet(p)
        out = d[["t", "date", "y"]].copy()
        out["v"], out["e"] = d["u_e3_v05"], d["u_e3_e05"]
    out = out[np.isfinite(out["v"])]
    return out.reset_index(drop=True)


def rv_series(asset):
    """Trailing RV_WIN-day realized vol ending at t-1, indexed by t."""
    df = load_series(asset)
    x = df[df.attrs.get("column", "logret")].to_numpy()
    rv = pd.Series(x).rolling(RV_WIN).std().shift(1)     # ends at t-1
    return rv.to_numpy()


def markov_states(y: np.ndarray):
    """2-state MarkovRegression on squared returns; 1 = high-variance state.
    Classification: LAGGED FILTERED argmax — state for day t is the filtered
    state at t-1 (uses data through t-1 only; ex-ante, mirrors the RV strata's
    shift(1); day 0 unassigned). Two rejected alternatives, both verified:
    smoothed probabilities condition on the contemporaneous outcome (20-35%
    high-state hit rates for every layer incl. baselines — selection artifact);
    predicted-marginal argmax degenerates to a constant state (stationary pull;
    never selects high vol). search_reps randomness seeded (rule 4)."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    np.random.seed(20260706)                 # statsmodels search_reps uses global RNG
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = MarkovRegression(y ** 2, k_regimes=2, trend="c",
                               switching_variance=True)
        res = mod.fit(em_iter=50, search_reps=5)
    fp = np.argmax(np.asarray(res.filtered_marginal_probabilities), axis=1)
    st = np.full(len(fp), -1)
    st[1:] = fp[:-1]                         # state_t = filtered state at t-1
    # relabel: state with higher mean squared return = 1
    if (y[st == 0] ** 2).mean() > (y[st == 1] ** 2).mean():
        st = np.where(st >= 0, 1 - st, -1)
    return st


def stratum_cell(y, v, e, alpha):
    """(n, hit_rate, uc_pass, fz0_mean) for one asset-stratum."""
    n = len(y)
    if n < 60:
        return None
    uc = kupiec(y, v, alpha)
    p_mc = mc_p("kupiec", uc["stat"], n, alpha)
    good = np.isfinite(e) & (e < v) & (e < 0)
    fzm = float(fz0(y[good], v[good], e[good], alpha).mean()) if good.mean() > 0.99 \
        else np.nan
    return {"n": n, "hit_rate": uc["hit_rate"], "uc_p_mc": p_mc,
            "uc_pass": p_mc > 0.05, "fz0_mean": fzm}


def main():
    mf = yaml.safe_load((S4 / "run_manifest.yaml").read_text())
    assets = mf["assets"]
    rv_cache = {a: rv_series(a) for a in assets}
    mk_cache = {}
    rows_rv, rows_mk = [], []
    for layer in LAYERS:
        for asset in assets:
            lf = layer_frame(layer, asset)
            if lf is None or len(lf) < 400:
                print(f"[strata] SKIP {layer}/{asset}: no series", flush=True)
                continue
            y, v, e = lf["y"].to_numpy(), lf["v"].to_numpy(), lf["e"].to_numpy()
            t = lf["t"].to_numpy().astype(int)
            rv = rv_cache[asset][t]
            okrv = np.isfinite(rv)
            q = np.nanquantile(rv[okrv], [0.25, 0.5, 0.75])   # full-OOS cutoffs
            stratum = np.digitize(rv, q)                       # 0..3
            for s in range(4):
                m = okrv & (stratum == s)
                cell = stratum_cell(y[m], v[m], e[m], ALPHA)
                if cell:
                    rows_rv.append({"layer": layer, "asset": asset,
                                    "stratum": f"Q{s+1}", **cell})
            if asset not in mk_cache:
                try:
                    mk_cache[asset] = markov_states(y)
                except Exception as ex:
                    print(f"[strata] MARKOV-FAIL {asset}: {type(ex).__name__}",
                          flush=True)
                    mk_cache[asset] = None
            st = mk_cache[asset]
            # v2.1 P1-0 fix 2 (review #10 §6): the v2.0 silent skip here
            # dropped the four rate assets out of the fhs/gjr_t Markov rows
            # (24- vs 28-asset denominators) with every gate green. A length
            # mismatch means a stale upstream layer — hard error, never skip.
            if st is not None and len(st) != len(y):
                raise RuntimeError(
                    f"[strata] {layer}/{asset}: markov-state length {len(st)} "
                    f"!= series length {len(y)} — stale upstream layer frame; "
                    f"rerun the §2.7 chain in order (review #10 P1-0)")
            if st is not None:
                for s, nm in ((0, "low"), (1, "high")):
                    cell = stratum_cell(y[st == s], v[st == s], e[st == s], ALPHA)
                    if cell:
                        rows_mk.append({"layer": layer, "asset": asset,
                                        "state": nm, **cell})
        print(f"[strata] {layer} done", flush=True)

    rv = pd.DataFrame(rows_rv)
    mk = pd.DataFrame(rows_mk)
    rv.to_csv(S4 / "vol_strata.csv", index=False)
    mk.to_csv(S4 / "markov_strata.csv", index=False)
    _summary(rv, mk)
    print(f"strata complete: {len(rv)} RV rows, {len(mk)} Markov rows", flush=True)


def _summary(rv: pd.DataFrame, mk: pd.DataFrame):
    L = ["# Stage-5 vol strata + Markov robustness (script-generated; design §B/§C.2,",
         "# executed at closeout per rulings §4)", "",
         "alpha=5%, E3. RV strata: trailing-21d realized vol ending at t-1",
         "(conditioning-only; full-OOS quartile cutoffs, disclosed). Cell =",
         "MC-UC pass rate over assets | mean FZ0 (valid-ES days).", "",
         "## RV quartiles", "",
         "| layer | Q1 (low) | Q2 | Q3 | Q4 (high) |", "|---|---|---|---|---|"]

    def cell(d):
        if not len(d):
            return "—"
        return (f"{d.uc_pass.mean():.0%} / {d.fz0_mean.mean():.3f} / "
                f"hit {d.hit_rate.median():.3f}")

    for lay in LAYERS:
        cs = [cell(rv[(rv.layer == lay) & (rv.stratum == f"Q{s+1}")])
              for s in range(4)]
        L.append(f"| {lay} | " + " | ".join(cs) + " |")
    L += ["", "## Markov 2-state robustness (same outputs, MS classification)", "",
          "| layer | low state | high state |", "|---|---|---|"]
    if len(mk):
        for lay in LAYERS:
            cl = cell(mk[(mk.layer == lay) & (mk.state == "low")])
            ch = cell(mk[(mk.layer == lay) & (mk.state == "high")])
            L.append(f"| {lay} | {cl} | {ch} |")
    else:
        L.append("| (no converged assets) | — | — |")
    n_mk = mk.asset.nunique() if len(mk) else 0
    L += ["", f"Markov classification converged on {n_mk}/32 assets;",
          "failures logged in strata run log. States = LAGGED FILTERED argmax",
          "(state_t from data through t-1; ex-ante, mirrors RV shift(1)).",
          "Two alternatives computed and REJECTED with evidence: (i) smoothed",
          "probabilities condition on the contemporaneous outcome and yield",
          "0.21-0.35 high-state hit rates for every layer incl. GJR-t/FHS —",
          "selection artifact, hard-rule-3 violation; (ii) predicted-marginal",
          "argmax degenerates to a constant state (never selects high vol).",
          "Per-asset detail: results/stage4/vol_strata.csv, markov_strata.csv.", ""]
    (ROOT / "docs" / "stage5_strata.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("vol_strata")
