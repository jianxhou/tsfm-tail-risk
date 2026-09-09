"""Matched-date sensitivity for the five late-start assets — v2.0 Phase-3
item 1b (review #9 A-P1-4: TSFM and econometric day sets differ where the
1,000-day baseline estimation window starts later than the 512-day TSFM
context; csi300 2,639 vs 2,180, eth/xrp 2,647 vs 2,159, btc/ltc 3,796 vs
3,308).

    python -m harness.run_matched_dates

For every (TSFM head, extraction rule, alpha) cell on those five assets, the
hit rate and the MC-calibrated Kupiec decision (B=10,000, seed 20260706 —
the production convention of run_grid_extract.mc_p) are recomputed twice
from the persisted forecasts: on the delivered full TSFM day set, and on the
intersection with the baseline day set. No new forecasts. The full-day-set
pass is a reproduction gate against the delivered backtests.csv (n and
hit_rate must match exactly).

Output: results/stage4/matched_dates.csv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.load import load_series
from harness.run_grid_compare import TSFM_TAUS, tsfm_ve_series
from precision.mc_critical import MCNull, kupiec_stat_from_hits

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results" / "stage4"
ASSETS = ("btc", "csi300", "eth", "ltc", "xrp")
BASELINE_NAMES = ("caviar_sav", "ewma94", "fhs", "garch_t", "gjr_t",
                  "hs250", "hs500")
ALPHAS = (0.01, 0.025, 0.05)
ERULES = ("e0", "e1", "e2", "e3")

_MC: dict = {}


def mc_p(n: int, alpha: float, stat: float) -> float:
    key = (n, round(alpha, 4))
    if key not in _MC:
        _MC[key] = MCNull("kupiec", n, alpha, B=10000, seed=20260706)
    return _MC[key].p_value(stat)


def uc_cell(y: np.ndarray, v: np.ndarray, alpha: float) -> dict:
    ok = np.isfinite(v)
    y, v = y[ok], v[ok]
    n = len(y)
    hits = (y <= v).astype(int)
    stat = kupiec_stat_from_hits(hits, alpha)
    p = mc_p(n, alpha, stat)
    return {"n": n, "hits": int(hits.sum()), "hit_rate": float(hits.mean()),
            "ucp": p, "uc_pass": bool(p > 0.05)}


def baseline_dates(asset: str) -> set:
    ref = None
    for b in BASELINE_NAMES:
        d = pd.read_parquet(S4 / "baselines" / f"{b}_{asset}.parquet")
        ds = set(pd.to_datetime(d["date"]))
        if ref is None:
            ref = ds
        elif ds != ref:
            sys.exit(f"baseline day sets differ on {asset}: {b}")
    return ref


def main() -> None:
    bt = pd.read_csv(S4 / "backtests.csv")
    rows = []
    t0 = time.time()
    for ai, asset in enumerate(ASSETS):
        fits = pd.read_parquet(S4 / "ctxfits" / f"{asset}.parquet").set_index("t")
        df = load_series(asset)
        x = df[df.attrs.get("column", "logret")].to_numpy()
        bdates = baseline_dates(asset)
        for model in sorted(TSFM_TAUS):
            for er in ERULES:
                ref = bt[(bt.forecaster == model) & (bt.asset == asset)
                         & (bt.erule == er)]
                if not len(ref):
                    continue          # cell absent from the delivered grid
                # per-alpha VaR paths (tsfm_ve_series is per-alpha; loop)
                for a in ALPHAS:
                    sa = tsfm_ve_series(model, asset, er, a, fits, x)
                    full = uc_cell(sa["y"].to_numpy(), sa["v"].to_numpy(), a)
                    r = ref[ref.alpha == a].iloc[0]
                    if full["n"] != int(r["n"]) or \
                            abs(full["hit_rate"] - float(r["hit_rate"])) > 1e-12:
                        sys.exit(f"REPRODUCTION FAIL {model}/{asset}/{er}/{a}: "
                                 f"n {full['n']} vs {r['n']}, hit "
                                 f"{full['hit_rate']} vs {r['hit_rate']}")
                    inb = pd.to_datetime(sa["date"]).isin(bdates).to_numpy()
                    mt = uc_cell(sa["y"].to_numpy()[inb], sa["v"].to_numpy()[inb], a)
                    rows.append({"forecaster": model, "asset": asset,
                                 "erule": er, "alpha": a,
                                 "n_full": full["n"], "hit_full": full["hit_rate"],
                                 "ucp_full": full["ucp"],
                                 "pass_full": full["uc_pass"],
                                 "n_matched": mt["n"],
                                 "hit_matched": mt["hit_rate"],
                                 "ucp_matched": mt["ucp"],
                                 "pass_matched": mt["uc_pass"]})
        el = time.time() - t0
        print(f"[matched-dates] {ai+1}/{len(ASSETS)} {asset} done, "
              f"{el/(ai+1)/60:.1f} min/asset", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(S4 / "matched_dates.csv", index=False)
    flips = out[out.pass_full != out.pass_matched]
    print(f"matched-dates complete: {len(out)} cells "
          f"(reproduction gate passed on every full-sample cell); "
          f"{len(flips)} UC decision flips full->matched", flush=True)
    if len(flips):
        print(flips[["forecaster", "asset", "erule", "alpha", "hit_full",
                     "hit_matched", "pass_full", "pass_matched"]].to_string(),
              flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("matched_dates")
