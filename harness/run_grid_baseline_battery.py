"""MC-calibrated battery on the econometric baseline forecasts (checklist item 9,
baseline side). Reuses the exact battery() the TSFM extraction uses, so baseline
and TSFM cells are directly comparable.

    python -m harness.run_grid_baseline_battery

Reads results/stage4/baselines/{baseline}_{asset}.parquet (VaR+ES at 3 alphas;
CAViaR is VaR-only) -> results/stage4/baseline_backtests.csv, one row per
(baseline, asset, alpha). erule column = 'param' (baselines have no E-rule).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from data.load import load_series
from harness.run_grid_extract import acol, battery

ROOT = Path(__file__).resolve().parent.parent
BL = ROOT / "results" / "stage4" / "baselines"
OUT = ROOT / "results" / "stage4"
ALPHAS = (0.01, 0.025, 0.05)
BASELINES = ["garch_t", "gjr_t", "ewma94", "hs250", "hs500", "fhs", "caviar_sav"]


def main() -> None:
    dm = yaml.safe_load((ROOT / "data" / "data_manifest.yaml").read_text())
    rows = []
    for pq in sorted(BL.glob("*.parquet")):
        name = pq.stem
        bl = next((b for b in BASELINES if name.startswith(b + "_")), None)
        if bl is None:
            continue
        asset = name[len(bl) + 1:]
        d = pd.read_parquet(pq)
        group = dm[asset]["group"]
        for a in ALPHAS:
            c = acol(a)
            vcol, ecol = f"v{c}", f"e{c}"
            if vcol not in d:
                continue
            e = d[ecol].to_numpy() if ecol in d and d[ecol].notna().any() else None
            b = battery(d["y"].to_numpy(), d[vcol].to_numpy(), e, a)
            rows.append({"forecaster": bl, "asset": asset, "alpha": a,
                         "erule": "param", "group": group, **b})
        print(f"[bl-battery] {bl}/{asset}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "baseline_backtests.csv", index=False)
    print(f"baseline battery complete: {len(rows)} cells", flush=True)


if __name__ == "__main__":
    main()
    # G6 freshness stamp (v2.1.1, review #11): record generation-time input
    # hashes; loud no-op inside distributed package copies (no .git)
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from harness.freshness import stamp_safe
    stamp_safe("baseline_battery")
