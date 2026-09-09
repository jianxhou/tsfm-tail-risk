"""v2.1 Phase-2 Stage-A acceptance (directive step 1; review #10 §2.7 ①②).

    python -m harness.check_e3_rerun_stability

Against the local v20-baseline:
  1. E0/E1/E2 rows of backtests.csv are BITWISE stable (every numeric
     column; the redefinition must not move any non-E3 cell);
  2. E3 rows: full (erule, alpha) row-set present (the in-run assert also
     enforces this); change summary reported (values are expected to move);
  3. e3_diagnostics tau_mass distribution matches the ruling: continuous
     assets constant 52/512 = 0.1015625; rate-asset means inside the
     0.082-0.088 band; panel minimum 0.0546875 (dgs10).
Exit 1 on any stability/distribution failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
S4 = ROOT / "results/stage4"
BASE = ROOT / "results/stage4.v20-baseline"
KEY = ["forecaster", "asset", "alpha", "erule"]
RATES = ("dgs2", "dgs5", "dgs10", "dgs30")


def main() -> int:
    old = pd.read_csv(BASE / "backtests.csv", float_precision="round_trip")
    new = pd.read_csv(S4 / "backtests.csv", float_precision="round_trip")
    ok = True

    ko = set(map(tuple, old[KEY].itertuples(index=False)))
    kn = set(map(tuple, new[KEY].itertuples(index=False)))
    if ko != kn:
        print(f"FAIL: row-key sets differ (old-only {len(ko-kn)}, new-only {len(kn-ko)})")
        return 1
    o = old.set_index(KEY).sort_index()
    n = new.set_index(KEY).sort_index()
    num_cols = [c for c in o.columns if o[c].dtype.kind in "fi"]

    # 1. E0/E1/E2 bitwise
    stable = o.index.get_level_values("erule") != "e3"
    bad = 0
    for c in num_cols:
        a, b = o.loc[stable, c].to_numpy(), n.loc[stable, c].to_numpy()
        neq = ~((a == b) | (pd.isna(a) & pd.isna(b)))
        if neq.any():
            bad += int(neq.sum())
            for idx in o.loc[stable].index[neq][:3]:
                print(f"  E0/E1/E2 drift {idx} {c}: {o.loc[idx, c]!r} -> {n.loc[idx, c]!r}")
    if bad:
        print(f"FAIL: {bad} E0/E1/E2 cells drifted"); ok = False
    else:
        print(f"1 PASS: all {int(stable.sum())} E0/E1/E2 rows bitwise stable "
              f"({len(num_cols)} numeric cols)")

    # 2. E3 row set + change summary
    e3o, e3n = o.loc[~stable], n.loc[~stable]
    assert len(e3o) == len(e3n)
    ch = 0
    for c in num_cols:
        a, b = e3o[c].to_numpy(), e3n[c].to_numpy()
        ch += int((~((a == b) | (pd.isna(a) & pd.isna(b)))).sum())
    uc_o = e3o["kupiec_p_mc"] > 0.05
    uc_n = e3n["kupiec_p_mc"] > 0.05
    flips = int((uc_o != uc_n).sum())
    print(f"2 INFO: E3 rows {len(e3n)}; {ch} numeric cells moved; "
          f"UC(MC,5%) flips {flips} ({int((~uc_o & uc_n).sum())} fail->pass, "
          f"{int((uc_o & ~uc_n).sum())} pass->fail)")

    # 3. tau_mass distribution per ruling
    lo, rate_means, cont_bad = np.inf, {}, []
    for f in sorted((S4 / "e3_diagnostics").glob("*.csv")):
        if f.stem == "smallsample_v21":
            continue
        d = pd.read_csv(f)
        tm = d["tau_fit"].dropna()
        lo = min(lo, tm.min())
        if f.stem in RATES:
            rate_means[f.stem] = tm.mean()
        else:
            if not np.allclose(tm, 0.1015625, atol=0):
                cont_bad.append(f.stem)
    n_assets = len(list((S4 / "e3_diagnostics").glob("*.csv"))) - 1
    if cont_bad:
        print(f"FAIL: continuous assets with non-constant tau_mass: {cont_bad}")
        ok = False
    else:
        print(f"3a PASS: all {n_assets - len(RATES)} continuous assets constant 0.1015625")
    band_bad = {a: m for a, m in rate_means.items() if not 0.082 <= round(m, 3) <= 0.088}
    if len(rate_means) != 4 or band_bad:
        print(f"FAIL: rate means outside 0.082-0.088 band: {band_bad} "
              f"(have {len(rate_means)}/4)")
        ok = False
    else:
        print(f"3b PASS: rate means {[f'{a}={m:.4f}' for a, m in sorted(rate_means.items())]}")
    if lo != 0.0546875:
        print(f"FAIL: panel min tau_mass {lo!r} != 0.0546875"); ok = False
    else:
        print("3c PASS: panel min tau_mass 0.0546875 (ruling record)")

    print("STAGE-A ACCEPTANCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
