"""v2.1 Phase-2 MANDATORY STOP-POINT check (review #10 §2.8).

    python -m harness.check_matched_invariant

Diffs results/stage4/repair_backtests_matched.csv against the local
v20-baseline copy:
  (a) the `n` column is unchanged on EVERY row (any E3-VaR-NaN-driven mask
      movement would surface here — that would be a BREAK);
  (b) the E3-independent arm rows are numerically unchanged bit-for-bit:
      {*+H, *+F1, garch_evt, zero_loc, const_loc, *+X4, fhs, gjr_t}
      (H/F1 do not consume E3 values; the matched-day mask depends only on
      the always-finite V columns);
  expected transmission (NOT a break, reported): {*+unrepaired, *+F2*,
  *+F3, *+F4} value changes.
Exit 0 = invariant proven (proof block printed for the impact report);
exit 1 = BREAK — stop the run and escalate for adjudication (directive step 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NEW = ROOT / "results/stage4/repair_backtests_matched.csv"
OLD = ROOT / "results/stage4.v20-baseline/repair_backtests_matched.csv"
KEY = ["forecaster", "asset", "alpha", "erule", "context_model"]
INVARIANT_EXACT = {"garch_evt", "zero_loc", "const_loc", "fhs", "gjr_t"}
INVARIANT_SUFFIX = ("+H", "+F1", "+X4")


def arm_class(f: str) -> str:
    if f in INVARIANT_EXACT or f.endswith(INVARIANT_SUFFIX):
        return "invariant"
    return "transmission"        # *+unrepaired, *+F2, *+F2g05/20, *+F3, *+F4


def main() -> int:
    old = pd.read_csv(OLD, float_precision="round_trip")
    new = pd.read_csv(NEW, float_precision="round_trip")
    for d in (old, new):
        d["context_model"] = d["context_model"].fillna("")
    ok = True

    # row-key identity
    ko = set(map(tuple, old[KEY].itertuples(index=False)))
    kn = set(map(tuple, new[KEY].itertuples(index=False)))
    if ko != kn:
        print(f"BREAK: row-key sets differ (old-only {len(ko-kn)}, "
              f"new-only {len(kn-ko)})")
        for k in sorted(ko ^ kn)[:10]:
            print("   ", k)
        return 1
    o = old.set_index(KEY).sort_index()
    n = new.set_index(KEY).sort_index()

    # (a) n column unchanged on EVERY row
    bad_n = (o["n"] != n["n"])
    if bad_n.any():
        print(f"BREAK (a): n changed on {int(bad_n.sum())} rows")
        print(o[bad_n].join(n[bad_n], lsuffix="_old", rsuffix="_new")
              [["n_old", "n_new"]].head(10))
        ok = False
    else:
        print(f"(a) PASS: n unchanged on all {len(o)} rows")

    # (b) invariant arms bitwise
    num_cols = [c for c in o.columns if o[c].dtype.kind in "fi"]
    inv_mask = o.index.get_level_values(0).map(arm_class) == "invariant"
    diffs = []
    for c in num_cols:
        a, b = o.loc[inv_mask, c].to_numpy(), n.loc[inv_mask, c].to_numpy()
        neq = ~((a == b) | (pd.isna(a) & pd.isna(b)))
        if neq.any():
            for idx in o.loc[inv_mask].index[neq][:5]:
                diffs.append((c, idx, o.loc[idx, c], n.loc[idx, c]))
    if diffs:
        print(f"BREAK (b): {len(diffs)}+ invariant-arm cells changed:")
        for c, idx, va, vb in diffs[:15]:
            print(f"    {idx} {c}: {va!r} -> {vb!r}")
        ok = False
    else:
        print(f"(b) PASS: all {int(inv_mask.sum())} invariant-arm rows "
              f"({sorted(set(f for f in o.index.get_level_values(0) if arm_class(f)=='invariant'))}) "
              f"bitwise unchanged across {len(num_cols)} numeric columns")

    # expected transmission summary (report material, not a gate)
    tr_mask = ~inv_mask
    changed = 0
    for c in num_cols:
        a, b = o.loc[tr_mask, c].to_numpy(), n.loc[tr_mask, c].to_numpy()
        changed += int((~((a == b) | (pd.isna(a) & pd.isna(b)))).sum())
    print(f"transmission arms (*+unrepaired/F2*/F3/F4): {changed} numeric "
          f"cell changes across {int(tr_mask.sum())} rows (expected, per "
          f"review §2.8)")

    print("MATCHED INVARIANT:", "PROVEN" if ok else "BROKEN — STOP AND REPORT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
