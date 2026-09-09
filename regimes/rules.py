"""FROZEN regime episode rules (hard rule 6, narrowed per docs/rule6_decision.md).

Every constant here is verbatim from docs/stage5_design.md §B (v2.1, signer-approved
2026-07-07). This module is pure and deterministic: given the frozen vol-index data
it emits a CLOSED episode set over the OOS window — no fitting, no discarding, no
dependence on any viewed result. It is committed BEFORE any event-time or
episode-conditioned computation beyond the disclosed pilot peek. Editing any
constant after the freeze commit is a hard-rule-6 violation requiring a CHANGELOG
deviation entry flagged to the signer.

Episode rule (frozen):
  onset  = first day of a run of >= SUSTAIN (3) consecutive closes with
           index > max(MULT (1.5) * trailing-MEDIAN_WIN (250)-day median, floor)
  floors = VIX 25, MOVE 100, crypto-RV 75 (% annualized)
  span   = onset .. onset + SPAN (60) trading days (60 days incl. onset)
  merge  = onsets within MERGE (20) trading days collapse to the earlier
  calm   = all days inside runs of >= CALM_RUN (60) consecutive days with
           index < trailing-250-day median (pooled, no single-window pick)
Index -> asset-class map: VIX for equity_index/equity_single/fx/commodity;
MOVE for rates; crypto-RV for crypto. crypto-RV = cross-sectional mean of the
available crypto assets' trailing RV_WIN (30)-day annualized realized vol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---- FROZEN CONSTANTS (verbatim, design §B) ----
MULT = 1.5
MEDIAN_WIN = 250
SUSTAIN = 3
SPAN = 60
MERGE = 20
CALM_RUN = 60
RV_WIN = 30
TRADING_YEAR = 252
FLOORS = {"vix": 25.0, "move": 100.0, "crypto_rv": 75.0}
CLASS_INDEX = {"equity_index": "vix", "equity_single": "vix", "fx": "vix",
               "commodity": "vix", "rates": "move", "crypto": "crypto_rv"}

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "parquet"
CRYPTO_ASSETS = ["btc", "eth", "ltc", "xrp"]


def _level(index_id: str) -> pd.Series:
    """Frozen index level series indexed by date."""
    if index_id in ("vix", "move"):
        d = pd.read_parquet(PARQUET / f"{index_id}.parquet")
        return pd.Series(d["level"].to_numpy(), index=pd.to_datetime(d["date"]))
    if index_id == "crypto_rv":
        # cross-sectional mean of available crypto assets' trailing-30d annualized RV
        rvs = []
        for a in CRYPTO_ASSETS:
            p = PARQUET / f"{a}.parquet"
            if not p.exists():
                continue
            d = pd.read_parquet(p)
            r = pd.Series(d["logret"].to_numpy(), index=pd.to_datetime(d["date"]))
            rv = r.rolling(RV_WIN).std() * np.sqrt(TRADING_YEAR)   # logret already x100
            rvs.append(rv)
        m = pd.concat(rvs, axis=1).mean(axis=1, skipna=True)
        return m.dropna()
    raise ValueError(f"unknown index {index_id}")


def _episodes_for_index(index_id: str) -> list[dict]:
    lvl = _level(index_id).sort_index()
    med = lvl.rolling(MEDIAN_WIN).median()
    thr = np.maximum(MULT * med, FLOORS[index_id])
    above = (lvl > thr) & med.notna()
    a = above.to_numpy()
    # runs of >= SUSTAIN consecutive True -> onset = first index of the run
    onsets = []
    i, n = 0, len(a)
    while i < n:
        if a[i]:
            j = i
            while j < n and a[j]:
                j += 1
            if j - i >= SUSTAIN:
                onsets.append(i)
            i = j
        else:
            i += 1
    # merge onsets within MERGE trading days
    merged = []
    for o in onsets:
        if merged and o - merged[-1] < MERGE:
            continue
        merged.append(o)
    dates = lvl.index
    eps = []
    for k, o in enumerate(merged):
        end = min(o + SPAN - 1, n - 1)
        eps.append({"index": index_id, "episode": f"{index_id}_ep{k+1}",
                    "onset": str(dates[o].date()), "end": str(dates[end].date()),
                    "onset_level": round(float(lvl.iloc[o]), 1),
                    "onset_threshold": round(float(thr.iloc[o]), 1),
                    "n_days": end - o + 1})
    return eps


def _calm_for_index(index_id: str) -> list[dict]:
    lvl = _level(index_id).sort_index()
    med = lvl.rolling(MEDIAN_WIN).median()
    below = (lvl < med) & med.notna()
    b = below.to_numpy()
    dates = lvl.index
    runs = []
    i, n = 0, len(b)
    while i < n:
        if b[i]:
            j = i
            while j < n and b[j]:
                j += 1
            if j - i >= CALM_RUN:
                runs.append({"index": index_id, "start": str(dates[i].date()),
                             "end": str(dates[j - 1].date()), "n_days": j - i})
            i = j
        else:
            i += 1
    return runs


def episode_set() -> pd.DataFrame:
    """The complete, closed episode set across all indices (frozen rule output)."""
    rows = []
    for idx in ("vix", "move", "crypto_rv"):
        rows.extend(_episodes_for_index(idx))
    return pd.DataFrame(rows)


def calm_set() -> pd.DataFrame:
    rows = []
    for idx in ("vix", "move", "crypto_rv"):
        rows.extend(_calm_for_index(idx))
    return pd.DataFrame(rows)


def class_index(asset_group: str) -> str:
    return CLASS_INDEX[asset_group]


if __name__ == "__main__":
    eps = episode_set()
    calm = calm_set()
    print(f"episodes: {len(eps)}  calm runs: {len(calm)}")
    print(eps.to_string(index=False))
