"""Quality-control checks for fetched series.

Stage-1 QC checks: missing days, splits, outliers, timezone.
Thresholds are group-specific (crypto trades 7d/week, single stocks gap harder
than indices, rates are measured in bp) — see PROFILES. Pure functions: inputs
in → dict of findings out. Report rendering lives in build.py.
"""

from __future__ import annotations

import pandas as pd

# Per-group QC expectations.
#   outlier      : |return| beyond this is flagged for eyeballing (genuine crash days
#                  are expected to appear here — the flag means "verify vs known event")
#   gap_days     : calendar gap larger than this is listed
#   weekend_ok   : weekend bars are normal (crypto 7d calendar; FX Sunday opens)
#   bpy          : plausible bars-per-year range
#   stale_run    : run of >= this many identical closes is flagged
#   events_ok    : split/dividend events are expected (stocks, ETF proxy)
#   unit         : label for report tables
PROFILES = {
    "equity_index":  dict(outlier=15.0, gap_days=7,  weekend_ok=False, bpy=(238, 262), stale_run=5,  events_ok=False, unit="logret×100"),
    "equity_single": dict(outlier=25.0, gap_days=7,  weekend_ok=False, bpy=(238, 262), stale_run=5,  events_ok=True,  unit="logret×100"),
    "fx":            dict(outlier=6.0,  gap_days=7,  weekend_ok=True,  bpy=(238, 320), stale_run=8,  events_ok=False, unit="logret×100"),
    "crypto":        dict(outlier=40.0, gap_days=3,  weekend_ok=True,  bpy=(350, 370), stale_run=3,  events_ok=False, unit="logret×100"),
    "commodity":     dict(outlier=25.0, gap_days=7,  weekend_ok=False, bpy=(230, 265), stale_run=5,  events_ok=False, unit="logret×100"),
    "rates":         dict(outlier=50.0, gap_days=7,  weekend_ok=False, bpy=(230, 262), stale_run=15, events_ok=False, unit="Δbp"),
    # vol indices are LEVELS (VIX points / MOVE bps), not returns — the outlier/stale
    # thresholds are relaxed and the "unit" flags the level semantics; structural
    # checks (dups, gaps, timezone, calendar coverage) still apply.
    "vol_index":     dict(outlier=1e9, gap_days=7,  weekend_ok=False, bpy=(230, 262), stale_run=15, events_ok=False, unit="level"),
}


def qc_series(sid: str, df: pd.DataFrame, ret: pd.Series, tz_name: str,
              splits: int, dividends: int, profile: dict) -> dict:
    """QC one series. `df` has date/close; `ret` is the transformed series
    (logret×100 or Δbp), index-aligned to date[1:]."""
    dates = pd.to_datetime(df["date"])
    f: dict = {"id": sid}

    f["n_obs"] = len(df)
    f["start"] = str(dates.iloc[0].date())
    f["end"] = str(dates.iloc[-1].date())
    f["timezone"] = tz_name
    f["unit"] = profile["unit"]

    # --- structural integrity ---
    f["dup_dates"] = int(dates.duplicated().sum())
    f["nonmonotonic"] = bool((dates.diff().dt.days.iloc[1:] <= 0).any())
    f["bad_close"] = int(df["close"].isna().sum())
    if profile["unit"].startswith("logret"):
        f["bad_close"] += int((df["close"] <= 0).sum())  # yields may be ≤0, prices may not
    f["weekend_bars"] = int(dates.dt.dayofweek.isin([5, 6]).sum())

    # --- calendar coverage ---
    gaps = dates.diff().dt.days
    big = gaps[gaps > profile["gap_days"]]
    f["gaps"] = [
        (str(dates.iloc[i - 1].date()), str(dates.iloc[i].date()), int(g))
        for i, g in zip(big.index, big.values)
    ]
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    f["bars_per_year"] = round(len(df) / years, 1) if years > 0 else float("nan")

    # --- transformed-series stats ---
    f["ret_mean"] = float(ret.mean())
    f["ret_std"] = float(ret.std())
    f["ret_skew"] = float(ret.skew())
    f["ret_kurt"] = float(ret.kurt())  # excess
    f["zero_ret"] = int((ret == 0).sum())
    f["masked"] = int(ret.isna().sum())   # cross-gap diffs set to NaN by the transform

    top = ret.abs().sort_values(ascending=False).head(5)
    f["top_moves"] = [
        (str(pd.to_datetime(df["date"].iloc[i]).date()), round(float(ret.loc[i]), 2))
        for i in top.index
    ]
    f["outliers"] = [t for t in f["top_moves"] if abs(t[1]) > profile["outlier"]]

    # stale runs: consecutive identical closes
    same = (df["close"].diff() == 0).astype(int)
    run = max_run = 0
    for v in same:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    f["max_stale_run"] = int(max_run)

    f["splits"] = splits
    f["dividends"] = dividends

    f["flags"] = _flags(f, profile)
    return f


def _flags(f: dict, p: dict) -> list[str]:
    flags = []
    if f["dup_dates"]:
        flags.append(f"{f['dup_dates']} duplicate dates")
    if f["nonmonotonic"]:
        flags.append("dates not strictly increasing")
    if f["bad_close"]:
        flags.append(f"{f['bad_close']} invalid closes")
    if f["weekend_bars"] and not p["weekend_ok"]:
        flags.append(f"{f['weekend_bars']} weekend bars (timezone mapping suspect)")
    if f["outliers"]:
        flags.append(f"|ret|>{p['outlier']}: {f['outliers']} — verify vs known events")
    if f["max_stale_run"] >= p["stale_run"]:
        flags.append(f"stale run of {f['max_stale_run']} identical closes")
    if f["splits"] and not p["events_ok"]:
        flags.append(f"{f['splits']} split events on a non-equity series")
    lo, hi = p["bpy"]
    if not (lo <= f["bars_per_year"] <= hi):
        flags.append(f"{f['bars_per_year']} bars/yr outside [{lo},{hi}]")
    if len(f["gaps"]) > 3:
        flags.append(f"{len(f['gaps'])} calendar gaps >{p['gap_days']}d")
    return flags
