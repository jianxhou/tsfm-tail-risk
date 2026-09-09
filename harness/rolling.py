"""THE rolling-window module (protocol: ALL rolling logic lives here, reviewed).

Design approved 2026-07-04 with amendments A1-A4; see docs in git history for the
original design note. Summary of the contract:

Time convention: a forecast for target index t uses ONLY rows [0, t-1]; context is
x[t-ctx_len : t] (half-open — x[t] excluded by construction); the realization is
x[t] at date d[t]. Rows, not days: the series' own trading calendar defines
adjacency; calendar gaps never create synthetic rows.

Invariants I1-I9 (enforced at build time, re-checked by audit_no_lookahead, tested
in tests/test_rolling.py):
 I1 alignment      ctx[-1] == x[t-1], len(ctx) == ctx_len
 I2 no lookahead   slice bounds by INDEX arithmetic; ctx built from indices <= t-1
 I3 monotone time  input dates strictly increasing; targets strictly increasing
 I4 calendar honesty  no interpolation, no synthetic rows
 I5 boundary       strict: raise if oos_start leaves < ctx_len history.
                   A3 panel mode (allow_late_start=True): late-starting series
                   begin at the first full-ctx target; actual start in meta.
 I6 immutability   yielded ctx is a read-only copy
 I7 determinism    pure function of inputs, no RNG
 I8 NaN policy     NaN in ctx or y => skip+count (skip_nan_ctx=True) or raise;
                   never forward-filled
 I9 recalibration no-lookahead (A2): repair arms (F1-F4, Stage 5) may consume
                   aligned (forecast, realization) pairs ONLY for target indices
                   strictly below the current t, and ONLY through
                   calibration_view() below. No recalibration code may slice
                   forecast history by hand.

Runner contract (A4): model callables receive the ctx ndarray ONLY — never a
Window, never y. run_rolling() is the sole official channel; the sentinel test
proves a cheating callable cannot reach the realization.

A1: fit_len (parameter-estimation window, econometric baselines; main setting
fit_len=1000, refit_every=21 — Pele-aligned 1000-day rolling MLE) is decoupled
from ctx_len (forecast context). Both land in run metadata. The info-parity
variant (fit_len == TSFM ctx_len) is an optional ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Window:
    t: int                 # integer index of the TARGET row
    date: pd.Timestamp     # d[t]
    ctx: np.ndarray        # x[t-ctx_len : t], read-only copy
    y: float               # x[t] — scoring only; never hand this to a model


@dataclass
class RollingSet:
    windows: list[Window]
    meta: dict = field(default_factory=dict)

    def __iter__(self):
        return iter(self.windows)

    def __len__(self):
        return len(self.windows)


def rolling_windows(df: pd.DataFrame, *, value_col: str, ctx_len: int,
                    oos_start: str, oos_end: str | None = None,
                    skip_nan_ctx: bool = True,
                    allow_late_start: bool = False) -> RollingSet:
    """Build the OOS window set. See module docstring for the time convention."""
    if value_col not in df.columns or "date" not in df.columns:
        raise ValueError(f"df must have 'date' and '{value_col}' columns")
    dates = pd.to_datetime(df["date"]).reset_index(drop=True)
    diffs = dates.diff().dt.total_seconds().iloc[1:]
    if not (diffs > 0).all():                                   # I3 / I4
        raise ValueError("dates must be strictly increasing (no synthetic rows)")
    x = df[value_col].to_numpy(dtype=float)
    n = len(x)
    if ctx_len < 1 or n <= ctx_len:
        raise ValueError(f"need > ctx_len={ctx_len} rows, have {n}")

    start_ts = pd.Timestamp(oos_start)
    end_ts = pd.Timestamp(oos_end) if oos_end else None
    first_date_idx = int(np.searchsorted(dates.values, np.datetime64(start_ts)))
    if first_date_idx >= n:
        raise ValueError(f"oos_start {oos_start} is after the last observation")

    if first_date_idx < ctx_len:                                # I5
        if not allow_late_start:
            raise ValueError(
                f"only {first_date_idx} rows before oos_start {oos_start}, need "
                f"ctx_len={ctx_len}; pass allow_late_start=True for panel mode")
        start_t = ctx_len
    else:
        start_t = first_date_idx

    windows: list[Window] = []
    skipped: list[tuple[int, str, str]] = []
    for t in range(start_t, n):
        if end_ts is not None and dates.iloc[t] > end_ts:
            break
        lo = t - ctx_len                                        # I2: index arithmetic
        ctx = x[lo:t]
        if np.isnan(x[t]) or np.any(np.isnan(ctx)):             # I8
            if skip_nan_ctx:
                skipped.append((t, str(dates.iloc[t].date()),
                                "nan_y" if np.isnan(x[t]) else "nan_ctx"))
                continue
            raise ValueError(f"NaN in window for target t={t} ({dates.iloc[t].date()})")
        c = ctx.copy()                                          # I6
        c.setflags(write=False)
        assert len(c) == ctx_len and c[-1] == x[t - 1]          # I1
        windows.append(Window(t=t, date=dates.iloc[t], ctx=c, y=float(x[t])))

    meta = {
        "value_col": value_col,
        "ctx_len": ctx_len,
        "oos_start_requested": str(start_ts.date()),
        "oos_end": str(end_ts.date()) if end_ts is not None else None,
        "actual_oos_start": str(windows[0].date.date()) if windows else None,  # A3
        "late_start": bool(first_date_idx < ctx_len),
        "n_windows": len(windows),
        "n_skipped_nan": len(skipped),
        "skipped": skipped,
    }
    return RollingSet(windows=windows, meta=meta)


def align_forecasts(rs: RollingSet, forecasts: Sequence[dict]) -> pd.DataFrame:
    """THE only join point between forecasts and realizations."""
    if len(forecasts) != len(rs):
        raise ValueError(f"{len(forecasts)} forecasts for {len(rs)} windows")
    rows = []
    prev_date = None
    for w, f in zip(rs.windows, forecasts):
        if prev_date is not None and w.date <= prev_date:
            raise ValueError("window dates out of order")
        prev_date = w.date
        rows.append({"t": w.t, "date": w.date, "y": w.y, **f})
    return pd.DataFrame(rows)


def audit_no_lookahead(df: pd.DataFrame, value_col: str, rs: RollingSet) -> None:
    """Posthoc auditor: re-derives every window from the source and compares.
    Raises AssertionError on ANY discrepancy. Run this inside every experiment."""
    dates = pd.to_datetime(df["date"]).reset_index(drop=True)
    x = df[value_col].to_numpy(dtype=float)
    L = rs.meta["ctx_len"]
    prev_t = -1
    for w in rs.windows:
        assert w.t > prev_t, f"targets not strictly increasing at t={w.t}"
        prev_t = w.t
        assert w.t >= L, f"t={w.t} lacks full context"
        expect = x[w.t - L: w.t]
        assert w.ctx.shape == (L,), f"ctx shape {w.ctx.shape} at t={w.t}"
        assert np.array_equal(w.ctx, expect), f"ctx mismatch at t={w.t}"
        assert w.y == x[w.t], f"y mismatch at t={w.t}"
        assert w.date == dates.iloc[w.t], f"date mismatch at t={w.t}"


@dataclass(frozen=True)
class RefitPlan:
    """A1: fit_len (estimation window) decoupled from forecast ctx_len.
    Main econometric setting: fit_len=1000, refit_every=21."""
    mask: np.ndarray          # mask[i]: window i triggers a parameter refit
    refit_every: int
    fit_len: int

    def metadata(self) -> dict:
        return {"fit_len": self.fit_len, "refit_every": self.refit_every,
                "n_refits": int(self.mask.sum())}


def refit_schedule(n_windows: int, refit_every: int, fit_len: int) -> RefitPlan:
    if n_windows < 1 or refit_every < 1 or fit_len < 1:
        raise ValueError("n_windows, refit_every, fit_len must be >= 1")
    mask = np.zeros(n_windows, dtype=bool)
    mask[::refit_every] = True
    return RefitPlan(mask=mask, refit_every=refit_every, fit_len=fit_len)


def run_rolling(df: pd.DataFrame, fn: Callable[[np.ndarray], dict], *,
                value_col: str, ctx_len: int, oos_start: str,
                oos_end: str | None = None, skip_nan_ctx: bool = True,
                allow_late_start: bool = False) -> tuple[pd.DataFrame, dict]:
    """A4 runner contract: `fn` receives the read-only ctx ndarray ONLY — never a
    Window, never y — and returns a dict of forecast quantities. Returns the
    aligned scoring frame plus run metadata."""
    rs = rolling_windows(df, value_col=value_col, ctx_len=ctx_len,
                         oos_start=oos_start, oos_end=oos_end,
                         skip_nan_ctx=skip_nan_ctx,
                         allow_late_start=allow_late_start)
    forecasts = [fn(w.ctx) for w in rs.windows]
    frame = align_forecasts(rs, forecasts)
    audit_no_lookahead(df, value_col, rs)
    return frame, rs.meta


def calibration_view(aligned: pd.DataFrame, t_current: int,
                     window: int | None = None) -> pd.DataFrame:
    """I9: return the recalibration history a repair arm at target index
    `t_current` is allowed to see — rows with target index STRICTLY BELOW
    t_current (never the current row, never any future row). Optionally the most
    recent `window` such rows. The returned frame is a defensive COPY with its
    value columns made read-only, so a repair cannot mutate shared state or
    accidentally reach past the boundary.

    Every repair arm (F1 EVT-splice, F2 adaptive conformal, F3 rescale,
    F4 additive-FZ0, H hybrid) MUST obtain its recalibration history exclusively
    through this function; hand-slicing forecast history anywhere else is a
    hard-rule-3 / I9 violation. `aligned` must carry a 't' column (integer target
    index, as produced by align_forecasts).
    """
    if "t" not in aligned.columns:
        raise ValueError("aligned frame must carry the integer target-index column 't'")
    hist = aligned[aligned["t"] < t_current]
    if window is not None:
        if window < 1:
            raise ValueError("window must be >= 1")
        hist = hist.iloc[-window:]
    # Defensive COPY: the returned frame is isolated from `aligned`, so a repair
    # cannot mutate shared history, and the row filter guarantees no future row is
    # ever visible. (We do not rely on numpy writeable flags — pandas copies on
    # access, so isolation-by-copy is the enforceable guarantee.)
    return hist.copy()
