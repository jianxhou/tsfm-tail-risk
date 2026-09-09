"""Invariant tests for harness/rolling.py (T1-T8 + A3/A4 amendments)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harness.rolling import (Window, align_forecasts, audit_no_lookahead,
                             refit_schedule, rolling_windows, run_rolling)

L = 32


def _df(n=300, start="2015-01-01", values=None):
    dates = pd.bdate_range(start, periods=n)
    x = np.arange(n, dtype=float) if values is None else values
    return pd.DataFrame({"date": dates, "logret": x})


# T1 — identity series: any off-by-one is an exact integer mismatch
def test_t1_identity_alignment():
    df = _df(300)
    rs = rolling_windows(df, value_col="logret", ctx_len=L, oos_start="2015-06-01")
    assert len(rs) > 100
    for w in rs.windows:
        assert w.y == float(w.t)
        assert np.array_equal(w.ctx, np.arange(w.t - L, w.t, dtype=float))
    audit_no_lookahead(df, "logret", rs)


# T2 — strict boundary
def test_t2_oos_boundary_strict():
    df = _df(300)
    with pytest.raises(ValueError, match="allow_late_start"):
        rolling_windows(df, value_col="logret", ctx_len=L, oos_start="2015-01-05")
    rs = rolling_windows(df, value_col="logret", ctx_len=L, oos_start="2015-06-01")
    dates = pd.to_datetime(df["date"])
    first_idx = int((dates >= "2015-06-01").idxmax())
    assert rs.windows[0].t == first_idx >= L


# T3 — NaN skip: exactly L+1 affected windows, others identical to clean run
def test_t3_nan_skip():
    x = np.arange(300, dtype=float)
    k = 200
    x_nan = x.copy(); x_nan[k] = np.nan
    kw = dict(value_col="logret", ctx_len=L, oos_start="2015-06-01")
    clean = rolling_windows(_df(300, values=x), **kw)
    dirty = rolling_windows(_df(300, values=x_nan), **kw)
    assert dirty.meta["n_skipped_nan"] == L + 1          # t in [k, k+L]
    assert {t for t, _, _ in dirty.meta["skipped"]} == set(range(k, k + L + 1))
    kept = {w.t for w in dirty.windows}
    assert kept == {w.t for w in clean.windows} - set(range(k, k + L + 1))
    with pytest.raises(ValueError, match="NaN"):
        rolling_windows(_df(300, values=x_nan), skip_nan_ctx=False, **kw)


# T4 — immutability
def test_t4_ctx_readonly():
    rs = rolling_windows(_df(300), value_col="logret", ctx_len=L, oos_start="2015-06-01")
    with pytest.raises(ValueError):
        rs.windows[0].ctx[0] = 999.0


# T5 — auditor must catch a corrupted (shifted) window
def test_t5_audit_catches_shift():
    df = _df(300)
    rs = rolling_windows(df, value_col="logret", ctx_len=L, oos_start="2015-06-01")
    w = rs.windows[5]
    x = df["logret"].to_numpy()
    bad = Window(t=w.t, date=w.date, ctx=x[w.t - L + 1: w.t + 1].copy(), y=w.y)
    rs.windows[5] = bad
    with pytest.raises(AssertionError, match="ctx mismatch"):
        audit_no_lookahead(df, "logret", rs)


# T6 — align: mismatch refused, happy path round-trips
def test_t6_align():
    rs = rolling_windows(_df(300), value_col="logret", ctx_len=L, oos_start="2015-06-01")
    with pytest.raises(ValueError, match="forecasts for"):
        align_forecasts(rs, [{"v": 1.0}] * (len(rs) - 1))
    frame = align_forecasts(rs, [{"v": float(w.t)} for w in rs.windows])
    assert (frame["v"] == frame["y"]).all()              # by construction here
    assert frame["date"].is_monotonic_increasing


# T7 — refit schedule pattern + A1 metadata
def test_t7_refit_schedule():
    plan = refit_schedule(10, refit_every=3, fit_len=1000)
    assert plan.mask.tolist() == [True, False, False] * 3 + [True]
    assert plan.metadata() == {"fit_len": 1000, "refit_every": 3, "n_refits": 4}


# T8 — real dgs30 data: masked 2006 cross-gap NaN produces the predicted skips
def test_t8_dgs30_real_gap():
    pq = Path(__file__).parent.parent / "data" / "parquet" / "dgs30.parquet"
    if not pq.exists():
        pytest.skip("dgs30 parquet not built locally")
    df = pd.read_parquet(pq)
    x = df["dbp"].to_numpy()
    (nan_idx,) = np.where(np.isnan(x))
    assert len(nan_idx) == 1
    k = int(nan_idx[0])
    ctx = 512
    dates = pd.to_datetime(df["date"])
    first_idx = int((dates >= "2007-01-01").idxmax())
    expected = {t for t in range(max(first_idx, ctx), len(x))
                if t - ctx <= k <= t}
    rs = rolling_windows(df, value_col="dbp", ctx_len=ctx, oos_start="2007-01-01")
    assert {t for t, _, _ in rs.meta["skipped"]} == expected
    audit_no_lookahead(df, "dbp", rs)


# A3 — late-start panel mode
def test_a3_late_start_panel_mode():
    df = _df(300, start="2020-06-01")                    # series starts after oos_start
    kw = dict(value_col="logret", ctx_len=L, oos_start="2016-01-01")
    with pytest.raises(ValueError, match="allow_late_start"):
        rolling_windows(df, **kw)
    rs = rolling_windows(df, allow_late_start=True, **kw)
    assert rs.windows[0].t == L                          # first full-ctx target
    assert rs.meta["late_start"] is True
    assert rs.meta["actual_oos_start"] == str(pd.to_datetime(df["date"]).iloc[L].date())


# A4 — sentinel: a cheating callable cannot reach y through the official channel
def test_a4_cheat_adapter_sentinel():
    rng = np.random.default_rng(20260704)
    x = np.cumsum(rng.standard_normal(600))              # random walk, continuous
    df = _df(600, values=x)
    seen_types = set()

    def cheat(c):
        seen_types.add(type(c).__name__)
        # try every plausible leak: attribute access, oversized view, mutation
        assert not hasattr(c, "y") and not hasattr(c, "date")
        assert len(c) == L
        try:
            c[-1] = 0.0
            mutated = True
        except ValueError:
            mutated = False
        assert not mutated
        return {"pred": float(c[-1])}                    # best a leak-free fn can do

    frame, meta = run_rolling(df, cheat, value_col="logret", ctx_len=L,
                              oos_start="2016-06-01")
    assert seen_types == {"ndarray"}                     # never a Window
    assert (frame["pred"] == frame["y"]).sum() == 0      # zero exact hits on a
    assert (frame["pred"] != frame["y"]).all()           # continuous random walk


# --- I9: calibration_view no-lookahead (Stage 5 repair-arm gate) ---

def _aligned(n=100):
    import pandas as pd
    return pd.DataFrame({"t": np.arange(n), "date": pd.bdate_range("2016-01-01", periods=n),
                         "y": np.arange(n, dtype=float), "v": -np.arange(n, dtype=float),
                         "e": -2.0 * np.arange(n, dtype=float)})


def test_i9_strictly_below_current():
    from harness.rolling import calibration_view
    a = _aligned(100)
    view = calibration_view(a, t_current=50)
    assert view["t"].max() == 49              # never the current row
    assert (view["t"] < 50).all()             # never future
    assert len(view) == 50


def test_i9_window_takes_most_recent():
    from harness.rolling import calibration_view
    a = _aligned(100)
    view = calibration_view(a, t_current=50, window=10)
    assert view["t"].tolist() == list(range(40, 50))


def test_i9_empty_at_start():
    from harness.rolling import calibration_view
    a = _aligned(100)
    assert len(calibration_view(a, t_current=0)) == 0


def test_i9_view_isolated_from_source():
    from harness.rolling import calibration_view
    a = _aligned(100)
    view = calibration_view(a, t_current=50)
    view.loc[view.index[0], "v"] = 999.0      # mutate the returned history
    assert a.loc[0, "v"] == 0.0               # source is unaffected (isolation)


def test_i9_sentinel_repair_cannot_see_future():
    """A repair driven ONLY through calibration_view cannot depend on any y_t' with
    t' >= t. Proof: build two aligned frames identical up to t=60 but with all
    future y flipped; a view-only repair must produce identical output at every
    t <= 60."""
    from harness.rolling import calibration_view
    rng = np.random.default_rng(0)
    a1 = _aligned(120); a1["y"] = rng.standard_normal(120)
    a2 = a1.copy(); a2.loc[a2["t"] >= 61, "y"] = -999.0    # scramble the future only
    def repair_offset(aligned, t):               # a stand-in repair: mean past y
        h = calibration_view(aligned, t, window=30)
        return float(h["y"].mean()) if len(h) else 0.0
    for t in range(0, 61):
        assert repair_offset(a1, t) == repair_offset(a2, t)
