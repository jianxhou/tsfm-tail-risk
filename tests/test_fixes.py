"""Reference tests for repair arms F1-F4+H and the GARCH-EVT comparator
(amendment B: green before grid). Coverage/mechanism assertions on synthetic
data with known properties, plus I9 compliance."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fixes import arms
from synth.paths import garch_t_path


def _garch_aligned(seed=1, n=1500, ctx=512, gamma=0.0):
    """Aligned frame from a GARCH-t path: a too-narrow normal-tail v/e (to be
    repaired) plus the model 'grid' set to the true conditional quantiles' scale."""
    y, sig, nu = garch_t_path(seed, n=n, gjr_gamma=gamma)
    # emulate a TSFM central grid tracking scale well: q50=0, q25/q75 = +/-0.674*sig
    q50 = np.zeros(n); q25 = -0.674 * sig; q75 = 0.674 * sig
    # a deliberately too-narrow NORMAL-tail E3-stand-in v/e (over-violates -> repair target)
    a = 0.05
    v = sig * stats.norm.ppf(a); e = -sig * stats.norm.pdf(stats.norm.ppf(a)) / a
    df = pd.DataFrame({"t": np.arange(n), "y": y, "v": v, "e": e,
                       "q25": q25, "q50": q50, "q75": q75, "sig": sig})
    return df.iloc[ctx:].reset_index(drop=True), sig[ctx:], nu


def test_f1_evt_repair_near_nominal_on_garch():
    """F1 (TSFM-filtered EVT) on a GARCH-t path recovers near-nominal 5% coverage."""
    df, sig, nu = _garch_aligned(seed=2)
    v, e = arms.f1_tsfm_filtered_evt(df, 0.05)
    ok = np.isfinite(v)
    hit = (df["y"].to_numpy()[ok] <= v[ok]).mean()
    assert abs(hit - 0.05) < 0.02, f"F1 hit {hit}"
    assert (e[ok] <= v[ok]).all()          # ES convention preserved


def test_f3_rescale_widens_too_narrow_input():
    """The normal-tail v over-violates on t-innovations; F3 must pick k>1 and
    bring coverage down toward nominal."""
    df, sig, nu = _garch_aligned(seed=3)
    base_hit = (df["y"] <= df["v"]).mean()
    v, e = arms.f3_rescale(df, 0.05)
    ok = np.isfinite(v)
    rep_hit = (df["y"].to_numpy()[ok] <= v[ok]).mean()
    assert base_hit > 0.05                  # input over-violates
    assert rep_hit < base_hit               # rescale reduces violations


def test_f2_conformal_drives_toward_nominal():
    df, sig, nu = _garch_aligned(seed=4)
    s_hat = float(np.median((df["q75"] - df["q25"]) / 1.349))
    v, e = arms.f2_adaptive_conformal(df, 0.05, s_hat)
    base = abs((df["y"] <= df["v"]).mean() - 0.05)
    rep = abs((df["y"].to_numpy() <= v).mean() - 0.05)
    assert rep < base                       # conformal improves coverage


def test_f4_optimizer_reduces_in_window_fz0_and_valid():
    """F4's GUARANTEED property is in-window FZ0 minimization; out-of-sample it is
    not guaranteed to beat base (esp. for a SCALE miscalibration an additive shift
    cannot fix — a real F3-vs-F4 distinction). Test the in-window guarantee + that
    the applied series stays a valid ES forecast (e<v<0), out-of-sample-competitive."""
    df, sig, nu = _garch_aligned(seed=5)
    s_hat = float(np.median((df["q75"] - df["q25"]) / 1.349))
    # in-window guarantee: fitted (q,r) has lower FZ0 than (0,0) on the fit window
    yh, vh, eh = df["y"].to_numpy()[:500], df["v"].to_numpy()[:500], df["e"].to_numpy()[:500]
    from scipy.optimize import minimize
    f0 = arms._fz0_shift([0.0, 0.0], yh, vh, eh, 0.05)
    best = minimize(arms._fz0_shift, [0.0, 0.0], args=(yh, vh, eh, 0.05),
                    method="Nelder-Mead", options={"maxiter": 1500}).fun
    assert best <= f0 + 1e-9
    # applied out-of-sample: valid convention, and competitive (within 5% of base)
    v, e = arms.f4_additive_fz0(df, 0.05, s_hat)
    from backtests.scores import fz0
    y = df["y"].to_numpy(); ok = np.isfinite(v) & (e < v) & (e < 0)
    assert ok.mean() > 0.95
    base = fz0(y, df["v"].to_numpy(), df["e"].to_numpy(), 0.05).mean()
    rep = fz0(y[ok], v[ok], e[ok], 0.05).mean()
    assert rep <= 1.05 * base


def test_h_hybrid_and_garch_evt_run_and_cover():
    df, sig, nu = _garch_aligned(seed=6)
    mu = np.zeros(len(df))
    vh, eh = arms.h_hybrid(df, sig, 0.05)
    vg, eg = arms.garch_evt(df, mu, sig, 0.05)
    for v, e in ((vh, eh), (vg, eg)):
        ok = np.isfinite(v)
        assert ok.sum() > 100
        assert abs((df["y"].to_numpy()[ok] <= v[ok]).mean() - 0.05) < 0.025
        assert (e[ok] <= v[ok]).all()


def test_v2_control_arms_run_and_cover():
    """(v2.0 review #9 §9.1 control arms) X4 (GARCH loc + TSFM scale),
    zero-location and constant-location (GARCH scale) run, keep the ES
    convention, and reach near-nominal 5% coverage on a GARCH-t path — the
    same property standard applied to H / GARCH-EVT."""
    df, sig, nu = _garch_aligned(seed=9)
    mu = np.zeros(len(df))
    for d in (arms.x4_multi(df, mu, (0.05,)),
              arms.zero_loc_multi(df, sig, (0.05,)),
              arms.const_loc_multi(df, sig, (0.05,))):
        v, e = d[0.05]
        ok = np.isfinite(v)
        assert ok.sum() > 100
        assert abs((df["y"].to_numpy()[ok] <= v[ok]).mean() - 0.05) < 0.025
        assert (e[ok] <= v[ok]).all()


def _shat_first500(d: pd.DataFrame) -> float:
    """Driver convention (v2.0 P0-2, design §A): ŝ = median central-grid scale
    over the FIRST 500 rows of the aligned frame, computed once, fixed."""
    return float(np.median(((d["q75"] - d["q25"]) / 1.349).iloc[:500]))


def test_shat_first_window_no_lookahead_sentinel():
    """(v2.0 P0-2) Upgraded leak sentinel: ŝ is computed INSIDE the tested call
    from the frame's first 500 rows (the driver convention), not passed in
    precomputed — the old sentinel fed the same ŝ to both calls and was
    structurally blind to a full-frame ŝ.

    (A) Perturb y after row 500: ŝ and the F2/F4 outputs on rows [:501] must be
        unchanged (later rows legitimately differ through the online recursion).
    (B) Perturb ONLY the central grid (q25/q75) after row 500: ŝ is the sole
        channel from the grid into F2/F4, so the ENTIRE output series must be
        unchanged. A full-frame ŝ fails (B) outright and (A) via gamma."""
    df, sig, nu = _garch_aligned(seed=8)
    assert len(df) > 700
    dfA = df.copy(); dfA.loc[dfA.index > 500, "y"] = -999.0
    dfB = df.copy()
    dfB.loc[dfB.index > 500, "q25"] = dfB.loc[dfB.index > 500, "q25"] * 3.0
    dfB.loc[dfB.index > 500, "q75"] = dfB.loc[dfB.index > 500, "q75"] * 3.0
    calls = {
        "F2": lambda d: arms.f2_adaptive_conformal(d, 0.05, _shat_first500(d)),
        "F4": lambda d: arms.f4_additive_fz0(d, 0.05, _shat_first500(d)),
    }
    assert _shat_first500(df) == _shat_first500(dfA) == _shat_first500(dfB)
    for name, fn in calls.items():
        v0, e0 = fn(df)
        vA, eA = fn(dfA)
        assert np.allclose(v0[:501], vA[:501], equal_nan=True), f"{name} v leaks (A)"
        assert np.allclose(e0[:501], eA[:501], equal_nan=True), f"{name} e leaks (A)"
        vB, eB = fn(dfB)
        assert np.allclose(v0, vB, equal_nan=True), f"{name} v leaks via ŝ (B)"
        assert np.allclose(e0, eB, equal_nan=True), f"{name} e leaks via ŝ (B)"


def test_arms_are_i9_clean_no_future_leak():
    """Scramble the future of an aligned frame; EVERY arm's output on rows
    strictly before the scramble point must be unchanged (windows are strictly
    below t, so rows [:k] never see row k+). Extended from F3-only to all arms
    per closeout audit A14 (CHANGELOG 2026-07-08)."""
    df, sig, nu = _garch_aligned(seed=7)
    k = 700
    df2 = df.copy(); df2.loc[df2["t"] >= df["t"].iloc[k], "y"] = -999.0
    s_hat = float(np.median((df["q75"] - df["q25"]) / 1.349))
    mu = np.zeros(len(df))
    calls = {
        "F1": lambda d: arms.f1_tsfm_filtered_evt(d, 0.05),
        "F2": lambda d: arms.f2_adaptive_conformal(d, 0.05, s_hat),
        "F2g2": lambda d: arms.f2_adaptive_conformal(d, 0.05, s_hat, gamma_mult=2.0),
        "F3": lambda d: arms.f3_rescale(d, 0.05),
        "F4": lambda d: arms.f4_additive_fz0(d, 0.05, s_hat),
        "H": lambda d: arms.h_hybrid(d, sig, 0.05),
        "GE": lambda d: arms.garch_evt(d, mu, sig, 0.05),
        "X4": lambda d: arms.x4_multi(d, mu, (0.05,))[0.05],     # v2.0 control arms
        "ZL": lambda d: arms.zero_loc_multi(d, sig, (0.05,))[0.05],
        "CL": lambda d: arms.const_loc_multi(d, sig, (0.05,))[0.05],
    }
    for name, fn in calls.items():
        v1, e1 = fn(df)
        v2, e2 = fn(df2)
        assert np.allclose(v1[:k], v2[:k], equal_nan=True), f"{name} v leaks"
        assert np.allclose(e1[:k], e2[:k], equal_nan=True), f"{name} e leaks"
