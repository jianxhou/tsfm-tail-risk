"""v2.1 E3 tail-mass battery (review #10 §2.6; docs/e3_tail_mass_ruling.md).

Covers: v2.0 back-compat regression at (tau_anchor=0.10, tau_mass=0.10);
continuous-window analytic mass 52/512; quantized-tie point-mass accounting;
degenerate/tied inputs fail loudly (no unbounded beta, no silent shrink, no
alpha-collapse); xi->0 exponential limit and xi>=1 VaR/ES separation at
non-nominal mass; mle/pwm/fail branch fixtures; alpha-in-mass hard failure;
CDF/round-trip and alpha-monotonicity properties; determinism; shift/scale
equivariance; no lookahead; the production tau_mass margin over max(ALPHAS)
on 4 rate + 4 continuous assets (full-panel scan lands with Phase 2); and the
shat_first_window production-import regression (I9 companion).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from erules.rules import (e3_es_from_fit, e3_quantile, e3_quantile_from_fit,
                          e3_var_es_from_fit, fit_gpd, fit_gpd_diag,
                          fit_gpd_pwm, gpd_tail_es, gpd_tail_quantile, mad)

FIX = Path(__file__).parent / "fixtures"
NU = 5.0
DECILES = [round(0.1 * k, 1) for k in range(1, 10)]
T5_GRID = {tau: float(stats.t.ppf(tau, NU)) for tau in DECILES + [0.5]}
ALPHAS = (0.01, 0.025, 0.05)
W = 512


def _continuous_ctx(seed=20260802, n=W):
    return np.random.default_rng(seed).standard_normal(n)


def _quantized_series(seed=20260729, n=3000):
    """1-bp-quantized GARCH-t bp-difference series (rates construction)."""
    from synth.paths import garch_t_path
    y, _, _ = garch_t_path(seed, n=n)
    lvl = np.round(4.0 + np.cumsum(y / 2.0) / 100.0, 2)
    return np.diff(lvl) * 100.0


# ---------- regression: v2.0 values reproduce at (0.10, 0.10) ----------

def test_v20_reference_reproduces_at_nominal_mass():
    """(tau_anchor=0.10, tau_mass=0.10) reproduces the frozen v2.0 production
    values (real ctxfits x real grids, spx+dgs30, 324 cells) to 1e-12,
    including the xi>=1 ES-NaN pattern."""
    ref = pd.read_csv(FIX / "e3_v20_reference.csv")
    assert len(ref) >= 300
    n_nan = 0
    for r in ref.itertuples():
        grid = {float(k): v for k, v in json.loads(r.grid_json).items()}
        v, e = e3_var_es_from_fit(grid, r.q50c, r.u, r.xi, r.beta, r.alpha,
                                  tau_anchor=0.10, tau_mass=0.10)
        if np.isnan(r.v_expected):
            assert np.isnan(v)
        else:
            assert abs(v - r.v_expected) <= 1e-12 * max(1.0, abs(r.v_expected))
        if np.isnan(r.e_expected):
            assert np.isnan(e) and r.xi >= 1.0
            n_nan += 1
        else:
            assert abs(e - r.e_expected) <= 1e-12 * max(1.0, abs(r.e_expected))
    assert n_nan == int(ref["e_expected"].isna().sum())


# ---------- continuous window: analytic mass and path consistency ----------

def test_continuous_window_mass_is_52_over_512():
    ctx = _continuous_ctx()
    u = float(np.quantile(ctx, 0.10))
    d = fit_gpd_diag(u - ctx[ctx < u], scale_ref=mad(ctx))
    assert d["status"] == "ok" and d["fit_branch"] == "mle"
    assert int((ctx < u).sum()) == 52          # linear interpolation, no ties
    assert d["n_raw_exc"] == 52 and d["tie_count"] == 0 and d["n_kept"] == 52
    assert d["n_kept"] / W == 0.1015625        # ruling clause 5, exact


def test_refit_and_from_fit_paths_agree():
    """e3_quantile (refit) == e3_quantile_from_fit fed the same fit + the
    empirical mass — VaR and ES flow through ONE path (ruling clause 6)."""
    ctx = _continuous_ctx()
    u = float(np.quantile(ctx, 0.10))
    xi, beta, n_kept = fit_gpd(u - ctx[ctx < u], scale_ref=mad(ctx))
    tm = n_kept / len(ctx)
    q50c = float(np.quantile(ctx, 0.5))
    for a in ALPHAS:
        direct = e3_quantile(T5_GRID, ctx, a)
        wired = e3_quantile_from_fit(T5_GRID, q50c, u, xi, beta, a, tau_mass=tm)
        assert direct == wired


def test_threshold_probability_identities():
    """Empirical CDF at u stays n_strict/W; the splice satisfies
    P(X<q_alpha) = tau_mass * Gbar(u - q_alpha) = alpha (round trip)."""
    ctx = _continuous_ctx(7)
    u = float(np.quantile(ctx, 0.10))
    xi, beta, n_kept = fit_gpd(u - ctx[ctx < u], scale_ref=mad(ctx))
    tm = n_kept / len(ctx)
    assert (ctx < u).mean() == pytest.approx(n_kept / W)   # no ties here
    for a in ALPHAS:
        q = gpd_tail_quantile(u, xi, beta, tm, a)
        y = u - q
        alpha_back = tm * float(stats.genpareto.sf(y, xi, scale=beta))
        assert alpha_back == pytest.approx(a, rel=1e-9)


# ---------- quantized ties: threshold-point-mass accounting ----------

def test_quantized_ties_are_booked_as_threshold_point_mass():
    x = _quantized_series()
    hit = 0
    for t in range(512, len(x), 7):
        ctx = x[t - 512:t]
        u = float(np.quantile(ctx, 0.10))
        n_strict = int((ctx < u).sum())
        d = fit_gpd_diag(u - ctx[ctx < u], scale_ref=mad(ctx))
        assert d["n_raw_exc"] == n_strict
        if not np.isfinite(d["n_kept"]):
            continue
        assert d["n_kept"] + d["tie_count"] == d["n_raw_exc"]   # explicit booking
        if d["tie_count"] > 0 and d["status"] == "ok":
            hit += 1
            tau_strict = n_strict / 512
            tau_fit = d["n_kept"] / 512
            assert tau_fit < tau_strict            # point mass left OUT of the fit
            # direction: smaller extrapolation mass -> shallower deep quantile
            q_old = gpd_tail_quantile(u, d["xi"], d["beta"], 0.10, 0.01)
            q_new = gpd_tail_quantile(u, d["xi"], d["beta"], tau_fit, 0.01)
            if tau_fit < 0.10:
                assert q_new > q_old
    assert hit > 100                               # the pathology is exercised


def test_heavy_ties_fail_loudly_not_silently():
    """A window whose lower tail is one repeated value must NOT produce an
    unbounded-beta fit or a silently shrunken sample: the failure is an
    explicit status/ValueError with the counts on record."""
    rng = np.random.default_rng(3)
    ctx = rng.standard_normal(W)
    lo = np.quantile(ctx, 0.12)
    ctx[ctx < lo] = lo - 1e-13                     # entire lower tail one value
    u = float(np.quantile(ctx, 0.10))
    exc = u - ctx[ctx < u]
    d = fit_gpd_diag(exc, scale_ref=mad(ctx))
    assert d["status"] != "ok"                     # explicit, not silent
    with pytest.raises(ValueError):
        fit_gpd(exc, scale_ref=mad(ctx))


def test_partial_ties_keep_alpha_resolution():
    """With a mixed kept/tie population the fit stays valid, beta respects its
    floor, and the three alphas map to three distinct quantiles."""
    rng = np.random.default_rng(11)
    kept = stats.genpareto.rvs(0.2, scale=1.0, size=40, random_state=rng) + 1e-3
    ties = np.full(30, 1e-15)
    exc = np.concatenate([kept, ties])
    d = fit_gpd_diag(exc, scale_ref=1.0)
    assert d["status"] == "ok" and d["n_kept"] == 40 and d["tie_count"] == 30
    assert d["beta"] >= d["beta_floor"]
    tm = d["n_kept"] / W
    qs = [gpd_tail_quantile(-1.0, d["xi"], d["beta"], tm, a) for a in ALPHAS]
    assert qs[0] < qs[1] < qs[2]                   # no alpha collapse


# ---------- limits, separation, branches, hard failures ----------

def test_xi_zero_exponential_limit_at_empirical_mass():
    tm = 0.1015625
    assert gpd_tail_quantile(-1.0, 0.0, 2.0, tm, 0.01) == pytest.approx(
        -1.0 - 2.0 * np.log(tm / 0.01), rel=1e-12)


def test_xi_ge1_var_kept_es_missing_at_empirical_mass():
    v, e = e3_var_es_from_fit(T5_GRID, 0.0, -1.2, 1.3, 0.7, 0.01,
                              tau_mass=0.1015625)
    assert np.isfinite(v) and np.isnan(e)
    with pytest.raises(ValueError):
        gpd_tail_es(-1.2, 1.3, 0.7, 0.1015625, 0.01)


def test_fit_branch_fixtures_mle_pwm_fail(monkeypatch):
    # mle: continuous exceedances
    rng = np.random.default_rng(5)
    x = stats.genpareto.rvs(0.2, scale=1.0, size=60, random_state=rng) + 1e-6
    assert fit_gpd_diag(x, scale_ref=1.0)["fit_branch"] == "mle"
    # pwm: force the MLE down to exercise the fallback branch
    monkeypatch.setattr(stats.genpareto, "fit",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mle down")))
    d = fit_gpd_diag(x, scale_ref=1.0)
    assert d["fit_branch"] == "pwm"
    assert (d["xi"], d["beta"]) == fit_gpd_pwm(x)
    # fail: PWM beta lands under the validity floor — pick scale_ref so the
    # floor (1e-6*scale_ref) sits at exactly 2x the PWM beta while the tie
    # filter (1e-9*scale_ref) still keeps every point
    tiny = np.linspace(2e-6, 3e-6, 30)
    _, beta_p = fit_gpd_pwm(tiny)
    sref = 2.0 * beta_p / 1e-6
    assert 1e-9 * sref < tiny.min()                # filter drops nothing
    d2 = fit_gpd_diag(tiny, scale_ref=sref)
    assert d2["n_kept"] == 30
    assert d2["fit_branch"] == "fail" and "PWM fallback" in d2["status"]
    assert np.isnan(d2["xi"]) and np.isnan(d2["beta"])
    monkeypatch.undo()
    # fail: too few exceedances (before any filtering)
    d3 = fit_gpd_diag(np.array([1.0, 2.0]), scale_ref=1.0)
    assert d3["fit_branch"] == "fail" and ">=20" in d3["status"]


def test_alpha_must_sit_strictly_inside_tail_mass():
    with pytest.raises(ValueError):
        gpd_tail_quantile(-1.0, 0.2, 1.0, 0.05, 0.05)      # alpha == mass
    with pytest.raises(ValueError):
        gpd_tail_quantile(-1.0, 0.2, 1.0, 0.04, 0.05)      # alpha > mass


def test_anchor_assert_survives_production_swallow():
    """Trap 1 (review #10 §2.4): a grid lacking the anchor must raise an
    AssertionError that production `except (ValueError, KeyError)` handlers
    CANNOT swallow."""
    no_anchor = {k: v for k, v in T5_GRID.items() if k != 0.1}
    with pytest.raises(AssertionError):
        try:
            e3_var_es_from_fit(no_anchor, 0.0, -1.5, 0.2, 0.8, 0.01,
                               tau_mass=0.0546875)
        except (ValueError, KeyError):              # the production wrapper
            pytest.fail("anchor error was swallowed into the NaN path")
    with pytest.raises(TypeError):
        # tau_mass is required-keyword: the v2.0 call shape must fail loudly
        e3_var_es_from_fit(T5_GRID, 0.0, -1.5, 0.2, 0.8, 0.01)


# ---------- properties: monotonicity, determinism, equivariance, lookahead ----

def test_var_monotone_in_alpha_end_to_end():
    ctx = _continuous_ctx(13)
    qs = [e3_quantile(T5_GRID, ctx, a) for a in ALPHAS]
    assert qs[0] < qs[1] < qs[2]


def test_deterministic_on_same_window():
    ctx = _continuous_ctx(17)
    assert e3_quantile(T5_GRID, ctx, 0.01) == e3_quantile(T5_GRID, ctx, 0.01)


def test_affine_equivariance():
    """Formula layer exact; end-to-end (through the MLE) to 1e-6 rel."""
    s, m = 2.5, -0.7
    # exact at the from_fit layer
    v = e3_quantile_from_fit(T5_GRID, -0.1, -1.4, 0.25, 0.9, 0.01,
                             tau_mass=0.1015625)
    grid_t = {k: s * q + m for k, q in T5_GRID.items()}
    v_t = e3_quantile_from_fit(grid_t, s * -0.1 + m, s * -1.4 + m, 0.25,
                               s * 0.9, 0.01, tau_mass=0.1015625)
    assert v_t == pytest.approx(s * v + m, rel=1e-12)
    # end-to-end through the refit: exact up to scipy MLE optimizer noise on
    # the rescaled problem (identical kept set and tau_mass by construction;
    # measured delta ~8e-6 rel)
    ctx = _continuous_ctx(19)
    v1 = e3_quantile(T5_GRID, ctx, 0.01)
    v2 = e3_quantile(grid_t, s * ctx + m, 0.01)
    assert v2 == pytest.approx(s * v1 + m, rel=5e-5)


def test_no_lookahead_beyond_window():
    x = _continuous_ctx(23, n=1000)
    t = 700
    before = e3_quantile(T5_GRID, x[t - W:t].copy(), 0.01)
    x[t:] = 99.0                                    # corrupt the future
    after = e3_quantile(T5_GRID, x[t - W:t], 0.01)
    assert before == after


# ---------- production margin: min(tau_mass) - max(ALPHAS) >= 0.003 ----------

def test_tau_mass_margin_on_rate_and_continuous_assets():
    """Ruling clause 9 margin test on 4 rate + 4 continuous assets over their
    full production window sets (cheap count replication of the fit_gpd tie
    filter, spot-checked against fit_gpd_diag; the full 32-asset scan lands
    with the Phase-2 rerun). Measured panel minimum 0.0546875 (dgs10)."""
    from data.load import load_series
    from harness.rolling import rolling_windows
    if not (Path(__file__).resolve().parent.parent
            / "data" / "parquet" / "dgs2.parquet").exists():
        pytest.skip("requires the rebuilt raw-data layer (data/parquet/); "
                    "the replication packages ship fetch code, not data — "
                    "rebuild via the data step in REPRODUCING.md §3")
    lo = np.inf
    for asset in ("dgs2", "dgs5", "dgs10", "dgs30", "spx", "ndx", "gold", "btc"):
        df = load_series(asset)
        rs = rolling_windows(df, value_col=df.attrs.get("column", "logret"),
                             ctx_len=W, oos_start="2016-01-01",
                             allow_late_start=True)
        for i, w in enumerate(rs.windows):
            ctx = w.ctx
            u = float(np.quantile(ctx, 0.10))
            exc = u - ctx[ctx < u]
            n_kept = int((exc >= 1e-9 * mad(ctx)).sum())
            if i % 500 == 0:                        # consistency spot check
                d = fit_gpd_diag(exc, scale_ref=mad(ctx))
                if np.isfinite(d["n_kept"]):
                    assert int(d["n_kept"]) == n_kept
            lo = min(lo, n_kept / len(ctx))
    assert lo - max(ALPHAS) >= 0.003, f"tau_mass margin violated: min={lo}"
    assert lo == pytest.approx(0.0546875, abs=1e-12)   # dgs10, ruling record


# ---------- I9 companion: production shat is importable and window-frozen ----

def test_shat_first_window_production_import():
    """Regression on the PRODUCTION function (review #10 §2.6 addendum): ŝ is
    the median central-grid scale over the FIRST 500 rows only — a later
    scale break must not move it (frozen first-window definition, v2.0 P0-2)."""
    from harness.run_grid_repairs import SHAT_WIN, shat_first_window
    n = 1600                                     # scale break dominates the
    scale = np.where(np.arange(n) < SHAT_WIN, 1.0, 9.0)   # full-frame median
    al = pd.DataFrame({"q25": -0.6745 * scale,
                       "q75": 0.6745 * scale})
    assert shat_first_window(al) == pytest.approx(1.0, rel=1e-12)
    # the full-frame median (the v2.0 P0-2 lookahead bug) would be 9.0
    full_median = float(np.median((al["q75"] - al["q25"]) / 1.349))
    assert full_median == pytest.approx(9.0, rel=1e-12)
    assert shat_first_window(al) != full_median
