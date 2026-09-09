"""E-rule acceptance tests per user spec (D2/D3): exact recovery, closed-form
deviation, and GPD parameter convergence — assertions, not smoke."""

import numpy as np
import pytest
from scipy import stats

from erules.rules import (e1_quantile, e2_quantile, e3_quantile,
                          e3_quantile_from_fit, e3_var_es_from_fit, fit_gpd,
                          fit_gpd_pwm, fit_nu, gpd_tail_quantile, mad)

NU = 5.0
DECILES = [round(0.1 * k, 1) for k in range(1, 10)]
T5_GRID = {tau: float(stats.t.ppf(tau, NU)) for tau in DECILES + [0.5]}


def test_e2_exact_recovery_on_t5_deciles():
    """Exact t(5) deciles + true nu -> q01/q025 recovered exactly."""
    for a in (0.01, 0.025):
        assert e2_quantile(T5_GRID, a, nu=NU) == pytest.approx(
            stats.t.ppf(a, NU), rel=1e-12)


def test_e1_deviation_equals_closed_form_normal_vs_t_gap():
    """E1 on the same t(5) deciles must deviate from truth by EXACTLY the
    analytic normal-vs-t gap: s_N·z_a − t_a(5), s_N = (q80−q20)/(z80−z20)."""
    s_n = (T5_GRID[0.8] - T5_GRID[0.2]) / (stats.norm.ppf(0.8) - stats.norm.ppf(0.2))
    for a in (0.01, 0.025):
        gap_closed_form = s_n * stats.norm.ppf(a) - stats.t.ppf(a, NU)
        e1 = e1_quantile(T5_GRID, a)
        assert e1 - stats.t.ppf(a, NU) == pytest.approx(gap_closed_form, rel=1e-12)
        assert e1 > stats.t.ppf(a, NU)   # normal tail is too shallow vs t(5)


def test_e3_gpd_fit_converges_to_true_params():
    """Large synthetic GPD exceedance sample -> (xi, beta) near truth."""
    rng = np.random.default_rng(20260704)
    xi_true, beta_true = 0.25, 1.5
    x = stats.genpareto.rvs(xi_true, loc=0, scale=beta_true,
                            size=200_000, random_state=rng)
    xi, beta, n_kept = fit_gpd(x)
    assert xi == pytest.approx(xi_true, abs=0.01)
    assert beta == pytest.approx(beta_true, rel=0.01)
    assert n_kept == len(x)                  # continuous: tie filter drops nothing


def test_pwm_recovers_gpd_params():
    """(v2.0 P0-1 test i) Hosking-Wallis PWM on a large seeded continuous GPD
    sample recovers (xi, beta) to tolerance."""
    rng = np.random.default_rng(20260729)
    xi_true, beta_true = 0.25, 1.5
    x = stats.genpareto.rvs(xi_true, loc=0, scale=beta_true,
                            size=200_000, random_state=rng)
    xi, beta = fit_gpd_pwm(x)
    assert xi == pytest.approx(xi_true, abs=0.02)
    assert beta == pytest.approx(beta_true, rel=0.02)


def test_fit_gpd_quantized_ties_no_boundary_fits():
    """(v2.0 P0-1 test ii) Regression on a 1-bp-quantized GARCH-t series built
    the way the rates series are (levels rounded to 1 bp, then bp differences,
    which leaves machine-precision near-ties at the POT threshold):
      - the OLD behavior (raw genpareto MLE on unfiltered exceedances) produces
        beta ~ 0 parameter-boundary fits on a systematic fraction of windows;
      - the NEW fit_gpd produces ZERO beta < 1e-6*MAD fits; windows it cannot
        fit raise ValueError (-> NaN) and stay a small minority; every
        successful fit yields a FINITE E3 VaR."""
    import warnings
    from synth.paths import garch_t_path
    y, _, _ = garch_t_path(20260729, n=3000)
    lvl = np.round(4.0 + np.cumsum(y / 2.0) / 100.0, 2)   # 1-bp-quantized levels
    x = np.diff(lvl) * 100.0                              # dbp with float near-ties
    n_win = n_tie = n_bold = n_bnew = n_raise = n_var = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(512, len(x), 7):
            ctx = x[t - 512:t]
            u = float(np.quantile(ctx, 0.10))
            exc = u - ctx[ctx < u]
            if len(exc) < 20:
                continue
            n_win += 1
            m = mad(ctx)
            n_tie += int((exc < 1e-9 * m).sum() > 0)
            try:                                          # OLD behavior premise
                _, _, b_old = stats.genpareto.fit(exc, floc=0.0)
                n_bold += int(b_old < 1e-6)
            except Exception:
                pass
            try:                                          # NEW behavior
                xi_n, b_n, kept_n = fit_gpd(exc, scale_ref=m)
                n_bnew += int(b_n < 1e-6 * m)
                q50c = float(np.median(ctx))
                if u != q50c:                             # E3 map defined
                    v = e3_quantile_from_fit(T5_GRID, q50c, u, xi_n, b_n, 0.01,
                                             tau_mass=kept_n / len(ctx))
                    n_var += int(np.isfinite(v))
                else:
                    n_var += 1                            # degenerate map, not GPD
            except ValueError:
                n_raise += 1
    assert n_win > 300 and n_tie > 100          # the tie pathology is present
    assert n_bold >= 10, f"premise lost: old MLE boundary fits = {n_bold}"
    assert n_bnew == 0, f"new fit still hits the boundary: {n_bnew}"
    assert n_raise <= 0.15 * n_win, f"too many invalid windows: {n_raise}/{n_win}"
    assert n_var == n_win - n_raise             # every successful fit -> finite VaR


def test_e3_var_es_split_on_xi_ge1():
    """(v2.0 P0-1 test for the split paths) xi >= 1: VaR finite, ES NaN;
    xi < 1: both finite. ES failure never voids the VaR."""
    q50c, u = 0.0, -1.2
    v, e = e3_var_es_from_fit(T5_GRID, q50c, u, 1.2, 0.8, 0.01, tau_mass=0.10)
    assert np.isfinite(v)
    assert np.isnan(e)
    v2, e2 = e3_var_es_from_fit(T5_GRID, q50c, u, 0.3, 0.8, 0.01, tau_mass=0.10)
    assert np.isfinite(v2) and np.isfinite(e2) and e2 < v2


def test_gpd_tail_quantile_analytic():
    # exponential limit xi->0: u - beta*ln(tau_u/alpha)
    assert gpd_tail_quantile(-1.0, 0.0, 2.0, 0.10, 0.01) == pytest.approx(
        -1.0 - 2.0 * np.log(10.0), rel=1e-12)


def test_e3_end_to_end_on_exact_t5_world():
    """ctx ~ t(5) (large), model grid = exact t(5) deciles -> E3 deep quantile
    lands near the true t(5) quantile (GPD tail approximation tolerance)."""
    rng = np.random.default_rng(7)
    ctx = stats.t.rvs(NU, size=100_000, random_state=rng)
    q = e3_quantile(T5_GRID, ctx, 0.01, tau_anchor=0.10)
    truth = stats.t.ppf(0.01, NU)
    assert q == pytest.approx(truth, rel=0.08)


def test_fit_nu_recovers_truth():
    rng = np.random.default_rng(11)
    ctx = stats.t.rvs(NU, size=50_000, random_state=rng)
    assert fit_nu(ctx) == pytest.approx(NU, rel=0.1)


def test_e2_es_matches_oracle_formula():
    """E2 ES on exact t(5) deciles + true nu == the standardized-t ES used by the
    frozen fixture generator (make_input.py), scale 1."""
    from erules.rules import e2_es
    for a in (0.01, 0.025, 0.05):
        t_a = stats.t.ppf(a, NU)
        truth = -stats.t.pdf(t_a, NU) * (NU + t_a**2) / ((NU - 1) * a)
        assert e2_es(T5_GRID, a, nu=NU) == pytest.approx(truth, rel=1e-12)


def test_e1_es_normal_identity():
    from erules.rules import e1_es
    grid = {tau: float(stats.norm.ppf(tau)) for tau in DECILES + [0.5]}
    for a in (0.01, 0.05):
        truth = -stats.norm.pdf(stats.norm.ppf(a)) / a
        assert e1_es(grid, a) == pytest.approx(truth, rel=1e-12)


def test_gpd_tail_es_vs_numerical_integration():
    from erules.rules import gpd_tail_es, gpd_tail_quantile
    u, xi, beta, tau_u, a = -1.0, 0.2, 1.5, 0.10, 0.01
    es = gpd_tail_es(u, xi, beta, tau_u, a)
    # numerical check: X = u - Y, Y~GPD; ES = mean of X | X < q_a
    q = gpd_tail_quantile(u, xi, beta, tau_u, a)
    # power tail (y^-1/xi): integrate on a log-spaced grid far into the tail,
    # else truncation biases E[Y | Y > y_a] downward
    ys = np.geomspace(u - q, 2e5, 800_000)
    dens = stats.genpareto.pdf(ys, xi, scale=beta)
    num = u - np.trapz(ys * dens, ys) / np.trapz(dens, ys)
    assert es == pytest.approx(num, rel=1e-3)
