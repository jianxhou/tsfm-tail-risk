"""Reference tests for the econometric baselines (amendment B: green before grid).
Assertions against definitions, closed forms, and coverage — not smoke."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from harness.baselines import (caviar_sav_forecast, ewma_var,
                              filtered_historical_simulation, fit_caviar_sav,
                              historical_simulation, _caviar_sav_path, _rq_loss)


# ---- HS ----

def test_hs_matches_definition():
    w = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    v, e = historical_simulation(w, 0.2)
    assert v == pytest.approx(np.quantile(w, 0.2))
    tail = w[w <= v]
    assert e == pytest.approx(tail.mean())


def test_hs_es_le_var():
    rng = np.random.default_rng(0)
    w = rng.standard_t(5, 2000)
    for a in (0.01, 0.05):
        v, e = historical_simulation(w, a)
        assert e <= v < 0


# ---- EWMA ----

def test_ewma_normal_closed_form_on_constant_series():
    """Constant |y| feed -> EWMA sigma converges; VaR/ES = sigma * normal tail."""
    w = np.full(4000, 1.0)          # y^2 = 1 -> s2 -> 1
    a = 0.05
    v, e = ewma_var(w, a, lam=0.94)
    assert v == pytest.approx(stats.norm.ppf(a), abs=1e-3)
    assert e == pytest.approx(-stats.norm.pdf(stats.norm.ppf(a)) / a, abs=1e-3)


def test_ewma_recursion_hand_value():
    w = np.array([2.0, 0.0, 0.0])   # seed s2=var(w)
    s2 = np.var(w)
    for y in w:
        s2 = 0.94 * s2 + 0.06 * y * y
    v, _ = ewma_var(w, 0.05, lam=0.94)
    assert v == pytest.approx(np.sqrt(s2) * stats.norm.ppf(0.05))


# ---- FHS ----

def test_fhs_reduces_to_hs_under_constant_vol():
    """If all |y| are equal, EWMA vol is constant, so FHS == HS scaled by 1."""
    w = np.array([-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0] * 50)
    a = 0.05
    v_fhs, _ = filtered_historical_simulation(w, a)
    # constant vol -> standardized resid quantile * sig_next; sig roughly const
    z = w / np.sqrt(np.var(w))
    assert v_fhs == pytest.approx(np.sqrt(np.var(w)) * np.quantile(z, a), rel=0.05)


def test_fhs_coverage_on_garch():
    """FHS on a simulated GARCH-t path covers near nominal out of sample."""
    rng = np.random.default_rng(1)
    n = 4000
    o, al, be, nu = 0.05, 0.09, 0.9, 7.0
    s2 = o / (1 - al - be); y = np.empty(n)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    prev = 0.0
    for t in range(n):
        s2 = o + al * prev**2 + be * s2
        y[t] = np.sqrt(s2) * z[t]; prev = y[t]
    a, W = 0.05, 500
    viol = 0; cnt = 0
    for t in range(W, n):
        v, _ = filtered_historical_simulation(y[t - W:t], a)
        viol += y[t] <= v; cnt += 1
    assert abs(viol / cnt - a) < 0.015


def test_ewma_fhs_share_sigma_path():
    """EWMA and FHS consume the SAME lambda=0.94 sigma recursion seeded at the
    sample variance; they differ only in tail form (normal quantile vs
    filtered empirical quantile). Pins the controlled forecast-form contrast
    of the paper's EWMA/FHS exhibit (v2.0 Phase-3 block 4)."""
    rng = np.random.default_rng(20260730)
    w = rng.standard_t(5, 1000)
    # independent reconstruction of the shared recursion
    s2 = np.var(w)
    sig_path = np.empty(len(w))
    for i, y in enumerate(w):
        sig_path[i] = np.sqrt(s2)
        s2 = 0.94 * s2 + 0.06 * y * y
    sig_next = np.sqrt(s2)
    for a in (0.01, 0.05):
        v_e, e_e = ewma_var(w, a, lam=0.94)
        v_f, e_f = filtered_historical_simulation(w, a, lam=0.94)
        # EWMA = terminal sigma * normal tail
        assert v_e == pytest.approx(sig_next * stats.norm.ppf(a), rel=1e-12)
        # FHS = SAME terminal sigma * filtered empirical tail
        z = w / sig_path
        zq = np.quantile(z, a)
        assert v_f == pytest.approx(sig_next * zq, rel=1e-12)
        assert e_f == pytest.approx(sig_next * z[z <= zq].mean(), rel=1e-12)
        # implied sigmas identical across the two estimators
        assert v_e / stats.norm.ppf(a) == pytest.approx(v_f / zq, rel=1e-12)


# ---- CAViaR-SAV ----

def test_rq_loss_check_function():
    y = np.array([1.0, -2.0, 3.0]); v = np.array([0.0, 0.0, 0.0])
    # a=0.05: u=1 (>=0): 1*0.05; u=-2 (<0): -2*(0.05-1)=1.9; u=3: 3*0.05=0.15
    assert _rq_loss(y, v, 0.05) == pytest.approx(0.05 + 1.9 + 0.15)


def test_caviar_rq_matches_bruteforce_on_iid_intercept_only():
    """With b1=b2=0 pinned, the RQ-optimal intercept is the empirical quantile."""
    rng = np.random.default_rng(2)
    y = rng.standard_normal(3000)
    a = 0.05
    grid = np.linspace(-3, 0, 601)
    losses = [_rq_loss(y, np.full_like(y, c), a) for c in grid]
    c_star = grid[int(np.argmin(losses))]
    assert c_star == pytest.approx(np.quantile(y, a), abs=0.05)


def test_caviar_unconditional_coverage_on_iid():
    """On iid data CAViaR-SAV fitted VaR gives near-nominal coverage."""
    rng = np.random.default_rng(3)
    y = rng.standard_t(6, 2500) * np.sqrt(4 / 6)
    a = 0.05
    fit = fit_caviar_sav(y, a, n_starts=8)
    v = _caviar_sav_path(fit.beta, y, fit.v1)
    cov = (y <= v).mean()
    assert abs(cov - a) < 0.02


def test_caviar_multistart_diagnostics_recorded():
    """v2.0 P0-3 semantics: valid_starts counts candidates passing the
    stability/validity conditions; converged_starts counts optimizer-reported
    convergence (res.success) — the old field merely counted finite objectives."""
    rng = np.random.default_rng(4)
    y = rng.standard_normal(1500)
    _, fit = caviar_sav_forecast(y, 0.05, n_starts=10)
    assert fit.n_starts >= 10
    assert fit.valid_starts >= 5            # healthy data: most starts valid
    assert fit.loss_spread >= 0.0           # spread across VALID starts
    assert 0 <= fit.best_start_idx < fit.valid_starts
    assert 0 <= fit.converged_starts <= fit.n_starts


def _fit_caviar_unconstrained(y, alpha, n_starts=8, seed=20260706, warmup=300):
    """The PRE-v2.0 unconstrained estimator, reconstructed here only to pin the
    regression premise (do not use for anything else)."""
    from scipy.optimize import minimize
    v1 = float(np.quantile(y[:min(warmup, len(y))], alpha))
    rng = np.random.default_rng(seed)
    uncond = abs(np.quantile(y, alpha))
    base = [np.array([v1 * 0.1, 0.9, -0.1 * uncond]),
            np.array([v1 * 0.5, 0.5, -0.3 * uncond]),
            np.array([v1, 0.0, 0.0])]
    rand = [np.array([rng.uniform(-1, 0) * uncond, rng.uniform(0, 0.95),
                      rng.uniform(-0.5, 0) * uncond]) for _ in range(n_starts - 3)]
    results = []
    for s in base + rand:
        r = minimize(lambda b: _rq_loss(y, _caviar_sav_path(b, y, v1), alpha), s,
                     method="Nelder-Mead",
                     options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-8})
        if np.isfinite(r.fun):
            results.append((r.fun, r.x))
    losses = [f for f, _ in results]
    return results[int(np.argmin(losses))][1], v1


def test_caviar_divergence_window_regression():
    """(v2.0 P0-3) Vendored dgs5 fit window that produced the delivered grid's
    divergent CAViaR path (stored v05 min -5,729,758.7). Premise: the OLD
    unconstrained estimator fits |b1| > 1 at alpha=5%; its in-sample path looks
    bounded, but the recursion rolled onto the NEXT window explodes (>1e3, with
    positive lower-tail VaR days) — the exact delivered-grid mechanism. The NEW
    constrained estimator on the same windows: |b1| < 1 structurally, valid
    starts, ZERO |v| > 1e3 anywhere, and strictly negative one-step VaR."""
    from harness.baselines import fit_caviar_sav
    fx = pd.read_csv("tests/fixtures/caviar_divergence_window.csv")
    w_fit, w_next = fx["ctx_fit"].to_numpy(), fx["ctx_next"].to_numpy()
    a = 0.05
    # premise (old behavior)
    beta_o, _ = _fit_caviar_unconstrained(w_fit, a)
    v1n = float(np.quantile(w_next[:300], a))
    p_old = _caviar_sav_path(beta_o, w_next, v1n)
    assert abs(beta_o[1]) > 1.0, f"premise lost: old b1={beta_o[1]:.5f}"
    assert np.max(np.abs(p_old)) > 1e3 and (p_old > 0).sum() > 0
    # new behavior on both windows
    for alpha in (0.01, 0.05):
        fit = fit_caviar_sav(w_fit, alpha, n_starts=8)
        assert abs(fit.beta[1]) < 1.0
        assert fit.valid_starts >= 1
        for w in (w_fit, w_next):
            v1 = float(np.quantile(w[:300], alpha))
            p = _caviar_sav_path(fit.beta, w, v1)
            v_next = fit.beta[0] + fit.beta[1] * p[-1] + fit.beta[2] * abs(w[-1])
            assert np.all(np.isfinite(p)) and np.max(np.abs(p)) < 1e3
            assert v_next < 0


def test_caviar_tracks_volatility_clustering():
    """On a genuinely clustered GARCH path, CAViaR-SAV must learn a reactive model:
    b2 < 0 (large |y| widens the lower-tail VaR) and b1 > 0 (persistence), and a
    fresh large shock must widen VaR on the next step."""
    rng = np.random.default_rng(5)
    n = 2500
    o, al, be, nu = 0.05, 0.15, 0.80, 6.0    # strong ARCH -> clear clustering
    s2 = o / (1 - al - be); y = np.empty(n)
    z = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)
    prev = 0.0
    for t in range(n):
        s2 = o + al * prev**2 + be * s2
        y[t] = np.sqrt(s2) * z[t]; prev = y[t]
    _, fit = caviar_sav_forecast(y, 0.05, n_starts=12)
    b0, b1, b2 = fit.beta
    assert b2 < 0, f"reaction coef b2={b2:.3f} should be negative"
    assert b1 > 0, f"persistence b1={b1:.3f} should be positive"
    # inject a fresh shock at the end and confirm next-step VaR widens
    y2 = y.copy(); y2[-1] = -10.0
    v_shock, _ = caviar_sav_forecast(y2, 0.05, n_starts=1)
    v_calm, _ = caviar_sav_forecast(y, 0.05, n_starts=1)
    assert v_shock < v_calm
