"""E0-E3 tail extraction rules (proposal §3.3, pre-registered).

Input per window: the model's native quantile grid {tau: q} (from stored D2
outputs) and, for E2/E3, the context array. Anchor-pair convention (fixed):
scale is read from (0.25, 0.75) when present, else (0.2, 0.8); location = q50.

E0  native output as returned by the official implementation for the requested
    level (per registry native_tail_behavior: bolt's clamped edge IS its E0;
    timesfm has NO E0 — deep tails go E1-E3 only).
E1  normal tail:  q_a = q50 + s_N * z_a,        s_N = (q_hi - q_lo)/(z_hi - z_lo)
E2  t tail:       q_a = q50 + s_t * t_a(nu),    s_t = (q_hi - q_lo)/(t_hi - t_lo),
    nu fitted on the context (MLE, location-scale t) unless supplied.
E3  GPD splice (fixed-anchor empirical-mass POT; docs/e3_tail_mass_ruling.md,
    2026-08-02): POT on the context's left tail at threshold
    u = ctx-quantile tau_anchor (fixed 0.10; also the grid lookup anchor of the
    location-scale map — the map is self-consistent only if the ctx 10%
    quantile is mapped onto the model grid's 10% quantile). GPD(xi, beta) fit
    on the KEPT exceedances u - x (machine-precision threshold ties removed as
    explicit threshold-point mass; they enter no deep-tail formula). The GPD's
    unconditional weight is the empirical mass of its own fit population,
    tau_mass = n_kept / W, computed per window: for q below the threshold,
    P(X < q) = tau_mass * Gbar_GPD(u - q). VaR and ES consume the SAME
    tau_mass through the same path (ruling clause 6); xi >= 1 voids the ES
    only, never the VaR (v2.0 P0-1 split preserved, clause 7). v2.0 had fit
    the filtered population but extrapolated with the nominal 0.10 mass
    (review #10 P0-S1); (tau_anchor=0.10, tau_mass=0.10) reproduces the v2.0
    values exactly and is covered by a regression test.

Tested against exact synthetic truths in tests/test_erules.py plus the v2.1
tail-mass battery in tests/test_e3_tail_mass.py (hard rule 1).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

_ANCHORS = ((0.25, 0.75), (0.2, 0.8))


def _anchor(grid: dict) -> tuple[float, float]:
    for lo, hi in _ANCHORS:
        if lo in grid and hi in grid:
            return lo, hi
    raise ValueError(f"grid lacks an anchor pair {_ANCHORS}: {sorted(grid)}")


def e1_quantile(grid: dict, alpha: float) -> float:
    lo, hi = _anchor(grid)
    s = (grid[hi] - grid[lo]) / (stats.norm.ppf(hi) - stats.norm.ppf(lo))
    return float(grid[0.5] + s * stats.norm.ppf(alpha))


def fit_nu(ctx: np.ndarray) -> float:
    nu, _, _ = stats.t.fit(np.asarray(ctx, dtype=float))
    return float(np.clip(nu, 2.1, 100.0))


def e2_quantile(grid: dict, alpha: float, nu: float) -> float:
    lo, hi = _anchor(grid)
    s = (grid[hi] - grid[lo]) / (stats.t.ppf(hi, nu) - stats.t.ppf(lo, nu))
    return float(grid[0.5] + s * stats.t.ppf(alpha, nu))


def mad(x: np.ndarray) -> float:
    """Median absolute deviation (unscaled)."""
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def fit_gpd_pwm(exceedances: np.ndarray) -> tuple[float, float]:
    """Hosking-Wallis (1987) probability-weighted-moment GPD (xi, beta), loc 0.

    b_r = E[X (1-F(X))^r] sample PWMs; k_hat = b0/(b0-2b1) - 2 in the HW
    parametrization, xi = -k under the scipy sign convention. Boundary-robust
    (in-sample moments are always finite), used as the fallback when MLE
    degenerates on quantized/tied exceedances (v2.0 P0-1)."""
    xs = np.sort(np.asarray(exceedances, dtype=float))
    n = len(xs)
    if n < 2:
        raise ValueError("PWM needs >=2 exceedances")
    b0 = float(xs.mean())
    w = (n - 1.0 - np.arange(n)) / (n - 1.0)
    b1 = float(np.sum(w * xs) / n)
    denom = b0 - 2.0 * b1
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("PWM GPD: b0 - 2*b1 <= 0, no valid fit")
    xi = 2.0 - b0 / denom
    beta = 2.0 * b0 * b1 / denom
    return float(xi), float(beta)


def fit_gpd_diag(exceedances: np.ndarray, scale_ref: float | None = None) -> dict:
    """GPD fit with full per-window diagnostics (v2.1, review #10 §2.5).

    Same numerics as v2.0 fit_gpd, operation for operation (tie filter -> MLE
    -> PWM fallback -> validity floor); never raises on fit failure — returns
    a dict so every window's branch and counts can be persisted:
      n_raw_exc   exceedance count before the tie filter
      tie_count   points removed as threshold-point mass (excess < 1e-9*scale;
                  quantization ties sitting at u — ruling clauses 2/3)
      n_kept      fit-population size (tau_mass numerator, ruling clause 5)
      scale_ref   resolved scale reference actually used
      beta_floor  1e-6*scale_ref validity floor
      fit_branch  'mle' | 'pwm' | 'fail'
      xi, beta    parameters (NaN when fit_branch == 'fail')
      status      'ok' or the v2.0-identical failure message
    """
    x = np.asarray(exceedances, dtype=float)
    d = {"n_raw_exc": int(len(x)), "tie_count": np.nan, "n_kept": np.nan,
         "scale_ref": np.nan, "beta_floor": np.nan, "fit_branch": "fail",
         "xi": np.nan, "beta": np.nan, "status": "ok"}
    if len(x) < 20 or np.any(x <= 0):
        d["status"] = "need >=20 strictly positive exceedances"
        return d
    scale = None
    for cand in (scale_ref, mad(x), float(np.mean(x))):
        if cand is not None and np.isfinite(cand) and cand > 0:
            scale = float(cand)
            break
    if scale is None:
        d["status"] = "no positive finite scale reference for the GPD fit"
        return d
    d["scale_ref"] = scale
    x = x[x >= 1e-9 * scale]
    d["n_kept"] = int(len(x))
    d["tie_count"] = d["n_raw_exc"] - d["n_kept"]
    if len(x) < 20:
        d["status"] = "fewer than 20 exceedances after machine-precision tie removal"
        return d
    beta_floor = 1e-6 * scale
    d["beta_floor"] = beta_floor
    try:
        xi, _, beta = stats.genpareto.fit(x, floc=0.0)
        branch = "mle"
    except Exception:
        xi, beta, branch = np.nan, np.nan, "fail"
    if not (np.isfinite(xi) and np.isfinite(beta)) or beta < beta_floor:
        try:
            xi, beta = fit_gpd_pwm(x)        # boundary/invalid MLE -> PWM refit
            branch = "pwm"
        except ValueError as exc:
            d["status"] = str(exc)
            return d
    if not (np.isfinite(xi) and np.isfinite(beta)) or beta < beta_floor:
        d["status"] = f"GPD fit invalid after PWM fallback (beta={beta:.3e})"
        return d
    d["fit_branch"] = branch
    d["xi"], d["beta"] = float(xi), float(beta)
    return d


def fit_gpd(exceedances: np.ndarray,
            scale_ref: float | None = None) -> tuple[float, float, int]:
    """GPD (xi, beta, n_kept) with location fixed at 0, boundary-robust.

    Exceedances must be > 0. `scale_ref` is the caller's data scale (MAD of the
    context/residual window the exceedances came from); when None, falls back to
    MAD(exceedances), then mean(exceedances). Guards, in order:
      1. tie removal: drop exceedances < 1e-9*scale_ref — machine-precision
         float ties at the POT threshold on quantized series (e.g. 1-bp rate
         moves) that make the MLE likelihood unbounded (beta -> 0 boundary);
         the dropped points are explicit threshold-point mass, booked at the
         body/tail boundary and excluded from every deep-tail formula
         (docs/e3_tail_mass_ruling.md clauses 2/3);
      2. MLE; on boundary/invalid fit (beta < 1e-6*scale_ref or non-finite
         params) refit with Hosking-Wallis PWM (fit_gpd_pwm);
      3. validity: >=20 exceedances after tie removal, finite params,
         beta >= 1e-6*scale_ref — else ValueError (callers NaN the window).
    v2.1 (review #10 P0-S1): also returns n_kept, the fit-population size —
    the ONLY legitimate numerator for the extrapolation mass
    tau_mass = n_kept / W. Emitting it from the fit itself guarantees the mass
    is measured on exactly the population the GPD was fit on.
    On continuous (tie-free) data no guard triggers and (xi, beta) is the
    plain scipy genpareto MLE, unchanged from the pre-v2.0 implementation."""
    d = fit_gpd_diag(exceedances, scale_ref)
    if d["status"] != "ok":
        raise ValueError(d["status"])
    return d["xi"], d["beta"], int(d["n_kept"])


def gpd_tail_quantile(u: float, xi: float, beta: float, tau_mass: float,
                      alpha: float) -> float:
    """Left-tail quantile below threshold u.

    tau_mass is the unconditional probability mass of the GPD fit population
    (empirical P(X < u) net of threshold-point mass, ruling clause 4);
    alpha must be strictly deeper: P(X < q) = tau_mass * Gbar_GPD(u - q)."""
    if not alpha < tau_mass:
        raise ValueError("alpha must be deeper than the threshold probability")
    r = tau_mass / alpha
    if abs(xi) < 1e-12:
        return float(u - beta * np.log(r))
    return float(u - (beta / xi) * (r ** xi - 1.0))


def _e3_ctx_fit(ctx: np.ndarray, tau_anchor: float):
    """Shared E3 window fit: threshold, kept-population GPD, empirical mass."""
    u = float(np.quantile(ctx, tau_anchor))
    exc = u - ctx[ctx < u]
    xi, beta, n_kept = fit_gpd(exc, scale_ref=mad(ctx))
    tau_mass = n_kept / len(ctx)                     # ruling clause 5: W = len(ctx)
    return u, xi, beta, tau_mass


def e3_quantile(grid: dict, ctx: np.ndarray, alpha: float,
                tau_anchor: float = 0.10) -> float:
    ctx = np.asarray(ctx, dtype=float)
    # trap-1 guard (review #10 §2.4): the anchor check is an assert OUTSIDE any
    # try — an anchor/mass mix-up must never be swallowed into a NaN row.
    assert tau_anchor in grid, f"model grid lacks the splice anchor tau_anchor={tau_anchor}"
    u, xi, beta, tau_mass = _e3_ctx_fit(ctx, tau_anchor)
    q_ctx = gpd_tail_quantile(u, xi, beta, tau_mass, alpha)
    q50c = float(np.quantile(ctx, 0.5))
    a = (grid[tau_anchor] - grid[0.5]) / (u - q50c)     # location-scale map
    return float(grid[0.5] + a * (q_ctx - q50c))


# --- ES companions (analytic tail integrals; tested vs closed forms) ---

def e1_es(grid: dict, alpha: float) -> float:
    lo, hi = _anchor(grid)
    s = (grid[hi] - grid[lo]) / (stats.norm.ppf(hi) - stats.norm.ppf(lo))
    z = stats.norm.ppf(alpha)
    return float(grid[0.5] - s * stats.norm.pdf(z) / alpha)


def e2_es(grid: dict, alpha: float, nu: float) -> float:
    lo, hi = _anchor(grid)
    s = (grid[hi] - grid[lo]) / (stats.t.ppf(hi, nu) - stats.t.ppf(lo, nu))
    t_a = stats.t.ppf(alpha, nu)
    es_std = -stats.t.pdf(t_a, nu) * (nu + t_a ** 2) / ((nu - 1.0) * alpha)
    return float(grid[0.5] + s * es_std)


def gpd_tail_es(u: float, xi: float, beta: float, tau_mass: float,
                alpha: float) -> float:
    """Left-tail ES below the alpha-quantile via POT mean-excess (xi < 1).

    Consumes the SAME tau_mass as gpd_tail_quantile through the same path —
    the mean-excess formula is closed under the mass substitution and the
    threshold-point mass enters no integral (ruling clause 6)."""
    if xi >= 1.0:
        raise ValueError("GPD ES undefined for xi >= 1")
    q = gpd_tail_quantile(u, xi, beta, tau_mass, alpha)
    y_a = u - q
    return float(u - (y_a + (beta + xi * y_a) / (1.0 - xi)))


def e3_es(grid: dict, ctx: np.ndarray, alpha: float,
          tau_anchor: float = 0.10) -> float:
    ctx = np.asarray(ctx, dtype=float)
    assert tau_anchor in grid, f"model grid lacks the splice anchor tau_anchor={tau_anchor}"
    u, xi, beta, tau_mass = _e3_ctx_fit(ctx, tau_anchor)
    es_ctx = gpd_tail_es(u, xi, beta, tau_mass, alpha)
    q50c = float(np.quantile(ctx, 0.5))
    a = (grid[tau_anchor] - grid[0.5]) / (u - q50c)
    return float(grid[0.5] + a * (es_ctx - q50c))


# --- E3 from PRECOMPUTED GPD params (u, xi, beta, tau_mass) + q50c ---
# Results-identical to e3_quantile/e3_es (same u/xi/beta/tau_mass on the same
# window), but skips the per-cell GPD re-fit. Callers with cached ctx fits use
# these to avoid ~15x redundant genpareto.fit calls (CHANGELOG 2026-07-07).
# v2.1: tau_mass is a REQUIRED keyword — every caller must pass the per-window
# empirical mass explicitly (from the same cache row as u/xi/beta); there is
# deliberately no default that could silently reproduce the v2.0 behavior.

def e3_quantile_from_fit(grid: dict, q50c: float, u: float, xi: float,
                         beta: float, alpha: float, tau_anchor: float = 0.10,
                         *, tau_mass: float) -> float:
    assert tau_anchor in grid, f"model grid lacks the splice anchor tau_anchor={tau_anchor}"
    q_ctx = gpd_tail_quantile(u, xi, beta, tau_mass, alpha)
    a = (grid[tau_anchor] - grid[0.5]) / (u - q50c)
    return float(grid[0.5] + a * (q_ctx - q50c))


def e3_es_from_fit(grid: dict, q50c: float, u: float, xi: float, beta: float,
                   alpha: float, tau_anchor: float = 0.10,
                   *, tau_mass: float) -> float:
    assert tau_anchor in grid, f"model grid lacks the splice anchor tau_anchor={tau_anchor}"
    es_ctx = gpd_tail_es(u, xi, beta, tau_mass, alpha)
    a = (grid[tau_anchor] - grid[0.5]) / (u - q50c)
    return float(grid[0.5] + a * (es_ctx - q50c))


def e3_var_es_from_fit(grid: dict, q50c: float, u: float, xi: float, beta: float,
                       alpha: float, tau_anchor: float = 0.10,
                       *, tau_mass: float) -> tuple[float, float]:
    """E3 (VaR, ES) with INDEPENDENT exception paths (v2.0 P0-1).

    The GPD quantile is well-defined for any xi, so VaR stays finite even when
    the mean-excess ES integral diverges (xi >= 1): ES-nonexistence never voids
    the VaR. ES is NaN exactly when xi >= 1 or its own map fails.

    v2.1 trap-1 guard (review #10 §2.4): the anchor-membership and mass-sanity
    checks are asserts OUTSIDE the try blocks — production callers wrap this
    function in `except (ValueError, KeyError)` day-NaN handlers, and an
    anchor/mass wiring error must fail the RUN, not silently NaN the E3 layer."""
    assert tau_anchor in grid, f"model grid lacks the splice anchor tau_anchor={tau_anchor}"
    assert np.isfinite(tau_mass) and 0.0 < tau_mass < 1.0, \
        f"tau_mass must be a finite probability, got {tau_mass!r}"
    try:
        v = e3_quantile_from_fit(grid, q50c, u, xi, beta, alpha, tau_anchor,
                                 tau_mass=tau_mass)
    except (ValueError, KeyError):
        v = float("nan")
    try:
        e = e3_es_from_fit(grid, q50c, u, xi, beta, alpha, tau_anchor,
                           tau_mass=tau_mass)
    except (ValueError, KeyError):
        e = float("nan")
    return v, e
