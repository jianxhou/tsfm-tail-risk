"""Training-free repair arms F1–F4 + hybrid H and the GARCH-EVT comparator
(design v2.1 §A; all constants pre-registered there). Each arm is a pure function
producing a repaired (v, e) series over the OOS targets. Window-history access is
via harness.rolling.calibration_view (I9) for F1/F3/F4/H/GARCH-EVT; F2 is a
sequential recursion that reads no window — it is leak-free by construction
(c updates on y[i] strictly after (v,e)[i] is emitted; audit A14). No arm reads
its own or any future row. Leak sentinel: tests/test_fixes.py covers every arm.

Sign convention: lower-tail v, e negative, e < v < 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from erules.rules import fit_gpd, gpd_tail_es, gpd_tail_quantile, mad
from harness.rolling import calibration_view

W = 500                 # calibration window (design §A)
REFIT = 21
TAU_U = 0.10            # POT threshold (F1/H/E3/comparator all 10%)
SEED = 20260706


# ---------- EVT-on-residuals core (F1, H, GARCH-EVT share this) ----------

def _evt_from_residuals(z_hist: np.ndarray, m_t: float, s_t: float, alphas):
    """POT-GPD on the residual window (fit ONCE), evaluated at each alpha. Returns
    (v_dict, e_dict) keyed by alpha. `alphas` may be a scalar (back-compat) or a
    list. NaN dicts if the fit fails. Fitting once and evaluating all alphas is
    results-identical to per-alpha fitting (the GPD is alpha-independent)."""
    scalar = np.isscalar(alphas)
    al = [alphas] if scalar else list(alphas)
    if len(z_hist) < 100:
        nan = {a: np.nan for a in al}
        return (nan[al[0]] if scalar else nan, nan[al[0]] if scalar else nan)
    u = float(np.quantile(z_hist, TAU_U))
    exc = u - z_hist[z_hist < u]
    vd, ed = {}, {}
    try:
        # v2.1 tail-mass ruling clause 9: the repair arms KEEP TAU_U=0.10 as
        # both anchor and extrapolation mass — on the W=500 standardized-
        # residual windows the empirical threshold mass is exactly 0.10 with
        # zero exceptions (measured: 8 assets, 17,268 windows), so the two
        # conventions coincide here (docs/e3_tail_mass_ruling.md).
        xi, beta, _ = fit_gpd(exc, scale_ref=mad(z_hist))   # v2.0 P0-1 tie-robust fit
        for a in al:
            vd[a] = float(m_t + s_t * gpd_tail_quantile(u, xi, beta, TAU_U, a))
            # gpd_tail_es raises when xi>=1 (ES undefined); NaN that ES, keep the VaR
            try:
                ed[a] = float(m_t + s_t * gpd_tail_es(u, xi, beta, TAU_U, a))
            except ValueError:
                ed[a] = np.nan
    except ValueError:
        vd = {a: np.nan for a in al}; ed = {a: np.nan for a in al}
    if scalar:
        return vd[al[0]], ed[al[0]]
    return vd, ed


def f1_tsfm_filtered_evt(aligned: pd.DataFrame, alpha: float):
    """F1: location m_t=q50, scale s_t=(q75-q25)/1.349 from the TSFM grid;
    EVT on standardized residuals over the trailing W. Substitution repair."""
    m = aligned["q50"].to_numpy()
    s = ((aligned["q75"] - aligned["q25"]) / 1.349).to_numpy()
    y = aligned["y"].to_numpy()
    vv = np.full(len(aligned), np.nan); ee = np.full(len(aligned), np.nan)
    for i in range(len(aligned)):
        h = calibration_view(aligned, aligned["t"].iloc[i], window=W)
        if len(h) < 100:
            continue
        mh = h["q50"].to_numpy(); sh = ((h["q75"] - h["q25"]) / 1.349).to_numpy()
        z = (h["y"].to_numpy() - mh) / sh
        vv[i], ee[i] = _evt_from_residuals(z, m[i], s[i], alpha)
    return vv, ee


def h_hybrid(aligned: pd.DataFrame, sigma: np.ndarray, alpha: float):
    """H: TSFM location m_t=q50, GARCH scale sigma_t; EVT on residuals. Isolates
    the scale source vs F1 (TSFM IQR) at fixed TSFM location."""
    m = aligned["q50"].to_numpy()
    vv = np.full(len(aligned), np.nan); ee = np.full(len(aligned), np.nan)
    for i in range(len(aligned)):
        h = calibration_view(aligned, aligned["t"].iloc[i], window=W)
        idx = h.index
        if len(h) < 100:
            continue
        mh = h["q50"].to_numpy(); sh = sigma[idx.to_numpy()]
        z = (h["y"].to_numpy() - mh) / sh
        vv[i], ee[i] = _evt_from_residuals(z, m[i], sigma[i], alpha)
    return vv, ee


def garch_evt(aligned: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, alpha: float):
    """Classical McNeil–Frey comparator: GARCH-t location mu_t + scale sigma_t;
    EVT on standardized residuals. No TSFM input."""
    vv = np.full(len(aligned), np.nan); ee = np.full(len(aligned), np.nan)
    for i in range(len(aligned)):
        h = calibration_view(aligned, aligned["t"].iloc[i], window=W)
        idx = h.index.to_numpy()
        if len(h) < 100:
            continue
        z = (h["y"].to_numpy() - mu[idx]) / sigma[idx]
        vv[i], ee[i] = _evt_from_residuals(z, mu[i], sigma[i], alpha)
    return vv, ee


# ---------- multi-alpha EVT (grid driver: fit GPD once per window, all alphas) ----------

def f1_multi(aligned: pd.DataFrame, alphas):
    m = aligned["q50"].to_numpy()
    s = ((aligned["q75"] - aligned["q25"]) / 1.349).to_numpy()
    out = {a: (np.full(len(aligned), np.nan), np.full(len(aligned), np.nan)) for a in alphas}
    for i in range(len(aligned)):
        h = calibration_view(aligned, aligned["t"].iloc[i], window=W)
        if len(h) < 100:
            continue
        mh = h["q50"].to_numpy(); sh = ((h["q75"] - h["q25"]) / 1.349).to_numpy()
        z = (h["y"].to_numpy() - mh) / sh
        vd, ed = _evt_from_residuals(z, m[i], s[i], alphas)
        for a in alphas:
            out[a][0][i] = vd[a]; out[a][1][i] = ed[a]
    return out


def h_multi(aligned: pd.DataFrame, sigma: np.ndarray, alphas):
    m = aligned["q50"].to_numpy()
    out = {a: (np.full(len(aligned), np.nan), np.full(len(aligned), np.nan)) for a in alphas}
    for i in range(len(aligned)):
        h = calibration_view(aligned, aligned["t"].iloc[i], window=W)
        idx = h.index.to_numpy()
        if len(h) < 100:
            continue
        z = (h["y"].to_numpy() - h["q50"].to_numpy()) / sigma[idx]
        vd, ed = _evt_from_residuals(z, m[i], sigma[i], alphas)
        for a in alphas:
            out[a][0][i] = vd[a]; out[a][1][i] = ed[a]
    return out


def garch_evt_multi(aligned: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, alphas):
    out = {a: (np.full(len(aligned), np.nan), np.full(len(aligned), np.nan)) for a in alphas}
    for i in range(len(aligned)):
        h = calibration_view(aligned, aligned["t"].iloc[i], window=W)
        idx = h.index.to_numpy()
        if len(h) < 100:
            continue
        z = (h["y"].to_numpy() - mu[idx]) / sigma[idx]
        vd, ed = _evt_from_residuals(z, mu[i], sigma[i], alphas)
        for a in alphas:
            out[a][0][i] = vd[a]; out[a][1][i] = ed[a]
    return out


# ---------- v2.0 decomposition CONTROL arms (external review #9 §9.1) ----------
# Post-hoc reviewer-suggested robustness arms, NOT pre-specified (CHANGELOG v2.0):
#   X4 — the 2x2 fourth cell: GARCH conditional-mean location mu_t + TSFM IQR
#        scale s_t (completes F1 / H / GARCH-EVT to the full location-source x
#        scale-source factorial; interaction term becomes estimable).
#   ZL — zero location + GARCH scale sigma_t.
#   CL — constant location (trailing-W mean of y, history-only) + GARCH scale.
# All three share the _evt_from_residuals core and the W/TAU_U constants.

def x4_multi(aligned: pd.DataFrame, mu: np.ndarray, alphas):
    """Fourth cell: location = GARCH mu_t, scale = TSFM IQR s_t; EVT on residuals."""
    s = ((aligned["q75"] - aligned["q25"]) / 1.349).to_numpy()
    out = {a: (np.full(len(aligned), np.nan), np.full(len(aligned), np.nan)) for a in alphas}
    for i in range(len(aligned)):
        h = calibration_view(aligned, aligned["t"].iloc[i], window=W)
        idx = h.index.to_numpy()
        if len(h) < 100:
            continue
        sh = ((h["q75"] - h["q25"]) / 1.349).to_numpy()
        z = (h["y"].to_numpy() - mu[idx]) / sh
        vd, ed = _evt_from_residuals(z, mu[i], s[i], alphas)
        for a in alphas:
            out[a][0][i] = vd[a]; out[a][1][i] = ed[a]
    return out


def zero_loc_multi(aligned: pd.DataFrame, sigma: np.ndarray, alphas):
    """Zero-location control: m_t = 0, scale = GARCH sigma_t; EVT on residuals."""
    out = {a: (np.full(len(aligned), np.nan), np.full(len(aligned), np.nan)) for a in alphas}
    for i in range(len(aligned)):
        h = calibration_view(aligned, aligned["t"].iloc[i], window=W)
        idx = h.index.to_numpy()
        if len(h) < 100:
            continue
        z = h["y"].to_numpy() / sigma[idx]
        vd, ed = _evt_from_residuals(z, 0.0, sigma[i], alphas)
        for a in alphas:
            out[a][0][i] = vd[a]; out[a][1][i] = ed[a]
    return out


def const_loc_multi(aligned: pd.DataFrame, sigma: np.ndarray, alphas):
    """Constant-location control: m_t = trailing-W mean of y (history-only,
    refreshed per window), scale = GARCH sigma_t; EVT on residuals."""
    out = {a: (np.full(len(aligned), np.nan), np.full(len(aligned), np.nan)) for a in alphas}
    for i in range(len(aligned)):
        h = calibration_view(aligned, aligned["t"].iloc[i], window=W)
        idx = h.index.to_numpy()
        if len(h) < 100:
            continue
        m = float(h["y"].to_numpy().mean())
        z = (h["y"].to_numpy() - m) / sigma[idx]
        vd, ed = _evt_from_residuals(z, m, sigma[i], alphas)
        for a in alphas:
            out[a][0][i] = vd[a]; out[a][1][i] = ed[a]
    return out


# ---------- transformation repairs on the E3 series (F2, F3, F4) ----------

def f2_adaptive_conformal(aligned: pd.DataFrame, alpha: float, s_hat: float,
                          gamma_mult: float = 1.0):
    """F2: online additive offset c on (v_E3, e_E3). c_{t+1}=c_t-γ(1{y<=v+c}-α),
    from OOS day 1 (warm at eval). γ = 0.01·s_hat·gamma_mult, fixed."""
    v = aligned["v"].to_numpy(); e = aligned["e"].to_numpy(); y = aligned["y"].to_numpy()
    gamma = 0.01 * s_hat * gamma_mult
    c = 0.0
    vv = np.empty(len(v)); ee = np.empty(len(v))
    for i in range(len(v)):
        vv[i] = v[i] + c; ee[i] = e[i] + c
        c = c - gamma * ((1.0 if y[i] <= v[i] + c else 0.0) - alpha)
    return vv, ee


def _pinball(y, v, alpha):
    u = y - v
    return np.sum(u * (alpha - (u < 0).astype(float)))


def f3_rescale(aligned: pd.DataFrame, alpha: float):
    """F3 control: per-α multiplicative k on (v,e), grid k∈[0.5,3.0] step .001
    minimizing window pinball; tie-break smallest k; refit every 21."""
    v = aligned["v"].to_numpy(); e = aligned["e"].to_numpy(); y = aligned["y"].to_numpy()
    ks = np.arange(0.5, 3.0 + 1e-9, 0.001)
    vv = np.empty(len(v)); ee = np.empty(len(v)); k = 1.0
    for i in range(len(v)):
        if i % REFIT == 0:
            h = calibration_view(aligned, aligned["t"].iloc[i], window=W).dropna(subset=["v"])
            if len(h) >= 100:
                yh = h["y"].to_numpy(); vh = h["v"].to_numpy()
                losses = np.array([_pinball(yh, kk * vh, alpha) for kk in ks])
                k = float(ks[int(np.argmin(losses))])
        vv[i] = k * v[i]; ee[i] = k * e[i]
    return vv, ee


def _fz0_shift(params, y, v, e, alpha):
    q, r = params
    V = v + q; E = e + r
    if np.any(E >= -1e-10) or np.any(E >= V - 1e-10):
        return 1e12
    hit = (y <= V).astype(float)
    return np.mean(hit * (y - V) / (alpha * E) + V / E + np.log(-E) - 1.0)


def f4_additive_fz0(aligned: pd.DataFrame, alpha: float, s_hat: float):
    """F4: (q,r) additive shifts minimizing window FZ0 (HMD_ES rolling_recalib
    convention, restated numerically); W=500, refit 21, NM warm+2 restarts."""
    from scipy.optimize import minimize
    v = aligned["v"].to_numpy(); e = aligned["e"].to_numpy(); y = aligned["y"].to_numpy()
    rng = np.random.default_rng(SEED)
    vv = np.empty(len(v)); ee = np.empty(len(v)); sol = [0.0, 0.0]
    for i in range(len(v)):
        if i % REFIT == 0:
            h = calibration_view(aligned, aligned["t"].iloc[i], window=W).dropna(subset=["v", "e"])
            if len(h) >= 100:
                yh, vh, eh = h["y"].to_numpy(), h["v"].to_numpy(), h["e"].to_numpy()
                best_f, best = np.inf, tuple(sol)
                starts = [sol] + [[rng.uniform(-s_hat, s_hat), rng.uniform(-s_hat, s_hat)]
                                  for _ in range(2)]
                for st in starts:
                    res = minimize(_fz0_shift, st, args=(yh, vh, eh, alpha),
                                   method="Nelder-Mead",
                                   options={"maxiter": 1500, "xatol": 1e-6, "fatol": 1e-9})
                    if res.fun < best_f:
                        best_f, best = res.fun, tuple(res.x)
                sol = list(best)
        vv[i] = v[i] + sol[0]; ee[i] = e[i] + sol[1]
    return vv, ee
