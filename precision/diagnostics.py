"""Finite-sample precision diagnostics for ES comparisons.

Implements the Pele & Mazurencu-Marinescu-Pele (2026) machinery per proposal §3.4:
every (asset, model, α) cell carries its effective tail count and plug-in precision
floor, and every pairwise ES comparison carries a precision-fragile flag.
Conventions mirror QuantLet/HMD_ES (rolling_c_robustness.py / pipeline.py).
"""

from __future__ import annotations

import numpy as np


def n_alpha(n: int, alpha: float) -> float:
    """Effective tail sample size nα — the information bound driver: SE ∝ (nα)^(-1/2)."""
    return float(n) * float(alpha)


def sigma_tail(y: np.ndarray, v: np.ndarray, min_hits: int = 3) -> float:
    """Tail dispersion Ĉ: sample std of exceedance depths (v - y | y <= v).

    QuantLet convention (pipeline.py): ddof=1, NaN when fewer than `min_hits` hits.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    depth = v[y <= v] - y[y <= v]
    if len(depth) < min_hits:
        return float("nan")
    return float(np.std(depth, ddof=1))


def precision_floor(sigma: float, n_alpha_: float) -> float:
    """Plug-in one-forecaster precision floor: Ĉ / sqrt(nα)."""
    return float(sigma) / np.sqrt(n_alpha_)


def fragile_flag(diff: float, sigma_a: float, sigma_b: float, n_alpha_: float) -> dict:
    """Pairwise precision-fragile screen (QuantLet rolling_c_robustness.py):

        bound = sqrt(Ĉ_a² + Ĉ_b²) / sqrt(nα);   fragile ⟺ |diff| < bound

    A fragile pair means the observed loss/recalibration difference sits below the
    finite-sample identifiability floor — no substantive ranking claim is allowed.
    """
    bound = np.sqrt(sigma_a ** 2 + sigma_b ** 2) / np.sqrt(n_alpha_)
    return {"bound": float(bound), "fragile": bool(abs(diff) < bound)}
