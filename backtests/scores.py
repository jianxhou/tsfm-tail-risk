"""Loss functions for quantile/ES forecast evaluation.

Sign convention throughout the repo: returns and forecasts in return units
(log-return ×100); VaR/ES forecasts for the LOWER tail are negative numbers,
with e < v < 0 required for FZ0.
"""

from __future__ import annotations

import numpy as np


def quantile_score(y: np.ndarray, v: np.ndarray, alpha: float) -> np.ndarray:
    """Pinball/check loss per observation: (1{y<=v} - α)(v - y). Lower is better."""
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    return (np.where(y <= v, 1.0, 0.0) - alpha) * (v - y)


def fz0(y: np.ndarray, v: np.ndarray, e: np.ndarray, alpha: float) -> np.ndarray:
    """FZ0 joint (VaR, ES) loss per observation (Fissler–Ziegel, 0-homogeneous;
    Patton–Ziegel–Chen 2019 eq. 4). Guard matches QuantLet HMD_ES `fz_loss`
    at params=(0,0): requires e_t < 0 and e_t < v_t (v may in principle be
    positive; log(-e) only needs e < 0).

        FZ0_t = 1{y<=v}·(y-v)/(α·e) + v/e + log(-e) - 1
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    e = np.asarray(e, dtype=float)
    if np.any(e >= 0) or np.any(e >= v):
        raise ValueError("FZ0 requires e < v < 0 elementwise")
    hit = np.where(y <= v, 1.0, 0.0)
    return hit * (y - v) / (alpha * e) + v / e + np.log(-e) - 1.0
