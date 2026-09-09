"""Monte-Carlo finite-sample critical values / p-values for the VaR backtests
(gate condition 2 — the judgment lock).

The classical UC/CC/DQ statistics use asymptotic chi-square reference distributions
that are size-distorted at the small effective sample sizes of deep-tail backtesting
(the pilot documented CC over-rejecting ~9-12% at alpha=5%, n=1250 under exact H0;
proposal red-team #5). Under the null "violations are iid Bernoulli(alpha),
independent of the information set", the exact finite-sample distribution of each
statistic depends only on (n, alpha) and can be simulated. This module returns
MC critical values and MC p-values, cached per (statistic, n, alpha, B, seed).

Design decisions:
- The null is the SAME iid-Bernoulli(alpha) null the classical tests assume, so
  simulated hits need no VaR path — the statistics are functions of the hit
  sequence alone (Kupiec, Christoffersen) or of hits + regressors (DQ, where the
  lagged-hit and constant regressors are used; the contemporaneous-VaR regressor
  is dropped under the iid null since VaR is exogenous and its scale cancels).
- Bucketed by exact n keeps memory bounded; callers pass their real n.
"""

from __future__ import annotations

import numpy as np

from backtests.var_tests import (christoffersen_cc, christoffersen_independence,
                                 dq_test, kupiec)

_STATS = {
    "kupiec": lambda h, a: kupiec_stat_from_hits(h, a),
    "cc": lambda h, a: cc_stat_from_hits(h, a),
    "ind": lambda h, a: christoffersen_independence(_y(h), _v())["stat"],
    "dq": lambda h, a: dq_stat_from_hits(h, a),
}


def _v():
    return None


def _y(hits: np.ndarray) -> np.ndarray:
    # encode a hit as y<=v: y=-1 on hit, +1 otherwise, against v=0
    return np.where(hits.astype(bool), -1.0, 1.0)


def _V(n: int) -> np.ndarray:
    return np.zeros(n)


def kupiec_stat_from_hits(hits: np.ndarray, alpha: float) -> float:
    return kupiec(_y(hits), _V(len(hits)), alpha)["stat"]


def cc_stat_from_hits(hits: np.ndarray, alpha: float) -> float:
    return christoffersen_cc(_y(hits), _V(len(hits)), alpha)["stat"]


def dq_stat_from_hits(hits: np.ndarray, alpha: float, lags: int = 4) -> float:
    # iid null: DQ on constant + lagged hits only (no contemporaneous VaR term,
    # which is exogenous and scale-free under the null)
    return dq_test(_y(hits), _V(len(hits)), alpha, lags=lags, include_var=False)["stat"]


class MCNull:
    """Simulated null distribution of a statistic at fixed (n, alpha)."""

    def __init__(self, statistic: str, n: int, alpha: float,
                 B: int = 20000, seed: int = 20260706):
        if statistic not in _STATS:
            raise ValueError(f"unknown statistic {statistic!r}")
        self.statistic, self.n, self.alpha, self.B, self.seed = statistic, n, alpha, B, seed
        rng = np.random.default_rng(seed)
        fn = _STATS[statistic]
        draws = np.empty(B)
        hits_mat = (rng.random((B, n)) < alpha)
        for b in range(B):
            draws[b] = fn(hits_mat[b], alpha)
        self.draws = np.sort(draws)

    def critical_value(self, level: float = 0.05) -> float:
        """Upper-tail critical value: reject if stat >= this."""
        return float(np.quantile(self.draws, 1.0 - level))

    def p_value(self, stat: float) -> float:
        """MC p-value P(T >= stat) with the (b+1)/(B+1) plug-in (never 0)."""
        ge = int((self.draws >= stat).sum())
        return (ge + 1) / (self.B + 1)


def mc_pvalue(statistic: str, hits_or_stat, n: int, alpha: float, *,
              is_stat: bool = False, B: int = 20000, seed: int = 20260706) -> float:
    null = MCNull(statistic, n, alpha, B=B, seed=seed)
    stat = float(hits_or_stat) if is_stat else _STATS[statistic](
        np.asarray(hits_or_stat), alpha)
    return null.p_value(stat)
