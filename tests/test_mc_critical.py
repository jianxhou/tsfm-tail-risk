"""MC-calibration acceptance (gate condition 2). Assertions, not smoke:
the MC test must be correctly SIZED under the exact null where the asymptotic
chi-square test is size-distorted, and must agree with chi-square where n*alpha
is large enough for the asymptotics to hold."""

import numpy as np
import pytest
from scipy import stats

from precision.mc_critical import MCNull, mc_pvalue


def _sim_size(statistic, n, alpha, use_mc, n_rep=1500, seed=7):
    """Empirical rejection rate at level 0.05 under exact iid-Bernoulli(alpha)."""
    rng = np.random.default_rng(seed)
    null = MCNull(statistic, n, alpha, B=8000, seed=999) if use_mc else None
    rej = 0
    for _ in range(n_rep):
        hits = rng.random(n) < alpha
        if use_mc:
            p = null.p_value(_stat(statistic, hits, alpha))
        else:
            p = _chi2_p(statistic, hits, alpha)
        rej += p < 0.05
    return rej / n_rep


def _stat(statistic, hits, alpha):
    from precision.mc_critical import _STATS
    return _STATS[statistic](np.asarray(hits), alpha)


def _chi2_p(statistic, hits, alpha):
    from precision.mc_critical import cc_stat_from_hits, kupiec_stat_from_hits
    if statistic == "kupiec":
        return stats.chi2.sf(kupiec_stat_from_hits(hits, alpha), 1)
    if statistic == "cc":
        return stats.chi2.sf(cc_stat_from_hits(hits, alpha), 2)
    raise ValueError


def test_mc_kupiec_correctly_sized_where_chi2_distorts():
    """alpha=5%, n=1250: MC size near 0.05; and MC is at least as well-sized as chi2."""
    n, a = 1250, 0.05
    size_mc = _sim_size("kupiec", n, a, use_mc=True)
    size_chi2 = _sim_size("kupiec", n, a, use_mc=False)
    assert abs(size_mc - 0.05) <= 0.02, f"MC size {size_mc}"
    assert abs(size_mc - 0.05) <= abs(size_chi2 - 0.05) + 0.01


def test_mc_cc_fixes_documented_overrejection():
    """The pilot's CC over-rejection (~9-12% at nominal 5%, n=1250) must be
    corrected by MC calibration to near 5%."""
    n, a = 1250, 0.05
    size_mc = _sim_size("cc", n, a, use_mc=True)
    size_chi2 = _sim_size("cc", n, a, use_mc=False)
    assert size_chi2 > 0.075, f"expected chi2 CC over-rejection, got {size_chi2}"
    assert abs(size_mc - 0.05) <= 0.02, f"MC CC size {size_mc}"


def test_mc_kupiec_pvalue_conservative_under_null():
    """Kupiec's statistic is DISCRETE (a function of the hit COUNT ~ Binomial),
    so its MC p-value cannot be continuous-uniform — with the P(T>=t) convention
    it is CONSERVATIVE. The correct acceptance property is size control:
    P(p <= u) <= u (+ MC/discreteness tolerance) at every level u."""
    n, a = 1000, 0.025
    null = MCNull("kupiec", n, a, B=8000, seed=1)
    rng = np.random.default_rng(3)
    ps = np.array([null.p_value(_stat("kupiec", rng.random(n) < a, a))
                   for _ in range(2000)])
    for u in (0.05, 0.10, 0.20):
        assert (ps <= u).mean() <= u + 0.03, f"level {u}: size {(ps<=u).mean():.3f}"


def test_mc_dq_pvalue_near_uniform_under_null():
    """DQ is a quadratic form in the hit PATTERN (not just the count), so it is
    finely grained and its MC p-value is close to continuous-uniform."""
    n, a = 1500, 0.05
    from precision.mc_critical import MCNull as _M
    null = _M("dq", n, a, B=8000, seed=1)
    rng = np.random.default_rng(3)
    ps = [null.p_value(_stat("dq", rng.random(n) < a, a)) for _ in range(800)]
    assert stats.kstest(ps, "uniform").statistic < 0.06


def test_mc_agrees_with_chi2_in_large_nalpha_limit():
    """Where n*alpha is large, the MC critical value approaches the chi2 one."""
    n, a = 20000, 0.05     # n*alpha = 1000
    cv_mc = MCNull("kupiec", n, a, B=12000, seed=2).critical_value(0.05)
    cv_chi2 = stats.chi2.ppf(0.95, 1)
    assert abs(cv_mc - cv_chi2) < 0.4


def test_mc_pvalue_never_zero_and_monotone():
    null = MCNull("kupiec", 1250, 0.01, B=5000, seed=5)
    assert null.p_value(1e9) == pytest.approx(1 / 5001)      # (0+1)/(B+1)
    assert null.p_value(-1.0) == pytest.approx(1.0)          # all draws >= -1
    assert null.p_value(5.0) >= null.p_value(10.0)


def test_dq_null_reproducible_and_bounded():
    p1 = mc_pvalue("dq", np.random.default_rng(0).random(1000) < 0.05, 1000, 0.05, B=3000)
    p2 = mc_pvalue("dq", np.random.default_rng(0).random(1000) < 0.05, 1000, 0.05, B=3000)
    assert p1 == p2 and 0 < p1 <= 1
