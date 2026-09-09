"""DQ size property test (interim reference until the GAS fixture lands).

Under H0 — violations iid Bernoulli(α), independent of the VaR path — the DQ
statistic is asymptotically χ²(k), so its p-values must be ≈ Uniform(0,1).
2000 seeded replications at α=5%, n=1000; assert the KS distance to U(0,1) is
small and the 5%-level rejection rate is near nominal.
"""

import numpy as np
from scipy import stats

from backtests.var_tests import dq_test

N_SIM = 2000
N = 1000
ALPHA = 0.05


def test_dq_pvalues_uniform_under_h0():
    rng = np.random.default_rng(20260704)
    pvals = np.empty(N_SIM)
    for i in range(N_SIM):
        v = -1.0 + 0.1 * rng.standard_normal(N)      # varying VaR path (no collinearity)
        hit = rng.random(N) < ALPHA                  # iid Bernoulli(α), indep. of v
        eps = np.abs(rng.standard_normal(N)) + 0.05
        y = np.where(hit, v - eps, v + eps)
        pvals[i] = dq_test(y, v, ALPHA, lags=4)["p"]

    ks = stats.kstest(pvals, "uniform")
    rej05 = float((pvals < 0.05).mean())

    # χ² asymptotics leave a little finite-sample distortion; bands are generous
    # but would catch any real implementation error (wrong df, wrong scaling).
    assert ks.statistic < 0.05, f"KS D={ks.statistic:.4f} (p={ks.pvalue:.4f})"
    assert 0.03 <= rej05 <= 0.075, f"5% rejection rate {rej05:.3f}"
