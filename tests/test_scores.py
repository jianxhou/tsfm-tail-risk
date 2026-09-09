"""quantile_score analytic cases; fz0 vs the QuantLet HMD_ES reference."""

import numpy as np
import pytest

from backtests.scores import fz0, quantile_score
from tests.conftest import series


def test_quantile_score_analytic():
    # y above the quantile: (0-α)(v-y) = α(y-v); below: (1-α)(v-y)
    assert quantile_score(np.array([1.0]), np.array([-2.0]), 0.05)[0] \
        == pytest.approx(0.05 * 3.0)
    assert quantile_score(np.array([-3.0]), np.array([-2.0]), 0.05)[0] \
        == pytest.approx(0.95 * 1.0)
    # tie counts as a hit, loss 0
    assert quantile_score(np.array([-2.0]), np.array([-2.0]), 0.05)[0] == 0.0


def test_quantile_score_minimized_at_true_quantile():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(200_000)
    a = 0.05
    q_true = float(np.quantile(y, a))
    s_true = quantile_score(y, np.full_like(y, q_true), a).mean()
    for q in (q_true - 0.2, q_true + 0.2):
        assert quantile_score(y, np.full_like(y, q), a).mean() > s_true


@pytest.mark.parametrize("fc", ["oracle", "hs"])
@pytest.mark.parametrize("alpha", [0.01, 0.025, 0.05])
def test_fz0_matches_quantlet(inp, quantlet_refs, fc, alpha):
    y, v, e = series(inp, fc, alpha)
    ours = fz0(y, v, e, alpha).mean()
    ql = quantlet_refs
    ref = float(ql[(ql.forecaster == fc) & (ql.alpha == alpha)
                   & (ql.quantity == "fz0_mean")].value.iloc[0])
    assert ours == pytest.approx(ref, rel=1e-10)


def test_fz0_rejects_bad_convention():
    with pytest.raises(ValueError):
        fz0(np.array([0.0]), np.array([2.0]), np.array([1.0]), 0.05)   # e > 0
    with pytest.raises(ValueError):
        fz0(np.array([0.0]), np.array([-2.0]), np.array([-1.0]), 0.05)  # e > v
