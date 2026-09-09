"""ER vs esback (MC tolerance — different RNG streams); AS-Z2 analytic + sanity."""

import numpy as np
import pytest

from backtests.es_tests import as_z2, er_backtest
from tests.conftest import ref_value, series

CASES = [(fc, a) for fc in ("oracle", "hs") for a in (0.01, 0.025, 0.05)]


@pytest.mark.parametrize("fc,alpha", CASES)
def test_er_matches_esback_within_mc_error(inp, r_refs, fc, alpha):
    """esback er_backtest 'simple' used B=2000 (internal set.seed(1)); ours uses
    numpy with B=20000. p-values agree up to bootstrap MC error: se ≈ 0.011 at
    B=2000, so |Δp| < 0.04 is a ~3.5σ band."""
    y, v, e = series(inp, fc, alpha)
    ours = er_backtest(y, v, e, n_boot=20000, seed=1)
    assert ours["p_twosided"] == pytest.approx(
        ref_value(r_refs, fc, alpha, "esback_er", "p2_simple"), abs=0.04)
    assert ours["p_onesided"] == pytest.approx(
        ref_value(r_refs, fc, alpha, "esback_er", "p1_simple"), abs=0.04)


def test_as_z2_analytic():
    # 2 hits of y=-4 with e=-2 among n=10 at α=0.2: Z2 = (2·2)/(10·0.2) - 1 = 1.0
    y = np.array([-4.0, -4.0] + [1.0] * 8)
    v = np.full(10, -3.0)
    e = np.full(10, -2.0)
    out = as_z2(y, v, e, alpha=0.2)
    assert out["n_hits"] == 2
    assert out["stat"] == pytest.approx(1.0)


def test_as_z2_oracle_near_zero_and_orders_forecasters(inp):
    """On the frozen path the oracle's Z2 must sit near 0; a deliberately halved
    ES (too optimistic) must push Z2 up by construction."""
    y, v, e = series(inp, "oracle", 0.025)
    z_ok = as_z2(y, v, e, 0.025)["stat"]
    z_bad = as_z2(y, v, e / 2.0, 0.025)["stat"]
    assert abs(z_ok) < 0.15
    assert z_bad > z_ok + 0.5


def test_as_z2_bootstrap_p_deterministic(inp):
    y, v, e = series(inp, "oracle", 0.05)
    a = as_z2(y, v, e, 0.05, n_boot=5000, seed=42)
    b = as_z2(y, v, e, 0.05, n_boot=5000, seed=42)
    assert a["p_onesided"] == b["p_onesided"]
    assert 0.05 < a["p_onesided"] <= 1.0  # oracle must not be rejected
