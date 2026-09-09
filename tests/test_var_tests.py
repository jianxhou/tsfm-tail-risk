"""kupiec / christoffersen vs rugarch::VaRTest; DQ interim checks (GAS fixture TODO)."""

import numpy as np
import pytest

from backtests.var_tests import (christoffersen_cc, christoffersen_independence,
                                 dq_test, kupiec)
from tests.conftest import ref_value, series

CASES = [(fc, a) for fc in ("oracle", "hs") for a in (0.01, 0.025, 0.05)]


@pytest.mark.parametrize("fc,alpha", CASES)
def test_kupiec_matches_rugarch(inp, r_refs, fc, alpha):
    y, v, _ = series(inp, fc, alpha)
    out = kupiec(y, v, alpha)
    assert out["stat"] == pytest.approx(
        ref_value(r_refs, fc, alpha, "rugarch_VaRTest", "uc_stat"), rel=1e-6)
    assert out["p"] == pytest.approx(
        ref_value(r_refs, fc, alpha, "rugarch_VaRTest", "uc_p"), abs=1e-8)


@pytest.mark.parametrize("fc,alpha", CASES)
def test_christoffersen_cc_matches_rugarch(inp, r_refs, fc, alpha):
    y, v, _ = series(inp, fc, alpha)
    out = christoffersen_cc(y, v, alpha)
    assert out["stat"] == pytest.approx(
        ref_value(r_refs, fc, alpha, "rugarch_VaRTest", "cc_stat"), rel=1e-6)
    assert out["p"] == pytest.approx(
        ref_value(r_refs, fc, alpha, "rugarch_VaRTest", "cc_p"), abs=1e-8)


def test_quantlet_kupiec_bug_documented(inp, quantlet_refs, r_refs):
    """HMD_ES pipeline.py::kupiec_p uses n·ln(1-α) in the null loglik where the POF
    formula requires (n-x)·ln(1-α); its LR is inflated by -2·x·ln(1-α). Our kupiec
    must match rugarch, NOT the QuantLet value; this test pins the divergence AND
    reproduces their number exactly once the bug term is added back."""
    y, v, _ = series(inp, "oracle", 0.05)
    ours = kupiec(y, v, 0.05)
    rugarch_p = ref_value(r_refs, "oracle", 0.05, "rugarch_VaRTest", "uc_p")
    ql = quantlet_refs
    ql_p = float(ql[(ql.forecaster == "oracle") & (ql.alpha == 0.05)
                    & (ql.quantity == "kupiec_p")].value.iloc[0])

    assert ours["p"] == pytest.approx(rugarch_p, abs=1e-8)
    assert abs(ours["p"] - ql_p) > 0.1  # the bug is material on this input

    # reconstruct their statistic: LR_theirs = LR_ours - 2·x·ln(1-α)
    from scipy import stats as st
    x = ours["hits"]
    lr_theirs = ours["stat"] - 2.0 * x * np.log(1 - 0.05)
    assert st.chi2.sf(lr_theirs, 1) == pytest.approx(ql_p, rel=1e-6)


def test_cc_decomposition(inp):
    y, v, _ = series(inp, "hs", 0.025)
    cc = christoffersen_cc(y, v, 0.025)
    assert cc["stat"] == pytest.approx(cc["uc"]["stat"] + cc["ind"]["stat"], rel=1e-12)
    ind = christoffersen_independence(y, v)
    assert ind["stat"] == pytest.approx(cc["ind"]["stat"], rel=1e-12)


def test_dq_basics(inp):
    """Interim DQ checks until the GAS reference lands (CHANGELOG TODO):
    oracle forecasts should not be rejected wildly; a hand-built persistent-hit
    series must be rejected hard."""
    y, v, _ = series(inp, "oracle", 0.05)
    out = dq_test(y, v, 0.05, lags=4)
    assert out["df"] == 6
    assert 0.0 <= out["p"] <= 1.0
    assert out["p"] > 0.001

    # clustered violations: 60 hits all in a row -> massive serial dependence
    n = 1000
    y_bad = np.zeros(n)
    v_bad = np.full(n, -1.0)
    y_bad[100:160] = -2.0
    bad = dq_test(y_bad, v_bad, 0.05, lags=4)
    assert bad["p"] < 1e-6
