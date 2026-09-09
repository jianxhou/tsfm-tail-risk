"""DM (hand-computed + properties), MCS (synthetic), precision diagnostics."""

import numpy as np
import pytest

from backtests.dm import dm_test
from backtests.mcs import mcs
from precision.diagnostics import fragile_flag, n_alpha, precision_floor, sigma_tail


# ---- DM ----

def test_dm_hand_case():
    """h=1, no Harvey: stat = d̄ / sqrt(γ0/n) with γ0 the ML (1/n) variance."""
    d = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 4.0, 2.0, 3.0])
    la, lb = d, np.zeros_like(d)
    out = dm_test(la, lb, h=1, harvey_correction=False)
    n, dbar = len(d), d.mean()
    gamma0 = ((d - dbar) ** 2).mean()
    assert out["stat"] == pytest.approx(dbar / np.sqrt(gamma0 / n), rel=1e-12)


def test_dm_antisymmetric_and_harvey_shrinks(inp=None):
    rng = np.random.default_rng(3)
    la, lb = rng.random(200), rng.random(200)
    plain = dm_test(la, lb, harvey_correction=False)
    flipped = dm_test(lb, la, harvey_correction=False)
    assert plain["stat"] == pytest.approx(-flipped["stat"], rel=1e-12)
    harvey = dm_test(la, lb, harvey_correction=True)
    assert abs(harvey["stat"]) < abs(plain["stat"])  # adj < 1 for h=1


# ---- MCS ----

def test_mcs_separated_models_eliminated():
    rng = np.random.default_rng(11)
    T = 600
    base = rng.standard_normal(T)
    L = np.column_stack([
        base + rng.standard_normal(T) * 0.1,          # good
        base + rng.standard_normal(T) * 0.1,          # good (equivalent)
        base + rng.standard_normal(T) * 0.1 + 1.0,    # clearly worse
    ])
    out = mcs(L, names=["a", "b", "bad"], level=0.10, n_boot=2000)
    assert "bad" not in out["mcs"]
    assert set(out["mcs"]) == {"a", "b"}
    assert out["pvalues"]["bad"] < 0.10


def test_mcs_equivalent_models_survive_and_deterministic():
    rng = np.random.default_rng(5)
    L = rng.standard_normal((400, 3)) * 0.5 + 1.0
    r1 = mcs(L, level=0.10, n_boot=2000, seed=99)
    r2 = mcs(L, level=0.10, n_boot=2000, seed=99)
    assert r1 == r2
    assert len(r1["mcs"]) == 3  # iid equal-mean losses: nothing should fall


# ---- precision diagnostics (QuantLet HMD_ES conventions) ----

def test_n_alpha_and_floor():
    assert n_alpha(2600, 0.01) == pytest.approx(26.0)
    assert precision_floor(2.0, 25.0) == pytest.approx(0.4)


def test_sigma_tail_hand_case():
    y = np.array([-3.0, -5.0, 1.0, 2.0, -0.5])
    v = np.array([-2.0, -2.0, -2.0, -2.0, -2.0])
    # exceedance depths: v-y on hits -> [1, 3]; std(ddof=1) = sqrt(2)
    assert sigma_tail(y, v, min_hits=2) == pytest.approx(np.sqrt(2.0))
    assert np.isnan(sigma_tail(y, v, min_hits=3))


def test_fragile_flag_quantlet_convention():
    # bound = sqrt(c1²+c2²)/sqrt(nα); QuantLet example: n=250, α=0.025 → nα=6.25
    out = fragile_flag(diff=0.5, sigma_a=1.0, sigma_b=1.0, n_alpha_=6.25)
    assert out["bound"] == pytest.approx(np.sqrt(2.0) / 2.5)
    assert out["fragile"] is True          # 0.5 < 0.566
    out2 = fragile_flag(diff=0.6, sigma_a=1.0, sigma_b=1.0, n_alpha_=6.25)
    assert out2["fragile"] is False


def test_fixture_oracle_vs_hs_fragility(inp):
    """End-to-end: on the frozen input at α=1% the oracle-vs-HS FZ0 gap should be
    screened by the precision machinery without error (value locked by fixture)."""
    from backtests.scores import fz0
    from tests.conftest import series
    y, v, e = series(inp, "oracle", 0.01)
    y2, v2, e2 = series(inp, "hs", 0.01)
    n = min(len(y), len(y2))
    d = fz0(y[-n:], v[-n:], e[-n:], 0.01).mean() - fz0(y2[-n:], v2[-n:], e2[-n:], 0.01).mean()
    na = n_alpha(n, 0.01)
    fl = fragile_flag(d, sigma_tail(y[-n:], v[-n:]), sigma_tail(y2[-n:], v2[-n:]), na)
    assert np.isfinite(fl["bound"]) and fl["bound"] > 0
