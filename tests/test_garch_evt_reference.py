"""A16 — external McNeil-Frey reference tests for the GARCH-EVT comparator.

Stage-5 closing verdict elevated A16 to a MANDATORY Stage-6 number-freeze
prerequisite: GARCH-EVT is the benchmark the hybrid repair H is measured
against, so its correctness is headline-grade and synthetic property tests are
insufficient for print. These tests cross-check both halves of the comparator
against independent external authorities:

  * the POT-GPD tail (erules.rules + fixes.arms._evt_from_residuals) vs
    evir 1.7.4 `riskmeasures` — A. McNeil's OWN package, i.e. the McNeil-Frey
    estimator itself;
  * the GARCH-t 1-step conditional-sigma filter (fixes.garch_filter) vs
    rugarch 1.5.5 (the reference financial GARCH engine) AND an independent
    GARCH(1,1) recursion.

Reference values are vendored CSVs (tests/fixtures/mcneilfrey_*_reference.csv),
generated once by make_mcneilfrey_references.R against frozen inputs
(make_mcneilfrey_input.py) — same vendor-the-output pattern as the rugarch /
esback / QuantLet fixtures. R is NOT required to run these tests.
"""
import numpy as np
import pandas as pd
import pytest

from erules.rules import fit_gpd, gpd_tail_quantile, gpd_tail_es
from fixes.arms import _evt_from_residuals
from fixes.garch_filter import garch_mu_sigma_path

FX = "tests/fixtures/"
TAU_U = 0.10


@pytest.fixture(scope="module")
def resid():
    return pd.read_csv(FX + "mcneilfrey_resid_input.csv")["z"].to_numpy()


@pytest.fixture(scope="module")
def evt_ref():
    return pd.read_csv(FX + "mcneilfrey_evt_reference.csv")


@pytest.fixture(scope="module")
def garch_ctx():
    return pd.read_csv(FX + "mcneilfrey_garch_input.csv")["logret"].to_numpy()


@pytest.fixture(scope="module")
def garch_ref():
    return pd.read_csv(FX + "mcneilfrey_garch_reference.csv").iloc[0]


def test_gpd_mle_matches_evir(resid, evt_ref):
    """Our scipy genpareto MLE (erules.fit_gpd) matches McNeil's evir gpd MLE
    on the identical exceedance sample (optimizer-level agreement)."""
    u = float(np.quantile(resid, TAU_U))
    xi, beta, _ = fit_gpd(u - resid[resid < u])
    assert abs(xi - evt_ref["xi_evir"][0]) < 2e-3, (xi, evt_ref["xi_evir"][0])
    assert abs(beta - evt_ref["beta_evir"][0]) < 2e-3, (beta, evt_ref["beta_evir"][0])


def test_pot_tail_formula_is_mcneilfrey_exact(resid, evt_ref):
    """DECISIVE identity: fed evir's OWN (xi, beta, u), our closed-form tail
    quantile/ES reproduce evir riskmeasures to machine precision — i.e. the
    formulas in erules.rules ARE the McNeil-Frey POT estimator, algebraically,
    independent of any MLE-optimizer difference."""
    u = float(np.quantile(resid, TAU_U))
    for _, r in evt_ref.iterrows():
        a, xi, beta = r["alpha"], r["xi_evir"], r["beta_evir"]
        v = gpd_tail_quantile(u, xi, beta, TAU_U, a)
        e = gpd_tail_es(u, xi, beta, TAU_U, a)
        assert abs(v - r["var_evir"]) < 1e-9, (a, v, r["var_evir"])
        assert abs(e - r["es_evir"]) < 1e-9, (a, e, r["es_evir"])


def test_evt_end_to_end_matches_evir(resid, evt_ref):
    """Full comparator tail path (_evt_from_residuals with m=0,s=1, i.e. the
    standardized VaR/ES the GARCH-EVT arm scales by sigma) vs evir end-to-end;
    residual gap is the MLE-optimizer difference only."""
    for _, r in evt_ref.iterrows():
        v, e = _evt_from_residuals(resid, 0.0, 1.0, float(r["alpha"]))
        assert abs(v - r["var_evir"]) < 2e-3, (r["alpha"], v, r["var_evir"])
        assert abs(e - r["es_evir"]) < 2e-3, (r["alpha"], e, r["es_evir"])


def test_garch_filter_1step_matches_rugarch(garch_ctx, garch_ref):
    """fixes.garch_filter 1-step mu/sigma vs rugarch on the same window — two
    independent GARCH(1,1)-t engines agree on the conditional vol to <1%."""
    mu, sig = garch_mu_sigma_path([garch_ctx])
    assert abs(mu[0] - garch_ref["mu1_rugarch"]) < 1e-2, (mu[0], garch_ref["mu1_rugarch"])
    rel = abs(sig[0] - garch_ref["sigma1_rugarch"]) / garch_ref["sigma1_rugarch"]
    assert rel < 1e-2, f"1-step sigma rel diff {rel:.4%}"


def test_garch_filter_extraction_matches_recursion(garch_ctx):
    """Isolates code-correctness of the forecast extraction from optimizer
    noise: garch_filter's 1-step sigma must equal the GARCH(1,1) recursion
    sigma^2 = omega + alpha*eps_T^2 + beta*sigma_T^2 built independently from
    the arch fit's own params (catches horizon / variance-vs-vol / indexing)."""
    from arch import arch_model
    res = arch_model(garch_ctx, vol="GARCH", p=1, q=1, dist="t",
                     mean="Constant", rescale=False).fit(disp="off", show_warning=False)
    eps_T = garch_ctx[-1] - res.params["mu"]
    sig2 = (res.params["omega"] + res.params["alpha[1]"] * eps_T**2
            + res.params["beta[1]"] * res.conditional_volatility[-1]**2)
    _, sig = garch_mu_sigma_path([garch_ctx])
    assert abs(sig[0] - np.sqrt(sig2)) < 1e-8, (sig[0], np.sqrt(sig2))
