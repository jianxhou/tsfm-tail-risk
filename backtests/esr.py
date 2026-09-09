"""Bayer–Dimitriadis (2022) ESR regression backtest — PENDING IMPLEMENTATION.

Status (2026-07-04): reference p-values for versions 1–3 are already frozen in
tests/fixtures/r_references.csv (esback 0.3.1 asymptotic, no bootstrap). A faithful
Python port needs the esreg joint (VaR, ES) M-estimator (Fissler–Ziegel loss,
two-step Nelder-Mead + misspecification-robust covariance with the nid/scl_sp
sigma estimator). Per hard rule 1 this function must not be used scientifically
until it is pytest-green against those fixtures; tests/test_esr.py is skipped with
this reason until then.
"""

from __future__ import annotations


def esr_backtest(y, v, e, alpha: float, version: int = 1) -> dict:
    raise NotImplementedError(
        "ESR backtest pending: port esreg estimator, verify vs r_references.csv "
        "(esback_esr1..3 rows) before any scientific use."
    )
