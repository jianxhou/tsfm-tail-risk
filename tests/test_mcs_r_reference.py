"""MCS cross-validation against the R `MCS` package (Bernardi & Catania).

Closes the stage-1 registered TODO (backtests/mcs.py, CHANGELOG 2026-07-04):
on the frozen loss input (tests/fixtures/mcs_input_losses.csv, seed 20260704,
decisive design: A ~ B ~ 0.4 se apart, C-F 9+ se worse), the Python
implementation's superior set must equal the R package's superior set at both
test levels. Membership is the cross-implementation invariant; bootstrap
p-values differ by RNG stream and are not compared numerically. The R side is
frozen in tests/fixtures/mcs_r_reference.csv (MCS 0.2.0, B=5000, Tmax, k=10;
regenerate with tests/fixtures/make_mcs_r_reference.R).
"""
import csv
from pathlib import Path

import numpy as np
import pytest

from backtests.mcs import mcs

FIX = Path(__file__).parent / "fixtures"


def _load():
    rows = list(csv.reader(open(FIX / "mcs_input_losses.csv")))
    names, data = rows[0], np.array(rows[1:], dtype=float)
    ref = list(csv.DictReader(open(FIX / "mcs_r_reference.csv")))
    return names, data, ref


@pytest.mark.parametrize("level", [0.10, 0.25])
def test_mcs_superior_set_matches_r_package(level):
    names, data, ref = _load()
    out = mcs(data, names=names, level=level, n_boot=5000)
    r_superior = sorted(r["model"] for r in ref
                        if float(r["alpha"]) == level and r["in_superior"] == "1")
    assert sorted(out["mcs"]) == r_superior
    # the fixture was designed to be decisive; guard against a degenerate
    # reference (e.g., everything surviving) silently weakening the test
    assert r_superior == ["A", "B"]


def test_r_reference_is_decisive():
    _, _, ref = _load()
    ps = {(r["model"], r["alpha"]): float(r["mcs_p"]) for r in ref}
    for (m, a), p in ps.items():
        assert p <= 0.001 or p >= 0.49, (m, a, p)  # no borderline rows
