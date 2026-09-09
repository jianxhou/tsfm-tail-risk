"""v2.1 Phase-1 signer re-check fixups (B2-i / B2-h / B2-d), regression-locked.

B2-h: every driver that reads the ctxfits cache directly must hit the same
stale-schema hard failure as ctx_fits (the re-check demonstrated the four
drivers bypassed the guard). B2-d: the per-cell `except Exception` skip
handlers in the two repair drivers must re-raise AssertionError, else the
trap-1 anchor/mass/schema asserts are swallowed into cell skips. B2-i: h2's
_fidelity takes tau_mass through its signature.

Caveat on record (Phase-1 report erratum): all of these are `assert`
statements per review #10 §2.4's prescribed form — under `python -O` they
are stripped; the project never runs the pipeline with -O.
"""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from erules.rules import e3_var_es_from_fit
from harness.run_grid_extract import load_ctxfits

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "harness"
DRIVERS = ("run_grid_compare.py", "run_mcs_levels.py",
           "run_grid_repairs.py", "run_repair_series.py")

NU = 5.0
T5_GRID = {round(0.1 * k, 1): float(stats.t.ppf(round(0.1 * k, 1), NU))
           for k in range(1, 10)} | {0.5: 0.0}


def _old_schema_parquet(tmp_path) -> Path:
    """The re-check's construction: a v2.0-era ctxfits parquet (no
    n_kept/tau_mass columns)."""
    p = tmp_path / "spx.parquet"
    pd.DataFrame({"t": [512, 513], "nu": [5.0, 5.1], "u": [-1.1, -1.2],
                  "xi": [0.2, 0.25], "beta": [0.8, 0.9]}).to_parquet(p)
    return p


def test_load_ctxfits_hard_fails_on_old_schema(tmp_path):
    old = _old_schema_parquet(tmp_path)
    with pytest.raises(AssertionError, match="stale v2.0 ctxfits schema"):
        load_ctxfits(old)
    # and passes on the v2.1 schema
    new = tmp_path / "new.parquet"
    pd.DataFrame({"t": [512], "nu": [5.0], "u": [-1.1], "xi": [0.2],
                  "beta": [0.8], "n_kept": [50], "tau_mass": [50 / 512]}
                 ).to_parquet(new)
    fits = load_ctxfits(new)
    assert fits.index.name == "t" and "tau_mass" in fits.columns


def test_all_four_drivers_route_ctxfits_reads_through_guard():
    """No driver may read a ctxfits parquet around the guard: every ctxfits
    read site in the four drivers uses load_ctxfits, and none constructs a
    bare pd.read_parquet on a ctxfits path. Combined with the behavior test
    above, an old-schema cache hard-fails ALL FOUR drivers."""
    for name in DRIVERS:
        src = (HARNESS / name).read_text()
        bare = [ln.strip() for ln in src.splitlines()
                if "read_parquet" in ln and "ctxfits" in ln.lower()
                and "load_ctxfits" not in ln]
        assert not bare, f"{name}: unguarded ctxfits read(s): {bare}"
        n_guarded = len(re.findall(r"load_ctxfits\(", src))   # call sites only
        assert n_guarded >= 1, f"{name}: no guarded ctxfits read found"


def _handlers_of_trys(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.handlers:
            yield node.handlers


def test_percell_exception_handlers_reraise_assertionerror():
    """B2-d structural proof: in both repair drivers, every handler catching
    (bare) Exception is preceded, in the same try, by an AssertionError
    handler whose body is a bare `raise`."""
    def htype(h):
        if h.type is None:
            return "BARE"
        return h.type.id if isinstance(h.type, ast.Name) else "OTHER"

    checked = 0
    for name in ("run_grid_repairs.py", "run_repair_series.py"):
        for handlers in _handlers_of_trys(HARNESS / name):
            names = [htype(h) for h in handlers]
            if "Exception" in names or "BARE" in names:
                i = names.index("Exception") if "Exception" in names else names.index("BARE")
                pre = names[:i]
                assert "AssertionError" in pre, (
                    f"{name}: except Exception without a preceding "
                    f"AssertionError re-raise")
                h = handlers[pre.index("AssertionError")]
                assert len(h.body) == 1 and isinstance(h.body[0], ast.Raise) \
                    and h.body[0].exc is None, f"{name}: AssertionError handler must bare-raise"
                checked += 1
    assert checked >= 2      # at least one per driver


def test_anchor_assert_penetrates_cell_skip_pattern():
    """B2-d behavioral proof, replicating the drivers' fixed handler shape:
    the trap-1 anchor assert escapes the per-cell skip."""
    no_anchor = {k: v for k, v in T5_GRID.items() if k != 0.1}
    with pytest.raises(AssertionError):
        try:
            e3_var_es_from_fit(no_anchor, 0.0, -1.5, 0.2, 0.8, 0.01,
                               tau_mass=0.0546875)
        except AssertionError:
            raise
        except Exception:          # the cell-skip catcher
            pytest.fail("cell-skip handler swallowed the anchor assert")


def test_h2_fidelity_signature_takes_tau_mass():
    """B2-i: _fidelity threads tau_mass (the Phase-1 landing referenced it
    without passing it — NameError on any gpd_ok window)."""
    import inspect

    from harness.run_mechanism_h2 import _fidelity
    params = list(inspect.signature(_fidelity).parameters)
    assert "tau_mass" in params
    assert params.index("tau_mass") == params.index("beta") + 1
    acc = []
    _fidelity(acc, "h", "g", 0, 512, T5_GRID, 1.0, 5.0, 5.0,
              0.0, -1.2, 0.2, 0.8, 52 / 512, True)
    assert len(acc) == 3 and all(np.isfinite(r["e3_v"]) for r in acc)
