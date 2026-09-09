"""Canonical loader for the parquet store. ALL downstream consumers go through
load_series() — it is the enforcement point for data quarantine.

A series flagged `quarantined: true` in the manifest (set via universe.yaml,
propagated by build.py) HARD-FAILS on load until the flag is lifted after source
cross-verification. There is deliberately no bypass switch: fixing the data or
lifting the flag in universe.yaml (with a changelog note) is the only way through.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data_manifest.yaml"


class QuarantinedSeriesError(RuntimeError):
    pass


def manifest() -> dict:
    return yaml.safe_load(MANIFEST.read_text())


def load_series(sid: str) -> pd.DataFrame:
    """Load one series (date, close, logret|dbp) with its manifest entry attached
    as .attrs. Raises QuarantinedSeriesError for quarantined series and KeyError
    for unknown ids."""
    m = manifest()
    if sid not in m:
        raise KeyError(f"'{sid}' not in data_manifest.yaml")
    entry = m[sid]
    if entry.get("quarantined"):
        raise QuarantinedSeriesError(
            f"series '{sid}' is quarantined: {entry.get('quarantine_reason', '?')} "
            f"— fix the source and lift the flag in universe.yaml (changelog note "
            f"required) before use")
    df = pd.read_parquet(ROOT.parent / entry["parquet"])
    df.attrs.update(entry)
    return df
