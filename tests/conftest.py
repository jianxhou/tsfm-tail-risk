from pathlib import Path

import pandas as pd
import pytest

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def inp() -> pd.DataFrame:
    return pd.read_csv(FIX / "input_series.csv")


@pytest.fixture(scope="session")
def r_refs() -> pd.DataFrame:
    return pd.read_csv(FIX / "r_references.csv")


@pytest.fixture(scope="session")
def quantlet_refs() -> pd.DataFrame:
    return pd.read_csv(FIX / "quantlet_references.csv")


def series(inp: pd.DataFrame, fc: str, alpha: float):
    """(y, v, e) for one forecaster/alpha, NaN rows dropped (hs starts at t=250)."""
    tag = f"{alpha:g}".replace("0.", "")
    sub = inp[["y", f"v_{fc}_{tag}", f"e_{fc}_{tag}"]].dropna()
    return (sub["y"].to_numpy(), sub[f"v_{fc}_{tag}"].to_numpy(),
            sub[f"e_{fc}_{tag}"].to_numpy())


def ref_value(refs: pd.DataFrame, fc: str, alpha: float, test: str, quantity: str) -> float:
    row = refs[(refs.forecaster == fc) & (refs.alpha == alpha)
               & (refs.test == test) & (refs.quantity == quantity)]
    assert len(row) == 1, f"missing reference {fc}/{alpha}/{test}/{quantity}"
    return float(row.value.iloc[0])
