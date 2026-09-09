"""Frozen loss-matrix input for the MCS <-> R `MCS` package cross-validation.

Design (deterministic, seed 20260704): T=750 days, m=6 models, iid Gaussian
losses with mean separations chosen to be decisive at T=750 (A and B are
statistically equivalent, ~0.4 se apart; C-F sit 9+ se above), so both
implementations must return the superior set {A, B} at conventional levels
regardless of bootstrap RNG differences.

Run: python tests/fixtures/make_mcs_input.py  (rewrites mcs_input_losses.csv)
"""
from pathlib import Path

import numpy as np

SEED = 20260704
T = 750
MEANS = {"A": 1.000, "B": 1.002, "C": 1.05, "D": 1.08, "E": 1.15, "F": 1.25}
NOISE_SD = 0.10


def main() -> None:
    rng = np.random.default_rng(SEED)
    cols = {name: mu + NOISE_SD * rng.standard_normal(T)
            for name, mu in MEANS.items()}
    out = Path(__file__).parent / "mcs_input_losses.csv"
    header = ",".join(cols)
    rows = [",".join(f"{cols[n][t]:.10f}" for n in cols) for t in range(T)]
    out.write_text(header + "\n" + "\n".join(rows) + "\n")
    print(f"wrote {out} ({T} rows x {len(cols)} models)")


if __name__ == "__main__":
    main()
