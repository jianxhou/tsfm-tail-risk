"""Generate FZ0 / Kupiec reference outputs from the Pele QuantLet HMD_ES code.

    python tests/fixtures/make_quantlet_references.py /path/to/HMD_ES

Imports fz_loss and kupiec_p from the cloned QuantLet/HMD_ES repo (their exact
implementations — proposal D1: our FZ0 must match this reference) and evaluates
them on the frozen input_series.csv. Output committed as quantlet_references.csv;
the HMD_ES commit used is recorded in the CSV.

Their FZ0 convention: V, E in return units and negative (loss tail), E < V < 0;
fz_loss(params=(q,r), ...) shifts V+q, E+r — with params=(0,0) it is the plain
mean FZ0 loss of the forecasts themselves.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ALPHAS = (0.01, 0.025, 0.05)


def main(repo: str) -> None:
    repo_path = Path(repo).resolve()
    sha = subprocess.run(["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    sys.path.insert(0, str(repo_path / "code"))
    from pipeline import fz_loss, kupiec_p  # noqa: E402  (QuantLet reference impl)

    inp = pd.read_csv(HERE / "input_series.csv")
    rows = []
    for fc in ("oracle", "hs"):
        for a in ALPHAS:
            tag = f"{a:g}".replace("0.", "")
            v = inp[f"v_{fc}_{tag}"].to_numpy()
            e = inp[f"e_{fc}_{tag}"].to_numpy()
            y = inp["y"].to_numpy()
            ok = ~np.isnan(v) & ~np.isnan(e)
            y, v, e = y[ok], v[ok], e[ok]

            rows.append(dict(forecaster=fc, alpha=a, quantity="fz0_mean",
                             value=fz_loss((0.0, 0.0), y, v, e, a), hmd_es_commit=sha))
            rows.append(dict(forecaster=fc, alpha=a, quantity="kupiec_p",
                             value=kupiec_p(y, v, a), hmd_es_commit=sha))

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "quantlet_references.csv", index=False, float_format="%.12g")
    print(f"wrote {len(out)} rows @ HMD_ES {sha[:12]}")
    print(out.drop(columns='hmd_es_commit').to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
