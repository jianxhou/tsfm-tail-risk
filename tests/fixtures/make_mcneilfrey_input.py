"""A16 fixture-input generator (frozen once, vendored).

Produces the two frozen inputs the external McNeil-Frey cross-check runs on:
  - mcneilfrey_garch_input.csv : a fixed 1000-obs SPX logret window (the GARCH
    filter's context) — input to fixes.garch_filter, cross-checked vs rugarch.
  - mcneilfrey_resid_input.csv : the 1000 in-sample GARCH-t standardized
    residuals from that window (the sample McNeil-Frey's POT-GPD step operates
    on) — input to the EVT tail, cross-checked vs evir (McNeil's own package).

Window: SPX logret x100 rows [4000:5000] (deterministic slice; dates ~2015-11
..2019-11). n=1000 => the 10% POT threshold has EXACTLY 100 strict exceedances
(rate 0.10), so our tau_u=0.10 convention aligns bit-for-bit with evir's
empirical exceedance rate lambda=n.exceed/n. Asserted below.

Run once; the CSVs are the vendored fixture. arch MLE is deterministic (no RNG),
so this is reproducible, but the frozen CSVs are the source of truth for tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

SPX = "data/parquet/spx.parquet"
LO, HI = 4000, 5000

def main() -> None:
    df = pd.read_parquet(SPX)
    win = df.iloc[LO:HI].reset_index(drop=True)
    ctx = win["logret"].to_numpy(dtype=float)
    assert len(ctx) == 1000, len(ctx)

    # GARCH(1,1)-t, constant mean, rescale=False — IDENTICAL spec to
    # fixes.garch_filter.garch_mu_sigma_path / run_grid_baselines.garch_family.
    res = arch_model(ctx, vol="GARCH", p=1, q=1, dist="t",
                     mean="Constant", rescale=False).fit(disp="off", show_warning=False)
    z = (ctx - res.params["mu"]) / res.conditional_volatility  # in-sample std resid

    # rate check: exactly 100 strict exceedances below the 10% quantile
    u = float(np.quantile(z, 0.10))
    n_exc = int(np.sum(z < u))
    assert n_exc == 100, f"expected 100 exceedances, got {n_exc} (retune window)"
    assert len(np.unique(z)) == len(z), "ties in z would break the rate alignment"

    pd.DataFrame({"date": win["date"].astype(str), "logret": ctx}).to_csv(
        "tests/fixtures/mcneilfrey_garch_input.csv", index=False)
    pd.DataFrame({"z": z}).to_csv(
        "tests/fixtures/mcneilfrey_resid_input.csv", index=False)

    print(f"wrote inputs. n={len(ctx)}  u(10%)={u:.6f}  n_exc={n_exc}")
    print(f"arch params: mu={res.params['mu']:.6f} omega={res.params['omega']:.6f} "
          f"alpha={res.params['alpha[1]']:.6f} beta={res.params['beta[1]']:.6f} "
          f"nu={res.params['nu']:.6f}")
    print(f"1-step sigma forecast = "
          f"{float(np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1,0])):.6f}")

if __name__ == "__main__":
    main()
