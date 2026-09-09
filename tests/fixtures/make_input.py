"""Generate the frozen backtest fixture input (committed as input_series.csv).

One seeded GARCH(1,1)-t path plus two deterministic forecasters:
  oracle : true conditional VaR/ES from the simulating recursion
  hs250  : rolling 250-day historical simulation (empirical quantile / tail mean)

Every statistical function in backtests/ and precision/ is unit-tested against
reference outputs computed on THIS file by R (rugarch/GAS/esback) and by the
Pele QuantLet HMD_ES code — see make_r_references.R / make_quantlet_references.py.
Regenerating this file invalidates all reference CSVs; don't touch it casually.

Run:  python tests/fixtures/make_input.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260704
N_BURN = 500
N = 1250                      # ~5 trading years kept after burn-in
OMEGA, ALPHA_G, BETA_G = 0.05, 0.09, 0.90
NU = 8.0
ALPHAS = (0.01, 0.025, 0.05)
HS_WIN = 250

OUT = Path(__file__).parent / "input_series.csv"


def main() -> None:
    rng = np.random.default_rng(SEED)
    t_scale = np.sqrt((NU - 2.0) / NU)          # unit-variance standardized t
    z = rng.standard_t(NU, size=N_BURN + N) * t_scale

    sig2 = np.empty(N_BURN + N)
    y = np.empty(N_BURN + N)
    sig2[0] = OMEGA / (1.0 - ALPHA_G - BETA_G)
    y[0] = np.sqrt(sig2[0]) * z[0]
    for t in range(1, N_BURN + N):
        sig2[t] = OMEGA + ALPHA_G * y[t - 1] ** 2 + BETA_G * sig2[t - 1]
        y[t] = np.sqrt(sig2[t]) * z[t]

    # one-step-ahead conditional sigma for the kept sample
    sig_fc = np.sqrt(sig2[N_BURN:])
    y_kept = y[N_BURN:]

    df = pd.DataFrame({"t": np.arange(N), "y": y_kept})

    for a in ALPHAS:
        q_std = stats.t.ppf(a, NU) * t_scale                      # standardized t quantile
        es_std = (-stats.t.pdf(stats.t.ppf(a, NU), NU)
                  * (NU + stats.t.ppf(a, NU) ** 2) / ((NU - 1.0) * a)) * t_scale
        tag = f"{a:g}".replace("0.", "")
        df[f"v_oracle_{tag}"] = sig_fc * q_std
        df[f"e_oracle_{tag}"] = sig_fc * es_std

        v_hs = np.full(N, np.nan)
        e_hs = np.full(N, np.nan)
        for t in range(HS_WIN, N):
            w = y_kept[t - HS_WIN:t]
            q = np.quantile(w, a)
            tail = w[w <= q]
            v_hs[t] = q
            e_hs[t] = tail.mean() if len(tail) else q
        df[f"v_hs_{tag}"] = v_hs
        df[f"e_hs_{tag}"] = e_hs

    df.to_csv(OUT, index=False, float_format="%.10g")
    print(f"wrote {OUT} shape={df.shape} seed={SEED}")
    for a in ALPHAS:
        tag = f"{a:g}".replace("0.", "")
        hit = (y_kept <= df[f"v_oracle_{tag}"]).mean()
        print(f"  oracle hit rate @ {a:g}: {hit:.4f} (nominal {a:g})")


if __name__ == "__main__":
    main()
