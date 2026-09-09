"""GARCH-t conditional sigma path (fit_len=1000, refit 21, params held between
refits) — the filter shared by the GARCH-EVT comparator and the hybrid arm H.
Same estimation caliber (A1) as run_grid_baselines.garch_family.
"""

from __future__ import annotations

import warnings

import numpy as np


def garch_mu_sigma_path(ctx_windows, refit_every: int = 21):
    """Given the sequence of fit-length context windows (one per OOS target,
    most-recent-last), return (mu, sigma): the constant mean and one-step-ahead
    conditional sigma at each target. Params re-estimated every `refit_every`
    windows, held between (A1 caliber, matches garch_family)."""
    from arch import arch_model
    warnings.filterwarnings("ignore")
    params = None
    mu = np.empty(len(ctx_windows)); sig = np.empty(len(ctx_windows))
    for i, ctx in enumerate(ctx_windows):
        if i % refit_every == 0 or params is None:
            params = arch_model(ctx, vol="GARCH", p=1, q=1, dist="t",
                                mean="Constant", rescale=False
                                ).fit(disp="off", show_warning=False).params
        fixed = arch_model(ctx, vol="GARCH", p=1, q=1, dist="t",
                           mean="Constant", rescale=False).fix(params)
        mu[i] = float(params["mu"])
        sig[i] = float(np.sqrt(fixed.forecast(horizon=1, reindex=False).variance.values[-1, 0]))
    return mu, sig


def garch_sigma_path(ctx_windows, refit_every: int = 21):
    return garch_mu_sigma_path(ctx_windows, refit_every)[1]
