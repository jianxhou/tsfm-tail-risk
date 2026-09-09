"""Hansen–Lunde–Nason (2011) Model Confidence Set with the T_max statistic
and stationary bootstrap (Politis–Romano) — seeded, deterministic given inputs.

Cross-validated against the R `MCS` package (Bernardi & Catania, MCS 0.2.0)
2026-07-21 on the frozen fixture input: superior sets identical at levels
0.10 and 0.25 (tests/test_mcs_r_reference.py; fixtures
tests/fixtures/mcs_input_losses.csv + mcs_r_reference.csv). This closes the
stage-1 registered TODO (CHANGELOG 2026-07-04).
"""

from __future__ import annotations

import numpy as np


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) index matrix; geometric block lengths, circular wrap."""
    p = 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_boot, n))
    cont = rng.random(size=(n_boot, n)) >= p       # continue previous block?
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(cont[:, t], (idx[:, t - 1] + 1) % n, starts[:, t])
    return idx


def mcs(losses: np.ndarray, names: list[str] | None = None, level: float = 0.10,
        n_boot: int = 5000, mean_block: float = 10.0, seed: int = 20260704) -> dict:
    """MCS over a (T, m) loss matrix (lower loss = better).

    Iterative T_max procedure: at each step compute t_i = sqrt(T)·d̄_i· / se_i where
    d̄_i· is model i's average loss differential vs the current-set average; the
    equivalence-test p-value comes from the recentered bootstrap distribution of
    max_i t_i; while it falls below `level`, eliminate argmax t_i. MCS p-values are
    the running maximum of elimination p-values (Hansen et al. 2011, eq. 9).
    """
    L = np.asarray(losses, dtype=float)
    T, m = L.shape
    names = list(names) if names is not None else [f"m{i}" for i in range(m)]
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(T, n_boot, mean_block, rng)

    active = list(range(m))
    pvals: dict[str, float] = {}
    p_running = 0.0
    elim_order: list[str] = []

    while len(active) > 1:
        La = L[:, active]                                   # (T, k)
        dbar_i = La.mean(axis=0) - La.mean()                # d̄_i· vs set average
        # bootstrap the same quantity
        Lb = La[idx]                                        # (B, T, k)
        mb = Lb.mean(axis=1)                                # (B, k)
        db = mb - mb.mean(axis=1, keepdims=True)
        var_i = ((db - dbar_i) ** 2).mean(axis=0)           # bootstrap variance of d̄_i·
        se = np.sqrt(np.maximum(var_i, 1e-300))
        t_i = dbar_i / se
        t_boot = (db - dbar_i) / se                         # recentered
        tmax_boot = t_boot.max(axis=1)
        p_eq = float((tmax_boot >= t_i.max()).mean())

        p_running = max(p_running, p_eq)
        worst = int(np.argmax(t_i))
        if p_eq >= level:
            for j in active:
                pvals[names[j]] = max(pvals.get(names[j], 0.0), p_running)
            break
        pvals[names[active[worst]]] = p_running
        elim_order.append(names[active[worst]])
        active.pop(worst)

    if len(active) == 1:
        pvals[names[active[0]]] = 1.0
    survivors = [names[j] for j in active]
    return {"mcs": survivors, "pvalues": pvals, "eliminated": elim_order,
            "level": level}
