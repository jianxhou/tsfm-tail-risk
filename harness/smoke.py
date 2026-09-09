"""Adapter smoke runner (Stage 1): short seeded series -> shape / monotone / finite.

    python -m harness.smoke [adapter ...]     # default: all four pilot adapters

Quantile-native pilot models are probed at deep-tail + central levels where the
official implementation accepts them; TimesFM-2.5 is probed on its native decile
grid (its head refuses non-decile levels by design — that fact IS the point).
Prints one result line per adapter; exits nonzero if any adapter fails.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

SEED = 20260704
CTX_LEN = 512

PROBES = {
    "chronos_bolt": ("harness.chronos_adapters", "ChronosBoltAdapter",
                     [0.01, 0.025, 0.05, 0.5, 0.95]),
    "chronos_2": ("harness.chronos_adapters", "Chronos2Adapter",
                  [0.01, 0.025, 0.05, 0.5, 0.95]),
    "timesfm_2_5": ("harness.timesfm_adapter", "TimesFM25Adapter",
                    [0.1, 0.2, 0.5, 0.8, 0.9]),
    "moirai_2_0": ("harness.moirai_adapter", "Moirai2Adapter",
                   [0.01, 0.025, 0.05, 0.5, 0.95]),
    # samples-native: value = n samples to draw
    "lag_llama": ("harness.lagllama_adapter", "LagLlamaAdapter", 200),
    # mechanism panel (samples-native; probed through predict_quantiles = the
    # S=1000 empirical extraction the grid uses)
    "chronos_base": ("harness.chronos_base_adapter", "ChronosBaseAdapter",
                     [0.01, 0.025, 0.05, 0.5, 0.95]),
    "moirai_1_1": ("harness.moirai11_adapter", "Moirai11Adapter",
                   [0.01, 0.025, 0.05, 0.5, 0.95]),
}


def run_one(key: str) -> bool:
    mod_name, cls_name, levels = PROBES[key]
    rng = np.random.default_rng(SEED)
    ctx = (rng.standard_t(5, size=CTX_LEN) * 1.2).astype(np.float32)

    t0 = time.time()
    try:
        mod = __import__(mod_name, fromlist=[cls_name])
        adapter = getattr(mod, cls_name)()
        if isinstance(levels, int):                       # samples-native probe
            n = levels
            s = adapter.predict_samples(ctx, n)
            dt = time.time() - t0
            assert s.shape == (n,), f"shape {s.shape} != ({n},)"
            assert np.all(np.isfinite(s)), "non-finite samples"
            emp = {a: float(np.quantile(s, a)) for a in (0.01, 0.5, 0.95)}
            print(f"[PASS] {key:<12} {dt:6.1f}s  n={n} "
                  + ", ".join(f"emp_q{a:g}={v:+.3f}" for a, v in emp.items()))
            return True
        q = adapter.predict_quantiles(ctx, levels)
        dt = time.time() - t0

        assert q.shape == (len(levels),), f"shape {q.shape} != ({len(levels)},)"
        assert np.all(np.isfinite(q)), f"non-finite values: {q}"
        assert np.all(np.diff(q) >= -1e-9), f"quantile crossing: {q}"
        pairs = ", ".join(f"q{lv:g}={v:+.3f}" for lv, v in zip(levels, q))
        print(f"[PASS] {key:<12} {dt:6.1f}s  {pairs}")
        return True
    except Exception as err:  # noqa: BLE001 - smoke reports every failure mode
        dt = time.time() - t0
        print(f"[FAIL] {key:<12} {dt:6.1f}s  {type(err).__name__}: {err}")
        traceback.print_exc(limit=3)
        return False


if __name__ == "__main__":
    keys = sys.argv[1:] or list(PROBES)
    ok = all([run_one(k) for k in keys])
    sys.exit(0 if ok else 1)
