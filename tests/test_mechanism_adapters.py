"""Mechanism-panel adapter tests (proposal §3.1 sub-panel).

Integration tests: they load the pinned Chronos-base and Moirai-1.1 checkpoints,
so they are slow and network-dependent (skipped if the model stack / weights are
unavailable). The logic plus optional model-stack faithfulness checks are
exercised by the faithfulness test, which is the scientific gate for the
Chronos-base fast sampler.
"""

from __future__ import annotations

import numpy as np
import pytest

SEED = 20260706
CTX_LEN = 512


def _ctx():
    rng = np.random.default_rng(SEED)
    return (rng.standard_t(5, size=CTX_LEN) * 1.2).astype(np.float32)


def _load(mod_name, cls_name):
    try:
        mod = __import__(mod_name, fromlist=[cls_name])
        return getattr(mod, cls_name)()
    except Exception as e:                       # missing weights / deps
        pytest.skip(f"{cls_name} unavailable: {type(e).__name__}: {str(e)[:80]}")


@pytest.mark.slow
def test_chronos_base_shape_monotone_finite():
    a = _load("harness.chronos_base_adapter", "ChronosBaseAdapter")
    levels = [0.01, 0.025, 0.05, 0.5, 0.95]
    q = a.predict_quantiles(_ctx(), levels)
    assert q.shape == (len(levels),)
    assert np.all(np.isfinite(q))
    assert np.all(np.diff(q) >= -1e-9), f"quantile crossing: {q}"
    s = a.predict_samples(_ctx(), 1000)
    assert s.shape == (1000,) and np.all(np.isfinite(s))


@pytest.mark.slow
def test_moirai_1_1_shape_monotone_finite():
    a = _load("harness.moirai11_adapter", "Moirai11Adapter")
    levels = [0.01, 0.025, 0.05, 0.5, 0.95]
    q = a.predict_quantiles(_ctx(), levels)
    assert q.shape == (len(levels),)
    assert np.all(np.isfinite(q))
    assert np.all(np.diff(q) >= -1e-9), f"quantile crossing: {q}"


@pytest.mark.slow
def test_chronos_base_fast_sampler_faithful_to_native():
    """Scientific gate (distribution-level): the fast categorical path must
    reproduce the FULL distribution the native chronos `predict()` samples from,
    not just a tail quantile. On real SPX windows we compare the native SEEDED
    empirical CDF (S draws) against the exact reconstructed categorical CDF via
    the Kolmogorov (sup-CDF) distance.

    Threshold from the finite-sample floor: under the null (identical laws) the
    DKW inequality bounds P(KS > eps) <= 2 exp(-2 S eps^2); at S=1000, eps=0.07
    that tail is ~1e-4, whereas a genuine reconstruction bias produces a
    persistent KS far above it. This replaces the earlier single-quantile check,
    whose 1-sigma tolerance sat AT the S=1000 MC floor (~0.05 return points at
    1%) and was therefore flaky. Native draws are seeded so the gate is
    reproducible run-to-run.
    """
    from pathlib import Path
    if not (Path(__file__).resolve().parents[1] / "data/parquet/spx.parquet").is_file():
        pytest.skip("requires the rebuilt raw-data layer (data/parquet/spx.parquet); "
                    "see REPRODUCING.md — not distributed with the public copy")

    import torch

    from data.load import load_series
    a = _load("harness.chronos_base_adapter", "ChronosBaseAdapter")
    x = load_series("spx")["logret"].to_numpy()

    S, CHUNK, EPS = 1000, 200, 0.07

    def native_seeded(ctx, seed):
        torch.manual_seed(seed)                 # deterministic native draws
        parts = []
        with torch.no_grad():
            for _ in range(S // CHUNK):
                s = a.pipe.predict(inputs=torch.tensor(ctx).unsqueeze(0),
                                   prediction_length=1, num_samples=CHUNK)
                parts.append(np.asarray(s).reshape(-1))
        return np.sort(np.concatenate(parts))

    for t in (600, 2500, 3500):
        ctx = x[t - CTX_LEN:t].astype(np.float32)
        vals, probs = a.categorical(ctx)
        order = np.argsort(vals)
        sv, cdf = vals[order], np.cumsum(probs[order])       # exact categorical CDF
        nat = native_seeded(ctx, SEED + t)
        femp = np.searchsorted(nat, sv, side="right") / nat.size
        ks = float(np.max(np.abs(femp - cdf)))               # Kolmogorov distance
        print(f"[faithfulness] t={t} KS={ks:.4f} (S={S}, EPS={EPS})")
        assert ks < EPS, (
            f"t={t}: native-vs-exact KS={ks:.4f} exceeds the DKW floor EPS={EPS} "
            f"(S={S}) — the fast categorical reconstruction is not faithful")
