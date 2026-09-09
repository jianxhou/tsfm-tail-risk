"""Condition-1 acceptance: the persistent Lag-Llama server returns bit-identical
output to the subprocess-per-call path (so switching to it changes cost, never
numbers). Integration test — skips if the quarantined venv is absent."""

from pathlib import Path

import numpy as np
import pytest

VENV = Path(__file__).resolve().parent.parent / ".venv-lag-llama" / "bin" / "python"
pytestmark = pytest.mark.skipif(not VENV.exists(),
                                reason=".venv-lag-llama not built")


def test_server_matches_subprocess_bit_identical():
    from harness.lagllama_adapter import LagLlamaAdapter, LagLlamaServerAdapter
    rng = np.random.default_rng(20260706)
    ctxs = [(rng.standard_t(5, size=512) * 1.2).astype(np.float32) for _ in range(2)]
    levels = [0.01, 0.025, 0.05, 0.95]

    sub = LagLlamaAdapter(seed=123)
    sub_q = [sub.predict_quantiles(c, levels) for c in ctxs]
    with LagLlamaServerAdapter(seed=123) as srv:
        srv_q = [srv.predict_quantiles(c, levels) for c in ctxs]

    for a, b in zip(sub_q, srv_q):
        assert np.array_equal(a, b), f"server != subprocess: {a} vs {b}"


def test_server_survives_many_calls_one_load():
    """The point of the persistent worker: many windows, one process."""
    from harness.lagllama_adapter import LagLlamaServerAdapter
    rng = np.random.default_rng(1)
    with LagLlamaServerAdapter(seed=7) as srv:
        for _ in range(8):
            q = srv.predict_quantiles(
                (rng.standard_t(5, size=512) * 1.2).astype(np.float32),
                [0.01, 0.05, 0.5])
            assert q.shape == (3,) and np.all(np.diff(q) > 0)
        assert srv.proc.poll() is None   # still the same live process
