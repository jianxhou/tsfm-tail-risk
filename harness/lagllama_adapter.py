"""Adapter for Lag-Llama (Student-t parametric head, sample-native).

Runs via subprocess in the quarantined .venv-lag-llama (its deps downgrade
uni2ts's — CHANGELOG 2026-07-04). Loads the checkpoint per call (~2s); a
persistent-worker mode is a Stage 4 TODO if the full grid needs it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from harness.base import ModelAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PY = REPO_ROOT / ".venv-lag-llama" / "bin" / "python"
WORKER = Path(__file__).resolve().parent / "lagllama_worker.py"
SERVER = Path(__file__).resolve().parent / "lagllama_server.py"


class LagLlamaAdapter(ModelAdapter):
    model_id = "lag_llama"
    native = "samples"

    def __init__(self, seed: int = 20260704, timeout_s: int = 300):
        if not VENV_PY.exists():
            raise RuntimeError(
                f"{VENV_PY} missing — rebuild via envs/lag-llama-requirements.txt")
        self.seed = seed
        self.timeout_s = timeout_s

    def _call(self, payload: dict) -> dict:
        proc = subprocess.run(
            [str(VENV_PY), str(WORKER)],
            input=json.dumps(payload).encode(), capture_output=True,
            timeout=self.timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"lag-llama worker failed:\n{proc.stderr.decode()[-800:]}")
        return json.loads(proc.stdout.decode())

    def predict_quantiles(self, ctx: np.ndarray, levels) -> np.ndarray:
        """ANALYTIC parametric-head quantiles: the worker captures the Student-t
        (df, loc, scale) that the official forward pass produces and inverts it in
        closed form — this IS the model's native distribution, no sampling."""
        ctx = self._check_ctx(ctx)
        levels = self._check_levels(levels)
        out = self._call({"mode": "params", "ctx": ctx.tolist(),
                          "levels": levels, "seed": self.seed})
        q = np.asarray(out["quantiles"], dtype=float)
        if q.shape != (len(levels),):
            raise RuntimeError(f"worker returned shape {q.shape}")
        return q

    def predict_samples(self, ctx: np.ndarray, n: int) -> np.ndarray:
        ctx = self._check_ctx(ctx)
        if n < 1:
            raise ValueError("n must be >= 1")
        out = self._call({"mode": "samples", "ctx": ctx.tolist(),
                          "n": n, "seed": self.seed})
        samples = np.asarray(out["samples"], dtype=float)
        if samples.shape != (n,):
            raise RuntimeError(f"worker returned shape {samples.shape}, wanted ({n},)")
        return samples


class LagLlamaServerAdapter(ModelAdapter):
    """Persistent-worker adapter (gate condition 1): one checkpoint load serves
    all windows over a streaming pipe. Use as a context manager for the full-grid
    run; falls back to the same JSON protocol as the subprocess path so results
    are bit-identical (verified in tests/test_lagllama_server.py)."""

    model_id = "lag_llama"
    native = "samples"

    def __init__(self, seed: int = 20260704, ready_timeout_s: int = 120):
        if not VENV_PY.exists():
            raise RuntimeError(f"{VENV_PY} missing — rebuild venv")
        self.seed = seed
        self.proc = subprocess.Popen(
            [str(VENV_PY), str(SERVER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=1, text=True)
        # block until the checkpoint is loaded and a ping round-trips
        self._await_ready(ready_timeout_s)

    def _await_ready(self, timeout_s: int) -> None:
        import select
        self.proc.stdin.write(json.dumps({"mode": "ping"}) + "\n")
        self.proc.stdin.flush()
        r, _, _ = select.select([self.proc.stdout], [], [], timeout_s)
        if not r:
            self.close()
            raise RuntimeError("lagllama_server did not become ready in time")
        resp = json.loads(self.proc.stdout.readline())
        if not resp.get("pong"):
            raise RuntimeError(f"unexpected handshake: {resp}")

    def _rpc(self, payload: dict) -> dict:
        if self.proc.poll() is not None:
            raise RuntimeError(f"lagllama_server died (code {self.proc.returncode}): "
                               f"{self.proc.stderr.read()[-500:]}")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def predict_quantiles(self, ctx: np.ndarray, levels) -> np.ndarray:
        ctx = self._check_ctx(ctx)
        levels = self._check_levels(levels)
        out = self._rpc({"mode": "params", "ctx": ctx.tolist(),
                         "levels": levels, "seed": self.seed})
        q = np.asarray(out["quantiles"], dtype=float)
        if q.shape != (len(levels),):
            raise RuntimeError(f"server returned shape {q.shape}")
        return q

    def predict_samples(self, ctx: np.ndarray, n: int) -> np.ndarray:
        ctx = self._check_ctx(ctx)
        out = self._rpc({"mode": "samples", "ctx": ctx.tolist(),
                         "n": int(n), "seed": self.seed})
        return np.asarray(out["samples"], dtype=float)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
