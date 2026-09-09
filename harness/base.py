"""Unified adapter interface for TSFM models (Stage 1 checklist).

Contract (one-step-ahead, h=1, per proposal scope):
    predict_quantiles(ctx, levels) -> np.ndarray shape (len(levels),)
    predict_samples(ctx, n)        -> np.ndarray shape (n,)

`ctx` is a 1D float array of past values (log-returns ×100), most recent last.
A quantile-native model may not implement predict_samples and vice versa —
adapters expose what the model natively provides (NotImplementedError otherwise);
E-rule machinery (erules/) decides how tails are extracted on top. Adapters must
never silently substitute one representation for the other; whether a model's
native interface honors deep-tail levels (E0 availability) is a pilot-D2 fact to
freeze, not something to paper over here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ModelAdapter(ABC):
    model_id: str = "?"        # registry key in harness/registry.yaml
    native: str = "?"          # 'quantiles' | 'samples'

    @abstractmethod
    def predict_quantiles(self, ctx: np.ndarray, levels) -> np.ndarray: ...

    def predict_samples(self, ctx: np.ndarray, n: int) -> np.ndarray:
        raise NotImplementedError(f"{self.model_id} is {self.native}-native")

    @staticmethod
    def _check_ctx(ctx: np.ndarray) -> np.ndarray:
        ctx = np.asarray(ctx, dtype=np.float32)
        if ctx.ndim != 1 or len(ctx) < 16 or not np.all(np.isfinite(ctx)):
            raise ValueError("ctx must be a 1D finite array with >=16 points")
        return ctx

    @staticmethod
    def _check_levels(levels) -> list[float]:
        levels = [float(x) for x in levels]
        if not levels or any(not (0.0 < x < 1.0) for x in levels):
            raise ValueError("levels must be in (0,1)")
        if levels != sorted(levels):
            raise ValueError("levels must be ascending")
        return levels
