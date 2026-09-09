"""Adapter for TimesFM-2.5 (200M, torch build from the google-research/timesfm
GitHub package @4a6c5cda; PyPI 2.0.2 cannot load the 2.5 checkpoint).
"""

from __future__ import annotations

import numpy as np

from harness.base import ModelAdapter

TFM_REPO = "google/timesfm-2.5-200m-pytorch"
TFM_REV = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"


class TimesFM25Adapter(ModelAdapter):
    model_id = "timesfm_2_5"
    native = "quantiles"

    def __init__(self):
        import timesfm
        self._timesfm = timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            TFM_REPO, revision=TFM_REV)
        self._compiled_len: int | None = None
        # head order: [mean, q0.1, q0.2, ..., q0.9]
        self._head_levels = [round(0.1 * k, 1) for k in range(1, 10)]

    def _compile_for(self, n: int) -> None:
        """Compile with max_context == len(ctx). When the input is SHORTER than the
        compiled max_context, the 2.5 torch padding path returns all-NaN forecasts
        (observed with torch 2.4.1, timesfm @4a6c5cda — see CHANGELOG 2026-07-04),
        so padding is never allowed through this adapter."""
        if self._compiled_len != n:
            self.model.compile(self._timesfm.ForecastConfig(
                max_context=n,
                max_horizon=1,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
            ))
            self._compiled_len = n

    def predict_quantiles(self, ctx: np.ndarray, levels) -> np.ndarray:
        """Returns ONLY what the official head emits (deciles 0.1–0.9). No
        interpolation here — whether deep-tail levels are natively available
        (E0) is a pilot-D2 fact; extraction beyond the grid is erules/ work."""
        ctx = self._check_ctx(ctx)
        levels = self._check_levels(levels)
        missing = [lv for lv in levels
                   if not any(abs(lv - h) < 1e-9 for h in self._head_levels)]
        if missing:
            raise NotImplementedError(
                f"timesfm-2.5 native head exposes deciles {self._head_levels}; "
                f"requested non-native levels {missing} need an E-rule")
        self._compile_for(len(ctx))
        _, quantiles = self.model.forecast(horizon=1, inputs=[ctx])
        if not np.all(np.isfinite(np.asarray(quantiles))):
            raise RuntimeError("timesfm-2.5 returned non-finite forecast")
        qrow = np.asarray(quantiles)[0, 0, :]        # (1+9,) mean + 9 deciles
        grid = {h: float(qrow[1 + i]) for i, h in enumerate(self._head_levels)}
        return np.array([grid[min(self._head_levels, key=lambda h: abs(h - lv))]
                         for lv in levels])
