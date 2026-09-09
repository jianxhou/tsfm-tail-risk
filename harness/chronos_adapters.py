"""Adapters for the Chronos family (Bolt: bounded quantile grid; Chronos-2: current
flagship). Revisions pinned in registry.yaml — keep in sync.
"""

from __future__ import annotations

import numpy as np
import torch

from harness.base import ModelAdapter

BOLT_REPO, BOLT_REV = "amazon/chronos-bolt-base", "5d9f166d69f47aef3401367a7b842e78fe97b121"
C2_REPO, C2_REV = "amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"


class _ChronosQuantileAdapter(ModelAdapter):
    native = "quantiles"

    def __init__(self, repo: str, revision: str, device: str = "cpu"):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            repo, revision=revision, device_map=device, torch_dtype=torch.float32)

    @staticmethod
    def _shape(ctx: np.ndarray) -> torch.Tensor:
        return torch.tensor(ctx).unsqueeze(0)            # bolt: (batch, length)

    def predict_quantiles(self, ctx: np.ndarray, levels) -> np.ndarray:
        ctx = self._check_ctx(ctx)
        levels = self._check_levels(levels)
        q, _ = self.pipe.predict_quantiles(
            inputs=self._shape(ctx),
            prediction_length=1,
            quantile_levels=levels,
        )
        # bolt returns a (1,1,L) tensor; chronos-2 returns a list of (1,L) tensors
        q0 = q[0] if isinstance(q, list) else q
        arr = np.asarray(q0.detach().cpu() if hasattr(q0, "detach") else q0, dtype=float)
        return arr.reshape(-1)[-len(levels):]


class ChronosBoltAdapter(_ChronosQuantileAdapter):
    model_id = "chronos_bolt"

    def __init__(self, device: str = "cpu"):
        super().__init__(BOLT_REPO, BOLT_REV, device)


class Chronos2Adapter(_ChronosQuantileAdapter):
    model_id = "chronos_2"

    def __init__(self, device: str = "cpu"):
        super().__init__(C2_REPO, C2_REV, device)

    @staticmethod
    def _shape(ctx: np.ndarray) -> torch.Tensor:
        return torch.tensor(ctx).reshape(1, 1, -1)       # (n_series, n_variates, length)
