"""Adapter for Moirai-2.0 (Salesforce, quantile output head — arXiv:2511.11698).
Runs through the gluonts predictor path of uni2ts 2.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness.base import ModelAdapter

MOIRAI_REPO = "Salesforce/moirai-2.0-R-small"
MOIRAI_REV = "30f43ff08c8494f4943ae1521e9d4e94a0fbb389"


class Moirai2Adapter(ModelAdapter):
    model_id = "moirai_2_0"
    native = "quantiles"

    def __init__(self, context_length: int = 1024):
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        self.context_length = context_length
        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(MOIRAI_REPO, revision=MOIRAI_REV),
            prediction_length=1,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self.predictor = self.model.create_predictor(batch_size=1)

    def _forecast(self, ctx: np.ndarray):
        from gluonts.dataset.common import ListDataset
        ds = ListDataset(
            [{"start": pd.Period("2000-01-03", freq="D"), "target": ctx}],
            freq="D",
        )
        return next(iter(self.predictor.predict(ds)))

    def predict_quantiles(self, ctx: np.ndarray, levels) -> np.ndarray:
        ctx = self._check_ctx(ctx)
        levels = self._check_levels(levels)
        fc = self._forecast(ctx)
        return np.array([float(np.asarray(fc.quantile(lv)).reshape(-1)[0])
                         for lv in levels])
