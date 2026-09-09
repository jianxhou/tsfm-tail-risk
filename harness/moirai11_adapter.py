"""Adapter for Moirai-1.1-R-small (Salesforce) — the MIXTURE distribution head
(contrast partner for Moirai-2.0's quantile head; Moirai 2.0 dropped the mixture
for a quantile output, arXiv:2511.11698). Mechanism panel, proposal §3.1.

Extraction (signer rule: analytic quantiles preferred, else sample and record).
The official uni2ts 1.x path (`MoiraiForecast.create_predictor` -> gluonts
`SampleForecast`) exposes the mixture only through samples; there is no
analytic-quantile interface for the mixture on this path. We therefore draw
S=1000 samples and take empirical quantiles (D3 MC-error convention, identical
to how every sampling head is treated), and RECORD `analytic_available=False`.

patch_size="auto" is Moirai 1.x's own default selection (no free parameter is
chosen by us); num_samples=1000 per the registered sampling convention. Torch
RNG is seeded for reproducibility (protocol: all randomness seeded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from harness.base import ModelAdapter

REPO = "Salesforce/moirai-1.1-R-small"
REV = "0c24ab99db2c1a70ea2a0fc03bf113329772ac64"   # id-asserted via HF API 2026-07-04

ANALYTIC_AVAILABLE = False   # mixture exposed only through samples on the official path


class Moirai11Adapter(ModelAdapter):
    model_id = "moirai_1_1"
    native = "samples"

    def __init__(self, context_length: int = 512, num_samples: int = 1000,
                 patch_size="auto", seed: int = 20260706):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        torch.manual_seed(seed)
        self.context_length = context_length
        self.num_samples = num_samples
        self.model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(REPO, revision=REV),
            prediction_length=1,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            patch_size=patch_size,
            num_samples=num_samples,
        )
        self.predictor = self.model.create_predictor(batch_size=1)

    def _forecast(self, ctx: np.ndarray):
        from gluonts.dataset.common import ListDataset
        ds = ListDataset(
            [{"start": pd.Period("2000-01-03", freq="D"), "target": ctx}],
            freq="D",
        )
        return next(iter(self.predictor.predict(ds)))

    def predict_samples(self, ctx: np.ndarray, n: int) -> np.ndarray:
        ctx = self._check_ctx(ctx)
        s = np.asarray(self._forecast(ctx).samples, dtype=float).reshape(-1)
        if n <= len(s):
            return s[:n]
        raise ValueError(f"requested {n} samples > predictor num_samples {len(s)}")

    def predict_quantiles(self, ctx: np.ndarray, levels) -> np.ndarray:
        levels = self._check_levels(levels)
        s = self.predict_samples(ctx, self.num_samples)
        return np.quantile(s, levels)
