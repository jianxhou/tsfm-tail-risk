"""Adapter for Chronos-base (amazon/chronos-t5-base) — the original token-quantized
SAMPLING head (contrast partner for Chronos-Bolt's bounded quantile grid).

Mechanism panel, proposal §3.1. The registered extraction is S=1000 empirical
quantiles (D3 MC-error convention), matching how every sampling head is treated.

Faithful-but-fast sampling
--------------------------
For a one-step forecast, Chronos T5 samples num_samples iid draws from a SINGLE
categorical over the token-center values (`ChronosModel.forward` -> HF `generate`
with `min_new_tokens=1, do_sample=True, temperature, top_k, top_p`). The naive
`pipe.predict(num_samples=1000)` re-runs the 512-token encoder on a
1000-expanded batch, which is ~40-170 s/window and OOMs at S=1000 on CPU/MPS.

We instead run the encoder + one decoder step ONCE, reconstruct the exact
predictive categorical over token values (applying the model's own
temperature/top_k/top_p and the min_new_tokens EOS-suppression), and draw S=1000
from it. This is DISTRIBUTION-IDENTICAL to native `predict(num_samples=1000)`:
validated in tests/test_mechanism_adapters.py by the Kolmogorov (sup-CDF)
distance between the seeded native empirical CDF and this exact categorical:
KS sits at the S=1000 sampling floor (of order 0.02--0.03, well inside the
DKW threshold 0.07), at ~56x speed. The exact categorical (`categorical()`)
is also exposed for the H2
token-quantization analysis.
"""

from __future__ import annotations

import numpy as np
import torch

from harness.base import ModelAdapter

REPO = "amazon/chronos-t5-base"
REV = "ad294eaacead15db499b740ea4122266dd2a81a2"   # id-asserted via HF API 2026-07-04


class ChronosBaseAdapter(ModelAdapter):
    model_id = "chronos_base"
    native = "samples"

    def __init__(self, device: str = "cpu", seed: int = 20260706, n_threads: int = 4):
        from chronos import ChronosPipeline
        if n_threads:
            torch.set_num_threads(n_threads)
        self.pipe = ChronosPipeline.from_pretrained(
            REPO, revision=REV, device_map=device, torch_dtype=torch.float32)
        self._hf = self.pipe.model.model            # T5ForConditionalGeneration
        self._cfg = self.pipe.model.config          # ChronosConfig (temperature/top_k/top_p)
        self._tok = self.pipe.tokenizer             # MeanScaleUniformBins
        self._ds = self._hf.config.decoder_start_token_id
        self._rng = np.random.default_rng(seed)

    # -- exact single-step predictive categorical over token-center VALUES --------
    # v2.0 P0-5 refactor: the single forward (_step_logits) is separated from the
    # decoding configuration (decode_probs) so the SAME logits can be decoded
    # under {vendor default top_k=50, larger k, untruncated} post hoc.
    # categorical() composes the two with config defaults and is result-identical
    # to the pre-v2.0 implementation (faithfulness gate unchanged).

    def _step_logits(self, ctx: np.ndarray):
        """One encoder + one decoder step; returns (values, logits) with the
        min_new_tokens=1 EOS suppression applied, BEFORE temperature/top_k/top_p.
        Device-safe (inputs moved to the model device, logits returned on CPU)."""
        ctx = self._check_ctx(ctx)
        c = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        tids, amask, scale = self._tok.context_input_transform(c)
        dev = next(self._hf.parameters()).device
        with torch.no_grad():
            out = self._hf(input_ids=tids.to(dev), attention_mask=amask.to(dev),
                           decoder_input_ids=torch.tensor([[self._ds]], device=dev))
        logits = out.logits[0, -1, :].float().cpu().clone()
        logits[self._cfg.eos_token_id] = -float("inf")      # min_new_tokens=1
        ids = torch.arange(self._cfg.n_tokens)
        vals = self._tok.output_transform(ids.reshape(1, 1, -1), scale).reshape(-1)
        return vals.numpy().astype(float), logits

    def decode_probs(self, logits: "torch.Tensor", temperature: float | None = None,
                     top_k="config", top_p="config") -> np.ndarray:
        """Apply a decoding configuration to one-step logits -> probs (numpy).
        Defaults ('config') reproduce the pinned vendor configuration exactly;
        top_k=None disables truncation (untruncated categorical)."""
        lg = logits.clone()
        temperature = self._cfg.temperature if temperature is None else temperature
        top_k = self._cfg.top_k if isinstance(top_k, str) else top_k
        top_p = self._cfg.top_p if isinstance(top_p, str) else top_p
        lg = lg / temperature
        if top_k:
            kth = torch.topk(lg, top_k).values[-1]
            lg[lg < kth] = -float("inf")
        if top_p is not None and top_p < 1.0:
            sp, si = torch.sort(lg, descending=True)
            cum = torch.softmax(sp, dim=-1).cumsum(dim=-1)
            drop = cum - torch.softmax(sp, dim=-1) > top_p
            sp[drop] = -float("inf")
            lg = torch.full_like(lg, -float("inf")).scatter(0, si, sp)
        return torch.softmax(lg, dim=-1).numpy().astype(float)

    def categorical(self, ctx: np.ndarray):
        """Return (values, probs): the exact one-step predictive distribution the
        native sampler draws from, after the model's temperature/top_k/top_p and
        the min_new_tokens=1 EOS suppression. `values` are the token centers
        rescaled by the per-series mean-abs scale; `probs` sum to 1."""
        vals, logits = self._step_logits(ctx)
        return vals, self.decode_probs(logits)

    def predict_samples(self, ctx: np.ndarray, n: int) -> np.ndarray:
        vals, probs = self.categorical(ctx)
        probs = probs / probs.sum()             # exact-1 for rng.choice (float32 drift)
        idx = self._rng.choice(len(probs), size=int(n), replace=True, p=probs)
        return vals[idx]

    def predict_quantiles(self, ctx: np.ndarray, levels, n_samples: int = 1000) -> np.ndarray:
        """S=1000 empirical quantiles (registered sampling-head convention, D3)."""
        levels = self._check_levels(levels)
        s = self.predict_samples(ctx, n_samples)
        return np.quantile(s, levels)
