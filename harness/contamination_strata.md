# Contamination strata: post-cutoff power consequences (D2, 2026-07-04)

Machine-computed from `data/parquet/` (rows strictly after each model's registered
cutoff/upper bound, pilot assets spx+btc+nvda; regenerate with the one-liner in git
history of this file). Cutoff sources: `harness/registry.yaml`.

| model | cutoff (type) | n spx | n btc | n nvda | n pooled | nα@1% | nα@2.5% | nα@5% |
|---|---|---|---|---|---|---|---|---|
| timesfm_2_5 | 2023-11-30 (DISCLOSED) | 647 | 947 | 647 | 2241 | 22.4 | 56.0 | 112.1 |
| lag_llama | 2023-10-31 (upper bound) | 668 | 977 | 668 | 2313 | 23.1 | 57.8 | 115.7 |
| chronos_2 | 2025-10-31 (upper bound) | 166 | 246 | 166 | 578 | 5.8 | 14.5 | 28.9 |
| moirai_2_0 | 2025-11-30 (upper bound) | 147 | 216 | 147 | 510 | 5.1 | 12.8 | 25.5 |
| chronos_bolt | 2025-11-21 (upper bound, revision date) | 151 | 225 | 151 | 527 | 5.3 | 13.2 | 26.4 |

## Fixed protocol for pilot memo GO#4 (per user, D2)

- Consistency check runs at **α = 5%, three assets POOLED, direction test only**
  (does the post-cutoff slice agree in sign with the full-sample conclusion).
- **α = 1% on the ~8-month strata (chronos_bolt / chronos_2 / moirai_2_0) is
  inherently precision-fragile**: pooled nα ≈ 5–6 sits at the Pele & M-M-P floor
  where no substantive claim is allowed — recorded ex ante, no 1% claims will be
  made on these strata.
- **TimesFM-2.5 is the anchor model for contamination analysis**: the only
  DISCLOSED calendar cutoff (2023-11) with a usable stratum (pooled nα@5% ≈ 112,
  @2.5% ≈ 56). Per-model dual reporting hangs off this anchor.

## Corpus overlap audit — conclusions (docs/giftevalpretrain_overlap.md)

Scope caveat: audit covers DISCLOSED corpora only; "no disclosed overlap" is the
strongest supportable statement. Chronos-family TSMixup augmentation (and
KernelSynth) is an undisclosed component on top of listed datasets.

- GiftEvalPretrain (152 components): only financial component = Monash
  `bitcoin_with_missing` (ends ~2021-07) → **BTC has disclosed overlap** for
  chronos_2 / moirai_2_0 / timesfm_2_5.
- chronos_bolt corpus (autogluon/chronos_datasets, 53 sets): `exchange_rate` +
  `monash_fred_md`; **no disclosed BTC overlap**. lag_llama corpus
  (dataset_list.py @ df7531a, 27 sets): `exchange_rate` only; **no disclosed BTC
  overlap**. FX-class exposure matters at full-grid stage, not the pilot trio.
- SPX and NVDA: no disclosed overlap in any audited component (scope caveat
  applies).

## BTC contamination protocol (fixed, D3): difference-in-differences

Breakpoint = **2021-07** (end of the Monash bitcoin component). Groups:
BTC-exposed = {chronos_2, moirai_2_0, timesfm_2_5}; BTC-unexposed (disclosed) =
{chronos_bolt, lag_llama}. For each BTC calibration metric m (hit-rate deviation,
QS, FZ0) compute

    DiD = [m_post − m_pre]_exposed − [m_post − m_pre]_unexposed

A contamination signature is DiD > 0 (exposed models deteriorate more once the
in-corpus window ends), robust across ≥2 metrics. The plain pre/post split within
one model is NOT evidence by itself (regimes moved too); the unexposed group is
the regime control. Read at α=5% (nα limits per the table above).
