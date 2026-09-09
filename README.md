# Do Time Series Foundation Models Know Their Tails?

Zero-shot Value-at-Risk (VaR) and Expected Shortfall (ES) from time series foundation models (TSFMs): an audit of how the forecast's *output form* shapes tail-risk validity, with training-free repairs.

**Report:** [`paper/tsfm_tail_risk.pdf`](paper/tsfm_tail_risk.pdf) (working paper, September 2026, 65 pages including appendices).

Author: Jianxiu Hou (jianxiuhou9@gmail.com).

## The question

A zero-shot TSFM reports a 1% return quantile for tomorrow. That number depends on two things: what the model has learned about the conditional distribution, and what its output head can *express*. Heads differ (a bounded quantile grid, a token-sampling head, a parametric Student-t head, a mixture head, a calibrated quantile head), and standard central-quantile evaluations conflate the two. This project asks whether the probabilistic representation a head emits constrains whether a zero-shot tail forecast is identifiable at all, how sensitive it is to the rule used to read the tail off the interface, and whether it can be repaired without training.

## What was done

- **Models.** Five TSFMs at pinned Hugging Face revisions with documented pretraining cutoffs ([`harness/registry.yaml`](harness/registry.yaml)), plus a two-model mechanism sub-panel on a 12-asset subset that supplies within-family head contrasts.

  | Model | Output head | Role |
  |---|---|---|
  | Chronos-Bolt (base) | bounded quantile grid (deciles 0.1–0.9) | core panel |
  | Chronos-2 | quantile head | core panel |
  | TimesFM-2.5 (200M) | quantile head | core panel |
  | Moirai-2.0 (R-small) | quantile head | core panel |
  | Lag-Llama | parametric Student-t head | core panel |
  | Chronos T5-base | token-quantized sampling head | mechanism sub-panel |
  | Moirai-1.1 (R-small) | mixture head | mechanism sub-panel |

- **Data.** 32 daily series across six asset classes (equity indices, single stocks, FX, crypto, commodities, rates) from public sources (Yahoo Finance chart API, FRED), log returns ×100 (rates as first differences in basis points), with a QC report and a manifest ([`data/`](data/)).
- **Targets.** One-step VaR and ES at α ∈ {1%, 2.5%, 5%}, rolling windows. All rolling-window logic lives in one module, [`harness/rolling.py`](harness/rolling.py), so that no code path can look ahead.
- **Extraction rules E0–E3.** Four pre-specified rules for turning an output interface into a deep-tail forecast: the native output (E0), a normal tail (E1), a Student-t tail with ν fitted on the context (E2), and a peaks-over-threshold GPD splice (E3). The rule is treated as an experimental axis, not a nuisance ([`erules/rules.py`](erules/rules.py)).
- **Comparators.** Seven econometric risk models: GARCH(1,1)-t, GJR-GARCH-t, EWMA(0.94), historical simulation (250 and 500 days), filtered historical simulation, CAViaR-SAV.
- **Backtests.** Kupiec, Christoffersen (independence and conditional coverage), Engle–Manganelli DQ, quantile and FZ0 scores, Acerbi–Székely Z2, the ESR regression backtest, Diebold–Mariano, and Model Confidence Sets, with Monte-Carlo-calibrated test sizes; effective tail counts nα and precision-fragile flags for ES comparisons. Every statistic is unit-tested against vendored reference outputs (R `esback`/`rugarch`, QuantLet) before use ([`backtests/`](backtests/), [`precision/`](precision/), [`tests/`](tests/)).
- **Contamination control.** Evidence overlapping the models' disclosed pretraining corpora is kept separate from strictly post-cutoff evidence and from a synthetic GARCH-t oracle arm ([`synth/`](synth/)).
- **Regimes.** Event-time calibration decay and recovery around volatility episodes defined by a rule frozen before results were viewed ([`regimes/rules.py`](regimes/rules.py)).
- **Repairs.** Four training-free repair arms (F1 EVT splice, F2 adaptive conformal, F3 rescale, F4 additive FZ0 recalibration) and a hybrid H that keeps the TSFM's conditional location but substitutes the GARCH conditional scale before fitting an EVT tail, compared with classical GARCH–EVT under one protocol ([`fixes/`](fixes/)).

## Findings

1. **Native deep tails range from unusable by construction to partially calibrated, and some rankings depend on the extraction rule.** Chronos-Bolt clamps any request below its trained grid to the grid edge, which yields a 12.7% violation rate at a nominal 1%. Lag-Llama produces distinct tail quantiles, but none of its 96 asset–tail cells passes unconditional coverage under the GPD splice. The quantile heads avoid those failures, yet their Student-t and GPD-splice rankings disagree (Kendall τ < 0.5) on 4 of the 31 assets comparable at α = 1%. Extraction is therefore part of what is measured.
2. **Failures are associated with output form.** Two within-family contrasts (the Chronos lineage's grid, token-sampling and quantile heads; Moirai's mixture versus quantile head) and one cross-form contrast (quantile head versus parametric Student-t head) link the observed failures to the head. The evidence is associative, not causal. Regime failures are head-specific rather than a class property.
3. **The conditional scale is the binding shortfall; location is second-order at the daily one-step horizon.** Replacing the scale helps all five models, most strongly the weakest. The hybrid H is the only training-free arm that meets the joint criterion on all five models, and it matches classical GARCH–EVT without exceeding it.

The practical conclusion is that deep-tail comparisons between foundation models are not well defined without stating the extraction rule, and that a decision guide can map each output form to its defensible extraction rule, the minimum repair the evidence supports, or a recommendation to replace the head.

## Contributions

- **C1. Interface and extraction identification.** A mechanism analysis built on the within-family and cross-form contrasts, with the extraction rules E0–E3 as an experimental axis and a pre-committed test of the token-quantization hypothesis.
- **C2. Tail-risk validation and regime evidence.** A comparison of TSFMs with econometric risk models using coverage tests, dynamic-quantile diagnostics and Model Confidence Sets; an ES layer with FZ0 loss, ESR backtests, effective tail counts and precision-fragile flags; and event-time calibration decay and recovery under a frozen episode rule.
- **C3. Component substitution and deployment implications.** The repair arms F1–F4 and the hybrid H compared with GARCH–EVT under one protocol, and a decision guide from output form to extraction rule and repair.

## Repository layout

```
data/        fetchers, QC, asset universe, data manifest (parquet store is regenerated by data/build.py)
harness/     adapter interface (base.py), per-model adapters, registry.yaml (pinned revisions + cutoffs),
             rolling.py (the single no-lookahead window module), baselines.py, run_*.py grid and analysis
             drivers, the vendored R ESR engine
erules/      E0–E3 tail extraction rules
backtests/   VaR tests, ES tests, ESR, quantile/FZ0 scores, Diebold–Mariano, Model Confidence Set
precision/   effective tail counts, precision floors, fragile flags; Monte-Carlo critical values
regimes/     frozen volatility-episode rules and realized-vol strata
fixes/       repair arms F1–F4, hybrid H, GARCH–EVT comparator
synth/       GARCH-t / GJR-t simulated oracle arm
tests/       pytest suite; fixtures/ vendors the reference outputs the statistics are checked against
results/     stage4/ (grid backtests, repairs, regimes, ESR, MCS, diagnostics) and mechanism/ (sub-panel
             forecasts and the decoding probe); every table and figure is generated from these files
paper/       tsfm_tail_risk.pdf, figures/ (table and figure scripts), evidence_pack.md (the frozen number
             pack the scripts assert against), draft/generated and draft/figures (script outputs)
envs/        lock files for the main environment and the isolated Lag-Llama environment
docs/        the two analysis notes the table script parses
```

## Design discipline

- The extraction and testing protocol, the α levels, the model panel, the regime rules and the out-of-sample windows were fixed before results were viewed.
- Every Hugging Face revision is pinned and each model's pretraining cutoff is recorded with its source.
- All randomness is seeded and logged in run metadata; run provenance is frozen in `results/stage4/run_manifest.yaml` and generation-time freshness sidecars.
- Numbers in tables and figures are script-generated from `results/`; every aggregate that also appears in the frozen evidence pack is asserted equal at build time.
- Public data only.

## Reproducing

```bash
conda env create -f environment.yml
conda activate tsfm-tails
pytest -q
```

Lag-Llama runs in an isolated virtual environment built from `envs/lag-llama-requirements.txt`; the harness calls it through a subprocess worker. The R engine for the ESR backtest needs `esback` (see `harness/README_R_engines.md`); its frozen outputs are in `results/stage4/esr_results.csv`.

Data are fetched with `python -m data.build`, which writes the parquet store, the QC report and the manifest. The grid is driven by the `harness/run_grid_*.py` scripts (forecast, extract, baselines, compare, repairs) followed by the regime, strata, ESR and MCS drivers; model weights are downloaded at the pinned revisions on first use. The full five-model grid was run on a rented GPU; Chronos-Bolt, Lag-Llama and all statistics run on a laptop. Tables and figures are rebuilt with `paper/figures/make_tables.py` and `paper/figures/make_figures.py`.

The per-day forecast, baseline and repair layers (several hundred megabytes of parquet) are not tracked here; they are regenerated by the drivers above. The aggregate CSVs behind every table and figure are tracked under `results/`.
