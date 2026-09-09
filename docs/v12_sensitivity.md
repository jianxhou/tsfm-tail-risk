# v1.2 external-review sensitivity analyses (script-generated; do not hand-edit)

Source: results/stage4/repair_backtests_matched.csv (matched sample; zero new forecasts). alpha=1%, E3. Cluster bootstrap: 6 asset classes resampled with replacement, B=2000, seed 20260717. FHS pass (matched) = 78.1%.

## A. Decomposition uncertainty (per-asset paired FZ0 differences)

| head | n assets | n(scale>loc) | mean scale (F1-H) [95% CI] | mean loc (H-GE) [95% CI] |
|---|---|---|---|---|
| chronos_2 | 32 | 22 | +0.027 [-0.002, +0.055] | +0.004 [-0.004, +0.013] |
| moirai_2_0 | 32 | 24 | +0.048 [+0.011, +0.075] | +0.004 [-0.001, +0.010] |
| lag_llama | 32 | 30 | +0.152 [+0.124, +0.179] | +0.009 [+0.001, +0.018] |
| chronos_bolt | 32 | 22 | +0.034 [+0.003, +0.058] | +0.003 [-0.003, +0.008] |
| timesfm_2_5 | 32 | 21 | +0.052 [+0.012, +0.083] | +0.003 [-0.007, +0.013] |

### A2. Cluster-valid inference (v2.0 P0-6: CR1 cluster-robust SE, t(5) 95% CI, restricted wild cluster bootstrap p — Rademacher, full 2^6=64 enumeration, deterministic; G=6 asset classes)

| head | scale CR1 SE | scale t | scale t(5) CI | scale WCR p | loc CR1 SE | loc t | loc t(5) CI | loc WCR p |
|---|---|---|---|---|---|---|---|---|
| chronos_2 | 0.0163 | +1.63 | [-0.0153, +0.0685] | 0.188 | 0.0047 | +0.81 | [-0.0082, +0.0159] | 0.438 |
| moirai_2_0 | 0.0173 | +2.80 | [+0.0039, +0.0926] | 0.125 | 0.0029 | +1.38 | [-0.0034, +0.0114] | 0.219 |
| lag_llama | 0.0157 | +9.71 | [+0.1118, +0.1923] | 0.031 | 0.0046 | +1.97 | [-0.0027, +0.0208] | 0.125 |
| chronos_bolt | 0.0147 | +2.33 | [-0.0036, +0.0720] | 0.156 | 0.0030 | +0.91 | [-0.0050, +0.0104] | 0.438 |
| timesfm_2_5 | 0.0195 | +2.70 | [+0.0025, +0.1025] | 0.125 | 0.0054 | +0.53 | [-0.0110, +0.0167] | 0.625 |

## B. Registered-criterion threshold sensitivity for H

Grid: margin {5,10,15}pp x FZ0 tolerance {1.00,1.025,1.05,1.10} x fraction {1/2,2/3,3/4} = 36 combinations. Registered combo = (10pp, 1.05, 2/3): H joint success 5/5 heads; coverage 69/72/81/72/72; b-fractions .84/.81/.78/.81/.78. Drift vs the pre-v2.0 frozen values (expected — v2.0 P0-1/2/3 recomputation, CHANGELOG): moirai_2_0: pr 0.75 -> 0.72.

H passes on all five heads in **14 of 36** combinations.

| margin (pp) | frac | tol=1.00 | tol=1.025 | tol=1.05 | tol=1.10 |
|---|---|---|---|---|---|
| 5 | 1/2 | 0 | 1 | 1 | 1 |
| 5 | 2/3 | 0 | 0 | 1 | 1 |
| 5 | 3/4 | 0 | 0 | 1 | 1 |
| 10 | 1/2 | 0 | 5 | 5 | 5 |
| 10 | 2/3 | 0 | 3 | 5 | 5 |
| 10 | 3/4 | 0 | 0 | 5 | 5 |
| 15 | 1/2 | 0 | 5 | 5 | 5 |
| 15 | 2/3 | 0 | 3 | 5 | 5 |
| 15 | 3/4 | 0 | 0 | 5 | 5 |

Margin-5pp criticality: at margin 5pp the condition-(a) threshold rises to 73.1% and 4 head(s) fall below it (chronos_2 at 69%; moirai_2_0 at 72%; chronos_bolt at 72%; timesfm_2_5 at 72%), so 1 of five heads pass; for Chronos-2+H condition (a) FAILS and the joint verdict is FAILURE. Per-head critical condition-(a) margins (head fails once the margin tightens past): lag_llama below -3.12 pp, moirai_2_0 below 6.25 pp, chronos_bolt below 6.25 pp, timesfm_2_5 below 6.25 pp, chronos_2 below 9.38 pp.

## C. Bare pass counts behind the §10.2 H coverage rates

| head | pass count | rate |
|---|---|---|
| chronos_2 | 22/32 | 69% |
| moirai_2_0 | 23/32 | 72% |
| lag_llama | 26/32 | 81% |
| chronos_bolt | 23/32 | 72% |
| timesfm_2_5 | 23/32 | 72% |
