# Stage-5 prescription table v2 (script-generated; matched evaluation sample per design §A — OOS day 501+, common day set; supersedes the delivered mixed-sample table, CHANGELOG 2026-07-08)

alpha=1%. Coverage = fraction of assets passing MC UC+CC at 5%. FZ0 = mean loss over assets with valid ES cells; '(n=k)' marks means over k<32 assets (F4's additive shift breaks e<v on >1% of days in those cells — audit A5). '‡' = precision-fragile (median n_alpha < 10; matched median n = 2139). References on the SAME matched sample: FHS pass = 78%, GARCH-EVT pass = 72%.

Footnote (FHS reference drop — prerequisite d): the FHS criterion-(a) pass rate is 91% on the delivered full-OOS sample but 78% here — an attribution, not cherry-picking. The matched sample's 500-day burn-in (design §A common day set) removes the calm 2016–17 segment where FHS covered well, shortening per-asset n (spx 2639→2139). The matched FHS pass sets the (a) threshold at 68.1%; the delivered value would set a higher 80.6% bar (see the dual-sample section). Disclosed to preempt any suspicion the matched sample was chosen to depress the FHS reference.

## Headline heads (native deep-tail-distinguishable; condition 4)

| head | arm | cov E3 | FZ0 E3 | cov E2 | FZ0 E2 | frag |
|---|---|---|---|---|---|---|
| chronos_2 | unrepaired | 69% | 1.837 (n=30) | 53% | 1.907 |  |
| chronos_2 | F1 | 78% | 1.877 | — | — |  |
| chronos_2 | F2 | 72% | 1.836 (n=30) | 56% | 1.904 |  |
| chronos_2 | F3 | 53% | 1.887 (n=30) | 41% | 1.948 |  |
| chronos_2 | F4 | 78% | 2.166 (n=10) | 72% | 2.188 (n=15) |  |
| chronos_2 | H | 69% | 1.850 | — | — |  |
| moirai_2_0 | unrepaired | 59% | 1.836 (n=30) | 53% | 1.918 |  |
| moirai_2_0 | F1 | 62% | 1.899 | — | — |  |
| moirai_2_0 | F2 | 62% | 1.837 (n=30) | 50% | 1.916 |  |
| moirai_2_0 | F3 | 50% | 1.926 (n=30) | 44% | 1.995 |  |
| moirai_2_0 | F4 | 53% | 2.010 (n=9) | 56% | 2.454 (n=7) |  |
| moirai_2_0 | H | 72% | 1.851 | — | — |  |
| lag_llama | unrepaired | 6% | 2.148 (n=30) | 16% | 2.170 |  |
| lag_llama | F1 | 69% | 2.008 | — | — |  |
| lag_llama | F2 | 6% | 2.056 (n=30) | 22% | 2.106 |  |
| lag_llama | F3 | 31% | 2.040 (n=30) | 22% | 2.134 |  |
| lag_llama | F4 | 50% | 1.992 (n=12) | 44% | 2.062 (n=9) |  |
| lag_llama | H | 81% | 1.856 | — | — |  |

Caveat (audit A13): lag_llama is headline-eligible under the frozen condition-4 classification (distinguishable head) but sits OUTSIDE the E3 caliber arbitration, which covers the 4 quantile heads only; its rows carry that reservation. timesfm_2_5 has no native deep tail (interface) and is excluded from the headline product per the frozen manifest; its repair rows live in the CSV and the criterion table below.

## Exhibit: chronos_bolt (clamped head; condition 5 — via E1-E3 only, not a headline ranking entry). E2-vs-E3 dispersion per design §A.

| arm | cov E3 | FZ0 E3 | cov E2 | FZ0 E2 | |cov E3−E2| |
|---|---|---|---|---|---|
| unrepaired | 28% | 1.897 (n=30) | 56% | 1.936 | 28pp |
| F1 | 69% | 1.884 | — | — | — |
| F2 | 38% | 1.876 (n=30) | 56% | 1.928 | 19pp |
| F3 | 50% | 1.915 (n=30) | 34% | 1.984 | 16pp |
| F4 | 59% | 2.072 (n=10) | 59% | 2.247 (n=15) | 0pp |
| H | 72% | 1.849 | — | — | — |

## F2 gamma robustness (ruling 2 — mandatory column; gamma = 0.01·s_hat·mult)

| head | cov 0.5x | cov 1x | cov 2x | FZ0 0.5x | FZ0 1x | FZ0 2x |
|---|---|---|---|---|---|---|
| chronos_2 | 75% | 72% | 78% | 1.836 (n=30) | 1.836 (n=30) | 1.835 (n=30) |
| moirai_2_0 | 59% | 62% | 69% | 1.836 (n=30) | 1.837 (n=30) | 1.837 (n=30) |
| lag_llama | 6% | 6% | 16% | 2.092 (n=30) | 2.056 (n=30) | 2.010 (n=30) |
| chronos_bolt | 31% | 38% | 47% | 1.885 (n=30) | 1.876 (n=30) | 1.864 (n=30) |
| timesfm_2_5 | 66% | 69% | 69% | 1.844 (n=30) | 1.843 (n=30) | 1.842 (n=30) |

lag_llama attribution (ruling 2): 0.5x: mean hit 0.0224, UC-only 0, CC-only 0, both 30; 1x: mean hit 0.0207, UC-only 0, CC-only 0, both 30; 2x: mean hit 0.0186, UC-only 0, CC-only 0, both 27. Far-from-nominal hit rates with dual UC+CC failures indicate gamma scale mismatch (adaptation too slow for the head's bias), not the honest-ACI clustering limitation; the 2x column is the direct test.

## Success criterion (design §A; (a) pass-rate >= FHS−10pp; (b) FZ0 <= 1.05·min(GJR-t, FHS) per asset on >= 2/3 of non-fragile assets; NaN FZ0 counts as (b)-fail)

(a) threshold = 68.1%. All assets non-fragile at the matched n.

| head | arm | pass rate | (a) | (b) frac | joint |
|---|---|---|---|---|---|
| chronos_2 | F1 | 78% | PASS | 0.62 | fails |
| chronos_2 | F2 | 72% | PASS | 0.50 | fails |
| chronos_2 | F3 | 53% | fail | 0.28 | fails |
| chronos_2 | F4 | 78% | PASS | 0.22 | fails |
| chronos_2 | H | 69% | PASS | 0.84 | **SUCCEEDS** |
| moirai_2_0 | F1 | 62% | fail | 0.47 | fails |
| moirai_2_0 | F2 | 62% | fail | 0.53 | fails |
| moirai_2_0 | F3 | 50% | fail | 0.19 | fails |
| moirai_2_0 | F4 | 53% | fail | 0.09 | fails |
| moirai_2_0 | H | 72% | PASS | 0.81 | **SUCCEEDS** |
| lag_llama | F1 | 69% | PASS | 0.12 | fails |
| lag_llama | F2 | 6% | fail | 0.06 | fails |
| lag_llama | F3 | 31% | fail | 0.06 | fails |
| lag_llama | F4 | 50% | fail | 0.00 | fails |
| lag_llama | H | 81% | PASS | 0.78 | **SUCCEEDS** |
| chronos_bolt | F1 | 69% | PASS | 0.53 | fails |
| chronos_bolt | F2 | 38% | fail | 0.38 | fails |
| chronos_bolt | F3 | 50% | fail | 0.25 | fails |
| chronos_bolt | F4 | 59% | fail | 0.09 | fails |
| chronos_bolt | H | 72% | PASS | 0.81 | **SUCCEEDS** |
| timesfm_2_5 | F1 | 59% | fail | 0.44 | fails |
| timesfm_2_5 | F2 | 69% | PASS | 0.41 | fails |
| timesfm_2_5 | F3 | 50% | fail | 0.19 | fails |
| timesfm_2_5 | F4 | 53% | fail | 0.09 | fails |
| timesfm_2_5 | H | 72% | PASS | 0.78 | **SUCCEEDS** |

GARCH-EVT self-test (prerequisite c, matched sample): pass rate 72% >= 68.1% -> (a) PASS; FZ0 <= 1.05·min(GJR-t,FHS) on 88% of assets >= 2/3 -> (b) PASS; joint = **SUCCEEDS**. The classical McNeil–Frey comparator meets its own bar on the paper sample. (On the delivered sample it fails (a) only — 75% vs the inflated 80.6% bar — with (b) still 0.91.)

## Dual-sample criterion robustness (prerequisite b)

The success criterion re-run per (head, arm) on the matched sample (paper) and the delivered mixed full-OOS sample. (a)-threshold: matched 68.1% (FHS 78%), delivered 80.6% (FHS 91%; burn-in-inflated, footnote d). The delivered sample is superseded (audit A2) and shown only to demonstrate the headline is not a sample-choice artifact.

| head | arm | matched pass | matched joint | delivered pass | delivered joint |
|---|---|---|---|---|---|
| chronos_2 | F1 | 78% | fails | 84% | fails |
| chronos_2 | F2 | 72% | fails | 78% | fails |
| chronos_2 | F3 | 53% | fails | 47% | fails |
| chronos_2 | F4 | 78% | fails | 78% | fails |
| chronos_2 | H | 69% | SUCCEEDS | 78% | fails |
| moirai_2_0 | F1 | 62% | fails | 66% | fails |
| moirai_2_0 | F2 | 62% | fails | 69% | fails |
| moirai_2_0 | F3 | 50% | fails | 56% | fails |
| moirai_2_0 | F4 | 53% | fails | 59% | fails |
| moirai_2_0 | H | 72% | SUCCEEDS | 84% | SUCCEEDS |
| lag_llama | F1 | 69% | fails | 88% | fails |
| lag_llama | F2 | 6% | fails | 3% | fails |
| lag_llama | F3 | 31% | fails | 28% | fails |
| lag_llama | F4 | 50% | fails | 44% | fails |
| lag_llama | H | 81% | SUCCEEDS | 91% | SUCCEEDS |
| chronos_bolt | F1 | 69% | fails | 75% | fails |
| chronos_bolt | F2 | 38% | fails | 28% | fails |
| chronos_bolt | F3 | 50% | fails | 38% | fails |
| chronos_bolt | F4 | 59% | fails | 72% | fails |
| chronos_bolt | H | 72% | SUCCEEDS | 81% | SUCCEEDS |
| timesfm_2_5 | F1 | 59% | fails | 69% | fails |
| timesfm_2_5 | F2 | 69% | fails | 59% | fails |
| timesfm_2_5 | F3 | 50% | fails | 53% | fails |
| timesfm_2_5 | F4 | 53% | fails | 50% | fails |
| timesfm_2_5 | H | 72% | SUCCEEDS | 81% | SUCCEEDS |

H headline: matched 5/5, delivered 4/5. The sole verdict change across the two samples is chronos_2+H: it clears the matched 68.1% (a)-bar (69%, margin +0.6pp) but misses the higher delivered 80.6% bar (78%, margin -2.5pp), while its (b) fraction holds either way (0.84 matched / 0.78 delivered). This is a threshold effect from FHS's calm-segment-inflated delivered pass rate, not an H degradation. H is the only arm that succeeds on any head in either sample (F1–F4 fail jointly throughout); no F1–F4 cell changes verdict between samples.

## Location vs scale decomposition — F1 vs H vs GARCH-EVT (ruling 3)

F1 = TSFM location + TSFM-IQR scale; H = TSFM location + GARCH scale; GARCH-EVT = GARCH location + GARCH scale. Binding reading (ruling 3): **at the daily one-step horizon, location is a second-order term; the binding shortfall of TSFM heads is the conditional scale. Substituting the GARCH sigma (H) lets TSFM-anchored EVT match classical GARCH-EVT.** Mean FZ0 at alpha=1%, E3, matched sample.

| head | F1 FZ0 | H FZ0 | GARCH-EVT FZ0 | scale term (F1−H) | location term (H−GE) |
|---|---|---|---|---|---|
| chronos_2 | 1.877 | 1.850 | 1.847 | +0.027 | +0.004 |
| moirai_2_0 | 1.899 | 1.851 | 1.847 | +0.048 | +0.004 |
| lag_llama | 2.008 | 1.856 | 1.847 | +0.152 | +0.009 |
| chronos_bolt | 1.884 | 1.849 | 1.847 | +0.034 | +0.003 |
| timesfm_2_5 | 1.902 | 1.850 | 1.847 | +0.052 | +0.003 |

## DM-on-FZ0 double bar (registered; HAC h=1, per asset, alpha=1%, matched sample)

| head | pair | assets GE/F4 better (p<0.05) | assets F1 better (p<0.05) | median stat |
|---|---|---|---|---|
| chronos_2 | F1_vs_garch_evt | 4/32 | 0/32 | +0.42 |
| chronos_2 | F1_vs_F4 | 0/32 | 11/32 | -1.43 |
| moirai_2_0 | F1_vs_garch_evt | 6/32 | 0/32 | +1.35 |
| moirai_2_0 | F1_vs_F4 | 0/32 | 14/32 | -1.80 |
| lag_llama | F1_vs_garch_evt | 19/32 | 0/32 | +2.36 |
| lag_llama | F1_vs_F4 | 0/32 | 12/32 | -1.28 |
| chronos_bolt | F1_vs_garch_evt | 5/32 | 0/32 | +0.75 |
| chronos_bolt | F1_vs_F4 | 0/32 | 8/32 | -1.32 |
| timesfm_2_5 | F1_vs_garch_evt | 8/32 | 0/32 | +1.13 |
| timesfm_2_5 | F1_vs_F4 | 0/32 | 12/32 | -1.51 |

(mean_diff > 0 = second element better on FZ0.)

## MCS membership at alpha=1% (10% level, per (head, asset); share of assets where arm is in the confidence set)

| head | F1 | F2 | F3 | F4 | H | garch_evt | fhs |
|---|---|---|---|---|---|---|---|
| chronos_2 | 94% | 88% | 69% | 78% | 100% | 100% | 100% |
| moirai_2_0 | 94% | 91% | 50% | 66% | 100% | 100% | 100% |
| lag_llama | 47% | 44% | 28% | 34% | 97% | 100% | 94% |
| chronos_bolt | 88% | 88% | 53% | 78% | 100% | 100% | 100% |
| timesfm_2_5 | 84% | 88% | 50% | 66% | 100% | 100% | 100% |
