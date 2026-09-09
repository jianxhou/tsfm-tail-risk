# Paper evidence pack — Stage 6 step 1 (number-freeze review)

Machine-extracted per hard rule 11: every value below is parsed from a committed, script-generated source at build time (no hand-typed numbers); every commit hash + sha256 is read from git / the file at build time. Regenerate with `python paper/build_evidence_pack.py`. A `*` on a hash marks a source with uncommitted changes at build time (should be none in the frozen pack).

Legend per number line: **value → source file → generating script → commit hash (sha256-12)**. Organized by the proposal-§12 skeleton.

## 0. Design-compliance checklist (hard rule 12 — acceptance starts here)

| pre-registered clause (governing design) | implementation evidence | commit hash | sha256-12 |
|---|---|---|---|
| α levels {1%, 2.5%, 5%} (proposal §3.3) | `results/stage4/run_manifest.yaml` | `3a6f980` | `3cda980734b7` |
| E-rules E0–E3 frozen (proposal §3.3) | `erules/rules.py` | `4ee13e7` | `a7e3fabffeed` |
| CORE 5-head panel pinned + cutoffs (proposal §3.1, hard rule 2) | `harness/registry.yaml` | `c748942` | `6ae15b816a54` |
| OOS 2016-01-01, ctx 512, fit 1000/refit 21, horizon 1 (proposal §3.2) | `results/stage4/run_manifest.yaml` | `3a6f980` | `3cda980734b7` |
| No-lookahead: all rolling logic in one reviewed module (hard rule 3) | `harness/rolling.py` | `9df9f79` | `696155f094c8` |
| Regime episodes frozen before results (hard rule 6) | `regimes/rules.py` | `70c0729` | `19b33adf3a45` |
| Backtest battery unit-tested vs reference fixtures (hard rule 1) | `tests/test_var_tests.py` | `8330dcc` | `99640b9ae1f2` |
| MC-calibrated critical values (proposal §3.4) | `tests/test_mc_critical.py` | `9190dcb` | `20850825189d` |
| Precision floor / n_alpha / fragile flag (Pele; proposal §3.4) | `precision/diagnostics.py` | `16ba8c5` | `ef1c1eeff653` |
| Contamination-aware: disclosed-corpus overlap audit (proposal §4) | `docs/giftevalpretrain_overlap.md` | `3e05e42` | `48ce70b2b962` |
| Synthetic-DGP oracle arm (proposal §5/condition 10) | `docs/stage4_synth_arbitration.md` | `eccbf26` | `f4feb16a40bb` |
| Repairs F1–F4 + hybrid H + GARCH-EVT comparator (proposal §6) | `fixes/arms.py` | `4ee13e7` | `8108cd78156b` |
| Leak sentinel over every repair arm (audit A14) | `tests/test_fixes.py` | `358a51f` | `0dd08eafaed3` |
| Success criterion, design §A, dual-sample (close_decision b) | `docs/stage5_prescription.md` | `eccbf26` | `24326615cce7` |
| A16 external McNeil–Frey reference + garch_filter unit test (close_decision a) | `tests/test_garch_evt_reference.py` | `4ee13e7` | `24167661e6b4` |
| Numbers script-generated from results/ (hard rule 5) | `paper/build_evidence_pack.py` | `9ace69d` | `0db393babf22` |

All sixteen clauses map to a tracked artifact; the two Stage-6 additions (A16 external fixture; dual-sample success criterion) close the closing-verdict prerequisites. Full provenance ledger at the foot.

## §1 Introduction — thesis & headline scope

- **Assets in the full grid**: `32` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **Assets quarantined (FX bad-tick)**: `eurusd, usdjpy` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **CORE TSFM heads**: `5` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **Econometric baselines**: `7` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **VaR/ES levels α**: `0.01, 0.025, 0.05` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **Extraction-dependent ranking cells (τ<0.5, α=1%)**: `4/31 (v2.1: dgs30 exits via E3 ES-invalidity)` → `docs/stage4_kendall_tau.md` → `harness/run_grid_arbitration.py` → `eccbf26` (`45c4afbd7ddd`)
- **Hybrid H meets success criterion (heads, matched sample)**: `5/5` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)

## §3 Models & Head Taxonomy

- **chronos_bolt HF revision**: `5d9f166d69f47aef3401367a7b842e78fe97b121` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **chronos_2 HF revision**: `29ec3766d36d6f73f0696f85560a422f50e8498c` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **timesfm_2_5 HF revision**: `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **moirai_2_0 HF revision**: `30f43ff08c8494f4943ae1521e9d4e94a0fbb389` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **lag_llama HF revision**: `72dcfc29da106acfe38250a60f4ae29d1e56a3d9` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **chronos_bolt head_type / native tail**: `bounded_quantile_grid / clamp-to-decile-edge` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **lag_llama head_type / native tail**: `parametric_student_t / analytic-parametric` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **bolt E0−E2 hit spread (clamped pathology)**: `11.4pp` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **lag_llama E0−E2 hit spread (too-narrow pathology)**: `1.7pp` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **chronos_2 E0−E2 hit spread (healthy)**: `0.0pp` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **timesfm_2_5 native deep tail**: `reject-non-decile` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **moirai_2_0 E0−E2 hit spread (healthy)**: `0.7pp` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **lag_llama native 5% VaR violation rate on SPX (too-narrow)**: `~9.8%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)

## §4 Data & Contamination-Aware Design

- **OOS start**: `2016-01-01` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **Context length**: `512` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **Fit length / refit cadence**: `1000 / 21` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **Forecast horizon**: `1` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **Global seed**: `20260706` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **Config hash (frozen grid)**: `d8d9e7c8f58dc5d2` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **GiftEvalPretrain financial components (of 152)**: `**btc** (BTC-USD daily)  …(btc only)` → `docs/giftevalpretrain_overlap.md` → `(D2 audit; machine-sourced from HF dataset APIs)` → `3e05e42` (`48ce70b2b962`)
- **BTC-exposed heads (disclosed corpus)**: `chronos_2, moirai_2_0, timesfm_2_5 (bitcoin_with_missing)` → `docs/giftevalpretrain_overlap.md` → `(D2 audit; machine-sourced from HF dataset APIs)` → `3e05e42` (`48ce70b2b962`)
- **BTC E2 hit@5% chronos_2 (exposed)**: `6.43%` → `docs/stage4_btc_did.md` → `harness/run_grid_arbitration.py` → `a88a364` (`5e078c26d0d9`)
- **BTC E2 hit@5% lag_llama (unexposed)**: `6.43%` → `docs/stage4_btc_did.md` → `harness/run_grid_arbitration.py` → `a88a364` (`5e078c26d0d9`)
- **BTC E2 hit@5% chronos_bolt (unexposed)**: `5.66%` → `docs/stage4_btc_did.md` → `harness/run_grid_arbitration.py` → `a88a364` (`5e078c26d0d9`)
- **BTC E2 hit@5% timesfm_2_5 (exposed)**: `5.35%` → `docs/stage4_btc_did.md` → `harness/run_grid_arbitration.py` → `a88a364` (`5e078c26d0d9`)
- **BTC E2 hit@5% moirai_2_0 (exposed)**: `4.14%` → `docs/stage4_btc_did.md` → `harness/run_grid_arbitration.py` → `a88a364` (`5e078c26d0d9`)
- **Series fetched + QC'd (incl. 2 quarantined; excl. vol indices)**: `34` → `data/data_manifest.yaml` → `data/build.py` → `b04bf5a` (`698566aaeea9`)
- **Sample end (spx)**: `2026-07-02` → `data/data_manifest.yaml` → `data/build.py` → `b04bf5a` (`698566aaeea9`)
- **Pretraining-cutoff bounds (bolt/c2/timesfm/moirai/lag_llama)**: `2025-11-21, 2025-10, 2023-11, 2025-11, 2023-10` → `harness/registry.yaml` → `(pinned by hand at Stage 1; ids machine-asserted, hard rule 11)` → `c748942` (`6ae15b816a54`)
- **BTC split breakpoint**: `2021-07-01` → `docs/stage4_btc_did_split.md` → `harness/run_btc_did_split.py` → `d49e1ed` (`429645f9112b`)
- **BTC exposed group hit pre / post (E2, α=5%)**: `5.56% / 5.03%` → `docs/stage4_btc_did_split.md` → `harness/run_btc_did_split.py` → `d49e1ed` (`429645f9112b`)
- **BTC unexposed group hit pre / post (E2, α=5%)**: `6.03% / 6.07%` → `docs/stage4_btc_did_split.md` → `harness/run_btc_did_split.py` → `d49e1ed` (`429645f9112b`)
- **BTC DiD (exposed−unexposed, post−pre)**: `-0.57pp` → `docs/stage4_btc_did_split.md` → `harness/run_btc_did_split.py` → `d49e1ed` (`429645f9112b`)
- **BTC split n days pre / post**: `1966 / 1830` → `docs/stage4_btc_did_split.md` → `harness/run_btc_did_split.py` → `d49e1ed` (`429645f9112b`)

## §5 Evaluation Protocol

- **E-rules pre-registered**: `e0, e1, e2, e3` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **MC-calibrated critical values**: `enabled` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **ESR engine**: `vendored-r-engine` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **GAS cross-check**: `dropped-per-3.1` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **Sampling-head MC MAE q01, S=250 → 4000**: `0.233 → 0.046` → `docs/pilot_d3_calibration.md` → `harness/run_pilot_d3.py` → `c4bb08c` (`45154dd55f50`)
- **E3 GPD bootstrap std q01, ctx=512**: `0.338 (std)` → `docs/pilot_d3_calibration.md` → `harness/run_pilot_d3.py` → `c4bb08c` (`45154dd55f50`)
- **Backtest battery (unit-tested vs fixtures)**: `kupiec, christoffersen_cc, dq, quantile_score, fz0, as_z2, esr, dm, mcs` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **E3 POT threshold τ_u (ctx-quantile)**: `0.10` → `erules/rules.py` → `erules/rules.py` → `4ee13e7` (`a7e3fabffeed`)
- **E2 ν clip lower bound**: `2.1` → `erules/rules.py` → `erules/rules.py` → `4ee13e7` (`a7e3fabffeed`)
- **Repair calibration window W / refit cadence**: `500 / 21` → `docs/stage5_design.md` → `(design doc v2.1, signer-approved 2026-07-07)` → `d282d05` (`b564e01b18f6`)
- **Matched-sample evaluation start (OOS day)**: `501` → `docs/stage5_design.md` → `(design doc v2.1, signer-approved 2026-07-07)` → `d282d05` (`b564e01b18f6`)
- **Matched-sample median n per asset**: `2139` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **Success criterion (a): pass-rate floor**: `FHS − 10pp` → `docs/stage5_design.md` → `(design doc v2.1, signer-approved 2026-07-07)` → `d282d05` (`b564e01b18f6`)
- **Success criterion (b): FZ0 tolerance × best baseline**: `1.05` → `docs/stage5_design.md` → `(design doc v2.1, signer-approved 2026-07-07)` → `d282d05` (`b564e01b18f6`)
- **Success criterion (b): asset fraction floor**: `2/3` → `docs/stage5_design.md` → `(design doc v2.1, signer-approved 2026-07-07)` → `d282d05` (`b564e01b18f6`)
- **MC null replications / test level**: `10,000 / 0.05` → `docs/stage5_design.md` → `(design doc v2.1, signer-approved 2026-07-07)` → `d282d05` (`b564e01b18f6`)
- **Day-cluster bootstrap B (regime bands)**: `2000` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **Event window (trading days post-onset)**: `60` → `harness/run_regime_analysis.py` → `harness/run_regime_analysis.py` → `9ace69d` (`5bc0948ed0c9`)
- **Episode rule constants (frozen)**: `MULT=1.5, MEDIAN_WIN=250, SUSTAIN=3, SPAN=60, MERGE=20, CALM_RUN=60, floors={'vix': 25.0, 'move': 100.0, 'crypto_rv': 75.0}.` → `docs/stage5_episode_set.md` → `regimes/rules.py (episode enumeration over frozen vol data)` → `70c0729` (`43f7c3323378`)
- **Episodes (rule output, closed set)**: `28 (VIX 11, MOVE 3, crypto_rv 14)` → `docs/stage5_episode_set.md` → `regimes/rules.py (episode enumeration over frozen vol data)` → `70c0729` (`43f7c3323378`)
- **Burn-in effect on per-asset nα@1%**: `~26 to ~21` → `docs/stage5_design.md` → `(design doc v2.1, signer-approved 2026-07-07)` → `d282d05` (`b564e01b18f6`)

## §6 Results: VaR audit (Kupiec MC pass fraction, by head × E-rule)

- **chronos_2 E3**: `84%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **moirai_2_0 E3**: `92%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **timesfm_2_5 E3**: `70%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **chronos_bolt E3 (clamped)**: `20%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **lag_llama E3 (too-narrow)**: `0%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **fhs param**: `100%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **gjr_t param**: `68%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **bolt E0 native hit (mean)**: `12.7%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **lag_llama E0 native hit (mean)**: `3.8%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **garch_t param**: `66%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **caviar_sav param**: `95%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **hs500 param**: `82%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **chronos_2 E1 (normal-tail exhibit)**: `2%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **timesfm_2_5 E2**: `70%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **chronos_2 E0 native hit (mean)**: `1.2%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **moirai_2_0 E0 native hit (mean)**: `1.4%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)

## §7 Results: ES with precision certificates

- **chronos_2 mean FZ0 (α=1%, E2)**: `1.876` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **gjr_t mean FZ0 (α=1%, E2)**: `1.799` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **lag_llama mean FZ0 (α=1%, E2)**: `2.112` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **Median n_alpha (identifiability)**: `26.4` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **moirai_2_0 ESR pass E3**: `85%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **lag_llama ESR pass E3**: `11%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **xi≥1 E3 incidence — rates max frac (dgs30)**: `0.0232 (dgs30)` → `docs/stage5_xi_incidence.md` → `harness/run_xi_incidence.py` → `e77f63d` (`1b304b3d4946`)
- **xi≥1 E3 incidence — every non-rates class**: `0.0000` → `docs/stage5_xi_incidence.md` → `harness/run_xi_incidence.py` → `e77f63d` (`1b304b3d4946`)
- **synth arbitration moirai α=1% (closer extractor; E3 more faithful)**: `E3` → `docs/stage4_synth_arbitration.md` → `synth/run_arbitration.py` → `eccbf26` (`f4feb16a40bb`)
- **fhs mean FZ0 (α=1%, E2)**: `1.814` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **garch_t mean FZ0 (α=1%, E2)**: `1.800` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **moirai_2_0 mean FZ0 (α=1%, E2)**: `1.884` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **timesfm_2_5 mean FZ0 (α=1%, E2)**: `1.873` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **chronos_bolt mean FZ0 (α=1%, E2)**: `1.890` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **Extraction-dependent cells, all α (τ<0.5)**: `8 of 93` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **Kendall τ mean (α=1%)**: `0.723` → `docs/stage4_kendall_tau.md` → `harness/run_grid_arbitration.py` → `eccbf26` (`45c4afbd7ddd`)
- **Synth arbitration mean |ES error| (E3 vs E2)**: `~1.2-1.4 vs ~1.6-2.1` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **xi≥1 E3 rates breakdown**: `dgs30 2.3% (61/2625); dgs2 0.9% (23/2625); dgs10 0.0% (1/2625)` → `docs/stage5_xi_incidence.md` → `harness/run_xi_incidence.py` → `e77f63d` (`1b304b3d4946`)
- **xi≥1 residual-site windows (grid-wide)**: `natgas x chronos_bolt [f1] 1/2449 (0.04%); spx x lag_llama [h] 1/2447 (0.04%)` → `docs/stage5_xi_incidence.md` → `harness/run_xi_incidence.py` → `e77f63d` (`1b304b3d4946`)

## §8 Regimes & Decay (τ_rec, VIX pool, α=5%, trailing-21 median)

- **moirai_2_0 τ_rec**: `24` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **fhs τ_rec**: `24` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **chronos_2 τ_rec**: `26` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **gjr_t τ_rec**: `29` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **chronos_bolt τ_rec**: `30` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **timesfm_2_5 τ_rec**: `31` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **lag_llama τ_rec (slowest)**: `31` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **lag_llama recovered/censored/never**: `7/4/0` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **VIX pool power (median mde_mult, w=21, α=5%)**: `1.49` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **lag_llama COVID trailing-21 rate at t=21**: `0.395` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **lag_llama vol-strata Q1→Q4 MC-UC pass**: `69% → 0%` → `docs/stage5_strata.md` → `harness/run_vol_strata.py` → `eccbf26` (`dd76a3d7066b`)
- **chronos_2 vol-strata Q1/Q4 MC-UC pass**: `88% / 88%` → `docs/stage5_strata.md` → `harness/run_vol_strata.py` → `eccbf26` (`dd76a3d7066b`)
- **timesfm_2_5 vol-strata Q4 MC-UC pass**: `47%` → `docs/stage5_strata.md` → `harness/run_vol_strata.py` → `eccbf26` (`dd76a3d7066b`)
- **chronos_bolt recovered/censored/never (21, α=5%)**: `11/0/0` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **chronos_2 recovered/censored/never (21, α=5%)**: `8/0/3` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **timesfm_2_5 recovered/censored/never (21, α=5%)**: `11/0/0` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **moirai_2_0 recovered/censored/never (21, α=5%)**: `8/1/2` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **gjr_t recovered/censored/never (21, α=5%)**: `9/0/2` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **fhs recovered/censored/never (21, α=5%)**: `6/1/4` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **chronos_2+F1 recovered/censored/never (21, α=5%; never-elevated 7/10)**: `3/0/7` → `docs/stage5_regime.md` → `harness/run_regime_analysis.py` → `eccbf26` (`6721198ec040`)
- **vix_ep17 (COVID) onset**: `2020-02-24` → `docs/stage5_episode_set.md` → `regimes/rules.py (episode enumeration over frozen vol data)` → `70c0729` (`43f7c3323378`)
- **vix_ep23 (2025-04) onset**: `2025-04-03` → `docs/stage5_episode_set.md` → `regimes/rules.py (episode enumeration over frozen vol data)` → `70c0729` (`43f7c3323378`)
- **moirai_2_0 vol-strata Q4 MC-UC pass**: `69%` → `docs/stage5_strata.md` → `harness/run_vol_strata.py` → `eccbf26` (`dd76a3d7066b`)
- **chronos_bolt vol-strata Q4 MC-UC pass**: `53%` → `docs/stage5_strata.md` → `harness/run_vol_strata.py` → `eccbf26` (`dd76a3d7066b`)
- **gjr_t vol-strata Q4 MC-UC pass**: `88%` → `docs/stage5_strata.md` → `harness/run_vol_strata.py` → `eccbf26` (`dd76a3d7066b`)
- **fhs vol-strata Q4 MC-UC pass**: `72%` → `docs/stage5_strata.md` → `harness/run_vol_strata.py` → `eccbf26` (`dd76a3d7066b`)
- **Markov low/high MC-UC pass — chronos_2**: `93% / 89%` → `docs/stage5_strata.md` → `harness/run_vol_strata.py` → `eccbf26` (`dd76a3d7066b`)
- **Markov low/high MC-UC pass — timesfm_2_5**: `86% / 43%` → `docs/stage5_strata.md` → `harness/run_vol_strata.py` → `eccbf26` (`dd76a3d7066b`)
- **Markov low/high MC-UC pass — lag_llama**: `0% / 0%` → `docs/stage5_strata.md` → `harness/run_vol_strata.py` → `eccbf26` (`dd76a3d7066b`)
- **Markov classification convergence**: `28/32` → `docs/stage5_strata.md` → `harness/run_vol_strata.py` → `eccbf26` (`dd76a3d7066b`)

## §9 Mechanism Attribution — head-type contrasts

- **Within-Chronos contrast: chronos_2 (token/quantile, healthy) vs chronos_bolt (bounded grid, clamped) — E0−E2 spread**: `0.0pp vs 11.4pp` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **Quantile-vs-parametric contrast: moirai_2_0 vs lag_llama E3 VaR pass**: `92% vs 0%` → `docs/stage4_final_table.md` → `harness/run_grid_final.py` → `1d9cfcb` (`d8c1cefad822`)
- **GAP — designed within-family panel (chronos_base, moirai_1_1) NOT run**: `registry stubs, cutoff=TODO-D2 — see freeze note` → `harness/registry.yaml` → `(pinned by hand at Stage 1; ids machine-asserted, hard rule 11)` → `c748942` (`6ae15b816a54`)

## §10 Training-Free Repair (matched sample, α=1%, E3)

- **H succeeds (heads) — matched criterion**: `5/5` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **chronos_2+H joint verdict**: `**SUCCEEDS**` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **lag_llama+H joint verdict**: `**SUCCEEDS**` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **F1 best case (lag_llama+F1) joint**: `fails` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **(a) threshold = FHS−10pp (matched)**: `68.1%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **Scale term lag_llama (F1−H)**: `+0.152` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **Location term lag_llama (H−GE)**: `+0.009` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **H FZ0 vs GARCH-EVT FZ0 (lag_llama)**: `1.856 vs 1.847` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **MCS membership H (lag_llama)**: `97%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **DM lag_llama F1_vs_garch_evt (GE better / F1 better)**: `19/32 / 0/32` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **F2 gamma-2x lag_llama coverage (substitution-only)**: `16%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **H (a) pass / (b) frac — chronos_2**: `69% / 0.84` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **H (a) pass / (b) frac — moirai_2_0**: `72% / 0.81` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **H (a) pass / (b) frac — lag_llama**: `81% / 0.78` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **H (a) pass / (b) frac — chronos_bolt**: `72% / 0.81` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **H (a) pass / (b) frac — timesfm_2_5**: `72% / 0.78` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **F1 (b) frac (c2/moirai/lag_llama/bolt/timesfm)**: `0.62 / 0.47 / 0.12 / 0.53 / 0.44` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **Scale/location terms — chronos_2**: `+0.027 / +0.004` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **Scale/location terms — moirai_2_0**: `+0.048 / +0.004` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **Scale/location terms — chronos_bolt**: `+0.034 / +0.003` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **Scale/location terms — timesfm_2_5**: `+0.052 / +0.003` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **H FZ0 range across heads (vs GE 1.847)**: `1.850 / 1.851 / 1.856 / 1.849 / 1.850` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **Unrepaired cov E3 — chronos_2**: `69%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **Unrepaired cov E3 — moirai_2_0**: `59%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **Unrepaired cov E3 — lag_llama**: `6%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **chronos_bolt exhibit unrepaired cov E3 / |cov E3−E2|**: `28% / 28pp` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **chronos_2+H margin (matched)**: `+0.6pp` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **chronos_2+H margin (delivered)**: `-2.5pp` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **(a) threshold (delivered sample)**: `80.6%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **F2 lag_llama mean hit by γ (0.5x/1x/2x)**: `0.0224 / 0.0207 / 0.0186` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **F2 lag_llama FZ0 by γ (0.5x/1x/2x)**: `2.092 (n=30) / 2.056 (n=30) / 2.010 (n=30)` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **H MCS membership (c2/moirai/bolt/timesfm)**: `100% / 100% / 100% / 100%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **F1 MCS membership lag_llama (drop)**: `47%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)

## §11 Discussion — self-consistency of the comparator

- **GARCH-EVT self-test verdict (prereq c, matched)**: `SUCCEEDS` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **GARCH-EVT pass rate (a)**: `72%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **GARCH-EVT (b) fraction**: `88%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **FHS reference drop (delivered → matched)**: `91% → 78%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **GE delivered-sample: fails (a) only**: `75% vs 80.6%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **GE delivered-sample (b) fraction**: `0.91` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **FHS footnote: spx n (delivered→matched)**: `2639→2139` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)

## Appendix — pins, tests, synthetic arm, A16 external McNeil–Frey fixture

- **Gate verdict / conditions**: `GO / 11` → `results/stage4/run_manifest.yaml` → `harness/freeze_run_manifest.py` → `3a6f980` (`3cda980734b7`)
- **evir GPD ξ (McNeil–Frey POT ref)**: `0.2317` → `tests/fixtures/mcneilfrey_evt_reference.csv` → `tests/fixtures/make_mcneilfrey_references.R (evir 1.7.4)` → `8ce91fd` (`06eb7fcb3043`)
- **evir POT VaR α=5% (z-scale, ref)**: `-1.6863` → `tests/fixtures/mcneilfrey_evt_reference.csv` → `tests/fixtures/make_mcneilfrey_references.R (evir 1.7.4)` → `8ce91fd` (`06eb7fcb3043`)
- **rugarch 1-step σ (GARCH-t ref)**: `0.4892` → `tests/fixtures/mcneilfrey_garch_reference.csv` → `tests/fixtures/make_mcneilfrey_references.R (rugarch 1.5.5)` → `8ce91fd` (`2472eee54b02`)
- **A16 cross-check tests (POT formula ≡ evir to 1e-9; σ vs rugarch <1%; extraction exact)**: `5 collected` → `tests/test_garch_evt_reference.py` → `tests/test_garch_evt_reference.py` → `4ee13e7` (`24167661e6b4`)
- **Full statistics suite (pytest --collect-only, build time)**: `139 collected` → `tests/test_garch_evt_reference.py` → `tests/test_garch_evt_reference.py` → `4ee13e7` (`24167661e6b4`)
- **R reference stack (R / rugarch / esback)**: `R version 4.4.2 / rugarch 1.5.5 / esback 0.3.1` → `tests/fixtures/r_versions.txt` → `tests/fixtures/make_r_references.R` → `8330dcc` (`294e39c1edc6`)
- **QuantLet HMD_ES reference commit (FZ0 fixtures)**: `2490253e` → `tests/fixtures/README.md` → `(fixtures documentation; kupiec_p divergence verified by hand + rugarch/GAS, 2026-07-04)` → `9df9f79` (`5189421d2797`)
- **kupiec_p divergence example (HMD_ES vs textbook, oracle α=5%)**: `their LR 6.79 / p 0.0093 vs correct LR 0.10 / p 0.75` → `tests/fixtures/README.md` → `(fixtures documentation; kupiec_p divergence verified by hand + rugarch/GAS, 2026-07-04)` → `9df9f79` (`5189421d2797`)

## §9M Mechanism sub-panel — 12-asset subset, executed 2026-07-16/17

- **chronos_base HF revision**: `ad294eaacead15db499b740ea4122266dd2a81a2` → `harness/registry.yaml` → `(pinned by hand at Stage 1; ids machine-asserted, hard rule 11)` → `c748942` (`6ae15b816a54`)
- **moirai_1_1 HF revision**: `0c24ab99db2c1a70ea2a0fc03bf113329772ac64` → `harness/registry.yaml` → `(pinned by hand at Stage 1; ids machine-asserted, hard rule 11)` → `c748942` (`6ae15b816a54`)
- **chronos_base pretraining-cutoff upper bound**: `2025-11-21` → `harness/registry.yaml` → `(pinned by hand at Stage 1; ids machine-asserted, hard rule 11)` → `c748942` (`6ae15b816a54`)
- **moirai_1_1 pretraining-cutoff upper bound**: `2025-01-21` → `harness/registry.yaml` → `(pinned by hand at Stage 1; ids machine-asserted, hard rule 11)` → `c748942` (`6ae15b816a54`)
- **chronos_base E0 probe: categorical support / tail centers below q05**: `50 / 3` → `harness/registry.yaml` → `(pinned by hand at Stage 1; ids machine-asserted, hard rule 11)` → `c748942` (`6ae15b816a54`)
- **moirai_1_1 analytic quantiles exposed by official path**: `False` → `harness/registry.yaml` → `(pinned by hand at Stage 1; ids machine-asserted, hard rule 11)` → `c748942` (`6ae15b816a54`)
- **chronos_base E0@1% mean hit (12A)**: `32.4%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **chronos_bolt E0@1% mean hit (12A)**: `12.6%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **chronos_2 E0@1% mean hit (12A)**: `1.3%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **chronos_base E0−E2 spread @1% (12A, pp)**: `9.6` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **chronos_bolt E0−E2 spread @1% (12A, pp)**: `11.3` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **chronos_2 E0−E2 spread @1% (12A, pp)**: `0.0` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **chronos_base UC pooled pass by rule (12A, E0/E1/E2/E3)**: `0%/0%/0%/0%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **chronos_bolt UC pooled pass by rule (12A, E0/E1/E2/E3)**: `0%/0%/47%/19%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **chronos_2 UC pooled pass by rule (12A, E0/E1/E2/E3)**: `83%/0%/56%/92%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **chronos_base UC pass, all rules pooled (12A)**: `0/144` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **moirai_1_1 UC pass, all rules pooled (12A)**: `23/144` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **Moirai migration E0@1% mean hit (1.1 vs 2.0, 12A)**: `1.86% vs 1.38%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **Moirai migration UC pooled E0 (1.1 vs 2.0, 12A)**: `8% vs 42%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **Moirai migration UC pooled E2 (1.1 vs 2.0, 12A)**: `47% vs 64%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **Moirai migration UC pooled E3 (1.1 vs 2.0, 12A)**: `8% vs 89%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **Moirai migration CC pooled E3 (1.1 vs 2.0, 12A)**: `8% vs 78%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **Moirai migration mean FZ0 E2@1% (1.1 vs 2.0, 12A)**: `2.000 vs 1.955` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **Moirai migration mean qs E0@5% (1.1 vs 2.0, 12A)**: `0.2993 vs 0.2855` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **moirai_1_1 per-α E0 mean hit (12A, 1/2.5/5%)**: `1.9% / 4.1% / 7.5%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **chronos_base per-α E0 mean hit (12A, 1/2.5/5%)**: `32.4% / 33.5% / 34.7%` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)
- **H2 verdict (frozen 3-criterion convention)**: `NOT SUPPORTED` → `docs/mechanism_h2_results.md` → `harness/run_mechanism_h2.py` → `eccbf26` (`9455b604b7ed`)
- **chronos_base tail support below q05 (median distinct token values)**: `2` → `docs/mechanism_h2_results.md` → `harness/run_mechanism_h2.py` → `eccbf26` (`9455b604b7ed`)
- **chronos_base tail nearest-nbr gap (median, ret units)**: `0.01457` → `docs/mechanism_h2_results.md` → `harness/run_mechanism_h2.py` → `eccbf26` (`9455b604b7ed`)
- **Deep-tail resolution sweep, median distinct VaR (base/bolt/c2)**: `3 / 1 / 34` → `docs/mechanism_h2_results.md` → `harness/run_mechanism_h2.py` → `eccbf26` (`9455b604b7ed`)
- **Resolution ratio chronos_2 / chronos_base**: `11.3x` → `docs/mechanism_h2_results.md` → `harness/run_mechanism_h2.py` → `eccbf26` (`9455b604b7ed`)
- **chronos_base synth VaR fidelity mean|err| E0 (1/2.5/5%)**: `3.600 / 2.810 / 2.218` → `docs/mechanism_h2_results.md` → `harness/run_mechanism_h2.py` → `eccbf26` (`9455b604b7ed`)
- **Forensic: chronos_base spx q01 vs realized ctx q01 (t=5343)**: `-0.093 vs -4.956` → `docs/chronos_base_forensic.md` → `docs/chronos_base_forensic.py (independent official-pipeline probe; signer-ordered forensic)` → `54826c6` (`5b54f5832aa7`)
- **Forensic: chronos_base btc q01 vs realized (t=2410)**: `-0.568 vs -9.099` → `docs/chronos_base_forensic.md` → `docs/chronos_base_forensic.py (independent official-pipeline probe; signer-ordered forensic)` → `54826c6` (`5b54f5832aa7`)
- **Forensic: chronos_base nvda q01 vs realized (t=5343)**: `-0.651 vs -7.979` → `docs/chronos_base_forensic.md` → `docs/chronos_base_forensic.py (independent official-pipeline probe; signer-ordered forensic)` → `54826c6` (`5b54f5832aa7`)
- **Forensic probe: sine (pred median, truth, spread)**: `-2.931 vs -2.939, spread 0.023` → `results/mechanism/chronos_base_controlled_probes.csv` → `harness/run_chronos_base_decoding.py (--controlled-probes, seeded; double-run byte-identical)` → `6332f59` (`61edc602a334`)
- **Forensic probe: trend spread**: `0.656` → `results/mechanism/chronos_base_controlled_probes.csv` → `harness/run_chronos_base_decoding.py (--controlled-probes, seeded; double-run byte-identical)` → `6332f59` (`61edc602a334`)
- **Forensic probe: noise spread vs realized q10**: `0.545 (realized q10 -1.61)` → `results/mechanism/chronos_base_controlled_probes.csv` → `harness/run_chronos_base_decoding.py (--controlled-probes, seeded; double-run byte-identical)` → `6332f59` (`61edc602a334`)
- **Forensic verdict (branch)**: `WITHDRAWN: native sampling-interface scale pathology (P1-5 supersession banner; decoding-configuration audit)` → `docs/chronos_base_forensic.md` → `docs/chronos_base_forensic.py (independent official-pipeline probe; signer-ordered forensic)` → `54826c6` (`5b54f5832aa7`)
- **Mechanism panel EXECUTED (supersedes the §9 GAP entry above)**: `12-asset subset, 2 heads, E0-E3, MC battery — results/mechanism/` → `docs/mechanism_contrasts.md` → `harness/run_mechanism_contrasts.py` → `1d9cfcb` (`da2f7cb4f7f2`)

## §D8 anchor closures — panel composition, protocol constants, table facts

- **Asset classes in the 32-asset panel**: `6` → `data/data_manifest.yaml` → `data/build.py` → `b04bf5a` (`698566aaeea9`)
- **Panel class breakdown (per class, evaluated assets)**: `equity_index 8, equity_single 8, commodity 4, crypto 4, fx 4, rates 4` → `data/data_manifest.yaml` → `data/build.py` → `b04bf5a` (`698566aaeea9`)
- **FX quarantine bad-tick vintage**: `2008` → `data/data_manifest.yaml` → `data/build.py` → `b04bf5a` (`698566aaeea9`)
- **Sampling-head production draw count S**: `1000` → `docs/pilot_d3_calibration.md` → `harness/run_pilot_d3.py` → `c4bb08c` (`45154dd55f50`)
- **E2 ν refit cadence (windows)**: `21` → `harness/run_grid_extract.py` → `harness/run_grid_extract.py` → `9ace69d` (`64b5ccdd1b4b`)
- **lag_llama+F1 repaired coverage (matched)**: `69%` → `docs/stage5_prescription.md` → `harness/run_prescription.py` → `eccbf26` (`24326615cce7`)
- **tab:es precision-fragile cells (every displayed row)**: `0 (all 11 rows)` → `paper/draft/generated/tab_es_precision.tex` → `paper/figures/make_tables.py` → `c520de3` (`3703f63827d2`)
- **Forensic probe predicted medians (trend / noise)**: `+25.48 / -0.050` → `results/mechanism/chronos_base_controlled_probes.csv` → `harness/run_chronos_base_decoding.py (--controlled-probes, seeded; double-run byte-identical)` → `6332f59` (`61edc602a334`)

## §D9 v1.2 sensitivity — decomposition CIs, criterion grid, bare counts

- **H MC-UC+CC pass counts (C2/M2/LL/Bolt/TFM, matched)**: `22/32 / 23/32 / 26/32 / 23/32 / 23/32` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **Criterion grid: combos with H passing on all five heads**: `14 of 36` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **C2+H criticality at margin 5pp (threshold vs pass rate)**: `73.1%` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **Assets with scale>loc per head (C2/M2/LL/Bolt/TFM, of 32)**: `22 / 24 / 30 / 22 / 21` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **lag_llama scale term 95% CI (cluster bootstrap)**: `[+0.124, +0.179]` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **lag_llama location term 95% CI (cluster bootstrap)**: `[+0.001, +0.018]` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **Cluster bootstrap spec (classes resampled, B, seed)**: `6 asset classes resampled with replacement, B=2000, seed 20260717` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)

## §D10 verification basis by statistic — itemized (stage-7 item 12)

- **kupiec (UC)**: `external fixture: rugarch 1.5.5, exact` → `tests/test_var_tests.py` → `—` → `8330dcc` (`99640b9ae1f2`)
- **christoffersen ind/cc**: `external fixture: rugarch 1.5.5, exact` → `tests/test_var_tests.py` → `—` → `8330dcc` (`99640b9ae1f2`)
- **dq (4 lags)**: `property test: H0 p-values ~ U(0,1), 2000 seeded replications; no dependable external fixture (GAS refs never validated, stage-1 record)` → `tests/test_dq_size.py` → `—` → `f9c24fc` (`6ea268b94f99`)
- **quantile_score (pinball)**: `analytic cases + minimized-at-true-quantile property` → `tests/test_scores.py` → `—` → `8330dcc` (`41f342fcfc50`)
- **fz0**: `external fixture: QuantLet HMD_ES, exact` → `tests/test_scores.py` → `—` → `8330dcc` (`41f342fcfc50`)
- **er (exceedance residuals)**: `external fixture: esback, within bootstrap MC tolerance` → `tests/test_es_tests.py` → `—` → `8330dcc` (`ae3e490f2a02`)
- **as_z2 (Acerbi-Szekely Z2)**: `analytic + sanity tests, MC-calibrated critical values; no authoritative external package` → `tests/test_es_tests.py` → `—` → `8330dcc` (`ae3e490f2a02`)
- **esr (strict ESR)**: `R-engine authoritative: vendored esback 0.3.1 engine, output frozen (SHA-256 in harness/README_R_engines.md); python port pending, skipped by design` → `tests/test_esr.py` → `—` → `b1e9576` (`4829d6180884`)
- **dm (Diebold-Mariano)**: `analytic hand case + antisymmetry/Harvey-correction properties` → `tests/test_dm_mcs_precision.py` → `—` → `8330dcc` (`99f1f09c08db`)
- **mcs (T_max, stationary bootstrap)**: `external fixture: R MCS 0.2.0 superior-set agreement at levels 0.10/0.25 on the frozen input (2026-07-21) + synthetic property tests` → `tests/test_mcs_r_reference.py` → `—` → `c2e3603` (`21546f54f8fd`)
- **n_alpha / precision floor / fragile flag**: `analytic (Pele-Mazurencu formulas) + unit tests` → `tests/test_dm_mcs_precision.py` → `—` → `8330dcc` (`99f1f09c08db`)

## §D11 v1.5 per-α VaR audit at E3 (item 1d)

- **chronos_bolt E3 non-rejection (pooled/1%/2.5%/5%/all-3)**: `19.8% / 34.4% / 18.8% / 6.2% / 6.2%` → `paper/draft/generated/tab_var_audit_alpha.tex` → `—` → `eccbf26` (`faee8d9954a1`)
- **chronos_2 E3 non-rejection (pooled/1%/2.5%/5%/all-3)**: `84.4% / 87.5% / 78.1% / 87.5% / 75.0%` → `paper/draft/generated/tab_var_audit_alpha.tex` → `—` → `eccbf26` (`faee8d9954a1`)
- **timesfm_2_5 E3 non-rejection (pooled/1%/2.5%/5%/all-3)**: `69.8% / 84.4% / 68.8% / 56.2% / 40.6%` → `paper/draft/generated/tab_var_audit_alpha.tex` → `—` → `eccbf26` (`faee8d9954a1`)
- **moirai_2_0 E3 non-rejection (pooled/1%/2.5%/5%/all-3)**: `91.7% / 90.6% / 87.5% / 96.9% / 81.2%` → `paper/draft/generated/tab_var_audit_alpha.tex` → `—` → `eccbf26` (`faee8d9954a1`)
- **lag_llama E3 non-rejection (pooled/1%/2.5%/5%/all-3)**: `0.0% / 0.0% / 0.0% / 0.0% / 0.0%` → `paper/draft/generated/tab_var_audit_alpha.tex` → `—` → `eccbf26` (`faee8d9954a1`)

## §D12 v1.5 robustness — decomposition CIs + DM Newey-West (items 10a/10b)

- **LOO scale means all positive (5 heads)**: `True` → `docs/v15_robustness.md` → `—` → `8bfc6f0` (`154211fe9d27`)
- **Unclustered scale CI excludes zero**: `chronos_2=yes, moirai_2_0=yes, lag_llama=yes, chronos_bolt=yes, timesfm_2_5=yes` → `docs/v15_robustness.md` → `—` → `8bfc6f0` (`154211fe9d27`)
- **DM decisions significant at 5% (frozen / NW / flips)**: `310 / 270 / 42` → `docs/v15_robustness.md` → `—` → `8bfc6f0` (`154211fe9d27`)
- **DM flips by direction (to-significant / to-insignificant)**: `1 / 41` → `docs/v15_robustness.md` → `—` → `8bfc6f0` (`154211fe9d27`)
- **NW-significant F1-vs-GARCH-EVT rows favoring F1**: `2` → `docs/v15_robustness.md` → `—` → `8bfc6f0` (`154211fe9d27`)

## §D13 v1.6 additions — margin narrative, ctx ablation, ESR grid, GE self-check, IQR

- **Margin-5pp failing heads (pass rates)**: `chronos_2 at 69%; moirai_2_0 at 72%; chronos_bolt at 72%; timesfm_2_5 at 72%` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **First-fail margins (C2 / M2+Bolt+TFM)**: `9.38 / 6.25` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **Bolt E0 clamp signature under ctx {256,512,2048} (q01=q025=q05)**: `13.39% / 14.33% (identical across the three requested levels per row)` → `docs/pilot_d6_ctx_ablation.md` → `—` → `f88b3a1` (`b7773bf9da77`)
- **ESR E3 non-rejection by head (Bolt/C2/TFM/M2/LL, % of 96 cells)**: `38.5 / 41.7 / 82.3 / 85.4 / 11.5` → `paper/draft/generated/tab_esr.tex` → `paper/figures/make_tables.py` → `1d9cfcb` (`751f9a8bfea6`)
- **GARCH-EVT criterion self-check (pass rate / (b) frac)**: `72% / 0.88` → `paper/draft/generated/tab_success.tex` → `—` → `47ba6e2` (`0e1e8fa686e5`)
- **tau_rec IQR at α=5% (M2/FHS/C2/GJR/Bolt/TFM/LL)**: `24--31 / 22--32 / 24--30 / 24--37 / 24--40 / 24--38 / 30--37` → `paper/draft/generated/tab_taurec.tex` → `—` → `eccbf26` (`e894773b62de`)

## §D14 v2.0 Phase-2 — decoding configs, cluster-valid inference, control arms, ξ recount, CAViaR, Pele completion

- **Chronos-base decoding mean hit @1% (topk50/topk500/untruncated, 12A)**: `32.34% / 7.59% / 1.41%` → `results/mechanism/chronos_base_topk.csv` → `harness/run_chronos_base_decoding.py` → `7e69616` (`6472017e08a3`)
- **Chronos-base decoding UC pass @1% (topk50/topk500/untruncated)**: `0/12 / 0/12 / 6/12` → `results/mechanism/chronos_base_topk.csv` → `harness/run_chronos_base_decoding.py` → `7e69616` (`6472017e08a3`)
- **Chronos-base untruncated UC pass (1%/2.5%/5%)**: `6/12 / 10/12 / 11/12` → `results/mechanism/chronos_base_topk.csv` → `harness/run_chronos_base_decoding.py` → `7e69616` (`6472017e08a3`)
- **Sampled-E0 cross-validation (exact topk50 vs delivered sampled mean hit @1%)**: `32.34% vs 32.40%` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **Probe spx q01 by decoding config (topk50/topk500/untrunc vs realized ctx / bolt clamp)**: `-0.093 / -1.488 / -2.475 vs -4.956 / -1.120` → `results/mechanism/chronos_base_probes.csv` → `harness/run_chronos_base_decoding.py` → `7e69616` (`2b18bd704584`)
- **CR1 scale t (C2/M2/LL/Bolt/TFM)**: `+1.63 / +2.80 / +9.71 / +2.33 / +2.70` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **Scale t(5) CI excludes zero**: `3/5 (moirai_2_0, lag_llama, timesfm_2_5)` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **Scale restricted-WCR p (C2/M2/LL/Bolt/TFM; attainable floor 2/64)**: `0.188 / 0.125 / 0.031 / 0.156 / 0.125` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **Scale WCR zero-exclusion**: `1/5 (lag_llama)` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **Location WCR p (C2/M2/LL/Bolt/TFM)**: `0.438 / 0.219 / 0.125 / 0.438 / 0.625` → `docs/v12_sensitivity.md` → `—` → `b172b5d` (`70f3f8efd90c`)
- **Location max CR1 t**: `1.97` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **Scale-term zero-exclusion by procedure (delivered cluster CI / CR1-t(5) / restricted WCR)**: `4/5 / 3/5 / 1/5` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **Matched scale-term means all positive (C2/M2/LL/Bolt/TFM)**: `+.0266 / +.0483 / +.1520 / +.0342 / +.0525` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **Control-arm mean FZ0 (garch_evt / zero_loc / const_loc; α=1% matched)**: `1.8467 / 1.8457 / 1.8460` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **X4 interaction terms (C2/M2/LL/Bolt/TFM)**: `−.0034/+.0011/+.0096/−.0010/+.0019` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **X4 UC+CC pass (C2/M2/LL/Bolt/TFM); zero/const_loc; garch_evt**: `78/59/69/66/56; 69/69; garch_evt 72` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **ξ≥1 windows old → new**: `586 → 85` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **Boundary-artifact share of old ξ≥1 (β<1e-6 MLE fits)**: `490 (83.6%)` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **ξ≥1 per-asset (new)**: `dgs30 61, dgs2 23, dgs10 1, dgs5 0` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **dgs30 true ξ≥1 day share vs the battery ES-validity margin**: `2.3% > 1%` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **E3 day sets unified (dgs2/dgs5/dgs10/dgs30 → all)**: `2487/2597/2601/2229 → 2,625` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **E3 decision flips (UC / CC)**: `UC 11 (all pass→fail) / CC 12 (10 pass→fail, 2 fail→pass)` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **E3 ES aggregability (32 asset x 5 head x 3 α grid)**: `15 NaN cells (dgs30); complete-panel assets 31 of 32` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **Table-2 E3 CC pass one-decimal (Bolt/C2/TFM/M2/LL, old→new)**: `16.7→16.7 / 72.9→70.8 / 50.0→46.9 / 70.8→67.7 / 0→0` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **CAViaR-SAV UC pass (pooled old→new; per-α new)**: `94.8% (α=1% 93.8; 2.5% 93.8; 5% 96.9)` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **CAViaR diagnostics totals**: `32 assets: failed_fits 0, invalid_days 0, min valid_starts 8/8` → `results/stage4/caviar_diag_summary.csv` → `harness/run_caviar_diag_summary.py` → `0004192` (`1d2b23fcf610`)
- **CAViaR day-set identity vs garch_t**: `96/96` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **Baseline pooled UC band (FHS/CAViaR/HS-500/GJR/GARCH-t, % of 96)**: `100.0 / 94.8 / 82.3 / 67.7 / 65.6` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **Pele-completion violation rates (32 TimesFM-2.5 grids; mean/min/max)**: `1.88% / 0.91% / 2.69%` → `results/stage4/pele_completion.csv` → `harness/run_pele_completion.py` → `dd89d3c` (`f5917f4a4722`)
- **Pele-completion window count (32 assets)**: `87013` → `results/stage4/pele_completion.csv` → `harness/run_pele_completion.py` → `dd89d3c` (`f5917f4a4722`)
- **E1/E2 anchor pairs (primary; TimesFM decile fallback)**: `(0.25, 0.75) primary; (0.2, 0.8) fallback` → `erules/rules.py` → `erules/rules.py` → `4ee13e7` (`a7e3fabffeed`)
- **H1 registered text (proposal §3.6, verbatim)**: `- **H1(有界分位头)**:α≤2.5% 深尾输出为外推假设的函数而非模型知识的函数;E0–E3 间的发散度按头型交叉分类;族内对照 Chronos vs Chronos-Bolt、Moirai-1.1 vs 2.0 隔离头因素。` → `docs/TSFM_tails_proposal_v2_GO.md` → `(frozen design document)` → `b7dbc8b` (`d45928ddbb40`)
- **Chronology dates (CHANGELOG): Stage-4 final-table rulings / Stage-5 design approval / mechanism sub-panel authorization**: `2026-07-07 / 2026-07-07 / 2026-07-16` → `CHANGELOG.md` → `(governance log)` → `3296a55` (`67f52d172c66`)
- **Threshold-timing element (stage5_design §A, verbatim)**: `thresholds set knowing Stage-4 comparator values but before any repair outcome` → `docs/stage5_design.md` → `(design doc v2.1, signer-approved 2026-07-07)` → `d282d05` (`b564e01b18f6`)
- **Mechanism dgs2/dgs10 re-extraction cell flips (decoupled E3 + new ctxfits)**: `UC 2 (both moirai_1_1: dgs10 α=1% fail→pass p .0500→.2710; dgs2 α=2.5% pass→fail p .6133→.0469) / CC 0; pooled per-(head, rule) UC/CC pass rates unchanged` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)
- **TimesFM-2.5 mean 1% violation by rule (E1/E2/E3, x nominal)**: `3.40x / 1.04x / 1.12x` → `docs/v20_impact_report.md` → `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` → `f610466` (`84bf90dc2f8e`)

## §D15 v2.0 Phase-3 — day sets, DQ/CC, UC power, cluster CIs, dual-level MCS, regime CIs, cutoff shares, EWMA/FHS

- **Fetched series composition (total = eval + quarantined FX + vol-index)**: `36 = 32 + 2 + 2` → `data/data_manifest.yaml` → `data/build.py` → `b04bf5a` (`698566aaeea9`)
- **Bolt/Lag FX-class exposure component (disclosed corpora)**: `exchange_rate — 8-currency daily benchmark 1990–2010, pre-OOS` → `docs/giftevalpretrain_overlap.md` → `(D2 audit; machine-sourced from HF dataset APIs)` → `3e05e42` (`48ce70b2b962`)
- **Late-start TSFM vs baseline day sets (csi300; eth/xrp; btc/ltc)**: `2639/2180; 2647/2159; 3796/3308` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Matched-date sensitivity (cells; flips; toward passing/failing)**: `285 cells; 12 flips; 9/3` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **DQ pass (asym) E3 — healthy heads (C2 / Moirai)**: `42.7% / 40.6%` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **DQ pass (asym) — leading baselines (GJR/FHS/GARCH-t/CAViaR)**: `58.3% / 53.1% / 50.0% / 45.8%` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **CC pass (MC) E3 — healthy heads (C2 / Moirai)**: `76.0% / 74.0%` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **HS-500 conditional record (DQ pass / CC pass)**: `2.1% / 29.2%` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **MC-UC acceptance region (n=2639, a=1%, production B=10,000)**: `17..37 violations = [0.644x, 1.402x]` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Exact-binomial UC power vs 1.25x nominal (n=2639, a=1%)**: `0.212` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **FHS a=1% multiples (mean; per-asset range)**: `1.07x; 0.76x to 1.28x` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **GJR-t a=1% mean multiple**: `1.27x` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Healthy-head E3 a=1% mean multiples (C2 / Moirai)**: `1.01x / 0.97x` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Cluster-bootstrap 95% CI, C2-Moirai E3 pass-rate diff**: `[-18.8, +3.1]pp` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Cluster-bootstrap 95% CIs vs TimesFM (C2-TFM / Moirai-TFM, E3)**: `[+4.2, +25.0]pp / [+11.5, +33.3]pp` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **E2 pairwise cluster CIs (all three pairs)**: `[-27.1, +19.8]pp; [-24.0, +3.1]pp; [-26.0, +11.5]pp — all include 0` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Within-asset ICC / design effect / effective cells (E3, three heads)**: `0.54/2.07/46; 0.19/1.39/69; 0.17/1.34/71` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Moirai E3 UC rate vs the seven baselines (above five, below two)**: `91.7% > hs500 82.3%, gjr 67.7%, garch 65.6%, ewma 43.8%, hs250 54.2%; < fhs 100.0%, caviar 94.8%` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **EWMA vs FHS controlled contrast (UC a=1%; mean FZ0 a=1%)**: `2/32 vs 32/32; 1.9604 vs 1.8140` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **EWMA ESR non-rejection**: `4/32 = 12.5% (P1-4 option A: alpha=1% caliber; pooled 8/96 = 8.3%)` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Asym-vs-MC UC decisions (differ / total; direction)**: `15/2,496; 0 pass->fail, 15 fail->pass` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Asym-vs-MC CC decisions at a=1% (differ/832; direction)**: `33/832; 33 asym-accept->MC-reject, 0 reverse` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **n-alpha floors (panel min / matched min; cells below 10)**: `21.59 / 16.59; 0 / 0` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Lag-Llama F1-vs-GARCH-EVT DM at a=1% (NW headline; baseline variance)**: `18/32 favor the comparator, 0/32 favor F1; baseline 19/32, 0/32` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Repair MCS at the 25% level, a=1% (F3 / F4 / H in-MCS shares)**: `35% / 41% / 97%` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Raw recovery medians a=5% (Moirai / FHS / C2; native half-day)**: `24.5 / 24.0 / 25.5` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Episode-bootstrap 95% CIs for medians a=5% (Moirai / FHS)**: `[24, 34] / [22, 38]` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Pretraining-bound exposure shares (Bolt/C2/Moirai/TFM/Lag; of asset-days)**: `94.07% / 93.49% / 94.24% / 74.73% / 73.91% of 87013` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Chronos-2 dual-rule post-bound sample (arXiv bound -> revision bound)**: `5,662 -> 641` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Second-cluster vs weak baselines mean FZ0 (EWMA / HS-500)**: `1.960 / 2.018` → `paper/draft/generated/tab_es_precision.tex` → `paper/figures/make_tables.py` → `c520de3` (`3703f63827d2`)
- **EWMA MCS membership at the 25% level (a=1%/2.5%/5%)**: `75% / 94% / 94%` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Lag-Llama MCS membership (10% levels; 25% levels)**: `50%/22%/9%; 34%/6%/6%` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Chronos-2 MCS membership at the 25% level (a=2.5% / 5%)**: `81% / 66%` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **GARCH sigma vs rugarch measured relative difference**: `0.39%` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Asymptotic DQ size under the iid null (n=2639, a=1%, B=10,000)**: `0.1178` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)

## §D16 v2.1 Phase-3 — empirical-mass E3 (tau_mass disclosure, direction account, seeded-ESR round)

- **E3 tau_mass — continuous series (28 assets)**: `52/512 = 0.1015625, constant over all 28 continuous assets; zero threshold ties` → `results/stage4/e3_diagnostics/spx.csv` → `harness/run_grid_extract.py (per-window E3 diagnostics)` → `ee25dfb` (`1fd853545835`)
- **E3 tau_mass — rate series (per-asset means; panel min; dgs30 windows)**: `dgs2 0.0846, dgs5 0.0882, dgs10 0.0870, dgs30 0.0822; panel min 0.0546875 (dgs10); dgs30 per-window 0.061-0.102, 5.4% of windows > 0.10` → `results/stage4/e3_diagnostics/dgs30.csv` → `harness/run_grid_extract.py (per-window E3 diagnostics)` → `ee25dfb` (`61eeb42f0bdf`)
- **Repair-side threshold mass (TAU_U=0.10 exact on W=500 calib windows)**: `8 assets, 17,268 windows` → `docs/e3_tail_mass_ruling.md` → `(signer ruling 2026-08-02, verbatim; protocol correction)` → `4fcf4fa` (`40fc0652ba87`)
- **Rate-asset direction account (20 cells, E3 @1%: no-reject / reject-low / reject-high)**: `11 / 5 / 4 (reject-high all lag_llama)` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Kendall low-tau composition (tau<0.5 cells by alpha; @1% assets)**: `8 cells: 4 @1%, 2 @2.5%, 2 @5%; @1% assets btc, eth, nvda, xrp` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)
- **Moirai-2.0 E0 mean violation @1% — 32-asset main panel**: `1.39%` → `docs/v20_phase3_analyses.md` → `harness/run_p3_stats.py` → `8bfc6f0` (`c19b8212c526`)

## Provenance ledger — every source file (hard rule 11)

| source file | generating script | commit hash | sha256-12 |
|---|---|---|---|
| `CHANGELOG.md` | `(governance log)` | `3296a55` | `67f52d172c66` |
| `data/data_manifest.yaml` | `data/build.py` | `b04bf5a` | `698566aaeea9` |
| `docs/TSFM_tails_proposal_v2_GO.md` | `(frozen design document)` | `b7dbc8b` | `d45928ddbb40` |
| `docs/chronos_base_forensic.md` | `docs/chronos_base_forensic.py (independent official-pipeline probe; signer-ordered forensic)` | `54826c6` | `5b54f5832aa7` |
| `docs/e3_tail_mass_ruling.md` | `(signer ruling 2026-08-02, verbatim; protocol correction)` | `4fcf4fa` | `40fc0652ba87` |
| `docs/giftevalpretrain_overlap.md` | `(D2 audit; machine-sourced from HF dataset APIs)` | `3e05e42` | `48ce70b2b962` |
| `docs/mechanism_contrasts.md` | `harness/run_mechanism_contrasts.py` | `1d9cfcb` | `da2f7cb4f7f2` |
| `docs/mechanism_h2_results.md` | `harness/run_mechanism_h2.py` | `eccbf26` | `9455b604b7ed` |
| `docs/pilot_d3_calibration.md` | `harness/run_pilot_d3.py` | `c4bb08c` | `45154dd55f50` |
| `docs/pilot_d6_ctx_ablation.md` | `—` | `f88b3a1` | `b7773bf9da77` |
| `docs/stage4_btc_did.md` | `harness/run_grid_arbitration.py` | `a88a364` | `5e078c26d0d9` |
| `docs/stage4_btc_did_split.md` | `harness/run_btc_did_split.py` | `d49e1ed` | `429645f9112b` |
| `docs/stage4_final_table.md` | `harness/run_grid_final.py` | `1d9cfcb` | `d8c1cefad822` |
| `docs/stage4_kendall_tau.md` | `harness/run_grid_arbitration.py` | `eccbf26` | `45c4afbd7ddd` |
| `docs/stage4_synth_arbitration.md` | `synth/run_arbitration.py` | `eccbf26` | `f4feb16a40bb` |
| `docs/stage5_design.md` | `(design doc v2.1, signer-approved 2026-07-07)` | `d282d05` | `b564e01b18f6` |
| `docs/stage5_episode_set.md` | `regimes/rules.py (episode enumeration over frozen vol data)` | `70c0729` | `43f7c3323378` |
| `docs/stage5_prescription.md` | `harness/run_prescription.py` | `eccbf26` | `24326615cce7` |
| `docs/stage5_regime.md` | `harness/run_regime_analysis.py` | `eccbf26` | `6721198ec040` |
| `docs/stage5_strata.md` | `harness/run_vol_strata.py` | `eccbf26` | `dd76a3d7066b` |
| `docs/stage5_xi_incidence.md` | `harness/run_xi_incidence.py` | `e77f63d` | `1b304b3d4946` |
| `docs/v12_sensitivity.md` | `—` | `b172b5d` | `70f3f8efd90c` |
| `docs/v15_robustness.md` | `—` | `8bfc6f0` | `154211fe9d27` |
| `docs/v20_impact_report.md` | `(v2.0 Phase-1 impact report — script-read values; authorization = docs/external_review_9_adjudication.md)` | `f610466` | `84bf90dc2f8e` |
| `docs/v20_phase3_analyses.md` | `harness/run_p3_stats.py` | `8bfc6f0` | `c19b8212c526` |
| `erules/rules.py` | `erules/rules.py` | `4ee13e7` | `a7e3fabffeed` |
| `fixes/arms.py` | `fixes/arms.py` | `4ee13e7` | `8108cd78156b` |
| `harness/registry.yaml` | `(pinned by hand at Stage 1; ids machine-asserted, hard rule 11)` | `c748942` | `6ae15b816a54` |
| `harness/rolling.py` | `harness/rolling.py` | `9df9f79` | `696155f094c8` |
| `harness/run_grid_extract.py` | `harness/run_grid_extract.py` | `9ace69d` | `64b5ccdd1b4b` |
| `harness/run_regime_analysis.py` | `harness/run_regime_analysis.py` | `9ace69d` | `5bc0948ed0c9` |
| `paper/build_evidence_pack.py` | `—` | `9ace69d` | `0db393babf22` |
| `paper/draft/generated/tab_es_precision.tex` | `paper/figures/make_tables.py` | `c520de3` | `3703f63827d2` |
| `paper/draft/generated/tab_esr.tex` | `paper/figures/make_tables.py` | `1d9cfcb` | `751f9a8bfea6` |
| `paper/draft/generated/tab_success.tex` | `—` | `47ba6e2` | `0e1e8fa686e5` |
| `paper/draft/generated/tab_taurec.tex` | `—` | `eccbf26` | `e894773b62de` |
| `paper/draft/generated/tab_var_audit_alpha.tex` | `—` | `eccbf26` | `faee8d9954a1` |
| `precision/diagnostics.py` | `—` | `16ba8c5` | `ef1c1eeff653` |
| `regimes/rules.py` | `regimes/rules.py` | `70c0729` | `19b33adf3a45` |
| `results/mechanism/chronos_base_controlled_probes.csv` | `harness/run_chronos_base_decoding.py (--controlled-probes, seeded; double-run byte-identical)` | `6332f59` | `61edc602a334` |
| `results/mechanism/chronos_base_probes.csv` | `harness/run_chronos_base_decoding.py` | `7e69616` | `2b18bd704584` |
| `results/mechanism/chronos_base_topk.csv` | `harness/run_chronos_base_decoding.py` | `7e69616` | `6472017e08a3` |
| `results/stage4/caviar_diag_summary.csv` | `harness/run_caviar_diag_summary.py` | `0004192` | `1d2b23fcf610` |
| `results/stage4/e3_diagnostics/dgs30.csv` | `harness/run_grid_extract.py (per-window E3 diagnostics)` | `ee25dfb` | `61eeb42f0bdf` |
| `results/stage4/e3_diagnostics/spx.csv` | `harness/run_grid_extract.py (per-window E3 diagnostics)` | `ee25dfb` | `1fd853545835` |
| `results/stage4/pele_completion.csv` | `harness/run_pele_completion.py` | `dd89d3c` | `f5917f4a4722` |
| `results/stage4/run_manifest.yaml` | `harness/freeze_run_manifest.py` | `3a6f980` | `3cda980734b7` |
| `tests/fixtures/README.md` | `(fixtures documentation; kupiec_p divergence verified by hand + rugarch/GAS, 2026-07-04)` | `9df9f79` | `5189421d2797` |
| `tests/fixtures/mcneilfrey_evt_reference.csv` | `tests/fixtures/make_mcneilfrey_references.R (evir 1.7.4)` | `8ce91fd` | `06eb7fcb3043` |
| `tests/fixtures/mcneilfrey_garch_reference.csv` | `tests/fixtures/make_mcneilfrey_references.R (rugarch 1.5.5)` | `8ce91fd` | `2472eee54b02` |
| `tests/fixtures/r_versions.txt` | `tests/fixtures/make_r_references.R` | `8330dcc` | `294e39c1edc6` |
| `tests/test_dm_mcs_precision.py` | `—` | `8330dcc` | `99f1f09c08db` |
| `tests/test_dq_size.py` | `—` | `f9c24fc` | `6ea268b94f99` |
| `tests/test_es_tests.py` | `—` | `8330dcc` | `ae3e490f2a02` |
| `tests/test_esr.py` | `—` | `b1e9576` | `4829d6180884` |
| `tests/test_fixes.py` | `—` | `358a51f` | `0dd08eafaed3` |
| `tests/test_garch_evt_reference.py` | `tests/test_garch_evt_reference.py` | `4ee13e7` | `24167661e6b4` |
| `tests/test_mc_critical.py` | `—` | `9190dcb` | `20850825189d` |
| `tests/test_mcs_r_reference.py` | `—` | `c2e3603` | `21546f54f8fd` |
| `tests/test_scores.py` | `—` | `8330dcc` | `41f342fcfc50` |
| `tests/test_var_tests.py` | `—` | `8330dcc` | `99640b9ae1f2` |

Build-time integrity: 61 source files, 338 cited numbers across 21 sections.
