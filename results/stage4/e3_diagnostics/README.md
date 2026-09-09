# e3_diagnostics/

Per-asset, per-window E3 GPD fit diagnostics (`{asset}.csv`), written by
the extraction chain (`erules/rules.py` via `harness/run_grid_extract.py`)
under the fixed-anchor empirical-mass POT convention
(`docs/e3_tail_mass_ruling.md`): one row per rolling window with the
threshold, kept-exceedance count, `tau_mass`, and fitted `(xi, beta)`.

`smallsample_v21.csv` is NOT one of these: it is the v2.1 Phase-1
small-sample validation exhibit (SPX + dgs30 x 5 heads x 3 alpha,
old-vs-new-convention leg comparison; generator =
`harness/run_e3_smallsample_v21.py`, consumer =
`harness/check_e3_rerun_stability.py`). It predates the per-window
schema and lives here for historical adjacency only; do not parse it as a
fit-diagnostic file. (Placement note added v2.1.1, review #11; the file
was deliberately left in place rather than moved so that phase-report and
adjudication paths remain valid.)
