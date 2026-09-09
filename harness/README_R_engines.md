# R engines and reference stack (generation-time dependencies)

The Python side consumes frozen CSV outputs; R is needed only to (re)generate
them or the vendored test fixtures. All versions below are machine-sourced
(from the frozen outputs themselves or from the installed libraries).

## ESR engine (`harness/run_esr_engine.R`)

- Command: `Rscript harness/run_esr_engine.R <lib_path> [<stage4_root>]`
  - `<lib_path>`: an R library containing `esback` (prepended to `.libPaths()`).
  - `<stage4_root>`: defaults to `./results/stage4` under the working
    directory. Path-parameterized 2026-07-21 (stage-7 prep).
  - Statistical call: `esback::esr_backtest`, version 1, two-sided
    asymptotic p, B=0. The point-estimation layer (esreg's iterated local
    search) is stochastic, so the engine seeds R's RNG per cell
    (`cell_seed()` in the script: base 20260803 + a base-31 byte hash of
    the cell id, mod 2^31-1); the output is exactly reproducible
    run-to-run. Before v2.1 Phase 3 the engine was unseeded — a real
    reproducibility defect (the earlier "deterministic" claim here was
    wrong; adjudication §8 ISSUE i) — although on any single store the
    resulting p-wobble was engine noise, not a results change.
- Engine stack at generation: R 4.4.2 (2024-10-31); `esback` 0.3.1
  (recorded per row in the frozen output: all 2,016 rows carry
  `esback_version = 0.3.1`).
- Frozen output: `results/stage4/esr_results.csv`, 2,016 cells, SHA-256
  `deacbe0302ce4797261b01a2750eaa122a57813972d2938f4fb097e86f87296b`
  (seeded engine, v2.1 Phase 3; proven byte-identical across two full
  independent runs).
- `esback` install (if absent): CRAN
  `install.packages("esback", lib = "<lib_path>", repos = "https://cloud.r-project.org")`
  (0.3.1 is the current CRAN release; if CRAN has moved on, install 0.3.1
  from the CRAN archive to match the fixture-anchored version).

## Reference-fixture stack (`tests/fixtures/`, vendored local library)

- Local library: `tests/fixtures/rlib_local/` (gitignored; rebuild with the
  commands in `tests/test_garch_evt_reference.py` header).
- Versions (machine-read from the installed library): `rugarch` 1.5.5,
  `evir` 1.7.4; `esback` 0.3.1 fixtures were generated before vendoring and
  are frozen under `tests/fixtures/`.
- The paper's pinned reference stack statement (R 4.4.2, rugarch 1.5.5,
  esback 0.3.1) is anchored at evidence-pack entry ep:238.
