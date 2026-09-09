# Vendored ESR engine (gate condition 8, signer decision ②):
#   Rscript harness/run_esr_engine.R <lib_path> [<stage4_root>]
# <stage4_root> defaults to ./results/stage4 relative to the working
# directory (path-parameterized 2026-07-21, stage-7 prep; the computation is
# byte-identical to the frozen generation run — see CHANGELOG).
# Reads <stage4_root>/esr_inputs/index.csv (+ per-cell CSVs), runs
# esback::esr_backtest version 1 (Bayer-Dimitriadis "strict ESR"; two-sided
# asymptotic p, B=0; esreg's iterated-local-search point estimation draws
# from R's RNG, so each cell is seeded via cell_seed() below — the output is
# exactly reproducible run-to-run), writes results/stage4/esr_results.csv.
# R/esback is a GENERATION-TIME dependency only; outputs are frozen and consumed
# by the Python side (final-table merge). esback version recorded per row.
# Run discipline: per-cell progress to stdout, measured-rate ETA lines every 50.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) .libPaths(c(args[1], .libPaths()))
suppressMessages(library(esback))

# Per-cell RNG seeding (v2.1 Phase 3; adjudication §8 ISSUE i): esreg's
# iterated local search is stochastic, so an unseeded engine is not
# reproducible end-to-end. Derivation: seed_i = (20260803 + H(cell_id))
# mod (2^31 - 1), where H is the base-31 polynomial byte hash
# H <- (H*31 + utf8_byte) mod (2^31 - 1) over the cell_id string —
# platform-independent, order-sensitive, and exact in double arithmetic
# (every intermediate < 2^36 < 2^53). Base seed 20260803 = ruling date.
cell_seed <- function(cid, base = 20260803) {
  m <- 2147483647
  h <- 0
  for (b in utf8ToInt(cid)) h <- (h * 31 + b) %% m
  as.integer((base + h) %% m)
}

root <- if (length(args) >= 2) args[2] else file.path(getwd(), "results", "stage4")
idx <- read.csv(file.path(root, "esr_inputs", "index.csv"), stringsAsFactors = FALSE)
ver <- as.character(packageVersion("esback"))
n <- nrow(idx)
res <- vector("list", n)
t0 <- Sys.time()

for (i in seq_len(n)) {
  cid <- idx$cell_id[i]
  d <- read.csv(file.path(root, "esr_inputs", paste0(cid, ".csv")))
  set.seed(cell_seed(cid))
  p <- tryCatch({
    out <- esr_backtest(r = d$y, q = d$v, e = d$e, alpha = idx$alpha[i],
                        version = 1, B = 0)
    out$pvalue_twosided_asymptotic
  }, error = function(e) NA_real_)
  res[[i]] <- data.frame(cell_id = cid, forecaster = idx$forecaster[i],
                         erule = idx$erule[i], asset = idx$asset[i],
                         alpha = idx$alpha[i], n = idx$n[i],
                         esr_p = p, esback_version = ver,
                         stringsAsFactors = FALSE)
  if (i %% 50 == 0 || i == n) {
    el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    eta <- el / i * (n - i)
    cat(sprintf("[esr] %d/%d done, %.1f s/cell, ETA %.0f min\n",
                i, n, el / i, eta / 60))
    flush(stdout())
  }
}

out <- do.call(rbind, res)
write.csv(out, file.path(root, "esr_results.csv"), row.names = FALSE)
cat(sprintf("esr engine complete: %d cells, %d NA\n", n, sum(is.na(out$esr_p))))
