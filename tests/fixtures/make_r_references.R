# Generate R reference outputs for the backtest library (one-time; outputs committed).
#
#   Rscript tests/fixtures/make_r_references.R [lib_path]
#
# Reads  input_series.csv  (frozen; see make_input.py)
# Writes r_references.csv  (long format: forecaster, alpha, test, quantity, value)
#        r_versions.txt    (R + package versions used)
#
# Packages: rugarch (VaRTest: Kupiec UC / Christoffersen CC; ESTest),
#           GAS (BacktestVaR: UC/CC/DQ with 4 lags),
#           esback (esr_backtest 1-3, er_backtest, cc_backtest; Bayer-Dimitriadis).
# Bootstrap-based tests are seeded for exact reproducibility.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) .libPaths(c(args[1], .libPaths()))

suppressMessages({
  library(rugarch)
  library(esback)
})
# GAS is optional (best-effort baseline per proposal §3.1; binary install failing
# 2026-07-04 — GAS reference rows are TODO until it installs, see CHANGELOG).
has_gas <- suppressMessages(requireNamespace("GAS", quietly = TRUE))
if (has_gas) suppressMessages(library(GAS))

here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
inp <- read.csv(file.path(here, "input_series.csv"))

alphas <- c(0.01, 0.025, 0.05)
tags   <- c("01", "025", "05")
rows <- list()
add <- function(fc, a, test, quantity, value) {
  rows[[length(rows) + 1]] <<- data.frame(
    forecaster = fc, alpha = a, test = test, quantity = quantity,
    value = as.numeric(value), stringsAsFactors = FALSE)
}

for (fc in c("oracle", "hs")) {
  for (k in seq_along(alphas)) {
    a <- alphas[k]; tag <- tags[k]
    v <- inp[[paste0("v_", fc, "_", tag)]]
    e <- inp[[paste0("e_", fc, "_", tag)]]
    y <- inp$y
    ok <- !is.na(v) & !is.na(e)
    y <- y[ok]; v <- v[ok]; e <- e[ok]

    # rugarch: Kupiec UC + Christoffersen CC
    vt <- VaRTest(alpha = a, actual = y, VaR = v)
    add(fc, a, "rugarch_VaRTest", "uc_stat", vt$uc.LRstat)
    add(fc, a, "rugarch_VaRTest", "uc_p",    vt$uc.LRp)
    add(fc, a, "rugarch_VaRTest", "cc_stat", vt$cc.LRstat)
    add(fc, a, "rugarch_VaRTest", "cc_p",    vt$cc.LRp)

    # rugarch: ES test (asymptotic + seeded bootstrap)
    set.seed(20260704)
    et <- ESTest(alpha = a, actual = y, ES = e, VaR = v, boot = TRUE, n.boot = 2000)
    add(fc, a, "rugarch_ESTest", "p_asym", et$p.value)
    add(fc, a, "rugarch_ESTest", "boot_p", et$boot.p.value)

    # GAS: UC / CC / DQ(4 lags) — skipped when GAS unavailable (TODO)
    if (has_gas) {
      bv <- BacktestVaR(data = y, VaR = v, alpha = a, Lags = 4)
      add(fc, a, "GAS_BacktestVaR", "uc_stat", bv$LRuc["Test"])
      add(fc, a, "GAS_BacktestVaR", "uc_p",    bv$LRuc["Pvalue"])
      add(fc, a, "GAS_BacktestVaR", "cc_stat", bv$LRcc["Test"])
      add(fc, a, "GAS_BacktestVaR", "cc_p",    bv$LRcc["Pvalue"])
      add(fc, a, "GAS_BacktestVaR", "dq_stat", bv$DQ$stat)
      add(fc, a, "GAS_BacktestVaR", "dq_p",    bv$DQ$pvalue)
      add(fc, a, "GAS_BacktestVaR", "ae",      bv$AE)
    }

    # esback: ESR backtests (asymptotic p-values; no bootstrap)
    for (ver in 1:3) {
      es <- esr_backtest(r = y, q = v, e = e, alpha = a, version = ver, B = 0)
      add(fc, a, paste0("esback_esr", ver), "p2_asym", es$pvalue_twosided_asymptotic)
    }

    # esback: exceedance residuals (seeded bootstrap)
    set.seed(20260704)
    er <- er_backtest(r = y, q = v, e = e, B = 2000)
    add(fc, a, "esback_er", "p2_simple", er$pvalue_twosided_simple)
    add(fc, a, "esback_er", "p1_simple", er$pvalue_onesided_simple)
    add(fc, a, "esback_er", "p2_std",    er$pvalue_twosided_standardized)
    add(fc, a, "esback_er", "p1_std",    er$pvalue_onesided_standardized)

    cc <- cc_backtest(r = y, q = v, e = e, alpha = a)
    add(fc, a, "esback_cc", "p1_simple",   cc$pvalue_onesided_simple)
    add(fc, a, "esback_cc", "p2_simple",   cc$pvalue_twosided_simple)
    add(fc, a, "esback_cc", "p1_general",  cc$pvalue_onesided_general)
    add(fc, a, "esback_cc", "p2_general",  cc$pvalue_twosided_general)
  }
}

out <- do.call(rbind, rows)
write.csv(out, file.path(here, "r_references.csv"), row.names = FALSE)

vers <- c(R.version.string,
          paste("rugarch", as.character(packageVersion("rugarch"))),
          if (has_gas) paste("GAS", as.character(packageVersion("GAS")))
          else "GAS NOT INSTALLED (references TODO — see CHANGELOG 2026-07-04)",
          paste("esback",  as.character(packageVersion("esback"))))
writeLines(vers, file.path(here, "r_versions.txt"))
cat("wrote", nrow(out), "reference rows\n")
print(vers)
