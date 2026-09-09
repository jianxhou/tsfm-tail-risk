# One-shot R reference for the MCS cross-validation (stage-1 registered TODO,
# closed 2026-07-21): runs Bernardi & Catania's `MCS` package on the frozen
# loss input and freezes the superior sets + package version as a fixture.
#   Rscript tests/fixtures/make_mcs_r_reference.R <lib_path>
args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) .libPaths(c(args[1], .libPaths()))
suppressMessages(library(MCS))

fixdir <- dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)))
Loss <- as.matrix(read.csv(file.path(fixdir, "mcs_input_losses.csv")))
ver <- as.character(packageVersion("MCS"))

rows <- list()
for (alpha in c(0.10, 0.25)) {
  set.seed(20260704)
  ssm <- MCSprocedure(Loss = Loss, alpha = alpha, B = 5000,
                      statistic = "Tmax", k = 10, verbose = FALSE)
  show <- ssm@show   # columns: Avg.Loss / p-Value for H_{0,M_k} / MCS p-Value
  for (m in rownames(show)) {
    p <- show[m, "MCS p-Value"]
    rows[[length(rows) + 1]] <- data.frame(
      alpha = alpha, model = m, avg_loss = show[m, "Avg.Loss"], mcs_p = p,
      in_superior = as.integer(p >= alpha),
      mcs_package_version = ver, stringsAsFactors = FALSE)
  }
}
out <- do.call(rbind, rows)
write.csv(out, file.path(fixdir, "mcs_r_reference.csv"), row.names = FALSE)
cat(sprintf("MCS %s reference written: %d superior-set rows\n", ver, nrow(out)))
