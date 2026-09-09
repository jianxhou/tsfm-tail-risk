# A16 external McNeil-Frey reference generator (vendored; run once).
#
# Two independent authorities cross-check the GARCH-EVT comparator (fixes/arms.py
# garch_evt) + its two building blocks (erules POT-GPD tail, fixes/garch_filter):
#   evir 1.7.4  (A. McNeil's own EVT package; riskmeasures = the McNeil-Frey
#                POT VaR/ES estimator)  -> the tail half
#   rugarch 1.5.5 (reference financial GARCH engine)  -> the filter half
#
# Reads the frozen inputs (make_mcneilfrey_input.py) and writes:
#   mcneilfrey_evt_reference.csv   (evir xi,beta,threshold,VaR,ES per alpha, z-scale)
#   mcneilfrey_garch_reference.csv (rugarch params + 1-step mu,sigma)
# Sign convention of the Python code under test: LOWER tail, v/e negative.
# We map to evir's UPPER-tail loss convention via L = -z, threshold = -u.

.libPaths(Sys.getenv("R_LIBS_USER"))
suppressPackageStartupMessages({library(evir); library(rugarch)})

fx <- "tests/fixtures/"

## ---- EVT half: evir riskmeasures on the frozen residual sample ----
z  <- read.csv(paste0(fx, "mcneilfrey_resid_input.csv"))$z
u  <- as.numeric(quantile(z, 0.10, type = 7))   # numpy default = R type 7
L  <- -z                                         # losses -> upper tail
thr <- -u
fit <- gpd(L, threshold = thr)                   # McNeil POT-GPD MLE
n_exceed <- fit$n.exceed
xi   <- as.numeric(fit$par.ests["xi"])
beta <- as.numeric(fit$par.ests["beta"])

alphas <- c(0.05, 0.025, 0.01)
ps     <- 1 - alphas                              # upper-tail cumulative probs
rm     <- riskmeasures(fit, ps)                   # cols: p, quantile, sfall (loss scale)
evt <- data.frame(
  alpha     = alphas,
  xi_evir   = xi,
  beta_evir = beta,
  u         = u,                                  # z-scale lower threshold
  n_exceed  = n_exceed,
  n         = length(z),
  var_evir  = -rm[, "quantile"],                  # back to z-scale (negative)
  es_evir   = -rm[, "sfall"]
)
write.csv(evt, paste0(fx, "mcneilfrey_evt_reference.csv"), row.names = FALSE)

## ---- filter half: rugarch GARCH(1,1)-t, constant mean, 1-step ahead ----
ctx <- read.csv(paste0(fx, "mcneilfrey_garch_input.csv"))$logret
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "std")
gfit <- ugarchfit(spec, data = ctx, solver = "hybrid",
                  fit.control = list(scale = 0))
fc   <- ugarchforecast(gfit, n.ahead = 1)
cf   <- coef(gfit)
garch <- data.frame(
  mu_rugarch    = as.numeric(cf["mu"]),
  omega_rugarch = as.numeric(cf["omega"]),
  alpha_rugarch = as.numeric(cf["alpha1"]),
  beta_rugarch  = as.numeric(cf["beta1"]),
  nu_rugarch    = as.numeric(cf["shape"]),
  sigma1_rugarch = as.numeric(sigma(fc)[1]),
  mu1_rugarch    = as.numeric(fitted(fc)[1]))
write.csv(garch, paste0(fx, "mcneilfrey_garch_reference.csv"), row.names = FALSE)

cat("evir:  xi=", round(xi,6), " beta=", round(beta,6),
    " n.exceed=", n_exceed, "\n", sep="")
print(evt[, c("alpha","var_evir","es_evir")])
cat("rugarch: mu=", round(garch$mu_rugarch,6),
    " omega=", round(garch$omega_rugarch,6),
    " alpha=", round(garch$alpha_rugarch,6),
    " beta=", round(garch$beta_rugarch,6),
    " nu=", round(garch$nu_rugarch,6),
    " sigma1=", round(garch$sigma1_rugarch,6), "\n", sep="")
