# Generate R reference outputs for the CBPS compatibility tests.
#
# Requires the CBPS package. Run from the repository root:
#   Rscript tests/density/cbps/fixtures/generate_fixtures.R
suppressMessages({library(CBPS); library(jsonlite)})

here <- "tests/density/cbps/fixtures"
cases <- list(
  list(name = "binary_ate_over_twostep", data = "binary", factor = TRUE, ATT = 0, method = "over", twostep = TRUE),
  list(name = "binary_ate_exact_twostep", data = "binary", factor = TRUE, ATT = 0, method = "exact", twostep = TRUE),
  list(name = "binary_ate_over_cue", data = "binary", factor = TRUE, ATT = 0, method = "over", twostep = FALSE),
  list(name = "binary_att1_over_twostep", data = "binary", factor = TRUE, ATT = 1, method = "over", twostep = TRUE),
  list(name = "binary_att1_exact_twostep", data = "binary", factor = TRUE, ATT = 1, method = "exact", twostep = TRUE),
  list(name = "three_level_over_twostep", data = "three_level", factor = TRUE, ATT = 0, method = "over", twostep = TRUE),
  list(name = "three_level_exact_twostep", data = "three_level", factor = TRUE, ATT = 0, method = "exact", twostep = TRUE),
  list(name = "four_level_over_twostep", data = "four_level", factor = TRUE, ATT = 0, method = "over", twostep = TRUE),
  list(name = "continuous_over_twostep", data = "continuous", factor = FALSE, ATT = 0, method = "over", twostep = TRUE),
  list(name = "continuous_exact_twostep", data = "continuous", factor = FALSE, ATT = 0, method = "exact", twostep = TRUE),
  list(name = "continuous_over_cue", data = "continuous", factor = FALSE, ATT = 0, method = "over", twostep = FALSE)
)

for (case in cases) {
  d <- read.csv(file.path(here, paste0(case$data, ".csv")))
  if (case$factor) d$treat <- factor(d$treat)
  set.seed(1)
  fit <- suppressWarnings(CBPS(treat ~ x0 + x1 + x2 + x3, data = d, ATT = case$ATT,
                               method = case$method, twostep = case$twostep, standardize = TRUE))
  out <- list(
    data = case$data, ATT = case$ATT, method = case$method, twostep = case$twostep,
    coefficients = unname(as.matrix(coef(fit))),
    fitted_values = unname(as.matrix(fit$fitted.values)),
    weights = unname(as.vector(fit$weights)),
    J = as.numeric(fit$J), mle_J = as.numeric(fit$mle.J),
    converged = as.numeric(fit$converged)
  )
  if (!case$factor) out$sigmasq <- as.numeric(fit$sigmasq)
  write_json(out, file.path(here, paste0(case$name, ".json")), digits = NA, auto_unbox = TRUE)
  cat(case$name, "J =", fit$J, "\n")
}
