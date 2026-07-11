#!/usr/bin/env Rscript
# Thin CLI wrapper around acmgscaler::calibrate() for one dataset at a time.
#
# Sources the package's R/*.R files directly from its source checkout
# (base-R only, per its own README -- no install.packages needed) rather
# than requiring the package to be installed.
#
# Usage:
#   Rscript run_acmgscaler.R <acmgscaler_dir> <input_csv> <output_csv> <prior> [<thresholds_csv>]
#
# <input_csv> must have columns: class (P/B/other), value (numeric score).
# Writes <output_csv> = calibrate(df, value='value', prior=<prior>)$likelihood_ratios,
# i.e. one row per input row with value_lr, value_lr_lower, value_lr_upper,
# value_evidence columns appended, in the original row order.
#
# If <thresholds_csv> is given, also writes $score_thresholds (one row per
# evidence tier, with the score interval [value_lower, value, value_upper]
# for that tier) -- used to draw threshold lines on a per-dataset figure.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: run_acmgscaler.R <acmgscaler_dir> <input_csv> <output_csv> <prior> [<thresholds_csv>]")
}
acmgscaler_dir <- args[1]
input_csv <- args[2]
output_csv <- args[3]
prior <- as.numeric(args[4])
thresholds_csv <- if (length(args) >= 5) args[5] else NA

for (f in c("density_ratio.R", "add_evidence_levels.R", "calibrate.R")) {
  source(file.path(acmgscaler_dir, "R", f))
}

df <- read.csv(input_csv, stringsAsFactors = FALSE)
result <- calibrate(df, value = "value", prior = prior, group = NULL)
write.csv(result$likelihood_ratios, output_csv, row.names = FALSE)
if (!is.na(thresholds_csv)) {
  write.csv(result$score_thresholds, thresholds_csv, row.names = FALSE)
}
