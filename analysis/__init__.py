import os as _os

# The manuscript results were computed with rule posteriors rounded to 3
# decimals in the c search (src/assay_calibration/fit_utils/evidence_thresholds.py);
# the package default is now exact. Pin the manuscript setting for everything
# that imports this package unless the caller has set it explicitly.
_os.environ.setdefault("EXCALIBR_TAVTIGIAN_POSTERIOR_DECIMALS", "3")
