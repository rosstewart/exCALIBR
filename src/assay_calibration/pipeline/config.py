"""
Configuration for Assay Calibration Pipeline
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional

@dataclass
class PipelineConfig:
    """Main configuration for the calibration pipeline"""

    # Input/Output
    dataset_csv: str
    dataset_name: str
    output_dir: str = "./calibration_output"

    # Precomputed fits (skip bootstrap fitting)
    precomputed_fits: Optional[str] = None  # Path to gzipped JSON of bootstrap fits
    splits_file: Optional[str] = None       # Path to precomputed splits pickle (for OOB)

    # Bootstrap parameters
    n_bootstraps: int = 1000
    num_fits_per_bootstrap: int = 100

    # Master seed for full reproducibility (train/val splits, EM initializations,
    # and E-step Monte Carlo draws are all derived from this). None (default)
    # preserves the historical unseeded behaviour.
    seed: Optional[int] = None

    # Model parameters
    components: List[int] = None  # [2], [3], or [2, 3]
    use_median_prior: bool = True
    use_2c_equation: bool = False  # Use EM estimation instead
    liberal_monotonicity: bool = True
    benign_method: Literal["benign", "avg", "synonymous"] = "avg"
    manual_prior: float = None  # If set, skip prior estimation and use this value
    population_type: str = "gnomAD"

    # Configurable conservative percentile (paired with 100 - pathogenic_percentile as the
    # upper bound). Replaces hardcoded 5/95 in conservative threshold/LR calculations.
    pathogenic_percentile: float = 5.0

    # Opt-in: restrict the "effective pathogenic sample" used for prior estimation to
    # pathogenic-labeled rows whose conservative (pathogenic_percentile-th) bootstrap
    # log-LR+ is > 0, then recompute the pathogenic sample's mixture weights (a single
    # M-step, frozen mixture components) restricted to just those rows. This is used in
    # place of the raw pathogenic weights everywhere downstream, so mixture components
    # dominated by "looks normal" pathogenic-labeled rows are naturally down-weighted
    # before the ordinary standard-EM/PU/NU prior estimation runs -- the estimator
    # itself is unchanged, only its pathogenic reference density is cleaned up first.
    # For PU-only datasets (no benign/synonymous sample), there is no fb curve to form a
    # true LR+ against, so the row-selection criterion falls back to
    # log_fp - log_f_population > 0 -- a prior-free separation check (see
    # compute_lr_filtered_pathogenic_mask's docstring).
    # EXPERIMENTAL / non-default: the pathomechanism prior (below) is the default
    # pathogenic-sample-cleaning strategy. This LR filter is kept for experimentation
    # only and is mutually exclusive with pathomechanism_prior.
    filter_pathogenic_sample_by_lr: bool = False

    # EXPERIMENTAL, opt-in alternative to filter_pathogenic_sample_by_lr's
    # ad-hoc row filter, for any dataset with a pathogenic-labeled sample.
    # Decomposes the pathogenic-labeled sample's score density as
    #   f_pathogenic_labeled(x) = gamma * f_D(x) + (1 - gamma) * f_N(x)
    # where f_D ("disease-mechanism-captured"/assay-relevant density) is
    # built from the per-component excess of the raw pathogenic weights
    # over the anchor weights (max(0, w_P - w_N), renormalized -- N gets
    # priority on any overlap, D only gets credit for a clear excess, a
    # deliberate conservative choice), f_N is FIXED to an anchor density
    # (anchored -- no label-switching), and gamma is then estimated via a
    # plain two-fixed-densities mixture-proportion EM (confirmed
    # init-robust, unlike an earlier free-weight-vector version -- see
    # compute_pathomechanism_pathogenic_density in fit_utils/point_ranges.py).
    # gamma is the estimated fraction of PLP-labeled variants whose
    # mechanism this assay captures (reported as
    # PLP_frac_pathomechanism_measured in the output calibration JSON).
    # f_D feeds the prior estimator in place of the raw pathogenic density.
    # The anchor is the benign/synonymous density when a benign/synonymous
    # sample exists (PN/standard mode), or, for PU-only datasets, the raw
    # gnomAD/population density directly -- the same anchor
    # filter_pathogenic_sample_by_lr's own PU fallback already uses
    # (log_fp - log_f_population), just formalized as a generative mixture.
    # No unmixing of gnomAD's own (typically small) disease-variant content
    # is attempted first: the excess construction is provably invariant to
    # it, at the cost of gamma_hat itself being a mildly conservative
    # (downward-biased) estimate of true mechanistic coverage when gnomAD
    # does carry real disease-relevant mass (see the docstring in
    # point_ranges.py for the derivation). For PU-only datasets, when the
    # corrected f_D has zero density everywhere in the (often small)
    # population sample, the prior for that bootstrap fit is floored to
    # 0.01 rather than discarded -- with few gnomAD points, finding no trace
    # of the disease-relevant component there is itself evidence of a
    # near-zero prior, not an absence of information.
    # Like filter_pathogenic_sample_by_lr, this substitution only affects
    # the scalar prior -- the LR+ curve used for per-variant evidence
    # assignment keeps using the raw pathogenic density. Mutually exclusive
    # with filter_pathogenic_sample_by_lr. This is the default
    # pathogenic-sample-cleaning strategy (ships on); pass
    # pathomechanism_prior=False (CLI --no-pathomechanism-prior) to disable it.
    pathomechanism_prior: bool = True

    # Per-dataset overrides (used in IGVF batch mode)
    scoreset_flipped_override: Optional[bool] = None  # Force flip state

    # BasicScoreset sample name override (order must match column ordering in data)
    sample_names: Optional[List[str]] = None

    # Per-sample M-step reweighting (also applied to val_ll for fit selection)
    # sample_proportions takes precedence over sample_balance_beta when both given.
    # sample_proportions=[2,1,1,1]: sample 0 contributes twice as much as others.
    # sample_balance_beta=1: all samples contribute equally regardless of size.
    sample_proportions: Optional[List[float]] = None
    sample_balance_beta: float = 0.0
    weighted_val_ll: bool = False  # if True, val_ll uses the same sample weighting as the M-step

    # Debug mode: verbose logging of component params, weights, flip detection, point ranges
    debug: bool = False

    # Viz-only mode: skip variant table + calibration save, only regenerate plots
    viz_only: bool = False

    # OOB evidence
    compute_oob: bool = False
    oob_min_samples: int = 10

    # Execution parameters
    execution_mode: Literal["slurm", "parallel", "single"] = "parallel"
    n_jobs: int = -1  # -1 uses all available CPUs

    # SLURM parameters (only used if execution_mode="slurm")
    slurm_account: str = "default"
    slurm_partition: str = "short"
    slurm_time_hours: int = 23
    slurm_mem_gb: int = 1
    slurm_cpus_per_task: int = 12
    slurm_conda_env: str = "assay_calibration"
    slurm_module_commands: List[str] = None

    # Model selection (only used if components=[2,3])
    auto_select_model: bool = True
    model_selection_alpha: float = 0.05
    use_conservative_selection: bool = True  # Use 5th percentile test

    # Output options
    save_bootstrap_fits: bool = False
    save_visualizations: bool = True
    point_values: List[int] = None
    score_range_points: int = 2000  # Number of interpolation points for score range

    # ACMG-mapping method: how LR+ thresholds are derived from the prior.
    #   "tavtigian"  — legacy C*-based integer-point system (default).
    #   "piecewise"  — prior-adaptive piecewise α; integer points preserved
    #                  with exact posterior at the four ACMG knots.
    #   "continuous" — posterior-based classification (B/LB/VUS/LP/P) directly
    #                  from LR+; no integer points.
    # "all" is handled at the orchestration layer (run_pipeline / run_igvf_batch)
    # by running the calibration step once per ACMG-mapping method.
    #   "strict_additive" — prior-dependent LSQ-optimal α(p); strictly additive
    #                      (1 VS = 2 St = 4 M = 8 Su in log-LR at each prior);
    #                      exact posteriors at shifting classification boundaries.
    acmg_mapping_method: Literal["tavtigian", "piecewise", "continuous",
                                 "strict_additive"] = "tavtigian"

    # ClinVar parameters
    clinvar_release: str = "2025"
    min_clinvar_star: int = 1
    synonymous_exclusive: bool = False

    # Progress reporting (optional — used by web backend)
    # If set, the pipeline writes structured JSON progress updates to this path.
    # Has no effect on pipeline behaviour or output when None (the default).
    progress_file: Optional[str] = None


    def __post_init__(self):
        if self.components is None:
            self.components = [2, 3]
        if self.point_values is None:
            self.point_values = [1, 2, 3, 4, 5, 6, 7, 8]

        # Validate components
        if not all(c in [2, 3, 4] for c in self.components):
            raise ValueError("Components must be 2, 3, or 4")

        # Validate manual_prior
        if self.manual_prior is not None:
            if not (0 < self.manual_prior < 1):
                raise ValueError(f"manual_prior must be in (0, 1), got {self.manual_prior}")

        # Validate pathogenic_percentile (must stay below its paired upper bound
        # 100 - pathogenic_percentile, i.e. below 50)
        if not (0 < self.pathogenic_percentile < 50):
            raise ValueError(
                f"pathogenic_percentile must be in (0, 50), got {self.pathogenic_percentile}"
            )

        # pathomechanism_prior and filter_pathogenic_sample_by_lr are mutually
        # exclusive pathogenic-sample-cleaning strategies for prior estimation.
        if self.pathomechanism_prior and self.filter_pathogenic_sample_by_lr:
            raise ValueError(
                "pathomechanism_prior and filter_pathogenic_sample_by_lr are mutually "
                "exclusive -- pass filter_pathogenic_sample_by_lr=False when enabling "
                "pathomechanism_prior."
            )
        # Validate population_type
        valid_pop_types = {
            'all_variants', 'all_nsSNV', 'all_missense_nsSNV',
            'gnomAD', 'gnomAD_nsSNV', 'gnomAD_missense_nsSNV'
        }
        if self.population_type not in valid_pop_types:
            raise ValueError(f"population_type must be one of {valid_pop_types}, got {self.population_type}")


        # If using SLURM, adjust job count
        if self.execution_mode == "slurm" and self.n_jobs == -1:
            self.n_jobs = 30  # Reasonable default for job generation


def resolve_prior_mode(filter_flag, pathomechanism_flag):
    """Resolve the two mutually-exclusive pathogenic-sample-cleaning flags.

    Both CLI flags parse with ``default=None`` (unset). The default mode is the
    pathomechanism prior (LR filter off, pathomechanism on). Specifying just one
    flag implicitly clears the other, so the experimental
    ``--filter-pathogenic-sample-by-lr`` can be used on its own without also
    having to pass ``--no-pathomechanism-prior``. Setting both explicitly is left
    to PipelineConfig's mutual-exclusion validation to reject.

    Returns a ``(filter_pathogenic_sample_by_lr, pathomechanism_prior)`` tuple.
    """
    f, p = filter_flag, pathomechanism_flag
    if f is None and p is None:
        return False, True          # default: pathomechanism prior
    if p is None:                    # only the LR filter was specified
        return f, (False if f else True)
    if f is None:                    # only the pathomechanism flag was specified
        return False, p
    return f, p                      # both explicit -> let PipelineConfig validate
