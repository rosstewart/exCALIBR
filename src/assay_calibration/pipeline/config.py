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

    # Model parameters
    components: List[int] = None  # [2], [3], or [2, 3]
    use_median_prior: bool = True
    use_2c_equation: bool = False  # Use EM estimation instead
    liberal_monotonicity: bool = True
    benign_method: Literal["benign", "avg", "synonymous"] = "avg"
    manual_prior: float = None  # If set, skip prior estimation and use this value
    population_type: str = "gnomAD"

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

    # ClinVar parameters
    clinvar_release: str = "2025"
    min_clinvar_star: int = 1

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
