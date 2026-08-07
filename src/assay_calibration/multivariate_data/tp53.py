"""
TP53 ingestion: raw annotated variants + reference cohorts -> BasicMultiScoreset.

Faithful port of process_tp53.ipynb's ingestion cell (the version that
produced tp53_processed.csv.gz, which analyze_tp53.ipynb reads directly and
reconstructs the same 16-dimension BasicMultiScoreset from).

Two dataset-specific quirks made explicit here, per the consolidation plan:
  - RPVS_ALL: toggles the RPV sample between the high-confidence subset
    (rpvs_high_conf, "confidence" != "low") and the full RPV list (rpvs_all).
    Default False (rpvs_high_conf), matching the notebook's default.
  - KawOligo's jitter (np.random.normal(0, 0.05, ...)) was originally
    unseeded (non-reproducible across runs); it now takes an explicit seed
    via a local RandomState, not global numpy RNG state. sigma is also now
    an explicit parameter (default widened to 0.1, up from the notebook's
    0.05): {Tetramer, Dimer, Monomer} are 0.5 apart, and sigma=0.05 makes
    each category a near-Dirac spike (~10 sigma from its neighbors) that a
    handful of shared continuous skew-normal mixture components -- which
    must also jointly fit 15 other continuous assay dimensions -- can't
    represent well without dedicating whole components to each spike.
    sigma=0.1 still keeps categories comfortably separated (~5 sigma gap)
    while giving components room to fit smooth, non-degenerate density.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..data_utils.dataset import BasicMultiScoreset
from .common import build_basicmultiscoreset_from_wide_dataframe

DEFAULT_COHORTS_PATH = "/data/ross/assay_calibration/TP53/tp53_reference_cohorts.csv"
DEFAULT_VARIANTS_PATH = "/data/ross/assay_calibration/TP53/tp53_annotated_variants.csv"

SCORE_COLS = [
    "WAF1nWT",
    "MDM2nWT",
    "BAXnWT",
    "h1433snWT",
    "AIP1nWT",
    "GADD45nWT",
    "NOXAnWT",
    "P53R2nWT",
    "Giac_NULL_Nut",
    "Giac_Etop",
    "Funk_RFS",
    "Kotler_RFS",
    "Giac_WT_Nut",
    "DN_score",
    "log_TempSens",
    "KawOligo",
]
SAMPLE_NAMES = ["P/LP", "B/LB", "gnomAD", "Synonymous", "RPV"]
KAWOLIGO_MAP = {"Tetramer": 0.0, "Dimer": 0.5, "Monomer": 1.0}
DEFAULT_KAWOLIGO_SEED = 0
DEFAULT_KAWOLIGO_JITTER_SIGMA = 0.1


def load_reference_cohorts(cohorts_path: str = DEFAULT_COHORTS_PATH) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_rpvs, df_hotspot) from tp53_reference_cohorts.csv."""
    df_rpvs_hotspot = pd.read_csv(cohorts_path)
    df_rpvs = df_rpvs_hotspot[df_rpvs_hotspot["category"] == "RPV"].copy()
    df_hotspot = df_rpvs_hotspot[df_rpvs_hotspot["category"] == "hotspot"].copy()
    df_rpvs["Variant"] = df_rpvs["Variant"].fillna(df_rpvs["cDNA"])
    df_rpvs["GVSr"] = df_rpvs["n_germline"] / df_rpvs["n_somatic"]
    return df_rpvs, df_hotspot


def build_tp53_dataframe(
    RPVS_ALL: bool = False,
    variants_path: str = DEFAULT_VARIANTS_PATH,
    cohorts_path: str = DEFAULT_COHORTS_PATH,
    kawoligo_seed: int = DEFAULT_KAWOLIGO_SEED,
    kawoligo_jitter_sigma: float = DEFAULT_KAWOLIGO_JITTER_SIGMA,
) -> pd.DataFrame:
    """Build the processed TP53 variant dataframe (one row per variant, with
    a 5-class ``sample_assignments`` column), equivalent to
    tp53_processed.csv.gz.
    """
    df_rpvs, df_hotspot = load_reference_cohorts(cohorts_path)
    df_rpvs_high = df_rpvs[df_rpvs["confidence"] != "low"]

    df_tp53 = pd.read_csv(variants_path)
    df_tp53["log_TempSens"] = np.log(df_tp53["TempSens"])

    rng = np.random.RandomState(kawoligo_seed)
    df_tp53["KawOligo"] = (
        df_tp53["KawOligo"].map(KAWOLIGO_MAP)
        + rng.normal(0, kawoligo_jitter_sigma, size=len(df_tp53))
    )

    rpvs_high_conf = df_rpvs_high.Variant.tolist()
    rpvs_all = df_rpvs.Variant.tolist()
    gvsr_map = (
        df_rpvs
        .assign(_has_gvsr=df_rpvs["GVSr"].notna())
        .sort_values(["Variant", "_has_gvsr"], ascending=[True, False])
        .drop_duplicates("Variant", keep="first")
        .set_index("Variant")["GVSr"]
    )

    df_tp53 = df_tp53[
        (df_tp53["spliceAI_max"] <= 0.2) | (df_tp53["spliceAI_max"].isna())
    ]

    sample_assignments = np.zeros((len(df_tp53), 5), dtype=bool)
    sample_assignments[:, 0] = df_tp53.ClinVar_category == "P/LP"
    sample_assignments[:, 1] = df_tp53.ClinVar_category == "B/LB"
    sample_assignments[:, 2] = df_tp53.gnomADv4_AF.notna()
    sample_assignments[:, 3] = df_tp53.variant_class == "synonymous"
    sample_assignments[:, 4] = (
        df_tp53.Variant.isin(rpvs_high_conf if not RPVS_ALL else rpvs_all)
        & (df_tp53.ClinVar_category != "B/LB")  # no known benigns in this class
    )

    df_tp53 = df_tp53.copy()
    df_tp53["GVSr"] = df_tp53["Variant"].map(gvsr_map)
    df_tp53["hotspot"] = df_tp53["Variant"].isin(df_hotspot["Variant"].tolist())
    df_tp53["sample_assignments"] = [
        ",".join(map(str, np.flatnonzero(row))) for row in sample_assignments
    ]
    return df_tp53


def build_tp53_multiscoreset(
    df_tp53: Optional[pd.DataFrame] = None,
    **build_kwargs,
) -> BasicMultiScoreset:
    """Build the 16-dimension BasicMultiScoreset for TP53.

    Uses the shared wide-format helper (common.py): every score dimension
    is a column of df_tp53, and all 16 dimensions share the same
    per-variant sample_assignments column -- confirmed equivalent to
    building 16 separate BasicScoresets and combining them via
    BasicMultiScoreset (the notebook's original approach): same
    n_variants, same sample_counts, same per-variant score matrix.
    """
    if df_tp53 is None:
        df_tp53 = build_tp53_dataframe(**build_kwargs)

    return build_basicmultiscoreset_from_wide_dataframe(
        df_tp53,
        score_cols=SCORE_COLS,
        id_col="Variant",
        sample_assignments_col="sample_assignments",
        dataset_names=SCORE_COLS,
        sample_names=SAMPLE_NAMES,
    )
