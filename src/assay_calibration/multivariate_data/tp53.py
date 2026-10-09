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
  - log_TempSens: the notebook's plain ``np.log(TempSens)`` silently turned
    every TempSens <= 0 into NaN (indistinguishable from a true missing
    measurement) -- TempSens is signed (observed range here is roughly
    [-437, 80]), and ~42% of its non-null values are <= 0, so a straight
    log discarded nearly half of the dimension's already-sparse real data.
    A signed-log transform (``sign(x) * log1p(|x|)``) was tried as a fix
    (keeps all rows, avoids the NaN-ing), but a rank-based separation check
    (Mann-Whitney AUC, P/LP vs B/LB) showed it's identical to raw TempSens's
    separation -- expected, since AUC only depends on rank order and any
    monotonic transform preserves that exactly. With no separation upside
    and a transform that only obscures the variable's natural units, the
    column (name kept as "log_TempSens" for backward compatibility with
    existing job configs/figure naming, though it is no longer logged) now
    carries raw, untransformed TempSens.
  - kawoligo_mode="categorical" (EXPERIMENTAL, opt-in; default "jitter" is
    unchanged): KawOligo is kept as integer level codes (Tetramer=0,
    Dimer=1, Monomer=2 -- the same order as KAWOLIGO_MAP, i.e. increasing
    pathogenicity), with NO jitter, and the multiscoreset is tagged with
    ``categorical_dims = {"KawOligo": KAWOLIGO_LEVELS}`` so the CFUSN fitter
    models it as a categorical dimension (fit_utils/cfusn/categorical.py)
    instead of a jittered continuous one. NOTE: KawOligo is ordinal, but
    that model treats levels as nominal (unordered); see categorical.py.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data_utils.dataset import BasicMultiScoreset, BasicScoreset

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
KAWOLIGO_LEVELS = ["Tetramer", "Dimer", "Monomer"]
DEFAULT_KAWOLIGO_SEED = 0
DEFAULT_KAWOLIGO_JITTER_SIGMA = 0.1
KAWOLIGO_MODES = ("jitter", "categorical")


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
    kawoligo_mode: str = "jitter",
) -> pd.DataFrame:
    """Build the processed TP53 variant dataframe (one row per variant, with
    a 5-class ``sample_assignments`` column), equivalent to
    tp53_processed.csv.gz. ``kawoligo_mode``: see the module docstring.
    """
    if kawoligo_mode not in KAWOLIGO_MODES:
        raise ValueError(f"kawoligo_mode must be one of {KAWOLIGO_MODES}, got {kawoligo_mode!r}")
    df_rpvs, df_hotspot = load_reference_cohorts(cohorts_path)
    df_rpvs_high = df_rpvs[df_rpvs["confidence"] != "low"]

    df_tp53 = pd.read_csv(variants_path)
    # No transform: raw TempSens, kept under the "log_TempSens" column name
    # for backward compatibility -- see module docstring's quirk note.
    df_tp53["log_TempSens"] = df_tp53["TempSens"]

    if kawoligo_mode == "categorical":
        df_tp53["KawOligo"] = df_tp53["KawOligo"].map(
            {level: float(code) for code, level in enumerate(KAWOLIGO_LEVELS)}
        )
    else:
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


def build_tp53_functional_scoresets(
    df_tp53: Optional[pd.DataFrame] = None,
    **build_kwargs,
) -> List[BasicScoreset]:
    """Per-dataset BasicScoreset list for TP53's 16 functional dimensions --
    the same data build_tp53_multiscoreset merges into one BasicMultiScoreset,
    stopped one level earlier so other callers (e.g.
    src/assay_calibration/multivariate_data/combined.py's "combined"
    gene-set, which otherwise builds a DIFFERENT, wrong TP53 functional
    panel from the generic integrated dataframe) can splice these into
    MultiPredictorFunctionalScoreset.from_predictor_and_functional
    alongside predictor dims. Confirmed equivalent to the merged
    build_tp53_multiscoreset path: building N independent single-column
    BasicScoresets from identical rows and recombining via
    BasicMultiScoreset is a no-op for ID-alignment (every one of them
    shares the exact same id/sample_assignments columns already), so
    n_variants/sample_counts/score matrix come out identical either way.
    """
    if df_tp53 is None:
        df_tp53 = build_tp53_dataframe(**build_kwargs)
    ids = df_tp53["Variant"].astype(str).to_numpy()
    sa = df_tp53["sample_assignments"].to_numpy()
    return [
        BasicScoreset(
            scores=df_tp53[col].to_numpy(dtype=float),
            sample_assignments=sa,
            ids=ids,
            scoreset_name=col,
        )
        for col in SCORE_COLS
    ]


def build_tp53_multiscoreset(
    df_tp53: Optional[pd.DataFrame] = None,
    **build_kwargs,
) -> BasicMultiScoreset:
    """Build the 16-dimension BasicMultiScoreset for TP53.

    Built from build_tp53_functional_scoresets (one BasicScoreset per
    dimension) rather than independently reparsing df_tp53's wide format,
    so the two stay in sync by construction -- confirmed equivalent to the
    previous direct build_basicmultiscoreset_from_wide_dataframe path (same
    n_variants, same sample_counts, same per-variant score matrix).

    With ``kawoligo_mode="categorical"`` (passed through to
    build_tp53_dataframe, or stating how a caller-supplied df_tp53 was
    built), the result carries ``categorical_dims = {"KawOligo": levels}``.
    """
    kawoligo_mode = build_kwargs.get("kawoligo_mode", "jitter")
    if df_tp53 is None:
        df_tp53 = build_tp53_dataframe(**build_kwargs)

    ms = BasicMultiScoreset(
        build_tp53_functional_scoresets(df_tp53),
        dataset_names=SCORE_COLS,
        sample_names=SAMPLE_NAMES,
    )
    if kawoligo_mode == "categorical":
        ms.categorical_dims = {"KawOligo": list(KAWOLIGO_LEVELS)}
    return ms
