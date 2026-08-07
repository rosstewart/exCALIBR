"""
Predictor + functional combined ingestion: join REVEL/MP2/AM predictor
dimensions with a gene's functional assay dimensions on protein_variant/
hgvs_p, via PredictorScoreset/MultiPredictorFunctionalScoreset.

No join logic of its own -- this is a thin caller of the two new dataset.py
classes. Functional dimensions are built the same way hpc/prepare.py's
existing ``multivariate`` subcommand builds them (one Scoreset per Dataset
value from the integrated variant-effect dataframe, population_type="gnomAD",
regularization_type=None, i.e. Scoreset's own defaults).
"""

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ..data_utils.dataset import Scoreset, PredictorScoreset, MultiPredictorFunctionalScoreset
from .predictors import PREDICTORS, PREDICTOR_DATASET_NAMES, load_predictor_data, df_to_basic_scoreset

DEFAULT_INTEGRATED_DATAFRAME = (
    "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz"
)

# PredictorScoreset/df_to_basic_scoreset's onehot column convention: 0=P/LP, 1=B/LB, 2=gnomAD.
_PREDICTOR_CLINVAR_COLS = (0, 1)


def build_functional_scoresets(
    df: pd.DataFrame,
    gene: str,
    datasets: List[str],
    clinvar_release: str = "2026",
    population_type: str = "gnomAD",
) -> List[Scoreset]:
    """Build one Scoreset per Dataset value for ``gene`` (same construction
    hpc/prepare.py's ``_process_multivariate_gene`` uses for its functional
    dimensions, minus the fitting step).
    """
    df_gene = df[df["Gene"] == gene]
    scoresets = []
    for ds_name in datasets:
        ds = Scoreset(
            df_gene[df_gene["Dataset"] == ds_name],
            clinvar_release=clinvar_release,
            min_clinvar_star=1,
            population_type=population_type,
        )
        if sum(1 for _ in ds.samples) < 2:
            continue
        scoresets.append(ds)
    return scoresets


def get_functionally_assayed_protein_variants(
    df: pd.DataFrame,
    gene: str,
    datasets: List[str],
    score_col: str = "auth_reported_score",
) -> Set[str]:
    """Protein-variant strings (``aa_ref + aa_pos + aa_alt``, e.g. "A119T",
    matching Scoreset._aa_subs's format) for every row with a non-null
    functional score across ``datasets`` -- regardless of ClinVar
    classification, i.e. every variant actually MEASURED by a functional
    assay, whether or not it happens to be one of that assay's defined
    P/LP/B/LB controls under any particular ClinVar release.

    Used to identify which variants a predictor CSV's own P/LP/B/LB label
    should be split off a duplicate row for (see
    split_predictor_clinvar_leakage) rather than trusted on the real,
    jointly-observed row.
    """
    df_gene = df[(df["Gene"] == gene) & (df["Dataset"].isin(datasets))]
    scored = df_gene[
        df_gene[score_col].notna()
        & df_gene["aa_ref"].notna() & df_gene["aa_pos"].notna() & df_gene["aa_alt"].notna()
    ]
    variants = (
        scored["aa_ref"].astype(str)
        + scored["aa_pos"].astype(float).astype(int).astype(str)
        + scored["aa_alt"].astype(str)
    )
    return set(variants)


def split_predictor_clinvar_leakage(
    predictor_scoreset: PredictorScoreset,
    functionally_assayed_variants: Set[str],
    leakage_cols=_PREDICTOR_CLINVAR_COLS,
    dup_suffix: str = "__predictor_dup",
) -> PredictorScoreset:
    """Split (not discard) a PredictorScoreset's own P/LP(0)/B/LB(1) label
    for any variant that was directly, functionally assayed for this gene
    (regardless of whether the functional assay's own ClinVar release/star
    threshold actually classified it as P/LP/B/LB), into two data points:

    - the ORIGINAL row (real variant id, still joins with the functional
      Scoreset on that id) keeps its predictor score but has its P/LP/B/LB
      flags cleared -- so the row jointly observed with the functional
      dimension(s) is governed only by the functional side's own,
      release-fixed ClinVar label (plus gnomAD, untouched, since population
      membership isn't ClinVar-release-dependent).
    - an ARTIFICIAL DUPLICATE row (synthetic id, so it can never join with
      any functional Scoreset -- its functional dimensions come out NaN in
      the combined matrix) carries the original, unmodified P/LP/B/LB
      label, so the predictor dimension's own marginal calibration still
      gets the full labeled training signal.

    Why split instead of just clearing the original: the predictor CSV's
    P/LP/B/LB label comes from an independent, unversioned ClinVar
    snapshot (see multivariate_data/predictors.py's docstring), not the
    same release the functional Scoreset was built with. For a variant
    with functional data, that release IS the fixed "gold standard"
    control set the functional assay's calibration is defined against --
    letting the predictor CSV's (possibly newer, or reclassified) label
    promote such a variant into the joint P/LP or B/LB sample would leak
    ClinVar information from outside that vintage into what's supposed to
    be a fixed control set. But since the marginal-likelihood EM only uses
    a row's *observed* dimensions, isolating the label on a
    functional-score-absent duplicate lets it inform the predictor
    dimension's own marginal without ever contributing to any
    cross-dimension (predictor-functional correlation) estimate -- nothing
    to leak, since it's never jointly observed with anything. Variants
    with no functional data at all are untouched (nothing to split).

    Mutates ``predictor_scoreset`` in place (and returns it): its
    ``scores``/``_ids``/``_sample_assignments`` arrays are extended with
    the duplicate rows, and the original rows' leakage columns are cleared.
    """
    ids = np.asarray(predictor_scoreset._ids)
    sa = predictor_scoreset._sample_assignments

    has_leaky_label = np.zeros(len(ids), dtype=bool)
    for col in leakage_cols:
        if col < sa.shape[1]:
            has_leaky_label |= sa[:, col]
    is_assayed = np.array([vid in functionally_assayed_variants for vid in ids])
    split_mask = has_leaky_label & is_assayed

    if not split_mask.any():
        return predictor_scoreset

    dup_scores = predictor_scoreset.scores[split_mask]
    dup_sample_assignments = sa[split_mask].copy()   # full original label, pre-clearing
    dup_ids = np.array([f"{vid}{dup_suffix}" for vid in ids[split_mask]])

    for col in leakage_cols:
        if col < sa.shape[1]:
            sa[split_mask, col] = False

    predictor_scoreset.scores = np.concatenate([predictor_scoreset.scores, dup_scores])
    predictor_scoreset._sample_assignments = np.concatenate([sa, dup_sample_assignments], axis=0)
    predictor_scoreset._ids = np.concatenate([ids, dup_ids])
    predictor_scoreset.sample_counts = predictor_scoreset._sample_assignments.sum(axis=0)
    return predictor_scoreset


def build_combined_multiscoreset(
    gene: str,
    functional_scoresets: List[Scoreset],
    functional_dataset_names: List[str],
    predictor_data_dir: str,
    predictors: Optional[List[str]] = None,
    functionally_assayed_variants: Optional[Set[str]] = None,
) -> Optional[MultiPredictorFunctionalScoreset]:
    """Join gene's predictor dimensions (REVEL/MP2/AM) with its functional
    dimensions into one MultiPredictorFunctionalScoreset.

    ``functionally_assayed_variants`` (see get_functionally_assayed_protein_
    variants), if given, splits predictor-CSV P/LP/B/LB leakage (see
    split_predictor_clinvar_leakage) for variants that have functional data
    -- predictor-only variants are unaffected and keep their own label.

    Returns None if no predictor CSVs are available for ``gene``.
    """
    predictors = list(predictors) if predictors is not None else list(PREDICTORS)

    by_gene: Dict[str, Dict[str, pd.DataFrame]] = load_predictor_data(
        predictor_data_dir, genes=[gene]
    )
    predictor_dfs = by_gene.get(gene, {})
    have = [p for p in predictors if p in predictor_dfs]
    if not have:
        return None

    predictor_scoresets = [df_to_basic_scoreset(predictor_dfs[p], p) for p in have]
    if functionally_assayed_variants:
        for ps in predictor_scoresets:
            split_predictor_clinvar_leakage(ps, functionally_assayed_variants)
    predictor_names = [PREDICTOR_DATASET_NAMES.get(p, p) for p in have]

    return MultiPredictorFunctionalScoreset.from_predictor_and_functional(
        predictor_scoresets,
        functional_scoresets,
        predictor_names=predictor_names,
        functional_names=list(functional_dataset_names),
    )
