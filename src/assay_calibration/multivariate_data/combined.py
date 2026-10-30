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
    regularization_type: Optional[str] = None,
    tp53_build_kwargs: Optional[dict] = None,
) -> List[Scoreset]:
    """Build one Scoreset per Dataset value for ``gene`` (same construction
    hpc/prepare.py's ``_process_multivariate_gene`` uses for its functional
    dimensions, minus the fitting step).

    ``regularization_type`` (default None, i.e. Scoreset's own default):
    pass "all_assayed" to retain VUS/unlabeled rows (Scoreset's default
    keep_mask drops any row without a P/LP/B/LB/gnomAD/Synonymous
    sample_assignment) -- needed by callers that want to score true
    ClinVar-VUS variants under the combined functional+predictor fit.

    TP53 special-case: the generic ``df``/``datasets`` (sourced from
    DEFAULT_INTEGRATED_DATAFRAME) gives a functional panel that does NOT
    match TP53's own dedicated "tp53" gene-set (different datasets
    entirely -- confirmed missing KawOligo/log_TempSens, which need
    preprocessing only src/assay_calibration/multivariate_data/tp53.py
    implements). Every "combined" ms-building call site in the repo calls
    this function directly before build_combined_multiscoreset, so fixing
    it here (rather than patching each call site) makes TP53's combined
    functional panel use the exact same data as "tp53" everywhere at once.
    ``datasets``/``clinvar_release``/``population_type``/
    ``regularization_type`` are ignored for TP53 -- tp53.py's ingestion
    reads its own dedicated raw files and has no equivalent knobs.

    TSC2 likewise uses its dedicated "tsc2" panel (tsc2.py: Biar et al. 2026
    VAMP-seq, immune SGE, cliPE) instead of TSC2_IGVF; ``datasets`` and
    ``population_type`` are ignored, ``clinvar_release`` sets its labels and
    ``regularization_type="all_assayed"`` keeps unlabelled rows.
    """
    if gene == "TP53":
        # tp53_build_kwargs (e.g. kawoligo_mode) are passed through to
        # tp53.build_tp53_dataframe; unset means tp53.py's defaults.
        from .tp53 import build_tp53_functional_scoresets
        return build_tp53_functional_scoresets(**(tp53_build_kwargs or {}))
    if gene == "TSC2":
        # Same idea for TSC2: the dedicated Biar et al. 2026 panel
        # (multivariate_data/tsc2.py, the "tsc2" gene-set) replaces the
        # integrated dataframe's TSC2_IGVF; its labels come from the same
        # integrated dataframe, at ``clinvar_release``.
        from .tsc2 import build_tsc2_functional_scoresets
        return build_tsc2_functional_scoresets(
            keep_unlabelled=regularization_type == "all_assayed",
            df_integrated=df, clinvar_release=clinvar_release)

    df_gene = df[df["Gene"] == gene]
    scoresets = []
    for ds_name in datasets:
        ds = Scoreset(
            df_gene[df_gene["Dataset"] == ds_name],
            clinvar_release=clinvar_release,
            min_clinvar_star=1,
            population_type=population_type,
            regularization_type=regularization_type,
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
    if gene == "TSC2":
        # build_functional_scoresets uses tsc2.py's panel for TSC2.
        from .tsc2 import functionally_assayed_ids
        return functionally_assayed_ids(df_integrated=df)
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


def select_variants(ms, mask: np.ndarray) -> None:
    """Restrict `ms` (a BasicMultiScoreset/MultiPredictorFunctionalScoreset)
    IN PLACE to exactly the rows where `mask` is True. `mask` is indexed
    against ms.kept_variants/ms.scores/ms.n_variants (i.e. AFTER the "at
    least one dimension observed" filter `_build()` already applies -- NOT
    against the raw pre-union `_all_ids`/`_scores_matrix`/`_missing_mask`).

    Row/variant analog of redundancy_collapse.py::select_dims's column
    restriction -- same direct-mutation style, just restricting rows instead
    of columns. Also keeps `_keep_mask` (defined against the full,
    pre-filter `_all_ids` length) internally consistent with the narrowed
    `kept_variants`/`scores`, since `keep_mask`/`full_scores` are still a
    public part of BasicMultiScoreset's API even though nothing in this
    module's own callers reads them.

    Used by build_combined_multiscoreset's `require_both_modalities` to
    implement "intersection across modalities, union within modality" for
    the combined gene-set: a variant observed in ANY functional assay AND
    ANY predictor is kept; a variant observed in only one modality (e.g. all
    3 predictor dims but zero functional dims, or vice versa) is dropped
    from the combined fit/scoring population entirely -- it still
    contributes to that modality's own functional-only/predictor-only union
    fit, just not to "combined".
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != ms.n_variants:
        raise ValueError(
            f"mask length {mask.shape[0]} does not match ms.n_variants {ms.n_variants}"
        )

    old_keep_idx = np.where(ms._keep_mask)[0]
    new_keep_mask = ms._keep_mask.copy()
    new_keep_mask[old_keep_idx[~mask]] = False
    ms._keep_mask = new_keep_mask

    ms._variants_kept = [v for v, m in zip(ms._variants_kept, mask) if m]
    ms._scores = ms._scores[mask]
    ms._missing = ms._missing[mask]
    ms._sample_assignments = ms._sample_assignments[mask]
    ms.n_variants = ms._scores.shape[0]
    ms._xlims = tuple(
        (float(np.nanmin(ms._scores[:, d])), float(np.nanmax(ms._scores[:, d])))
        for d in range(ms._scores.shape[1])
    )
    ms.global_score_min = tuple(lo for lo, _hi in ms._xlims)
    ms.global_score_max = tuple(hi for _lo, hi in ms._xlims)
    ms.sample_counts = ms._sample_assignments.sum(axis=0)


def combined_calibration_clusters(observations, dataset_names, min_overlap_rows=30):
    """Disjoint fit clusters for a combined (predictor + functional) dataset:
    the functional readouts' own clusters (Fit._select_all_calibration_clusters
    on the functional columns alone, so exactly the functional panel's
    clusters), each joined by every predictor column. Clustering all columns
    together instead lets the predictors -- scored on nearly every missense
    variant -- chain functional readouts that never share a variant into one
    cluster whose cross-covariances have no data behind them (BRCA2: 15 of 21
    functional pairs never co-observed). A functional readout outside every
    functional cluster is dropped, as in the functional panel. Genes without
    a multi-readout functional cluster (a single functional readout, e.g.
    JAG1/MSH2) keep the all-column clustering. Column indices returned
    sorted, clusters most informative first."""
    from ..fit_utils.fit import Fit
    pred_names = set(PREDICTOR_DATASET_NAMES.values())
    P = [i for i, n in enumerate(dataset_names) if n in pred_names]
    F = [i for i, n in enumerate(dataset_names) if n not in pred_names]
    obs = np.asarray(observations, dtype=float)
    if P and len(F) > 1:
        functional = Fit._select_all_calibration_clusters(obs[:, F], min_overlap_rows=min_overlap_rows)
        if functional:
            return [sorted(P + [F[i] for i in c]) for c in functional]
    return Fit._select_all_calibration_clusters(obs, min_overlap_rows=min_overlap_rows)


def build_combined_multiscoreset(
    gene: str,
    functional_scoresets: List[Scoreset],
    functional_dataset_names: List[str],
    predictor_data_dir: str,
    predictors: Optional[List[str]] = None,
    functionally_assayed_variants: Optional[Set[str]] = None,
    require_both_modalities: bool = True,
    predictor_transform: Optional[str] = None,
) -> Optional[MultiPredictorFunctionalScoreset]:
    """Join gene's predictor dimensions (REVEL/MP2/AM) with its functional
    dimensions into one MultiPredictorFunctionalScoreset.

    ``functionally_assayed_variants`` (see get_functionally_assayed_protein_
    variants), if given, splits predictor-CSV P/LP/B/LB leakage (see
    split_predictor_clinvar_leakage) for variants that have functional data
    -- predictor-only variants are unaffected and keep their own label.

    ``require_both_modalities`` (default True): drop any variant that isn't
    observed in AT LEAST ONE predictor dimension AND AT LEAST ONE functional
    dimension (union within each modality, intersection across modalities;
    see select_variants). Without this, MultiPredictorFunctionalScoreset's
    underlying BasicMultiScoreset uses a plain union of IDs across every
    input scoreset -- confirmed on real BRCA1 data (6 dims: 3 predictor + 3
    functional) that this lets single-modality rows dominate: only 60/2835
    variants (2.1%) had every dimension observed, while 1833 (64.6%) had all
    3 predictor dims but ZERO functional dims, and 383 (13.5%) had
    functional data but ZERO predictor dims -- 78% of the "combined"
    population was actually single-modality evidence, contaminating a fit
    that's supposed to isolate cross-modality (functional x predictor)
    correlation structure. Pass False to restore the old union-only
    behavior.

    ``predictor_transform``: see predictors.df_to_basic_scoreset (default
    $MV_PREDICTOR_TRANSFORM, else "none"); functional dimensions are untouched.

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

    predictor_scoresets = [df_to_basic_scoreset(predictor_dfs[p], p, transform=predictor_transform)
                           for p in have]
    if functionally_assayed_variants:
        for ps in predictor_scoresets:
            split_predictor_clinvar_leakage(ps, functionally_assayed_variants)
    predictor_names = [PREDICTOR_DATASET_NAMES.get(p, p) for p in have]

    if gene in ("TP53", "TSC2"):
        # build_functional_scoresets already swapped in tp53.py's (or tsc2.py's) own
        # dedicated panel for TP53 (see its docstring) -- the caller's
        # functional_dataset_names is still the OLD generic-panel name list
        # (same length, wrong names) since callers build it independently
        # of functional_scoresets. Use each scoreset's own name instead of
        # trusting the caller's stale list.
        functional_dataset_names = [s.scoreset_name for s in functional_scoresets]

    combined = MultiPredictorFunctionalScoreset.from_predictor_and_functional(
        predictor_scoresets,
        functional_scoresets,
        predictor_names=predictor_names,
        functional_names=list(functional_dataset_names),
    )

    # Categorical functional dims (TP53's KawOligo; see
    # tp53.build_tp53_functional_scoresets) carry over by name.
    cats = {s.scoreset_name: s.categorical_levels for s in functional_scoresets
            if getattr(s, "categorical_levels", None)}
    if cats:
        combined.categorical_dims = cats

    if require_both_modalities:
        n_predictor_dims = len(predictor_scoresets)
        observed = ~combined.missing
        has_predictor = observed[:, :n_predictor_dims].any(axis=1)
        has_functional = observed[:, n_predictor_dims:].any(axis=1)
        select_variants(combined, has_predictor & has_functional)

    return combined
