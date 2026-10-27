"""
Shared helpers for the multivariate predictor calibration flow.

Used by both hpc/prepare.py's predictor-mv job generation and the analysis
side (src/assay_calibration/multivariate_analysis/predictors.py) so the
on-disk layout, ID alignment, and sample-column convention stay in one place.
"""

import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data_utils.dataset import (
    BasicScoreset,
    BasicMultiScoreset,
)


PREDICTORS: Tuple[str, ...] = ("REVEL", "MP2", "AM")
PREDICTOR_DATASET_NAMES: Dict[str, str] = {
    "REVEL": "REVEL",
    "MP2": "MutPred2",
    "AM": "AlphaMissense",
}
SAMPLE_COLUMNS: List[str] = ["0", "1", "2"]  # P/LP, B/LB, gnomAD

DEFAULT_GENES: Tuple[str, ...] = (
    "BRCA1", "BRCA2", "F9", "JAG1", "MSH2", "SCN5A", "TP53", "TSC2",
)
DATASET_SUFFIX = "_predictors_mv"


# Monotone, one-to-one transforms of the [0,1]-bounded predictor scores, for
# fitting on an unbounded scale (the bound at 1 -- AlphaMissense especially --
# otherwise pins component Gamma to the ridge). A one-to-one transform leaves
# every variant's LR f_P/f_B unchanged (the Jacobian cancels), so evidence
# still maps back onto the raw scores. "rank_int" is the rank-based inverse
# normal transform over every scored row of the gene's predictor CSV: one
# fixed map per gene and predictor, never recomputed per bootstrap.
PREDICTOR_TRANSFORMS: Tuple[str, ...] = ("none", "logit", "rank_int")
PREDICTOR_TRANSFORM_ENV = "MV_PREDICTOR_TRANSFORM"
LOGIT_EPS = 5e-5  # half of AlphaMissense's 1e-4 score resolution


def resolve_predictor_transform(transform: Optional[str] = None) -> str:
    """``transform``, else $MV_PREDICTOR_TRANSFORM, else "none"."""
    t = transform or os.environ.get(PREDICTOR_TRANSFORM_ENV) or "none"
    if t not in PREDICTOR_TRANSFORMS:
        raise ValueError(f"unknown predictor transform {t!r}; choose from {PREDICTOR_TRANSFORMS}")
    return t


def transform_predictor_scores(scores: np.ndarray, reference: np.ndarray, transform: str) -> np.ndarray:
    """Apply ``transform`` to ``scores``; "rank_int" ranks them within
    ``reference`` (the gene's full score column for this predictor)."""
    s = np.asarray(scores, dtype=float)
    if transform == "none":
        return s
    if transform == "logit":
        if np.nanmin(s) < 0 or np.nanmax(s) > 1:
            raise ValueError("logit transform needs scores in [0, 1]")
        p = np.clip(s, LOGIT_EPS, 1 - LOGIT_EPS)
        return np.log(p) - np.log1p(-p)
    if transform == "rank_int":
        from scipy.stats import norm, rankdata
        ref = np.sort(np.asarray(reference, dtype=float)[~np.isnan(reference)])
        u, first = np.unique(ref, return_index=True)
        counts = np.diff(np.append(first, len(ref)))
        mid_rank = first + (counts + 1) / 2.0                 # average rank of ties, 1-based
        q = norm.ppf((mid_rank - 0.5) / len(ref))
        # scores in the reference map exactly; anything else interpolates
        return np.interp(s, u, q)
    raise ValueError(f"unknown predictor transform {transform!r}")


def predictor_dataset_label(gene: str) -> str:
    """Canonical key used in fits results JSON and on-disk save_dir."""
    return f"{gene}{DATASET_SUFFIX}"


def load_predictor_data(data_dir: str,
                        genes: Optional[List[str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load predictor CSVs into ``{gene: {predictor: df}}``.

    Skips predictor files that are missing from disk; the caller decides
    whether the gene has enough coverage to proceed.
    """
    target_genes = list(genes) if genes is not None else list(DEFAULT_GENES)

    by_gene: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
    for gene in target_genes:
        for predictor in PREDICTORS:
            csv_path = f"{data_dir}/{gene}/{gene}_{predictor}.csv.gz"
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, compression="gzip")
            required = {"protein_variant", "score", "sample_assignments"}
            missing = required - set(df.columns)
            if missing:
                print(f"  {gene}/{predictor}: missing columns {missing}, skipping")
                continue
            by_gene[gene][predictor] = df
    return dict(by_gene)


def df_to_basic_scoreset(df: pd.DataFrame, predictor: str, standardize: bool = False,
                         transform: Optional[str] = None) -> BasicScoreset:
    """Convert one per-predictor DataFrame into a BasicScoreset with ids.

    `protein_variant` is used as the alignment ID. Rows that don't fall
    in any of the SAMPLE_COLUMNS (i.e. unlabeled) are dropped so the
    EM only sees rows with a class assignment.

    ``standardize`` (default False): fit this predictor's EM on z-scored
    (0 mean, unit std) scores instead of the raw ~[0,1]-bounded scale --
    see BasicScoreset's own docstring/comment for why. Each predictor gets
    its own independent mean/std (computed from its own score column).

    ``transform`` (see PREDICTOR_TRANSFORMS; default $MV_PREDICTOR_TRANSFORM,
    else "none"): applied to the scores before anything else.
    """
    transform = resolve_predictor_transform(transform)
    df = df.copy()
    if transform != "none":
        reference = df["score"].to_numpy(dtype=float)
        df["score"] = transform_predictor_scores(reference, reference, transform)
    df = df.dropna(subset=["score", "protein_variant", "sample_assignments"])

    onehot = (
        df["sample_assignments"]
        .astype(str)
        .str.get_dummies(sep=",")
        .reindex(columns=SAMPLE_COLUMNS, fill_value=0)
        .astype(bool)
        .to_numpy()
    )
    keep = onehot.any(axis=1)
    if not keep.all():
        df = df.loc[keep].reset_index(drop=True)
        onehot = onehot[keep]

    return BasicScoreset(
        scores=df["score"].to_numpy(dtype=float),
        sample_assignments=onehot,
        ids=df["protein_variant"].astype(str).to_numpy(),
        scoreset_name=PREDICTOR_DATASET_NAMES.get(predictor, predictor),
        standardize=standardize,
    )


def build_basic_multi_scoreset(
    gene: str,
    predictor_dfs: Dict[str, pd.DataFrame],
    standardize: bool = False,
    transform: Optional[str] = None,
) -> Tuple[Optional[BasicMultiScoreset], object]:
    """Build a BasicMultiScoreset for one gene.

    Returns ``(ms, dataset_names)`` on success or ``(None, reason_str)``
    if the gene is missing any of the three predictors or any of the
    per-predictor BasicScoresets fails to build.

    ``standardize`` (default False), ``transform``: see df_to_basic_scoreset.
    """
    have = [p for p in PREDICTORS if p in predictor_dfs]
    missing = [p for p in PREDICTORS if p not in predictor_dfs]
    if missing:
        return None, f"missing predictor(s): {missing}"

    scoresets = []
    dataset_names = []
    for p in PREDICTORS:
        try:
            bs = df_to_basic_scoreset(predictor_dfs[p], p, standardize=standardize, transform=transform)
        except (ValueError, KeyError) as e:
            return None, f"BasicScoreset({p}) failed: {e}"
        if len(bs.scores) == 0:
            return None, f"BasicScoreset({p}) is empty after filtering"
        scoresets.append(bs)
        dataset_names.append(PREDICTOR_DATASET_NAMES[p])

    try:
        ms = BasicMultiScoreset(scoresets, dataset_names=dataset_names)
    except (ValueError, KeyError) as e:
        return None, f"BasicMultiScoreset failed: {e}"

    return ms, dataset_names


def load_predictor_ms(gene: str, data_dir: str, standardize: bool = False,
                      transform: Optional[str] = None) -> BasicMultiScoreset:
    """Convenience loader: build the BasicMultiScoreset for one gene.

    Raises ``ValueError`` if the gene cannot be assembled (missing
    predictors, empty CSVs, etc.) — easier to use from notebooks than
    the (ms, reason) tuple from build_basic_multi_scoreset.

    ``standardize`` (default False), ``transform``: see df_to_basic_scoreset.
    """
    by_gene = load_predictor_data(data_dir, genes=[gene])
    if gene not in by_gene:
        raise ValueError(f"No predictor CSVs found under {data_dir}/{gene}/")
    ms, info = build_basic_multi_scoreset(gene, by_gene[gene], standardize=standardize, transform=transform)
    if ms is None:
        raise ValueError(f"Could not build BasicMultiScoreset for {gene}: {info}")
    return ms
