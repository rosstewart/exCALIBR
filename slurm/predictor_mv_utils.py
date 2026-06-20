"""
Shared helpers for the multivariate predictor calibration flow.

Used by both prepare_batch_jobs_single_predictor_multivariate.py (job
generation) and the analysis side (analyze_mv_results.py / notebooks)
so the on-disk layout, ID alignment, and sample-column convention stay
in one place.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make src/ importable regardless of cwd
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.assay_calibration.data_utils.dataset import (
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


def df_to_basic_scoreset(df: pd.DataFrame, predictor: str) -> BasicScoreset:
    """Convert one per-predictor DataFrame into a BasicScoreset with ids.

    `protein_variant` is used as the alignment ID. Rows that don't fall
    in any of the SAMPLE_COLUMNS (i.e. unlabeled) are dropped so the
    EM only sees rows with a class assignment.
    """
    df = df.copy()
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
    )


def build_basic_multi_scoreset(
    gene: str,
    predictor_dfs: Dict[str, pd.DataFrame],
) -> Tuple[Optional[BasicMultiScoreset], object]:
    """Build a BasicMultiScoreset for one gene.

    Returns ``(ms, dataset_names)`` on success or ``(None, reason_str)``
    if the gene is missing any of the three predictors or any of the
    per-predictor BasicScoresets fails to build.
    """
    have = [p for p in PREDICTORS if p in predictor_dfs]
    missing = [p for p in PREDICTORS if p not in predictor_dfs]
    if missing:
        return None, f"missing predictor(s): {missing}"

    scoresets = []
    dataset_names = []
    for p in PREDICTORS:
        try:
            bs = df_to_basic_scoreset(predictor_dfs[p], p)
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


def load_predictor_ms(gene: str, data_dir: str) -> BasicMultiScoreset:
    """Convenience loader: build the BasicMultiScoreset for one gene.

    Raises ``ValueError`` if the gene cannot be assembled (missing
    predictors, empty CSVs, etc.) — easier to use from notebooks than
    the (ms, reason) tuple from build_basic_multi_scoreset.
    """
    by_gene = load_predictor_data(data_dir, genes=[gene])
    if gene not in by_gene:
        raise ValueError(f"No predictor CSVs found under {data_dir}/{gene}/")
    ms, info = build_basic_multi_scoreset(gene, by_gene[gene])
    if ms is None:
        raise ValueError(f"Could not build BasicMultiScoreset for {gene}: {info}")
    return ms
