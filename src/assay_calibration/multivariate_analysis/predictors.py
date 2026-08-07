"""
Shared analyze-core for the predictor-only (REVEL/MP2/AM) model and the
predictor+functional combined model -- both consume a
BasicMultiScoreset/MultiPredictorFunctionalScoreset built by
multivariate_data/predictors.py or multivariate_data/combined.py, so one
pair of thin wrappers covers both, reusing gene_set_analysis.py's core.

Not yet run against real fits: no predictor-mv fits exist on disk yet
(confirmed this session -- ingest CSVs exist, no `*predictors_mv*` fit
files do), and the combined model is new.
"""

from typing import Optional

import pandas as pd

from ..multivariate_data.predictors import load_predictor_ms, DATASET_SUFFIX
from ..multivariate_data.combined import build_functional_scoresets, build_combined_multiscoreset
from ..multivariate_data.common import gene_set_dataset_label
from .gene_set_analysis import run_gene_set_analysis


def analyze_predictor_gene(
    gene: str,
    fits_json_path: str,
    predictor_data_dir: str,
    ms=None,
    **run_kwargs,
):
    if ms is None:
        ms = load_predictor_ms(gene, predictor_data_dir)

    run_kwargs.setdefault("min_valid_boots", 1)
    run_kwargs.setdefault("reestimate_marginal_weights", False)

    return run_gene_set_analysis(ms, gene, fits_json_path, dataset_suffix=DATASET_SUFFIX, **run_kwargs)


def analyze_combined_gene(
    gene: str,
    fits_json_path: str,
    predictor_data_dir: str,
    functional_df: Optional[pd.DataFrame] = None,
    functional_datasets: Optional[list] = None,
    ms=None,
    **run_kwargs,
):
    if ms is None:
        if functional_df is None or functional_datasets is None:
            raise ValueError(
                "either ms=..., or both functional_df and functional_datasets, must be given"
            )
        functional_scoresets = build_functional_scoresets(functional_df, gene, functional_datasets)
        ms = build_combined_multiscoreset(
            gene, functional_scoresets, functional_datasets, predictor_data_dir,
        )
        if ms is None:
            raise ValueError(f"could not build combined MultiPredictorFunctionalScoreset for {gene}")

    run_kwargs.setdefault("min_valid_boots", 1)
    run_kwargs.setdefault("reestimate_marginal_weights", False)

    dataset_name = gene_set_dataset_label(gene, "combined")
    return run_gene_set_analysis(ms, gene, fits_json_path, dataset_name=dataset_name, **run_kwargs)
