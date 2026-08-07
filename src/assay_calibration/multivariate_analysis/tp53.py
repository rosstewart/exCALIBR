"""
Analyze driver for TP53: build the ingestion BasicMultiScoreset and run
MVCalibrationAnalysis against fits produced by
``hpc/prepare.py multivariate --gene-set tp53``.

RPV is sample index 4 (auxiliary_pathogenic_indices=[4]), matching
analyze_tp53.ipynb's usage. Not yet run against real fits -- no TP53 fits
exist under the new Fit.generate_fit_jobs-backed pipeline as of this
writing (see the consolidation plan). Ingestion itself
(multivariate_data/tp53.py) is independently verified against
tp53_processed.csv.gz's construction.
"""

from typing import Optional

from ..multivariate_data.tp53 import build_tp53_multiscoreset
from ..multivariate_data.common import gene_set_dataset_label
from .gene_set_analysis import run_gene_set_analysis


def analyze_tp53(
    fits_json_path: str,
    ms=None,
    RPVS_ALL: bool = False,
    **run_kwargs,
):
    if ms is None:
        ms = build_tp53_multiscoreset(RPVS_ALL=RPVS_ALL)

    run_kwargs.setdefault("path_percentile", 5)
    run_kwargs.setdefault("aux_path_percentile", 50)
    run_kwargs.setdefault("aux_ben_percentile", 50)
    run_kwargs.setdefault("min_valid_boots", 1)
    run_kwargs.setdefault("reestimate_marginal_weights", False)
    run_kwargs.setdefault("liberal_marginal_monotonicity", False)
    run_kwargs.setdefault("enforce_marginal_monotonicity", False)

    dataset_name = gene_set_dataset_label("TP53", "tp53")
    return run_gene_set_analysis(
        ms, "TP53", fits_json_path, dataset_name=dataset_name,
        auxiliary_pathogenic_indices=[4], **run_kwargs,
    )
