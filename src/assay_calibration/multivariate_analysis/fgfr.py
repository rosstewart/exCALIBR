"""
Analyze driver for FGFR: build the ingestion MultiScoreset and run
MVCalibrationAnalysis against fits produced by
``hpc/prepare.py multivariate --gene-set fgfr``.

Not yet run against real fits -- no FGFR fits exist under the new
Fit.generate_fit_jobs-backed pipeline as of this writing (see the
consolidation plan). Ingestion itself (multivariate_data/fgfr.py) is
independently verified against the raw annotated file.
"""

from typing import Dict, Optional

from ..multivariate_data.fgfr import build_fgfr_multiscoresets
from ..multivariate_data.common import gene_set_dataset_label
from .gene_set_analysis import run_gene_set_analysis


def analyze_fgfr_gene(
    gene: str,
    fits_json_path: str,
    gene_ms_map: Optional[Dict] = None,
    **run_kwargs,
):
    if gene_ms_map is None:
        gene_ms_map = build_fgfr_multiscoresets(genes=[gene])
    ms = gene_ms_map[gene]

    run_kwargs.setdefault("path_percentile", 25)
    run_kwargs.setdefault("min_valid_boots", 1)
    run_kwargs.setdefault("reestimate_marginal_weights", False)
    run_kwargs.setdefault("liberal_marginal_monotonicity", False)
    run_kwargs.setdefault("enforce_marginal_monotonicity", False)

    dataset_name = gene_set_dataset_label(gene, "fgfr")
    return run_gene_set_analysis(ms, gene, fits_json_path, dataset_name=dataset_name, **run_kwargs)
