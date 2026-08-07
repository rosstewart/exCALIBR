"""
Analyze driver for LABEL-seq: build the ingestion MultiScoreset for one gene
and run MVCalibrationAnalysis against fits produced by
``hpc/prepare.py multivariate --gene-set labelseq``.

Not yet run against real fits -- no LABEL-seq fits exist under the new
Fit.generate_fit_jobs-backed pipeline as of this writing (see the
consolidation plan). The existing labelseq_with_exc_points.flat.tsv.gz was
built from the historical ad hoc notebook fits and will need to be
regenerated once new fits exist, per the plan's open question on
downstream-artifact timing.
"""

from typing import Dict, Optional

from ..multivariate_data.labelseq import build_labelseq_multiscoresets
from ..multivariate_data.common import gene_set_dataset_label
from .gene_set_analysis import run_gene_set_analysis


def analyze_labelseq_gene(
    gene: str,
    fits_json_path: str,
    gene_ms_map: Optional[Dict] = None,
    **run_kwargs,
):
    if gene_ms_map is None:
        gene_ms_map = build_labelseq_multiscoresets()
    ms = gene_ms_map[gene]

    run_kwargs.setdefault("min_valid_boots", 1)
    run_kwargs.setdefault("reestimate_marginal_weights", False)
    run_kwargs.setdefault("liberal_marginal_monotonicity", False)
    run_kwargs.setdefault("enforce_marginal_monotonicity", False)

    dataset_name = gene_set_dataset_label(gene, "labelseq")
    return run_gene_set_analysis(ms, gene, fits_json_path, dataset_name=dataset_name, **run_kwargs)
