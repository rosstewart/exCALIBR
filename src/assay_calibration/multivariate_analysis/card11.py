"""
Analyze driver for CARD11: build the ingestion BasicMultiScoreset and run
MVCalibrationAnalysis against fits produced by
``hpc/prepare.py multivariate --gene-set card11``.

BENTA (GOF, index 4) and CADINS (LOF, index 5) are auxiliary pathogenic
cohorts in exactly the sense TP53's RPV (index 4) is: variants with
orthogonal, non-ClinVar evidence of pathogenicity that shouldn't be lumped
into the main P/LP sample but should still contribute evidence --
``auxiliary_pathogenic_indices=[4, 5]`` uses the identical
MVCalibrationAnalysis mechanism as TP53, just with two mechanism-specific
cohorts (GOF/LOF) instead of one (germline recurrence). No prior analyze
notebook exists for CARD11 to verify this against (prepare_CARD11_data.
ipynb only builds the BasicMultiScoreset, never fits or analyzes it) --
revisit the run_kwargs defaults below once real fits exist.
"""

from ..multivariate_data.card11 import build_card11_multiscoreset
from ..multivariate_data.common import gene_set_dataset_label
from .gene_set_analysis import run_gene_set_analysis


def analyze_card11(fits_json_path: str, ms=None, **run_kwargs):
    if ms is None:
        ms = build_card11_multiscoreset()

    run_kwargs.setdefault("min_valid_boots", 1)
    run_kwargs.setdefault("reestimate_marginal_weights", False)
    run_kwargs.setdefault("liberal_marginal_monotonicity", False)
    run_kwargs.setdefault("enforce_marginal_monotonicity", False)

    dataset_name = gene_set_dataset_label("CARD11", "card11")
    return run_gene_set_analysis(
        ms, "CARD11", fits_json_path, dataset_name=dataset_name,
        auxiliary_pathogenic_indices=[4, 5], **run_kwargs,
    )
