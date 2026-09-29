#!/usr/bin/env python3
"""
TP53's reduced-penetrance (RPV) sample reporting -- penetrance histogram,
RPV classification matrix, RPV quadrant plot, per-dimension marginal
densities, and confusion matrix (Figure fig:mv_tp53).

RPV is NOT a disease phenotype the way RET's MEN2/Hirschsprung or CARD11's
BENTA/CADINS are -- it's "reduced penetrance relative to the pathogenic/
benign controls," a different concept with its own custom framing. This
module is deliberately TP53-only and does not attempt a shared abstraction
with mv_analysis/phenotype_evidence.py's disease-group mechanism, even
though both ultimately sit on top of the same generic auxiliary-sample
machinery (`generate_gene_report`'s `rpv_samples` dict, which historically
also serves CARD11's BENTA/CADINS -- that usage belongs to
phenotype_evidence.py's disease-phenotype framing, not here).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
from src.assay_calibration.multivariate_analysis.report_gene import generate_gene_report

from mv_analysis import config as mv_config

# TP53's dedicated MultiScoreset builder's fixed sample-role index for RPV
# (src/assay_calibration/multivariate_data/tp53.py: SAMPLE_NAMES = ["P/LP",
# "B/LB", "gnomAD", "Synonymous", "RPV"]).
TP53_RPV_ROLE = 4


def run_tp53_rpv_report(
    results_json: str = None,
    output_dir: str = None,
    dataset_name: str = "TP53_tp53_mv",
    configs=None,
    ms=None,
    run_kwargs=None,
):
    """Thin wrapper around the already-working `generate_gene_report` for
    TP53's RPV sample -- reproduces the exact figure/table set already seen
    in tmp_scripts/tp53_pca10_run/figures/ (penetrance histogram, RPV
    classification/quadrant, per-dimension marginal densities, confusion),
    with zero new plotting logic.

    ``ms``, if given, skips rebuilding TP53's MultiScoreset (pass the
    output of `config.build_multiscoresets_for_gene_set("tp53")["TP53"]`
    if already built -- this applies TP53_DEFAULT_COLLAPSE_PRESET by
    default, matching what TP53_tp53_mv was actually fit on).

    Returns generate_gene_report's own {config: {"metrics":, "rpv_scores":}}.
    """
    results_json = results_json or mv_config.PAPER_RESULTS_JSON
    output_dir = output_dir or mv_config.PAPER_TP53_RPV_DIR
    if ms is None:
        ms = mv_config.build_multiscoresets_for_gene_set("tp53")["TP53"]

    analysis = build_gene_set_analysis(
        ms, "tp53", results_json, dataset_name=dataset_name,
        auxiliary_pathogenic_indices=[TP53_RPV_ROLE],
    )
    run_kwargs = dict(run_kwargs) if run_kwargs is not None else dict(
        path_percentile=5, min_valid_boots=1,
        reestimate_marginal_weights=False,
        enforce_marginal_monotonicity=False,
        liberal_marginal_monotonicity=False,
    )
    # RPV's vs_P compares two pathogenic-leaning classes, so the normal 5/95
    # two-sided haircut pulls both toward the null and can't resolve them --
    # see MVCalibrationAnalysis.score_rpv_penetrance's docstring: at 5/95,
    # low_pen_rpv is structurally unreachable (0/22 RPV variants classified
    # on real TP53 data); 50/50 is the documented, required convention for
    # RPV specifically. setdefault (not run_kwargs.get(...) or a plain
    # `run_kwargs or dict(...)` fallback above) so this is forced onto
    # WHATEVER run_kwargs the caller passes -- including analyze_mv_pipeline_
    # output.py's generic DEFAULT_RUN_KWARGS-derived dict, which is what the
    # actual CLI/production path (`run_tp53_special_figures` ->
    # `run_tp53_rpv_report(run_kwargs=run_kwargs)`) always supplies (never
    # None), so a plain "only applies to the None-default fallback" fix would
    # never actually reach production.
    run_kwargs.setdefault("aux_path_percentile", 50)
    run_kwargs.setdefault("aux_ben_percentile", 50)
    analysis.run(partial_pattern_mode="trust_global", **run_kwargs)

    return generate_gene_report(
        analysis, "TP53", output_dir, configs=configs,
        rpv_samples={"RPV": TP53_RPV_ROLE},
    )


if __name__ == "__main__":
    run_tp53_rpv_report()
