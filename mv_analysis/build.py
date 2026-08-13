"""
Thin wrapper around hpc/prepare.py's gene-set ingestion dispatch -- reused
directly (not duplicated) so mv_analysis and the job-generation pipeline
can never silently disagree on how a gene-set's MultiScoreset is built.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "hpc")):
    if p not in sys.path:
        sys.path.insert(0, p)

import prepare  # hpc/prepare.py
from src.assay_calibration.multivariate_data.common import gene_set_dataset_label


def build_ms(args):
    """(gene, ms, dataset_name) for one --gene-set/--gene combination,
    using the exact same dispatch run_mv_analysis.py (now this module's
    caller) always has.
    """
    if args.gene_set == "predictor-mv":
        from src.assay_calibration.multivariate_data.predictors import (
            load_predictor_ms, predictor_dataset_label,
        )
        if not args.gene:
            raise SystemExit("--gene required for --gene-set predictor-mv")
        ms = load_predictor_ms(args.gene, args.data_dir or args.predictor_data_dir)
        return args.gene, ms, predictor_dataset_label(args.gene)

    gene_ms_map = prepare._load_and_filter_gene_ms_map(args.gene_set, args)
    if args.gene_set == "tp53":
        gene = "TP53"
    elif args.gene_set == "card11":
        gene = "CARD11"
    elif args.gene_set == "fgfr" and not args.fgfr_separate:
        gene = "FGFR_combined"
    else:
        if not args.gene:
            raise SystemExit(
                f"--gene required for --gene-set {args.gene_set} "
                f"(available: {sorted(gene_ms_map)})"
            )
        gene = args.gene
    if gene not in gene_ms_map:
        raise SystemExit(f"{gene!r} not found; available: {sorted(gene_ms_map)}")
    return gene, gene_ms_map[gene], gene_set_dataset_label(gene, args.gene_set)
