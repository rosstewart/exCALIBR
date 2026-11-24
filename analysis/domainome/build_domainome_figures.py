#!/usr/bin/env python3
"""Supplementary Data 2-style and extra-fit-style figures for the Domainome
(Beltran et al. 2024) calibrations (Reviewer 3).

Reuses the manuscript builders, pointed at the Domainome outputs:
  - analysis.extended_data_appendix.build_appendix_pdf (SD2 "full" style,
    with all_variants=True and score_label="Abundance score")
    -> <figure_dir>/domainome_model_fit_visualizations.pdf
  - with --extra-fits, analysis.extra_gene_fits.build_extra_gene_fits
    (Part B2 style) -> <figure_dir>/<Dataset>_fits.png, _snv.png

Component counts come from --dataset-configs (dataset_configs_domainome.json).
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.extended_data_appendix import build_appendix_pdf  # noqa: E402
from analysis.extra_gene_fits import build_extra_gene_fits  # noqa: E402

BASE = Path("/data/ross/assay_calibration/domainome")

# Display order in the PDF (by gene, domains in sequence order).
PAGE_ORDER = [
    "MECP2_Beltran_2024_MBD",
    "CRX_Beltran_2024_homeobox",
    "TP63_Beltran_2024_SAM",
    "FHL1_Beltran_2024_LIM2",
    "FHL1_Beltran_2024_LIM3",
    "SATB2_Beltran_2024_CUT1",
    "SATB2_Beltran_2024_CUT2",
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=str(BASE / "exc_domainome_clinvar2025_calib"))
    p.add_argument("--dataset-tsv", default=str(BASE / "excalibr_domainome.integrated.tsv.gz"))
    p.add_argument("--precomputed-fits",
                   default=str(BASE / "jobs_domainome_clinvar2025_2c3c_1000b" / "bootstrap_results.json.gz"))
    p.add_argument("--dataset-configs", default=str(BASE / "dataset_configs_domainome.json"))
    p.add_argument("--figure-dir", default=str(BASE / "figures"))
    p.add_argument("--pdf-name", default="domainome_model_fit_visualizations.pdf")
    p.add_argument("--n-jobs", type=int, default=7)
    p.add_argument("--extra-fits", action="store_true",
                   help="Also write the Part B2-style <Dataset>_fits.png / _snv.png pairs")
    args = p.parse_args()

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    configured = set(json.load(open(args.dataset_configs)))
    datasets = [d for d in PAGE_ORDER if d in configured]
    datasets += sorted(configured - set(datasets))  # any not in PAGE_ORDER go last
    loader_kw = dict(output_dir=args.output_dir, dataset_tsv=args.dataset_tsv,
                     precomputed_fits=args.precomputed_fits, dataset_configs_path=args.dataset_configs)

    # Protein-level library (all 19 substitutions per position): plot every
    # scored variant, not only SNV-accessible ones; scores are aPCA abundance.
    build_appendix_pdf(
        dataset_list=datasets,
        output_path=figure_dir / args.pdf_name,
        plot_thresholds=True,
        n_jobs=args.n_jobs,
        all_variants=True,
        score_label="Abundance score",
        **loader_kw,
    )

    if not args.extra_fits:
        return
    build_extra_gene_fits(
        args.output_dir, args.dataset_tsv, args.precomputed_fits, args.dataset_configs, figure_dir,
        dataset_specs=[(d, d) for d in datasets],
    )


if __name__ == "__main__":
    main()
