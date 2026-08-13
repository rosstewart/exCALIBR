"""
RAD51D / XRCC2 / BARD1 extra fit plots — supplementary to Figure 4, not part
of it.

Moved out of `analysis.figure4.driver` so that package can ship standalone
(just Figure 4's own inputs) for others to reproduce that one figure without
needing this unrelated, repo-specific supplementary output. Called from
`analyze_pipeline_output.py` alongside the rest of the main analysis instead.

TODO: `fit_hist_snv_plot` module not found in repo, skipping. Only the import
statement `from fit_hist_snv_plot import plot_figure_panel_a, plot_figure_panel_b`
exists in the legacy script (test/auxiliary_fig_creation/pillar_project_figure4.py);
no file defining `fit_hist_snv_plot` was found anywhere in this repo. Rather
than fabricate an implementation, this prints a warning and returns if it's
still missing.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from analysis.legacy_fits import load_scoreset_and_fits


def build_extra_gene_fits(output_dir, dataset_tsv, precomputed_fits, dataset_configs_path, figure_dir):
    try:
        import fit_hist_snv_plot  # noqa: F401
    except ImportError:
        print(
            "  SKIP RAD51D/XRCC2/BARD1 extra fit plots: 'fit_hist_snv_plot' module "
            "not found in repo (only its import statement exists in the legacy "
            "script test/auxiliary_fig_creation/pillar_project_figure4.py) — "
            "TODO: port/locate this module if these plots are needed."
        )
        return

    # If fit_hist_snv_plot is ever added to the repo, wire it up here following
    # the same load_scoreset_and_fits(dataset) pattern as
    # figure4.driver._load_msh2_calibration, for datasets RAD51D_unpublished /
    # XRCC2_unpublished / BARD1_unpublished, then call
    # fit_hist_snv_plot.plot_figure_panel_a / plot_figure_panel_b and save to
    # figure_dir / {"xrcc2","rad51d","bard1"}_{fits,snv}.png.
    from fit_hist_snv_plot import plot_figure_panel_a, plot_figure_panel_b  # noqa: F401

    for dataset, tag in [
        ("RAD51D_unpublished", "rad51d"),
        ("XRCC2_unpublished", "xrcc2"),
        ("BARD1_unpublished", "bard1"),
    ]:
        try:
            scoreset, indv_summary, fits, score_range, n_c, n_samples, flipped = load_scoreset_and_fits(
                dataset, output_dir=output_dir, dataset_tsv=dataset_tsv,
                precomputed_fits=precomputed_fits, dataset_configs_path=dataset_configs_path,
            )
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  SKIP extra fit plot for {dataset}: {e}")
            continue

        minimal = dataset == "BARD1_unpublished"
        fig_a = plot_figure_panel_a(
            scoreset, indv_summary, fits, score_range, flipped, n_samples,
            layout='vertical', minimal=minimal, figsize=(6.6, 7.3),
        )
        fig_b = plot_figure_panel_b(
            scoreset, indv_summary, score_range, flipped,
            use_twin_axes=True, minimal=minimal, figsize=(6.6, 7.3),
        )
        fig_a.savefig(figure_dir / f"{tag}_fits.png", dpi=300, bbox_inches='tight')
        fig_b.savefig(figure_dir / f"{tag}_snv.png", dpi=300, bbox_inches='tight')
        plt.close(fig_a)
        plt.close(fig_b)
