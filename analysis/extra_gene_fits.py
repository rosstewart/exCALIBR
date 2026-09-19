"""
Extra per-dataset fit/SNV panel plots, supplementary to Figure 4 (not part of
it) — e.g. the RAD51D/XRCC2 pillar-project panels shown alongside Figure 4 in
the manuscript.

Moved out of `analysis.figure4.driver` so that package can ship standalone
(just Figure 4's own inputs) for others to reproduce that one figure without
needing this unrelated, repo-specific supplementary output. Called from
`analyze_pipeline_output.py` alongside the rest of the main analysis instead.

Which dataset(s) to plot is the caller's choice (see `analyze_pipeline_output.py`
section 7a, which specifies RAD51D_IGVF / XRCC2_IGVF) -- this module is
dataset-agnostic and just wires `analysis.legacy_fits.load_scoreset_and_fits`
output into `analysis.fit_hist_snv_plot.plot_figure_panel_a` /
`plot_figure_panel_b` (ported here from
`test/auxiliary_fig_creation/pillar_project_figure4.py`'s
`fit_hist_snv_plot` import, which previously had no corresponding module in
the repo).
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from analysis.legacy_fits import load_scoreset_and_fits
from analysis.fit_hist_snv_plot import plot_figure_panel_a, plot_figure_panel_b


def build_extra_gene_fits(output_dir, dataset_tsv, precomputed_fits, dataset_configs_path, figure_dir, dataset_specs):
    """Build the {fits,snv} panel pair for each dataset in `dataset_specs`.

    Parameters
    ----------
    dataset_specs : list of (dataset, tag) or (dataset, tag, minimal) tuples
        `dataset` is looked up via `analysis.legacy_fits.load_scoreset_and_fits`;
        `tag` names the two output files, `{tag}_fits.png` / `{tag}_snv.png`,
        saved under `figure_dir`. `minimal` (default False) is passed through
        to both panel functions (see their own docstrings for what it hides).
    """
    for spec in dataset_specs:
        dataset, tag = spec[0], spec[1]
        minimal = spec[2] if len(spec) > 2 else False

        try:
            scoreset, indv_summary, fits, score_range, n_c, n_samples, flipped = load_scoreset_and_fits(
                dataset, output_dir=output_dir, dataset_tsv=dataset_tsv,
                precomputed_fits=precomputed_fits, dataset_configs_path=dataset_configs_path,
            )
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  SKIP extra fit plot for {dataset}: {e}")
            continue

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
