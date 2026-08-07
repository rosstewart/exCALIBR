"""
Full per-gene report: confusion matrix + classification metrics, the
marginal-evidence/skew-normal-fit plot, and (for each low-penetrance-style
auxiliary sample given -- TP53's RPV, CARD11's BENTA/CADINS, etc.) that
sample's penetrance table + its plots -- one call per config, reusing the
single (expensive) precomputed bootstrap-fit data for both the main plot
and the quadrant plot so it isn't computed twice.

Default partial_pattern_mode is "local_unmixing" ("pu_unmix"): on TP53/
LABEL-seq(BRAF)/CARD11 it's empirically identical to every other mode
(checked directly against real fits), but it's the most statistically
principled of the four for cases where that stops being true, at no cost
here -- see the mv_calibration consolidation discussion this was designed
from.
"""

import os

import matplotlib.pyplot as plt

from ..plot_utils.utils import compute_classification_metrics
from . import eval_plot_utils as epu
from .visualize_fit import (
    precompute_mv_plot_data,
    precompute_mv_plot_data_cached,
    render_mv_plot_data,
    plot_rpv_quadrant,
    plot_component_densities,
)


def confusion_and_metrics(analysis, config):
    """(confusion_df, metrics_dict) for one config's primary P/LP vs B/LB."""
    r = analysis.results.get(config)
    if r is None:
        return None, None
    ms = analysis.ms
    sa = ms.sample_assignments
    points = r["points"]
    plp_mask = sa[:, analysis.p_idx].astype(bool) if analysis.p_idx is not None else None
    blb_mask = sa[:, analysis.b_idx].astype(bool) if analysis.b_idx is not None else None
    if plp_mask is None or blb_mask is None:
        return None, None
    sub_mask = plp_mask | blb_mask
    labels = plp_mask[sub_mask].astype(int)
    cm = epu.points_to_confusion(labels, points[sub_mask])
    metrics = compute_classification_metrics(cm)
    return cm, metrics


def _projection_for_dims(D):
    """(projection, max_lr_pairs) so D<=3 shows full pairwise combos and D>3
    shows only 1D marginals -- neither ever falls back to UMAP dim-reduction.

    D==2 is unaffected by this (render_mv_plot_data always routes D==2 through
    the native _plot_mv_2d regardless of the projection value).
    """
    if D <= 3:
        return "pairwise", 10
    return "marginal_only", 0


def generate_gene_report(
    analysis,
    gene: str,
    output_dir: str,
    configs=None,
    rpv_samples=None,
    n_grid=120,
    cache_dir=None,
    force_recompute=False,
):
    """Write, per config: confusion_matrix.txt (+ metrics), the marginal/
    skew-normal-fit plot (PNG), and -- for each entry in ``rpv_samples`` --
    that auxiliary sample's RPV-style penetrance table (CSV) plus its
    classification-matrix, penetrance-histogram, and quadrant plots (PNG).

    ``rpv_samples`` : dict {display_name: fixed_idx}, e.g. {"RPV": 4} for
    TP53, or {"BENTA": 4, "CADINS": 5} for CARD11 -- all entries reuse the
    SAME precomputed bootstrap-fit data for a given config (computed once),
    rather than recomputing it per auxiliary sample.

    ``cache_dir`` : if given, the expensive per-config bootstrap-sweep data
    (precompute_mv_plot_data's output) is cached there via
    precompute_mv_plot_data_cached -- reruns with the same analysis/config/
    n_grid reuse it instead of recomputing, so iterating on plot aesthetics
    is cheap. Defaults to ``{output_dir}/.precompute_cache`` when
    unspecified but non-None desired: pass ``cache_dir=False`` to disable
    caching entirely. ``force_recompute`` bypasses an existing cache entry.

    Returns {config: {"metrics": dict, "rpv_scores": {name: DataFrame}}}.
    """
    os.makedirs(output_dir, exist_ok=True)
    configs = configs or [c for c in analysis.configs if analysis.results.get(c) is not None]
    rpv_samples = rpv_samples or {}

    if cache_dir is None:
        cache_dir = os.path.join(output_dir, ".precompute_cache")

    D = analysis.ms.scores.shape[1]
    projection, max_lr_pairs = _projection_for_dims(D)

    report = {}
    for config in configs:
        print(f"\n=== {gene} / {config} ===")
        cm, metrics = confusion_and_metrics(analysis, config)
        if cm is not None:
            with open(os.path.join(output_dir, f"{gene}_{config}_confusion.txt"), "w") as f:
                f.write(f"{gene} / {config}\n\n")
                f.write(cm.to_string())
                f.write("\n\nMetrics:\n")
                for k, v in metrics.items():
                    f.write(f"  {k}: {v}\n")
            print("Confusion matrix:\n", cm)
            print("Metrics:", {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in metrics.items()})

            # Plotted confusion matrix (plain P/LP vs B/LB) -- always generated,
            # independent of whether any auxiliary/RPV-like sample is provided.
            fig_cm0, ax_cm0 = plt.subplots(figsize=(4.5, 3))
            epu._draw_cm_heatmap(cm, ax_cm0)
            ax_cm0.set_title(f"{gene} {config} — P/LP vs B/LB", fontsize=12, fontweight="bold")
            p = os.path.join(output_dir, f"{gene}_{config}_confusion.png")
            fig_cm0.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig_cm0)
            print(f"  Saved {p}")

        print("  Precomputing bootstrap-fit data for plots...")
        if cache_dir is False:
            precomputed = precompute_mv_plot_data(analysis, config, n_grid=n_grid, projection=projection)
        else:
            precomputed = precompute_mv_plot_data_cached(
                analysis, config, cache_dir, n_grid=n_grid, projection=projection,
                force_recompute=force_recompute,
            )

        fig, _info = render_mv_plot_data(precomputed, projection=projection, max_lr_pairs=max_lr_pairs)
        fig_path = os.path.join(output_dir, f"{gene}_{config}_mv_calibration.png")
        fig.savefig(fig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fig_path}")

        density_figs = plot_component_densities(precomputed)
        dataset_names = getattr(analysis.ms, "dataset_names", [f"dim{d}" for d in range(D)])
        for dim, (fig_d, _axes_d) in density_figs.items():
            dim_name = dataset_names[dim] if dim < len(dataset_names) else f"dim{dim}"
            safe_name = "".join(c if c.isalnum() else "_" for c in dim_name)
            p = os.path.join(output_dir, f"{gene}_{config}_dim{dim}_{safe_name}_densities.png")
            fig_d.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig_d)
            print(f"  Saved {p}")

        # P/LP variants the primary model itself called indeterminate (points==0)
        # -- shown as a 4th row alongside B/LB, P/LP, and the auxiliary sample.
        plp_mask = analysis.ms.sample_assignments[:, analysis.p_idx].astype(bool) \
            if analysis.p_idx is not None else None
        indet_mask = (plp_mask & (analysis.results[config]["points"] == 0)) \
            if plp_mask is not None else None

        rpv_scores_by_name = {}
        for rpv_sample_name, rpv_fixed_idx in rpv_samples.items():
            rpv_scores = analysis.score_rpv_penetrance(config, fixed_idx=rpv_fixed_idx)
            rpv_scores_by_name[rpv_sample_name] = rpv_scores
            rpv_csv = os.path.join(output_dir, f"{gene}_{config}_{rpv_sample_name}_scores.csv")
            rpv_scores.to_csv(rpv_csv)
            print(f"  Saved {rpv_csv} ({len(rpv_scores)} variants, "
                  f"class counts: {rpv_scores['rpv_class'].value_counts().to_dict()})")

            sample_idx_map = {"B/LB": analysis.b_idx, "P/LP": analysis.p_idx,
                               rpv_sample_name: rpv_fixed_idx}
            extra_plp_label = "P/LP (indet.)"

            fig_cm, _ = epu.plot_rpv_classification_matrix(
                rpv_scores, analysis.ms, sample_idx_map=sample_idx_map,
                class_label=rpv_sample_name,
                extra_plp_mask=indet_mask, extra_plp_label=extra_plp_label,
                title=f"{gene} {config} — {rpv_sample_name} classification")
            p = os.path.join(output_dir, f"{gene}_{config}_{rpv_sample_name}_classification.png")
            fig_cm.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig_cm)
            print(f"  Saved {p}")

            fig_hist, _ = epu.plot_penetrance_score_hist(
                rpv_scores, analysis.ms, sample_idx_map=sample_idx_map,
                class_label=rpv_sample_name,
                extra_plp_mask=indet_mask, extra_plp_label=extra_plp_label,
                title=f"{gene} {config} — {rpv_sample_name} penetrance score")
            p = os.path.join(output_dir, f"{gene}_{config}_{rpv_sample_name}_penetrance_hist.png")
            fig_hist.savefig(p, dpi=100, bbox_inches="tight")
            plt.close(fig_hist)
            print(f"  Saved {p}")

            fig_q = plot_rpv_quadrant(precomputed, fixed_idx=rpv_fixed_idx, class_label=rpv_sample_name)
            if fig_q is not None:
                fig_q, _ = fig_q
                p = os.path.join(output_dir, f"{gene}_{config}_{rpv_sample_name}_quadrant.png")
                fig_q.savefig(p, dpi=100, bbox_inches="tight")
                plt.close(fig_q)
                print(f"  Saved {p}")

        report[config] = {"metrics": metrics, "rpv_scores": rpv_scores_by_name}

    return report
