#!/usr/bin/env python3
"""
Per-gene 3D evidence-colored scatter for LABEL-seq genes (e.g. RET): one 3D subplot
per fixed sample class (P/LP, B/LB, gnomAD, Synonymous), plotting a gene's 3 score
dimensions against each other, points colored by MV Tavtigian evidence points.

Reuses the same evidence colormap (`POINT_CMAP`/`TwoSlopeNorm`) as every other
per-variant evidence plot in `visualize_fit.py`, and the same analysis-construction
pattern as `mv_analysis.gene_performance_scatter`'s LABEL-seq loop.
"""
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.assay_calibration.multivariate_data.labelseq import build_labelseq_multiscoresets
from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
from src.assay_calibration.multivariate_analysis.visualize_fit import POINT_CMAP

from mv_analysis.gene_performance_scatter import fast_results_json, RUN_KWARGS

SAMPLE_ROLE_NAMES = ["P/LP", "B/LB", "gnomAD", "Synonymous"]


def _resolve_dim(dataset_names, substr):
    matches = [i for i, name in enumerate(dataset_names) if substr in name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one dataset name containing {substr!r} among "
            f"{dataset_names}, found {len(matches)}: {[dataset_names[i] for i in matches]}"
        )
    return matches[0]


def _compute_3d_evidence_data(
    gene, x_dim, y_dim, z_dim, results_json, config="4c_unc", z_log10=True, ms_map=None,
    run_kwargs=None,
):
    """Runs the MV analysis once and returns everything needed to render one or
    more views (per-sample-class point coordinates/colors, shared axis limits,
    colormap norm) without re-running the bootstrap-scoring analysis per view.

    ``run_kwargs``, if given, overrides/extends the default RUN_KWARGS passed to
    ``analysis.run(...)`` (e.g. ``{"path_percentile": 25}`` to loosen the
    classification thresholds -- ``ben_percentile`` derives automatically as
    ``100 - path_percentile``)."""
    data = _compute_3d_evidence_data_multi(
        gene, x_dim, y_dim, z_dim, results_json, configs=[config], z_log10=z_log10,
        ms_map=ms_map, run_kwargs=run_kwargs,
    )
    out = dict(data)
    out["points"] = data["points_by_config"][config]
    return out


def _compute_3d_evidence_data_multi(
    gene, x_dim, y_dim, z_dim, results_json, configs=("3c_unc", "4c_unc", "5c_unc", "6c_unc"),
    z_log10=True, ms_map=None, run_kwargs=None,
):
    """Same as _compute_3d_evidence_data but scores every config in `configs`
    from a SINGLE analysis.run() call (which computes all fitted configs
    internally regardless), returning a {config: points} dict instead of one
    points array -- lets a multi-config figure (or a before/after percentile
    comparison) reuse one bootstrap-scoring pass rather than repeating it."""
    if ms_map is None:
        ms_map = build_labelseq_multiscoresets()
    ms = ms_map[gene]

    x_i = _resolve_dim(ms.dataset_names, x_dim)
    y_i = _resolve_dim(ms.dataset_names, y_dim)
    z_i = _resolve_dim(ms.dataset_names, z_dim)

    kwargs = {**RUN_KWARGS, **(run_kwargs or {})}
    with fast_results_json(results_json):
        analysis = build_gene_set_analysis(
            ms, gene, results_json, dataset_name=f"{gene}_labelseq_mv",
        )
        analysis.run(partial_pattern_mode="trust_global", **kwargs)
    points_by_config = {
        c: np.asarray(analysis.results[c]["points"], dtype=float) for c in configs
    }

    x_raw, y_raw, z_raw = ms.scores[:, x_i], ms.scores[:, y_i], ms.scores[:, z_i]
    if z_log10:
        if np.any(z_raw[~np.isnan(z_raw)] <= 0):
            raise ValueError(
                f"{gene}'s {z_dim!r} dimension has non-positive values -- "
                "log10 would produce NaN/-inf; pass z_log10=False or decide a floor."
            )
        z_raw = np.log10(z_raw)

    complete = ~(np.isnan(x_raw) | np.isnan(y_raw) | np.isnan(z_raw))
    sa = ms._sample_assignments

    role_masks = []
    for role, name in enumerate(SAMPLE_ROLE_NAMES):
        mask = sa[:, role].astype(bool) & complete
        if mask.sum() > 0:
            role_masks.append((name, mask))

    x_lim = (np.nanmax(x_raw[complete]), np.nanmin(x_raw[complete]))  # flipped
    y_lim = (np.nanmax(y_raw[complete]), np.nanmin(y_raw[complete]))  # flipped
    z_lim = (0.0, np.nanmax(z_raw[complete])) if z_log10 else (
        np.nanmin(z_raw[complete]), np.nanmax(z_raw[complete]))
    max_pt = max(analysis.point_values)
    pt_norm = TwoSlopeNorm(vmin=-max_pt, vcenter=0, vmax=max_pt)

    return {
        "x_raw": x_raw, "y_raw": y_raw, "z_raw": z_raw, "points_by_config": points_by_config,
        "role_masks": role_masks, "x_lim": x_lim, "y_lim": y_lim, "z_lim": z_lim,
        "pt_norm": pt_norm, "max_pt": max_pt,
        "x_label": x_dim, "y_label": y_dim,
        "z_label": f"log10({z_dim})" if z_log10 else z_dim,
        "analysis": analysis,
    }


def _evidence_style(pts, pt_norm, max_pt, min_size=8, max_size=60, min_alpha=0.12, max_alpha=0.95):
    """Per-point (facecolor RGBA, size) so zero-evidence points render small and
    near-transparent, and strong-evidence points render large and opaque --
    baking alpha into the RGBA array (rather than a single scalar `alpha=`)
    since Axes3D.scatter doesn't support a per-point alpha kwarg. Also returns
    a matching edgecolor RGBA array so faint points don't get a heavy outline."""
    frac = np.clip(np.abs(pts) / max_pt, 0.0, 1.0)
    sizes = min_size + (max_size - min_size) * frac
    alphas = min_alpha + (max_alpha - min_alpha) * frac
    face = POINT_CMAP(pt_norm(pts))
    face = np.array(face, dtype=float)
    face[:, 3] = alphas
    edge = np.tile([0.2, 0.2, 0.2, 1.0], (len(pts), 1))
    edge[:, 3] = alphas
    return face, edge, sizes


def _render_3d_evidence_row(fig, gs_row, data, ncols, elev, azim, row_label=None, points=None):
    """Renders one row of (sample-class) 3D subplots at a given (elev, azim), all
    sharing the same axis limits/colormap norm from `data`. Point size/opacity
    scale with |evidence points| (see _evidence_style) -- zero-evidence points are
    small and near-transparent, strong-evidence points are large and opaque.
    `depthshade=False` so matplotlib's own distance-based auto-dimming doesn't
    additionally vary the opacity of same-colored points. ``points``, if given,
    overrides ``data["points"]`` (used for multi-config figures where each row
    is a different config's points array, from the same shared `data`)."""
    if points is None:
        points = data["points"]
    for i, (name, mask) in enumerate(data["role_masks"]):
        ax = fig.add_subplot(gs_row[i], projection="3d")
        face, edge, sizes = _evidence_style(points[mask], data["pt_norm"], data["max_pt"])
        ax.scatter(
            data["x_raw"][mask], data["y_raw"][mask], data["z_raw"][mask],
            c=face, edgecolors=edge, linewidths=0.3, s=sizes, depthshade=False,
        )
        ax.set_xlim(data["x_lim"]); ax.set_ylim(data["y_lim"]); ax.set_zlim(data["z_lim"])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(data["x_label"], fontsize=8)
        ax.set_ylabel(data["y_label"], fontsize=8)
        ax.set_zlabel(data["z_label"], fontsize=8)
        title = f"{name} (n={mask.sum()})"
        if row_label:
            title = f"{row_label}\n{title}"
        ax.set_title(title, fontsize=10)


def plot_labelseq_3d_evidence(
    gene: str,
    x_dim: str, y_dim: str, z_dim: str,
    results_json: str,
    config: str = "4c_unc",
    z_log10: bool = True,
    elev: float = 20.0,
    azim: float = -60.0,
    save_path: Optional[str] = None,
    ms_map=None,
):
    """One 3D subplot per fixed sample class (P/LP, B/LB, gnomAD, Synonymous) for a
    LABEL-seq gene's 3 score dimensions, points colored by MV Tavtigian evidence
    points. ``x_dim``/``y_dim``/``z_dim`` are substrings matched against the gene's
    ``ms.dataset_names`` (e.g. "abundance_HSP90i").

    ``ms_map``, if given, skips rebuilding the ~10-minute LABEL-seq MultiScoreset
    dict (pass the output of ``build_labelseq_multiscoresets()`` if already built).
    """
    data = _compute_3d_evidence_data(
        gene, x_dim, y_dim, z_dim, results_json, config=config, z_log10=z_log10, ms_map=ms_map)

    n = len(data["role_masks"])
    fig = plt.figure(figsize=(5 * n, 5.5))
    gs = fig.add_gridspec(1, n)
    _render_3d_evidence_row(fig, [gs[0, i] for i in range(n)], data, n, elev, azim)

    fig.suptitle(f"{gene}: MV evidence by sample class", fontsize=12)
    mappable = plt.cm.ScalarMappable(norm=data["pt_norm"], cmap=POINT_CMAP)
    cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.6, pad=0.02)
    cbar.set_label("Evidence Points")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved 3D evidence plot to {save_path}")

    return fig


def plot_labelseq_3d_evidence_multiview(
    gene: str,
    x_dim: str, y_dim: str, z_dim: str,
    results_json: str,
    azims: Optional[Sequence[float]] = None,
    elev: float = 20.0,
    views: Optional[Sequence[tuple]] = None,
    config: str = "4c_unc",
    z_log10: bool = True,
    save_path: Optional[str] = None,
    ms_map=None,
):
    """Same as `plot_labelseq_3d_evidence` but rendered from multiple (elev, azim)
    views (one row per view), e.g. to also look "up" from below the low-activity
    floor rather than only rotating azimuth at a fixed elevation. Runs the
    underlying MV analysis only once, reused for every view (rendering is cheap;
    re-scoring bootstraps is not).

    Pass either:
      - ``views``: explicit list of (elev, azim) tuples, e.g.
        ``[(20, -60), (-20, -60), (60, -60)]`` to compare looking down, level,
        and up at the same azimuth; or
      - ``azims`` (legacy): a list of azimuths, all rendered at the single
        fixed ``elev`` -- kept for backwards compatibility.
    Defaults to a mix of azimuth rotation AND elevation (including negative,
    i.e. looking up from below the z=0 floor) if neither is given.
    """
    if views is None:
        if azims is not None:
            views = [(elev, a) for a in azims]
        else:
            views = [
                (20, -60), (20, 30), (20, 120),   # rotate around at a normal downward-looking angle
                (-20, -60), (-20, 30),             # look UP from below the low-activity floor
                (60, -60),                         # look down steeply from above
            ]

    data = _compute_3d_evidence_data(
        gene, x_dim, y_dim, z_dim, results_json, config=config, z_log10=z_log10, ms_map=ms_map)

    n_cols = len(data["role_masks"])
    n_rows = len(views)
    fig = plt.figure(figsize=(5 * n_cols, 5.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)

    for r, (v_elev, v_azim) in enumerate(views):
        _render_3d_evidence_row(
            fig, [gs[r, i] for i in range(n_cols)], data, n_cols, v_elev, v_azim,
            row_label=f"elev={v_elev:g}°, azim={v_azim:g}°",
        )

    fig.suptitle(f"{gene}: MV evidence by sample class (multiple orientations)", fontsize=14)
    mappable = plt.cm.ScalarMappable(norm=data["pt_norm"], cmap=POINT_CMAP)
    cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.4, pad=0.02)
    cbar.set_label("Evidence Points")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved multi-view 3D evidence plot to {save_path}")

    return fig


def plot_labelseq_3d_evidence_multiconfig(
    gene: str,
    x_dim: str, y_dim: str, z_dim: str,
    results_json: str,
    configs: Sequence[str] = ("3c_unc", "4c_unc", "5c_unc", "6c_unc"),
    z_log10: bool = True,
    elev: float = 20.0,
    azim: float = -60.0,
    save_path: Optional[str] = None,
    ms_map=None,
    run_kwargs=None,
):
    """One row per fitted config (3c/4c/5c/6c by default), one column per
    sample class -- lets you compare how the mixture-component count affects
    evidence assignment at a glance. Scores every config from a single
    analysis.run() call (all configs are always fitted together)."""
    data = _compute_3d_evidence_data_multi(
        gene, x_dim, y_dim, z_dim, results_json, configs=configs, z_log10=z_log10,
        ms_map=ms_map, run_kwargs=run_kwargs,
    )

    n_cols = len(data["role_masks"])
    n_rows = len(configs)
    fig = plt.figure(figsize=(5 * n_cols, 5.5 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)

    for r, cfg in enumerate(configs):
        _render_3d_evidence_row(
            fig, [gs[r, i] for i in range(n_cols)], data, n_cols, elev, azim,
            row_label=f"config={cfg}", points=data["points_by_config"][cfg],
        )

    fig.suptitle(f"{gene}: MV evidence by sample class, across configs", fontsize=14)
    mappable = plt.cm.ScalarMappable(norm=data["pt_norm"], cmap=POINT_CMAP)
    cbar = fig.colorbar(mappable, ax=fig.axes, shrink=0.4, pad=0.02)
    cbar.set_label("Evidence Points")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved multi-config 3D evidence plot to {save_path}")

    return fig


def plot_missing_dimension_evidence(
    gene: str,
    x_dim: str, y_dim: str, z_dim: str,
    results_json: str,
    config: str = "4c_unc",
    save_path: Optional[str] = None,
    ms_map=None,
):
    """Companion to plot_labelseq_3d_evidence: variants missing at least one
    of the 3 chosen dimensions can't be placed in the 3D scatter at all, but
    MVCalibrationAnalysis (partial_pattern_mode="trust_global") still scores
    them via whichever dimensions they DO have -- this plots those
    evidence-point values directly (no 3D coordinates needed), one row per
    sample class, points grouped/jittered by which dimension(s) are missing.

    Reuses the same fit/analysis as plot_labelseq_3d_evidence -- no new
    bootstrap fit is run, this only re-scores the existing 3D fit's already-
    computed per-variant points for the rows plot_labelseq_3d_evidence
    silently drops.
    """
    if ms_map is None:
        ms_map = build_labelseq_multiscoresets(genes=[gene])
    ms = ms_map[gene]

    x_i = _resolve_dim(ms.dataset_names, x_dim)
    y_i = _resolve_dim(ms.dataset_names, y_dim)
    z_i = _resolve_dim(ms.dataset_names, z_dim)

    with fast_results_json(results_json):
        analysis = build_gene_set_analysis(
            ms, gene, results_json, dataset_name=f"{gene}_labelseq_mv",
        )
        analysis.run(partial_pattern_mode="trust_global", **RUN_KWARGS)
    points = np.asarray(analysis.results[config]["points"], dtype=float)

    dim_names = [x_dim, y_dim, z_dim]
    dim_nan = {name: np.isnan(ms.scores[:, i]) for name, i in zip(dim_names, [x_i, y_i, z_i])}
    missing_any = dim_nan[x_dim] | dim_nan[y_dim] | dim_nan[z_dim]

    sa = ms._sample_assignments
    max_pt = max(analysis.point_values)
    pt_norm = TwoSlopeNorm(vmin=-max_pt, vcenter=0, vmax=max_pt)

    role_data = []
    for role, name in enumerate(SAMPLE_ROLE_NAMES):
        idx = np.where(sa[:, role].astype(bool) & missing_any)[0]
        if len(idx) == 0:
            continue
        patterns = np.array([
            ", ".join(d for d in dim_names if dim_nan[d][i]) for i in idx
        ])
        role_data.append((name, idx, patterns))

    if not role_data:
        print(f"{gene}: no sample-class variants are missing any of {dim_names} -- nothing to plot.")
        return None

    n = len(role_data)
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.8 * n + 1), sharex=True)
    if n == 1:
        axes = [axes]

    rng = np.random.RandomState(0)
    for ax, (name, idx, patterns) in zip(axes, role_data):
        uniq_patterns = sorted(set(patterns), key=lambda p: (-p.count(","), p))
        y_positions = {p: i for i, p in enumerate(uniq_patterns)}
        y_vals = np.array([y_positions[p] for p in patterns], dtype=float)
        y_jitter = y_vals + rng.uniform(-0.15, 0.15, size=len(y_vals))
        face, edge, sizes = _evidence_style(points[idx], pt_norm, max_pt)
        ax.scatter(points[idx], y_jitter, c=face, edgecolors=edge, linewidths=0.3, s=sizes)
        ax.set_yticks(range(len(uniq_patterns)))
        ax.set_yticklabels([f"missing: {p}" for p in uniq_patterns], fontsize=8)
        ax.set_ylim(-0.5, len(uniq_patterns) - 0.5)
        ax.axvline(0, color="#999", linewidth=0.5, linestyle="--")
        ax.set_title(f"{name} (n={len(idx)})", fontsize=10, loc="left")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("Evidence Points", fontsize=9)
    fig.suptitle(f"{gene}: evidence for variants missing >=1 of {{{', '.join(dim_names)}}}\n"
                 f"(excluded from the 3D scatter; scored via partial-pattern trust_global)",
                 fontsize=11)
    mappable = plt.cm.ScalarMappable(norm=pt_norm, cmap=POINT_CMAP)
    fig.colorbar(mappable, ax=axes, shrink=0.6, pad=0.02, label="Evidence Points")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved missing-dimension evidence plot to {save_path}")

    return fig


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--x-dim", required=True)
    ap.add_argument("--y-dim", required=True)
    ap.add_argument("--z-dim", required=True)
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--config", default="4c_unc")
    ap.add_argument("--no-log10-z", action="store_true")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--azims", default=None,
                     help="Comma-separated azimuths (e.g. '-60,0,60,120') -- if given, "
                          "renders one row per azimuth instead of a single view (--azim ignored).")
    ap.add_argument("--save-path", default=None)
    args = ap.parse_args()

    if args.azims:
        azims = [float(a) for a in args.azims.split(",")]
        plot_labelseq_3d_evidence_multiview(
            args.gene, args.x_dim, args.y_dim, args.z_dim, args.results_json,
            azims=azims, elev=args.elev, config=args.config,
            z_log10=not args.no_log10_z, save_path=args.save_path,
        )
    else:
        plot_labelseq_3d_evidence(
            args.gene, args.x_dim, args.y_dim, args.z_dim, args.results_json,
            config=args.config, z_log10=not args.no_log10_z,
            elev=args.elev, azim=args.azim, save_path=args.save_path,
        )
