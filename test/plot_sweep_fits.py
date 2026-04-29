"""
Side-by-side comparison of CFUSN fits across β values.

Produces one figure per predictor dimension (1-D marginals) and one figure
per predictor pair (2-D joint density contours + scatter).

Layout per 1-D figure:
  Rows    = samples (P/LP, B/LB, gnomAD)
  Columns = β configs  (+1 data-only column on the left)
  X-axis shared; Y-axis shared within each row.
  Each panel: sample histogram + fitted mixture density + component sub-curves.

Layout per 2-D pair figure:
  Same rows/columns as above.
  Each panel: scatter of focal sample (others muted) + fitted density contours.

Usage
-----
    python test/plot_sweep_fits.py /tmp/sweep_brca1.pkl \\
        --data-dir /path/to/predictor_scores/single_gene_calibration_data

    # save to /tmp/out/BRCA1_REVEL_sweep.png, BRCA1_REVEL_vs_MutPred2_sweep.png …
    python test/plot_sweep_fits.py /tmp/sweep_brca1.pkl \\
        --data-dir ... --out-dir /tmp/out
"""

import argparse
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[1]))

from visualize_fit import _sn_logpdf
from predictor_mv_utils import load_predictor_ms

SAMPLE_COLORS = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
SAMPLE_ALPHAS = [0.6, 0.6, 0.3, 0.5]
SAMPLE_CMAPS  = ['Reds', 'Blues', 'Greys', 'Greens']


# ──────────────────────────────────────
# Density helpers
# ──────────────────────────────────────

def _marginal_density(params, w_s, x_grid_2d, K):
    """1-D marginal mixture density.  x_grid_2d is (n, p) with NaN except one col."""
    comp_logpdfs = [_sn_logpdf(x_grid_2d, *params[c]) for c in range(K)]
    log_terms = [np.log(w_s[c] + 1e-300) + comp_logpdfs[c] for c in range(K)]
    density = np.exp(logsumexp(log_terms, axis=0))
    comp_densities = [np.exp(lt) for lt in log_terms]
    return density, comp_densities


def _joint_density_grid(params, w_s, xi_grid, xj_grid, dim_i, dim_j, p, K):
    """2-D joint marginal density on a grid in (dim_i, dim_j), NaN elsewhere."""
    X1, X2 = np.meshgrid(xi_grid, xj_grid, indexing='ij')
    grid_pts = np.full((X1.size, p), np.nan)
    grid_pts[:, dim_i] = X1.ravel()
    grid_pts[:, dim_j] = X2.ravel()
    log_d = logsumexp(
        [np.log(w_s[c] + 1e-300) + _sn_logpdf(grid_pts, *params[c]) for c in range(K)],
        axis=0,
    )
    return np.exp(log_d).reshape(len(xi_grid), len(xj_grid))


# ──────────────────────────────────────
# 1-D marginal figures (one per predictor)
# ──────────────────────────────────────

def _draw_hist(ax, obs, color, alpha):
    if len(obs):
        ax.hist(obs, bins=50, density=True, alpha=alpha,
                color=color, edgecolor="none")


def _plot_one_predictor(dim, predictor_name, rows, scores, sa, sample_names,
                        sample_counts, gene, K, p, x_grid, n_grid, dpi, out_path):
    S = len(sample_names)
    n_cols = 1 + len(rows)
    n_rows = S
    comp_colors = plt.cm.Set2(np.linspace(0, 1, max(K, 3)))

    obs_per_sample = []
    for s in range(S):
        obs = scores[sa[:, s].astype(bool), dim]
        obs_per_sample.append(obs[~np.isnan(obs)])

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        sharex=True, sharey="row",
        squeeze=False,
    )

    # Column 0: raw data only
    for row_idx in range(S):
        ax = axes[row_idx, 0]
        s = row_idx
        color = SAMPLE_COLORS[s % len(SAMPLE_COLORS)]
        _draw_hist(ax, obs_per_sample[s], color, SAMPLE_ALPHAS[s % len(SAMPLE_ALPHAS)])
        ax.grid(alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=7)
        ax.set_ylabel(f"{sample_names[s]}\n(n={len(obs_per_sample[s])})",
                      fontsize=8, fontweight="bold", color=color)
        if row_idx == 0:
            ax.set_title("data only", fontsize=8, fontweight="bold")
        if row_idx == n_rows - 1:
            ax.set_xlabel(predictor_name, fontsize=8)

    # Fit columns
    x_2d = np.full((n_grid, p), np.nan)
    x_2d[:, dim] = x_grid

    for fit_idx, r in enumerate(rows):
        col_idx = 1 + fit_idx
        params = [(np.asarray(mu, float), np.asarray(D, float), np.asarray(G, float))
                  for mu, D, G in r["best_params"]]
        weights = np.asarray(r["best_weights"], float)
        col_title = (f"kmeans\nβ={r['beta']:g}" if r["init_strategy"] == "kmeans"
                     else f"β = {r['beta']:g}")

        for row_idx in range(S):
            ax = axes[row_idx, col_idx]
            s = row_idx
            color = SAMPLE_COLORS[s % len(SAMPLE_COLORS)]
            _draw_hist(ax, obs_per_sample[s], color, SAMPLE_ALPHAS[s % len(SAMPLE_ALPHAS)])
            if s < weights.shape[0]:
                density, comp_dens = _marginal_density(params, weights[s], x_2d, K)
                ax.plot(x_grid, density, color=color, lw=2.0)
                for c in range(K):
                    if comp_dens[c].max() > density.max() * 0.01:
                        ax.plot(x_grid, comp_dens[c], color=comp_colors[c],
                                lw=0.9, ls="--", alpha=0.7)
            ax.grid(alpha=0.2, lw=0.4)
            ax.tick_params(labelsize=7)
            if row_idx == n_rows - 1:
                ax.set_xlabel(predictor_name, fontsize=8)
            if row_idx == 0:
                ax.set_title(col_title, fontsize=8, fontweight="bold")

    samples_str = ("  |  " + ", ".join(f"{n}={c}" for n, c in zip(sample_names, sample_counts))
                   if sample_counts else "")
    fig.suptitle(f"{gene}  K={K}  —  {predictor_name}{samples_str}", fontsize=10, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, out_path, dpi)


# ──────────────────────────────────────
# 2-D pair figures (one per predictor pair)
# ──────────────────────────────────────

def _plot_one_pair(dim_i, dim_j, pred_i, pred_j, rows, scores, sa,
                   sample_names, sample_counts, gene, K, p,
                   xi_grid, xj_grid, dpi, out_path, n_contour=6):
    S = len(sample_names)
    n_cols = 1 + len(rows)
    n_rows = S
    complete = ~(np.isnan(scores[:, dim_i]) | np.isnan(scores[:, dim_j]))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.2 * n_rows),
        sharex=True, sharey=True,
        squeeze=False,
    )

    def _scatter_bg(ax, focal_s):
        for s2 in range(S):
            if s2 == focal_s:
                continue
            mask2 = sa[:, s2].astype(bool) & complete
            if mask2.any():
                ax.scatter(scores[mask2, dim_i], scores[mask2, dim_j],
                           c="#D0D0D0", s=2, alpha=0.25, edgecolors="none", zorder=1)

    def _scatter_focal(ax, s):
        color = SAMPLE_COLORS[s % len(SAMPLE_COLORS)]
        mask = sa[:, s].astype(bool) & complete
        if mask.any():
            ax.scatter(scores[mask, dim_i], scores[mask, dim_j],
                       c=color, s=5, alpha=SAMPLE_ALPHAS[s % len(SAMPLE_ALPHAS)] * 0.8,
                       edgecolors="none", zorder=2)
        return mask.sum()

    # Column 0: raw data only
    for row_idx in range(S):
        ax = axes[row_idx, 0]
        s = row_idx
        color = SAMPLE_COLORS[s % len(SAMPLE_COLORS)]
        _scatter_bg(ax, s)
        n_s = _scatter_focal(ax, s)
        ax.grid(alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=7)
        ax.set_ylabel(f"{sample_names[s]}\n(n={n_s})\n{pred_j}",
                      fontsize=7, fontweight="bold", color=color)
        if row_idx == 0:
            ax.set_title("data only", fontsize=8, fontweight="bold")
        if row_idx == n_rows - 1:
            ax.set_xlabel(pred_i, fontsize=8)

    # Fit columns
    for fit_idx, r in enumerate(rows):
        col_idx = 1 + fit_idx
        params = [(np.asarray(mu, float), np.asarray(D, float), np.asarray(G, float))
                  for mu, D, G in r["best_params"]]
        weights = np.asarray(r["best_weights"], float)
        col_title = (f"kmeans\nβ={r['beta']:g}" if r["init_strategy"] == "kmeans"
                     else f"β = {r['beta']:g}")

        for row_idx in range(S):
            ax = axes[row_idx, col_idx]
            s = row_idx
            color = SAMPLE_COLORS[s % len(SAMPLE_COLORS)]
            cmap = SAMPLE_CMAPS[s % len(SAMPLE_CMAPS)]

            _scatter_bg(ax, s)
            _scatter_focal(ax, s)

            if s < weights.shape[0]:
                density = _joint_density_grid(
                    params, weights[s], xi_grid, xj_grid, dim_i, dim_j, p, K)
                peak = density.max()
                if peak > 0:
                    levels = np.linspace(peak * 0.02, peak * 0.90, n_contour)
                    if levels[-1] > levels[0]:
                        ax.contourf(xi_grid, xj_grid, density.T, levels=levels,
                                    cmap=cmap, alpha=0.30, zorder=0)
                        ax.contour(xi_grid, xj_grid, density.T, levels=levels,
                                   colors=[color], linewidths=0.8, alpha=0.8, zorder=3)

            ax.grid(alpha=0.2, lw=0.4)
            ax.tick_params(labelsize=7)
            if row_idx == n_rows - 1:
                ax.set_xlabel(pred_i, fontsize=8)
            if row_idx == 0:
                ax.set_title(col_title, fontsize=8, fontweight="bold")

    samples_str = ("  |  " + ", ".join(f"{n}={c}" for n, c in zip(sample_names, sample_counts))
                   if sample_counts else "")
    fig.suptitle(f"{gene}  K={K}  —  {pred_i} vs {pred_j}{samples_str}",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, out_path, dpi)


# ──────────────────────────────────────
# Shared save helper
# ──────────────────────────────────────

def _save_or_show(fig, out_path, dpi):
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────
# Main
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot β-sweep fits: 1-D marginals and 2-D pairwise density contours."
    )
    parser.add_argument("pickle_path", type=str,
                        help="Path to pickle from sweep_beta_brca1.py.")
    parser.add_argument("--data-dir", type=str,
                        default="/data/ross/assay_calibration/predictor_scores/"
                                "single_gene_calibration_data")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to save all figures. Opens windows if omitted.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--n-grid", type=int, default=300,
                        help="Grid points for 1-D density curves.")
    parser.add_argument("--n-grid-2d", type=int, default=80,
                        help="Grid points per axis for 2-D contour plots.")
    args = parser.parse_args()

    with open(args.pickle_path, "rb") as f:
        data = pickle.load(f)

    gene          = data["gene"]
    K             = data["components"]
    predictors    = data["predictors"]
    p             = len(predictors)
    sample_names  = data["sample_names"]
    sample_counts = data.get("sample_counts", [])
    results       = data["results"]

    print(f"Loading {gene} data from {args.data_dir} ...")
    ms     = load_predictor_ms(gene, args.data_dir)
    scores = ms.scores           # (N, p)
    sa     = ms.sample_assignments  # (N, S) bool

    anchored = sorted(
        [r for r in results if r["init_strategy"] == "anchored"],
        key=lambda r: r["beta"],
    )
    kmeans_results = [r for r in results if r["init_strategy"] == "kmeans"]
    rows = anchored + kmeans_results
    n_total = len(rows)
    rows = [r for r in rows if not r.get("failed")]
    n_skipped = n_total - len(rows)
    if n_skipped:
        print(f"  Skipping {n_skipped} failed config(s) from plot")
    if not rows:
        raise SystemExit("No results found in pickle.")

    def _safe(name):
        return name.replace("/", "_").replace(" ", "_")

    pad_frac = 0.03

    # ── 1-D marginal figures ──────────────────────────────────────
    x_grids = []
    for dim, predictor_name in enumerate(predictors):
        col = scores[:, dim]
        rng = float(np.nanmax(col) - np.nanmin(col))
        pad = pad_frac * rng
        x_grid = np.linspace(float(np.nanmin(col)) - pad,
                              float(np.nanmax(col)) + pad, args.n_grid)
        x_grids.append(x_grid)

        out_path = (Path(args.out_dir) / f"{gene}_{_safe(predictor_name)}_sweep.png"
                    if args.out_dir else None)
        _plot_one_predictor(
            dim, predictor_name, rows, scores, sa,
            sample_names, sample_counts, gene, K, p,
            x_grid, args.n_grid, args.dpi, out_path,
        )

    # ── 2-D pairwise figures ──────────────────────────────────────
    for dim_i, dim_j in combinations(range(p), 2):
        pred_i, pred_j = predictors[dim_i], predictors[dim_j]
        col_i, col_j = scores[:, dim_i], scores[:, dim_j]

        def _grid(col):
            rng = float(np.nanmax(col) - np.nanmin(col))
            return np.linspace(float(np.nanmin(col)) - pad_frac * rng,
                               float(np.nanmax(col)) + pad_frac * rng,
                               args.n_grid_2d)

        xi_grid = _grid(col_i)
        xj_grid = _grid(col_j)

        out_path = (Path(args.out_dir) / f"{gene}_{_safe(pred_i)}_vs_{_safe(pred_j)}_sweep.png"
                    if args.out_dir else None)
        print(f"Rendering pair: {pred_i} vs {pred_j} ...")
        _plot_one_pair(
            dim_i, dim_j, pred_i, pred_j, rows, scores, sa,
            sample_names, sample_counts, gene, K, p,
            xi_grid, xj_grid, args.dpi, out_path,
        )


if __name__ == "__main__":
    main()
