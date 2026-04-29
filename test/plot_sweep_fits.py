"""
Side-by-side comparison of CFUSN fits across β values — one file per predictor.

Produces one figure per predictor dimension (REVEL, MutPred2, AlphaMissense).

Layout per figure:
  Rows    = samples (P/LP, B/LB, gnomAD)  — one sample per row
  Columns = β configs                      — one β value per column
  X-axis shared across all panels so score ranges align.
  Y-axis shared within each row so density scale is comparable across β.

Each panel: histogram of that sample's scores (density=True) with the
fitted mixture density curve overlaid, plus dashed per-component sub-curves.

Usage
-----
    python test/plot_sweep_fits.py /tmp/sweep_brca1.pkl \\
        --data-dir /path/to/predictor_scores/single_gene_calibration_data

    # save to /tmp/sweep_fits_BRCA1_REVEL.png etc.
    python test/plot_sweep_fits.py /tmp/sweep_brca1.pkl \\
        --data-dir ... --out-dir /tmp
"""

import argparse
import pickle
import sys
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


def _marginal_density(params, w_s, x_grid_2d, K):
    """Per-sample marginal mixture density on an (n_grid, p) grid with NaN fill."""
    comp_logpdfs = [_sn_logpdf(x_grid_2d, *params[c]) for c in range(K)]
    log_terms = [np.log(w_s[c] + 1e-300) + comp_logpdfs[c] for c in range(K)]
    density = np.exp(logsumexp(log_terms, axis=0))
    comp_densities = [np.exp(lt) for lt in log_terms]
    return density, comp_densities


def _draw_hist(ax, obs, color, alpha):
    if len(obs):
        ax.hist(obs, bins=50, density=True, alpha=alpha,
                color=color, edgecolor="none")


def _plot_one_predictor(dim, predictor_name, rows, scores, sa, sample_names,
                        sample_counts, gene, K, p, x_grid, n_grid,
                        dpi, out_path):
    S = len(sample_names)
    # Column 0 = raw data only; columns 1..n_fits = β configs
    n_fit_cols = len(rows)
    n_cols = 1 + n_fit_cols
    n_rows = S

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        sharex=True, sharey="row",
        squeeze=False,
    )

    comp_colors = plt.cm.Set2(np.linspace(0, 1, max(K, 3)))

    # Pre-compute per-sample observations for this dim
    obs_per_sample = []
    for s in range(S):
        obs = scores[sa[:, s].astype(bool), dim]
        obs_per_sample.append(obs[~np.isnan(obs)])

    # Column 0: raw data only
    for row_idx in range(S):
        ax = axes[row_idx, 0]
        s = row_idx
        color = SAMPLE_COLORS[s % len(SAMPLE_COLORS)]
        alpha = SAMPLE_ALPHAS[s % len(SAMPLE_ALPHAS)]
        _draw_hist(ax, obs_per_sample[s], color, alpha)
        ax.grid(alpha=0.2, lw=0.4)
        ax.tick_params(labelsize=7)
        n_s = len(obs_per_sample[s])
        ax.set_ylabel(
            f"{sample_names[s]}\n(n={n_s})",
            fontsize=8, fontweight="bold", color=color,
        )
        if row_idx == 0:
            ax.set_title("data only", fontsize=8, fontweight="bold")
        if row_idx == n_rows - 1:
            ax.set_xlabel(predictor_name, fontsize=8)

    # Columns 1+: β configs
    x_2d = np.full((n_grid, p), np.nan)
    x_2d[:, dim] = x_grid

    for fit_idx, r in enumerate(rows):
        col_idx = 1 + fit_idx
        beta = r["beta"]
        init = r["init_strategy"]
        params = [(np.asarray(mu, float), np.asarray(D, float), np.asarray(G, float))
                  for mu, D, G in r["best_params"]]
        weights = np.asarray(r["best_weights"], float)  # (S, K)

        col_title = (
            f"kmeans\nβ={beta:g}" if init == "kmeans" else f"β = {beta:g}"
        )

        for row_idx in range(S):
            ax = axes[row_idx, col_idx]
            s = row_idx
            color = SAMPLE_COLORS[s % len(SAMPLE_COLORS)]
            alpha = SAMPLE_ALPHAS[s % len(SAMPLE_ALPHAS)]

            _draw_hist(ax, obs_per_sample[s], color, alpha)

            if s < weights.shape[0]:
                density, comp_dens = _marginal_density(params, weights[s], x_2d, K)
                ax.plot(x_grid, density, color=color, lw=2.0)
                for c in range(K):
                    if comp_dens[c].max() > density.max() * 0.01:
                        ax.plot(x_grid, comp_dens[c],
                                color=comp_colors[c], lw=0.9, ls="--", alpha=0.7)

            ax.grid(alpha=0.2, lw=0.4)
            ax.tick_params(labelsize=7)

            if row_idx == n_rows - 1:
                ax.set_xlabel(predictor_name, fontsize=8)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=8, fontweight="bold")

    samples_str = (
        "  |  " + ", ".join(f"{n}={c}" for n, c in zip(sample_names, sample_counts))
        if sample_counts else ""
    )
    fig.suptitle(
        f"{gene}  K={K}  —  {predictor_name}{samples_str}",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot β-sweep fits as histogram + density overlays (one file per predictor)."
    )
    parser.add_argument("pickle_path", type=str,
                        help="Path to pickle from sweep_beta_brca1.py.")
    parser.add_argument("--data-dir", type=str,
                        default="/data/ross/assay_calibration/predictor_scores/"
                                "single_gene_calibration_data")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to save figures. Files named "
                             "{gene}_{predictor}_sweep.png. Opens windows if omitted.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--n-grid", type=int, default=300)
    args = parser.parse_args()

    with open(args.pickle_path, "rb") as f:
        data = pickle.load(f)

    gene = data["gene"]
    K = data["components"]
    predictors = data["predictors"]
    p = len(predictors)
    sample_names = data["sample_names"]
    sample_counts = data.get("sample_counts", [])
    results = data["results"]

    print(f"Loading {gene} data from {args.data_dir} ...")
    ms = load_predictor_ms(gene, args.data_dir)
    scores = ms.scores          # (N, p)
    sa = ms.sample_assignments  # (N, S) bool

    anchored = sorted(
        [r for r in results if r["init_strategy"] == "anchored"],
        key=lambda r: r["beta"],
    )
    kmeans_results = [r for r in results if r["init_strategy"] == "kmeans"]
    rows = anchored + kmeans_results
    if not rows:
        raise SystemExit("No results found in pickle.")

    pad_frac = 0.03
    for dim, predictor_name in enumerate(predictors):
        col = scores[:, dim]
        rng = float(np.nanmax(col) - np.nanmin(col))
        pad = pad_frac * rng
        x_grid = np.linspace(float(np.nanmin(col)) - pad,
                              float(np.nanmax(col)) + pad,
                              args.n_grid)

        out_path = None
        if args.out_dir:
            safe_name = predictor_name.replace("/", "_").replace(" ", "_")
            out_path = Path(args.out_dir) / f"{gene}_{safe_name}_sweep.png"

        _plot_one_predictor(
            dim, predictor_name, rows, scores, sa,
            sample_names, sample_counts, gene, K, p,
            x_grid, args.n_grid, args.dpi, out_path,
        )


if __name__ == "__main__":
    main()
