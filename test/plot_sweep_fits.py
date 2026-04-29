"""
Side-by-side comparison of CFUSN fits across β values.

Rows = β values (anchored sweep, plus kmeans baseline if present).
Columns = predictor dimensions.

Each panel shows raw per-sample score histograms with the fitted
mixture density overlaid — one curve per sample, coloured by sample
type (P/LP red, B/LB blue, gnomAD grey).

Usage
-----
    python test/plot_sweep_fits.py /tmp/sweep_brca1.pkl \\
        --data-dir /path/to/predictor_scores/single_gene_calibration_data

    python test/plot_sweep_fits.py /tmp/sweep_brca1.pkl \\
        --data-dir ... --out /tmp/sweep_fits_brca1.png --show-components
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

from visualize_fit import _sn_logpdf, SAMPLE_COLORS
from predictor_mv_utils import load_predictor_ms


def _marginal_density(params, w_s, x_grid_2d, K):
    """Per-sample marginal mixture density on a pre-built (n_grid, p) grid.

    NaN-filled columns are ignored by _sn_logpdf (correct marginal).

    Returns
    -------
    density      : (n,) total mixture density
    comp_densities : list of K (n,) per-component densities (sample-weighted)
    """
    comp_logpdfs = [_sn_logpdf(x_grid_2d, *params[c]) for c in range(K)]
    log_terms = [np.log(w_s[c] + 1e-300) + comp_logpdfs[c] for c in range(K)]
    density = np.exp(logsumexp(log_terms, axis=0))
    comp_densities = [np.exp(lt) for lt in log_terms]
    return density, comp_densities


def main():
    parser = argparse.ArgumentParser(
        description="Plot β-sweep fits as histogram + density overlays."
    )
    parser.add_argument("pickle_path", type=str,
                        help="Path to pickle from sweep_beta_brca1.py.")
    parser.add_argument("--data-dir", type=str,
                        default="/data/ross/assay_calibration/predictor_scores/"
                                "single_gene_calibration_data",
                        help="Directory with {gene}/{gene}_{predictor}.csv.gz.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (PNG/PDF). Opens window if omitted.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--n-grid", type=int, default=300,
                        help="Density evaluation grid points per dimension.")
    parser.add_argument("--show-components", action="store_true",
                        help="Overlay dashed per-component sub-densities.")
    args = parser.parse_args()

    with open(args.pickle_path, "rb") as f:
        data = pickle.load(f)

    gene = data["gene"]
    K = data["components"]
    predictors = data["predictors"]
    p = len(predictors)
    sample_names = data["sample_names"]
    sample_counts = data.get("sample_counts", [])
    S = len(sample_names)
    results = data["results"]

    print(f"Loading {gene} data from {args.data_dir} ...")
    ms = load_predictor_ms(gene, args.data_dir)
    scores = ms.scores              # (N, p)
    sa = ms.sample_assignments      # (N, S) bool

    anchored = sorted(
        [r for r in results if r["init_strategy"] == "anchored"],
        key=lambda r: r["beta"],
    )
    kmeans_results = [r for r in results if r["init_strategy"] == "kmeans"]
    rows = anchored + kmeans_results
    if not rows:
        raise SystemExit("No results found in pickle.")

    n_rows = len(rows)
    n_cols = p

    pad_frac = 0.03
    x_grids = []
    for d in range(p):
        col = scores[:, d]
        rng = float(np.nanmax(col) - np.nanmin(col))
        pad = pad_frac * rng
        x_grids.append(np.linspace(float(np.nanmin(col)) - pad,
                                   float(np.nanmax(col)) + pad,
                                   args.n_grid))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    for row_idx, r in enumerate(rows):
        beta = r["beta"]
        init = r["init_strategy"]
        params = [(np.asarray(mu, float), np.asarray(D, float), np.asarray(G, float))
                  for mu, D, G in r["best_params"]]
        weights = np.asarray(r["best_weights"], float)  # (S, K)

        row_label = (
            f"kmeans  β={beta:g}" if init == "kmeans" else f"β = {beta:g}"
        )

        for col_idx in range(p):
            ax = axes[row_idx, col_idx]
            x_1d = x_grids[col_idx]
            x_2d = np.full((len(x_1d), p), np.nan)
            x_2d[:, col_idx] = x_1d

            for s in range(S):
                color = SAMPLE_COLORS[s % len(SAMPLE_COLORS)]

                obs = scores[sa[:, s].astype(bool), col_idx]
                obs = obs[~np.isnan(obs)]
                if len(obs):
                    ax.hist(obs, bins=40, density=True, alpha=0.18,
                            color=color, edgecolor="none")

                if s < weights.shape[0]:
                    density, comp_dens = _marginal_density(
                        params, weights[s], x_2d, K
                    )
                    label = sample_names[s] if (col_idx == 0 and row_idx == 0) else None
                    ax.plot(x_1d, density, color=color, lw=1.8, label=label)

                    if args.show_components:
                        comp_colors = plt.cm.Set2(np.linspace(0, 1, max(K, 3)))
                        for c in range(K):
                            peak = comp_dens[c].max()
                            if peak > density.max() * 0.01:
                                ax.plot(x_1d, comp_dens[c],
                                        color=comp_colors[c], lw=0.7,
                                        ls="--", alpha=0.65)

            ax.set_xlabel(predictors[col_idx], fontsize=8)
            ax.grid(alpha=0.2, lw=0.4)
            ax.tick_params(labelsize=7)

            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\n\nDensity", fontsize=8,
                              fontweight="bold")
            else:
                ax.set_ylabel("Density", fontsize=8)

            if row_idx == 0:
                ax.set_title(predictors[col_idx], fontsize=9, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, fontsize=7, loc="upper left",
                          framealpha=0.7)

    samples_str = (
        "  |  " + ", ".join(f"{n}={c}" for n, c in zip(sample_names, sample_counts))
        if sample_counts else ""
    )
    fig.suptitle(
        f"{gene}  K={K}  CFUSN β sweep — fitted mixture densities{samples_str}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
