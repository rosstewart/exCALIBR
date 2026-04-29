"""
Plot β-sensitivity sweep results from sweep_beta_brca1.py.

Reads the pickle produced by ``sweep_beta_brca1.py --save-pickle <path>``
and renders one figure with two rows:

  * Top row, K columns: per-component μ vs β, one line per predictor dim.
    Horizontal dashed lines = empirical mean of that component's anchor
    sample (the value the EM "should" land near). Diamond markers mark
    the kmeans baseline (β=0) for reference, if present.

  * Bottom row, full-width: best train LL vs β (anchored), with kmeans
    baseline as a horizontal reference line.

Usage
-----
    python test/plot_sweep_beta.py /tmp/sweep_brca1.pkl
    python test/plot_sweep_beta.py /tmp/sweep_brca1.pkl --out /tmp/sweep_brca1.png
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _anchor_sample_index(c, K, S):
    """Mirror default_anchor_groups in initializations.py.

    For K=3, S=4: comp 1 is anchored to {benign, synonymous}; here we
    return just sample idx 1 for the reference line (the dominant
    anchor sample). For everything else, comp c ← sample c.
    """
    if S == 4 and K <= 3 and c == 1:
        return 1
    return min(c, S - 1)


def main():
    parser = argparse.ArgumentParser(
        description="Plot β-sensitivity sweep results."
    )
    parser.add_argument("pickle_path", type=str,
                        help="Path to the pickle saved by sweep_beta_brca1.py.")
    parser.add_argument("--out", type=str, default=None,
                        help="Save figure (PNG/PDF/SVG). If omitted, opens an interactive window.")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    with open(args.pickle_path, "rb") as f:
        data = pickle.load(f)

    gene = data["gene"]
    K = data["components"]
    predictors = data["predictors"]
    p = len(predictors)
    sample_names = data["sample_names"]
    sample_counts = data["sample_counts"]
    sample_means = np.asarray(data.get("sample_means"))
    S = sample_means.shape[0] if sample_means is not None and sample_means.ndim == 2 else len(sample_names)
    results = data["results"]

    anchored = sorted(
        [r for r in results if r["init_strategy"] == "anchored"],
        key=lambda r: r["beta"],
    )
    kmeans = [r for r in results if r["init_strategy"] == "kmeans"]
    if not anchored:
        raise SystemExit("No anchored results found in pickle.")

    betas = np.array([r["beta"] for r in anchored])
    # means_arr[c, β_idx, dim]
    means_arr = np.array([
        [r["component_means_sorted"][c] for r in anchored]
        for c in range(K)
    ])
    lls_anch = np.array([r["best_ll"] for r in anchored])

    # Layout: top row = K subplots (one per component); bottom row = LL panel
    fig = plt.figure(figsize=(4 * K, 7))
    gs = fig.add_gridspec(2, K, height_ratios=[2, 1], hspace=0.35, wspace=0.3)

    colors = plt.cm.tab10.colors

    for c in range(K):
        ax = fig.add_subplot(gs[0, c])
        anchor_s = _anchor_sample_index(c, K, S)
        anchor_label = sample_names[anchor_s] if anchor_s < len(sample_names) else "—"

        # Anchored sweep: solid lines per dim
        for d in range(p):
            ax.plot(betas, means_arr[c, :, d], "o-",
                    color=colors[d % 10], label=predictors[d], lw=1.8, ms=5)

        # Empirical anchor-sample mean per dim: horizontal dashed reference
        if sample_means is not None and sample_means.ndim == 2 and anchor_s < S:
            for d in range(p):
                ax.axhline(
                    sample_means[anchor_s, d],
                    color=colors[d % 10], ls=":", lw=1.0, alpha=0.6,
                )

        # kmeans baseline (β=0): diamond markers
        for r in kmeans:
            for d in range(p):
                ax.plot([r["beta"]], [r["component_means_sorted"][c][d]],
                        marker="D", color=colors[d % 10], ms=7,
                        markerfacecolor="white", markeredgewidth=1.5)

        ax.set_title(f"Comp {c}  ←  {anchor_label}")
        ax.set_xlabel("β")
        if c == 0:
            ax.set_ylabel("component mean μ per dim")
            ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    # Bottom: LL vs β
    ax_ll = fig.add_subplot(gs[1, :])
    ax_ll.plot(betas, lls_anch, "o-", color="black", lw=1.8, ms=6, label="anchored")
    for r in kmeans:
        ax_ll.axhline(r["best_ll"], color="gray", ls="--", lw=1.0,
                      label=f"kmeans (β={r['beta']:g})")
        ax_ll.plot([r["beta"]], [r["best_ll"]], marker="D",
                   color="gray", ms=7, markerfacecolor="white", markeredgewidth=1.5)
    ax_ll.set_xlabel("β  (sample-balance strength)")
    ax_ll.set_ylabel("best train LL")
    ax_ll.set_title("Model fit quality across β")
    ax_ll.legend(fontsize=9)
    ax_ll.grid(alpha=0.3)

    samples_str = ", ".join(f"{n}={c}" for n, c in zip(sample_names, sample_counts))
    fig.suptitle(
        f"{gene}  K={K}  CFUSN β sweep    "
        f"|  samples: {samples_str}    "
        f"|  --:anchor empirical mean   ◇:kmeans baseline",
        fontsize=11,
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
