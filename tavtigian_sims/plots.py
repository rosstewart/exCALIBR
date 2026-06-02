"""
Plotting utilities for Tavtigian simulation results.

Quick-start
-----------
    import sys; sys.path.insert(0, "../src")
    from tavtigian_sims import SimulationSuite, prior_grid
    from tavtigian_sims.plots import *

    suite = SimulationSuite(priors=prior_grid("dense"), n_jobs=-1)
    suite.run()

    plot_C_star_profile(suite)            # C* U-shape vs prior
    plot_lr_thresholds(suite)             # LR+ thresholds for each tier
    plot_classification_posteriors(suite) # ground-truth check: do point
                                          # thresholds give correct posteriors?
    plot_feasibility_space(suite)         # Tavtigian 2018 Figure 1
    plot_combining_rule_heatmap(suite)    # which rules pass at each prior
    plot_lr_sensitivity(suite)            # U-shape of threshold stringency
    plot_dashboard(suite)                 # all-in-one 3×3
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .core import (
    POINT_VALUES, ACMG_TIER_CODES,
    CLASSIFICATION_BOUNDARIES, POSTERIOR_TARGETS,
)
from .suite import SimulationSuite
from .analysis import (
    boundary_validity, valid_prior_range,
    combining_rule_consistency, lr_threshold_sensitivity,
    feasibility_space, minimum_C_curve,
    find_C_star_minimum,
)

# ── shared style ───────────────────────────────────────────────────────────────

_TIER_COLOR = {8: "#d62728", 4: "#ff7f0e", 2: "#2ca02c", 1: "#1f77b4",
               3: "#9467bd", 5: "#8c564b", 6: "#e377c2", 7: "#7f7f7f"}
_TIER_LABEL = {
    8: "Very Strong (±8 pts)", 4: "Strong (±4 pts)",
    2: "Moderate (±2 pts)",    1: "Supporting (±1 pt)",
}

# Classification boundary style
_BND_STYLE = {
    "P_min":  ("≥10 pts  (P lower bound)",      "#d62728", "--", 0.99),
    "LP_min": ("≥6 pts   (LP lower bound)",     "#ff7f0e", "--", 0.90),
    "LP_max": ("≤9 pts   (LP/P boundary)",      "#aa0000", ":",  0.99),
    "LB_max": ("≤-1 pt   (LB upper bound)",     "#1f77b4", "--", 0.10),
    "LB_min": ("≥-6 pts  (LB lower bound)",     "#0055aa", ":",  0.01),
    "B_max":  ("≤-7 pts  (B upper bound)",      "#003388", "--", 0.01),
}

# Posterior target reference lines
_POST_REFS = [(0.99, "P threshold",  "#d62728", "--"),
              (0.90, "LP threshold", "#ff7f0e", "--"),
              (0.10, "LB threshold", "#1f77b4", "--"),
              (0.01, "B threshold",  "#003388", ":")]


def _df(s) -> pd.DataFrame:
    return s.to_dataframe() if isinstance(s, SimulationSuite) else s


def _vline(ax, x, color="black", ls="--", lw=1.1, alpha=0.5, label=None):
    ax.axvline(x, color=color, ls=ls, lw=lw, alpha=alpha, label=label)


def _apply_xscale(ax, log_scale: bool):
    if log_scale:
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, which="both")
    else:
        ax.set_xscale("linear")
        ax.grid(True, alpha=0.3)


# ── 1. C* profile ─────────────────────────────────────────────────────────────

def plot_C_star_profile(
    suite_or_df,
    figsize: Tuple = (9, 4.5),
    log_scale: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    C* (Tavtigian constant / O_PVSt) and total fail count vs prior.

    The theory predicts a U-shape with minimum at prior ≈ 0.25 (C_min = 81).
    With C_max = 350, the minimum hits the cap at ~0.10–0.32 (the valid range
    reported in Tavtigian 2018).  Green dots mark zero-fail priors.

    Interpretation
    --------------
    High C* → large LR+ thresholds → evidence must be very strong to count.
    The prior that minimises C* is the "most natural" prior for this framework.
    """
    df = _df(suite_or_df)
    info = find_C_star_minimum(df)
    opt = info["prior_at_min_C"]

    fig, (ax_c, ax_f) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax_c.plot(df["prior"], df["C_star"], color="steelblue", lw=2)
    zero = df[df["total_fails"] == 0]
    if len(zero):
        ax_c.scatter(zero["prior"], zero["C_star"],
                     color="limegreen", s=20, zorder=3, label="zero fails")
    _vline(ax_c, opt, label=f"C* minimum (prior={opt:.3f})")
    ax_c.axhline(81,  color="grey", ls=":", lw=1, alpha=0.7, label="C=81 (theoretical minimum)")
    ax_c.axhline(350, color="grey", ls="--", lw=1, alpha=0.5, label="C=350 (paper example)")
    ax_c.set_ylabel("C* (O_PVSt)", fontsize=11)
    ax_c.set_yscale("log")
    ax_c.legend(fontsize=8, loc="upper right")
    ax_c.set_title(
        "C* profile vs prior probability\n"
        "Minimum C* ≈ 81 at prior ≈ 0.25 (Tavtigian 2018 Fig 1)",
        fontsize=11,
    )

    colors = {"path_fails": "#d62728", "lp_fails": "#ff7f0e",
              "benign_fails": "#4daf4a", "lb_fails": "#1f77b4"}
    bottom = np.zeros(len(df))
    for col, color in colors.items():
        h = df[col].values
        ax_f.bar(df["prior"].values, h, bottom=bottom, color=color,
                 width=np.diff(np.append(df["prior"].values, 1))*0.8,
                 label=col.replace("_", " "), alpha=0.85)
        bottom += h
    _vline(ax_f, opt, label=None)
    ax_f.set_ylabel("Fail count", fontsize=11)
    ax_f.legend(fontsize=8, loc="upper right")

    for ax in (ax_c, ax_f):
        _apply_xscale(ax, log_scale)
    ax_f.set_xlabel("Prior probability of pathogenicity", fontsize=11)
    fig.tight_layout()
    return fig, np.array([ax_c, ax_f])


# ── 2. LR+ thresholds vs prior ────────────────────────────────────────────────

def plot_lr_thresholds(
    suite_or_df,
    codes: List[int] = ACMG_TIER_CODES,
    figsize: Tuple = (10, 5),
    log_scale: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    log10(LR+) thresholds for each ACMG evidence tier vs prior.

    Left: pathogenic tiers (+1,+2,+4,+8 pts).
    Right: benign tiers (−1,−2,−4,−8 pts).

    The thresholds move in parallel because they all scale with C* (= LR+_p8).
    The U-shape of C* directly translates to a U-shape in all thresholds.

    Clinical reading: at low priors, evidence tiers demand higher LR+ ratios;
    at high priors, benign tiers demand lower (more extreme) LR+ ratios.
    """
    df = _df(suite_or_df)
    opt = find_C_star_minimum(df)["prior_at_min_C"]

    fig, (ax_p, ax_b) = plt.subplots(1, 2, figsize=figsize)

    for k in codes:
        col = _TIER_COLOR[k]
        lbl = _TIER_LABEL.get(k, f"code {k}")
        ax_p.plot(df["prior"], df[f"log10_lr_p{k}"], color=col, lw=2, label=lbl)
        ax_b.plot(df["prior"], df[f"log10_lr_b{k}"], color=col, lw=2, label=lbl)

    for ax in (ax_p, ax_b):
        _vline(ax, opt, label=f"C* min (prior={opt:.3f})")
        _apply_xscale(ax, log_scale)
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set_xlabel("Prior probability of pathogenicity", fontsize=11)

    ax_p.set_ylabel("log₁₀(LR+)", fontsize=11)
    ax_b.set_ylabel("log₁₀(LR+)", fontsize=11)
    ax_p.set_title("Pathogenic evidence thresholds\n(higher = more stringent)", fontsize=11)
    ax_b.set_title("Benign evidence thresholds\n(lower = more stringent)", fontsize=11)
    ax_p.legend(fontsize=9)
    ax_b.legend(fontsize=9)
    fig.suptitle("LR+ thresholds inherit the U-shape of C*", fontsize=12)
    fig.tight_layout()
    return fig, np.array([ax_p, ax_b])


# ── 3. Classification boundary posteriors (ground-truth check) ────────────────

def plot_classification_posteriors(
    suite_or_df,
    figsize: Tuple = (12, 5),
    log_scale: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Ground-truth check: do the fixed Tavtigian-2020 point thresholds deliver
    the promised posterior probabilities across the full prior range?

    Each line shows the posterior achieved by a variant exactly at a
    classification boundary (e.g., 10 total points for P, 6 for LP).
    The shaded bands mark the correct posterior range.

    Failure = line exits its shaded band.

    Left panel:  pathogenic boundaries (P_min, LP_min, LP_max).
    Right panel: benign boundaries (LB_max, LB_min, B_max).
    """
    df = _df(suite_or_df)
    bv  = boundary_validity(df)
    opt = find_C_star_minimum(df)["prior_at_min_C"]

    fig, (ax_p, ax_b) = plt.subplots(1, 2, figsize=figsize)

    path_bnds = ["P_min", "LP_min", "LP_max"]
    ben_bnds  = ["LB_max", "LB_min", "B_max"]

    for ax, bnds in [(ax_p, path_bnds), (ax_b, ben_bnds)]:
        for name in bnds:
            label, color, ls, target = _BND_STYLE[name]
            sub = bv[bv["boundary"] == name].sort_values("prior")
            ax.plot(sub["prior"], sub["posterior"], color=color, ls=ls,
                    lw=2, label=label)
            ax.axhline(target, color=color, lw=0.7, alpha=0.4)

        _vline(ax, opt)
        _apply_xscale(ax, log_scale)
        ax.set_xlabel("Prior", fontsize=11)

    # shade target zones
    ax_p.axhspan(0.99, 1.01, color="#ffcccc", alpha=0.25, label="P zone (≥0.99)")
    ax_p.axhspan(0.90, 0.99, color="#ffe8cc", alpha=0.25, label="LP zone (0.90–0.99)")
    ax_b.axhspan(-0.01, 0.10, color="#cce0ff", alpha=0.25, label="LB zone (<0.10)")
    ax_b.axhspan(-0.01, 0.01, color="#99bbff", alpha=0.30, label="B zone (<0.01)")

    ax_p.set_ylim(0.80, 1.005)
    ax_b.set_ylim(-0.005, 0.15)
    ax_p.set_ylabel("Posterior P(pathogenic)", fontsize=11)
    ax_b.set_ylabel("Posterior P(pathogenic)", fontsize=11)
    ax_p.set_title("Pathogenic boundaries\n(shaded = target posterior zone)", fontsize=11)
    ax_b.set_title("Benign boundaries\n(shaded = target posterior zone)", fontsize=11)
    ax_p.legend(fontsize=8, loc="lower left")
    ax_b.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Classification boundary posteriors vs prior\n"
        "Lines exiting the shaded band = framework failure",
        fontsize=12,
    )
    fig.tight_layout()
    return fig, np.array([ax_p, ax_b])


# ── 4. Feasibility space (Tavtigian 2018 Figure 1) ────────────────────────────

def plot_feasibility_space(
    priors: Optional[np.ndarray] = None,
    C_range: Optional[np.ndarray] = None,
    figsize: Tuple = (8, 6),
    original: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Reproduce Tavtigian 2018 Figure 1: the (prior, O_PVSt) feasibility diagram.

    Blue region  : LP rules satisfied (all LP posteriors ≥ 0.90).
    Red region   : LB rule satisfied  (LB posteriors ≤ 0.10).
    Purple region: BOTH satisfied simultaneously.

    The two boundary curves intersect at (prior=0.25, C=81) — the minimum
    feasible C.  The paper's working point (prior=0.10, C=350) is also marked.
    """
    curves = minimum_C_curve(priors, original=original)

    if priors is None:
        priors = np.linspace(0.01, 0.45, 200)

    fig, ax = plt.subplots(figsize=figsize)

    lp_curve = curves.dropna(subset=["min_C_LP"])
    lb_curve = curves.dropna(subset=["min_C_LB"])

    ax.plot(lp_curve["prior"], lp_curve["min_C_LP"],
            color="steelblue", lw=2.5, label="LP boundary (min C satisfying LP≥0.90)")
    ax.plot(lb_curve["prior"], lb_curve["min_C_LB"],
            color="firebrick", lw=2.5, label="LB boundary (min C satisfying LB≤0.10)")

    # Fill feasibility region
    # Between the two curves (LP needs high C at low prior; LB needs high C at high prior)
    ax.fill_between(lp_curve["prior"],
                    lp_curve["min_C_LP"].fillna(0),
                    2000, alpha=0.10, color="steelblue", label="LP satisfied region")
    ax.fill_betweenx([0, 2000], 0, lb_curve.set_index("prior")["min_C_LB"].reindex(
        lp_curve["prior"], method="nearest").fillna(0).index.min(),
        alpha=0.08, color="firebrick")

    # Mark key points
    ax.scatter([0.25], [81], s=120, zorder=5, color="black",
               marker="^", label="Minimum solution (prior=0.25, C=81)")
    ax.scatter([0.10], [350], s=120, zorder=5, color="black",
               marker="o", label="Paper example (prior=0.10, C=350)")

    ax.set_xlabel("Prior probability of pathogenicity", fontsize=12)
    ax.set_ylabel("O_PVSt  (C*)", fontsize=12)
    ax.set_ylim(0, 1000)
    ax.set_xlim(0.0, 0.46)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "Tavtigian 2018 Figure 1 — Feasibility space\n"
        "LP and LB satisfied between the two curves",
        fontsize=12,
    )
    fig.tight_layout()
    return fig, ax


# ── 5. Combining-rule consistency heatmap ─────────────────────────────────────

def plot_combining_rule_heatmap(
    suite: SimulationSuite,
    figsize: Tuple = (13, 6),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Heatmap showing the posterior of each ACMG combining rule across priors.
    Target zones are shaded; rules that persistently fall outside their zone
    (Path(ii) and LP(i)) are highlighted.

    Columns = priors, rows = combining rules, colour = posterior value.
    """
    df = combining_rule_consistency(suite)
    cats = ["P", "LP", "LB", "B"]
    targets = {"P": (0.99, 1.01), "LP": (0.90, 0.99),
               "LB": (0.01, 0.10), "B": (-0.01, 0.01)}

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    priors_sorted = sorted(suite.to_dataframe()["prior"].unique())

    for ax, cat in zip(axes, cats):
        sub = df[df["category"] == cat]
        n_rules = sub["rule_idx"].nunique()
        matrix = np.full((n_rules, len(priors_sorted)), np.nan)
        for _, row in sub.iterrows():
            pidx = priors_sorted.index(
                min(priors_sorted, key=lambda p: abs(p - row["prior"]))
            )
            matrix[int(row["rule_idx"]), pidx] = row["posterior"]

        lo, hi = targets[cat]
        if cat in ("P", "LP"):
            cmap, vmin, vmax = "RdYlGn", lo - 0.05, hi + 0.01
        else:
            cmap, vmin, vmax = "RdYlGn_r", lo, hi + 0.05

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        # Mark inconsistent rules using the flag from combining_rule_consistency
        incon_idxs = set(
            sub.loc[sub["known_inconsistency"], "rule_idx"].astype(int).tolist()
        )
        for rule_idx in incon_idxs:
            ax.add_patch(plt.Rectangle((-0.5, rule_idx - 0.5),
                         len(priors_sorted), 1,
                         fill=False, edgecolor="yellow", lw=2))

        ax.set_title(f"{cat} rules\n(target {lo:.3f}–{hi:.3f})", fontsize=9)
        ax.set_yticks(range(n_rules))
        ax.set_yticklabels([f"rule {i}" for i in range(n_rules)], fontsize=7)
        step = max(len(priors_sorted) // 6, 1)
        ax.set_xticks(range(0, len(priors_sorted), step))
        ax.set_xticklabels(
            [f"{priors_sorted[i]:.2f}" for i in range(0, len(priors_sorted), step)],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_xlabel("Prior", fontsize=8)

    fig.suptitle(
        "Combining-rule posteriors vs prior\n"
        "Yellow border = known inconsistent rule (Tavtigian 2018)",
        fontsize=12,
    )
    fig.tight_layout()
    return fig, axes


# ── 6. LR+ threshold sensitivity (U-shape) ────────────────────────────────────

def plot_lr_sensitivity(
    suite_or_df,
    ref_prior: float = 0.10,
    codes: List[int] = ACMG_TIER_CODES,
    figsize: Tuple = (10, 5),
    log_scale: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Δlog10(LR+) relative to a reference prior — directly shows the U-shape
    hypothesis.

    Positive values: threshold is higher (more stringent) than at ref_prior.
    For pathogenic tiers:  U-shape opening upward → priors far from C*_min
                           demand stronger evidence.
    For benign tiers:      symmetric, opening downward (more negative = more
                           stringent in absolute terms).
    """
    df = _df(suite_or_df)
    sdf = lr_threshold_sensitivity(df, codes=codes, ref_prior=ref_prior)
    opt = find_C_star_minimum(df)["prior_at_min_C"]

    fig, (ax_p, ax_b) = plt.subplots(1, 2, figsize=figsize, sharey=False)

    for k in codes:
        col = _TIER_COLOR[k]
        lbl = _TIER_LABEL.get(k, f"code {k}")
        sub_p = sdf[(sdf["code"] == k) & (sdf["direction"] == "pathogenic")]
        sub_b = sdf[(sdf["code"] == k) & (sdf["direction"] == "benign")]
        ax_p.plot(sub_p["prior"], sub_p["delta_log10"], color=col, lw=2, label=lbl)
        ax_b.plot(sub_b["prior"], sub_b["delta_log10"], color=col, lw=2, label=lbl)

    for ax in (ax_p, ax_b):
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        _vline(ax, ref_prior, color="purple", label=f"ref prior={ref_prior}")
        _vline(ax, opt, label=f"C* min (prior={opt:.3f})")
        _apply_xscale(ax, log_scale)
        ax.set_xlabel("Prior", fontsize=11)

    ax_p.set_ylabel("Δlog₁₀(LR+) vs reference", fontsize=11)
    ax_b.set_ylabel("Δlog₁₀(LR+) vs reference", fontsize=11)
    ax_p.set_title("Pathogenic thresholds\n(+ve = harder than reference)", fontsize=11)
    ax_b.set_title("Benign thresholds\n(−ve = harder than reference)", fontsize=11)
    ax_p.legend(fontsize=9)
    ax_b.legend(fontsize=9)
    fig.suptitle(
        f"Threshold stringency relative to prior={ref_prior}  |  "
        f"U-shape minimum near prior={opt:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig, np.array([ax_p, ax_b])


# ── 7. All-in-one dashboard ───────────────────────────────────────────────────

def plot_dashboard(
    suite: SimulationSuite,
    figsize: Tuple = (18, 12),
    log_scale: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    3×3 overview combining all key views.

    [0,0] C* vs prior (U-shape)           [0,1] LR+ path thresholds
    [0,2] LR+ benign thresholds
    [1,0] P/LP boundary posteriors         [1,1] LB/B boundary posteriors
    [1,2] LR+ sensitivity (path)
    [2,0] Path combining-rule posteriors   [2,1] LP combining-rule posteriors
    [2,2] LB/B combining-rule posteriors
    """
    df = suite.to_dataframe()
    bv  = boundary_validity(df)
    cdf = combining_rule_consistency(suite)
    sdf = lr_threshold_sensitivity(df)
    opt = find_C_star_minimum(df)["prior_at_min_C"]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.38)

    ax_c    = fig.add_subplot(gs[0, 0])
    ax_lrp  = fig.add_subplot(gs[0, 1])
    ax_lrb  = fig.add_subplot(gs[0, 2])
    ax_pp   = fig.add_subplot(gs[1, 0])
    ax_bp   = fig.add_subplot(gs[1, 1])
    ax_sens = fig.add_subplot(gs[1, 2])
    ax_rP   = fig.add_subplot(gs[2, 0])
    ax_rLP  = fig.add_subplot(gs[2, 1])
    ax_rLB  = fig.add_subplot(gs[2, 2])

    # ── C* ──────────────────────────────────────────────────────────────────
    ax_c.plot(df["prior"], df["C_star"], color="steelblue", lw=2)
    ax_c.axhline(81, color="grey", ls=":", lw=1)
    ax_c.axhline(350, color="grey", ls="--", lw=0.8)
    _vline(ax_c, opt)
    ax_c.set_yscale("log")
    ax_c.set_title("C* vs prior (U-shape)", fontsize=10)
    ax_c.set_ylabel("C*", fontsize=9)

    # ── LR+ thresholds ──────────────────────────────────────────────────────
    for k in ACMG_TIER_CODES:
        col = _TIER_COLOR[k]
        ax_lrp.plot(df["prior"], df[f"log10_lr_p{k}"], color=col, lw=1.8,
                    label=_TIER_LABEL.get(k))
        ax_lrb.plot(df["prior"], df[f"log10_lr_b{k}"], color=col, lw=1.8)

    ax_lrp.set_title("Pathogenic LR+ thresholds", fontsize=10)
    ax_lrb.set_title("Benign LR+ thresholds", fontsize=10)
    ax_lrp.set_ylabel("log₁₀(LR+)", fontsize=9)
    ax_lrb.set_ylabel("log₁₀(LR+)", fontsize=9)
    ax_lrp.legend(fontsize=7)

    # ── classification boundary posteriors ──────────────────────────────────
    for name in ["P_min", "LP_min"]:
        lbl, col, ls, _ = _BND_STYLE[name]
        sub = bv[bv["boundary"] == name].sort_values("prior")
        ax_pp.plot(sub["prior"], sub["posterior"], color=col, ls=ls, lw=2, label=lbl)
    ax_pp.axhspan(0.99, 1.02, color="#ffcccc", alpha=0.25)
    ax_pp.axhspan(0.90, 0.99, color="#ffe8cc", alpha=0.25)
    ax_pp.set_ylim(0.82, 1.01)
    ax_pp.set_title("P/LP boundary posteriors", fontsize=10)
    ax_pp.set_ylabel("Posterior", fontsize=9)
    ax_pp.legend(fontsize=7)

    for name in ["LB_max", "B_max"]:
        lbl, col, ls, _ = _BND_STYLE[name]
        sub = bv[bv["boundary"] == name].sort_values("prior")
        ax_bp.plot(sub["prior"], sub["posterior"], color=col, ls=ls, lw=2, label=lbl)
    ax_bp.axhspan(-0.001, 0.10, color="#cce0ff", alpha=0.25)
    ax_bp.axhspan(-0.001, 0.01, color="#99bbff", alpha=0.35)
    ax_bp.set_ylim(-0.005, 0.14)
    ax_bp.set_title("LB/B boundary posteriors", fontsize=10)
    ax_bp.set_ylabel("Posterior", fontsize=9)
    ax_bp.legend(fontsize=7)

    # ── LR+ sensitivity ─────────────────────────────────────────────────────
    for k in ACMG_TIER_CODES:
        sub = sdf[(sdf["code"] == k) & (sdf["direction"] == "pathogenic")]
        ax_sens.plot(sub["prior"], sub["delta_log10"], color=_TIER_COLOR[k], lw=1.8)
    ax_sens.axhline(0, color="grey", lw=0.8, ls="--")
    ax_sens.set_title("Path. threshold stringency\n(Δlog₁₀ vs prior=0.10)", fontsize=10)
    ax_sens.set_ylabel("Δlog₁₀(LR+)", fontsize=9)

    # ── combining-rule posteriors ────────────────────────────────────────────
    rule_axes = {"P": ax_rP, "LP": ax_rLP}
    rule_axes_b = {"LB": ax_rLB}
    targets_p = {"P": (0.99, "#d62728"), "LP": (0.90, "#ff7f0e")}
    targets_b = {"LB": (0.10, "#1f77b4"), "B": (0.01, "#003388")}

    for cat, ax in rule_axes.items():
        sub = cdf[cdf["category"] == cat]
        target, tcol = targets_p[cat]
        for rule_idx, grp in sub.groupby("rule_idx"):
            is_incon = grp["known_inconsistency"].iloc[0]
            lw = 2.5 if is_incon else 1.2
            ls = "--" if is_incon else "-"
            lbl = f"rule {rule_idx} ⚠" if is_incon else f"rule {rule_idx}"
            ax.plot(grp["prior"], grp["posterior"], lw=lw, ls=ls, label=lbl)
        ax.axhline(target, color=tcol, ls=":", lw=1.2, alpha=0.7)
        ax.set_title(f"{cat} rule posteriors", fontsize=10)
        ax.set_ylabel("Posterior", fontsize=9)
        ax.legend(fontsize=6)

    sub_lb = cdf[cdf["category"].isin(["LB", "B"])]
    for rule_idx, grp in sub_lb.groupby(["category", "rule_idx"]):
        cat = rule_idx[0]
        ax_rLB.plot(grp["prior"], grp["posterior"],
                    label=f"{cat} rule {rule_idx[1]}", lw=1.5)
    ax_rLB.set_title("LB/B rule posteriors", fontsize=10)
    ax_rLB.set_ylabel("Posterior", fontsize=9)
    ax_rLB.legend(fontsize=6)

    # ── shared formatting ────────────────────────────────────────────────────
    for ax in [ax_c, ax_lrp, ax_lrb, ax_pp, ax_bp, ax_sens, ax_rP, ax_rLP, ax_rLB]:
        _apply_xscale(ax, log_scale)
        _vline(ax, opt, lw=0.9)
    for ax in [ax_rP, ax_rLP, ax_rLB, ax_sens]:
        ax.set_xlabel("Prior", fontsize=9)

    fig.suptitle(
        f"Tavtigian Framework Dashboard  |  C* minimum at prior ≈ {opt:.3f}",
        fontsize=14,
    )
    return fig, np.array([[ax_c, ax_lrp, ax_lrb],
                           [ax_pp, ax_bp, ax_sens],
                           [ax_rP, ax_rLP, ax_rLB]])
