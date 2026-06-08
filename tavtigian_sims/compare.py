"""
Side-by-side comparison plots: Tavtigian vs Piecewise vs Continuous.

Quick-start
-----------
    from tavtigian_sims import prior_grid
    from tavtigian_sims.compare import (
        run_all_methods,
        plot_three_way_comparison,
        plot_boundary_posteriors_three_way,
        plot_relative_stringency,
        plot_additivity_experiment,
        plot_combined_error_comparison,
    )
    priors = prior_grid("paper")
    t, pw, pw_add, cont = run_all_methods(priors, n_jobs=10)
    plot_three_way_comparison(t, pw, cont_df=cont)
    plot_boundary_posteriors_three_way(t, pw)
    plot_relative_stringency(t, pw, cont_df=cont)
    plot_additivity_experiment(priors, tav_df=t)
    plot_combined_error_comparison(priors, tav_df=t)
"""

from __future__ import annotations

from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from .suite import SimulationSuite
from .bayesian import PiecewiseSuite, PiecewiseAdditiveSuite, LPAnchoredSuite, ContinuousSuite
from .core import ACMG_TIER_CODES, CLASSIFICATION_BOUNDARIES, POSTERIOR_TARGETS
from .analysis import boundary_validity

# ── Colour palettes ───────────────────────────────────────────────────────────

# Per-code strength colours: pathogenic (warm reds), benign (cool blues)
# Extended to ±12 codes
STRENGTH_COLOR = {
    -12: '#2f5a70', -11: '#3a6b81', -10: '#4b91a6', -9: '#516d8a',
    -8: '#4b91a6', -7: '#5DA3BD', -6: '#6FAACE', -5: '#74ABCE',
    -4: '#7ab5d1', -3: '#99c8dc', -2: '#d0e8f0', -1: '#e4f1f6',
     0: '#e0e0e0',
     1: '#e6b1b8',  2: '#d68f99',  3: '#ca7682',  4: '#b85c6b',
     5: '#B1535F',  6: '#AA4E58',  7: '#A2484F',  8: '#943744',
     9: '#7d2e38', 10: '#6b2830', 11: '#59221f', 12: '#472015',
}

# Boundary colours (P, LP, LB, B) keyed to strength-colour anchors
_BND_COLOR = {
    "P":  STRENGTH_COLOR[8],
    "LP": STRENGTH_COLOR[5],
    "LB": STRENGTH_COLOR[-5],
    "B":  STRENGTH_COLOR[-8],
}

# Per-method line styles / labels
_METHOD_STYLE = {
    "tavtigian":          {"ls": "-",   "lw": 2.0, "alpha": 0.95, "label": "Tavtigian (C*)"},
    "piecewise":          {"ls": "--",  "lw": 2.0, "alpha": 0.85, "label": "Piecewise α (ACMG)"},
    "piecewise_additive": {"ls": "-.",  "lw": 2.0, "alpha": 0.85, "label": "Piecewise-Add (6·11·6)"},
    "lp_anchored":        {"ls": "-.",  "lw": 2.0, "alpha": 0.85, "label": "LP-Anchored (additive)"},
    "continuous":         {"ls": ":",   "lw": 2.5, "alpha": 0.80, "label": "Continuous (Bayes)"},
}

# Classification boundary T values for each method.
# Used to label plots and pull the right bnd_post_* columns.
_METHOD_BND_T = {
    "tavtigian":          {"P": 10, "LP":  6, "LB": -1, "B": -7},
    "piecewise":          {"P": 10, "LP":  6, "LB": -1, "B": -7},
    "piecewise_additive": {"P": 17, "LP": 11, "LB":  0, "B": -6},
    "lp_anchored":        {"P": 10, "LP":  6, "LB": -1, "B": -7},
}

_ALL_CODES = list(range(1, 13))   # 1..12


# ── Error metrics helper functions ────────────────────────────────────────────

def log_odds_error(posterior: np.ndarray, target: float) -> np.ndarray:
    """Signed log-odds error: log(post/(1-post)) - log(target/(1-target)).

    Measures error in the native space of Bayesian reasoning.
    Zero means exact. Positive means overestimate, negative means underestimate.
    """
    import math
    eps = 1e-10
    post_clip = np.clip(posterior, eps, 1.0 - eps)
    target_clip = np.clip(target, eps, 1.0 - eps)
    return np.log(post_clip / (1.0 - post_clip)) - math.log(target_clip / (1.0 - target_clip))


def abs_log_odds_error(posterior: np.ndarray, target: float) -> np.ndarray:
    """Absolute log-odds error."""
    return np.abs(log_odds_error(posterior, target))


def run_all_methods(priors: np.ndarray, n_jobs: int = -1) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all five suites; return (tav_df, pw_df, pw_add_df, lp_anch_df, cont_df)."""
    t_suite  = SimulationSuite(priors=priors, n_jobs=n_jobs).run()
    p_suite  = PiecewiseSuite(priors=priors).run()
    pa_suite = PiecewiseAdditiveSuite(priors=priors).run()
    lpa_suite = LPAnchoredSuite(priors=priors).run()
    c_suite  = ContinuousSuite(priors=priors).run()
    return (t_suite.to_dataframe(), p_suite.to_dataframe(),
            pa_suite.to_dataframe(), lpa_suite.to_dataframe(), c_suite.to_dataframe())


def _apply_xscale(ax, log_scale: bool):
    if log_scale:
        ax.set_xscale("log"); ax.grid(True, alpha=0.3, which="both")
    else:
        ax.set_xscale("linear"); ax.grid(True, alpha=0.3)


# ── 1. LR+ thresholds: combined pathogenic + benign on one plot ───────────────

def plot_three_way_comparison(
    tav_df:     Optional[pd.DataFrame] = None,
    pw_df:      Optional[pd.DataFrame] = None,
    pw_add_df:  Optional[pd.DataFrame] = None,
    lpa_df:     Optional[pd.DataFrame] = None,
    cont_df:    Optional[pd.DataFrame] = None,
    methods: List[str] = ("tavtigian", "piecewise", "lp_anchored", "continuous"),
    codes = "key",
    figsize  = (11, 6),
    log_scale: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """log₁₀(LR+) for evidence codes vs prior — pathogenic and benign on one plot.

    Parameters
    ----------
    methods : list of str
        Any subset of
        ``{"tavtigian", "piecewise", "piecewise_additive", "lp_anchored", "continuous"}``.
        Pass a single-element list to plot only one method.
        ``"continuous"`` is always drawn as thin black dotted reference lines
        regardless of whether it appears in ``methods`` — pass ``cont_df=None``
        to suppress it entirely.
    codes : ``"all"`` | ``"key"`` | list of int
        ``"all"``  → codes 1–12 (all extended codes)
        ``"key"``  → codes 1, 2, 4, 8 only (ACMG tiers; default)
        list       → explicit subset, e.g. ``[1, 2, 4, 8, 12]``
    """
    # Resolve codes
    if codes == "all":
        _codes = _ALL_CODES
    elif codes == "key":
        _codes = [1, 2, 4, 8]
    else:
        _codes = list(codes)

    _DF = {"tavtigian": tav_df, "piecewise": pw_df,
           "piecewise_additive": pw_add_df, "lp_anchored": lpa_df}

    fig, ax = plt.subplots(figsize=figsize)

    # Continuous ground-truth background: four thin black dotted lines
    if cont_df is not None:
        _cont_cols = {
            "P":  ("log10_lr_P",  0.99),
            "LP": ("log10_lr_LP", 0.90),
            "LB": ("log10_lr_LB", 0.10),
            "B":  ("log10_lr_B",  0.01),
        }
        for name, (col, target) in _cont_cols.items():
            if col in cont_df.columns:
                ax.plot(cont_df["prior"], cont_df[col],
                        color="black", lw=1.2, ls=":", alpha=0.35,
                        zorder=1,
                        label=f"Bayes {name} (post={target})" if name == "P" else None)

    # Per-method per-code lines
    for method in methods:
        if method == "continuous":
            continue   # drawn above as background
        df = _DF.get(method)
        if df is None:
            continue
        s = _METHOD_STYLE[method]
        for k in _codes:
            col_p = f"log10_lr_p{k}"
            col_b = f"log10_lr_b{k}"
            if col_p in df.columns:
                ax.plot(df["prior"], df[col_p],
                        color=STRENGTH_COLOR[k],
                        ls=s["ls"], lw=s["lw"], alpha=s["alpha"],
                        zorder=2)
            if col_b in df.columns:
                # log10_lr_b is already negative (LR+ < 1)
                ax.plot(df["prior"], df[col_b],
                        color=STRENGTH_COLOR[-k],
                        ls=s["ls"], lw=s["lw"], alpha=s["alpha"],
                        zorder=2)

    ax.axhline(0, color="grey", lw=0.6, ls=":")
    _apply_xscale(ax, log_scale)
    ax.set_xlabel("Prior probability of pathogenicity", fontsize=11)
    ax.set_ylabel("log₁₀(LR+)", fontsize=11)

    # Legend: one entry per method + colour gradient sentinels
    handles = []
    if cont_df is not None:
        handles.append(Line2D([0], [0], color="black", ls=":", lw=1.5,
                               alpha=0.5, label="Bayesian ground truth"))
    for method in methods:
        if method == "continuous":
            continue
        df = _DF.get(method)
        if df is not None:
            s = _METHOD_STYLE[method]
            handles.append(Line2D([0], [0], color="grey",
                                  ls=s["ls"], lw=s["lw"],
                                  label=s["label"]))
    handles.append(Line2D([0], [0], color=STRENGTH_COLOR[_codes[-1]], lw=3,
                          label=f"Code ±{_codes[-1]} (strongest)"))
    handles.append(Line2D([0], [0], color=STRENGTH_COLOR[_codes[0]], lw=3,
                          label=f"Code ±{_codes[0]} (weakest)"))

    ax.legend(handles=handles, fontsize=9, loc="best", framealpha=0.8)
    _methods_shown = [m for m in methods if m != "continuous"]
    _labels_shown  = [_METHOD_STYLE.get(m, {}).get("label", m) for m in _methods_shown]
    ax.set_title(
        f"LR+ thresholds vs prior  [{', '.join(_labels_shown)}]"
        f"  (pathogenic above 0, benign below 0)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, ax


# ── 2. Boundary posteriors: Tavtigian vs Piecewise ───────────────────────────

def plot_boundary_posteriors_three_way(
    tav_df:    Optional[pd.DataFrame] = None,
    pw_df:     Optional[pd.DataFrame] = None,
    pw_add_df: Optional[pd.DataFrame] = None,
    lpa_df:    Optional[pd.DataFrame] = None,  # [DEPRECATED] kept for compatibility
    cont_df:   Optional[pd.DataFrame] = None,   # accepted for backward compat, unused
    methods: List[str] = ("tavtigian", "piecewise"),
    figsize  = (12, 5),
    log_scale: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """Posterior at each method's own boundary T values vs prior.

    Each method is plotted at its OWN classification boundary T values:
      - Tavtigian and piecewise: T ∈ {10, 6, −1, −7}
      - Piecewise-Add (6·11·6): T ∈ {16, 10, −1, −7}

    Continuous is omitted (trivially flat by construction).
    Piecewise variants should be flat at the targets for every prior.
    Tavtigian drifts away from the targets at priors far from 0.10.
    """
    # Guard: if methods is accidentally a DataFrame (old call: (t, a, b))
    if not isinstance(methods, (list, tuple)) or (
            len(methods) > 0 and not isinstance(methods[0], str)):
        methods = ("tavtigian", "piecewise", "piecewise_additive")

    _DF = {"tavtigian": tav_df, "piecewise": pw_df,
           "piecewise_additive": pw_add_df, "lp_anchored": lpa_df}

    # Each method's boundary T values for the four named boundaries
    # (used in labels and to pull the right bnd_post_ column).
    _BND_NAMES = ("P_min", "LP_min", "LB_max", "B_max")
    _BND_TARGET = {"P_min": 0.99, "LP_min": 0.90, "LB_max": 0.10, "B_max": 0.01}
    _BND_COLOR_MAP = {
        "P_min":  _BND_COLOR["P"],  "LP_min": _BND_COLOR["LP"],
        "LB_max": _BND_COLOR["LB"], "B_max":  _BND_COLOR["B"],
    }

    fig, (ax_p, ax_b) = plt.subplots(1, 2, figsize=figsize)

    for method in methods:
        s   = _METHOD_STYLE.get(method, {"ls": "-", "lw": 1.5})
        lbl = s.get("label", method)
        df  = _DF.get(method)
        bnd_t = _METHOD_BND_T.get(method, {"P": 10, "LP": 6, "LB": -1, "B": -7})

        if df is None:
            continue

        if method == "tavtigian":
            bv = boundary_validity(df)
            for bname in _BND_NAMES:
                ax    = ax_p if bname in ("P_min", "LP_min") else ax_b
                color = _BND_COLOR_MAP[bname]
                # bv uses "P_min" / "LP_min" etc as "boundary" column
                sub = bv[bv["boundary"] == bname].sort_values("prior")
                bnd_key = bname.split("_")[0]   # "P", "LP", "LB", "B"
                T_val   = bnd_t.get(bnd_key, "?")
                if len(sub):
                    ax.plot(sub["prior"], sub["posterior"],
                            color=color, lw=s["lw"], ls=s["ls"],
                            label=f"{lbl}  T={T_val}")
        else:
            for bname in _BND_NAMES:
                ax    = ax_p if bname in ("P_min", "LP_min") else ax_b
                color = _BND_COLOR_MAP[bname]
                col   = f"bnd_post_{bname}"
                bnd_key = bname.split("_")[0]
                T_val   = bnd_t.get(bnd_key, "?")
                if col in df.columns:
                    ax.plot(df["prior"], df[col],
                            color=color, lw=s["lw"], ls=s["ls"],
                            label=f"{lbl}  T={T_val}")

    # Target reference lines (dotted, thin)
    for bname in _BND_NAMES:
        ax     = ax_p if bname in ("P_min", "LP_min") else ax_b
        target = _BND_TARGET[bname]
        color  = _BND_COLOR_MAP[bname]
        ax.axhline(target, color=color, lw=0.9, ls=":", alpha=0.45)

    # Shading
    ax_p.axhspan(0.99, 1.01, color=_BND_COLOR["P"],  alpha=0.08)
    ax_p.axhspan(0.90, 0.99, color=_BND_COLOR["LP"], alpha=0.06)
    ax_b.axhspan(-0.01, 0.10, color=_BND_COLOR["LB"], alpha=0.06)
    ax_b.axhspan(-0.01, 0.01, color=_BND_COLOR["B"],  alpha=0.10)

    ax_p.set_ylim(0.78, 1.005)
    ax_b.set_ylim(-0.005, 0.18)

    for ax, title in [(ax_p, "P / LP boundary posteriors"),
                       (ax_b, "LB / B boundary posteriors")]:
        _apply_xscale(ax, log_scale)
        ax.set_xlabel("Prior", fontsize=11)
        ax.set_ylabel("Posterior P(pathogenic)", fontsize=11)
        ax.set_title(title, fontsize=11)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Boundary posteriors at each method's own boundary T\n"
        "Piecewise variants: flat on targets · Tavtigian: drifts from p=0.10",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, np.array([ax_p, ax_b])


# ── 3. Relative stringency vs continuous reference ───────────────────────────

def plot_relative_stringency(
    tav_df:    pd.DataFrame,
    pw_df:     pd.DataFrame,
    pw_add_df: Optional[pd.DataFrame] = None,
    cont_df:   Optional[pd.DataFrame] = None,
    figsize=(13, 5),
    log_scale: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """log₁₀(LR+_method / LR+_continuous) at each method's own boundary T.

    Each panel shows one ACMG classification boundary.  The y-axis shows
    how many log₁₀ units above or below the analytical Bayesian LR+ the
    method's required LR+ sits.  Zero = exact match.

    Piecewise variants are always exactly 0 (their boundary LR+ equals the
    continuous target by construction).  Tavtigian drifts with prior.
    """
    # For each boundary, the continuous LR+ column name and the piecewise
    # methods' boundary column.  For piecewise_additive the T value differs
    # but the LR+ at that T still equals the continuous target.
    anchors = {
        "P":  ("P_min",  _BND_COLOR["P"]),
        "LP": ("LP_min", _BND_COLOR["LP"]),
        "LB": ("LB_max", _BND_COLOR["LB"]),
        "B":  ("B_max",  _BND_COLOR["B"]),
    }
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    for ax, (name, (bnd_name, color)) in zip(axes, anchors.items()):
        if cont_df is None:
            ax.set_title(f"{name} (no cont_df)", fontsize=10)
            continue
        cont_lr = cont_df[f"lr_{name}"].values

        # Tavtigian
        if tav_df is not None and f"bnd_lr_{bnd_name}" in tav_df.columns:
            tav_lr = tav_df[f"bnd_lr_{bnd_name}"].values
            T_tav  = _METHOD_BND_T["tavtigian"][name.split("_")[0]
                                                if "_" in name else name]
            ax.plot(tav_df["prior"], np.log10(tav_lr / cont_lr),
                    color=color, lw=2, ls="-",
                    label=f"Tavtigian (T={_METHOD_BND_T['tavtigian'].get(name, '?')})")

        # Piecewise original
        if pw_df is not None and f"bnd_lr_{bnd_name}" in pw_df.columns:
            pw_lr = pw_df[f"bnd_lr_{bnd_name}"].values
            T_pw  = _METHOD_BND_T["piecewise"].get(name, "?")
            ax.plot(pw_df["prior"], np.log10(pw_lr / cont_lr),
                    color=color, lw=2, ls="--",
                    label=f"Piecewise (T={T_pw})")

        # Piecewise additive
        if pw_add_df is not None and f"bnd_lr_{bnd_name}" in pw_add_df.columns:
            pw_add_lr = pw_add_df[f"bnd_lr_{bnd_name}"].values
            T_pa = _METHOD_BND_T["piecewise_additive"].get(name, "?")
            ax.plot(pw_add_df["prior"], np.log10(pw_add_lr / cont_lr),
                    color=color, lw=2, ls="-.",
                    label=f"Piecewise-Add (T={T_pa})")

        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{name} boundary", fontsize=10)
        ax.set_xlabel("Prior", fontsize=10)
        _apply_xscale(ax, log_scale)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("log₁₀(LR+_method / LR+_Bayesian)", fontsize=10)
    fig.suptitle(
        "Stringency vs the analytical Bayesian reference  (0 = exact match)\n"
        "Piecewise variants: always 0 at their own boundary T · Tavtigian: drifts",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, axes


# ── 4. log(LR+) vs T (additivity / smoothness) ───────────────────────────────

def plot_log_lr_curves(
    priors=(0.05, 0.10, 0.25, 0.50),
    T_grid=None,
    methods=("tavtigian", "piecewise", "continuous", "recanonical",
             "lsq", "spline"),
    tav_df: Optional[pd.DataFrame] = None,
    figsize=None,
) -> Tuple[plt.Figure, np.ndarray]:
    """log(LR+) vs total points T for each method at multiple priors."""
    import math
    from assay_calibration.fit_utils.bayesian_thresholds import piecewise_log_lr
    from .additivity import (
        recanonical_log_lr, single_alpha_lsq, cubic_spline_log_lr,
        ACMG_T_ANCHORS,
    )

    if T_grid is None:
        T_grid = np.linspace(-12, 18, 400)
    if figsize is None:
        figsize = (5.5 * len(priors), 4.2)

    fig, axes = plt.subplots(1, len(priors), figsize=figsize, sharey=False)
    axes = np.atleast_1d(axes)

    tav_C_at = {}
    if "tavtigian" in methods:
        if tav_df is not None and "C_star" in tav_df.columns:
            for p in priors:
                idx = int((tav_df["prior"] - float(p)).abs().idxmin())
                tav_C_at[float(p)] = float(tav_df.loc[idx, "C_star"])
        else:
            from assay_calibration.fit_utils.evidence_thresholds import (
                get_tavtigian_constant,
            )
            for p in priors:
                tav_C_at[float(p)] = float(get_tavtigian_constant(float(p)))

    for ax, p in zip(axes, priors):
        if "tavtigian" in methods and float(p) in tav_C_at:
            C = tav_C_at[float(p)]
            y = (math.log(C) / 8.0) * T_grid
            ax.plot(T_grid, y, lw=2, color="#d62728", ls="-",
                    label=f'Tavtigian (C={int(C)})')
        if "piecewise" in methods:
            y = piecewise_log_lr(T_grid, float(p))
            ax.plot(T_grid, y, lw=2, color="#1f77b4", ls="--", label="Piecewise")
        if "recanonical" in methods:
            y = recanonical_log_lr(T_grid, float(p))
            ax.plot(T_grid, y, lw=2, color="#9467bd", ls="-.", label="Recanonical")
        if "lsq" in methods:
            alpha = single_alpha_lsq(float(p))
            y = alpha * T_grid
            ax.plot(T_grid, y, lw=2, color="#8c564b", ls=":", label="Single-α LSQ")
        if "spline" in methods:
            y = cubic_spline_log_lr(T_grid, float(p))
            ax.plot(T_grid, y, lw=2, color="#e377c2", ls="--", label="Cubic spline")

        for name, T in ACMG_T_ANCHORS.items():
            ax.axvline(T, color="grey", ls=":", lw=0.7, alpha=0.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Total integer points T")
        ax.set_ylabel("log(LR+)")
        ax.set_title(f"prior = {p:.2f}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "log(LR+) vs T — additive methods are straight; piecewise kinks at "
        "T ∈ {−7,−1,6,10}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, np.asarray(axes)


# ── 5. Boundary posteriors at each method's own anchors ──────────────────────

def plot_boundary_posteriors_all(
    priors: np.ndarray,
    methods=("tavtigian", "piecewise", "continuous", "recanonical"),
    tav_df: Optional[pd.DataFrame] = None,
    figsize=(13, 5),
    log_scale: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """Posterior at each method's own boundary T values vs prior."""
    import math
    from assay_calibration.fit_utils.bayesian_thresholds import (
        bayes_posterior_from_lr, piecewise_posterior,
    )
    from .additivity import recanonical_boundaries, recanonical_posterior, ACMG_T_ANCHORS

    fig, (ax_p, ax_b) = plt.subplots(1, 2, figsize=figsize)
    labels = {
        "P":  ("≥ P  bound", _BND_COLOR["P"],  0.99),
        "LP": ("≥ LP bound", _BND_COLOR["LP"], 0.90),
        "LB": ("≤ LB bound", _BND_COLOR["LB"], 0.10),
        "B":  ("≤ B  bound", _BND_COLOR["B"],  0.01),
    }

    tav_C = None
    if "tavtigian" in methods:
        if tav_df is not None and "C_star" in tav_df.columns:
            tav_C = np.array([
                float(tav_df.loc[(tav_df["prior"] - float(p)).abs().idxmin(), "C_star"])
                for p in priors
            ])
        else:
            from assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant
            tav_C = np.array([float(get_tavtigian_constant(float(p))) for p in priors])

    for method in methods:
        s = _METHOD_STYLE.get(method, {"ls": "-"})
        for name, T_acmg in ACMG_T_ANCHORS.items():
            ax = ax_p if name in ("P", "LP") else ax_b
            lbl, color, target = labels[name]
            posts = np.empty_like(priors, dtype=float)
            for i, p in enumerate(priors):
                if method == "tavtigian" and tav_C is not None:
                    lr = tav_C[i] ** (T_acmg / 8.0)
                    posts[i] = bayes_posterior_from_lr(lr, float(p))
                elif method == "piecewise":
                    posts[i] = piecewise_posterior(T_acmg, float(p))
                elif method == "continuous":
                    posts[i] = target
                elif method == "recanonical":
                    T_recanon = recanonical_boundaries(float(p))[name]
                    posts[i] = recanonical_posterior(T_recanon, float(p))
                else:
                    posts[i] = np.nan
            ax.plot(priors, posts, color=color, lw=2, ls=s["ls"], alpha=0.85)
            ax.axhline(target, color=color, lw=0.6, ls=":", alpha=0.4)

    ax_p.set_ylim(0.78, 1.005)
    ax_b.set_ylim(-0.005, 0.18)
    for ax, title in [(ax_p, "P/LP boundaries"), (ax_b, "LB/B boundaries")]:
        _apply_xscale(ax, log_scale)
        ax.set_xlabel("Prior", fontsize=11)
        ax.set_ylabel("Posterior", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Boundary posteriors per method", fontsize=11)
    fig.tight_layout()
    return fig, np.array([ax_p, ax_b])


# ── 6. Recanonical boundaries vs prior ───────────────────────────────────────

def plot_recanonical_boundaries(
    priors: np.ndarray,
    figsize=(8, 5),
    log_scale: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Method D's boundary T values vs prior, with ACMG references."""
    from .additivity import recanonical_boundaries, ACMG_T_ANCHORS
    fig, ax = plt.subplots(figsize=figsize)
    cols = {"P": _BND_COLOR["P"], "LP": _BND_COLOR["LP"],
            "LB": _BND_COLOR["LB"], "B": _BND_COLOR["B"]}
    for name in ("P", "LP", "LB", "B"):
        Ts = np.array([recanonical_boundaries(float(p))[name] for p in priors])
        ax.plot(priors, Ts, lw=2, color=cols[name], label=f"T_{name} (Method D)")
        ax.axhline(ACMG_T_ANCHORS[name], color=cols[name], ls=":", lw=0.8,
                   alpha=0.6, label=f"T_{name} = {ACMG_T_ANCHORS[name]} (ACMG)")
    _apply_xscale(ax, log_scale)
    ax.set_xlabel("Prior", fontsize=11)
    ax.set_ylabel("Boundary T value", fontsize=11)
    ax.set_title(
        "Method D shifts its boundary T values with prior\n"
        "At prior=0.10 they land on (−6, 0, 11, 17); ACMG fixes (−7, −1, 6, 10)",
        fontsize=11,
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig, ax


# ── 7. Additivity / chokepoint experiment ────────────────────────────────────

def plot_additivity_experiment(
    priors: tuple = (0.05, 0.10, 0.25, 0.50),
    figsize: tuple = (17, 9),
    tav_df: Optional[pd.DataFrame] = None,
    log_scale: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """Diagnose the additivity / chokepoint structure of piecewise vs Tavtigian.

    Layout
    ------
    Row 0 (one panel per prior):
        Incremental Δlog₁₀(LR+) per code step k → k+1 for pathogenic codes 1–8.
        Grouped bars: piecewise (filled) vs Tavtigian (hatched).
        A flat profile = strictly additive; a step at T=6 exposes the chokepoint.

    Row 1 left — Combination violation at prior=0.10:
        violation(k_A, k_B) = log₁₀(LR+(k_A+k_B)) − log₁₀(LR+(k_A)) − log₁₀(LR+(k_B))
        For k_B ∈ {1, 2, 4}, vary k_A ∈ 1..8.
        Tavtigian = 0 everywhere (additive by construction).
        Piecewise deviates whenever k_A + k_B crosses a segment knot (T=6 or T=10).

    Row 1 right — Worst-case combination across priors:
        LR+(8) / LR+(4)² for piecewise and Tavtigian across the full prior range.
        Perfect additivity → ratio = 1.  Piecewise shows a ratio ≈ 0.5 at the
        canonical prior (0.10) because combining two 4-point pieces crosses the
        T=6 chokepoint, halving the expected LR+ product.

    Parameters
    ----------
    priors : tuple of float
        Priors at which to show the incremental-slope panels (row 0).
    tav_df : DataFrame, optional
        Precomputed Tavtigian suite DataFrame.  If None, C* is computed per-prior
        (slow).
    """
    import math
    from assay_calibration.fit_utils.bayesian_thresholds import piecewise_log_lr

    n_priors = len(priors)
    ncols = max(n_priors, 2)

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(2, ncols, figure=fig, hspace=0.45, wspace=0.35)

    axes_row0 = [fig.add_subplot(gs[0, i]) for i in range(n_priors)]
    ax_viol   = fig.add_subplot(gs[1, :ncols // 2])
    ax_ratio  = fig.add_subplot(gs[1, ncols // 2:])

    # Precompute Tavtigian C* at each display prior
    tav_C_at: dict = {}
    if tav_df is not None and "C_star" in tav_df.columns:
        for p in priors:
            idx = int((tav_df["prior"] - float(p)).abs().idxmin())
            tav_C_at[float(p)] = float(tav_df.loc[idx, "C_star"])
    else:
        try:
            from assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant
            for p in priors:
                tav_C_at[float(p)] = float(get_tavtigian_constant(float(p)))
        except Exception:
            pass

    # ── Row 0: incremental Δlog₁₀(LR+) per code step ─────────────────────────
    k_range = list(range(1, 9))   # codes 1..8, step k-1 → k
    x = np.array(k_range, dtype=float)
    bar_w = 0.35

    for ax, p in zip(axes_row0, priors):
        # Piecewise increments
        pw_deltas = []
        for k in k_range:
            d = (float(piecewise_log_lr(k,     float(p))) -
                 float(piecewise_log_lr(k - 1, float(p)))) / math.log(10)
            pw_deltas.append(d)

        # Tavtigian increment (constant = log10(C)/8)
        C = tav_C_at.get(float(p))
        if C is not None:
            tav_delta = math.log10(C) / 8.0
            tav_deltas = [tav_delta] * len(k_range)
        else:
            tav_deltas = [float("nan")] * len(k_range)

        bar_colors = [STRENGTH_COLOR[k] for k in k_range]

        ax.bar(x - bar_w / 2, pw_deltas, width=bar_w, color=bar_colors,
               alpha=0.90, label="Piecewise (ACMG)")
        if C is not None:
            ax.bar(x + bar_w / 2, tav_deltas, width=bar_w, color=bar_colors,
                   alpha=0.35, edgecolor="black", lw=0.5,
                   hatch="///", label="Tavtigian")

        # Mark the ACMG knot position (LP boundary at T=6)
        ax.axvline(6.5, color="#555", ls=":", lw=1.0, alpha=0.7, label="ACMG LP knot (T=6)")

        if C is not None:
            ax.axhline(tav_delta, color="grey", ls="--", lw=0.7, alpha=0.6)

        ax.set_title(f"Δlog₁₀(LR+) per step  (p={p:.2f})", fontsize=9)
        ax.set_xlabel("Code k  (step: k−1 → k)", fontsize=8)
        ax.set_ylabel("Δlog₁₀(LR+)", fontsize=8)
        ax.set_xticks(k_range)
        ax.grid(True, alpha=0.25, axis="y")
        if ax is axes_row0[0]:
            ax.legend(fontsize=6, loc="upper right")

    # ── Row 1 left: combination violation at p=0.10 ───────────────────────────
    p_ref = 0.10
    k_B_cases = [1, 2, 4]
    line_styles = ["-", "--", "-."]
    k_A_range = list(range(1, 9))

    for k_B, ls in zip(k_B_cases, line_styles):
        pw_viols  = []
        valid_kA = [ka for ka in k_A_range if ka + k_B <= 16]
        for k_A in valid_kA:
            k_tot = k_A + k_B
            log_sum_pw = float(piecewise_log_lr(k_tot, p_ref))
            log_A_pw   = float(piecewise_log_lr(k_A,   p_ref))
            log_B_pw   = float(piecewise_log_lr(k_B,   p_ref))
            pw_viols.append((log_sum_pw - log_A_pw - log_B_pw) / math.log(10))

        pts = np.array(valid_kA)
        ax_viol.plot(pts, pw_viols, ls=ls, lw=2, color="#1f77b4",
                     marker="o", ms=4, label=f"Piecewise  k_B={k_B}")
        ax_viol.axhline(0, color="#d62728", lw=1.0, ls="--", alpha=0.5,
                        zorder=0)  # Tavtigian ≡ 0 reference (drawn once below)

    ax_viol.axhline(0, color="grey", lw=0.9)
    ax_viol.axvline(6.5, color="#444", ls=":", lw=1.1, label="T=6 knot")
    ax_viol.set_xlabel("k_A  (first evidence code)", fontsize=10)
    ax_viol.set_ylabel(
        "log₁₀(LR+(k_A+k_B)) − log₁₀(LR+(k_A)) − log₁₀(LR+(k_B))",
        fontsize=8,
    )
    ax_viol.set_title(
        f"Combination violation at prior={p_ref:.2f}\n"
        "Piecewise deviates when sum crosses T=6 or T=10 knot · Tavtigian ≡ 0 (red dashed)",
        fontsize=9,
    )
    ax_viol.grid(True, alpha=0.3)
    ax_viol.legend(fontsize=7, ncol=2)

    # ── Row 1 right: worst-case ratio across priors (k_A=k_B=4) ─────────────
    if log_scale:
        prior_dense = np.logspace(-3, np.log10(0.80), 400)
    else:
        prior_dense = np.linspace(0.01, 0.45, 400)

    pw_ratios  = []
    for p in prior_dense:
        ll8_pw = float(piecewise_log_lr(8, float(p)))
        ll4_pw = float(piecewise_log_lr(4, float(p)))
        pw_ratios.append(math.exp(ll8_pw - 2.0 * ll4_pw))

    ax_ratio.plot(prior_dense, pw_ratios, color="#1f77b4", lw=2,
                  label="Piecewise (ACMG)")
    ax_ratio.axhline(1.0, color="#d62728", lw=2, ls="--",
                     label="Tavtigian ≡ 1 (additive)")
    ax_ratio.axhline(1.0, color="grey", lw=0.9, ls=":", alpha=0.5)
    ax_ratio.axvline(0.10, color="#888", lw=0.8, ls="--", alpha=0.5,
                     label="Prior = 0.10 (ACMG canonical)")

    _apply_xscale(ax_ratio, log_scale)

    ax_ratio.set_xlabel("Prior", fontsize=10)
    ax_ratio.set_ylabel("LR+(k=8)  /  LR+(k=4)²", fontsize=10)
    ax_ratio.set_title(
        "Two 'Strong' pieces combined  (k_A = k_B = 4 → T = 8)\n"
        "Ratio < 1: piecewise crossing T=6 knot underestimates product",
        fontsize=9,
    )
    ax_ratio.legend(fontsize=8)

    fig.suptitle(
        "Additivity diagnosis: piecewise chokepoints vs Tavtigian (strictly additive)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    all_axes = np.array(axes_row0 + [ax_viol, ax_ratio], dtype=object)
    return fig, all_axes


# ── 8. Combined error comparison: boundary drift vs combination error ─────────

def plot_combined_error_comparison(
    priors: np.ndarray = None,
    tav_df: Optional[pd.DataFrame] = None,
    combos: Optional[list] = None,
    figsize: tuple = (14, 12),
    log_scale: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """Compare Tavtigian boundary-drift error vs piecewise combination error.

    The two methods have complementary failure modes:

    * **Tavtigian**: strictly additive (zero combination error) but its
      per-tier LR+ thresholds are calibrated only at prior=0.10.  At other
      priors the boundary posteriors drift away from the ACMG targets.

    * **Piecewise α**: exact boundary posteriors at every prior, but combining
      two assays by integer-point addition introduces error when the total
      crosses a segment knot (most visibly at T=6 for two Strong pieces).

    Layout (2 rows × 2 columns)
    ----------------------------
    (0,0)  Single-assay boundary error vs prior.
           |posterior_method(T_boundary, p) − target| for T ∈ {10, 6, −1, −7}.
           Piecewise = 0 (flat, by construction).  Tavtigian drifts.

    (0,1)  Two-assay self-consistency error for piecewise vs prior.
           post(k_A+k_B) − Bayes(LR+(k_A, p) · LR+(k_B, p), p) — signed.
           Tavtigian ≡ 0 everywhere (additive: C^(kA+kB) = C^kA · C^kB).
           Piecewise deviates when the total T crosses a segment knot.

    (1,0)  Net winner: |err_tav_vs_pw_ref| − |err_pw_self|.
           Positive → piecewise wins; negative → Tavtigian wins.
           tav reference = piecewise per-code LR+ product; both on same scale.

    (1,1)  Bar breakdown of absolute errors at p=0.10 and p=0.25.

    Parameters
    ----------
    priors : 1-D array, optional
        Dense prior grid.  Defaults to logspace(−3, log10(0.80), 500).
    tav_df : DataFrame, optional
        Precomputed Tavtigian suite DataFrame.  If None, C* is recomputed
        per prior (slower).
    combos : list of (k_A, k_B) tuples, optional
        Evidence-code pairs to analyse.  Default: [(4,4), (2,4), (2,2), (6,6)].
    """
    import math
    from assay_calibration.fit_utils.bayesian_thresholds import (
        piecewise_log_lr, piecewise_posterior,
        bayes_posterior_from_lr,
    )
    from assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant

    # ── Defaults ─────────────────────────────────────────────────────────────
    if priors is None:
        priors = np.logspace(-3, np.log10(0.80), 500)
    if combos is None:
        combos = [(4, 4), (2, 4), (2, 2), (6, 6)]

    # ACMG boundary (T, target_posterior, display colour)
    _BND_INFO = [
        ("P",  10,  0.99, _BND_COLOR["P"]),
        ("LP",  6,  0.90, _BND_COLOR["LP"]),
        ("LB", -1,  0.10, _BND_COLOR["LB"]),
        ("B",  -7,  0.01, _BND_COLOR["B"]),
    ]

    # Combo line styles and colours
    _COMBO_LS   = ["-", "--", "-.", ":"]
    _COMBO_COLS = [STRENGTH_COLOR[8], STRENGTH_COLOR[6],
                   STRENGTH_COLOR[4], STRENGTH_COLOR[3]]

    # ── Retrieve C* across the prior grid ───────────────────────────────────
    if tav_df is not None and "C_star" in tav_df.columns:
        C_arr = np.array([
            float(tav_df.loc[(tav_df["prior"] - float(p)).abs().idxmin(), "C_star"])
            for p in priors
        ])
    else:
        C_arr = np.array([float(get_tavtigian_constant(float(p))) for p in priors])

    # ── Figure / axes ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_single  = axes[0, 0]   # single-assay boundary error
    ax_combo   = axes[0, 1]   # two-assay combination error
    ax_net     = axes[1, 0]   # which method wins
    ax_bar     = axes[1, 1]   # bar breakdown at two reference priors

    # ══════════════════════════════════════════════════════════════════════════
    # Panel (0,0): Single-assay boundary error (MAE + log-odds)
    # ══════════════════════════════════════════════════════════════════════════
    for name, T_bnd, target, color in _BND_INFO:
        # Tavtigian: posterior at the ACMG integer boundary T
        tav_posts = np.array([
            bayes_posterior_from_lr(C_arr[i] ** (T_bnd / 8.0), float(p))
            for i, p in enumerate(priors)
        ])
        tav_mae = np.abs(tav_posts - target)
        tav_loe = abs_log_odds_error(tav_posts, target)
        ax_single.plot(priors, tav_mae, color=color, lw=2, ls="-", alpha=0.7,
                       label=f"Tavtigian T={T_bnd} MAE")
        ax_single.plot(priors, tav_loe, color=color, lw=2, ls="--", alpha=0.5,
                       label=f"Tavtigian T={T_bnd} log-odds")

    # Piecewise is exactly 0 at all four boundaries — draw a single flat reference
    ax_single.axhline(0, color="grey", lw=1.2, ls=":", alpha=0.7,
                      label="Piecewise = 0 (exact by construction)")

    ax_single.set_xlabel("Prior", fontsize=10)
    ax_single.set_ylabel("Boundary error (MAE, log-odds units)", fontsize=10)
    ax_single.set_title(
        "Single-assay boundary error: MAE (solid) vs log-odds (dashed)\n"
        "Piecewise: always 0 · Tavtigian: drifts from calibrated p=0.10",
        fontsize=10,
    )
    _apply_xscale(ax_single, log_scale)
    ax_single.axvline(0.10, color="#888", lw=0.8, ls=":", alpha=0.6)
    ax_single.legend(fontsize=7, loc="upper right", ncol=2)

    # ══════════════════════════════════════════════════════════════════════════
    # Panel (0,1): Two-assay combination error vs prior
    # ══════════════════════════════════════════════════════════════════════════
    # combo_data: (k_A, k_B) -> (tav_errs_vs_pw, pw_self_errs)
    #
    # pw_self_errs   — piecewise self-consistency
    #                  = post_pw(k_A+k_B) − Bayes(LR_pw(kA,p) · LR_pw(kB,p), p)
    # tav_errs_vs_pw — Tavtigian combined posterior vs the same piecewise reference
    #                  (same denominator, so the two are directly comparable)
    #
    # Tavtigian self-consistency is always 0 (C^(kA+kB) = C^kA · C^kB),
    # but its combined posterior differs from the piecewise ground truth because
    # its per-code LR+ values are not calibrated at the ACMG posterior targets.
    combo_data = {}
    for (k_A, k_B), ls, color in zip(combos, _COMBO_LS, _COMBO_COLS):
        k_tot = k_A + k_B

        tav_errs_vs_pw = np.empty(len(priors))
        pw_self_errs   = np.empty(len(priors))

        for i, p in enumerate(priors):
            # Shared reference: piecewise per-code LR+ multiplied independently
            lr_A_pw = math.exp(float(piecewise_log_lr(k_A, float(p))))
            lr_B_pw = math.exp(float(piecewise_log_lr(k_B, float(p))))
            post_true_pw = bayes_posterior_from_lr(lr_A_pw * lr_B_pw, float(p))

            # Piecewise self-consistency
            lr_pw_comb   = math.exp(float(piecewise_log_lr(k_tot, float(p))))
            post_pw_comb = bayes_posterior_from_lr(lr_pw_comb, float(p))
            pw_self_errs[i] = post_pw_comb - post_true_pw

            # Tavtigian vs piecewise reference
            lr_tav_comb   = C_arr[i] ** (k_tot / 8.0)
            post_tav_comb = bayes_posterior_from_lr(lr_tav_comb, float(p))
            tav_errs_vs_pw[i] = post_tav_comb - post_true_pw

        combo_data[(k_A, k_B)] = (tav_errs_vs_pw, pw_self_errs)

        ax_combo.plot(priors, pw_self_errs,  color=color, lw=2, ls=ls)
        ax_combo.plot(priors, tav_errs_vs_pw, color=color, lw=1.0, ls=ls,
                      alpha=0.45)

    ax_combo.axhline(0, color="grey", lw=0.8, ls=":")
    ax_combo.axvline(0.10, color="#888", lw=0.8, ls=":", alpha=0.6)
    ax_combo.axvline(0.181, color=STRENGTH_COLOR[4], lw=1.0, ls=":",
                     alpha=0.5, label="Sign flip p≈0.181 (4,4)")

    ax_combo.set_xlabel("Prior", fontsize=10)
    ax_combo.set_ylabel("Combined posterior − true posterior", fontsize=10)
    ax_combo.set_title(
        "Two-assay combination error vs prior\n"
        "Piecewise self-consistency (solid) · Tavtigian vs PW reference (faint)",
        fontsize=10,
    )
    _apply_xscale(ax_combo, log_scale)

    combo_handles = [
        Line2D([0], [0], color=col, lw=2, ls=ls, label=f"Piecewise k=({kA},{kB})")
        for (kA, kB), ls, col in zip(combos, _COMBO_LS, _COMBO_COLS)
    ]
    combo_handles.append(
        Line2D([0], [0], color="grey", lw=1.5, ls="-", alpha=0.6,
               label="Tavtigian (faint, vs PW ref)")
    )
    ax_combo.legend(handles=combo_handles, fontsize=7, loc="best")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel (1,0): Net winner — |err_tav| − |err_pw| and |err_tav| − |err_pa|
    # ══════════════════════════════════════════════════════════════════════════
    for (k_A, k_B), ls, color in zip(combos, _COMBO_LS, _COMBO_COLS):
        tav_errs_vs_pw, pw_self_errs = combo_data[(k_A, k_B)]
        # Positive → piecewise wins; negative → Tavtigian wins
        net = np.abs(tav_errs_vs_pw) - np.abs(pw_self_errs)
        ax_net.plot(priors, net, color=color, lw=2, ls=ls,
                    label=f"k=({k_A},{k_B})")

    ax_net.axhline(0, color="grey", lw=1.0, ls="--")
    ax_net.axvline(0.10, color="#888", lw=0.8, ls=":", alpha=0.6,
                   label="p=0.10 (ACMG canonical)")

    ax_net.fill_between(
        [priors[0], priors[-1]], [0, 0], [0.30, 0.30],
        color="#e6b1b8", alpha=0.06, zorder=0,
    )
    ax_net.fill_between(
        [priors[0], priors[-1]], [-0.30, -0.30], [0, 0],
        color="#d0e8f0", alpha=0.06, zorder=0,
    )
    ax_net.text(priors[5], 0.003,
                "Piecewise wins\n(|err_tav| > |err_pw|)",
                fontsize=7, color=_BND_COLOR["LP"], va="bottom")
    ax_net.text(priors[5], -0.003,
                "Tavtigian wins\n(|err_pw| > |err_tav|)",
                fontsize=7, color=_BND_COLOR["LB"], va="top")

    ax_net.set_xlabel("Prior", fontsize=10)
    ax_net.set_ylabel("|err_Tavtigian| − |err_Piecewise|", fontsize=10)
    ax_net.set_title(
        "Net winner per combination (+ = piecewise better)\n"
        "Both measured vs same piecewise per-code LR+ reference",
        fontsize=10,
    )
    _apply_xscale(ax_net, log_scale)
    ax_net.legend(fontsize=8, loc="best")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel (1,1): Error breakdown at p=0.10 and p=0.25.
    # Both errors measured vs the same piecewise per-code LR+ reference.
    # Tavtigian: its combined posterior vs piecewise ground truth.
    # Piecewise: self-consistency (post(kA+kB) vs product of individual LR+).
    # ══════════════════════════════════════════════════════════════════════════
    ref_priors = [0.10, 0.25]
    x = np.arange(len(combos))
    bar_w = 0.18
    group_offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w

    tav_colors = ["#444444", "#888888"]   # dark/light for p=0.10, p=0.25
    pw_colors  = ["#1f77b4", "#aec7e8"]

    combo_labels = [f"k=({kA},{kB})" for kA, kB in combos]

    for j, p_ref in enumerate(ref_priors):
        i_ref = int((np.abs(priors - p_ref)).argmin())

        tav_abs = np.array([
            abs(combo_data[(kA, kB)][0][i_ref]) for kA, kB in combos
        ])
        pw_abs = np.array([
            abs(combo_data[(kA, kB)][1][i_ref]) for kA, kB in combos
        ])

        base_tav = x + group_offsets[j * 2]
        base_pw  = x + group_offsets[j * 2 + 1]

        ax_bar.bar(base_tav, tav_abs, width=bar_w,
                   color=tav_colors[j], alpha=0.85,
                   label=f"Tavtigian p={p_ref:.2f}")
        ax_bar.bar(base_pw, pw_abs, width=bar_w,
                   color=pw_colors[j], alpha=0.85, hatch="///",
                   edgecolor=pw_colors[j],
                   label=f"Piecewise p={p_ref:.2f}")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(combo_labels, fontsize=9)
    ax_bar.set_xlabel("Evidence combination (k_A, k_B)", fontsize=10)
    ax_bar.set_ylabel("|combined posterior − reference posterior|", fontsize=10)
    ax_bar.set_title(
        "Absolute combination error at two priors\n"
        "Tavtigian (solid) vs Piecewise self-consistency (hatched)\n"
        "Both vs piecewise per-code LR+ product as reference",
        fontsize=9,
    )
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.3, axis="y")

    # ── Finish ────────────────────────────────────────────────────────────────
    fig.suptitle(
        "Boundary-posterior drift (Tavtigian) vs combination error (Piecewise)\n"
        "Tavtigian: exact at p=0.10 · Piecewise: exact boundaries, small combo error",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, axes


# ── 9. Additivity dilemma: clinical intuition ────────────────────────────────

def plot_additivity_dilemma(
    priors: np.ndarray = None,
    tav_df: Optional[pd.DataFrame] = None,
    demo_priors: tuple = (0.05, 0.10, 0.25),
    figsize: tuple = (16, 10),
    log_scale: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """Head-to-head error comparison: Tavtigian posterior drift vs Piecewise additivity error.

    Both errors measured against the same ground truth (Bayesian posterior from
    piecewise per-code LR+ product). The question: which problem is bigger in practice?

    Layout (2 rows × 3 columns)
    ---------------------------
    Row 0 — SINGLE-ASSAY BOUNDARIES (single evidence at one tier):
      (0,0)  LP (T=6): Tavtigian posterior vs exact 0.90 target.
      (0,1)  P (T=10): Tavtigian posterior vs exact 0.99 target.
      (0,2)  |error| at all four boundaries for both methods.

    Row 1 — MULTI-ASSAY COMBINATIONS (two pieces of evidence):
      (1,0)  Example: "1 VS vs 2 Strong" (both → T=8).
             Bar chart: expected posterior, piecewise result, Tavtigian result.
             Ground truth = piecewise individual LR+ product.
      (1,1)  All 36 evidence combinations (pathogenic, benign, mixed).
             Error envelope: Tavtigian posterior error vs Piecewise additivity error.
             Which method is more accurate across the full space?
      (1,2)  Error comparison: which method has smaller error at each prior?
             Shows the crossover: where does one become better than the other?
    """
    import math
    from assay_calibration.fit_utils.bayesian_thresholds import (
        piecewise_log_lr, piecewise_posterior,
        bayes_posterior_from_lr,
    )
    from assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant

    if priors is None:
        priors = np.linspace(0.01, 0.45, 300)

    # Retrieve C* across the prior grid
    if tav_df is not None and "C_star" in tav_df.columns:
        C_arr = np.array([
            float(tav_df.loc[(tav_df["prior"] - float(p)).abs().idxmin(), "C_star"])
            for p in priors
        ])
    else:
        C_arr = np.array([float(get_tavtigian_constant(float(p))) for p in priors])

    # Demo C* for bar chart (retrieve for each demo_prior)
    demo_C_tav = {}
    if tav_df is not None and "C_star" in tav_df.columns:
        for p in demo_priors:
            demo_C_tav[p] = float(
                tav_df.loc[(tav_df["prior"] - float(p)).abs().idxmin(), "C_star"]
            )
    else:
        for p in demo_priors:
            demo_C_tav[p] = float(get_tavtigian_constant(float(p)))

    _BND_INFO = [
        ("P",  10, 0.99, _BND_COLOR["P"]),
        ("LP",  6, 0.90, _BND_COLOR["LP"]),
        ("LB", -1, 0.10, _BND_COLOR["LB"]),
        ("B",  -7, 0.01, _BND_COLOR["B"]),
    ]

    # Full combination space: pathogenic, benign, and mixed.
    # Codes use the ACMG tier integers: ±1, ±2, ±4, ±8.
    # Negative codes = benign evidence (LR+ < 1, reduces pathogenicity probability).
    _path_codes = [1, 2, 4, 8]
    _ben_codes  = [-1, -2, -4, -8]
    path_combos   = [(kA, kB) for kA in range(1, 9) for kB in range(kA, 9)]
    benign_combos = [(-kA, -kB) for kA in [1, 2, 4, 8] for kB in [1, 2, 4, 8] if kB >= kA]
    mixed_combos  = [(kA, -kB) for kA in [1, 2, 4, 8] for kB in [1, 2, 4, 8]]
    combos_all = path_combos + benign_combos + mixed_combos

    def _segment(t):
        """Piecewise segment index: 0 = B–LB (T≤−1), 1 = LB–LP (−1<T≤6), 2 = LP–P (T>6)."""
        return 0 if t <= -1 else (1 if t <= 6 else 2)

    combo_types = []
    for kA, kB in combos_all:
        k_tot = kA + kB
        if kA > 0 and kB > 0:          # both pathogenic
            if _segment(kA) == _segment(kB) == _segment(k_tot):
                combo_types.append("within_path")
            else:
                combo_types.append("cross_path")
        elif kA < 0 and kB < 0:         # both benign
            if _segment(kA) == _segment(kB) == _segment(k_tot):
                combo_types.append("within_ben")
            else:
                combo_types.append("cross_ben")
        else:                            # mixed pathogenic + benign
            combo_types.append("mixed")

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    ax_lp, ax_p, ax_drift = axes[0]
    ax_bars, ax_envelope, ax_frac = axes[1]

    # ══════════════════════════════════════════════════════════════════════════
    # Row 0 left/middle: LP and P boundary posteriors
    # ══════════════════════════════════════════════════════════════════════════
    for ax, (name, T_bnd, target, color), ylim_lo in [
        (ax_lp, _BND_INFO[1], 0.88),   # LP  ylim starts at 0.88 (shows 0.90 clearly)
        (ax_p,  _BND_INFO[0], 0.985),  # P   ylim starts at 0.985
    ]:
        tav_posts = np.array([
            bayes_posterior_from_lr(C_arr[i] ** (T_bnd / 8.0), float(p))
            for i, p in enumerate(priors)
        ])

        ax.plot(priors, tav_posts, color=color, lw=2.5, ls="-",
                label="Tavtigian (C*)")
        # Piecewise is exactly the target by construction — draw as dashed line
        ax.axhline(target, color="#1f77b4", lw=2.5, ls="--",
                   label=f"Piecewise (exact = {target:.0%})", alpha=0.9)
        ax.axvline(0.10, color="#888", lw=1.0, ls=":", alpha=0.7,
                   label="p=0.10 (calibration prior)")
        ax.fill_between(priors, tav_posts, target,
                        where=(tav_posts < target) if target > 0.5 else (tav_posts > target),
                        alpha=0.12, color=color, label="Tavtigian error")
        _apply_xscale(ax, log_scale)
        ax.set_xlabel("Gene-specific prior", fontsize=10)
        ax.set_ylabel("Posterior probability", fontsize=10)
        ax.set_ylim(ylim_lo, 1.002)
        ax.set_title(
            f'"{name}" classification: should mean {target:.0%} confidence\n'
            f'(ACMG boundary T = {T_bnd})',
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="best")

    # ══════════════════════════════════════════════════════════════════════════
    # Row 0 right: boundary drift summary — all four ACMG boundaries
    # ══════════════════════════════════════════════════════════════════════════
    for name, T_bnd, target, color in _BND_INFO:
        tav_posts = np.array([
            bayes_posterior_from_lr(C_arr[i] ** (T_bnd / 8.0), float(p))
            for i, p in enumerate(priors)
        ])
        ax_drift.plot(priors, np.abs(tav_posts - target), color=color, lw=2,
                      label=f"Tavtigian  {name} (T={T_bnd})")
    ax_drift.axhline(0, color="#1f77b4", lw=2, ls="--", alpha=0.85,
                     label="Piecewise = 0 (exact at every prior)")
    ax_drift.axvline(0.10, color="#888", lw=1.0, ls=":", alpha=0.7)
    ax_drift.set_xlabel("Gene-specific prior", fontsize=10)
    ax_drift.set_ylabel("|posterior − target|", fontsize=10)
    ax_drift.set_title(
        "Tavtigian boundary error at all four ACMG thresholds\n"
        "Calibrated only at p=0.10 — drifts everywhere else",
        fontsize=10,
    )
    _apply_xscale(ax_drift, log_scale)
    ax_drift.legend(fontsize=7, loc="upper right")

    # ── Bar chart: error for representative combos of each type ─────────────────
    # Four representative combinations, one per combo type:
    #   within_path:  M + M → T=4    (stays in LB–LP segment; piecewise error = 0)
    #   cross_path:   S + S → T=8    (crosses T=6 knot; piecewise has kink error)
    #   within_ben:   S_B+S_B → T=−8 (stays in B–LB segment; piecewise error = 0)
    #   mixed:        VS + S_B → T=4  (Tavtigian error = 0 by construction)
    rep_combos = [
        (2, 2,  "M+M\n(within LB–LP)"),
        (4, 4,  "S+S\n(cross T=6)"),
        (-4, -4, "S$_B$+S$_B$\n(within B–LB)"),
        (8, -4, "VS+S$_B$\n(mixed)"),
    ]
    bar_w   = 0.30
    x_rep   = np.arange(len(rep_combos))
    offsets = np.array([-0.5, 0.5]) * bar_w

    p_ref = 0.10
    C_ref = demo_C_tav[p_ref]

    pw_heights  = []
    tav_heights = []
    for kA, kB, _ in rep_combos:
        k_tot = kA + kB
        lr_A = math.exp(float(piecewise_log_lr(kA, float(p_ref))))
        lr_B = math.exp(float(piecewise_log_lr(kB, float(p_ref))))
        ref  = float(bayes_posterior_from_lr(lr_A * lr_B, float(p_ref)))
        pw_heights.append(abs(float(piecewise_posterior(k_tot, float(p_ref))) - ref))
        tav_heights.append(abs(float(bayes_posterior_from_lr(
            C_ref ** (k_tot / 8.0), float(p_ref))) - ref))

    ax_bars.bar(x_rep + offsets[0], pw_heights,  width=bar_w,
                color="#1f77b4", alpha=0.85, label="Piecewise (additivity error)")
    ax_bars.bar(x_rep + offsets[1], tav_heights, width=bar_w,
                color="#d62728", alpha=0.85, label="Tavtigian (posterior error)")

    ax_bars.axhline(0, color="grey", lw=0.8, ls=":", alpha=0.5)
    ax_bars.set_xticks(x_rep)
    ax_bars.set_xticklabels([lbl for _, _, lbl in rep_combos], fontsize=9)
    ax_bars.set_ylabel("|error| vs Bayesian ground truth", fontsize=10)
    ax_bars.set_title(
        f"Representative combos at p = {p_ref:.2f}\n"
        "Within-segment: piecewise error = 0 (exact additivity)\n"
        "Tavtigian: error only from posterior drift, not combination arithmetic",
        fontsize=9,
    )
    ax_bars.legend(fontsize=8, loc="upper right")
    ax_bars.grid(True, alpha=0.25, axis="y")

    # ══════════════════════════════════════════════════════════════════════════
    # Compute error matrices for ALL combo types (pathogenic, benign, mixed)
    # ══════════════════════════════════════════════════════════════════════════
    n_p = len(priors)
    n_c = len(combos_all)
    pw_err_mat  = np.empty((n_c, n_p))
    tav_err_mat = np.empty((n_c, n_p))

    for ci, (kA, kB) in enumerate(combos_all):
        k_tot = kA + kB
        for pi, p in enumerate(priors):
            lr_A = math.exp(float(piecewise_log_lr(kA, float(p))))
            lr_B = math.exp(float(piecewise_log_lr(kB, float(p))))
            ref  = float(bayes_posterior_from_lr(lr_A * lr_B, float(p)))
            pw_err_mat[ci, pi]  = abs(float(piecewise_posterior(k_tot, float(p))) - ref)
            tav_err_mat[ci, pi] = abs(float(bayes_posterior_from_lr(
                C_arr[pi] ** (k_tot / 8.0), float(p)
            )) - ref)

    # ── Middle panel: median error per combo type ─────────────────────────────
    # For each type, plot both piecewise and Tavtigian error.
    # Key expectation:
    #   within_path / within_ben: piecewise error ≈ 0 (exact additivity in segment)
    #   cross_path / cross_ben:   piecewise error > 0 (kink at knot)
    #   mixed:                    Tavtigian error ≈ 0 (C^(a+b) = C^a * C^b always)
    type_spec = [
        ("within_path", "#1f77b4", "--",  "#1f77b4", "-",  "Within-seg (path)"),
        ("cross_path",  "#ff7f0e", "--",  "#ff7f0e", "-",  "Cross-knot (path)"),
        ("within_ben",  "#4b91a6", ":",   "#4b91a6", "-.", "Within-seg (ben)"),
        ("cross_ben",   "#9467bd", ":",   "#9467bd", "-.", "Cross-knot (ben)"),
        ("mixed",       "#2ca02c", "--",  "#2ca02c", "-",  "Mixed (path+ben)"),
    ]
    for ctype, pw_col, pw_ls, tav_col, tav_ls, lbl in type_spec:
        idx = [ci for ci, t in enumerate(combo_types) if t == ctype]
        if not idx:
            continue
        pw_med  = np.median(pw_err_mat[idx, :],  axis=0)
        tav_med = np.median(tav_err_mat[idx, :], axis=0)
        ax_envelope.plot(priors, pw_med,  color=pw_col,  lw=1.8, ls=pw_ls,
                         alpha=0.85, label=f"PW {lbl}")
        ax_envelope.plot(priors, tav_med, color=tav_col, lw=1.8, ls=tav_ls,
                         alpha=0.85, label=f"Tav {lbl}")

    ax_envelope.axhline(0, color="black", lw=0.5)
    ax_envelope.axvline(0.10, color="#888", lw=1.0, ls=":", alpha=0.7,
                        label="p=0.10 (Tav calibrated)")
    ax_envelope.set_xlabel("Gene-specific prior", fontsize=10)
    ax_envelope.set_ylabel("|posterior error|", fontsize=10)
    ax_envelope.set_title(
        "Median error by combination type: Piecewise (dashed) vs Tavtigian (solid)\n"
        "Within-segment: PW error = 0 (exact additivity) | Cross-knot: PW has bounded error\n"
        "Mixed (path+ben): Tavtigian error = 0 (C$^{a+b}$ = C$^a$·C$^b$ by construction)",
        fontsize=9,
    )
    _apply_xscale(ax_envelope, log_scale)
    ax_envelope.legend(fontsize=6, loc="upper right", ncol=2)

    # ── Right panel: error difference (Piecewise − Tavtigian) per combo type ──
    # Positive = piecewise better (smaller error); Negative = Tavtigian better.
    type_diff_spec = [
        ("within_path", "#1f77b4", "-",  "Within-seg path"),
        ("cross_path",  "#ff7f0e", "--", "Cross-knot path"),
        ("within_ben",  "#4b91a6", "-.", "Within-seg ben"),
        ("cross_ben",   "#9467bd", ":",  "Cross-knot ben"),
        ("mixed",       "#2ca02c", "-",  "Mixed"),
    ]
    for ctype, color, ls, lbl in type_diff_spec:
        idx = [ci for ci, t in enumerate(combo_types) if t == ctype]
        if not idx:
            continue
        pw_med  = np.median(pw_err_mat[idx, :],  axis=0)
        tav_med = np.median(tav_err_mat[idx, :], axis=0)
        ax_frac.plot(priors, pw_med - tav_med, color=color, lw=2.0, ls=ls, label=lbl)

    ax_frac.axhline(0, color="grey", lw=1.2, ls="-", alpha=0.8)
    ax_frac.axvline(0.10, color="#888", lw=1.0, ls=":", alpha=0.7,
                    label="p=0.10 (Tav calibrated)")

    # Shade the two regions
    ymax = ax_frac.get_ylim()[1] if ax_frac.get_ylim()[1] > 0 else 0.05
    ax_frac.text(0.02, 0.97, "← Piecewise wins", transform=ax_frac.transAxes,
                 fontsize=8, color="#1f77b4", va="top")
    ax_frac.text(0.02, 0.03, "← Tavtigian wins", transform=ax_frac.transAxes,
                 fontsize=8, color="#d62728", va="bottom")

    ax_frac.set_xlabel("Gene-specific prior", fontsize=10)
    ax_frac.set_ylabel("Error difference (PW − Tav)", fontsize=10)
    ax_frac.set_title(
        "Error advantage by combination type\n"
        "Above 0: Piecewise more accurate | Below 0: Tavtigian more accurate",
        fontsize=10,
    )
    _apply_xscale(ax_frac, log_scale)
    ax_frac.legend(fontsize=8, loc="center right")

    fig.suptitle(
        "Error comparison by combination type: where does each method fail?\n"
        "Within-knot: piecewise IS additive (same slope) | Across-knot: piecewise breaks (kinks) | Mixed: Tavtigian additivity test",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    return fig, axes


# ── 9b. Combination paths: tier ratios and boundary paths ────────────────────

def plot_combination_paths(
    priors: np.ndarray = None,
    tav_df: Optional[pd.DataFrame] = None,
    figsize: tuple = (16, 12),
    log_scale: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """Dissect how piecewise and Tavtigian handle evidence combination.

    Two complementary questions:

    Q1 — Do Su:M:S:VS follow 1:2:4:8 in LR+ terms?
      A perfectly additive method has log_lr(k) = α·k, so
      log_lr(2k)/log_lr(k) = 2 for every tier doubling.
      Piecewise fails this because each segment has a non-zero y-intercept
      (the segment does not pass through the origin), and because
      VS (code k=8) lives in a different segment from Su/M/S (codes 1,2,4).

    Q2 — When combining standard evidence to reach the LP (T=6) or P (T=10)
      boundary, does each method give the correct posterior?
      Two sub-cases are distinguished:
        (a) All codes in the middle segment (1,2,4) sum to ≤ 6, then
            further evidence pushes the total above 6 → crosses the T=6 knot.
        (b) Paths that include VS (code 8), which is already in the upper
            segment above T=6.

      For piecewise: the POINT-SUM posterior is exact at every boundary by
      construction.  The LR-PRODUCT of individual codes may differ from the
      point-sum posterior (combination error).
      For Tavtigian: the LR-product always equals the point-sum (additive),
      but the point-sum posterior drifts from the target at priors ≠ 0.10.

    Layout (3 rows × 2 columns)
    ---------------------------
    (0,0)  Per-point log-LR by tier (log_lr(k, p) / k vs prior).
           Tavtigian: single flat line — tier ratios exact.
           Piecewise: different values for lower-segment codes (1,2,4) vs
           upper-segment code 8 (VS).  Inconsistency visible as split.

    (0,1)  "Doubling" ratio: LR+(2k) / LR+(k)² vs prior.
           Should be 1.0 for a perfectly additive method.
           Tavtigian = 1.0 always.
           Piecewise deviates, most for S→VS (4+4→8, crosses T=6 knot).

    (1,0)  LP boundary (T=6) combination paths.
           Signed error = post_pw(T_total) − Bayes(∏LR+(ki), p) for each
           standard path to T=6 (S+M, 3M, S+Su+Su).
           Tavtigian reference line: post_tav(T=6) − 0.90 (boundary drift).

    (1,1)  P boundary (T=10), lower-segment paths only (no VS).
           Paths where all codes are ≤4, sum crosses T=6 knot to reach T=10.
           (S+S+M, S+S+Su+Su, S+M+M+M, ...).
           Same error metric vs Tavtigian boundary drift.

    (2,0)  P boundary (T=10), paths involving VS (code 8).
           (VS+M, VS+Su+Su).

    (2,1)  Summary: for each panel's paths, max |piecewise LR-product error|
           vs |Tavtigian boundary drift| across all priors.
           Shows directly which is the larger error and by how much.
    """
    import math
    from assay_calibration.fit_utils.bayesian_thresholds import (
        piecewise_log_lr, piecewise_posterior,
        bayes_posterior_from_lr,
    )
    from assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant

    if priors is None:
        priors = np.linspace(0.01, 0.45, 300)

    # Retrieve C* across the prior grid
    if tav_df is not None and "C_star" in tav_df.columns:
        C_arr = np.array([
            float(tav_df.loc[(tav_df["prior"] - float(p)).abs().idxmin(), "C_star"])
            for p in priors
        ])
    else:
        C_arr = np.array([float(get_tavtigian_constant(float(p))) for p in priors])

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    ax_perpt, ax_doubling = axes[0]
    ax_lp6, ax_p10_lower = axes[1]
    ax_p10_vs, ax_summary = axes[2]

    # Colour coding: Su=pink, M=salmon, S=red, VS=dark red
    tier_colors = {1: STRENGTH_COLOR[2], 2: STRENGTH_COLOR[4],
                   4: STRENGTH_COLOR[6], 8: STRENGTH_COLOR[8]}
    tier_labels = {1: "Su (k=1)", 2: "M (k=2)", 4: "S (k=4)", 8: "VS (k=8)"}

    # ══════════════════════════════════════════════════════════════════════════
    # (0,0) Per-point log-LR by tier
    # ══════════════════════════════════════════════════════════════════════════
    for k, color in tier_colors.items():
        pw_perpt = np.array([
            float(piecewise_log_lr(k, float(p))) / k for p in priors
        ])
        ax_perpt.plot(priors, pw_perpt, color=color, lw=2.5, ls="--",
                      label=f"Piecewise {tier_labels[k]}")

    # Tavtigian: log(C*)/8 per point for all tiers (single flat level per prior)
    tav_perpt = np.log(C_arr) / 8.0
    ax_perpt.plot(priors, tav_perpt, color="#d62728", lw=2, ls="-",
                  label="Tavtigian (same for all tiers)")

    ax_perpt.axvline(0.10, color="#888", lw=1, ls=":", alpha=0.6)
    _apply_xscale(ax_perpt, log_scale)
    ax_perpt.set_xlabel("Prior", fontsize=10)
    ax_perpt.set_ylabel("log(LR+) per point  [log_lr(k) / k]", fontsize=10)
    ax_perpt.set_title(
        "Q1: LR+ value per point, by tier\n"
        "Tavtigian: one flat value for all tiers (exact 1:2:4:8)\n"
        "Piecewise: splits — VS (upper segment) ≠ Su/M/S (middle segment)",
        fontsize=9,
    )
    ax_perpt.legend(fontsize=8, loc="best")
    ax_perpt.grid(True, alpha=0.25)

    # ══════════════════════════════════════════════════════════════════════════
    # (0,1) Doubling ratios  LR+(2k) / LR+(k)²
    # ══════════════════════════════════════════════════════════════════════════
    doubling_pairs = [(1, "Su+Su → M  (1+1=2)"),
                      (2, "M+M → S    (2+2=4)"),
                      (4, "S+S → VS   (4+4=8)")]

    for k, label in doubling_pairs:
        color = tier_colors[k]
        pw_ratio = np.array([
            math.exp(float(piecewise_log_lr(2*k, float(p)))
                     - 2*float(piecewise_log_lr(k, float(p))))
            for p in priors
        ])
        ax_doubling.plot(priors, pw_ratio, color=color, lw=2.5, ls="--",
                         label=f"Piecewise: {label}")

    ax_doubling.axhline(1.0, color="#d62728", lw=2.0, ls="-",
                        label="Tavtigian ≡ 1.0 (exact doubling)")
    ax_doubling.axvline(0.10, color="#888", lw=1, ls=":", alpha=0.6)

    _apply_xscale(ax_doubling, log_scale)
    ax_doubling.set_xlabel("Prior", fontsize=10)
    ax_doubling.set_ylabel("LR+(2k)  /  LR+(k)²", fontsize=10)
    ax_doubling.set_title(
        "Q1: Does tier doubling hold?  LR+(2k) / LR+(k)²\n"
        "= 1 means 2×Supporting = Moderate, 2×Strong = VS, etc.\n"
        "S+S→VS crosses the T=6 knot — largest deviation",
        fontsize=9,
    )
    ax_doubling.legend(fontsize=8, loc="best")
    ax_doubling.grid(True, alpha=0.25)

    # ══════════════════════════════════════════════════════════════════════════
    # Helper: combination path error
    #   signed_err = piecewise_posterior(T_total, p)
    #                  − Bayes(∏ LR+(k_i, p), p)
    #   For Tavtigian, LR-product = point-sum (additive), so its
    #   "combination error" is 0, but we show boundary drift instead.
    # ══════════════════════════════════════════════════════════════════════════
    def _pw_path_err(codes, p):
        T_total = sum(codes)
        lrs = [math.exp(float(piecewise_log_lr(k, float(p)))) for k in codes]
        lr_prod = 1.0
        for lr in lrs:
            lr_prod *= lr
        return float(piecewise_posterior(T_total, float(p))) - float(
            bayes_posterior_from_lr(lr_prod, float(p))
        )

    def _tav_boundary_drift(T_bnd, target, p, C):
        return float(bayes_posterior_from_lr(C ** (T_bnd / 8.0), float(p))) - target

    # Combo line styles by path
    _path_ls = ["-", "--", "-.", ":"]
    _path_colors = [STRENGTH_COLOR[8], STRENGTH_COLOR[6],
                    STRENGTH_COLOR[4], STRENGTH_COLOR[2]]

    # ══════════════════════════════════════════════════════════════════════════
    # (1,0)  LP boundary paths (T=6)
    # ══════════════════════════════════════════════════════════════════════════
    lp_paths = [
        ([4, 2],       "S + M"),
        ([2, 2, 2],    "M + M + M"),
        ([4, 1, 1],    "S + Su + Su"),
        ([2, 2, 1, 1], "M + M + Su + Su"),
    ]
    for (codes, label), ls, col in zip(lp_paths, _path_ls, _path_colors):
        errs = np.array([_pw_path_err(codes, p) for p in priors])
        ax_lp6.plot(priors, errs, color=col, lw=2, ls=ls,
                    label=f"PW {label}")

    # Tavtigian boundary drift at T=6 (comparison reference)
    tav_lp_drift = np.array([
        _tav_boundary_drift(6, 0.90, p, C_arr[i])
        for i, p in enumerate(priors)
    ])
    ax_lp6.plot(priors, tav_lp_drift, color="#888", lw=2.0, ls="-",
                alpha=0.7, label="Tav LP boundary drift (post−0.90)")

    ax_lp6.axhline(0, color="black", lw=0.7, ls=":")
    ax_lp6.axvline(0.10, color="#888", lw=1, ls=":", alpha=0.6)
    _apply_xscale(ax_lp6, log_scale)
    ax_lp6.set_xlabel("Prior", fontsize=10)
    ax_lp6.set_ylabel("post(T_total) − Bayes(LR₁·LR₂·…)", fontsize=10)
    ax_lp6.set_title(
        "Q2: LP boundary (T=6) combination paths\n"
        "Piecewise: point-sum = 0.90 exactly · LR-product may differ\n"
        "Grey = Tavtigian boundary drift (its failure mode)",
        fontsize=9,
    )
    ax_lp6.legend(fontsize=7, loc="best")
    ax_lp6.grid(True, alpha=0.25)

    # ══════════════════════════════════════════════════════════════════════════
    # (1,1)  P boundary (T=10), lower-segment codes only — crosses T=6 knot
    # ══════════════════════════════════════════════════════════════════════════
    p10_lower_paths = [
        ([4, 4, 2],       "S + S + M"),
        ([4, 4, 1, 1],    "S + S + Su + Su"),
        ([4, 2, 2, 2],    "S + M + M + M"),
        ([2, 2, 2, 2, 2], "5× M"),
    ]
    for (codes, label), ls, col in zip(p10_lower_paths, _path_ls, _path_colors):
        errs = np.array([_pw_path_err(codes, p) for p in priors])
        ax_p10_lower.plot(priors, errs, color=col, lw=2, ls=ls,
                          label=f"PW {label}")

    tav_p_drift = np.array([
        _tav_boundary_drift(10, 0.99, p, C_arr[i])
        for i, p in enumerate(priors)
    ])
    ax_p10_lower.plot(priors, tav_p_drift, color="#888", lw=2.0, ls="-",
                      alpha=0.7, label="Tav P boundary drift (post−0.99)")

    ax_p10_lower.axhline(0, color="black", lw=0.7, ls=":")
    ax_p10_lower.axvline(0.10, color="#888", lw=1, ls=":", alpha=0.6)
    _apply_xscale(ax_p10_lower, log_scale)
    ax_p10_lower.set_xlabel("Prior", fontsize=10)
    ax_p10_lower.set_ylabel("post(T_total) − Bayes(LR₁·LR₂·…)", fontsize=10)
    ax_p10_lower.set_title(
        "Q2: P boundary (T=10), lower-segment paths only (no VS)\n"
        "All codes ≤4, total crosses T=6 knot on way to T=10\n"
        "Grey = Tavtigian boundary drift at T=10",
        fontsize=9,
    )
    ax_p10_lower.legend(fontsize=7, loc="best")
    ax_p10_lower.grid(True, alpha=0.25)

    # ══════════════════════════════════════════════════════════════════════════
    # (2,0)  P boundary (T=10), paths involving VS (code 8)
    # ══════════════════════════════════════════════════════════════════════════
    p10_vs_paths = [
        ([8, 2],    "VS + M"),
        ([8, 1, 1], "VS + Su + Su"),
    ]
    for (codes, label), ls, col in zip(p10_vs_paths, _path_ls, _path_colors):
        errs = np.array([_pw_path_err(codes, p) for p in priors])
        ax_p10_vs.plot(priors, errs, color=col, lw=2, ls=ls,
                       label=f"PW {label}")

    ax_p10_vs.plot(priors, tav_p_drift, color="#888", lw=2.0, ls="-",
                   alpha=0.7, label="Tav P boundary drift (post−0.99)")
    ax_p10_vs.axhline(0, color="black", lw=0.7, ls=":")
    ax_p10_vs.axvline(0.10, color="#888", lw=1, ls=":", alpha=0.6)
    _apply_xscale(ax_p10_vs, log_scale)
    ax_p10_vs.set_xlabel("Prior", fontsize=10)
    ax_p10_vs.set_ylabel("post(T_total) − Bayes(LR₁·LR₂·…)", fontsize=10)
    ax_p10_vs.set_title(
        "Q2: P boundary (T=10), paths involving VS (code 8)\n"
        "VS in upper segment, companion codes in middle segment\n"
        "Grey = Tavtigian boundary drift at T=10",
        fontsize=9,
    )
    ax_p10_vs.legend(fontsize=7, loc="best")
    ax_p10_vs.grid(True, alpha=0.25)

    # ══════════════════════════════════════════════════════════════════════════
    # (2,1)  Summary: max |piecewise combo error| vs |Tavtigian boundary drift|
    # Show bar chart at two reference priors (0.10 and 0.25) for each path set
    # ══════════════════════════════════════════════════════════════════════════
    all_path_sets = [
        ("LP (T=6)",    lp_paths,       6,  0.90),
        ("P lower",     p10_lower_paths, 10, 0.99),
        ("P + VS",      p10_vs_paths,   10, 0.99),
    ]
    ref_priors_sum = [0.10, 0.25]
    x_groups = np.arange(len(all_path_sets))
    bar_w = 0.22
    grp_off = np.array([-1, 0, 1]) * bar_w
    sum_cols = ["#444", "#888", "#1f77b4"]
    sum_labels = ["p=0.10 Tav drift", "p=0.25 Tav drift",
                  "p=0.25 PW max combo err"]

    for j, (p_ref, col, lbl) in enumerate(zip(
            ref_priors_sum + [0.25], sum_cols, sum_labels)):
        i_ref = int(np.abs(priors - p_ref).argmin())
        heights = []
        for _, paths, T_bnd, tgt in all_path_sets:
            if j < 2:
                # Tavtigian boundary drift
                heights.append(abs(_tav_boundary_drift(
                    T_bnd, tgt, priors[i_ref], C_arr[i_ref]
                )))
            else:
                # Max piecewise combo error across paths at p=0.25
                heights.append(max(
                    abs(_pw_path_err(codes, priors[i_ref]))
                    for codes, _ in paths
                ))
        ax_summary.bar(x_groups + grp_off[j], heights, width=bar_w,
                       color=col, alpha=0.85, label=lbl)

    ax_summary.set_xticks(x_groups)
    ax_summary.set_xticklabels([ps[0] for ps in all_path_sets], fontsize=9)
    ax_summary.set_ylabel("|posterior error|", fontsize=10)
    ax_summary.set_title(
        "Summary: Tavtigian boundary drift vs piecewise combo error\n"
        "at p=0.10 and p=0.25 · Piecewise combo error measured as max across paths",
        fontsize=9,
    )
    ax_summary.legend(fontsize=8)
    ax_summary.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        "Evidence combination: tier ratios (Q1) and boundary paths (Q2)\n"
        "Piecewise: exact LP/P posteriors via point-sum, small LR-product mismatch  ·  "
        "Tavtigian: exact LR ratios, drifting LP/P posteriors",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    return fig, axes


# ── 10. Slope geometry: why the kinks are unavoidable ────────────────────────

def plot_slope_geometry(
    demo_priors: tuple = (0.05, 0.10, 0.25),
    figsize: tuple = (12, 9),
    log_scale: bool = False,
    tav_df: Optional[pd.DataFrame] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Show the log(LR+) vs T geometry: Tavtigian straight line vs Piecewise kinks.

    Visualizes the core additivity tradeoff:
    - Tavtigian: straight (additive) but misses Bayesian target dots at non-canonical priors
    - Piecewise: kinked line that hits all four target dots exactly at every prior

    Layout (2 rows × 3 columns, one column per prior)
    -----------------------------------------------
    Row 0 — log(LR+) vs T curves:
        Target dots show where Bayesian posterior equals 0.01, 0.10, 0.90, 0.99.
        Tavtigian = straight through all T, hits dots only at p=0.10.
        Piecewise = kinked, hits dots exactly at every prior.
    Row 1 — implied posterior vs T:
        Shows classification regions (P/LP/LB/B) and where each method lands.
    """
    import math
    from assay_calibration.fit_utils.bayesian_thresholds import (
        piecewise_log_lr, piecewise_posterior,
        bayes_posterior_from_lr, bayes_lr_for_posterior,
    )
    from assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant

    T_grid = np.linspace(-10, 14, 500)
    _ACMG_T = {"B": -7, "LB": -1, "LP": 6, "P": 10}
    _ACMG_targets = {"B": 0.01, "LB": 0.10, "LP": 0.90, "P": 0.99}

    fig, axes = plt.subplots(2, len(demo_priors), figsize=figsize)

    for col, p in enumerate(demo_priors):
        # Resolve C*
        if tav_df is not None and "C_star" in tav_df.columns:
            C = float(tav_df.loc[
                (tav_df["prior"] - float(p)).abs().idxmin(), "C_star"
            ])
        else:
            C = float(get_tavtigian_constant(float(p)))

        ax_log = axes[0, col]
        ax_post = axes[1, col]

        # ── Row 0: log(LR+) vs T ──────────────────────────────────────────
        # Bayesian anchor log(LR+) values (what we NEED at each knot T)
        target_log_lrs = {
            name: math.log(bayes_lr_for_posterior(q, float(p)))
            for name, q in _ACMG_targets.items()
        }

        # Tavtigian
        tav_slope = math.log(C) / 8.0
        ax_log.plot(T_grid, tav_slope * T_grid,
                    color="#d62728", lw=2.5, ls="-", label=f"Tavtigian (C={int(C)})")

        # Piecewise (ACMG knots)
        pw_log_lr = piecewise_log_lr(T_grid, float(p))
        ax_log.plot(T_grid, pw_log_lr,
                    color="#1f77b4", lw=2.5, ls="--", label="Piecewise α (ACMG)")

        # Bayesian target log(LR+) at each ACMG T: mark with dots
        for name, T_bnd in _ACMG_T.items():
            tgt_ll = target_log_lrs[name]
            color  = _BND_COLOR[name]
            ax_log.plot(T_bnd, tgt_ll, "o", color=color,
                        ms=9, zorder=5)
            ax_log.axhline(tgt_ll, color=color, lw=0.7, ls=":", alpha=0.4)
            ax_log.axvline(T_bnd,  color=color, lw=0.7, ls=":", alpha=0.4)

        ax_log.axhline(0, color="black", lw=0.5)
        ax_log.axvline(0, color="black", lw=0.5)
        ax_log.set_xlim(T_grid[0], T_grid[-1])
        ax_log.set_xlabel("Total evidence points T", fontsize=10)
        ax_log.set_ylabel("log(LR+)", fontsize=10)
        ax_log.set_title(
            f"log(LR+) vs T   at prior p = {p:.2f}\n"
            "Straight line = additive · Dots = Bayesian targets",
            fontsize=9,
        )
        ax_log.grid(True, alpha=0.25)
        if col == 0:
            ax_log.legend(fontsize=8, loc="upper left")
        else:
            # Annotate what changes between panels
            if tav_slope * _ACMG_T["LP"] < target_log_lrs["LP"] - 0.05:
                ax_log.annotate(
                    "Tavtigian misses\nLP target ↓",
                    xy=(6, tav_slope * 6),
                    xytext=(3, tav_slope * 6 - 1.5),
                    arrowprops=dict(arrowstyle="->", color="#d62728"),
                    color="#d62728", fontsize=7,
                )

        # ── Row 1: posterior vs T ─────────────────────────────────────────
        tav_posts = np.array([
            float(bayes_posterior_from_lr(math.exp(tav_slope * t), float(p)))
            for t in T_grid
        ])
        pw_posts  = piecewise_posterior(T_grid, float(p))

        ax_post.plot(T_grid, tav_posts,
                     color="#d62728", lw=2.5, ls="-",  label="Tavtigian")
        ax_post.plot(T_grid, pw_posts,
                     color="#1f77b4", lw=2.5, ls="--", label="Piecewise α")

        # Classification region shading
        ax_post.axhspan(0.99, 1.01, color=_BND_COLOR["P"],  alpha=0.12)
        ax_post.axhspan(0.90, 0.99, color=_BND_COLOR["LP"], alpha=0.08)
        ax_post.axhspan(0.10, 0.90, color="white",          alpha=0.0)
        ax_post.axhspan(0.01, 0.10, color=_BND_COLOR["LB"], alpha=0.08)
        ax_post.axhspan(-0.02, 0.01, color=_BND_COLOR["B"], alpha=0.12)

        # ACMG boundary horizontal lines
        for name, target in _ACMG_targets.items():
            ax_post.axhline(target, color=_BND_COLOR[name],
                            lw=1.2, ls=":", alpha=0.6)
            # ACMG T boundary vertical lines
            ax_post.axvline(_ACMG_T[name], color=_BND_COLOR[name],
                            lw=0.8, ls=":", alpha=0.5)

        # Region labels (only on leftmost panel)
        if col == 0:
            for name, target, label_text in [
                ("P",  0.995, "P"),  ("LP", 0.945, "LP"),
                ("LB", 0.055, "LB"), ("B",  0.005, "B"),
            ]:
                ax_post.text(T_grid[0] + 0.5, target, label_text,
                             color=_BND_COLOR[name], fontsize=9,
                             fontweight="bold", va="center")

        ax_post.set_xlim(T_grid[0], T_grid[-1])
        ax_post.set_ylim(-0.03, 1.05)
        ax_post.set_xlabel("Total evidence points T", fontsize=10)
        ax_post.set_ylabel("Posterior P(pathogenic)", fontsize=10)
        ax_post.set_title(
            f"Classification posterior  p = {p:.2f}\n"
            "Piecewise hits ACMG boundaries exactly · "
            "Tavtigian misses at p ≠ 0.10",
            fontsize=9,
        )
        ax_post.grid(True, alpha=0.20)
        if col == 0:
            ax_post.legend(fontsize=8, loc="center left")

    fig.suptitle(
        "Why the dilemma is unavoidable: a straight log(LR+) line "
        "cannot pass through all four Bayesian targets simultaneously\n"
        "Piecewise connects the dots exactly (kinks); "
        "Tavtigian stays straight (misses the dots at priors ≠ 0.10)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    return fig, axes
