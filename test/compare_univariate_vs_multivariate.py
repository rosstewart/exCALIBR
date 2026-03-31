"""
Compare univariate aggregated vs multivariate joint calibration.
All plots show DISCRETE point assignments (integers -8..+8).
Uses 5th percentile LR+ for pathogenic, 95th for benign, median prior.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mvn, norm
from collections import Counter

# ──────────────────────────────────────
# Discrete colormap for points -8..+8
# ──────────────────────────────────────
_blues = plt.cm.Blues_r(np.linspace(0.1, 0.9, 8))
_reds = plt.cm.Reds(np.linspace(0.1, 0.9, 8))
_white = np.array([[0.95, 0.95, 0.95, 1.0]])
_all_colors = np.vstack([_blues, _white, _reds])  # 17 colors: -8..-1, 0, +1..+8
DISCRETE_CMAP = LinearSegmentedColormap.from_list("discrete_evidence", _all_colors, N=17)
DISCRETE_BOUNDS = np.arange(-8.5, 9.5, 1)  # 18 boundaries for 17 bins
DISCRETE_NORM = BoundaryNorm(DISCRETE_BOUNDS, DISCRETE_CMAP.N)

SAMPLE_COLORS = {'P/LP': '#CA7682', 'B/LB': '#1D7AAB', 'Pop': '#A0A0A0', 'Syn': '#6BAA75'}
MISCLASS_MARKERS = {'FP': 'X', 'FN': 'X'}  # wrong-direction marker



# ──────────────────────────────────────
# Alignment
# ──────────────────────────────────────

def align_variants(ms, uv_aspa, mv_result):
    """Build per-variant comparison table."""
    uv_aspa = uv_aspa.copy()
    uv_aspa["POS"] = pd.to_numeric(uv_aspa["POS"], errors="coerce")
    uv_aspa["#CHROM"] = uv_aspa["#CHROM"].astype(str)
    uv_aspa["REF"] = uv_aspa["REF"].astype(str)
    uv_aspa["ALT"] = uv_aspa["ALT"].astype(str)

    uv_lookup = {}
    for _, row in uv_aspa.iterrows():
        key = (str(row["#CHROM"]), float(row["POS"]), str(row["REF"]), str(row["ALT"]))
        uv_lookup[key] = {
            "uv_points": row.get("points", np.nan),
            "uv_evidence": row.get("evidence", ""),
            "uv_dataset": row.get("dataset_highest_evidence", ""),
            "protein_variant": row.get("protein_variant", ""),
        }

    sa = ms._sample_assignments
    scores = ms.scores
    mv_pts = mv_result["points"] if mv_result is not None else np.zeros(len(scores))

    records = []
    for i, vk in enumerate(ms._variants_kept):
        gene, chrom, pos, ref, alt = vk
        chrom_str = f"chr{chrom}" if not str(chrom).startswith("chr") else str(chrom)
        uv_key = (chrom_str, float(pos), str(ref), str(alt))
        uv_info = uv_lookup.get(uv_key, {})

        complete_both = not np.isnan(scores[i]).any()
        has_dim0 = not np.isnan(scores[i, 0]) if scores.shape[1] > 0 else False
        has_dim1 = not np.isnan(scores[i, 1]) if scores.shape[1] > 1 else False

        records.append({
            "gene": gene, "chrom": chrom_str, "pos": pos, "ref": ref, "alt": alt,
            "is_plp": bool(sa[i, 0]),
            "is_blb": bool(sa[i, 1]),
            "is_pop": bool(sa[i, 2]),
            "is_syn": bool(sa[i, 3]),
            "uv_points": uv_info.get("uv_points", np.nan),
            "uv_evidence": uv_info.get("uv_evidence", ""),
            "protein_variant": uv_info.get("protein_variant", ""),
            "mv_points": mv_pts[i],
            "score_dim0": scores[i, 0], "score_dim1": scores[i, 1],
            "complete_both": complete_both,
            "has_dim0": has_dim0, "has_dim1": has_dim1,
        })
    return pd.DataFrame(records)


# ──────────────────────────────────────
# Metrics
# ──────────────────────────────────────

def compute_metrics(labels, points, name=""):
    mask = np.isfinite(points)
    labels, points = np.asarray(labels)[mask], np.asarray(points, float)[mask]
    N = len(labels)
    if N == 0:
        return {"name": name, "N": 0}

    TP = (labels & (points > 0)).sum()
    FN = (labels & (points < 0)).sum()
    FP = (~labels & (points > 0)).sum()
    TN = (~labels & (points < 0)).sum()
    unc_p = (labels & (points == 0)).sum()
    unc_b = (~labels & (points == 0)).sum()
    det = TP + TN + FP + FN

    return {
        "name": name, "N": int(N),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN),
        "uncertain": int(unc_p + unc_b),
        "coverage": det / N if N else 0,
        "accuracy": (TP + TN) / det if det else 0,
        "sensitivity": TP / (TP + FN) if (TP + FN) else 0,
        "specificity": TN / (TN + FP) if (TN + FP) else 0,
        "MCC": ((TP * TN) - (FP * FN)) / np.sqrt(float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))) if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) > 0 else 0,
        "DOR": (TP * TN) / (FP * FN) if FP > 0 and FN > 0 else float("inf"),
    }


# ──────────────────────────────────────
# Scatter helper with misclassification marks
# ──────────────────────────────────────

def _scatter_with_misclass(ax, df, x_col, y_col, pts_col, title, xlabel, ylabel,
                            show_missing=True, annotate_misclass=True):
    """
    Scatter colored by discrete points.
    Misclassified variants: P/LP with negative points or B/LB with positive points
    get a red/blue X marker on top. Missing-dim variants shown as ticks on edges.
    """
    eval_mask = df["is_plp"] | df["is_blb"]
    pts = df[pts_col].fillna(0).values

    # Complete cases
    complete = df["complete_both"]
    mask = eval_mask & complete
    if mask.any():
        sc = ax.scatter(
            df.loc[mask, x_col], df.loc[mask, y_col],
            c=pts[mask], cmap=DISCRETE_CMAP, norm=DISCRETE_NORM,
            s=14, alpha=0.6, edgecolors="none", zorder=2,
        )

    # Missing dim0 only (show on bottom edge)
    if show_missing:
        m0 = eval_mask & df["has_dim0"] & ~df["has_dim1"]
        if m0.any():
            y_edge = ax.get_ylim()[0] + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]) if ax.get_ylim()[1] != ax.get_ylim()[0] else 0
            ax.scatter(
                df.loc[m0, x_col],
                np.full(m0.sum(), y_edge if y_edge != 0 else df[y_col].min()),
                c=pts[m0], cmap=DISCRETE_CMAP, norm=DISCRETE_NORM,
                s=10, alpha=0.4, marker="|", zorder=2,
            )
        # Missing dim1 only (show on left edge)
        m1 = eval_mask & ~df["has_dim0"] & df["has_dim1"]
        if m1.any():
            x_edge = ax.get_xlim()[0] + 0.03 * (ax.get_xlim()[1] - ax.get_xlim()[0]) if ax.get_xlim()[1] != ax.get_xlim()[0] else 0
            ax.scatter(
                np.full(m1.sum(), x_edge if x_edge != 0 else df[x_col].min()),
                df.loc[m1, y_col],
                c=pts[m1], cmap=DISCRETE_CMAP, norm=DISCRETE_NORM,
                s=10, alpha=0.4, marker="_", zorder=2,
            )

    # Misclassifications: big X markers
    if annotate_misclass:
        # FP: B/LB assigned positive
        fp = eval_mask & df["is_blb"] & (pts > 0) & complete
        if fp.any():
            ax.scatter(df.loc[fp, x_col], df.loc[fp, y_col],
                       marker="X", s=60, c="red", edgecolors="black",
                       linewidths=0.5, zorder=4, label=f"FP ({fp.sum()})")
        # FN: P/LP assigned negative
        fn = eval_mask & df["is_plp"] & (pts < 0) & complete
        if fn.any():
            ax.scatter(df.loc[fn, x_col], df.loc[fn, y_col],
                       marker="X", s=60, c="blue", edgecolors="black",
                       linewidths=0.5, zorder=4, label=f"FN ({fn.sum()})")

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(lw=0.2, alpha=0.3)
    if annotate_misclass and (fp.any() or fn.any()):
        ax.legend(fontsize=7, loc="upper left", framealpha=0.7)


def _marginal_discrete_strip(ax, df, score_col, pts_col, has_col, is_plp_col,
                               is_blb_col, title, xlabel):
    """
    Marginal 1D strip plot: x = score, y = discrete points.
    Color by point value, mark misclassifications.
    """
    eval_mask = df[is_plp_col] | df[is_blb_col]
    has = df[has_col]
    m = eval_mask & has
    if not m.any():
        ax.set_title(title)
        return

    scores = df.loc[m, score_col].values
    pts = df.loc[m, pts_col].fillna(0).values.astype(int)
    is_plp = df.loc[m, is_plp_col].values
    is_blb = df.loc[m, is_blp_col].values if is_blp_col in df else df.loc[m, is_blb_col].values

    jitter = np.random.normal(0, 0.2, len(pts))
    ax.scatter(scores, pts + jitter, c=pts, cmap=DISCRETE_CMAP, norm=DISCRETE_NORM,
               s=8, alpha=0.5, edgecolors="none", zorder=2)

    # Misclassifications
    fp = is_blb & (pts > 0)
    fn = is_plp & (pts < 0)
    if fp.any():
        ax.scatter(scores[fp], pts[fp] + jitter[fp], marker="X", s=40,
                   c="red", edgecolors="black", linewidths=0.3, zorder=4)
    if fn.any():
        ax.scatter(scores[fn], pts[fn] + jitter[fn], marker="X", s=40,
                   c="blue", edgecolors="black", linewidths=0.3, zorder=4)

    ax.axhline(0, color="gray", lw=0.8, ls="-", alpha=0.4)
    ax.set_yticks(range(-8, 9, 2))
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Points", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(lw=0.2, alpha=0.3)


# ──────────────────────────────────────
# Main comparison plot
# ──────────────────────────────────────

def plot_comparison(comp_df, metrics_uv, metrics_mv, ms, config_name=""):
    """
    Full comparison figure. All discrete.

    Row 0: [UV vs MV scatter] [Point distributions] [Metrics bars] [Metrics table]
    Row 1: [UV 2D points]     [MV 2D points]       [UV marginal dim0] [MV marginal dim0]
    Row 2: [UV marginal dim1 strip] [MV marginal dim1 strip] [changed variants] [legend/summary]
    """
    dataset_names = getattr(ms, 'dataset_names', ['Dim 0', 'Dim 1'])
    eval_mask = comp_df["is_plp"] | comp_df["is_blb"]
    eval_df = comp_df[eval_mask].copy()

    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    # ─── (0,0): UV vs MV point scatter ───
    ax = fig.add_subplot(gs[0, 0])
    uv = eval_df["uv_points"].fillna(0).values
    mv = eval_df["mv_points"].values
    jitter_u = np.random.normal(0, 0.15, len(uv))
    jitter_m = np.random.normal(0, 0.15, len(mv))
    colors = np.where(eval_df["is_plp"], SAMPLE_COLORS["P/LP"], SAMPLE_COLORS["B/LB"])
    # Mark complete vs partial
    complete = eval_df["complete_both"].values
    sizes = np.where(complete, 15, 8)
    alphas = np.where(complete, 0.6, 0.3)
    for c_val, s_val, a_val, label in [
        (True, 15, 0.6, "Complete"), (False, 8, 0.3, "Partial")
    ]:
        m = complete == c_val
        if m.any():
            ax.scatter(uv[m] + jitter_u[m], mv[m] + jitter_m[m],
                       c=colors[m], s=s_val, alpha=a_val, edgecolors="none",
                       marker="o" if c_val else "^", label=label)
    ax.plot([-9, 9], [-9, 9], "k--", lw=0.8, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.3)
    agree = (np.sign(uv) == np.sign(mv)).mean()
    exact = (uv.astype(int) == mv.astype(int)).mean()
    ax.text(0.03, 0.97, f"Sign agree: {agree*100:.1f}%\nExact match: {exact*100:.1f}%",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("UV Aggregated Points")
    ax.set_ylabel("MV Joint Points")
    ax.set_title("UV vs MV Points\n(red=P/LP, blue=B/LB, △=partial)", fontsize=10, fontweight="bold")
    ax.set_xlim(-9, 9); ax.set_ylim(-9, 9)
    ax.set_xticks(range(-8, 9, 2)); ax.set_yticks(range(-8, 9, 2))
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(lw=0.2, alpha=0.3)

    # ─── (0,1): Point distribution ───
    ax = fig.add_subplot(gs[0, 1])
    all_pts = list(range(-8, 9))
    uv_c = Counter(uv.astype(int))
    mv_c = Counter(mv.astype(int))
    x_pos = np.arange(len(all_pts))
    w = 0.35
    ax.bar(x_pos - w/2, [uv_c.get(p, 0) for p in all_pts], w,
           label="UV Agg", color="#A0A0A0", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.bar(x_pos + w/2, [mv_c.get(p, 0) for p in all_pts], w,
           label="MV Joint", color="#3498db", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x_pos[::2])
    ax.set_xticklabels([str(p) for p in all_pts[::2]], fontsize=7)
    ax.set_xlabel("Points"); ax.set_ylabel("Count")
    ax.set_title("Point Distribution (P/LP + B/LB)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(lw=0.2, alpha=0.3, axis="y")

    # ─── (0,2): Metrics bars ───
    ax = fig.add_subplot(gs[0, 2])
    mnames = ["accuracy", "sensitivity", "specificity", "MCC", "coverage"]
    uv_v = [metrics_uv.get(m, 0) for m in mnames]
    mv_v = [metrics_mv.get(m, 0) for m in mnames]
    x = np.arange(len(mnames))
    ax.bar(x - 0.18, uv_v, 0.32, label="UV Agg", color="#A0A0A0")
    ax.bar(x + 0.18, mv_v, 0.32, label="MV Joint", color="#3498db")
    for i, (u, m_) in enumerate(zip(uv_v, mv_v)):
        ax.text(i - 0.18, u + 0.02, f"{u:.2f}", ha="center", fontsize=6)
        ax.text(i + 0.18, m_ + 0.02, f"{m_:.2f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(mnames, fontsize=7, rotation=15)
    ax.set_ylabel("Value"); ax.set_title("Metrics", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7); ax.set_ylim(0, 1.15)
    ax.grid(lw=0.2, alpha=0.3, axis="y")

    # ─── (0,3): Table ───
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    rows = []
    for m in ["N", "TP", "TN", "FP", "FN", "uncertain", "coverage",
              "accuracy", "sensitivity", "specificity", "MCC", "DOR"]:
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) and v != float("inf") else str(v)
        rows.append([m, fmt(metrics_uv.get(m, "")), fmt(metrics_mv.get(m, ""))])
    tbl = ax.table(cellText=rows, colLabels=["Metric", "UV", "MV"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.2)
    ax.set_title("Full Metrics", fontsize=10, fontweight="bold", pad=15)

    # ─── (1,0): UV 2D discrete points ───
    ax = fig.add_subplot(gs[1, 0])
    _scatter_with_misclass(ax, comp_df, "score_dim0", "score_dim1", "uv_points",
                            f"UV Aggregated Points", dataset_names[0], dataset_names[1])

    # ─── (1,1): MV 2D discrete points ───
    ax = fig.add_subplot(gs[1, 1])
    _scatter_with_misclass(ax, comp_df, "score_dim0", "score_dim1", "mv_points",
                            f"MV Joint Points ({config_name})", dataset_names[0], dataset_names[1])

    # ─── (1,2): UV marginal dim0 strip ───
    ax = fig.add_subplot(gs[1, 2])
    eval_m = comp_df["is_plp"] | comp_df["is_blb"]
    has0 = comp_df["has_dim0"] & ~comp_df["has_dim1"] & eval_m
    if has0.any():
        sub = comp_df[has0]
        scores0 = sub["score_dim0"].values
        pts_uv = sub["uv_points"].fillna(0).values.astype(int)
        jit = np.random.normal(0, 0.2, len(pts_uv))
        col = np.where(sub["complete_both"], 1.0, 0.4)
        ax.scatter(scores0, pts_uv + jit, c=pts_uv, cmap=DISCRETE_CMAP, norm=DISCRETE_NORM,
                   s=8, alpha=col * 0.6, edgecolors="none")
        # Misclass
        fp = sub["is_blb"].values & (pts_uv > 0)
        fn = sub["is_plp"].values & (pts_uv < 0)
        if fp.any():
            ax.scatter(scores0[fp], pts_uv[fp] + jit[fp], marker="X", s=35,
                       c="red", edgecolors="black", linewidths=0.3, zorder=4)
        if fn.any():
            ax.scatter(scores0[fn], pts_uv[fn] + jit[fn], marker="X", s=35,
                       c="blue", edgecolors="black", linewidths=0.3, zorder=4)
    ax.axhline(0, color="gray", lw=0.6, alpha=0.4)
    ax.set_yticks(range(-8, 9, 2))
    ax.set_xlabel(dataset_names[0], fontsize=8); ax.set_ylabel("UV Points", fontsize=8)
    ax.set_title(f"UV Marginal {dataset_names[0]}", fontsize=9, fontweight="bold")
    ax.grid(lw=0.2, alpha=0.3)

    # ─── (1,3): MV marginal dim0 strip ───
    ax = fig.add_subplot(gs[1, 3])
    if has0.any():
        sub = comp_df[has0]
        pts_mv = sub["mv_points"].fillna(0).values.astype(int)
        col = np.where(sub["complete_both"], 1.0, 0.4)
        ax.scatter(scores0, pts_mv + jit, c=pts_mv, cmap=DISCRETE_CMAP, norm=DISCRETE_NORM,
                   s=8, alpha=col * 0.6, edgecolors="none")
        fp = sub["is_blb"].values & (pts_mv > 0)
        fn = sub["is_plp"].values & (pts_mv < 0)
        if fp.any():
            ax.scatter(scores0[fp], pts_mv[fp] + jit[fp], marker="X", s=35,
                       c="red", edgecolors="black", linewidths=0.3, zorder=4)
        if fn.any():
            ax.scatter(scores0[fn], pts_mv[fn] + jit[fn], marker="X", s=35,
                       c="blue", edgecolors="black", linewidths=0.3, zorder=4)
    ax.axhline(0, color="gray", lw=0.6, alpha=0.4)
    ax.set_yticks(range(-8, 9, 2))
    ax.set_xlabel(dataset_names[0], fontsize=8); ax.set_ylabel("MV Points", fontsize=8)
    ax.set_title(f"MV Marginal {dataset_names[0]}", fontsize=9, fontweight="bold")
    ax.grid(lw=0.2, alpha=0.3)

    # ─── (2,0): UV marginal dim1 strip ───
    ax = fig.add_subplot(gs[2, 0])
    has1 = ~comp_df["has_dim0"] & comp_df["has_dim1"] & eval_m
    if has1.any():
        sub1 = comp_df[has1]
        scores1 = sub1["score_dim1"].values
        pts_uv1 = sub1["uv_points"].fillna(0).values.astype(int)
        jit1 = np.random.normal(0, 0.2, len(pts_uv1))
        col1 = np.where(sub1["complete_both"], 1.0, 0.4)
        ax.scatter(scores1, pts_uv1 + jit1, c=pts_uv1, cmap=DISCRETE_CMAP, norm=DISCRETE_NORM,
                   s=8, alpha=col1 * 0.6, edgecolors="none")
        fp = sub1["is_blb"].values & (pts_uv1 > 0)
        fn = sub1["is_plp"].values & (pts_uv1 < 0)
        if fp.any():
            ax.scatter(scores1[fp], pts_uv1[fp] + jit1[fp], marker="X", s=35,
                       c="red", edgecolors="black", linewidths=0.3, zorder=4)
        if fn.any():
            ax.scatter(scores1[fn], pts_uv1[fn] + jit1[fn], marker="X", s=35,
                       c="blue", edgecolors="black", linewidths=0.3, zorder=4)
    ax.axhline(0, color="gray", lw=0.6, alpha=0.4)
    ax.set_yticks(range(-8, 9, 2))
    ax.set_xlabel(dataset_names[1], fontsize=8); ax.set_ylabel("UV Points", fontsize=8)
    ax.set_title(f"UV Marginal {dataset_names[1]}", fontsize=9, fontweight="bold")
    ax.grid(lw=0.2, alpha=0.3)

    # ─── (2,1): MV marginal dim1 strip ───
    ax = fig.add_subplot(gs[2, 1])
    if has1.any():
        pts_mv1 = sub1["mv_points"].fillna(0).values.astype(int)
        ax.scatter(scores1, pts_mv1 + jit1, c=pts_mv1, cmap=DISCRETE_CMAP, norm=DISCRETE_NORM,
                   s=8, alpha=col1 * 0.6, edgecolors="none")
        fp = sub1["is_blb"].values & (pts_mv1 > 0)
        fn = sub1["is_plp"].values & (pts_mv1 < 0)
        if fp.any():
            ax.scatter(scores1[fp], pts_mv1[fp] + jit1[fp], marker="X", s=35,
                       c="red", edgecolors="black", linewidths=0.3, zorder=4)
        if fn.any():
            ax.scatter(scores1[fn], pts_mv1[fn] + jit1[fn], marker="X", s=35,
                       c="blue", edgecolors="black", linewidths=0.3, zorder=4)
    ax.axhline(0, color="gray", lw=0.6, alpha=0.4)
    ax.set_yticks(range(-8, 9, 2))
    ax.set_xlabel(dataset_names[1], fontsize=8); ax.set_ylabel("MV Points", fontsize=8)
    ax.set_title(f"MV Marginal {dataset_names[1]}", fontsize=9, fontweight="bold")
    ax.grid(lw=0.2, alpha=0.3)

    # ─── (2,2): Changed direction variants ───
    ax = fig.add_subplot(gs[2, 2])
    complete_eval = eval_m & comp_df["complete_both"]
    if complete_eval.any():
        ce = comp_df[complete_eval]
        uv_s = np.sign(ce["uv_points"].fillna(0).values)
        mv_s = np.sign(ce["mv_points"].values)
        same = uv_s == mv_s
        ax.scatter(ce.loc[same.astype(bool), "score_dim0"],
                   ce.loc[same.astype(bool), "score_dim1"],
                   c="gray", s=5, alpha=0.15, edgecolors="none")
        changed = ~same
        if changed.any():
            ch_colors = np.where(ce["is_plp"].values[changed], SAMPLE_COLORS["P/LP"], SAMPLE_COLORS["B/LB"])
            ax.scatter(ce["score_dim0"].values[changed], ce["score_dim1"].values[changed],
                       c=ch_colors, s=25, alpha=0.8, edgecolors="black", linewidths=0.3)
    ax.set_xlabel(dataset_names[0], fontsize=8); ax.set_ylabel(dataset_names[1], fontsize=8)
    n_changed = (~same).sum() if complete_eval.any() else 0
    ax.set_title(f"Direction Changes ({n_changed})\n(P/LP=red, B/LB=blue)", fontsize=9, fontweight="bold")
    ax.grid(lw=0.2, alpha=0.3)

    # ─── (2,3): Legend / summary ───
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    legend_elements = [
        Patch(facecolor=SAMPLE_COLORS["P/LP"], label="P/LP variant"),
        Patch(facecolor=SAMPLE_COLORS["B/LB"], label="B/LB variant"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="red",
               markersize=10, markeredgecolor="black", label="FP (B/LB → path evidence)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="blue",
               markersize=10, markeredgecolor="black", label="FN (P/LP → ben evidence)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=8, label="Partial (missing 1 dim)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, label="Complete (both dims)"),
    ]
    ax.legend(handles=legend_elements, loc="center", fontsize=9, frameon=True)

    n_total = len(eval_df)
    n_complete = eval_df["complete_both"].sum()
    n_partial = n_total - n_complete
    ax.text(0.5, 0.15, f"Eval variants: {n_total}\n"
            f"Complete: {n_complete} ({100*n_complete/n_total:.0f}%)\n"
            f"Partial: {n_partial} ({100*n_partial/n_total:.0f}%)",
            ha="center", va="center", transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    gene = getattr(ms, 'scoreset_name', comp_df["gene"].iloc[0] if len(comp_df) else "")
    fig.suptitle(f"{gene}: UV Aggregated vs MV Joint ({config_name})",
                 fontsize=13, fontweight="bold", y=1.01)
    return fig



def plot_lr_curves(ms, mv_result, config_name="", figsize=None,
                   smoothing_window=None):
    """
    Diagnostic plot showing LR+ percentile curves along each score dimension,
    split by complete (joint 2D) vs partial (marginal-only) observations.

    Layout (2 rows × 3 cols):
      Row 0: [Dim0 LR+ curves]  [Dim1 LR+ curves]  [Density fraction]
      Row 1: [Dim0 partial vs complete detail] [Dim1 partial vs complete detail] [Summary table]

    For each dimension:
      - Smoothed 5th and 95th percentile LR+ as a function of score
      - Separate curves for complete (2D) and partial (marginal-only) variants
      - Pathogenic/benign threshold lines
      - Rug plot showing P/LP and B/LB variant locations

    Parameters
    ----------
    ms : MultiScoreset
    mv_result : dict
        From analysis.results[config]. Must contain lr_5th, lr_95th, lr_median,
        tau_p_log, tau_b_log, and optionally path_fraction.
    config_name : str
    figsize : tuple or None
    smoothing_window : int or None
        Window size for uniform_filter1d smoothing. If None, auto-computed.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import uniform_filter1d
    import numpy as np
    from collections import Counter

    scores = ms.scores
    sa = ms._sample_assignments
    N, K = scores.shape
    dataset_names = getattr(ms, 'dataset_names', [f'Dim {d}' for d in range(K)])

    lr_5th = mv_result['lr_5th']
    lr_95th = mv_result['lr_95th']
    lr_median = mv_result['lr_median']
    tau_p_log = mv_result['tau_p_log']
    tau_b_log = mv_result['tau_b_log']
    points = mv_result['points']
    path_fraction = mv_result.get('path_fraction')
    path_pctile = mv_result.get('path_percentile', 5)
    ben_pctile = mv_result.get('ben_percentile', 95)

    # Also grab raw (uncapped) if available
    lr_5th_raw = mv_result.get('lr_5th_raw')
    lr_95th_raw = mv_result.get('lr_95th_raw')
    has_raw = lr_5th_raw is not None

    # Observation masks
    has_dim = [~np.isnan(scores[:, d]) for d in range(K)]
    complete = np.all(~np.isnan(scores), axis=1)
    path_mask = sa[:, 0]
    ben_mask = sa[:, 1]
    if sa.shape[1] > 3:
        neg_mask = sa[:, 1] | sa[:, 3]  # B/LB + synonymous
    else:
        neg_mask = sa[:, 1]

    if figsize is None:
        figsize = (20, 14)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    def _smooth(x, y, win=None):
        """Sort by x and smooth y."""
        order = np.argsort(x)
        xs, ys = x[order], y[order].astype(float)
        if win is None:
            win = max(len(xs) // 40, 5)
        ys_smooth = uniform_filter1d(ys, win)
        return xs, ys_smooth

    # ═══════════════════════════════════════════════════
    # Row 0: LR+ curves per dimension (all variants)
    # ═══════════════════════════════════════════════════
    for dim in range(min(K, 2)):
        ax = fig.add_subplot(gs[0, dim])

        valid = has_dim[dim] & np.isfinite(lr_median)
        # Partial only: has this dim but missing the other
        other_dim_row0 = 1 - dim if K == 2 else None
        if other_dim_row0 is not None:
            is_partial = valid & ~has_dim[other_dim_row0]
        else:
            is_partial = valid & ~complete

        # ── Partial cases (marginal-only) ──
        if is_partial.any():
            _, p5_p = _smooth(scores[is_partial, dim], lr_5th[is_partial], smoothing_window)
            xs_p, p95_p = _smooth(scores[is_partial, dim], lr_95th[is_partial], smoothing_window)

            ax.plot(xs_p, p5_p, color='#d7191c', lw=1.5,
                    label=f'{path_pctile}th pctile — path (n={is_partial.sum()})')
            ax.plot(xs_p, p95_p, color='#2c7bb6', lw=1.5,
                    label=f'{ben_pctile}th pctile — ben')

        # ── Threshold lines ──
        ax.axhline(0, color='gray', lw=1, ls='-', alpha=0.5)
        for pv_idx, pv in enumerate([1, 4, max(mv_result.get('points', [8]).max(), 1)]):
            if pv - 1 < len(tau_p_log):
                ax.axhline(tau_p_log[pv - 1], color='red', ls='--', lw=0.5, alpha=0.4)
                ax.axhline(tau_b_log[pv - 1], color='blue', ls='--', lw=0.5, alpha=0.4)
                if pv_idx == 0:
                    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else scores[valid, dim].min(),
                            tau_p_log[pv - 1], f' +{pv}', fontsize=6, color='red', va='bottom')
                    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else scores[valid, dim].min(),
                            tau_b_log[pv - 1], f' -{pv}', fontsize=6, color='blue', va='top')

        # ── Rug plots for P/LP and B/LB ──
        ylim = ax.get_ylim()
        rug_y_top = ylim[1] - 0.02 * (ylim[1] - ylim[0])
        rug_y_bot = ylim[0] + 0.02 * (ylim[1] - ylim[0])

        plp_dim = path_mask & has_dim[dim]
        blb_dim = neg_mask & has_dim[dim]
        if plp_dim.any():
            ax.plot(scores[plp_dim, dim], np.full(plp_dim.sum(), rug_y_top),
                    '|', color='#CA7682', alpha=0.4, ms=4, mew=0.5)
        if blb_dim.any():
            ax.plot(scores[blb_dim, dim], np.full(blb_dim.sum(), rug_y_bot),
                    '|', color='#1D7AAB', alpha=0.4, ms=4, mew=0.5)

        # ── Y limits: ± the 8-point threshold ──
        ylim_bound = max(abs(tau_p_log[-1]), abs(tau_b_log[-1])) * 1.05
        ax.set_ylim(-ylim_bound, ylim_bound)

        ax.set_xlabel(dataset_names[dim], fontsize=9)
        ax.set_ylabel('log LR+', fontsize=9)
        ax.set_title(f'LR+ Curves — {dataset_names[dim]}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6, loc='best', framealpha=0.7)
        ax.grid(lw=0.2, alpha=0.3)

    # ═══════════════════════════════════════════════════
    # Row 0, Col 2: Density fraction (if available)
    # ═══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 2])
    if path_fraction is not None and K >= 2:
        c_mask = complete & (path_mask | neg_mask)
        if c_mask.any():
            sc = ax.scatter(scores[c_mask, 0], scores[c_mask, 1],
                            c=path_fraction[c_mask], cmap='RdYlBu_r',
                            vmin=0, vmax=0.5, s=10, alpha=0.6)
            plt.colorbar(sc, ax=ax, label='f_path / (f_path + f_ben)', shrink=0.8)

            # Mark P/LP in benign zone
            thresh = mv_result.get('density_fraction_threshold', 0.05)
            plp_benign = path_mask & complete & (path_fraction < thresh)
            if plp_benign.any():
                ax.scatter(scores[plp_benign, 0], scores[plp_benign, 1],
                           marker='x', s=40, c='red', zorder=4,
                           label=f'P/LP in benign zone ({plp_benign.sum()})')
                ax.legend(fontsize=7, framealpha=0.7)

            ax.set_xlabel(dataset_names[0], fontsize=8)
            ax.set_ylabel(dataset_names[1], fontsize=8)
            ax.set_title(f'Density Fraction\n(threshold={thresh})', fontsize=10, fontweight='bold')
            ax.grid(lw=0.2, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No density fraction\n(run with cap_by_density=True)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Density Fraction', fontsize=10, fontweight='bold')
        ax.axis('off')

    # ═══════════════════════════════════════════════════
    # Row 1: Detailed partial vs complete LR+ by class
    # ═══════════════════════════════════════════════════
    for dim in range(min(K, 2)):
        ax = fig.add_subplot(gs[1, dim])
        valid = has_dim[dim] & np.isfinite(lr_5th)

        # Only partial: has this dim but missing the other
        other_dim = 1 - dim if K == 2 else None
        if other_dim is not None:
            is_partial_this = valid & ~has_dim[other_dim]
        else:
            is_partial_this = valid & ~complete

        # P/LP variants: show their 5th percentile LR+ (pathogenic threshold)
        plp_sub = is_partial_this & path_mask
        if plp_sub.any():
            ax.scatter(scores[plp_sub, dim], lr_5th[plp_sub],
                       c='#CA7682', marker='^', s=18, alpha=0.7,
                       edgecolors='none', zorder=2,
                       label=f'P/LP {path_pctile}th (n={plp_sub.sum()})')
            # Mark FN (P/LP assigned benign)
            fn_sub = plp_sub & (points < 0)
            if fn_sub.any():
                ax.scatter(scores[fn_sub, dim], lr_5th[fn_sub],
                           marker='X', s=50, c='blue', edgecolors='black',
                           linewidths=0.4, zorder=5,
                           label=f'FN ({fn_sub.sum()})')

        # B/LB variants: show their 95th percentile LR+ (benign threshold)
        blb_sub = is_partial_this & neg_mask
        if blb_sub.any():
            ax.scatter(scores[blb_sub, dim], lr_95th[blb_sub],
                       c='#1D7AAB', marker='^', s=18, alpha=0.7,
                       edgecolors='none', zorder=2,
                       label=f'B/LB {ben_pctile}th (n={blb_sub.sum()})')
            # Mark FP (B/LB assigned pathogenic)
            fp_sub = blb_sub & (points > 0)
            if fp_sub.any():
                ax.scatter(scores[fp_sub, dim], lr_95th[fp_sub],
                           marker='X', s=50, c='red', edgecolors='black',
                           linewidths=0.4, zorder=5,
                           label=f'FP ({fp_sub.sum()})')

        # Thresholds
        ax.axhline(0, color='gray', lw=1, ls='-', alpha=0.5)
        for pv in [1, 4]:
            if pv - 1 < len(tau_p_log):
                ax.axhline(tau_p_log[pv - 1], color='red', ls='--', lw=0.5, alpha=0.4)
                ax.axhline(tau_b_log[pv - 1], color='blue', ls='--', lw=0.5, alpha=0.4)

        # Y limits: ± the 8-point threshold
        ylim_bound = max(abs(tau_p_log[-1]), abs(tau_b_log[-1])) * 1.05
        ax.set_ylim(-ylim_bound, ylim_bound)

        other_name = dataset_names[other_dim] if other_dim is not None else '?'
        ax.set_xlabel(dataset_names[dim], fontsize=9)
        ax.set_ylabel(f'log LR+ ({path_pctile}th / {ben_pctile}th pctile)', fontsize=9)
        ax.set_title(f'Partial Only — {dataset_names[dim]}\n'
                     f'(missing {other_name}, △=partial, X=misclass)',
                     fontsize=9, fontweight='bold')
        ax.legend(fontsize=5.5, loc='best', framealpha=0.7, ncol=2)
        ax.grid(lw=0.2, alpha=0.3)

    # ═══════════════════════════════════════════════════
    # Row 1, Col 2: Summary statistics table
    # ═══════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    # Compute per-subset metrics
    rows = []
    rows.append(['', 'Complete', 'Partial', 'All'])
    rows.append(['', '(joint 2D)', '(marginal)', ''])

    for label, mask in [('P/LP', path_mask), ('B/LB+Syn', neg_mask)]:
        for metric_name, metric_fn in [
            ('N', lambda m: m.sum()),
            ('Correct %', lambda m: (
                (points[m] > 0).mean() * 100 if label == 'P/LP'
                else (points[m] < 0).mean() * 100) if m.any() else 0),
            ('Wrong %', lambda m: (
                (points[m] < 0).mean() * 100 if label == 'P/LP'
                else (points[m] > 0).mean() * 100) if m.any() else 0),
            ('Uncertain %', lambda m: (points[m] == 0).mean() * 100 if m.any() else 0),
        ]:
            comp_m = mask & complete
            part_m = mask & ~complete
            all_m = mask
            rows.append([
                f'{label} {metric_name}',
                f'{metric_fn(comp_m):.1f}' if isinstance(metric_fn(comp_m), float) else str(metric_fn(comp_m)),
                f'{metric_fn(part_m):.1f}' if isinstance(metric_fn(part_m), float) else str(metric_fn(part_m)),
                f'{metric_fn(all_m):.1f}' if isinstance(metric_fn(all_m), float) else str(metric_fn(all_m)),
            ])
        rows.append(['', '', '', ''])  # spacer

    tbl = ax.table(cellText=rows, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.1, 1.3)

    # Style header rows
    for col in range(4):
        tbl[0, col].set_facecolor('#e6e6e6')
        tbl[0, col].set_text_props(fontweight='bold')
        tbl[1, col].set_facecolor('#f0f0f0')

    ax.set_title('Complete vs Partial Breakdown', fontsize=10, fontweight='bold', pad=15)

    gene = getattr(ms, 'scoreset_name', '')
    fig.suptitle(f'{gene}: LR+ Curve Diagnostic ({config_name})',
                 fontsize=13, fontweight='bold', y=1.01)

    return fig

