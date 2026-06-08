#!/usr/bin/env python
"""
Generate all figures for report.tex.

Run from the repo root:
    python tavtigian_sims/report/generate_figures.py

Saves PDFs to tavtigian_sims/report/figures/.
Also prints the numerical summary tables used in the report body.
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tavtigian_sims import prior_grid
from tavtigian_sims.compare import (
    run_all_methods,
    plot_three_way_comparison,
    plot_boundary_posteriors_three_way,
    plot_additivity_experiment,
    plot_combined_error_comparison,
    plot_additivity_dilemma,
    plot_slope_geometry,
    plot_combination_paths,
)

FIG_DIR = os.path.join(_HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {name}")

# ── Run simulations ──────────────────────────────────────────────────────────

print("Running simulations...")
priors = prior_grid("paper")   # 300 priors from 0.01 to 0.45
t, pw, pw_add, lpa, cont = run_all_methods(priors, n_jobs=-1)
print(f"  Done. {len(priors)} priors, {len(t)} Tavtigian rows.")

# ── Figure 1: LR+ thresholds ─────────────────────────────────────────────────

print("\nFigure 1: LR+ thresholds...")
fig, _ = plot_three_way_comparison(
    t, pw, pw_add_df=pw_add, cont_df=cont,
    methods=("tavtigian", "piecewise"),
    codes="key",
    log_scale=False,
    figsize=(10, 5),
)
savefig(fig, "lr_thresholds.pdf")

# ── Figure 2: Boundary posteriors ────────────────────────────────────────────

print("Figure 2: Boundary posteriors...")
fig, _ = plot_boundary_posteriors_three_way(
    t, pw, pw_add_df=pw_add,
    methods=("tavtigian", "piecewise"),
    log_scale=False,
    figsize=(11, 5),
)
savefig(fig, "boundary_posteriors.pdf")

# ── Figure 3: Slope geometry ─────────────────────────────────────────────────

print("Figure 3: Slope geometry...")
fig, _ = plot_slope_geometry(
    demo_priors=(0.05, 0.10, 0.25),
    tav_df=t,
    figsize=(13, 8),
)
savefig(fig, "slope_geometry.pdf")

# ── Figure 4: Additivity by combination type ─────────────────────────────────

print("Figure 4: Additivity dilemma (by type)...")
fig, _ = plot_additivity_dilemma(
    priors=priors,
    tav_df=t,
    demo_priors=(0.05, 0.10, 0.25),
    figsize=(18, 10),
)
savefig(fig, "additivity_by_type.pdf")

# ── Figure 5: Combined error comparison (MAE + log-odds) ─────────────────────

print("Figure 5: Combined error comparison...")
fig, _ = plot_combined_error_comparison(
    priors=priors,
    tav_df=t,
    figsize=(14, 12),
)
savefig(fig, "error_comparison.pdf")

# ── Figure 6: Combination paths ──────────────────────────────────────────────

print("Figure 6: Combination paths...")
fig, _ = plot_combination_paths(
    priors=priors,
    tav_df=t,
    figsize=(14, 12),
)
savefig(fig, "combination_paths.pdf")

# ── Numerical tables ─────────────────────────────────────────────────────────

print("\n" + "="*70)
print("TABLE 1: Tavtigian boundary posteriors at key priors")
print("  (target: LP=0.90, P=0.99, LB=0.10, B=0.01)")
print("="*70)

from assay_calibration.fit_utils.bayesian_thresholds import bayes_posterior_from_lr

key_priors = [0.01, 0.05, 0.10, 0.25, 0.45]
boundaries = [("LP", 6, 0.90), ("P", 10, 0.99), ("LB", -1, 0.10), ("B", -7, 0.01)]

# Get C* at each key prior from tav_df
def get_C(p):
    idx = int((t["prior"] - float(p)).abs().idxmin())
    return float(t.loc[idx, "C_star"])

rows = []
for p_key in key_priors:
    C = get_C(p_key)
    row = {"prior": p_key, "C*": int(C)}
    for name, T_bnd, target in boundaries:
        post = float(bayes_posterior_from_lr(C ** (T_bnd / 8.0), p_key))
        row[f"Tav {name}"] = round(post, 4)
        row[f"PW {name}"] = round(target, 4)   # piecewise is exact by construction
    rows.append(row)

df_table = pd.DataFrame(rows).set_index("prior")
print(df_table.to_string())

from assay_calibration.fit_utils.bayesian_thresholds import piecewise_log_lr, piecewise_posterior
from scipy.stats import wilcoxon

_path_tiers = [1, 2, 4, 8]
_ben_tiers  = [1, 2, 4, 8]
path_combos   = [(kA, kB) for kA in range(1, 9) for kB in range(kA, 9)]
benign_combos = [(-kA, -kB) for kA in _ben_tiers for kB in _ben_tiers if kB >= kA]
mixed_combos  = [(kA, -kB) for kA in _path_tiers for kB in _ben_tiers]
all_combos    = path_combos + benign_combos + mixed_combos

def segment(t_val):
    """Piecewise segment: 0=B–LB (T≤−1), 1=LB–LP (−1<T≤6), 2=LP–P (T>6)."""
    return 0 if t_val <= -1 else (1 if t_val <= 6 else 2)

def combo_type(kA, kB):
    k_tot = kA + kB
    if kA > 0 and kB > 0:
        return "within_path" if segment(kA) == segment(kB) == segment(k_tot) else "cross_path"
    elif kA < 0 and kB < 0:
        if segment(kA) == segment(kB) == segment(k_tot):
            return "within_ben" if k_tot >= -7 else "cross_ben"
        return "cross_ben"
    return "mixed"

def logit(x, eps=1e-9):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))

def log_odds_err(pred, true):
    return abs(logit(pred) - logit(true))

# ── Compute per-combo errors at all key priors ────────────────────────────────

COMBO_TYPES = ["within_path", "cross_path", "within_ben", "cross_ben", "mixed"]
SUMMARY_PRIORS = [0.01, 0.05, 0.10, 0.25, 0.45]

# errs[prior][ct] = {"pw": [...], "tav": [...], "delta": [...]}  (log-odds errors)
errs_by_prior = {}
for p in SUMMARY_PRIORS:
    C_p = get_C(p)
    errs_by_prior[p] = {ct: {"pw": [], "tav": [], "delta": []} for ct in COMBO_TYPES}
    for kA, kB in all_combos:
        k_tot = kA + kB
        lr_A  = math.exp(float(piecewise_log_lr(kA, p)))
        lr_B  = math.exp(float(piecewise_log_lr(kB, p)))
        ref   = float(bayes_posterior_from_lr(lr_A * lr_B, p))
        pw_p  = float(piecewise_posterior(k_tot, p))
        tav_p = float(bayes_posterior_from_lr(C_p ** (k_tot / 8.0), p))
        pw_e  = log_odds_err(pw_p,  ref)
        tav_e = log_odds_err(tav_p, ref)
        ct = combo_type(kA, kB)
        errs_by_prior[p][ct]["pw"].append(pw_e)
        errs_by_prior[p][ct]["tav"].append(tav_e)
        errs_by_prior[p][ct]["delta"].append(pw_e - tav_e)  # >0 means Tav better

# ── TABLE 2: p=0.10 snapshot ──────────────────────────────────────────────────

print("\n" + "="*80)
print("TABLE 2: Combination errors at p=0.10 by type (median |log-odds error|)")
print("="*80)
print(f"{'Type':<20} {'N':>4} {'PW med':>8} {'PW max':>8} {'Tav med':>8} {'Tav max':>8}")
print("-"*60)
for ct in COMBO_TYPES:
    d = errs_by_prior[0.10][ct]
    pw  = np.array(d["pw"])
    tav = np.array(d["tav"])
    print(f"{ct:<20} {len(pw):>4} {np.median(pw):>8.3f} {np.max(pw):>8.3f}"
          f" {np.median(tav):>8.3f} {np.max(tav):>8.3f}")

# ── TABLE 2b: Summary across priors with Wilcoxon ────────────────────────────

print("\n" + "="*110)
print("TABLE 2b: Median log-odds combination errors by type × prior  (** = winner;  W-stat and p from Wilcoxon signed-rank)")
print("="*110)

wilcoxon_rows = []   # collected for LaTeX table
for ct in COMBO_TYPES:
    print(f"\n{ct}")
    print(f"  {'prior':>6}  {'PW med':>8} {'Tav med':>8}  {'winner':>6}  {'W-stat':>8} {'p-value':>10}")
    wrows = {"ct": ct}
    for p in SUMMARY_PRIORS:
        d = errs_by_prior[p][ct]
        pw_arr  = np.array(d["pw"])
        tav_arr = np.array(d["tav"])
        delta   = np.array(d["delta"])  # pw - tav; >0 means Tav better
        pw_med  = np.median(pw_arr)
        tav_med = np.median(tav_arr)
        winner  = "PW" if pw_med < tav_med else "Tav"
        # Wilcoxon signed-rank on (pw - tav) differences; null = median diff = 0
        if len(delta) >= 2 and not np.all(delta == 0):
            try:
                stat, pval = wilcoxon(delta, alternative="two-sided")
            except Exception:
                stat, pval = np.nan, np.nan
        else:
            stat, pval = np.nan, np.nan
        mark = "**" if pw_med < tav_med else "  "
        print(f"  {p:>6.2f}  {mark}{pw_med:>7.3f}  {tav_med:>7.3f}  {winner:>6}  {stat:>8.1f} {pval:>10.4f}")
        wrows[p] = {"pw": pw_med, "tav": tav_med, "stat": stat, "pval": pval}
    wilcoxon_rows.append(wrows)

print("\nAll figures saved to:", FIG_DIR)

# ── Generate supplementary LaTeX tables ──────────────────────────────────────

print("\nGenerating supplementary combination error tables...")

# ACMG tier label lookup (pathogenic)
def path_label(k):
    labels = {1: "Su", 2: "M", 3: "Su+M?", 4: "S", 5: "Su+S?", 6: "M+S?",
              7: "Su+M+S?", 8: "VS"}
    return labels.get(abs(k), str(k))

def ben_label(k):
    labels = {1: "Su$_B$", 2: "M$_B$†", 4: "S$_B$", 8: "VS$_B$†"}
    return labels.get(abs(k), str(k))

def code_label(k):
    if k > 0:
        return path_label(k)
    else:
        return ben_label(-k)

KEY_PRIORS = [0.01, 0.10, 0.25, 0.45]

def make_combo_rows(combos, priors_list, C_dict):
    """Return list of dicts for LaTeX table rows (log-odds errors)."""
    rows = []
    for kA, kB in combos:
        k_tot = kA + kB
        row = {"kA": kA, "kB": kB, "k_tot": k_tot,
               "label_A": code_label(kA), "label_B": code_label(kB)}
        for p in priors_list:
            C = C_dict[p]
            lr_A  = math.exp(float(piecewise_log_lr(kA, float(p))))
            lr_B  = math.exp(float(piecewise_log_lr(kB, float(p))))
            ref   = float(bayes_posterior_from_lr(lr_A * lr_B, float(p)))
            pw_p  = float(piecewise_posterior(k_tot, float(p)))
            tav_p = float(bayes_posterior_from_lr(C ** (k_tot / 8.0), float(p)))
            row[f"pw_{p}"]  = log_odds_err(pw_p,  ref)
            row[f"tav_{p}"] = log_odds_err(tav_p, ref)
        rows.append(row)
    return rows

# Build C* lookup
C_lookup = {p: float(t.loc[(t["prior"] - float(p)).abs().idxmin(), "C_star"])
            for p in KEY_PRIORS}
print(f"  C* values: {C_lookup}")

# Build combo sets
PATH_TIERS = [1, 2, 4, 8]
BEN_TIERS  = [1, 2, 4, 8]   # -1=-Su_B, -2=-M_B†, -4=-S_B, -8=-VS_B†

path_combos_full  = [(kA, kB) for kA in range(1,9) for kB in range(kA, 9)]
benign_combos_full = [(-kA, -kB) for kA in BEN_TIERS for kB in BEN_TIERS if kB >= kA]
mixed_combos_full  = [(kA, -kB) for kA in PATH_TIERS for kB in BEN_TIERS]

def fmt_err(v, bold=False):
    """Format a log-odds error to 3 sig figs."""
    if v < 0.001:
        s = r"$<$0.001"
    elif v < 1.0:
        s = f"{v:.3f}"
    else:
        s = f"{v:.2f}"
    return r"\textbf{" + s + r"}" if bold else s

def write_latex_table(rows, caption, label, out_lines):
    # Column spec: label cols + separator + (rr + hspace) per prior
    col_spec = r"llr"
    for i in range(len(KEY_PRIORS)):
        col_spec += r" @{\hspace{0.8em}} rr"

    prior_headers = " & ".join([f"\\multicolumn{{2}}{{c}}{{$p={p}$}}" for p in KEY_PRIORS])
    sub_headers   = " & ".join([r"\textbf{PW} & \textbf{Tav}"] * len(KEY_PRIORS))

    out_lines.append(r"\begin{longtable}{" + col_spec + r"}")
    out_lines.append(r"\caption{" + caption + r"} \label{" + label + r"} \\")
    out_lines.append(r"\toprule")
    out_lines.append(r"$k_A$ & $k_B$ & $T$ & " + prior_headers + r" \\")
    out_lines.append(r" & & & " + sub_headers + r" \\")
    out_lines.append(r"\midrule \endfirsthead")
    out_lines.append(r"\toprule")
    out_lines.append(r"$k_A$ & $k_B$ & $T$ & " + prior_headers + r" \\")
    out_lines.append(r" & & & " + sub_headers + r" \\")
    out_lines.append(r"\midrule \endhead")
    out_lines.append(r"\midrule \multicolumn{" + str(3+2*len(KEY_PRIORS)) +
                     r"}{r}{\small continued\ldots} \\ \endfoot")
    out_lines.append(r"\bottomrule \endlastfoot")

    for row in rows:
        cells = [row["label_A"], row["label_B"], str(row["k_tot"])]
        for p in KEY_PRIORS:
            pw_v  = row[f"pw_{p}"]
            tav_v = row[f"tav_{p}"]
            pw_wins = pw_v <= tav_v
            cells.append(fmt_err(pw_v,  bold=pw_wins))
            cells.append(fmt_err(tav_v, bold=not pw_wins))
        out_lines.append(" & ".join(cells) + r" \\")

    out_lines.append(r"\end{longtable}")
    out_lines.append("")


# Generate the three tables
path_rows   = make_combo_rows(path_combos_full,  KEY_PRIORS, C_lookup)
benign_rows = make_combo_rows(benign_combos_full, KEY_PRIORS, C_lookup)
mixed_rows  = make_combo_rows(mixed_combos_full,  KEY_PRIORS, C_lookup)

lines = [
    r"% This file is auto-generated by generate_figures.py",
    r"",
]

write_latex_table(
    path_rows,
    caption=(r"Combination errors for all pathogenic code pairs. "
             r"$k_A \leq k_B$, both in $\{1,2,3,4,5,6,7,8\}$. "
             r"Codes 3,5,6,7 are intermediate strengths not corresponding to "
             r"named ACMG tiers; codes 1,2,4,8 = Su, M, S, VS. "
             r"PW = piecewise additivity error; Tav = Tavtigian posterior error. "
             r"Both measured vs.\ the Bayesian posterior from the piecewise per-code "
             r"LR$^+$ product."),
    label="tab:s_path",
    out_lines=lines,
)

write_latex_table(
    benign_rows,
    caption=(r"Combination errors for benign code pairs. "
             r"Standard ACMG benign tiers are Su$_B$ ($-1$ pt) and S$_B$ ($-4$ pt); "
             r"M$_B$ ($-2$ pt, $\dagger$) and VS$_B$ ($-8$ pt, $\dagger$) are "
             r"non-standard hypothetical extensions included for completeness. "
             r"Negative $T$ totals indicate net benign evidence."),
    label="tab:s_ben",
    out_lines=lines,
)

write_latex_table(
    mixed_rows,
    caption=(r"Combination errors for mixed (pathogenic $+$ benign) code pairs. "
             r"$k_A > 0$ (pathogenic tier), $k_B < 0$ (benign tier). "
             r"Tavtigian has zero internal combination error for mixed evidence "
             r"by construction ($C^{a+b} = C^a \cdot C^b$), but its per-code "
             r"LR$^+$ thresholds are miscalibrated at non-canonical priors, "
             r"so the posterior errors are still large."),
    label="tab:s_mixed",
    out_lines=lines,
)

supp_path = os.path.join(_HERE, "supplement_tables.tex")
with open(supp_path, "w") as f:
    f.write("\n".join(lines))
print(f"  Saved supplement_tables.tex ({len(path_rows)+len(benign_rows)+len(mixed_rows)} combo rows)")
