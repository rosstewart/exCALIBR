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
    plot_slope_geometry,
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
t, acmg = run_all_methods(priors, n_jobs=-1)
print(f"  Done. {len(priors)} priors, {len(t)} Tavtigian rows.")

# ── Figure 1: LR+ thresholds ─────────────────────────────────────────────────

print("\nFigure 1: LR+ thresholds...")
fig, _ = plot_three_way_comparison(
    t, acmg,
    methods=("tavtigian", "acmg_bayes"),
    codes="key",
    log_scale=False,
    figsize=(10, 5),
)
savefig(fig, "lr_thresholds.pdf")

# ── Figure 2: Boundary posteriors ────────────────────────────────────────────

print("Figure 2: Boundary posteriors...")
fig, _ = plot_boundary_posteriors_three_way(
    t, acmg,
    methods=("tavtigian", "acmg_bayes"),
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
        row[f"AB {name}"] = round(target, 4)   # ACMG-Bayes is exact by construction
    rows.append(row)

df_table = pd.DataFrame(rows).set_index("prior")
print(df_table.to_string())

print("\nAll figures saved to:", FIG_DIR)
