"""
Small shared helpers used by every figure-producing module in analysis/ —
consolidates what used to be separately duplicated `_save`/`_pretty` helpers
in confusion.py, evidence.py, scatter.py, calibration_plots.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def sample_matches(df: pd.DataFrame, category: str) -> pd.Series:
    """Boolean mask: does each variant's `sample` column include `category`?

    `sample` is pipe-separated multi-label (a variant can genuinely belong to
    more than one category at once, e.g. both "Synonymous" and "population" —
    see variant_evidence.py::_build_standard_table) — never compare it with
    `==` directly, since that only ever matches single-label rows and
    silently drops every multi-label variant from both sides of the
    comparison.
    """
    return df["sample"].str.split("|").apply(lambda cats: category in cats)


def effective_points(df_sub: pd.DataFrame, use_oob: bool, label: str = "", context: str = "") -> pd.Series:
    """Return the best available evidence points per variant.

    When use_oob=True: use oob_points when not NaN, fall back to standard_points.
    When use_oob=False: always use standard_points.

    Logs how many variants fell back to in-bag scoring (no OOB match) when
    use_oob=True, so silent OOB->in-bag substitution is never invisible —
    used identically by confusion.py (confusion matrices) and evidence.py
    (evidence-distribution arrays) so both stay consistent.
    """
    if not use_oob or "oob_points" not in df_sub.columns:
        return df_sub["standard_points"]
    has_oob = df_sub["oob_points"].notna()
    n_fallback = int((~has_oob).sum())
    if n_fallback:
        tag = f"[{label}] " if label else ""
        print(f"  {tag}{context}: {n_fallback}/{len(df_sub)} variant(s) used in-bag "
              f"scoring (no OOB match)")
    result = df_sub["standard_points"].copy()
    result[has_oob] = df_sub.loc[has_oob, "oob_points"]
    return result


def is_notebook() -> bool:
    """True when running inside a Jupyter/IPython kernel rather than a plain script."""
    return "ipykernel" in sys.modules or "IPython" in sys.modules


def pretty_method(method: str) -> str:
    return {
        "tavtigian": "Tavtigian",
        "acmg_bayes": "ACMG-Bayes",
        # Legacy tags from runs predating the ACMG-Bayes consolidation.
        "piecewise": "Piecewise [legacy]",
        "continuous": "Continuous [legacy]",
        "strict_additive": "Strict Additive [legacy]",
        "default": "ExCALIBR",
    }.get(method, method.replace("_", " ").title())


def save_and_show(fig, path: Path, dpi: int = 300):
    """Save `fig` to `path`, then display it in the current cell's output when
    running as a notebook (before closing it — matplotlib_inline's own
    post-cell display hook only picks up figures that are still open, so
    saving-then-closing immediately, as this used to do, meant nothing ever
    rendered in Jupyter even though the PNG was written correctly to disk)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")
    if is_notebook():
        plt.show()
    else:
        plt.close(fig)
