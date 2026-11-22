#!/usr/bin/env python
"""
LABEL-seq ClinVar VUS: independent (univariate ExCALIBR) evidence -> joint
(ExCALIBR-MV) evidence, as a strength-tier Sankey. For the LABEL-seq-specific
manuscript, not the ExCALIBR-MV paper (LABEL-seq is held back from that one).

Source data: the archived full-panel run (with LABEL-seq and CARD11),
`jobs_final_tau_100b_3f_101826/final/figures/vus_reclassification/
functional_true_clinvar_vus_points.csv`, written by
analysis/compute_clinvar_vus_points_by_panel.py. Its LABEL-seq rows carry, per
ClinVar VUS:
  - `our_points`: ExCALIBR-MV points (best-MCC config per gene, trust_global,
    5th-percentile path);
  - `indep_points`: the paper's independent baseline -- each assay calibrated
    alone with univariate ExCALIBR, combined by the non-conflicting rule, on
    the same rows as the joint model (report._uv_nonconflicting_points).
Nothing is re-fitted or re-scored here.

VUS definition. The archived CSV selected VUS by ClinVar 2026
(`clinvar_sig_2026 == "Uncertain significance"`, no star filter). The project
rule is ClinVar 2025, so the default figure keeps only rows that are ALSO
"Uncertain significance" in ClinVar 2025 (`--vus-release 2025`). Variants that
are VUS in 2025 but not in 2026 were never scored in the archive and cannot be
added without re-scoring. `--vus-release 2026` reproduces the archived set.

Controls (--sample PLP / BLB / BLB_nonsyn): P/LP or B/LB from the archived
strength/strength_points.csv (see load_controls).

Outputs (OUTPUT_DIR): labelseq_vus_uv_vs_mv_sankey_{release}.{pdf,png},
labelseq_vus_uv_vs_mv_flows_{release}.csv (tier x tier counts) and
labelseq_vus_uv_vs_mv_summary_{release}.csv (per-gene and pooled summary).

Usage
-----
    python analysis/plot_labelseq_uv_vs_mv_vus_sankey.py [--sample VUS|PLP|BLB|BLB_nonsyn] [--vus-release 2025|2026] [--slides]
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mv_analysis.sankey_plot_flat import STRENGTH_COLOR, plot_categorical_sankey

ARCHIVE_CSV = Path("/data/ross/assay_calibration/multivariate/jobs_final_tau_100b_3f_101826/"
                   "final/figures/vus_reclassification/functional_true_clinvar_vus_points.csv")
STRENGTH_CSV = Path("/data/ross/assay_calibration/multivariate/jobs_final_tau_100b_3f_101826/"
                    "final/figures/strength/strength_points.csv")
OUTPUT_DIR = Path("/data/ross/assay_calibration/labelseq/vus_uv_vs_mv")
LABEL_CACHE = OUTPUT_DIR / "labelseq_clinvar_labels.csv"

# ACMG strength tiers on the points scale (Supporting 1, Moderate 2-3,
# Strong 4-7, Very Strong 8; mirrored for benign), most pathogenic on top.
_TIERS = [
    ("Very Strong (+8)",  8,  8, STRENGTH_COLOR[8]),
    ("Strong (+4 to +7)", 4,  7, STRENGTH_COLOR[4]),
    ("Moderate (+2, +3)", 2,  3, STRENGTH_COLOR[2]),
    ("Supporting (+1)",   1,  1, STRENGTH_COLOR[1]),
    ("No evidence (0)",      0,  0, "#e0e0e0"),
    ("Supporting (-1)",  -1, -1, STRENGTH_COLOR[-1]),
    ("Moderate (-2, -3)", -3, -2, STRENGTH_COLOR[-2]),
    ("Strong (-4 to -7)", -7, -4, STRENGTH_COLOR[-4]),
    ("Very Strong (-8)", -8, -8, STRENGTH_COLOR[-8]),
]
_TIER_ORDER = [t[0] for t in _TIERS]
_TIER_COLOR = {t[0]: t[3] for t in _TIERS}


def to_tier(points: pd.Series) -> pd.Series:
    p = points.round().clip(-8, 8).astype(int)
    out = pd.Series(index=p.index, dtype=object)
    for name, lo, hi, _ in _TIERS:
        out[(p >= lo) & (p <= hi)] = name
    return out


def load_clinvar_labels() -> pd.DataFrame:
    """(gene, protein variant) -> is ClinVar VUS in 2025 / 2026, from the
    LABEL-seq source dataframe (slow to build, so cached). A protein variant
    has one row per SNV x assay, and SNVs can carry different ClinVar records,
    so a variant counts as VUS if ANY of its rows is "Uncertain significance"
    -- the same any-row rule the archive used to select 2026 VUS
    (compute_clinvar_vus_points_by_panel.labelseq_vus_keys)."""
    if LABEL_CACHE.exists():
        return pd.read_csv(LABEL_CACHE)
    from src.assay_calibration.multivariate_data.labelseq import build_labelseq_dataframe
    df = build_labelseq_dataframe()
    df = df.assign(gene=df["Gene"].str.lower(),
                   vus_2025=df["clinvar_sig_2025"] == "Uncertain significance",
                   vus_2026=df["clinvar_sig_2026"] == "Uncertain significance")
    lab = df.groupby(["gene", "variant"])[["vus_2025", "vus_2026"]].any().reset_index()
    lab.to_csv(LABEL_CACHE, index=False)
    return lab


def load_controls(sample: str, exclude_synonymous: bool = False) -> pd.DataFrame:
    """P/LP or B/LB controls from the archived strength_points.csv (functional
    panel, LABEL-seq rows). Same footing as the paper's strength figure: the
    canonical joint fit's points and the non-conflicting independent points,
    both on the evaluation rows (labels as used by those fits; in-sample for
    both methods, not out-of-bag)."""
    df = pd.read_csv(STRENGTH_CSV)
    df = df[(df["panel"] == "functional") & (df["gene_set"] == "labelseq") & (df["sample"] == sample)].copy()
    df = df.rename(columns={"joint_points": "our_points"})
    df["gene"] = df["gene"].str.lower()
    if exclude_synonymous:
        df = df[~df["variant"].str.contains("=", regex=False)].copy()
    df["uv_tier"] = to_tier(df["indep_points"])
    df["mv_tier"] = to_tier(df["our_points"])
    return df


def load_vus(release: str) -> pd.DataFrame:
    df = pd.read_csv(ARCHIVE_CSV)
    df = df[df["gene_set"] == "labelseq"].copy()
    if release == "2025":
        lab = load_clinvar_labels()
        df = df.merge(lab, left_on=["gene", "variant_key"], right_on=["gene", "variant"], how="left")
        assert df["vus_2026"].fillna(False).all(), "archive rows should all be 2026 VUS"
        df = df[df["vus_2025"].fillna(False).astype(bool)].copy()
    df["uv_tier"] = to_tier(df["indep_points"])
    df["mv_tier"] = to_tier(df["our_points"])
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def _one(g):
        uv, mv = g["indep_points"].round(), g["our_points"].round()
        return pd.Series({
            "n_variants": len(g),
            "uv_any_evidence": int((uv != 0).sum()),
            "mv_any_evidence": int((mv != 0).sum()),
            "uv_pathogenic": int((uv > 0).sum()),
            "mv_pathogenic": int((mv > 0).sum()),
            "uv_benign": int((uv < 0).sum()),
            "mv_benign": int((mv < 0).sum()),
            "uv_moderate_or_stronger": int((uv.abs() >= 2).sum()),
            "mv_moderate_or_stronger": int((mv.abs() >= 2).sum()),
            "gained_evidence": int(((uv == 0) & (mv != 0)).sum()),
            "lost_evidence": int(((uv != 0) & (mv == 0)).sum()),
            "direction_flip": int(((uv * mv) < 0).sum()),
            "same_tier": int((g["uv_tier"] == g["mv_tier"]).sum()),
        })
    per_gene = df.groupby("gene").apply(_one)
    per_gene.index = per_gene.index.str.upper()
    pooled = _one(df).to_frame("ALL").T
    return pd.concat([pooled, per_gene]).astype(int)


def make_figure(df: pd.DataFrame, stem: str, slides: bool = False):
    flow = (df.groupby(["uv_tier", "mv_tier"]).size().reset_index(name="count")
            .rename(columns={"uv_tier": "source", "mv_tier": "target"}))
    figsize = (11, 10) if slides else (8.5, 7.5)
    fig, ax = plt.subplots(figsize=figsize)
    plot_categorical_sankey(
        flow, _TIER_ORDER, _TIER_ORDER, _TIER_COLOR, _TIER_COLOR, ax=ax,
        source_fontsize=15 if slides else 9, target_fontsize=15 if slides else 9,
        show_target_counts=True, gap_frac=0.02 if slides else 0.012,
        min_frac=0.03 if slides else 0.015, gradient=True,
    )
    hdr = 18 if slides else 11
    ax.text(0.0, -0.035, "ExCALIBR", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=hdr, fontweight="bold")
    ax.text(1.0, -0.035, "ExCALIBR-MV", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=hdr, fontweight="bold")
    fig.tight_layout()
    out_stem = f"{stem}_slides" if slides else stem
    for ext in ("pdf", "png"):
        out = OUTPUT_DIR / f"{out_stem}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)
    return flow


def main(sample: str = "VUS", release: str = "2025", slides: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if sample == "VUS":
        df = load_vus(release)
        tag = f"vus_uv_vs_mv_{{}}_clinvar{release}"
        what = f"ClinVar {release} VUS"
    else:
        name = {"PLP": "P/LP", "BLB": "B/LB", "BLB_nonsyn": "B/LB"}[sample]
        df = load_controls(name, exclude_synonymous=sample == "BLB_nonsyn")
        tag = f"{sample.lower()}_uv_vs_mv_{{}}"
        what = f"{name} controls" + (" (non-synonymous)" if sample == "BLB_nonsyn" else "")
    print(f"{what}: {len(df):,} variants, {df['gene'].nunique()} genes")
    stem = f"labelseq_{tag.format('sankey')}"
    flow = make_figure(df, stem)
    if slides:
        make_figure(df, stem, slides=True)
    (flow.pivot_table(index="source", columns="target", values="count", fill_value=0)
     .reindex(index=_TIER_ORDER, columns=_TIER_ORDER, fill_value=0)
     .rename_axis(index="ExCALIBR", columns="ExCALIBR-MV")
     .to_csv(OUTPUT_DIR / f"labelseq_{tag.format('flows')}.csv"))
    summary = summarize(df)
    summary.to_csv(OUTPUT_DIR / f"labelseq_{tag.format('summary')}.csv")
    print(summary.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", choices=["VUS", "PLP", "BLB", "BLB_nonsyn"], default="VUS")
    ap.add_argument("--vus-release", choices=["2025", "2026"], default="2025",
                    help="VUS only; controls keep the labels the archived fits used.")
    ap.add_argument("--slides", action="store_true")
    args = ap.parse_args()
    main(args.sample, args.vus_release, args.slides)
