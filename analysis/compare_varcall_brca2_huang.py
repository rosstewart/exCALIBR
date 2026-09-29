#!/usr/bin/env python
"""
ExCALIBR (OOB) vs Huang et al.'s own VarCall evidence calls for BRCA2_Huang_2025_SGE.

BRCA2_Huang_2025_SGE carries no author functional annotations in the master
TSV (every auth_reported_func_class* / Interval N column is NaN), so it never
appears in analyze_pipeline_output.py's ExCALIBR-vs-author comparisons. Huang
et al.'s Table S3 does report their own evidence call per variant ("Functional
category", last column), so this compares them directly against the SAME
ClinVar P/LP and B/LB controls, [B/LB, P/LP] x [Normal, IR, Abnormal]:

  - ExCALIBR (OOB): the canonical calibration, on auth_reported_score;
  - ExCALIBR (in-bag, VarCall score): ExCALIBR re-run on the same four
    fitting samples but on VarCall's "Model based functional score" instead --
    produced by run_varcall_score_calibration_brca2_huang.py. In-bag, not OOB:
    that run is --preset light (20 bootstraps), so each variant's OOB call rests
    on a median of ~7 bootstraps and only 60.6% of them agree with its in-bag
    point. Included whenever that run's output exists; otherwise the comparison
    is two-way;
  - VarCall: direction from the category's leading letter ("P..." ->
    pathogenic, "B..." -> benign, "VUS" -> indeterminate).

Canonical ExCALIBR evidence direction comes from OOB points, falling back to
in-bag where a variant has no OOB score (logged). Outputs a side-by-side confusion figure,
a metrics table (LaTeX via latex_performance_table_multi) and a CSV that also
carries VUS coverage, which the LaTeX table has no row for.

Standalone on purpose -- not part of analyze_pipeline_output.py/.ipynb.

Usage
-----
python analysis/compare_varcall_brca2_huang.py
python analysis/compare_varcall_brca2_huang.py --figure-dir /tmp/varcall --method default
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from analysis import config
from analysis.confusion import (
    build_confusion_matrix, build_vus_coverage, make_confusion_figure, make_confusion_grid_figure,
)
from analysis.discovery import discover_outputs, load_all_variants
from analysis.manuscript_stats import compute_aggregate_metrics_multi, latex_performance_table_multi
from analysis.plot_common import sample_matches, save_latex_table
from analysis.run_varcall_score_calibration_brca2_huang import (
    DEFAULT_WORK_DIR, VARCALL_SCORE_DATASET,
)

DATASET = "BRCA2_Huang_2025_SGE"
DEFAULT_VARCALL = "/data/ross/assay_calibration/BRCA2_Huang/BRCA2_Huang_VarCall.csv"
VARCALL_CATEGORY_COL = "Functional category"
VARCALL_HGVS_C_COL = "Coding sequence change (c.)"
FILE_STEM = "excalibr_vs_varcall_brca2_huang"

# method key -> display label, in column/panel order
METHOD_LABELS = {
    "excalibr": "ExCALIBR (OOB)",
    "excalibr_varcall_score": "ExCALIBR (in-bag, VarCall score)",
    "varcall": "VarCall",
}
# OOB where there are enough bootstraps for it (canonical run: 1,000); in-bag
# for the light VarCall-score run -- see the module docstring.
USE_OOB = {"excalibr": True, "excalibr_varcall_score": False}


def load_varcall(path: str) -> pd.DataFrame:
    """Table S3 -> one row per variant with `hgvs_c_short`, `varcall_category`,
    `varcall_points` (-1 benign / 0 VUS / +1 pathogenic).

    Row 0 of the CSV is a title line, header names carry trailing spaces
    ("Functional category "), and the table ends in footnote rows whose
    category is empty -- those are dropped."""
    vc = pd.read_csv(path, skiprows=1)
    vc.columns = [c.strip() for c in vc.columns]
    vc = vc[vc[VARCALL_CATEGORY_COL].notna()].copy()

    category = vc[VARCALL_CATEGORY_COL].astype(str).str.strip()
    points = pd.Series(0, index=vc.index)
    points[category.str.startswith("P")] = 1
    points[category.str.startswith("B")] = -1
    unexpected = sorted(set(category[points == 0]) - {"VUS"})
    if unexpected:
        raise ValueError(f"unexpected VarCall categories (neither P*/B*/VUS): {unexpected}")

    out = pd.DataFrame({
        "hgvs_c_short": vc[VARCALL_HGVS_C_COL].astype(str).str.strip(),
        "varcall_category": category,
        "varcall_points": points.astype(int),
    })
    if out["hgvs_c_short"].duplicated().any():
        raise ValueError("VarCall has duplicate c. changes -- join key is not unique")
    return out


def load_excalibr(output_dir: Path, dataset: str, dataset_configs: dict, method, dataset_tsv=None):
    """One dataset's kept-sample-group + VUS variants (with oob_points) from
    load_all_variants, keyed by `hgvs_c_short`.

    variant_id is f"{urn}_{Gene}_{Chrom}_{hgvs_c}" with hgvs_c
    "NM_000059.4:c.7436-10T>A" -- the text after the last ":" is exactly
    Table S3's "Coding sequence change (c.)"."""
    tree, model_selections, calibrations = discover_outputs(Path(output_dir))
    if dataset not in tree:
        raise FileNotFoundError(f"{dataset} not found in {output_dir}")
    df = load_all_variants(
        tree=tree, model_selections=model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=[dataset], calibrations=calibrations, min_controls=0,
        dataset_tsv=dataset_tsv,
    )
    if df.empty:
        raise ValueError(f"no variants loaded for {dataset} from {output_dir}")
    method = method or sorted(df["method"].unique())[0]
    df = df[df["method"] == method].copy()
    df["hgvs_c_short"] = df["variant_id"].astype(str).str.rsplit(":", n=1).str[-1]
    if df["hgvs_c_short"].duplicated().any():
        raise ValueError(f"{dataset}: duplicate c. changes among loaded variants")
    print(f"ExCALIBR [{dataset} / {method}]: {len(df):,} variants")
    return df, method


def points_confusion_matrix(df: pd.DataFrame, points_col: str) -> pd.DataFrame:
    """Same [BLB, PLP] x [Normal, IR, Abnormal] frame build_confusion_matrix
    produces, tallied from an arbitrary points column -- mirrors
    comparison_methods.build_acmgscaler_confusion_matrix."""
    df_plp = df[sample_matches(df, "Pathogenic/Likely Pathogenic")]
    df_blb = df[sample_matches(df, "Benign/Likely Benign")]

    def _counts(sub):
        pts = sub[points_col]
        return [int((pts < 0).sum()), int((pts == 0).sum()), int((pts > 0).sum())]

    return pd.DataFrame(
        [_counts(df_blb), _counts(df_plp)], index=["BLB", "PLP"], columns=["Normal", "IR", "Abnormal"],
    )


def _vus_determinate_pct(coverage):
    if coverage is None:
        return None
    n_det, n_vus = coverage
    return 100 * n_det / n_vus if n_vus else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR, help="ExCALIBR pipeline output directory")
    parser.add_argument("--dataset-configs", default=config.DATASET_CONFIGS)
    parser.add_argument("--varcall", default=DEFAULT_VARCALL, help="Huang et al. Table S3 CSV")
    parser.add_argument("--varcall-score-dir", default=DEFAULT_WORK_DIR,
                        help="run_varcall_score_calibration_brca2_huang.py --work-dir")
    parser.add_argument("--figure-dir", default=str(Path(config.FIGURE_DIR) / "clinvar_comparisons" / "varcall"))
    parser.add_argument("--method", default=None, help="ExCALIBR method (default: first discovered)")
    args = parser.parse_args()

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with open(args.dataset_configs) as f:
        dataset_configs = json.load(f)

    # --- 1. ExCALIBR (canonical) + VarCall, joined on the c. change -----------
    df, method = load_excalibr(args.output_dir, DATASET, dataset_configs, args.method)
    varcall = load_varcall(args.varcall)
    df = df.merge(varcall, on="hgvs_c_short", how="left")
    n_unmatched = int(df["varcall_points"].isna().sum())
    print(f"VarCall: {len(varcall):,} calls; matched {len(df) - n_unmatched:,}/{len(df):,} ExCALIBR variants")
    if n_unmatched:
        examples = df.loc[df["varcall_points"].isna(), "variant_id"].head(5).tolist()
        raise ValueError(f"{n_unmatched} ExCALIBR variants have no VarCall call, e.g. {examples}")
    frames = {"excalibr": df, "varcall": df}

    # --- 2. ExCALIBR re-run on VarCall's score, if it has been run -------------
    vs_dir = Path(args.varcall_score_dir)
    vs_table = vs_dir / f"{VARCALL_SCORE_DATASET}.tsv.gz"
    if (vs_dir / VARCALL_SCORE_DATASET).is_dir() and vs_table.exists():
        # Rebuilt from its own temp table (the score column differs from the
        # master TSV's); canonical n_c/benign_method, same as the source dataset.
        df_vs, _ = load_excalibr(
            vs_dir, VARCALL_SCORE_DATASET, {VARCALL_SCORE_DATASET: dataset_configs[DATASET]},
            method, dataset_tsv=str(vs_table),
        )
        # Same variants, same ClinVar labels -- only the score was swapped.
        if set(df_vs["hgvs_c_short"]) != set(df["hgvs_c_short"]):
            only_a = len(set(df["hgvs_c_short"]) - set(df_vs["hgvs_c_short"]))
            only_b = len(set(df_vs["hgvs_c_short"]) - set(df["hgvs_c_short"]))
            raise ValueError(f"variant sets differ between runs ({only_a} only canonical, "
                             f"{only_b} only VarCall-score)")
        frames["excalibr_varcall_score"] = df_vs
    else:
        print(f"\nNOTE: no VarCall-score ExCALIBR run under {vs_dir} -- two-way comparison only. "
              f"Run analysis/run_varcall_score_calibration_brca2_huang.py to add it.")
    methods = [m for m in METHOD_LABELS if m in frames]

    # --- 3. Matrices + VUS coverage on the same ClinVar controls -------------
    mats, vus = {}, {}
    for m in methods:
        if m == "varcall":
            mats[m] = points_confusion_matrix(df, "varcall_points")
            vus[m] = build_vus_coverage(df, points_col="varcall_points")
        else:
            mats[m] = build_confusion_matrix(frames[m], use_oob=USE_OOB[m], label=METHOD_LABELS[m])
            vus[m] = build_vus_coverage(frames[m], use_oob=USE_OOB[m], label=METHOD_LABELS[m])
        if mats[m] is None:
            sys.exit(f"ERROR: {METHOD_LABELS[m]} has no ClinVar P/LP or B/LB controls")
    row_totals = {m: tuple(mats[m].sum(axis=1)) for m in methods}
    if len(set(row_totals.values())) != 1:
        raise AssertionError(f"control totals differ across methods: {row_totals}")

    for m in methods:
        print(f"\n{METHOD_LABELS[m]}:\n{mats[m]}")

    # VarCall strength x ClinVar class, for context the direction-only matrix hides.
    clinvar_class = pd.Series("other", index=df.index)
    clinvar_class[sample_matches(df, "Benign/Likely Benign")] = "B/LB"
    clinvar_class[sample_matches(df, "Pathogenic/Likely Pathogenic")] = "P/LP"
    if "is_vus" in df.columns:
        clinvar_class[df["is_vus"].fillna(False).astype(bool)] = "VUS"
    print("\nVarCall category x ClinVar class:")
    print(pd.crosstab(df["varcall_category"], clinvar_class))

    # --- 4. Figure -------------------------------------------------------------
    if len(methods) == 2:
        make_confusion_figure(
            danzs_m1=[mats["excalibr"]], danzs_m2=[mats["varcall"]], dataset_names=[DATASET],
            label1=method, label2="varcall",
            title1=METHOD_LABELS["excalibr"], title2="VarCall (Huang et al.)",
            # VarCall is evidence, not a functional annotation -- override the
            # author-panel axis ("Functional Annotation", Normal/Abnormal) so
            # both panels read the same.
            xlabel="Evidence Direction", xticklabels=["Benign", "Indeterminate", "Pathogenic"],
            figure_dir=figure_dir, filename=f"{FILE_STEM}.png",
            vus_coverages_m1=[vus["excalibr"]], vus_coverages_m2=[vus["varcall"]],
        )
    else:
        make_confusion_grid_figure(
            panels=[(m, [mats[m]]) for m in methods],
            figure_dir=figure_dir, filename=f"{FILE_STEM}.png",
            vus_coverages=[[vus[m]] for m in methods],
            titles=[METHOD_LABELS[m] if m != "varcall" else "VarCall (Huang et al.)" for m in methods],
        )

    # --- 5. Metrics table --------------------------------------------------------
    agg, own_counts, n_matched = compute_aggregate_metrics_multi(
        {m: [mats[m]] for m in methods}, [DATASET],
    )
    caption = (
        r"ExCALIBR (out-of-bag) versus Huang et al.\ VarCall evidence for BRCA2\_Huang\_2025\_SGE, "
        r"evaluated against the same ClinVar P/LP and B/LB controls. VarCall categories starting "
        r"with P are pathogenic evidence, B benign, VUS indeterminate."
    )
    if "excalibr_varcall_score" in methods:
        caption += (r" ExCALIBR (VarCall score) is ExCALIBR re-calibrated on the same four fitting "
                    r"samples using VarCall's model-based functional score in place of the "
                    r"author-reported score; its evidence is in-bag (20 bootstraps are too few "
                    r"for stable out-of-bag calls), so it is optimistic relative to the OOB column.")
    latex = latex_performance_table_multi(
        agg, {m: METHOD_LABELS[m] for m in methods}, own_counts, 1, n_matched,
        caption=caption, label="tab:excalibr_vs_varcall_brca2_huang",
    )
    save_latex_table(latex, figure_dir / f"{FILE_STEM}_performance.tex")

    metric_keys = [
        "total", "determinate", "coverage", "accuracy", "sensitivity", "specificity", "mcc",
        "lr_plus_standard", "lr_plus_pathogenic", "lr_plus_benign",
        "dor_standard", "dor_pathogenic", "dor_benign",
    ]
    table = pd.DataFrame({METHOD_LABELS[m]: {k: agg[m][k] for k in metric_keys} for m in methods})
    table.loc["vus_n"] = [vus[m][1] if vus[m] else None for m in methods]
    table.loc["vus_determinate_pct"] = [_vus_determinate_pct(vus[m]) for m in methods]
    csv_path = figure_dir / f"{FILE_STEM}_performance.csv"
    table.to_csv(csv_path, index_label="metric")
    print(f"\nMetrics vs ClinVar controls:\n{table.to_string(float_format=lambda x: f'{x:.3f}')}")
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    main()
