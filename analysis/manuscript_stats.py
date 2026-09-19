"""
Manuscript-ready statistics: text/LaTeX summaries computed dynamically from
real, currently-loaded data (never hardcoded numbers) — ported from
test/plot_author_calibration_confusion.py.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _fmt_int(x) -> str:
    return f"{x:,}"


def _fmt_pct(x) -> str:
    return f"{100 * x:.1f}\\%"


def _fmt3(x) -> str:
    return f"{x:.3f}"


def _fmt1(x) -> str:
    return f"{x:.1f}"


def _format_dataset_label(dataset_name: str) -> str:
    r"""LaTeX dataset label: \textit{Gene} (Author Year) -- e.g.
    BRCA1_Findlay_2018_clinvar_2018 -> \textit{BRCA1} (Findlay 2018). Reuses
    analysis.gene_table's existing gene/author/year parsing (already handles
    multi-gene datasets like CALM1_CALM2_CALM3 and unpublished/IGVF ones)
    instead of re-deriving that logic. Falls back to the dataset name's
    second underscore-delimited token (e.g. "IGVF") when no author/year
    pattern is found (extract_author_from_dataset returns None for those,
    per its own docstring).
    """
    from analysis.gene_table import (
        extract_author_from_dataset, extract_gene_from_dataset, extract_year_from_dataset,
    )

    gene = extract_gene_from_dataset(dataset_name)
    author = extract_author_from_dataset(dataset_name)
    year = extract_year_from_dataset(dataset_name)

    if author is not None:
        author = author.replace(" et al.", "")
        label = f"{author} {year}" if year else author
    else:
        parts = dataset_name.split("_")
        label = parts[1] if len(parts) > 1 else dataset_name
    return f"\\textit{{{gene}}} ({label})"


def latex_performance_table_clinvar(groups: List[Tuple[str, Dict, Dict]]) -> str:
    """LaTeX table comparing ExCALIBR evidence vs. author annotations on
    P/LP and B/LB variants — one (ExCALIBR, Author) column pair per entry in
    `groups` (each a `(group_label, danz_metrics, auth_metrics)` tuple, the
    metrics dicts from
    src.assay_calibration.plot_utils.utils.print_aggregate_performance).
    A single group renders as a plain 2-column table (no grouped header
    row); multiple groups (e.g. "All variants" + "Both determinate") share
    one table with a `\\multicolumn` header row spanning each group's own
    ExCALIBR/Author pair, so the same metric can be compared side by side
    across scopes without duplicating the table.

    Returns the LaTeX source as a string (also printed, matching the legacy
    script's behavior).
    """
    def _rows_for(danz: Dict, auth: Dict) -> List[Tuple[str, str, str]]:
        return [
            ("Total variants", _fmt_int(danz["total"]), _fmt_int(auth["total"])),
            ("Determinate",
             f'{_fmt_int(danz["determinate"])} ({_fmt_pct(danz["coverage"])})',
             f'{_fmt_int(auth["determinate"])} ({_fmt_pct(auth["coverage"])})'),
            ("Indeterminate",
             f'{_fmt_int(danz["uncertain"])} ({_fmt_pct(1 - danz["coverage"])})',
             f'{_fmt_int(auth["uncertain"])} ({_fmt_pct(1 - auth["coverage"])})'),
            ("\\midrule", "", ""),
            ("Accuracy", _fmt3(danz["accuracy"]), _fmt3(auth["accuracy"])),
            ("Sensitivity", _fmt3(danz["sensitivity"]), _fmt3(auth["sensitivity"])),
            ("Specificity", _fmt3(danz["specificity"]), _fmt3(auth["specificity"])),
            ("MCC", _fmt3(danz["mcc"]), _fmt3(auth["mcc"])),
            ("\\midrule", "", ""),
            ("LR$^+$ (P vs. B)", _fmt1(danz["lr_plus_standard"]), _fmt1(auth["lr_plus_standard"])),
            ("LR$^+$ (P vs. R)", _fmt1(danz["lr_plus_pathogenic"]), _fmt1(auth["lr_plus_pathogenic"])),
            ("LR$^+$ (B vs. R)", _fmt1(danz["lr_plus_benign"]), _fmt1(auth["lr_plus_benign"])),
            ("DOR (P vs. B)", _fmt1(danz["dor_standard"]), _fmt1(auth["dor_standard"])),
            ("DOR (P vs. R)", _fmt1(danz["dor_pathogenic"]), _fmt1(auth["dor_pathogenic"])),
            ("DOR (B vs. R)", _fmt1(danz["dor_benign"]), _fmt1(auth["dor_benign"])),
        ]

    per_group_rows = [_rows_for(danz, auth) for _, danz, auth in groups]
    n_metric_rows = len(per_group_rows[0])
    assert all(len(r) == n_metric_rows for r in per_group_rows), "groups must report the same metric rows"

    n_cols = 1 + 2 * len(groups)
    col_spec = "l" + "cc" * len(groups)
    header_lines = []
    if len(groups) > 1:
        multicol = " & ".join(f"\\multicolumn{{2}}{{c}}{{{label}}}" for label, _, _ in groups)
        header_lines.append(f" & {multicol} \\\\")
    sub_header = "Metric & " + " & ".join("ExCALIBR & Author" for _ in groups) + " \\\\"
    header_lines.append(sub_header)

    lines = [
        r"\begin{table}[!tb]",
        r"\centering",
        r"\caption{Performance comparison of out-of-bag \excalibr-calibrated evidence vs. author-provided functional annotations on P/LP and B/LB variants. Pathogenic (P) and benign (B) refer to the direction of evidence assigned; R = rest.}",
        r"\label{tab:author_performance}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        *header_lines,
        r"\midrule",
    ]
    for row_idx in range(n_metric_rows):
        metric_label = per_group_rows[0][row_idx][0]
        if metric_label == "\\midrule":
            lines.append(r"\midrule")
            continue
        cells = [metric_label]
        for rows in per_group_rows:
            cells.extend(rows[row_idx][1:])
        assert len(cells) == n_cols
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    print(latex)
    return latex


def compute_aggregate_metrics_multi(matrices_by_method: Dict[str, list], dataset_names: list):
    """N-method generalization of src.assay_calibration.plot_utils.utils.compute_aggregate_metrics
    (which is hardcoded to exactly "danz"/"auth").

    `matrices_by_method`: {method_label: [confusion_df_or_None, ...]}, each
    list the same length/order as `dataset_names` (one entry per dataset, as
    e.g. `conf_by_method[primary_method]` / `acmgscaler_conf_raw` /
    `manual_prior_conf_raw` already are in analyze_pipeline_output.py).

    Returns (agg_metrics_by_method, own_calibrated_counts, n_matched):
      - agg_metrics_by_method: {method_label: compute_classification_metrics(...)}
        computed on each method's matrices summed only over the datasets
        where EVERY method in `matrices_by_method` has a non-None matrix
        (matched/intersection semantics, same as
        analyze_pipeline_output.py's own `_compare_vs_excalibr`).
      - own_calibrated_counts: {method_label: count of non-None matrices
        across the FULL `dataset_names` list} -- each method's own
        denominator, independent of the other methods in the table.
      - n_matched: size of the intersection actually used for agg_metrics_by_method.
    """
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics

    methods = list(matrices_by_method.keys())
    n = len(dataset_names)

    own_calibrated_counts = {
        m: sum(1 for mat in matrices_by_method[m] if mat is not None) for m in methods
    }

    matched_idx = [
        i for i in range(n) if all(matrices_by_method[m][i] is not None for m in methods)
    ]

    aggregates: Dict[str, pd.DataFrame] = {}
    for i in matched_idx:
        for m in methods:
            mat = matrices_by_method[m][i]
            aggregates[m] = mat.copy() if m not in aggregates else aggregates[m] + mat

    agg_metrics_by_method = {
        m: compute_classification_metrics(aggregates[m]) for m in methods if m in aggregates
    }
    return agg_metrics_by_method, own_calibrated_counts, len(matched_idx)


def latex_performance_table_multi(
    agg_metrics_by_method: Dict[str, Dict],
    method_labels: Dict[str, str],
    own_calibrated_counts: Dict[str, int],
    n_total_datasets: int,
    n_matched: int,
    caption: str,
    label: str,
) -> str:
    """N-column generalization of latex_performance_table_clinvar, for
    comparing ExCALIBR against an arbitrary number of other methods (rather
    than exactly one, "Author Annotations") -- same row layout, plus a
    leading "Datasets calibratable" row (own_calibrated_counts[m]/n_total_datasets
    per column) so it's visible at a glance how big each method's own
    calibratable subset is, not just the common intersection the metrics
    below it are pooled over (n_matched, folded into the caption).

    Column order = `method_labels`' insertion order (must match the keys of
    `agg_metrics_by_method`/`own_calibrated_counts`).
    """
    methods = list(method_labels.keys())

    def _fmt1(x) -> str:
        return f"{x:.1f}"

    rows = [
        ("Datasets calibratable",
         *[f'{own_calibrated_counts[m]:,}/{n_total_datasets:,}' for m in methods]),
        ("Total variants",
         *[_fmt_int(agg_metrics_by_method[m]["total"]) for m in methods]),
        ("Determinate",
         *[f'{_fmt_int(agg_metrics_by_method[m]["determinate"])} ({_fmt_pct(agg_metrics_by_method[m]["coverage"])})'
           for m in methods]),
        ("Indeterminate",
         *[f'{_fmt_int(agg_metrics_by_method[m]["uncertain"])} ({_fmt_pct(1 - agg_metrics_by_method[m]["coverage"])})'
           for m in methods]),
        ("\\midrule",),
        ("Accuracy", *[_fmt3(agg_metrics_by_method[m]["accuracy"]) for m in methods]),
        ("Sensitivity", *[_fmt3(agg_metrics_by_method[m]["sensitivity"]) for m in methods]),
        ("Specificity", *[_fmt3(agg_metrics_by_method[m]["specificity"]) for m in methods]),
        ("MCC", *[_fmt3(agg_metrics_by_method[m]["mcc"]) for m in methods]),
        ("\\midrule",),
        ("LR$^+$ (P vs. B)", *[_fmt1(agg_metrics_by_method[m]["lr_plus_standard"]) for m in methods]),
        ("LR$^+$ (P vs. R)", *[_fmt1(agg_metrics_by_method[m]["lr_plus_pathogenic"]) for m in methods]),
        ("LR$^+$ (B vs. R)", *[_fmt1(agg_metrics_by_method[m]["lr_plus_benign"]) for m in methods]),
        ("DOR (P vs. B)", *[_fmt1(agg_metrics_by_method[m]["dor_standard"]) for m in methods]),
        ("DOR (P vs. R)", *[_fmt1(agg_metrics_by_method[m]["dor_pathogenic"]) for m in methods]),
        ("DOR (B vs. R)", *[_fmt1(agg_metrics_by_method[m]["dor_benign"]) for m in methods]),
    ]

    col_spec = "l" + "c" * len(methods)
    header = "Metric & " + " & ".join(method_labels[m] for m in methods) + " \\\\"

    lines = [
        r"\begin{table}[!tb]",
        r"\centering",
        rf"\caption{{{caption} Aggregate metrics computed over the {n_matched} "
        r"dataset(s) common to all methods shown.}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for r in rows:
        if r[0] == "\\midrule":
            lines.append(r"\midrule")
        else:
            lines.append(f"{r[0]} & " + " & ".join(r[1:]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    print(latex)
    return latex


def latex_performance_table_clingen(groups: List[Tuple[str, Dict]]) -> str:
    """LaTeX table for the ClinGen ground-truth confusion (analysis/clingen.py) —
    one (ExCALIBR, Author) column pair per entry in `groups` (each a
    `(group_label, conf_dict)` tuple, where `conf_dict` is
    {'excalibr': 2x3 array, 'auth': 2x3 array}, e.g.
    {'excalibr': convert_3x2_to_2x3(confusion['excalibr']),
     'auth': convert_3x2_to_2x3(confusion['auth'])}). Computes metrics
    directly from the 2x3 confusion arrays via compute_classification_metrics,
    so every number is dynamic. A single group renders as a plain 2-column
    table; multiple groups (e.g. "PS3/BS3 excluded" + "PS3/BS3 included")
    share one table with a `\\multicolumn` header row spanning each group's
    own ExCALIBR/Author pair.
    """
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics

    def _rows_for(conf_dict: Dict) -> List[Tuple[str, str, str]]:
        ex = compute_classification_metrics(pd.DataFrame(conf_dict['excalibr']))
        au = compute_classification_metrics(pd.DataFrame(conf_dict['auth']))
        total_ex, total_au = ex['total'], au['total']

        def pct(n, d):
            return 100 * n / d if d > 0 else 0.0

        return [
            ("Total variants", _fmt_int(total_ex), _fmt_int(total_au)),
            ("Determinate",
             f"{ex['determinate']:,} ({pct(ex['determinate'], total_ex):.1f}\\%)",
             f"{au['determinate']:,} ({pct(au['determinate'], total_au):.1f}\\%)"),
            ("Indeterminate",
             f"{ex['uncertain']:,} ({pct(ex['uncertain'], total_ex):.1f}\\%)",
             f"{au['uncertain']:,} ({pct(au['uncertain'], total_au):.1f}\\%)"),
            ("\\midrule", "", ""),
            ("Accuracy", _fmt3(ex['accuracy']), _fmt3(au['accuracy'])),
            ("Sensitivity", _fmt3(ex['sensitivity']), _fmt3(au['sensitivity'])),
            ("Specificity", _fmt3(ex['specificity']), _fmt3(au['specificity'])),
            ("MCC", _fmt3(ex['mcc']), _fmt3(au['mcc'])),
            ("\\midrule", "", ""),
            ("LR$^+$ (P vs. B)", _fmt1(ex['lr_plus_standard']), _fmt1(au['lr_plus_standard'])),
            ("LR$^+$ (P vs. R)", _fmt1(ex['lr_plus_pathogenic']), _fmt1(au['lr_plus_pathogenic'])),
            ("LR$^+$ (B vs. R)", _fmt1(ex['lr_plus_benign']), _fmt1(au['lr_plus_benign'])),
            ("DOR (P vs. B)", _fmt1(ex['dor_standard']), _fmt1(au['dor_standard'])),
            ("DOR (P vs. R)", _fmt1(ex['dor_pathogenic']), _fmt1(au['dor_pathogenic'])),
            ("DOR (B vs. R)", _fmt1(ex['dor_benign']), _fmt1(au['dor_benign'])),
        ]

    per_group_rows = [_rows_for(conf_dict) for _, conf_dict in groups]
    n_metric_rows = len(per_group_rows[0])
    assert all(len(r) == n_metric_rows for r in per_group_rows), "groups must report the same metric rows"

    n_cols = 1 + 2 * len(groups)
    col_spec = "l" + "cc" * len(groups)
    header_lines = []
    if len(groups) > 1:
        multicol = " & ".join(f"\\multicolumn{{2}}{{c}}{{{label}}}" for label, _ in groups)
        header_lines.append(f" & {multicol} \\\\")
    header_lines.append("Metric & " + " & ".join("ExCALIBR & Author" for _ in groups) + " \\\\")

    lines = [
        r"\begin{table}[!tb]",
        r"\centering",
        r"\caption{Performance comparison of out-of-bag \excalibr-calibrated evidence vs. author-provided functional annotations on P/LP and B/LB variants, against ClinGen expert-panel ground truth. Pathogenic (P) and benign (B) refer to the direction of evidence assigned; R = rest.}",
        r"\label{tab:clingen_performance}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        *header_lines,
        r"\midrule",
    ]
    for row_idx in range(n_metric_rows):
        metric_label = per_group_rows[0][row_idx][0]
        if metric_label == "\\midrule":
            lines.append(r"\midrule")
            continue
        cells = [metric_label]
        for rows in per_group_rows:
            cells.extend(rows[row_idx][1:])
        assert len(cells) == n_cols
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    print(latex)
    return latex


def latex_bootstrap_reduction_metrics_table(table_df: pd.DataFrame) -> str:
    """LaTeX table of standard classification metrics (accuracy/determinate
    %/sensitivity/specificity/DOR/MCC), median [IQR] across datasets, at
    each bootstrap-count level -- analysis.robustness.bootstrap_reduction_metrics_table's
    output. Same median [IQR] cell style as latex_robustness_metrics_table,
    except the "spread" pooled here is cross-dataset (at a fixed level), not
    cross-seed -- bootstrap-count reduction computes exactly ONE calibration
    per (dataset, level), so there's no seed-to-seed spread within a cell
    (see analysis.robustness's own module docstring for that section).
    """
    def _fmt_pct_cell(p25, p50, p75, n):
        if not np.isfinite(p50):
            return "--"
        if n <= 1 or not np.isfinite(p25):
            return f"{100 * p50:.1f}\\%"
        return f"{100 * p50:.1f}\\% [{100 * p25:.1f}, {100 * p75:.1f}]"

    def _fmt_num_cell(p25, p50, p75, n, nd=2):
        if not np.isfinite(p50):
            return "--"
        if n <= 1 or not np.isfinite(p25):
            return f"{p50:.{nd}f}"
        return f"{p50:.{nd}f} [{p25:.{nd}f}, {p75:.{nd}f}]"

    lines = [
        r"\begin{table}[!tb]",
        r"\centering",
        r"\caption{Calibration performance as the number of bootstrap fits per dataset is "
        r"reduced. Median [IQR] across all datasets at each bootstrap-count level.}",
        r"\label{tab:bootstrap_reduction_metrics}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Bootstraps & $n$ & Accuracy & Determinate \% & Sensitivity & Specificity & DOR & MCC \\",
        r"\midrule",
    ]
    for _, row in table_df.iterrows():
        n = row["n_datasets"]
        lines.append(
            f"{int(row['num_bootstraps']):,} & {int(n):,} & "
            f"{_fmt_pct_cell(row['accuracy_p25'], row['accuracy_p50'], row['accuracy_p75'], n)} & "
            f"{_fmt_pct_cell(row['coverage_p25'], row['coverage_p50'], row['coverage_p75'], n)} & "
            f"{_fmt_pct_cell(row['sensitivity_p25'], row['sensitivity_p50'], row['sensitivity_p75'], n)} & "
            f"{_fmt_pct_cell(row['specificity_p25'], row['specificity_p50'], row['specificity_p75'], n)} & "
            f"{_fmt_num_cell(row['dor_standard_p25'], row['dor_standard_p50'], row['dor_standard_p75'], n, nd=1)} & "
            f"{_fmt_num_cell(row['mcc_p25'], row['mcc_p50'], row['mcc_p75'], n, nd=3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    print(latex)
    return latex


def latex_fit_number_comparison_table(table_df: pd.DataFrame, metric: str = "delta_std") -> str:
    """LaTeX table of analysis.robustness.summarize_delta_std_table's output
    (median [P25, P75] of `metric` across every (dataset, n_c) row, at each
    restart-count/num_fits level) -- the tabular twin of
    analysis.robustness.plot_fit_number_comparison_curve.

    `metric`: "delta_std" (dimensionless, in units of that (dataset, n_c)'s
    own restart-to-restart SD -- the most interpretable across datasets,
    default), "delta" (raw train_ll units, not comparable across datasets
    with different likelihood scales), or "geometric_mean_lr_pct" (bounded
    (0%, 100%] geometric-mean likelihood ratio vs. the all-restarts
    baseline -- see analysis.robustness.compute_delta_std_column's docstring
    for why this is explicitly NOT a classification-style quality percentage).
    """
    metric_labels = {
        "delta_std": r"$\Delta$ / restart SD",
        "delta": r"$\Delta$ train LL",
        "geometric_mean_lr_pct": r"Geometric-mean LR (\%)",
    }
    metric_label = metric_labels.get(metric, metric)

    def fmt(x, nd=3):
        return f"{x:.{nd}f}" if np.isfinite(x) else "--"

    lines = [
        r"\begin{table}[!tb]",
        r"\centering",
        rf"\caption{{Fit quality ({metric_label}) vs. restart count, median [P25, P75] across "
        r"every (dataset, $n_c$) row at each restart-count level.}",
        r"\label{tab:fit_number_comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        rf"Restarts & $n$ & Median & P25 & P75 \\",
        r"\midrule",
    ]
    for _, row in table_df.iterrows():
        lines.append(
            f"{int(row['num_fits']):,} & {int(row['n']):,} & "
            f"{fmt(row['median'])} & {fmt(row['p25'])} & {fmt(row['p75'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    print(latex)
    return latex


def latex_robustness_metrics_table(table_df: pd.DataFrame, perturbation_type: str) -> str:
    """LaTeX table of accuracy / determinate-% / max evidence strength
    (median [IQR] across seeds) for every (base_dataset, level) cell of the
    downsample/discordance robustness confusion-matrix grid
    (analysis.robustness.plot_robustness_confusion_matrix_grid) -- same
    data, tidy table form for the manuscript instead of a figure caption.

    `table_df`: analysis.robustness.robustness_seed_metrics_table's output
    (columns base_dataset, level_label, n_seeds, accuracy_p25/p50/p75,
    coverage_p25/p50/p75, maxp_p25/p50/p75, maxb_p25/p50/p75), already in
    the grid's own row order.
    """
    def _fmt_pct_cell(p25, p50, p75, n_seeds):
        if n_seeds <= 1:
            return f"{p50:.1f}\\%"
        return f"{p50:.1f}\\% [{p25:.1f}, {p75:.1f}]"

    def _fmt_signed_cell(p25, p50, p75, n_seeds):
        if p50 is None:
            return "--"
        if n_seeds <= 1:
            return f"{p50:+.0f}"
        return f"{p50:+.0f} [{p25:+.0f}, {p75:+.0f}]"

    title = "Downsampling" if perturbation_type == "downsample" else "Label discordance"
    lines = [
        r"\begin{table}[!tb]",
        r"\centering",
        rf"\caption{{Robustness ({title.lower()}) confusion-matrix accuracy, determinate "
        r"coverage, and max pathogenic/benign evidence strength, median [IQR] across 10 seeds per level. "
        r"Metrics are computed with respect to the original control samples.}}",
        rf"\label{{tab:robustness_{perturbation_type}_metrics}}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Dataset & Level & Accuracy & Determinate \% & Max Pathogenic & Max Benign \\",
        r"\midrule",
    ]
    prev_dataset = None
    for _, row in table_df.iterrows():
        if prev_dataset is not None and row["base_dataset"] != prev_dataset:
            lines.append(r"\midrule")
        prev_dataset = row["base_dataset"]
        acc_cell = _fmt_pct_cell(row["accuracy_p25"], row["accuracy_p50"], row["accuracy_p75"], row["n_seeds"])
        cov_cell = _fmt_pct_cell(row["coverage_p25"], row["coverage_p50"], row["coverage_p75"], row["n_seeds"])
        maxp_cell = _fmt_signed_cell(row["maxp_p25"], row["maxp_p50"], row["maxp_p75"], row["n_seeds"])
        maxb_cell = _fmt_signed_cell(row["maxb_p25"], row["maxb_p50"], row["maxb_p75"], row["n_seeds"])
        lines.append(
            f"{_format_dataset_label(row['base_dataset'])} & {row['level_label']} & {acc_cell} & {cov_cell} & "
            f"{maxp_cell} & {maxb_cell} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    print(latex)
    return latex


def latex_moi_comparison_table(moi_comparison_df: pd.DataFrame, label: str) -> str:
    """LaTeX table of the AD-vs-AR mode-of-inheritance per-metric comparison
    (analyze_pipeline_output.py section 11's `_compute_moi_comparison` --
    columns metric, mean_AD, mean_AR, median_AD, median_AR, n_AD, n_AR,
    p_value)."""
    header_cells = ["Metric", "Mean AD", "Mean AR", "Median AD", "Median AR", "$n$ AD", "$n$ AR", "$p$-value"]
    colspec = "l" + "c" * (len(header_cells) - 1)
    lines = [
        r"\begin{table}[!tb]",
        r"\centering",
        rf"\caption{{Mode-of-inheritance (AD vs. AR) performance comparison, {label}.}}",
        rf"\label{{tab:moi_comparison_{'dual_dropped' if 'dropped' in label else 'dual_counted'}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]

    def _fmt_metric_or_dash(x, is_dor):
        if x is None:
            return "--"
        return _fmt_int(round(x)) if is_dor else _fmt3(x)

    for _, row in moi_comparison_df.iterrows():
        is_dor = row["metric"] == "DOR"
        cells = [
            row["metric"],
            _fmt_metric_or_dash(row["mean_AD"], is_dor),
            _fmt_metric_or_dash(row["mean_AR"], is_dor),
            _fmt_metric_or_dash(row["median_AD"], is_dor),
            _fmt_metric_or_dash(row["median_AR"], is_dor),
            _fmt_int(row["n_AD"]),
            _fmt_int(row["n_AR"]),
            _fmt3(row["p_value"]) if row["p_value"] is not None else "--",
        ]
        assert len(cells) == len(header_cells)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    print(latex)
    return latex


def latex_splice_ablation_summary_table(summary_df: pd.DataFrame, baseline_condition: str = "thresh_0.2") -> str:
    """LaTeX table of analysis.splice_ablation.compute_splice_ablation_aggregate_summary's
    output (Table A: each condition's own population + own calibration,
    pooled across every dataset) -- one row per condition, with percent
    change relative to `baseline_condition` shown alongside each metric/
    population count. Full metric set stays in the returned DataFrame for
    anyone who wants more than this published subset (mirrors
    plot_splice_ablation_curve's own default 3-metric subset vs. the fuller
    run_splice_ablation_analysis DataFrame).
    """
    def _fmt_metric(row, col):
        base = "--" if pd.isna(row[f"pct_change_{col}"]) else f"{row[f'pct_change_{col}']:+.1f}\\%"
        return f"{row[col]:.3f} ({base})"

    def _fmt_pct_only(row, col):
        return "--" if pd.isna(row[f"pct_change_{col}"]) else f"{row[f'pct_change_{col}']:+.1f}\\%"

    lines = [
        r"\begin{table}[!tb]", r"\centering",
        r"\caption{SpliceAI-threshold / VEP-splice-filter ablation: pooled aggregate performance "
        rf"per condition, with percent change relative to {baseline_condition} (VEP filter on, "
        r"SpliceAI threshold 0.2 -- Scoreset.splicing\_filter's own defaults). "
        r"Metrics are computed with respect to the original control samples.}",
        r"\label{tab:splice_ablation_summary}",
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r"Condition & Accuracy ($\Delta\%$) & Coverage ($\Delta\%$) & DOR ($\Delta\%$) & "
        r"$\Delta$PLP\% & $\Delta$BLB\% & $\Delta$gnomAD\% & $\Delta$Synonymous\% \\",
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['condition_label']} & {_fmt_metric(row, 'accuracy')} & {_fmt_metric(row, 'coverage')} & "
            f"{_fmt_metric(row, 'dor_standard')} & {_fmt_pct_only(row, 'n_plp')} & "
            f"{_fmt_pct_only(row, 'n_blb')} & {_fmt_pct_only(row, 'n_gnomad')} & "
            f"{_fmt_pct_only(row, 'n_synonymous')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)
    print(latex)
    return latex


def latex_splice_ablation_fixed_population_table(
    summary_df: pd.DataFrame, baseline_condition: str = "thresh_0.2",
) -> str:
    """LaTeX table of analysis.splice_ablation.compute_splice_ablation_fixed_population_summary's
    output (Table B: `baseline_condition`'s own variant population held
    fixed, every other condition's calibration re-applied to it) -- isolates
    the calibration-threshold effect from population composition drift. No
    population-count columns: those are identical across every row by
    construction, so showing them would be misleading, not just redundant.
    """
    def _fmt_metric(row, col):
        base = "--" if pd.isna(row[f"pct_change_{col}"]) else f"{row[f'pct_change_{col}']:+.1f}\\%"
        return f"{row[col]:.3f} ({base})"

    lines = [
        r"\begin{table}[!tb]", r"\centering",
        r"\caption{SpliceAI-threshold / VEP-splice-filter ablation, fixed reference population: "
        rf"every condition's own calibration re-applied to {baseline_condition}'s fixed variant "
        r"population (VEP filter on, SpliceAI threshold 0.2 -- Scoreset.splicing\_filter's own "
        r"defaults), isolating the calibration-threshold effect from population composition "
        r"drift (see the companion pooled-population table for that). "
        r"Metrics are computed with respect to the original control samples.}",
        r"\label{tab:splice_ablation_fixed_population}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Condition & Accuracy ($\Delta\%$) & Coverage ($\Delta\%$) & DOR ($\Delta\%$) \\",
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"{row['condition_label']} & {_fmt_metric(row, 'accuracy')} & "
            f"{_fmt_metric(row, 'coverage')} & {_fmt_metric(row, 'dor_standard')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)
    print(latex)
    return latex
