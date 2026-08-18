"""
Manuscript-ready statistics: text/LaTeX summaries computed dynamically from
real, currently-loaded data (never hardcoded numbers) — ported from
test/plot_author_calibration_confusion.py.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd


def _fmt_int(x) -> str:
    return f"{x:,}"


def _fmt_pct(x) -> str:
    return f"{100 * x:.1f}\\%"


def _fmt3(x) -> str:
    return f"{x:.3f}"


def _fmt1(x) -> str:
    return f"{x:.1f}"


def latex_performance_table_clinvar(danz: Dict, auth: Dict) -> str:
    """LaTeX table comparing calibrated (ExCALIBR) evidence vs. author
    annotations on P/LP and B/LB variants — the exact table from
    test/plot_author_calibration_confusion.py, taking the metrics dicts
    returned by src.assay_calibration.plot_utils.utils.print_aggregate_performance
    (danz_agg_metrics / auth_agg_metrics) rather than hardcoded values.

    Returns the LaTeX source as a string (also printed, matching the legacy
    script's behavior).
    """
    total = danz["total"]  # same for both by design

    rows = [
        ("Total variants", _fmt_int(total), _fmt_int(total)),
        ("Determinate assignments",
         f'{_fmt_int(danz["determinate"])} ({_fmt_pct(danz["coverage"])})',
         f'{_fmt_int(auth["determinate"])} ({_fmt_pct(auth["coverage"])})'),
        ("Indeterminate assignments",
         f'{_fmt_int(danz["uncertain"])} ({_fmt_pct(1 - danz["coverage"])})',
         f'{_fmt_int(auth["uncertain"])} ({_fmt_pct(1 - auth["coverage"])})'),
        ("\\midrule", "", ""),
        ("Accuracy", _fmt3(danz["accuracy"]), _fmt3(auth["accuracy"])),
        ("Sensitivity", _fmt3(danz["sensitivity"]), _fmt3(auth["sensitivity"])),
        ("Specificity", _fmt3(danz["specificity"]), _fmt3(auth["specificity"])),
        ("MCC", _fmt3(danz["mcc"]), _fmt3(auth["mcc"])),
        ("\\midrule", "", ""),
        ("LR$^+$ (pathogenic vs. benign)", _fmt1(danz["lr_plus_standard"]), _fmt1(auth["lr_plus_standard"])),
        ("LR$^+$ (pathogenic vs. rest)", _fmt1(danz["lr_plus_pathogenic"]), _fmt1(auth["lr_plus_pathogenic"])),
        ("LR$^+$ (benign vs. rest)", _fmt1(danz["lr_plus_benign"]), _fmt1(auth["lr_plus_benign"])),
        ("DOR (pathogenic vs. benign)", _fmt1(danz["dor_standard"]), _fmt1(auth["dor_standard"])),
        ("DOR (pathogenic vs. rest)", _fmt1(danz["dor_pathogenic"]), _fmt1(auth["dor_pathogenic"])),
        ("DOR (benign vs. rest)", _fmt1(danz["dor_benign"]), _fmt1(auth["dor_benign"])),
    ]

    lines = [
        r"\begin{table}[!tb]",
        r"\centering",
        r"\caption{Performance comparison of out-of-bag \excalibr-calibrated evidence vs. author-provided functional annotations on P/LP and B/LB variants. Pathogenic and benign refer to the direction of evidence assigned.}",
        r"\label{tab:author_performance}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Calibrated Evidence & Author Annotations \\",
        r"\midrule",
    ]
    for r in rows:
        if r[0] == "\\midrule":
            lines.append(r"\midrule")
        else:
            lines.append(f"{r[0]} & {r[1]} & {r[2]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    print(latex)
    return latex


def latex_performance_table_clingen(conf_dict: Dict) -> str:
    """LaTeX table for the ClinGen ground-truth confusion (analysis/clingen.py) —
    ported verbatim from test/plot_author_calibration_confusion.py's second
    latex_performance_table(conf_dict). Computes metrics directly from the
    2x3 confusion arrays via compute_classification_metrics, so every number
    is dynamic.

    conf_dict: {'excalibr': 2x3 array, 'auth': 2x3 array} — e.g.
    {'excalibr': convert_3x2_to_2x3(confusion['excalibr']),
     'auth': convert_3x2_to_2x3(confusion['auth'])}
    """
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics

    ex = compute_classification_metrics(pd.DataFrame(conf_dict['excalibr']))
    au = compute_classification_metrics(pd.DataFrame(conf_dict['auth']))

    def pct(n, d):
        return 100 * n / d if d > 0 else 0.0

    def fmt(x, nd=3):
        return f"{x:.{nd}f}"

    total = ex['total']  # same for both

    table = f"""
\\begin{{table}}[!tb]
\\centering
\\caption{{Performance comparison of out-of-bag \\excalibr-calibrated evidence vs. author-provided functional annotations on P/LP and B/LB variants, against ClinGen expert-panel ground truth. Pathogenic and benign refer to the direction of evidence assigned.}}
\\label{{tab:clingen_performance}}
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Calibrated Evidence & Author Annotations \\\\
\\midrule
Total variants & {total:,} & {total:,} \\\\
Determinate assignments & {ex['determinate']:,} ({pct(ex['determinate'], total):.1f}\\%) & {au['determinate']:,} ({pct(au['determinate'], total):.1f}\\%) \\\\
Indeterminate assignments & {ex['uncertain']:,} ({pct(ex['uncertain'], total):.1f}\\%) & {au['uncertain']:,} ({pct(au['uncertain'], total):.1f}\\%) \\\\
\\midrule
Accuracy & {fmt(ex['accuracy'])} & {fmt(au['accuracy'])} \\\\
Sensitivity & {fmt(ex['sensitivity'])} & {fmt(au['sensitivity'])} \\\\
Specificity & {fmt(ex['specificity'])} & {fmt(au['specificity'])} \\\\
MCC & {fmt(ex['mcc'])} & {fmt(au['mcc'])} \\\\
\\midrule
LR$^+$ (pathogenic vs. benign) & {fmt(ex['lr_plus_standard'], 1)} & {fmt(au['lr_plus_standard'], 1)} \\\\
LR$^+$ (pathogenic vs. rest) & {fmt(ex['lr_plus_pathogenic'], 1)} & {fmt(au['lr_plus_pathogenic'], 1)} \\\\
LR$^+$ (benign vs. rest) & {fmt(ex['lr_plus_benign'], 1)} & {fmt(au['lr_plus_benign'], 1)} \\\\
DOR (pathogenic vs. benign) & {fmt(ex['dor_standard'], 1)} & {fmt(au['dor_standard'], 1)} \\\\
DOR (pathogenic vs. rest) & {fmt(ex['dor_pathogenic'], 1)} & {fmt(au['dor_pathogenic'], 1)} \\\\
DOR (benign vs. rest) & {fmt(ex['dor_benign'], 1)} & {fmt(au['dor_benign'], 1)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    print(table)
    return table


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
        r"coverage, and max pathogenic/benign evidence strength, median [IQR] across 10 seeds per level.}}",
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
            f"{row['base_dataset']} & {row['level_label']} & {acc_cell} & {cov_cell} & "
            f"{maxp_cell} & {maxb_cell} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    latex = "\n".join(lines)
    print(latex)
    return latex
