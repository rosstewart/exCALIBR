"""The independent baselines the paper compares joint calibration against, per
evidence type, as per-gene results rows (evidence-direction threshold):

  functional  every assay calibrated with ExCALIBR, combined by the
              nonconflicting rule (the results tables' "UV non-conflicting")
  predictor   each predictor calibrated alone (Chen et al.), since practice is
              to pick one predictor rather than combine them
  combined    functional evidence (nonconflicting rule) plus one predictor's

The nonconflicting combination of the three predictors is kept as an extra
baseline ("Nonconflicting") for predictors and combined evidence; it is not a
practice anyone follows, so it is reported as secondary.

Rows carry gene, cluster, TP, TN, FP, FN, n_pathogenic, n_benign, total,
determinate, mcc, accuracy, sensitivity, specificity and the ratio columns of
the results tables. Sources: the per-gene results tables,
predictor/predictor_baseline_comparison.csv
(run_predictor_baseline_comparison) and
combined/combined_single_predictor_comparison.csv
(run_combined_single_predictor_comparison).
"""
from pathlib import Path

import pandas as pd

from mv_analysis import config

EVIDENCE_DIRECTION = "evidence_direction (>=1 / <=-1)"
# predictor code in the comparison tables -> display name
PREDICTORS = (("REVEL", "REVEL"), ("AM", "AlphaMissense"), ("MP2", "MutPred2"))
TABLES = {"functional": "functional/functional_all_results_table.csv",
          "predictor": "predictor/predictor_results_table.csv",
          "combined": "combined/combined_results_table.csv"}


def baselines(panel, figures_dir=None, secondary=False, pknn=False):
    """[(label, rows)] in display order; with secondary=True the predictors'
    nonconflicting combination is appended for predictor and combined, and
    with pknn=True genome-wide P-KNN as published (in-sample;
    pknn/pknn_genomewide_ingene_rows.csv), alone for predictors or added to
    functional evidence for combined."""
    figs = Path(figures_dir or config.PAPER_FIGURES_DIR)
    if panel == "functional":
        t = pd.read_csv(figs / TABLES[panel])
        t = t[(t["threshold"] == EVIDENCE_DIRECTION) & (t["method"] == "UV non-conflicting")]
        return [("Independent", t)]
    if panel == "predictor":
        t = pd.read_csv(figs / "predictor/predictor_baseline_comparison.csv")
        t = t[t["source"] == config.PREDICTOR_UV_SOURCE]
        if "cluster" not in t.columns:
            t = t.assign(cluster="gene")
        out = [(name, t[t["predictor"] == code]) for code, name in PREDICTORS]
        out += _pknn_rows(figs, panel, "P-KNN genome-wide") if pknn else []
        if secondary:
            u = pd.read_csv(figs / TABLES[panel])
            u = u[(u["threshold"] == EVIDENCE_DIRECTION) & (u["method"] == "UV non-conflicting")]
            out.append(("Nonconflicting", u))
        return out
    if panel == "combined":
        t = pd.read_csv(figs / "combined/combined_single_predictor_comparison.csv")
        if "threshold" in t.columns:
            t = t[t["threshold"] == EVIDENCE_DIRECTION]
        if "cluster" not in t.columns:
            t = t.assign(cluster="gene")
        out = [(f"Functional + {name}", t[t["predictor"] == code]) for code, name in PREDICTORS]
        out += _pknn_rows(figs, panel, "Functional + P-KNN genome-wide") if pknn else []
        if secondary:
            u = pd.read_csv(figs / TABLES[panel])
            u = u[(u["threshold"] == EVIDENCE_DIRECTION) & (u["method"] == "UV non-conflicting")]
            out.append(("Nonconflicting", u))
        return out
    raise ValueError(panel)


def _pknn_rows(figs, panel, label):
    f = figs / "pknn/pknn_genomewide_ingene_rows.csv"
    if not f.exists():
        return []
    t = pd.read_csv(f)
    return [(label, t[t["panel"] == panel])]


def joint(panel, figures_dir=None):
    """Joint calibration's per-gene rows at each gene's canonical component count."""
    figs = Path(figures_dir or config.PAPER_FIGURES_DIR)
    t = pd.read_csv(figs / TABLES[panel])
    t = t[(t["threshold"] == EVIDENCE_DIRECTION) & t["method"].str.startswith("MV")]
    return config.select_canonical_mv_rows(t)


def undefined_oob_mcc(figures_dir=None):
    """{(key, method)} (key: panel__gene[__cN]) whose out-of-bag MCC is undefined (0/0): no
    determinate call, or every determinate call in one direction or on one
    control class. The comparison tables report these as 0, which would read
    as chance-level; figures drop them. Source:
    production_oob/production_oob_comparison.csv (methods MV, P-KNN, classifiers)."""
    figs = Path(figures_dir or config.PAPER_FIGURES_DIR)
    f = figs / "production_oob/production_oob_comparison.csv"
    if not f.exists():
        return set()
    o = pd.read_csv(f)
    o = o[o["threshold"] == EVIDENCE_DIRECTION] if "threshold" in o.columns else o
    den = (o["TP"] + o["FP"]) * (o["TN"] + o["FN"]) * (o["TP"] + o["FN"]) * (o["TN"] + o["FP"])
    return set(map(tuple, o.loc[den == 0, ["key", "method"]].values))


def drop_undefined_oob(p, pairs, figures_dir=None):
    """Rows of a per-gene table (column key) with each (column,
    method) of pairs set to NaN where that method's out-of-bag MCC is undefined."""
    bad = undefined_oob_mcc(figures_dir)
    p = p.copy()
    for col, method in pairs:
        m = [(k, method) in bad for k in p["key"]]
        p.loc[m, col] = float("nan")
    return p



COUNTS = ("TP", "TN", "FP", "FN", "uncertain", "total", "determinate", "n_pathogenic", "n_benign")


def gene_level(rows, by=("gene",)):
    """One row per gene from per-group rows (BRCA2's assays form two groups
    whose evaluated control variants do not overlap, although their fitting
    sets share the predictor-scored variants): confusion counts summed over the gene's
    groups and the metrics recomputed, so per-gene medians and gene counts
    weigh each gene once. by: identifying columns kept (e.g. panel, method)."""
    import numpy as np
    by = [c for c in by if c in rows.columns]
    cnt = [c for c in COUNTS if c in rows.columns]
    g = rows.groupby(by, as_index=False, sort=False)[cnt].sum()
    tp, tn, fp, fn = (g[c].astype(float) for c in ("TP", "TN", "FP", "FN"))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    g["mcc"] = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
    g["accuracy"] = (tp + tn) / (tp + tn + fp + fn).where(lambda x: x > 0)
    g["sensitivity"] = tp / (tp + fn).where(lambda x: x > 0)
    g["specificity"] = tn / (tn + fp).where(lambda x: x > 0)
    if "determinate" in g.columns and "total" in g.columns:
        g["det_rate"] = g["determinate"] / g["total"].where(lambda x: x > 0)
    g["cluster"] = "gene"
    g["n_groups"] = rows.groupby(by, sort=False).size().to_numpy()
    return g


def mcc_defined(x):
    """MCC is defined only when the determinate calls cover both control
    classes and both directions (else 0/0)."""
    return ((x["TP"] + x["FP"]) > 0) & ((x["TN"] + x["FN"]) > 0) & ((x["TP"] + x["FN"]) > 0) & ((x["TN"] + x["FP"]) > 0)


def oob_gene_level(figures_dir=None):
    """production_oob_comparison.csv at the gene level (groups merged by
    gene_level), MCC set to NaN where undefined, genes lacking a control
    class dropped; columns panel, gene, method, mcc, det_rate and counts."""
    figs = Path(figures_dir or config.PAPER_FIGURES_DIR)
    o = pd.read_csv(figs / "production_oob/production_oob_comparison.csv")
    if "threshold" in o.columns:
        o = o[o["threshold"] == EVIDENCE_DIRECTION]
    g = gene_level(o, by=("panel", "gene", "method"))
    g = g[(g["n_pathogenic"] > 0) & (g["n_benign"] > 0)]
    return g.assign(mcc=g["mcc"].where(mcc_defined(g)))


def oob_pair(panel, a="MV", b="P-KNN", figures_dir=None):
    """Per gene, out-of-bag MCC and determinate rate of methods a and b
    (gene level), on genes where both MCCs are defined."""
    g = oob_gene_level(figures_dir)
    g = g[g["panel"] == panel]
    w = g.pivot_table(index="gene", columns="method", values=["mcc", "det_rate"], aggfunc="first")
    if (("mcc", a) not in w.columns) or (("mcc", b) not in w.columns):
        return pd.DataFrame()
    out = pd.DataFrame({"a_mcc": w[("mcc", a)], "b_mcc": w[("mcc", b)],
                        "a_det": w[("det_rate", a)], "b_det": w[("det_rate", b)]})
    return out.dropna(subset=["a_mcc", "b_mcc"])


def func_pknn_gw_gene_level(figures_dir=None):
    """pknn/combined_func_plus_pknn_genomewide.csv at the gene level: per
    gene, out-of-bag MCC and determinate rate of joint calibration (mv_oob)
    and of functional evidence plus genome-wide P-KNN (func_pknn_gw), groups
    merged from counts, on genes where both MCCs are defined."""
    figs = Path(figures_dir or config.PAPER_FIGURES_DIR)
    f = figs / "pknn/combined_func_plus_pknn_genomewide.csv"
    if not f.exists():
        return pd.DataFrame()
    x = pd.read_csv(f)
    cols = {}
    for tag in ("mv_oob", "func_pknn_gw"):
        g = gene_level(x.rename(columns={f"{tag}_{c}": c for c in ("TP", "TN", "FP", "FN", "total", "determinate")}))
        g = g[(g["n_pathogenic"] > 0) & (g["n_benign"] > 0)].set_index("gene")
        cols[f"{tag}_mcc"] = g["mcc"].where(mcc_defined(g))
        cols[f"{tag}_det"] = g["det_rate"]
    return pd.DataFrame(cols).dropna(subset=["mv_oob_mcc", "func_pknn_gw_mcc"])
