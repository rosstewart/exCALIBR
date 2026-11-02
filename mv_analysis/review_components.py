"""Component count chosen by held-out log-likelihood vs by MCC, side by side.

The paper picks each model's component count (3 or 4) by MCC on the controls
(config.select_canonical_mv_rows; TP53 functional fixed at 4). Held-out
log-likelihood (val_ll, stored for every bootstrap fit) chooses without the
labels' classification performance: 4 components when the median paired
val_ll[4c] - val_ll[3c] over bootstraps is positive.

  component_selection.csv   run_component_selection's table recomputed on the
                            current fits (written here, not in the paper's
                            directory): per model both counts' MCC and
                            determinate rate, the held-out difference, and
                            the count each rule picks.
  oob_by_rule.csv           out-of-bag metrics per model under each rule. For
                            models where the rules disagree, the other count's
                            out-of-bag matrix is built exactly as
                            production_oob._mv_one builds the canonical one
                            (same bootstrap splits), into components/oob/.
  summary.csv               per evidence type: median out-of-bag MCC (gene
                            level, defined only), determinate rate and errors
                            under each rule, and how often the rules agree.

Usage: python -m mv_analysis.review_components
"""
import contextlib
import io
import shutil
import warnings

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis import review_common as R

OUT = R.out_dir("components")
TABLES = ("functional/functional_all_results_table.csv", "predictor/predictor_results_table.csv",
          "combined/combined_results_table.csv")


def selection_table():
    """run_component_selection on copies of the paper's results tables."""
    from mv_analysis import analyze_mv_pipeline_output as A
    for rel in TABLES:
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f"{config.PAPER_FIGURES_DIR}/{rel}", OUT / rel)
    old = config.PAPER_FIGURES_DIR
    config.PAPER_FIGURES_DIR = str(OUT)
    try:
        A.run_component_selection()
    finally:
        config.PAPER_FIGURES_DIR = old
    return pd.read_csv(OUT / "component_selection.csv")


def mv_matrix(key, cfg):
    """production_oob._mv_one for another config: L, priors, OOB masks of the
    same bootstrap splits -> components/oob/<key>/mv_<cfg>.npz."""
    from mv_analysis.comparison_methods import pknn_runs as RP
    from mv_analysis.gene_performance_scatter import fast_results_json
    from mv_analysis.mv_cockpit import DEFAULT_RUN_KWARGS
    from src.assay_calibration.multivariate_analysis import mv_calibration as mvc
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    f = OUT / "oob" / key / f"mv_{cfg}.npz"
    if f.exists():
        return dict(np.load(f))
    d, meta = RP.load_prep(key)
    gs, gene, cl = meta["gene_set"], meta["gene"], meta["cluster_idx"]
    rj = config.PAPER_RESULTS_JSON
    with contextlib.redirect_stdout(io.StringIO()):
        ms = config.build_ms_for_gene_cluster(gs, gene, cluster_idx=cl, results_json=rj)
    an = build_gene_set_analysis(ms, gene.lower(), rj, dataset_name=meta["dataset_name"], gene_set=gs)
    an.configs = [cfg]
    seeds = [int(k) for k in an.raw_boots]
    valid, orig = [], mvc._bootstrap_job

    def _rec(*a, **kw):
        r = orig(*a, **kw)
        valid.append(r is not None and r[0] is not None)
        return r
    mvc._bootstrap_job = _rec
    try:
        with fast_results_json(rj), contextlib.redirect_stdout(io.StringIO()):
            an.run(partial_pattern_mode="trust_global", n_jobs=1, **DEFAULT_RUN_KWARGS)
    finally:
        mvc._bootstrap_job = orig
    r = an.results[cfg]
    kept = [s for s, v in zip(seeds, valid) if v]
    L = np.asarray(r["lr_matrix"], float)
    assert len(kept) == L.shape[0] == len(r["priors"])
    splits = RP._paper_oob_splits(ms, kept, int(cfg.split("c")[0]))
    oob = np.zeros((len(kept), L.shape[1]), bool)
    for i, b in enumerate(kept):
        oob[i, splits[b]] = True
    f.parent.mkdir(parents=True, exist_ok=True)
    out = dict(L=L, priors=np.asarray(r["priors"], float), seeds=np.asarray(kept), oob=oob,
               point_values=np.asarray(an.point_values), insample_points=np.asarray(r["points"], int))
    np.savez_compressed(f, **out)
    return out


def _oob_metrics(key, mv):
    from mv_analysis.production_oob import _points
    from mv_analysis.review_priors import _metrics
    d, _ = R.prep(key)
    rows = np.flatnonzero(d["eval_mask"])
    L, oob, priors, pv = mv["L"], mv["oob"].astype(bool), mv["priors"], mv["point_values"]
    pts, keep = [], []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for j in rows:
            o = oob[:, j]
            if not o.any():
                continue
            lv = L[o, j]
            pts.append(_points(np.nanpercentile(lv, 5), np.nanpercentile(lv, 95), float(np.median(priors[o])), pv))
            keep.append(j)
    return _metrics(np.asarray(pts, float), np.asarray(d["plp"], bool)[keep])


def run():
    from mv_analysis import baselines as B
    warnings.filterwarnings("ignore")
    sel = selection_table()
    rows = []
    for key, m in R.paper_keys():
        canon = m["config"]
        k_canon = int(canon[0])
        cl = "gene" if m["cluster_idx"] == 0 else f"cluster{m['cluster_idx'] + 1}"
        s = sel[(sel["gene"].astype(str).str.upper() == m["gene"].upper()) & (sel["cluster"] == cl)
                & (sel["panel"] == m["panel"])]
        if m["gene_set"] in ("tp53",):
            s = s[s["gene_set"] == "tp53"]
        k_held = int(s["k_by_heldout"].iloc[0]) if len(s) and np.isfinite(s["k_by_heldout"].iloc[0]) else k_canon
        if config.fixed_mv_config(m["gene"], m["gene_set"]):
            k_held = k_canon  # an author-fixed config (TP53 functional, 4c) stays under both rules
        base = _oob_metrics(key, R.mv_npz(key))
        rows.append(dict(key=key, panel=m["panel"], gene=m["gene"], rule="mcc (paper)", k=k_canon, **base))
        if k_held == k_canon:
            rows.append(dict(key=key, panel=m["panel"], gene=m["gene"], rule="heldout", k=k_held, **base))
        else:
            alt = mv_matrix(key, f"{k_held}c_unc")
            rows.append(dict(key=key, panel=m["panel"], gene=m["gene"], rule="heldout", k=k_held,
                             **_oob_metrics(key, alt)))
        print(key, canon, "->", k_held, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "oob_by_rule.csv", index=False)
    summ = []
    for (panel, rule), x in t.groupby(["panel", "rule"]):
        x = x[(x["TP"] + x["FN"] > 0) & (x["TN"] + x["FP"] > 0)]
        g = B.gene_level(x, by=("panel", "rule", "gene"))
        summ.append(dict(panel=panel, rule=rule, n_genes=len(g), median_mcc=g.loc[B.mcc_defined(g), "mcc"].median(),
                         median_det_rate=(g["determinate"] / g["total"]).median(), errors=int((x.FP + x.FN).sum())))
    agree = t.pivot_table(index="key", columns="rule", values="k").dropna()
    summ = pd.DataFrame(summ)
    summ["models_agreeing"] = f"{int((agree['heldout'] == agree['mcc (paper)']).sum())}/{len(agree)}"
    summ.to_csv(OUT / "summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
