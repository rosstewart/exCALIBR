"""Phase 2a: compare bootstrap schemes on real data (production rule, out of bag).

Layout (built by the v2 bundle builder): <root>/<scheme>/<subrun>/tau_0.1/ is
one condition (markers in <root>/<scheme>/<subrun>/, strata layouts and
results in tau_0.1/). Aggregate each condition first:
    python hpc/aggregate_results.py <root>/<scheme>/<subrun>/tau_0.1

    python -m mv_analysis.bootscheme_compare score <condition dir>   # one condition (own process)
    python -m mv_analysis.bootscheme_compare all <root> [--workers 4]
    python -m mv_analysis.bootscheme_compare compare <root> [<root2> ...]
    python -m mv_analysis.bootscheme_compare annotate <condition dir>   # add consequence columns to an old CSV

``compare`` takes several bundle roots, e.g. the Phase 2a bundle and the
standard-model companion (jobs_v2_standard_*, scheme label "standard"), and
writes its tables to the first root.

``score`` writes <condition>/bootscheme_oob.csv: one row per evaluated control
and config with its out-of-bag points, the median and 5th/95th percentile of
its out-of-bag log LR, the thresholds, and why it is withheld when it is
("median" = the median log LR does not reach +-1 point; "spread" = it does,
but the 5th/95th percentile does not). ``compare`` summarises per scheme x
dataset and pooled per scheme: MCC, DOR, determinate share, errors, median
spread, and the withheld split; the decision rule is in the plan (highest
determinate share whose MCC and errors are no worse than the best scheme's).
"""
import argparse
import contextlib
import io
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _ds_parse(ds):
    """(gene, gene set, cluster index) of a fitted dataset name."""
    gene = ds.split("_")[0]
    rest = ds[len(gene) + 1:]
    gs = ("fgfr" if "_fgfr_" in ds else "predictor" if rest.startswith("predictors")
          else "integrated" if rest.startswith(("mv", "cluster")) else rest.split("_")[0])
    cl = int(rest.split("_cluster")[1].split("_")[0]) - 1 if "_cluster" in rest else 0
    return gene, gs, cl


_DF_CACHE = {}


def _row_classes(ms, gene):
    """Per ms row: consequence class (missense / PTV / synonymous / other) and
    splice flag (SpliceAI > 0.2, only possible in splice-detecting assays),
    from the same function the strata are built with."""
    import pandas as _pd
    from src.assay_calibration.fit_utils.strata import row_annotations
    df = None
    if getattr(ms, "_row_atoms", None) is not None and getattr(ms, "row_meta", None) is None:
        from src.assay_calibration.multivariate_data.combined import DEFAULT_INTEGRATED_DATAFRAME as P
        if P not in _DF_CACHE:
            _DF_CACHE[P] = _pd.read_csv(P, sep="\t", low_memory=False)
        df = _DF_CACHE[P]
    return row_annotations(ms, df, gene=gene)


def annotate(cond):
    """Add consequence / splice columns to an existing <cond>/bootscheme_oob.csv
    (rows were scored before score() wrote them); no rescoring."""
    os.environ["MV_PAPER_FITS_DIR"] = str(cond)
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mv_analysis import config
    f = Path(cond) / "bootscheme_oob.csv"
    d = pd.read_csv(f)
    cons, spl = np.empty(len(d), object), np.zeros(len(d), bool)
    for ds, idx in d.groupby("dataset").groups.items():
        gene, gs, cl = _ds_parse(ds)
        with contextlib.redirect_stdout(io.StringIO()):
            ms = config.build_ms_for_gene_cluster(gs, gene, cluster_idx=cl, results_json=config.PAPER_RESULTS_JSON)
        a = _row_classes(ms, gene)
        r = d.loc[idx, "row"].to_numpy()
        cons[idx] = a["consequence"].to_numpy()[r]
        spl[idx] = a["splice"].to_numpy()[r]
    d["consequence"], d["splice"] = cons, spl
    d.to_csv(f, index=False)
    print(f"{cond}: annotated {len(d)} rows")


def score(cond):
    os.environ["MV_PAPER_FITS_DIR"] = str(cond)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mv_analysis import config, report
    from mv_analysis.comparison_methods.pknn_runs import job_pickle_oob_splits
    from mv_analysis.gene_performance_scatter import fast_results_json
    from mv_analysis.mv_cockpit import DEFAULT_RUN_KWARGS
    from mv_analysis.production_oob import _log_c
    from src.assay_calibration.multivariate_analysis import mv_calibration as mvc
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    import gzip
    import json
    rj = config.PAPER_RESULTS_JSON
    with gzip.open(rj, "rt") as f:
        keys = sorted(json.load(f).keys())
    # MV_SCORE_TAG=<tag>: write bootscheme_oob_<tag>.csv (e.g. a consequence-
    # specific arm scored with MV_STRATA_ARM=<arm> MV_STRATA_EVAL_ARM=pooled).
    # MV_PRIOR_FROM_CSV=<csv>: take each control's prior threshold from that
    # earlier scoring (e.g. the pooled arm's bootscheme_oob.csv) instead of
    # this arm's own prior.
    tag = os.environ.get("MV_SCORE_TAG")
    prior_thr = None
    if os.environ.get("MV_PRIOR_FROM_CSV"):
        pc = pd.read_csv(os.environ["MV_PRIOR_FROM_CSV"])
        prior_thr = {(r.dataset, r.config, int(r.row)): float(r.thr1) for r in pc.itertuples()}
    rows = []
    for ds in keys:
        gene, gs, cl = _ds_parse(ds)
        with contextlib.redirect_stdout(io.StringIO()):
            ms = config.build_ms_for_gene_cluster(gs, gene, cluster_idx=cl, results_json=rj)
            ann = _row_classes(ms, gene)
            an = build_gene_set_analysis(ms, gene.lower(), rj, dataset_name=ds, gene_set=gs)
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
        per_cfg = len(seeds)
        for ci, cfg in enumerate(an.configs):
            r = an.results.get(cfg)
            if r is None:
                continue
            v = valid[ci * per_cfg:(ci + 1) * per_cfg]
            kept = [s for s, ok in zip(seeds, v) if ok]
            L = np.asarray(r["lr_matrix"], float)
            if len(kept) != L.shape[0]:
                print(f"{ds}/{cfg}: kept {len(kept)} vs L {L.shape[0]}; skipped")
                continue
            splits = job_pickle_oob_splits(ds, kept, fits_dir=cond)
            oob = np.zeros(L.shape, bool)
            for i, b in enumerate(kept):
                oob[i, splits[b]] = True
            priors = np.asarray(r["priors"], float)
            ev, lab = report._eval_labels(ms, an.p_idx, an.b_idx, gene_set=gs)
            for j, y in zip(np.flatnonzero(ev), lab):
                o = oob[:, j]
                if not o.any():
                    continue
                lv = L[o, j]
                lc = _log_c(round(float(np.median(priors[o])), 4))
                if prior_thr is not None:
                    if (ds, cfg, int(j)) not in prior_thr:
                        continue
                    lc = 8 * prior_thr[(ds, cfg, int(j))]
                lo, hi, med = np.nanpercentile(lv, 5), np.nanpercentile(lv, 95), np.nanmedian(lv)
                pts = 0
                for pv in an.point_values:
                    if lo >= pv / 8 * lc:
                        pts = pv
                for pv in an.point_values:
                    if hi <= -pv / 8 * lc:
                        pts = -pv
                why = ""
                if pts == 0:
                    why = "spread" if abs(med) >= lc / 8 else "median"
                rows.append(dict(dataset=ds, gene_set=gs, config=cfg, row=int(j), label=int(y), n_oob=int(o.sum()),
                                 points=pts, med_log_lr=med, p5=lo, p95=hi, thr1=lc / 8, withheld=why,
                                 consequence=ann["consequence"].iat[j], splice=bool(ann["splice"].iat[j])))
    out = Path(cond) / (f"bootscheme_oob_{tag}.csv" if tag else "bootscheme_oob.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{cond}: {len(rows)} rows -> {out}")


def _conditions(root):
    return sorted(p.parent for p in Path(root).glob("*/*/tau_0.1/bootstrap_results.json.gz"))


def all_(root, workers):
    from concurrent.futures import ThreadPoolExecutor
    conds = _conditions(root)

    def go(c):
        return subprocess.run(["nice", "-n", "19", sys.executable, "-m", "mv_analysis.bootscheme_compare", "score",
                               str(c)], cwd=str(Path(__file__).resolve().parents[1])).returncode
    with ThreadPoolExecutor(workers) as ex:
        codes = list(ex.map(go, conds))
    print(dict(zip(map(str, conds), codes)))


def _summ(d):
    tp = ((d.points > 0) & (d.label == 1)).sum(); tn = ((d.points < 0) & (d.label == 0)).sum()
    fp = ((d.points > 0) & (d.label == 0)).sum(); fn = ((d.points < 0) & (d.label == 1)).sum()
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return pd.Series(dict(n=len(d), mcc=(tp * tn - fp * fn) / den if den else np.nan,
                          dor=(tp * tn) / (fp * fn) if fp * fn else np.inf, determinate=(d.points != 0).mean(),
                          errors=int(fp + fn), spread=(d.p95 - d.p5).median(),
                          withheld_median=(d.withheld == "median").mean(),
                          withheld_spread=(d.withheld == "spread").mean()))


def _summ_conf(d):
    """_summ plus the full confusion table: P/LP and B/LB controls by call
    (pathogenic evidence / none / benign evidence)."""
    s = _summ(d)
    P, B = d[d.label == 1], d[d.label == 0]
    s["n_plp"], s["n_blb"] = len(P), len(B)
    s["plp_path"], s["plp_none"], s["plp_ben"] = (P.points > 0).sum(), (P.points == 0).sum(), (P.points < 0).sum()
    s["blb_path"], s["blb_none"], s["blb_ben"] = (B.points > 0).sum(), (B.points == 0).sum(), (B.points < 0).sum()
    return s


def compare(*roots):
    root = roots[0]
    frames = []
    for c in [c for r in roots for c in _conditions(r)]:
        f = c / "bootscheme_oob.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["scheme"] = c.parent.parent.name
            frames.append(d)
    D = pd.concat(frames, ignore_index=True)
    # Score every scheme on the same controls: rows evaluated, with the same
    # label, in every scheme of the dataset (strata runs exclude rows labelled
    # both P/LP and B/LB, which the standard model evaluates as P/LP).
    n_sch = D.groupby("dataset").scheme.nunique()
    g = D.groupby(["dataset", "row"]).agg(n=("scheme", "nunique"), nl=("label", "nunique")).reset_index()
    g = g[(g.n == g.dataset.map(n_sch)) & (g.nl == 1)]
    keep = D.set_index(["dataset", "row"]).index.isin(g.set_index(["dataset", "row"]).index)
    dropped = D[~keep].groupby(["dataset", "scheme"]).row.nunique()
    if len(dropped):
        print("controls not common to every scheme (excluded):")
        print(dropped.to_string())
    D = D[keep]
    pd.set_option("display.width", 250)
    per = D.groupby(["scheme", "dataset", "config"]).apply(_summ).round(3)
    pooled = D.groupby(["scheme", "config"]).apply(_summ).round(3)
    print(per.to_string()); print(); print(pooled.to_string())
    per.to_csv(Path(root) / "bootscheme_per_dataset.csv")
    pooled.to_csv(Path(root) / "bootscheme_pooled.csv")
    if "consequence" in D.columns and D["consequence"].notna().all():
        D = D.assign(cclass=D["consequence"].astype(str) + np.where(D["splice"].astype(bool), " (splice)", ""))
        byc = D.groupby(["scheme", "config", "cclass"]).apply(_summ_conf).round(3)
        byc_ds = D.groupby(["scheme", "config", "dataset", "cclass"]).apply(_summ_conf).round(3)
        print(); print(byc.to_string())
        byc.to_csv(Path(root) / "bootscheme_by_consequence.csv")
        byc_ds.to_csv(Path(root) / "bootscheme_by_consequence_per_dataset.csv")
    else:
        print("no consequence columns (run `annotate` on each condition): consequence breakdown skipped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "annotate", "all", "compare"])
    ap.add_argument("path", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd != "compare" and len(a.path) != 1:
        ap.error(f"{a.cmd} takes one path")
    {"score": lambda: score(a.path[0]), "annotate": lambda: annotate(a.path[0]), "all": lambda: all_(a.path[0], a.workers),
     "compare": lambda: compare(*a.path)}[a.cmd]()
