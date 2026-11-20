"""Phase 2a: compare bootstrap schemes on real data (production rule, out of bag).

Layout (built by the v2 bundle builder): <root>/<scheme>/<subrun>/tau_0.1/ is
one condition (markers in <root>/<scheme>/<subrun>/, strata layouts and
results in tau_0.1/). Aggregate each condition first:
    python hpc/aggregate_results.py <root>/<scheme>/<subrun>/tau_0.1

    python -m mv_analysis.bootscheme_compare score <condition dir>   # one condition (own process)
    python -m mv_analysis.bootscheme_compare all <root> [--workers 4]
    python -m mv_analysis.bootscheme_compare compare <root> [<root2> ...]

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
    rows = []
    for ds in keys:
        gene = ds.split("_")[0]
        rest = ds[len(gene) + 1:]
        gs = "integrated" if rest.startswith("mv") else rest.split("_")[0]
        cl = int(rest.split("_cluster")[1].split("_")[0]) - 1 if "_cluster" in rest else 0
        with contextlib.redirect_stdout(io.StringIO()):
            ms = config.build_ms_for_gene_cluster(gs, gene, cluster_idx=cl, results_json=rj)
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
                                 points=pts, med_log_lr=med, p5=lo, p95=hi, thr1=lc / 8, withheld=why))
    out = Path(cond) / "bootscheme_oob.csv"
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
    pd.set_option("display.width", 250)
    per = D.groupby(["scheme", "dataset", "config"]).apply(_summ).round(3)
    pooled = D.groupby(["scheme", "config"]).apply(_summ).round(3)
    print(per.to_string()); print(); print(pooled.to_string())
    per.to_csv(Path(root) / "bootscheme_per_dataset.csv")
    pooled.to_csv(Path(root) / "bootscheme_pooled.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "all", "compare"])
    ap.add_argument("path", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd != "compare" and len(a.path) != 1:
        ap.error(f"{a.cmd} takes one path")
    {"score": lambda: score(a.path[0]), "all": lambda: all_(a.path[0], a.workers),
     "compare": lambda: compare(*a.path)}[a.cmd]()
