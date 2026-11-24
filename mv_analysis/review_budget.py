"""Choosing the joint model's restart and bootstrap budget from the cluster
pilots (hpc/review_jobs.sh).

restarts <run_dir>
    Reads every bootstrap_<seed>_restart_val_ll.json (all restarts' held-out
    log-likelihoods, in the order the greedy search ran them: fit_idx) under
    run_dir. For each model, component count and r, the shortfall of the best
    of the first r restarts from the best of all of them, per variant
    (median and 90th percentile over bootstraps). The minimum r is the first
    whose 90th-percentile shortfall is below TOL_LL for every model.
    -> review/budget/restarts.csv, restarts_minimum.csv, fig_restarts.png

bootstraps <final_dir>
    For each model in a merged bootstrap_results.json.gz with ~1000
    bootstraps, the paper's analysis on random subsets of 100, 200, 300, 500
    and 1000 bootstraps (5 subsets each): the share of model rows whose points
    match the 1000-bootstrap points, the controls' determinate rate and the
    median 5th-95th percentile width of the log LR+. The bootstrap count is
    where these stop changing.
    -> review/budget/bootstraps.csv, fig_bootstraps.png

Usage: python -m mv_analysis.review_budget restarts <run_dir>
       python -m mv_analysis.review_budget bootstraps <final_dir>
"""
import argparse
import contextlib
import gzip
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis import review_common as R

OUT = R.out_dir("budget")
TOL_LL = 1e-3  # per-variant held-out log-likelihood shortfall tolerated
SUBSETS = (100, 150, 200, 300, 500, 1000)  # those up to the run's bootstrap count are used
N_REP = 5


def restarts(run_dir):
    rows = []
    for f in Path(run_dir).rglob("bootstrap_*_restart_val_ll.json"):
        model = f.parent.name
        seed = int(f.name.split("_")[1])
        recs = pd.DataFrame(json.loads(f.read_text()))
        for label, x in recs.groupby("label"):
            v = x.sort_values("fit_idx")["val_ll"].astype(float).to_numpy()
            if not np.isfinite(v).any():
                continue
            best = np.nanmax(v)
            run = np.fmax.accumulate(np.where(np.isfinite(v), v, -np.inf))
            for r, b in enumerate(run, start=1):
                rows.append(dict(model=model, K=label, bootstrap=seed, r=r, shortfall=best - b,
                                 n_restarts=len(v)))
    t = pd.DataFrame(rows)
    s = t.groupby(["model", "K", "r"])["shortfall"].agg(median="median", p90=lambda x: np.quantile(x, 0.9),
                                                       frac_optimal=lambda x: float(np.mean(x <= 0)))
    s = s.reset_index()
    s.to_csv(OUT / "restarts.csv", index=False)
    ok = s[s["p90"] <= TOL_LL].groupby(["model", "K"])["r"].min().rename("min_r").reset_index()
    ok.to_csv(OUT / "restarts_minimum.csv", index=False)
    print(ok.to_string(index=False))
    print("minimum restarts over all models:", int(ok["min_r"].max()) if len(ok) else None)
    _fig(s, "r", "p90", "restarts (in search order)", "90th pct shortfall in held-out LL / variant",
         OUT / "fig_restarts.png", by=["model", "K"])
    return s


def bootstraps(final_dir, seed=0):
    from mv_analysis.mv_cockpit import DEFAULT_RUN_KWARGS
    from mv_analysis.gene_performance_scatter import fast_results_json
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    rj = str(Path(final_dir) / "bootstrap_results.json.gz")
    raw = json.load(gzip.open(rj, "rt"))
    rng = np.random.default_rng(seed)
    rows = []
    for key, m in R.paper_keys():
        if m["dataset_name"] not in raw:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            ms = config.build_ms_for_gene_cluster(m["gene_set"], m["gene"], cluster_idx=m["cluster_idx"],
                                                  results_json=rj)
        d, _ = R.prep(key)
        ev = np.asarray(d["eval_mask"], bool)
        ref = None
        for B in sorted(SUBSETS, reverse=True):
            for rep in range(1 if B >= len(raw[m["dataset_name"]]) else N_REP):
                an = build_gene_set_analysis(ms, m["gene"].lower(), rj, dataset_name=m["dataset_name"],
                                             gene_set=m["gene_set"])
                an.configs = [m["config"]]
                ks = list(an.raw_boots)
                keep = set(rng.choice(ks, min(B, len(ks)), replace=False))
                an.raw_boots = {k: v for k, v in an.raw_boots.items() if k in keep}
                with fast_results_json(rj), contextlib.redirect_stdout(io.StringIO()):
                    an.run(partial_pattern_mode="trust_global", n_jobs=4, **DEFAULT_RUN_KWARGS)
                r = an.results[m["config"]]
                pts = np.asarray(r["points"], float)
                L = np.asarray(r["lr_matrix"], float)
                with np.errstate(all="ignore"):
                    width = np.nanpercentile(L, 95, axis=0) - np.nanpercentile(L, 5, axis=0)
                if ref is None:
                    ref = pts
                rows.append(dict(key=key, B=min(B, len(ks)), rep=rep, agree_with_max=float(np.mean(pts == ref)),
                                 det_rate_controls=float(np.mean(pts[ev] != 0)),
                                 median_width=float(np.nanmedian(width))))
        print(key, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "bootstraps.csv", index=False)
    s = t.groupby(["key", "B"])[["agree_with_max", "det_rate_controls", "median_width"]].mean().reset_index()
    _fig(s, "B", "agree_with_max", "bootstraps", "share of points equal to the largest budget's",
         OUT / "fig_bootstraps.png", by=["key"])
    return t


def _fig(s, x, y, xl, yl, path, by):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, g in s.groupby(by):
        ax.plot(g[x], g[y], "o-", ms=3, lw=0.8, label=" ".join(map(str, k if isinstance(k, tuple) else (k,))))
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.legend(fontsize=5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["restarts", "bootstraps"])
    ap.add_argument("path")
    a = ap.parse_args()
    (restarts if a.mode == "restarts" else bootstraps)(a.path)
