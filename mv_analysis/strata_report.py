"""Consequence-stratified mixture report (plan Phase 5) for one stratified run.

    MV_PAPER_FITS_DIR=<run>/tau_0.1 python -m mv_analysis.strata_report [--out DIR] [--arms ...]

Writes to <PAPER_FIGURES_DIR>/strata/ (or --out):

  strata_layout.csv     per dataset and stratum: rows, membership, consequence,
                        splice flag, and the composite each arm puts it in
                        ("" = auxiliary in that arm)
  arm_evidence.csv      per dataset x arm x control group: P/LP and B/LB counts,
                        in-sample and out-of-bag determinate share, errors, MCC.
                        Control groups: all evaluated controls, then by
                        consequence (missense / PTV / synonymous / other), and
                        high-SpliceAI B/LB on their own. Every arm is evaluated
                        on the gene set's main-arm controls (identical sets).
  single_modality.csv   combined gene set only: out-of-bag evidence for the
                        main arm's missense controls by modality (both,
                        functional only, predictor only) -- the held-out test
                        of controls with one modality.
  per_variant.csv       the rows behind the two tables above.

Out-of-bag points use the production rule (5th / 95th percentile of the
out-of-bag log LR at the out-of-bag median prior) with splits read from the
run's job pickles; without pickles only in-sample columns are filled.
"""
import argparse
import contextlib
import gzip
import io
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ARMS_BY_SET = {"combined": ["missense"],
               "default": ["pooled", "missense", "pooled_no_splice_blb", "pooled_no_splice_syn_blb",
                           "pooled_benign_pool"]}


def _parse_key(ds):
    gene = ds.split("_")[0]
    rest = ds[len(gene) + 1:]
    gs = ("fgfr" if "_fgfr_" in ds else "predictor" if rest.startswith("predictors")
          else "integrated" if rest.startswith(("mv", "cluster")) else rest.split("_")[0])
    cl = int(rest.split("_cluster")[1].split("_")[0]) - 1 if "_cluster" in rest else 0
    return gene, gs, cl


def _mcc(tp, tn, fp, fn):
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / d if d else np.nan


def _score(gene, gs, cl, ds, arm, rj):
    """Build, stratify (arm) and run one dataset; returns (ms, an, results)."""
    from mv_analysis import config
    from mv_analysis.gene_performance_scatter import fast_results_json
    from mv_analysis.mv_cockpit import DEFAULT_RUN_KWARGS
    from src.assay_calibration.multivariate_analysis import mv_calibration as mvc
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    os.environ["MV_STRATA_ARM"] = arm
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
    kept = {cfg: [s for s, ok in zip(seeds, valid[i * per_cfg:(i + 1) * per_cfg]) if ok]
            for i, cfg in enumerate(an.configs)}
    return ms, an, kept


def _oob_points(L, priors, oob_row, pvals):
    from mv_analysis.production_oob import _log_c
    if not oob_row.any():
        return np.nan
    lv = L[oob_row]
    lc = _log_c(round(float(np.median(priors[oob_row])), 4))
    lo, hi = np.nanpercentile(lv, 5), np.nanpercentile(lv, 95)
    pts = 0
    for pv in pvals:
        if lo >= pv / 8 * lc:
            pts = pv
    for pv in pvals:
        if hi <= -pv / 8 * lc:
            pts = -pv
    return pts


def dataset_rows(ds, arms, rj, fits_dir):
    from mv_analysis.comparison_methods.pknn_runs import job_pickle_oob_splits
    from src.assay_calibration.multivariate_data.predictors import PREDICTOR_DATASET_NAMES
    from src.assay_calibration.fit_utils.strata import ROLES
    gene, gs, cl = _parse_key(ds)
    layout, rows = [], []
    for arm in arms:
        ms, an, kept = _score(gene, gs, cl, ds, arm, rj)
        st = an.strata
        if st is None:
            raise RuntimeError(f"{ds}: not a stratified run (no strata)")
        if arm == arms[0]:
            comps = {a: st.composition(a) for a in arms}
            for c, (name, (memb, csq, splice)) in enumerate(zip(st.names, st.keys)):
                rec = dict(dataset=ds, gene=gene, gene_set=gs, stratum=name, n=int(st.counts[c]),
                           consequence=csq, splice=bool(splice),
                           membership="+".join(r for r, b in zip(ROLES, memb[:4]) if b) or "none")
                for a in arms:
                    rec[f"arm_{a}"] = "+".join(ROLES[r] for r in range(4) if comps[a][r, c] > 0)
                layout.append(rec)
        X = np.asarray(ms.scores, float)
        pred = [i for i, n in enumerate(ms.dataset_names) if n in set(PREDICTOR_DATASET_NAMES.values())]
        func = [i for i in range(X.shape[1]) if i not in pred]
        has_p = ~np.isnan(X[:, pred]).all(axis=1) if pred else np.zeros(len(X), bool)
        has_f = ~np.isnan(X[:, func]).all(axis=1) if func else np.zeros(len(X), bool)
        modality = np.where(has_p & has_f, "both", np.where(has_f, "functional only", "predictor only"))
        ev = (np.asarray(ms._eval_plp) | np.asarray(ms._eval_blb)) & (has_p | has_f)
        rs = st.row_stratum
        for cfg in an.configs:
            r = an.results.get(cfg)
            if r is None:
                continue
            L = np.asarray(r["lr_matrix"], float)
            oob = None
            try:
                splits = job_pickle_oob_splits(ds, kept[cfg], fits_dir=fits_dir)
                oob = np.zeros(L.shape, bool)
                for i, b in enumerate(kept[cfg]):
                    oob[i, splits[b]] = True
            except (FileNotFoundError, KeyError):
                pass
            priors = np.asarray(r["priors"], float)
            for j in np.flatnonzero(ev):
                k = rs[j]
                csq, splice = (st.keys[k][1], st.keys[k][2]) if k >= 0 else ("other", False)
                rows.append(dict(dataset=ds, gene=gene, gene_set=gs, arm=arm, config=cfg, row=int(j),
                                 label=int(bool(ms._eval_plp[j])), consequence=csq, splice=bool(splice),
                                 modality=modality[j], insample=int(r["points"][j]),
                                 oob=_oob_points(L[:, j], priors, oob[:, j], an.point_values)
                                 if oob is not None else np.nan))
    return layout, rows


def _summ(d, col):
    x = d[d[col].notna()]
    p = x[col].to_numpy(); y = x.label.to_numpy()
    tp = int(((p > 0) & (y == 1)).sum()); tn = int(((p < 0) & (y == 0)).sum())
    fp = int(((p > 0) & (y == 0)).sum()); fn = int(((p < 0) & (y == 1)).sum())
    return {f"{col}_n": len(x), f"{col}_determinate": float((p != 0).mean()) if len(x) else np.nan,
            f"{col}_errors": fp + fn, f"{col}_mcc": _mcc(tp, tn, fp, fn)}


def summarise(V):
    out = []
    groups = [("all", lambda d: d)] + [(c, (lambda c: lambda d: d[d.consequence == c])(c))
                                       for c in ("missense", "PTV", "synonymous", "other")]
    groups.append(("B/LB high SpliceAI", lambda d: d[(d.label == 0) & d.splice]))
    for (ds, arm, cfg), d in V.groupby(["dataset", "arm", "config"]):
        for gname, f in groups:
            s = f(d)
            if not len(s):
                continue
            out.append(dict(dataset=ds, arm=arm, config=cfg, controls=gname, n_P=int((s.label == 1).sum()),
                            n_B=int((s.label == 0).sum()), **_summ(s, "insample"), **_summ(s, "oob")))
    A = pd.DataFrame(out)
    sm = []
    C = V[(V.gene_set == "combined") & (V.consequence == "missense")]
    for (ds, arm, cfg, mod), d in C.groupby(["dataset", "arm", "config", "modality"]):
        sm.append(dict(dataset=ds, arm=arm, config=cfg, modality=mod, n_P=int((d.label == 1).sum()),
                       n_B=int((d.label == 0).sum()), **_summ(d, "oob")))
    return A, pd.DataFrame(sm)


def main():
    warnings.filterwarnings("ignore")
    from mv_analysis import config
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    if not config.STRATA:
        raise SystemExit(f"{config.PAPER_FITS_DIR} is not a stratified run (no strata.txt marker)")
    rj = config.PAPER_RESULTS_JSON
    out = Path(args.out or Path(config.PAPER_FIGURES_DIR) / "strata")
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(rj, "rt") as f:
        keys = sorted(json.load(f).keys())
    L_all, V_all = [], []
    for ds in keys:
        if args.datasets and ds not in args.datasets:
            continue
        gs = _parse_key(ds)[1]
        arms = ARMS_BY_SET["combined" if gs == "combined" else "default"]
        if gs == "predictor":
            arms = ["pooled"]
        try:
            lay, rows = dataset_rows(ds, arms, rj, config.PAPER_FITS_DIR)
        except Exception as e:
            print(f"{ds}: {e}")
            continue
        L_all += lay
        V_all += rows
        print(f"{ds}: {len(lay)} strata, {len(rows)} control rows over {len(arms)} arm(s)")
    os.environ.pop("MV_STRATA_ARM", None)
    V = pd.DataFrame(V_all)
    pd.DataFrame(L_all).to_csv(out / "strata_layout.csv", index=False)
    V.to_csv(out / "per_variant.csv", index=False)
    if len(V):
        A, S = summarise(V)
        A.to_csv(out / "arm_evidence.csv", index=False)
        S.to_csv(out / "single_modality.csv", index=False)
        pd.set_option("display.width", 220)
        print(A[A.controls == "all"].round(3).to_string(index=False))
        if len(S):
            print(S.round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
