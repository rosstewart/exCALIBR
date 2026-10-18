"""EXPERIMENT v2: factorial vs canonical with the selection fixes.

Changes from run_factorial_panel.py (v1, kept for reproducibility):
  1. no single-state axes (states per axis in {2..5})
  2. Gamma prior strength is a candidate choice (PRIORS)
  3. final configuration chosen by ACCURACY, like canonical's K: the
     label-free search gives a shortlist (configs within 1 SE of the best
     held-out LL); each shortlisted config x prior is fitted on the bootstraps
     and the one with the best MCC is kept.
     Honest reporting: selection uses bootstraps SELECT_BOOTS (ensemble of
     those fits), and the reported metrics use the disjoint REPORT_BOOTS --
     for BOTH models (canonical picks 3c/4c the same way).

    nice -n 10 taskset -c 0-7 python hpc/run_factorial_panel_v2.py all --key PALB2 --workers 8
    python hpc/run_factorial_panel_v2.py collect
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, gzip, json, pickle, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_factorial_panel as V1  # noqa: E402  (data loading, splits, scoring helpers, v1 canonical baselines)

OUT = Path(os.environ.get("FACTORIAL_PANEL_V2_OUT",
                          "/data/ross/assay_calibration/multivariate/factorial_panel_v2_101326"))
SELECT_BOOTS = tuple(range(0, 10))
REPORT_BOOTS = tuple(range(10, 20))
PRIORS = (0.05, 0.005)
SHORTLIST = 4


def gdir(key):
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── select: label-free shortlist ────────────────────────────────────────────
def stage_select(key, workers):
    from src.assay_calibration.fit_utils.factorial.axis_search import AxisSearch
    path = gdir(key) / "shortlist.json"
    if path.exists():
        return json.load(open(path))
    t0 = time.time()
    ms, *_ = V1.load(key)
    jobs = [V1.job_for(ms, b) for b in range(V1.N_SPLITS)]
    names = [list(ms.dataset_names)[i] for i in jobs[0]["calibrated_dims"]]
    cat_cols, levels, _ = V1.cat_info(jobs[0])
    X_all = np.asarray(ms.scores, float)[:, jobs[0]["calibrated_dims"]]
    labeled = np.asarray(ms.sample_assignments, bool)[:, :2].any(1)
    srch = AxisSearch([V1.split_from_job(j) for j in jobs], cat_cols, levels, X_all, labeled, kind="gaussian",
                      seeds=V1.SEEDS, workers=workers, fm_kw=dict(tau=V1.TAU_DEFAULT),
                      log=lambda m: print(f"[{key}] {m}", flush=True))
    srch.run()
    srch.save(gdir(key) / "axis_selection.json", names)
    cands = []
    for part, ns, kind in srch.shortlist(SHORTLIST):
        for prior in PRIORS:
            cands.append({"partition": [[names[i] for i in a] for a in part], "states": list(ns), "kind": kind,
                          "prior_n0_frac": prior, "mean_val_ll": srch._mean((part, ns, kind))})
    json.dump(cands, open(path, "w"), indent=1)
    print(f"[{key}] shortlist: {len(cands)} candidates ({time.time() - t0:.0f}s)", flush=True)
    for i, c in enumerate(cands):
        print(f"[{key}]   fc{i}: {c['partition']} states={c['states']} kind={c['kind']} prior={c['prior_n0_frac']} "
              f"heldout={c['mean_val_ll']:.4f}", flush=True)
    return cands


# ── fit every shortlisted candidate on all bootstraps ────────────────────────
def _fit_one(args):
    key, ci, b = args
    warnings.filterwarnings("ignore")
    from src.assay_calibration.fit_utils.factorial import FactorialMixture
    from src.assay_calibration.fit_utils.factorial.export import to_fit_dict
    from src.assay_calibration.fit_utils.factorial.axis_search import _axis_dicts
    ck = gdir(key) / "fits" / f"fc{ci}_b{b}.pkl"
    if ck.exists():
        return ck
    ck.parent.mkdir(exist_ok=True)
    c = json.load(open(gdir(key) / "shortlist.json"))[ci]
    ms, *_ = V1.load(key)
    job = V1.job_for(ms, b)
    names = [list(ms.dataset_names)[i] for i in job["calibrated_dims"]]
    part = [[names.index(n) for n in a] for a in c["partition"]]
    cat_cols, levels, spec = V1.cat_info(job)
    Xtr, SAtr, Xva, SAva = V1.split_from_job(job)
    best = None
    for seed in V1.SEEDS:
        try:
            m = FactorialMixture(_axis_dicts(part, c["states"], cat_cols, levels, c["kind"]), tau=V1.TAU_DEFAULT,
                                 seed=seed, prior_n0_frac=c["prior_n0_frac"]).fit(Xtr, SAtr)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            continue
        va = m.loglik(Xva, SAva) / len(Xva)
        if best is None or va > best[0]:
            best = (va, m)
    va, m = best
    fit = to_fit_dict(m, job["kwargs"]["_scale_mean"], job["kwargs"]["_scale_std"], cat_spec=spec)
    entry = {"bootstrap_seed": b, "num_components": len(m.cells), "fit_idx": 0, "fit": fit, "val_ll": float(va),
             "calibrated_dims": list(job["calibrated_dims"])}
    pickle.dump(entry, open(ck, "wb"))
    return ck


def stage_fit(key, workers):
    t0 = time.time()
    cands = json.load(open(gdir(key) / "shortlist.json"))
    _, _, _, dataset, _ = V1.load(key)
    boots = SELECT_BOOTS + REPORT_BOOTS
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_fit_one, [(key, ci, b) for ci in range(len(cands)) for b in boots]))
    with gzip.open(gdir(key) / "factorial_results.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({dataset: {str(b): {f"fc{ci}": pickle.load(open(gdir(key) / "fits" / f"fc{ci}_b{b}.pkl", "rb"))
                                      for ci in range(len(cands))} for b in boots}}, fh)
    print(f"[{key}] fit {len(cands)} candidates x {len(boots)} boots ({time.time() - t0:.0f}s)", flush=True)


# ── score: select on SELECT_BOOTS, report on REPORT_BOOTS ───────────────────
def _score_boots(key, label, ms, results_json, dataset, configs, aux, boots, gene, gs):
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    from src.assay_calibration.multivariate_analysis.report_gene import confusion_and_metrics
    an = build_gene_set_analysis(ms, gene, str(results_json), dataset_name=dataset,
                                 auxiliary_pathogenic_indices=aux, gene_set=gs)
    an.raw_boots = {k: v for k, v in an.raw_boots.items() if int(k) in set(boots)}
    an.configs = [c for c in configs if c in an.configs]
    kw = dict(V1.RUN_KW)
    if aux:
        kw.update(aux_path_percentile=50, aux_ben_percentile=50)
    an.run(**kw)
    sa = np.asarray(ms.sample_assignments, bool)
    out = {}
    for cfg in an.configs:
        if an.results.get(cfg) is None:
            continue
        _, met = confusion_and_metrics(an, cfg)
        pts = np.asarray(an.results[cfg]["points"])
        out[cfg] = {"key": key, "model": label, "config": cfg, "n_boot": len(an.raw_boots),
                    **{k: met[k] for k in ("TP", "FN", "TN", "FP", "coverage", "mcc")},
                    "auroc_points": V1._auroc_points(pts[sa[:, an.p_idx]].astype(float),
                                                     pts[sa[:, an.b_idx]].astype(float))}
    return out


def _pick(rows):
    """Best MCC; ties -> higher coverage."""
    return max(rows.values(), key=lambda r: (round(r["mcc"], 6), r["coverage"]))


def stage_score(key):
    import config as mv_config
    ms_f, gene, gs, dataset, aux = V1.load(key, True)
    ms_c, *_ = V1.load(key, False)
    cands = json.load(open(gdir(key) / "shortlist.json"))
    fjson = gdir(key) / "factorial_results.json.gz"
    fcfgs = [f"fc{i}" for i in range(len(cands))]
    local = V1.OUT / key / "canonical_results.json.gz"
    cjson = local if local.exists() else (V1.TP53_RAWTEMP_CANONICAL if gs == "tp53" else Path(mv_config.PAPER_RESULTS_JSON))
    with gzip.open(cjson, "rt") as fh:
        ccfgs = sorted(next(iter(json.load(fh)[dataset].values())).keys())
    res = {}
    for label, ms, js, cfgs in (("factorial", ms_f, fjson, fcfgs), ("canonical", ms_c, cjson, ccfgs)):
        sel = _score_boots(key, label, ms, js, dataset, cfgs, aux, SELECT_BOOTS, gene, gs)
        rep = _score_boots(key, label, ms, js, dataset, cfgs, aux, REPORT_BOOTS, gene, gs)
        chosen = _pick(sel)["config"]
        res[label] = {"chosen": chosen, "select": sel, "report": rep}
    out = {"key": key, "shortlist": cands, **res}
    json.dump(out, open(gdir(key) / "scores.json", "w"), indent=1, default=float)
    for label in ("factorial", "canonical"):
        c = res[label]["chosen"]
        r = res[label]["report"][c]
        extra = ""
        if label == "factorial":
            i = int(c[2:])
            extra = f" [{cands[i]['partition']} states={cands[i]['states']} {cands[i]['kind']} prior={cands[i]['prior_n0_frac']}]"
        print(f"[{key}] {label:9s} chosen {c:7s} REPORT: TP={r['TP']} FN={r['FN']} TN={r['TN']} FP={r['FP']} "
              f"cov={r['coverage']:.3f} mcc={r['mcc']:.3f}{extra}", flush=True)
    return out


def collect():
    import pandas as pd
    rows = []
    for key in V1.PANEL:
        p = OUT / key / "scores.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        for label in ("factorial", "canonical"):
            c = s[label]["chosen"]
            r = dict(s[label]["report"][c])
            r["select_mcc"] = s[label]["select"][c]["mcc"]
            rows.append(r)
    if not rows:
        print("no scores yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "panel_comparison_v2.csv", index=False)
    pd.set_option("display.width", 220)
    print(df[["key", "model", "config", "TP", "FN", "TN", "FP", "coverage", "mcc", "select_mcc"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("select", "fit", "score", "all", "collect"))
    ap.add_argument("--key", choices=list(V1.PANEL))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    if args.stage == "collect":
        return collect()
    if args.stage in ("select", "all"):
        stage_select(args.key, args.workers)
    if args.stage in ("fit", "all"):
        stage_fit(args.key, args.workers)
    if args.stage in ("score", "all"):
        stage_score(args.key)


if __name__ == "__main__":
    main()
