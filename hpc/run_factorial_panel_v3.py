"""EXPERIMENT v3: factorial vs canonical, label-free selection by 5-fold CV.

  select  axes / states per axis / state kind / Gamma prior strength, all by
          5-fold cross-validated held-out log-likelihood on ALL the data
          (fit_utils/factorial/axis_search_cv.py); no MCC anywhere. If one
          axis is selected the gene falls back to canonical (identical
          structure -- nothing for the factorial model to add).
  fit     the selected model on bootstraps 0..N-1 (production resamples)
  score   factorial vs canonical (3c, 4c, and production's own MCC-chosen
          config) on all N bootstraps: production points (5th/95th
          percentile) AND variance-free out-of-bag points (median log LR+
          over the bootstraps that held the variant out), with the OOB
          calibration slope as a diagnostic.

    nice -n 10 taskset -c 0-7 python hpc/run_factorial_panel_v3.py all --key PALB2 --workers 8
    python hpc/run_factorial_panel_v3.py collect
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, gzip, json, pickle, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_factorial_panel as V1  # noqa: E402
from factorial_variance_check import _points, _metrics, _calibration_slope  # noqa: E402

OUT = Path(os.environ.get("FACTORIAL_PANEL_V3_OUT",
                          "/data/ross/assay_calibration/multivariate/factorial_panel_v3_101426"))
N_BOOT = 20


def gdir(key):
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _all_data_job(ms):
    from src.assay_calibration.fit_utils.fit import Fit
    return Fit(ms).generate_fit_jobs([4], num_fits=1, bootstrap=False, check_monotonic=False, master_seed=0,
                                     bootstrap_seed=0, max_em_iters=10, verbose_overlap=False)[0]


# ── select ──────────────────────────────────────────────────────────────────
def stage_select(key, workers):
    from src.assay_calibration.fit_utils.factorial.axis_search_cv import CVSearch
    path = gdir(key) / "selection.json"
    if path.exists():
        return json.load(open(path))
    t0 = time.time()
    ms, *_ = V1.load(key)
    job = _all_data_job(ms)
    names = [list(ms.dataset_names)[i] for i in job["calibrated_dims"]]
    cat_cols, levels, _ = V1.cat_info(job)
    X, SA = job["train_observations"], job["train_sample_assignments"].astype(bool)
    labeled = SA[:, :2].any(1)
    srch = CVSearch(X, SA, cat_cols, levels, labeled, workers=workers, seeds=V1.SEEDS,
                    fm_kw=dict(tau=V1.TAU_DEFAULT), log=lambda m: print(f"[{key}] {m}", flush=True))
    part, ns, kind, prior = srch.run()
    sel = {"partition": [[names[i] for i in a] for a in part], "states": list(ns), "kind": kind,
           "prior_n0_frac": prior, "n_axes": len(part), "canonical_fallback": len(part) == 1,
           "mean_cv_ll": srch.mean((part, ns, kind, prior))}
    json.dump({"selected": sel, "dims": names, "scores": srch.table(names)}, open(path, "w"), indent=1)
    print(f"[{key}] selected {sel['partition']} states={sel['states']} kind={kind} prior={prior} "
          f"cv_ll={sel['mean_cv_ll']:.4f}{'  -> ONE AXIS: canonical fallback' if sel['canonical_fallback'] else ''} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return json.load(open(path))


# ── fit ─────────────────────────────────────────────────────────────────────
def _fit_one(args):
    key, b = args
    warnings.filterwarnings("ignore")
    from src.assay_calibration.fit_utils.factorial import FactorialMixture
    from src.assay_calibration.fit_utils.factorial.export import to_fit_dict
    from src.assay_calibration.fit_utils.factorial.axis_search import _axis_dicts
    ck = gdir(key) / "fits" / f"b{b}.pkl"
    if ck.exists():
        return ck
    ck.parent.mkdir(exist_ok=True)
    sel = json.load(open(gdir(key) / "selection.json"))["selected"]
    ms, *_ = V1.load(key)
    job = V1.job_for(ms, b)
    names = [list(ms.dataset_names)[i] for i in job["calibrated_dims"]]
    part = [[names.index(n) for n in a] for a in sel["partition"]]
    cat_cols, levels, spec = V1.cat_info(job)
    Xtr, SAtr, Xva, SAva = V1.split_from_job(job)
    best = None
    for seed in V1.SEEDS:
        try:
            m = FactorialMixture(_axis_dicts(part, sel["states"], cat_cols, levels, sel["kind"]), tau=V1.TAU_DEFAULT,
                                 seed=seed, prior_n0_frac=sel["prior_n0_frac"]).fit(Xtr, SAtr)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            continue
        va = m.loglik(Xva, SAva) / len(Xva)
        if best is None or va > best[0]:
            best = (va, m)
    va, m = best
    fit = to_fit_dict(m, job["kwargs"]["_scale_mean"], job["kwargs"]["_scale_std"], cat_spec=spec)
    pickle.dump({"bootstrap_seed": b, "num_components": len(m.cells), "fit_idx": 0, "fit": fit, "val_ll": float(va),
                 "calibrated_dims": list(job["calibrated_dims"])}, open(ck, "wb"))
    return ck


def stage_fit(key, workers):
    sel = json.load(open(gdir(key) / "selection.json"))["selected"]
    if sel["canonical_fallback"]:
        print(f"[{key}] one axis selected -> canonical fallback, no factorial fits", flush=True)
        return
    t0 = time.time()
    _, _, _, dataset, _ = V1.load(key)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_fit_one, [(key, b) for b in range(N_BOOT)]))
    with gzip.open(gdir(key) / "factorial_results.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({dataset: {str(b): {"factorial": pickle.load(open(gdir(key) / "fits" / f"b{b}.pkl", "rb"))}
                             for b in range(N_BOOT)}}, fh)
    print(f"[{key}] fit {N_BOOT} bootstraps ({time.time() - t0:.0f}s)", flush=True)


# ── score ───────────────────────────────────────────────────────────────────
def _score_model(key, label, ms, js, dataset, configs, aux, gene, gs, oob):
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    from src.assay_calibration.multivariate_analysis.report_gene import confusion_and_metrics
    an = build_gene_set_analysis(ms, gene, str(js), dataset_name=dataset, auxiliary_pathogenic_indices=aux, gene_set=gs)
    an.raw_boots = {k: v for k, v in an.raw_boots.items() if int(k) < N_BOOT}
    order = [int(k) for k in an.raw_boots]
    an.configs = [c for c in configs if c in an.configs]
    kw = dict(V1.RUN_KW)
    if aux:
        kw.update(aux_path_percentile=50, aux_ben_percentile=50)
    an.run(**kw)
    sa = np.asarray(ms.sample_assignments, bool)
    P, B = sa[:, an.p_idx], sa[:, an.b_idx]
    out = {}
    for cfg in an.configs:
        r = an.results.get(cfg)
        if r is None:
            continue
        L = np.asarray(r["lr_matrix"], float)
        assert L.shape[0] == len(order), f"{key}/{cfg}: {L.shape[0]} valid of {len(order)} bootstraps"
        mask = np.zeros_like(L, bool)
        for i, b in enumerate(order):
            mask[i, list(oob[b])] = True
        with np.errstate(all="ignore"):
            med_oob = np.nanmedian(np.where(mask, L, np.nan), axis=0)
        has = np.isfinite(med_oob)
        tau_p, tau_b = np.asarray(r["tau_p_log"]), np.asarray(r["tau_b_log"])
        lab = (P | B) & has
        out[cfg] = {"key": key, "model": label, "config": cfg,
                    "pct": _metrics(np.asarray(r["points"]), P, B),
                    "oob": _metrics(_points(np.where(has, med_oob, 0.0), tau_p, tau_b), P, B, rows=has),
                    "calib_slope_oob": _calibration_slope(med_oob[lab], P[lab].astype(int)),
                    "lr_spread": float(np.nanmedian((np.nanpercentile(L, 95, axis=0)
                                                     - np.nanpercentile(L, 5, axis=0))[P | B]))}
    return out


def stage_score(key):
    import config as mv_config
    ms_f, gene, gs, dataset, aux = V1.load(key, True)
    ms_c, *_ = V1.load(key, False)
    sel = json.load(open(gdir(key) / "selection.json"))["selected"]
    oob = {b: set(np.asarray(V1.job_for(ms_c, b)["val_variant_indices"]).tolist()) for b in range(N_BOOT)}
    local = V1.OUT / key / "canonical_results.json.gz"
    cjson = local if local.exists() else (V1.TP53_RAWTEMP_CANONICAL if gs == "tp53" else Path(mv_config.PAPER_RESULTS_JSON))
    with gzip.open(cjson, "rt") as fh:
        ccfgs = sorted(next(iter(json.load(fh)[dataset].values())).keys())
    res = {"key": key, "selected": sel,
           "canonical": _score_model(key, "canonical", ms_c, cjson, dataset, ccfgs, aux, gene, gs, oob)}
    res["canonical_production_choice"] = max(res["canonical"], key=lambda c: (round(res["canonical"][c]["pct"]["mcc"], 6),
                                                                             res["canonical"][c]["pct"]["coverage"]))
    if not sel["canonical_fallback"]:
        res["factorial"] = _score_model(key, "factorial", ms_f, gdir(key) / "factorial_results.json.gz", dataset,
                                        ["factorial"], aux, gene, gs, oob)
    json.dump(res, open(gdir(key) / "scores.json", "w"), indent=1, default=float)
    rows = list(res["canonical"].values()) + list(res.get("factorial", {}).values())
    for r in rows:
        tag = "*" if r["model"] == "canonical" and r["config"] == res["canonical_production_choice"] else " "
        print(f"[{key}] {r['model']:9s} {r['config']:9s}{tag} pct: mcc={r['pct']['mcc']:.3f} cov={r['pct']['coverage']:.3f} "
              f"| oob: mcc={r['oob']['mcc']:.3f} cov={r['oob']['coverage']:.3f} auc={r['oob']['auroc']:.3f} "
              f"| slope={r['calib_slope_oob']:.2f} spread={r['lr_spread']:.2f}", flush=True)


def collect():
    import pandas as pd
    rows = []
    for key in V1.PANEL:
        p = OUT / key / "scores.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        c = s["canonical"][s["canonical_production_choice"]]
        f = (s.get("factorial") or {}).get("factorial")
        sel = s["selected"]
        row = {"key": key, "axes": sel["n_axes"], "states": sel["states"], "kind": sel["kind"],
               "prior": sel["prior_n0_frac"], "fallback": sel["canonical_fallback"], "canon_cfg": c["config"],
               "C_pct_mcc": c["pct"]["mcc"], "C_pct_cov": c["pct"]["coverage"],
               "C_oob_mcc": c["oob"]["mcc"], "C_oob_cov": c["oob"]["coverage"], "C_slope": c["calib_slope_oob"]}
        if f:
            row.update({"F_pct_mcc": f["pct"]["mcc"], "F_pct_cov": f["pct"]["coverage"], "F_oob_mcc": f["oob"]["mcc"],
                        "F_oob_cov": f["oob"]["coverage"], "F_slope": f["calib_slope_oob"]})
        rows.append(row)
    if not rows:
        print("no scores yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "panel_comparison_v3.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


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
