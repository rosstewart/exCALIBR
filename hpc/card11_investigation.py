"""EXPERIMENT (analysis only; production untouched): why do CARD11's BENTA (GOF)
and CADINS (LOF) variants get little MV evidence, and do more canonical
components or the factorial model (LOF axis x GOF axis) help?

Same 20 production bootstrap resamples for every model:
  canon     canonical refits K = 3..6 (production fitter, 3 restarts, best held-out LL)
  select    factorial selection by 5-fold CV held-out LL (v3); also forces the
            LOF | GOF split so it can be scored even if CV prefers one axis
  fit       factorial fits for the CV-selected and forced 2-axis configs
  score     per model: primary (P/LP vs B/LB) points and each aux sample's own
            points (vs B/LB) at production 5/95 and at 50/50, on BENTA, CADINS,
            P/LP, B/LB

    nice -n 10 taskset -c 0-11 python hpc/card11_investigation.py all --workers 12
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, gzip, io, json, pickle, sys, time, warnings, contextlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_factorial_panel as V1  # noqa: E402
import run_factorial_panel_v3 as V3  # noqa: E402

KEY = "CARD11"
OUT = Path("/data/ross/assay_calibration/multivariate/card11_investigation_101526")
V1.OUT = V3.OUT = OUT
V1.PANEL[KEY] = ("card11", "CARD11")
AUX = [4, 5]
NAMES = {0: "P/LP", 1: "B/LB", 4: "BENTA", 5: "CADINS"}
N_BOOT = V3.N_BOOT
KS = (3, 4, 5, 6)
_load = V1.load


def _load_card11(key, for_factorial=True):
    ms, gene, gs, dataset, _ = _load(key, for_factorial)
    return ms, gene, gs, dataset, AUX


V1.load = _load_card11


def gdir(sub=""):
    d = OUT / sub if sub else OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── factorial: CV selection + forced LOF | GOF ──────────────────────────────
def stage_select(workers):
    from src.assay_calibration.fit_utils.factorial.axis_search_cv import CVSearch, KINDS, PRIORS
    path = OUT / "selection_all.json"
    if path.exists():
        return json.load(open(path))
    ms, *_ = V1.load(KEY)
    job = V3._all_data_job(ms)
    names = [list(ms.dataset_names)[i] for i in job["calibrated_dims"]]
    X, SA = job["train_observations"], job["train_sample_assignments"].astype(bool)
    srch = CVSearch(X, SA, [], [], SA[:, :2].any(1), workers=workers, seeds=V1.SEEDS,
                    fm_kw=dict(tau=V1.TAU_DEFAULT), log=lambda m: print(f"[{KEY}] {m}", flush=True))
    cv_sel = srch.run()
    # forced split: best states/kind/prior for [[LOF], [GOF]] by the same CV score
    split = [[0], [1]]
    srch._score([(split, [a, b], k, p) for a in (2, 3, 4) for b in (2, 3, 4) for k in KINDS for p in PRIORS])
    forced = srch.best(lambda k: len(k[0]) == 2)
    joint = srch.best(lambda k: len(k[0]) == 1)

    def desc(k):
        part, ns, kind, prior = k
        return {"partition": [[names[i] for i in a] for a in part], "states": list(ns), "kind": kind,
                "prior_n0_frac": prior, "mean_cv_ll": srch.mean(k), "canonical_fallback": len(part) == 1}
    out = {"cv_selected": desc(cv_sel), "forced_split": desc(forced), "best_one_axis": desc(joint),
           "scores": srch.table(names)}
    json.dump(out, open(path, "w"), indent=1)
    for k in ("cv_selected", "forced_split", "best_one_axis"):
        print(f"[{KEY}] {k}: {out[k]}", flush=True)
    return out


def _fit_one(args):
    tag, b = args
    warnings.filterwarnings("ignore")
    ck = gdir(f"factorial_{tag}/fits") / f"b{b}.pkl"
    if ck.exists():
        return ck
    sub = f"factorial_{tag}"
    real = V3.gdir
    V3.gdir = lambda key: gdir(sub)
    try:
        return V3._fit_one((KEY, b))
    finally:
        V3.gdir = real


def stage_fit(workers):
    sels = json.load(open(OUT / "selection_all.json"))
    tags = ["forced_split"] + (["cv_selected"] if not sels["cv_selected"]["canonical_fallback"]
                               and sels["cv_selected"] != sels["forced_split"] else [])
    _, _, _, dataset, _ = V1.load(KEY)
    for t in tags:                      # written once, before the workers read it
        json.dump({"selected": {**sels[t], "canonical_fallback": False}},
                  open(gdir(f"factorial_{t}") / "selection.json", "w"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_fit_one, [(t, b) for t in tags for b in range(N_BOOT)]))
    for t in tags:
        with gzip.open(gdir(f"factorial_{t}") / "factorial_results.json.gz", "wt", encoding="utf-8") as fh:
            json.dump({dataset: {str(b): {f"fac_{t}": pickle.load(open(gdir(f"factorial_{t}/fits") / f"b{b}.pkl", "rb"))}
                                 for b in range(N_BOOT)}}, fh)


# ── score ───────────────────────────────────────────────────────────────────
def _dist(pts):
    v, c = np.unique(pts, return_counts=True)
    return {int(a): int(b) for a, b in zip(v, c)}


def _score(js, ms, dataset, configs, pct):
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    from src.assay_calibration.multivariate_analysis.report_gene import confusion_and_metrics
    an = build_gene_set_analysis(ms, "CARD11", str(js), dataset_name=dataset, auxiliary_pathogenic_indices=AUX,
                                 gene_set="card11")
    an.raw_boots = {k: v for k, v in an.raw_boots.items() if int(k) < N_BOOT}
    an.configs = [c for c in configs if c in an.configs]
    kw = dict(V1.RUN_KW)
    if pct is not None:
        kw.update(aux_path_percentile=pct, aux_ben_percentile=100 - pct)
    with contextlib.redirect_stdout(io.StringIO()):
        an.run(**kw)
    sa = np.asarray(ms.sample_assignments, bool)
    out = {}
    for cfg in an.configs:
        r = an.results.get(cfg)
        if r is None:
            continue
        _, met = confusion_and_metrics(an, cfg)
        pts = np.asarray(r["points"])
        row = {"config": cfg, "n_valid": int(r.get("n_valid", 0) or 0), "mcc": met["mcc"],
               "coverage": met["coverage"], "prior": float(np.nanmedian(r["priors"])) if "priors" in r else None,
               "primary": {NAMES[s]: _dist(pts[sa[:, s]]) for s in (0, 1, 4, 5)}}
        for fx, ar in r.get("aux_results", {}).items():
            ap = np.asarray(ar["points"])
            row[f"aux_{NAMES[fx]}"] = {"prior": ar["median_prior"], "on_aux": _dist(ap[sa[:, fx]]),
                                        "on_B/LB": _dist(ap[sa[:, 1]]),
                                        "frac_aux_pos": float((ap[sa[:, fx]] > 0).mean()),
                                        "frac_BLB_pos": float((ap[sa[:, 1]] > 0).mean())}
        out[cfg] = row
    return out


def stage_score():
    ms, _, _, dataset, _ = V1.load(KEY, False)
    models = [("canonical", OUT / KEY / "canonical_results.json.gz", [f"{k}c_unc" for k in KS])]
    for t in ("forced_split", "cv_selected"):
        p = OUT / f"factorial_{t}" / "factorial_results.json.gz"
        if p.exists():
            models.append((f"factorial_{t}", p, [f"fac_{t}"]))
    res = {}
    for label, js, cfgs in models:
        for pct in (5, 50):
            res[f"{label}@aux{pct}"] = _score(js, ms, dataset, cfgs, pct)
    json.dump(res, open(OUT / "scores.json", "w"), indent=1, default=float)
    for m, rows in res.items():
        for cfg, r in rows.items():
            aux = "  ".join(f"{k}: aux+ {r[k]['frac_aux_pos']:.2f} BLB+ {r[k]['frac_BLB_pos']:.2f}"
                            for k in ("aux_BENTA", "aux_CADINS") if k in r)
            print(f"{m:28s} {cfg:22s} mcc={r['mcc']:.3f} cov={r['coverage']:.3f} "
                  f"primary BENTA={r['primary']['BENTA']} CADINS={r['primary']['CADINS']} | {aux}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("canon", "select", "fit", "score", "all"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    t0 = time.time()
    if args.stage in ("canon", "all"):
        V1.stage_canon(KEY, N_BOOT, args.workers, Ks=KS)
    if args.stage in ("select", "all"):
        stage_select(args.workers)
    if args.stage in ("fit", "all"):
        stage_fit(args.workers)
    if args.stage in ("score", "all"):
        stage_score()
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
