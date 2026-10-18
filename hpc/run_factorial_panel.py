"""EXPERIMENT: factorial (axis-tied) mixture vs canonical, matched, on a gene panel.

Per gene (``--key``), three resumable stages:

  select  automatic axes + states per axis + state kind
          (fit_utils/factorial/axis_search.py) on 5 OOB splits that are
          production's own bootstrap resamples (generate_fit_jobs,
          master_seed=0, bootstrap_seed=0..4); cached as axis_selection.json
  fit     the selected structure on bootstraps 0..N-1 (the same resamples the
          canonical fits used), 2 seeds each, best held-out LL kept; one
          config per pathogenic-sample tau (single-mechanism candidates:
          tau_path in TAU_PATH); checkpointed per (config, bootstrap)
  score   factorial configs and the canonical configs, restricted to the same
          N bootstraps, through the unchanged MVCalibrationAnalysis; metrics
          on POINTS (TP/FN/TN/FP, coverage, MCC, point-AUROC)

    nice -n 10 taskset -c 0-7 python hpc/run_factorial_panel.py all --key PALB2 --n-boot 20 --workers 8
    python hpc/run_factorial_panel.py collect          # panel_comparison.csv + figure

Everything here is opt-in experimentation; no production code path changes.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import gzip
import json
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p_ in (REPO, REPO / "mv_analysis", REPO / "tests" / "cfusn_simulations"):
    sys.path.insert(0, str(p_))

OUT = Path(os.environ.get("FACTORIAL_PANEL_OUT", "/data/ross/assay_calibration/multivariate/factorial_panel_101226"))
TP53_RAWTEMP_CANONICAL = Path(os.environ.get(
    "TP53_RAWTEMP_CANONICAL_JSON",
    "/tmp/claude-9996/-home-rcstewart-exCALIBR/72bc95ba-f2d2-4da2-8132-de5a9da67a97/scratchpad/factorial_od/"
    "canonical_rawtemp_tp53_bootstrap_results.json.gz"))

PANEL = {
    "TP53": ("tp53", "TP53"),
    "PALB2": ("integrated", "PALB2"), "BRCA1": ("integrated", "BRCA1"), "KCNH2": ("integrated", "KCNH2"),
    "LDLR": ("integrated", "LDLR"), "GCK": ("integrated", "GCK"), "CHEK2": ("integrated", "CHEK2"),
    "F9": ("integrated", "F9"),
    "shp2": ("labelseq", "shp2"), "braf": ("labelseq", "braf"), "kras": ("labelseq", "kras"),
    "BRCA2_combined": ("combined", "BRCA2"), "F9_combined": ("combined", "F9"), "TP53_combined": ("combined", "TP53"),
}
TAU_DEFAULT = 5.0
TAU_PATH = (None, 50.0, 1000.0)            # None = default tau for every sample
PATHOGENIC_NAMES = ("P/LP", "RPV")
N_SPLITS = 5
SEEDS = (0, 1)
RUN_KW = dict(path_percentile=5, min_valid_boots=1, reestimate_marginal_weights=False,
              enforce_marginal_monotonicity=False, liberal_marginal_monotonicity=False,
              partial_pattern_mode="trust_global", n_jobs=8)


# ── data ─────────────────────────────────────────────────────────────────────
def load(key, for_factorial=True):
    """(ms, gene, gene_set, canonical dataset name, aux pathogenic fixed indices)."""
    warnings.filterwarnings("ignore")
    import config as mv_config
    gs, gene = PANEL[key]
    if gs == "tp53" and for_factorial:
        import sim_kawoligo_categorical_tp53 as KC
        ms = KC.build_ms("categorical")                 # KawOligo as a categorical dim
    else:
        ms = mv_config.build_ms_for_gene_cluster(gs, gene, results_json=mv_config.PAPER_RESULTS_JSON)
    dataset = ("TP53_tp53_mv" if gs == "tp53"
               else mv_config.canonical_dataset_name(gene, gs, results_json=mv_config.PAPER_RESULTS_JSON))
    aux = [4] if gs == "tp53" else None
    return ms, gene, gs, dataset, aux


def job_for(ms, b):
    from src.assay_calibration.fit_utils.fit import Fit
    return Fit(ms).generate_fit_jobs([4], num_fits=1, bootstrap=True, bootstrap_seed=b, check_monotonic=False,
                                     master_seed=0, max_em_iters=10, verbose_overlap=False)[0]


def split_from_job(job):
    return (job["train_observations"], job["train_sample_assignments"].astype(bool),
            job["val_observations"], job["val_sample_assignments"].astype(bool))


def cat_info(job):
    spec = job["kwargs"].get("cat_spec")
    if not spec:
        return [], [], None
    return list(spec["positions"]), [len(l) for l in spec["level_names"]], spec


def gdir(key):
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── stage 1: select ─────────────────────────────────────────────────────────
def stage_select(key, workers):
    from src.assay_calibration.fit_utils.factorial.axis_search import AxisSearch
    path = gdir(key) / "axis_selection.json"
    if path.exists():
        print(f"[{key}] selection cached", flush=True)
        return json.load(open(path))
    t0 = time.time()
    ms, *_ = load(key)
    jobs = [job_for(ms, b) for b in range(N_SPLITS)]
    names = [list(ms.dataset_names)[i] for i in jobs[0]["calibrated_dims"]]
    cat_cols, levels, _ = cat_info(jobs[0])
    X_all = np.asarray(ms.scores, float)[:, jobs[0]["calibrated_dims"]]
    sa = np.asarray(ms.sample_assignments, bool)
    labeled = sa[:, :2].any(1)
    srch = AxisSearch([split_from_job(j) for j in jobs], cat_cols, levels, X_all, labeled, kind="gaussian",
                      seeds=SEEDS, workers=workers, fm_kw=dict(tau=TAU_DEFAULT),
                      log=lambda m: print(f"[{key}] {m}", flush=True))
    part, ns, kind = srch.run()
    srch.save(path, names)
    print(f"[{key}] selected {[[names[i] for i in a] for a in part]} states={list(ns)} kind={kind} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return json.load(open(path))


# ── stage 2: fit ────────────────────────────────────────────────────────────
def _config_name(tau_path):
    return "factorial" if tau_path is None else f"factorial_taup{int(tau_path)}"


def _fit_one(args):
    key, b, tau_path = args
    warnings.filterwarnings("ignore")
    from src.assay_calibration.fit_utils.factorial import FactorialMixture
    from src.assay_calibration.fit_utils.factorial.export import to_fit_dict
    from src.assay_calibration.fit_utils.factorial.axis_search import _axis_dicts
    ck = gdir(key) / "fits" / f"{_config_name(tau_path)}_b{b}.pkl"
    if ck.exists():
        return ck
    ck.parent.mkdir(exist_ok=True)
    sel = json.load(open(gdir(key) / "axis_selection.json"))
    ms, *_ = load(key)
    job = job_for(ms, b)
    names = [list(ms.dataset_names)[i] for i in job["calibrated_dims"]]
    part = [[names.index(n) for n in a] for a in sel["selected"]["partition"]]
    ns, kind = sel["selected"]["states"], sel["selected"]["kind"]
    cat_cols, levels, spec = cat_info(job)
    Xtr, SAtr, Xva, SAva = split_from_job(job)
    snames = list(ms.sample_names)
    tbs = ({i: tau_path for i, n in enumerate(snames) if n in PATHOGENIC_NAMES and i < SAtr.shape[1]}
           if tau_path is not None else None)
    best = None
    for seed in SEEDS:
        m = FactorialMixture(_axis_dicts(part, ns, cat_cols, levels, kind), tau=TAU_DEFAULT, seed=seed,
                             tau_by_sample=tbs).fit(Xtr, SAtr)
        va = m.loglik(Xva, SAva) / len(Xva)
        if best is None or va > best[0]:
            best = (va, m)
    va, m = best
    d = np.diff(m.objectives) / np.maximum(np.abs(m.objectives[:-1]), 1)
    fit = to_fit_dict(m, job["kwargs"]["_scale_mean"], job["kwargs"]["_scale_std"], cat_spec=spec)
    fit["monotone_min_rel_step"] = float(d.min()) if len(d) else 0.0
    fit["overshoot_stopped"] = m.overshoot
    entry = {"bootstrap_seed": b, "num_components": len(m.cells), "fit_idx": 0, "fit": fit,
             "val_ll": float(va), "calibrated_dims": list(job["calibrated_dims"])}
    pickle.dump(entry, open(ck, "wb"))
    return ck


def stage_fit(key, n_boot, workers, taus=TAU_PATH):
    t0 = time.time()
    _, _, _, dataset, _ = load(key)
    todo = [(key, b, t) for t in taus for b in range(n_boot)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_fit_one, todo))
    for t in taus:
        cfg = _config_name(t)
        boots = {}
        for b in range(n_boot):
            boots[str(b)] = {cfg: pickle.load(open(gdir(key) / "fits" / f"{cfg}_b{b}.pkl", "rb"))}
        with gzip.open(gdir(key) / f"{cfg}_results.json.gz", "wt", encoding="utf-8") as fh:
            json.dump({dataset: boots}, fh)
    print(f"[{key}] fit {len(todo)} done ({time.time() - t0:.0f}s)", flush=True)


# ── stage 2b: canonical refit (when the stored production fit is stale) ─────
def _canon_one(args):
    key, K, b, f = args
    warnings.filterwarnings("ignore")
    from src.assay_calibration.fit_utils.fit import Fit
    ck = gdir(key) / "canon_fits" / f"{K}c_b{b}_f{f}.pkl"
    if ck.exists():
        return ck
    ck.parent.mkdir(exist_ok=True)
    ms, *_ = load(key, for_factorial=False)
    jobs = Fit(ms).generate_fit_jobs([K], num_fits=3, bootstrap=True, bootstrap_seed=b, check_monotonic=False,
                                     master_seed=0, max_em_iters=3000, verbose_overlap=False)
    res = Fit.execute_fit_job([j for j in jobs if j["fit_idx"] == f][0])
    pickle.dump(res, open(ck, "wb"))
    return ck


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items() if k not in ("kmeans", "initial_params")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, set):
        return sorted(x)
    return x


def stage_canon(key, n_boot, workers, Ks=(3, 4)):
    """Production canonical fits (3 restarts, best held-out LL) on the same
    bootstraps, for genes whose stored paper fit no longer matches the data."""
    t0 = time.time()
    _, _, _, dataset, _ = load(key, for_factorial=False)
    todo = [(key, K, b, f) for K in Ks for b in range(n_boot) for f in range(3)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_canon_one, todo))
    boots = {}
    for b in range(n_boot):
        for K in Ks:
            cands = [pickle.load(open(gdir(key) / "canon_fits" / f"{K}c_b{b}_f{f}.pkl", "rb")) for f in range(3)]
            cands = [c for c in cands if c and np.isfinite(c["val_ll"])]
            if cands:
                boots.setdefault(str(b), {})[f"{K}c_unc"] = _jsonable(max(cands, key=lambda c: c["val_ll"]))
    with gzip.open(gdir(key) / "canonical_results.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({dataset: boots}, fh)
    print(f"[{key}] canonical refit {len(todo)} fits ({time.time() - t0:.0f}s)", flush=True)


# ── stage 3: score ──────────────────────────────────────────────────────────
def _auroc_points(pos, neg):
    from scipy.stats import rankdata
    if not len(pos) or not len(neg):
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _score_json(key, label, ms, results_json, dataset, configs, aux, n_boot, gene, gs):
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    from src.assay_calibration.multivariate_analysis.report_gene import confusion_and_metrics
    rows = []
    an = build_gene_set_analysis(ms, gene, str(results_json), dataset_name=dataset,
                                 auxiliary_pathogenic_indices=aux, gene_set=gs)
    keep = [k for k in sorted(an.raw_boots, key=int) if int(k) < n_boot]
    an.raw_boots = {k: an.raw_boots[k] for k in keep}
    an.configs = [c for c in configs if c in an.configs]
    kw = dict(RUN_KW)
    if aux:
        kw.update(aux_path_percentile=50, aux_ben_percentile=50)
    an.run(**kw)
    sa = np.asarray(ms.sample_assignments, bool)
    for cfg in an.configs:
        if an.results.get(cfg) is None:
            continue
        cm, met = confusion_and_metrics(an, cfg)
        pts = np.asarray(an.results[cfg]["points"])
        row = {"key": key, "model": label, "config": cfg, "n_boot": len(keep),
               **{k: met[k] for k in ("TP", "FN", "TN", "FP", "coverage", "mcc")},
               "auroc_points": _auroc_points(pts[sa[:, an.p_idx]].astype(float), pts[sa[:, an.b_idx]].astype(float)),
               "n_P": int(sa[:, an.p_idx].sum()), "n_B": int(sa[:, an.b_idx].sum())}
        rows.append(row)
    return rows


def stage_score(key, n_boot):
    import config as mv_config
    rows = []
    ms_f, gene, gs, dataset, aux = load(key, for_factorial=True)
    ms_c, *_ = load(key, for_factorial=False)
    for t in TAU_PATH:
        cfg = _config_name(t)
        p = gdir(key) / f"{cfg}_results.json.gz"
        if p.exists():
            rows += _score_json(key, "factorial", ms_f, p, dataset, [cfg], aux, n_boot, gene, gs)
    local = gdir(key) / "canonical_results.json.gz"
    canon_json = (local if local.exists() else
                  TP53_RAWTEMP_CANONICAL if gs == "tp53" else Path(mv_config.PAPER_RESULTS_JSON))
    try:
        with gzip.open(canon_json, "rt") as fh:
            canon_cfgs = sorted(next(iter(json.load(fh)[dataset].values())).keys())
        rows += _score_json(key, "canonical" + ("_refit" if canon_json == local else ""), ms_c, canon_json,
                            dataset, canon_cfgs, aux, n_boot, gene, gs)
    except ValueError as e:                       # stale stored fit: keep the factorial rows, flag it
        print(f"[{key}] canonical baseline unusable ({e}); run the 'canon' stage", flush=True)
    json.dump(rows, open(gdir(key) / "scores.json", "w"), indent=1, default=float)
    for r in rows:
        print(f"[{key}] {r['model']:9s} {r['config']:18s} TP={r['TP']} FN={r['FN']} TN={r['TN']} FP={r['FP']} "
              f"cov={r['coverage']:.3f} mcc={r['mcc']:.3f} auroc_pts={r['auroc_points']:.3f}", flush=True)
    return rows


def collect():
    import pandas as pd
    rows = []
    for key in PANEL:
        p = OUT / key / "scores.json"
        if p.exists():
            rows += json.load(open(p))
    if not rows:
        print("no scores yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "panel_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("select", "fit", "canon", "score", "all", "collect"))
    ap.add_argument("--key", choices=list(PANEL))
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    if args.stage == "collect":
        return collect()
    if args.stage in ("select", "all"):
        stage_select(args.key, args.workers)
    if args.stage in ("fit", "all"):
        stage_fit(args.key, args.n_boot, args.workers)
    if args.stage == "canon":
        stage_canon(args.key, args.n_boot, args.workers)
    if args.stage in ("score", "all", "canon"):
        stage_score(args.key, args.n_boot)


if __name__ == "__main__":
    main()
