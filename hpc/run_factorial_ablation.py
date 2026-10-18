"""EXPERIMENT: which factorial setting makes it less accurate than canonical on
some genes? Ablation ladder toward canonical's configuration, same 20
bootstraps (production resamples), scored like hpc/run_factorial_panel.py.

  B  one axis, K = canonical's best config, cfusn (q=2) states
  C  B + 10x weaker Gamma prior (prior_n0_frac 0.005)
  D  C + no weight pseudo-count (alpha_w ~ 0)
  E  D + production-style restart cycling (gaussian / msn / cfusn, best held-out kept)

    nice -n 10 taskset -c 0-9 python hpc/run_factorial_ablation.py --key F9 --workers 10
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, gzip, json, pickle, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_factorial_panel as R  # noqa: E402

VARIANTS = {
    "B": dict(kinds=("cfusn",), prior_n0_frac=0.05, alpha_w=0.05),
    "C": dict(kinds=("cfusn",), prior_n0_frac=0.005, alpha_w=0.05),
    "D": dict(kinds=("cfusn",), prior_n0_frac=0.005, alpha_w=1e-8),
    "E": dict(kinds=("gaussian", "msn", "cfusn"), prior_n0_frac=0.005, alpha_w=1e-8),
}


def _fit(args):
    key, var, K, b = args
    warnings.filterwarnings("ignore")
    from src.assay_calibration.fit_utils.factorial import FactorialMixture
    from src.assay_calibration.fit_utils.factorial.export import to_fit_dict
    from src.assay_calibration.fit_utils.factorial.axis_search import _axis_dicts
    ck = R.gdir(key) / "ablation" / f"{var}_b{b}.pkl"
    if ck.exists():
        return ck
    ck.parent.mkdir(exist_ok=True)
    ms, *_ = R.load(key)
    job = R.job_for(ms, b)
    cat_cols, levels, spec = R.cat_info(job)
    Xtr, SAtr, Xva, SAva = R.split_from_job(job)
    p = Xtr.shape[1]
    cfg = VARIANTS[var]
    best = None
    for kind in cfg["kinds"]:
        for seed in R.SEEDS:
            try:
                m = FactorialMixture(_axis_dicts([list(range(p))], [K], cat_cols, levels, kind), tau=R.TAU_DEFAULT,
                                     seed=seed, prior_n0_frac=cfg["prior_n0_frac"],
                                     alpha_w=cfg["alpha_w"]).fit(Xtr, SAtr)
            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                continue
            va = m.loglik(Xva, SAva) / len(Xva)
            if best is None or va > best[0]:
                best = (va, m)
    va, m = best
    fit = to_fit_dict(m, job["kwargs"]["_scale_mean"], job["kwargs"]["_scale_std"], cat_spec=spec)
    entry = {"bootstrap_seed": b, "num_components": K, "fit_idx": 0, "fit": fit, "val_ll": float(va),
             "calibrated_dims": list(job["calibrated_dims"])}
    pickle.dump(entry, open(ck, "wb"))
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, choices=list(R.PANEL))
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    key = args.key
    scores = json.load(open(R.OUT / key / "scores.json"))
    canon = max((r for r in scores if r["model"].startswith("canonical")), key=lambda r: r["mcc"])
    K = int(canon["config"].split("c")[0])
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(_fit, [(key, v, K, b) for v in VARIANTS for b in range(args.n_boot)]))
    ms_f, gene, gs, dataset, aux = R.load(key, True)
    rows = []
    for v in VARIANTS:
        boots = {str(b): {f"abl{v}": pickle.load(open(R.gdir(key) / "ablation" / f"{v}_b{b}.pkl", "rb"))}
                 for b in range(args.n_boot)}
        pj = R.gdir(key) / "ablation" / f"abl{v}_results.json.gz"
        with gzip.open(pj, "wt", encoding="utf-8") as fh:
            json.dump({dataset: boots}, fh)
        rows += R._score_json(key, f"factorial_abl{v}", ms_f, pj, dataset, [f"abl{v}"], aux, args.n_boot, gene, gs)
    json.dump(rows, open(R.gdir(key) / "ablation" / "scores.json", "w"), indent=1, default=float)
    for r in [x for x in scores if x["config"] in ("factorial", canon["config"])] + rows:
        print(f"[{key}] {r['model']:20s} {r['config']:10s} TP={r['TP']} FN={r['FN']} TN={r['TN']} FP={r['FP']} "
              f"cov={r['coverage']:.3f} mcc={r['mcc']:.3f}", flush=True)
    print(f"[{key}] done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
