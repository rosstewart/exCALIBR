"""Semi-synthetic test of how well classifier probabilities are calibrated,
used directly ("raw") or after ExCALIBR calibration (cross-fitted, as in the
paper's classifier baselines), against a known true likelihood ratio.

World, per dataset (a paper evaluation key with production out-of-bag fits):
  - Variants are the dataset's real score vectors (all model rows: controls,
    population and synonymous variants), so the dimensions, scales and the
    pattern of missing measurements are real.
  - The true log LR+ of a row is the median over the joint model's bootstrap
    fits of its log LR+ (production_oob mv.npz "L"); it is a fixed function
    of the row's scores. With prior pi (median bootstrap prior), each row's
    class is drawn as Y ~ Bernoulli(pi LR / (pi LR + 1 - pi)). In this world
    P(Y=1 | x) is exactly that posterior, so the true LR+ relative to the
    world's prior pi_w = mean posterior is LR * odds(pi) / odds(pi_w) -- exact
    whether or not the joint model fits the real data.
  - Controls: as many P/LP and B/LB as the real dataset has, drawn uniformly
    within each class (ascertainment that does not depend on the scores, so it
    leaves the LR unchanged but sets the training class balance, as ClinVar
    does). Population sample: a uniform draw of the real population sample's
    size (classes in world proportion). Synonymous sample (if the dataset has
    one): a uniform draw of B rows of the real synonymous size.
  - Test rows: every row not used as a control.

Methods (mv_analysis/comparison_methods/supervised.py), for each classifier:
  raw       logit(p) - logit(training P/LP fraction) (fit_predict), points at
            the TRUE prior (an oracle the raw method cannot have)
  ExCALIBR  cross-fitted ExCALIBR calibration (fit_excalibr_crossfit), points
            at ExCALIBR's own estimated prior
  local     local likelihood ratio on a held-out third (fit_calibrate_predict),
            points at the true prior
Points use the production threshold rule (production_oob._points) on a single
estimate. Truth: the true log LR+ at the world prior.

Outputs: <PAPER_PRODUCTION_OOB_DIR>/classifier_calibration_sim/
  runs/<key>__rep<r>.npz      per test row: true log LR+, each method's log LR+
  summary_by_key.csv, summary.csv   (stage "summarize")

Stages:
  python -m mv_analysis.classifier_calibration_sim tasks --task-file F [--reps 20] [--keys ...]
  python -m mv_analysis.classifier_calibration_sim run --task-file F --task-index i [--tasks-per-index n] [--workers n]
  python -m mv_analysis.classifier_calibration_sim run [--reps 2] [--keys ...] [--workers n]   (local)
  python -m mv_analysis.classifier_calibration_sim summarize
Cluster: hpc/prepare_classifier_sim.py, hpc/submit_classifier_sim.sh.
"""
import argparse
import contextlib
import io
import json
import os
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis.comparison_methods import pknn_runs as RP
from mv_analysis.production_oob import OUT as OOB_DIR, _log_c, _points, _single_threaded, keys_with_both_classes

SIM_DIR = Path(os.environ.get("MV_CLS_SIM_DIR", Path(config.PAPER_PRODUCTION_OOB_DIR) / "classifier_calibration_sim"))
RUNS = SIM_DIR / "runs"
KINDS = ("raw", "ExCALIBR", "local")


def _logit(p):
    return np.log(p / (1 - p))


def world(key, rep):
    """The simulated dataset for (key, rep): scores, labels, samples, truth."""
    d, meta = RP.load_prep(key)
    mv = np.load(OOB_DIR / key / "mv.npz")
    X = d["X"]
    keep = ~np.isnan(X).all(1)
    X = X[keep]
    sa = np.asarray(d["sa"], bool)[keep]
    names = [str(n) for n in d["sample_names"]]
    from mv_analysis.comparison_methods.supervised import sample_columns
    col = sample_columns(names)
    llr0 = np.nanmedian(mv["L"], 0)[keep]
    pi = float(np.median(mv["priors"]))
    post = 1 / (1 + np.exp(-(np.clip(llr0, -50, 50) + _logit(pi))))
    pi_w = float(post.mean())
    llr_true = llr0 + _logit(pi) - _logit(pi_w)
    rng = np.random.default_rng([rep, zlib.crc32(key.encode())])
    y = rng.random(len(post)) < post
    n_p = int(d["plp"][d["labelled"]].sum())
    n_b = int(d["blb"][d["labelled"]].sum())
    ip, ib = np.flatnonzero(y), np.flatnonzero(~y)
    n_p, n_b = min(n_p, len(ip) // 2), min(n_b, len(ib) // 2)  # leave test rows of each class
    ctrl = np.zeros(len(y), bool)
    ctrl[rng.choice(ip, n_p, replace=False)] = True
    ctrl[rng.choice(ib, n_b, replace=False)] = True
    S = np.zeros((len(y), 4), bool)
    S[:, 0] = ctrl & y
    S[:, 1] = ctrl & ~y
    n_pop = int(sa[:, col["population"]].sum()) if col["population"] is not None else 0
    S[rng.choice(len(y), min(n_pop, len(y)), replace=False), 2] = True
    n_syn = int(sa[:, col["synonymous"]].sum()) if col["synonymous"] is not None else 0
    if n_syn:
        S[rng.choice(ib, min(n_syn, len(ib)), replace=False), 3] = True
    sample_names = ["P/LP", "B/LB", "population"] + (["synonymous"] if n_syn else [])
    return dict(X=X, y=y.astype(int), S=S[:, :len(sample_names)], sample_names=sample_names, ctrl=ctrl,
                llr_true=llr_true, prior_true=pi_w, prior_model=pi, n_p=n_p, n_b=n_b)


def run_one(key, rep):
    """Fit every classifier under each calibration on world(key, rep) ->
    runs/<key>__rep<rep>.npz (test rows only)."""
    _single_threaded()
    os.environ.setdefault("TQDM_DISABLE", "1")
    warnings.filterwarnings("ignore")
    from mv_analysis.comparison_methods import supervised
    f = RUNS / f"{key}__rep{rep}.npz"
    if f.exists():
        return key, rep
    w = world(key, rep)
    X, y, ctrl = w["X"], w["y"], w["ctrl"]
    te = ~ctrl
    out = dict(llr_true=w["llr_true"][te].astype(np.float32), y=y[te].astype(np.int8),
               prior_true=np.float64(w["prior_true"]), prior_model=np.float64(w["prior_model"]),
               n_p=w["n_p"], n_b=w["n_b"])
    for m in supervised.MODELS:
        quiet = contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())
        for kind in KINDS:
            try:
                with quiet[0], quiet[1]:
                    if kind == "raw":
                        r = supervised.fit_predict(m, X[ctrl], y[ctrl], X[te], seed=rep)
                    elif kind == "local":
                        r = supervised.fit_calibrate_predict(m, X[ctrl], y[ctrl], X[te], seed=rep)
                    else:
                        r = supervised.fit_excalibr_crossfit(m, X, y, w["S"], w["sample_names"], ctrl, te,
                                                             exclude=np.zeros(len(y), bool), seed=rep)
            except Exception as e:  # recorded, never silently dropped
                r = f"error: {type(e).__name__}: {e}"[:300]
            if r is None:
                r = "too few controls"
            if isinstance(r, str):
                out[f"fail_{kind}_{m}"] = np.array(r)
            elif kind == "ExCALIBR":
                out[f"llr_{kind}_{m}"], out[f"prior_{kind}_{m}"] = r[0].astype(np.float32), np.float64(r[1])
            else:
                out[f"llr_{kind}_{m}"] = np.asarray(r, np.float32)
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, f)
    return key, rep


def tasks(keys=None, reps=20):
    return [[k, r] for k in keys_with_both_classes(keys) if (OOB_DIR / k / "mv.npz").exists()
            for r in range(reps) if not (RUNS / f"{k}__rep{r}.npz").exists()]


def _pts(llr, prior, pvals):
    return np.array([_points(v, v, prior, pvals) for v in llr], float)


def summarize():
    """Per (key, classifier, calibration): accuracy of log LR+ and of points
    against the truth on test rows, pooled over replicates."""
    from mv_analysis.comparison_methods import supervised
    c = pd.read_csv(Path(config.PAPER_PRODUCTION_OOB_DIR) / "production_oob_comparison.csv")
    panel = dict(c[["key", "panel"]].drop_duplicates().itertuples(index=False))
    pvals = None
    recs, fails = [], []
    for f in sorted(RUNS.glob("*.npz")):
        key, rep = f.stem.rsplit("__rep", 1)
        if pvals is None:
            pvals = np.load(OOB_DIR / key / "mv.npz")["point_values"]
        z = np.load(f)
        t, pt = z["llr_true"].astype(float), float(z["prior_true"])
        tp = _pts(t, pt, pvals)
        for m in supervised.MODELS:
            for kind in KINDS:
                if f"fail_{kind}_{m}" in z.files:
                    fails.append(dict(key=key, rep=int(rep), model=m, kind=kind, reason=str(z[f"fail_{kind}_{m}"])))
                    continue
                e = z[f"llr_{kind}_{m}"].astype(float)
                prior = float(z[f"prior_{kind}_{m}"]) if kind == "ExCALIBR" else pt
                ep = _pts(e, prior, pvals)
                ok = np.isfinite(e) & np.isfinite(t)
                # continuous points: log LR+ in units of one point at each side's prior, capped at
                # +-8 (beyond the very strong threshold every log LR+ gives the same evidence)
                ct = np.clip(t[ok] / (_log_c(round(pt, 4)) / 8), -8, 8)
                ce = np.clip(e[ok] / (_log_c(round(prior, 4)) / 8), -8, 8)
                slope = float(np.polyfit(ce, ct, 1)[0]) if ce.std() > 0 else np.nan
                a, b = ep[ok], tp[ok]
                # evidence stronger than the truth supports: wrong sign, or same sign and more points
                over = (a != 0) & ((np.sign(a) != np.sign(b)) | (np.abs(a) > np.abs(b)))
                under = (b != 0) & (np.sign(a) != -np.sign(b)) & (np.abs(a) < np.abs(b))
                recs.append(dict(key=key, panel=panel.get(key, key.split("__")[0]), rep=int(rep), model=m, kind=kind,
                                 n_test=int(ok.sum()), n_p=int(z["n_p"]), n_b=int(z["n_b"]),
                                 prior_true=pt, prior_used=prior,
                                 mae_points=float(np.mean(np.abs(ce - ct))),
                                 bias_points=float(np.mean(ce - ct)),
                                 slope_true_on_est=slope,
                                 pts_exact=float(np.mean(a == b)),
                                 overstated=float(np.mean(over)),
                                 overstated_2plus=float(np.mean(over & (np.abs(a - b) >= 2))),
                                 understated=float(np.mean(under)),
                                 wrong_sign=float(np.mean(a * b < 0)),
                                 path_overclaim=float(np.mean((a > 0) & (a > b))),
                                 benign_overclaim=float(np.mean((a < 0) & (a < b)))))
    r = pd.DataFrame(recs)
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fails).to_csv(SIM_DIR / "failures.csv", index=False)
    by_key = r.groupby(["panel", "key", "model", "kind"]).mean(numeric_only=True).reset_index()
    by_key.to_csv(SIM_DIR / "summary_by_key.csv", index=False)
    cols = ["mae_points", "bias_points", "slope_true_on_est", "pts_exact", "overstated", "overstated_2plus",
            "understated", "wrong_sign", "path_overclaim", "benign_overclaim"]
    s = by_key.groupby(["panel", "model", "kind"])[cols].median().reset_index()
    s.to_csv(SIM_DIR / "summary.csv", index=False)
    print(s.round(3).to_string())
    print(f"{len(fails)} failed fits; wrote {SIM_DIR}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tasks", "run", "summarize"])
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--task-file")
    ap.add_argument("--task-index", type=int)
    ap.add_argument("--tasks-per-index", type=int, default=1)
    a = ap.parse_args()
    if a.stage == "tasks":
        t = tasks(a.keys, a.reps)
        Path(a.task_file).write_text(json.dumps(t))
        print(f"{len(t)} tasks -> {a.task_file}")
        return
    if a.stage == "summarize":
        summarize()
        return
    if a.task_file:
        t = json.loads(Path(a.task_file).read_text())
        if a.task_index is not None:
            t = t[a.task_index * a.tasks_per_index:(a.task_index + 1) * a.tasks_per_index]
    else:
        t = tasks(a.keys, a.reps)
    print(f"run: {len(t)} tasks", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max(1, min(a.workers, len(t)))) as ex:
        for k, r in ex.map(run_one, *zip(*t)) if t else []:
            print(f"  done {k} rep {r}", flush=True)


if __name__ == "__main__":
    main()
