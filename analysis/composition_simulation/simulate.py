"""Control-composition simulation (E3): which P/LP / PTV sample choices give correct
evidence for MISSENSE variants?

Generative model (one LoF assay, low score = abnormal; unit-variance normals):
    N  functionally normal          mean 0
    H  hypomorphic                  mean -kappa_h
    L  complete loss of function    mean -kappa
    T  truncating (PTV) LoF         mean -kappa_t   (kappa_t = kappa: PTVs look like missense LoF)
Missense benign = N. Missense pathogenic = (1-h) L + h H.
PTV pathogenic = (1-e) T + e H (e = NMD-escape-like fraction scoring like a hypomorph).
Synonymous = N. Population (gnomAD missense) = pi * missense pathogenic + (1-pi) N.
Assayed PTVs (the extra sample) = all PTVs, including the PTV P/LP.

Samples: P/LP = n_mis missense pathogenic + n_ptv_plp PTV pathogenic; B/LB missense
benign; gnomAD; synonymous; optional PTV sample.

Arms (each fit with the production Fit.run and calibrated with the production
process_component_fits):
    all       P/LP = all P/LP (missense + PTV)
    mis       P/LP = missense P/LP only (refit)
    reweight  "all" fits; P/LP weights re-estimated post-fit from each bootstrap's
              missense P/LP training rows (components fixed)
    mis_ptv   P/LP = missense only, plus a fit-only PTV sample (all assayed PTVs)

Truth: the missense-specific lr+(x) = f_misP(x) / f_N(x) and the prior pi.
Metrics on a large draw of missense population variants ("VUS"):
    overclaim: share of variants whose conservative (5th-percentile) estimated
               log lr+ exceeds the true log lr+ (pathogenic side, true lr+ > 1)
    log-LR error at the 5th percentile, prior error, and points vs the points the
    true lr+ and true prior would give (same mapping for both).

Run a pilot:  python -m analysis.composition_simulation.simulate --pilot
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

warnings.filterwarnings("ignore")
OUT = Path("/data/ross/assay_calibration/composition_simulation")
NAMES = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "gnomAD", "Synonymous", "ptv"]
POINTS = [1, 2, 3, 4, 5, 6, 7, 8]

SCENARIOS = {
    # PTVs distributed exactly like missense pathogenic: only count effects expected
    "null": dict(kappa=3.0, kappa_h=1.5, kappa_t=3.0, h=0.3, e=0.3),
    # PTVs more extreme than missense LoF, no hypomorph/escape difference
    "ptv_extreme": dict(kappa=3.0, kappa_h=1.5, kappa_t=4.5, h=0.3, e=0.3),
    # missense has hypomorphs, PTVs do not (all complete LoF)
    "mis_hypomorph": dict(kappa=3.0, kappa_h=1.5, kappa_t=3.0, h=0.4, e=0.0),
    # PTVs include many NMD-escape (hypomorph-like) variants, missense mostly complete LoF
    "ptv_escape": dict(kappa=3.0, kappa_h=1.5, kappa_t=3.0, h=0.1, e=0.5),
}


def mis_path_pdf(x, p):
    return (1 - p["h"]) * sps.norm.pdf(x, -p["kappa"]) + p["h"] * sps.norm.pdf(x, -p["kappa_h"])


def draw_mis_path(rng, n, p):
    hyp = rng.random(n) < p["h"]
    return np.where(hyp, -p["kappa_h"], -p["kappa"]) + rng.standard_normal(n)


def draw_ptv(rng, n, p):
    esc = rng.random(n) < p["e"]
    return np.where(esc, -p["kappa_h"], -p["kappa_t"]) + rng.standard_normal(n)


def make_data(rng, p, n_mis, n_ptv_plp, n_blb, n_gnomad, n_syn, n_ptv_unlab, pi):
    mis_p = draw_mis_path(rng, n_mis, p)
    ptv_p = draw_ptv(rng, n_ptv_plp, p)
    blb = rng.standard_normal(n_blb)
    g_path = rng.random(n_gnomad) < pi
    gnomad = np.where(g_path, draw_mis_path(rng, n_gnomad, p), rng.standard_normal(n_gnomad))
    syn = rng.standard_normal(n_syn)
    ptv_u = draw_ptv(rng, n_ptv_unlab, p)
    return dict(mis_p=mis_p, ptv_p=ptv_p, blb=blb, gnomad=gnomad, syn=syn, ptv_u=ptv_u)


def build_scoreset(d, arm):
    """BasicScoreset rows + (rows x 5) membership for an arm. PTV P/LP rows are also members
    of the PTV sample in mis_ptv (overlap resolved by the production one-hot)."""
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    blocks = [("mis_p", [0]), ("ptv_p", [0] if arm in ("all", "reweight") else ([4] if arm == "mis_ptv" else None)),
              ("blb", [1]), ("gnomad", [2]), ("syn", [3]), ("ptv_u", [4] if arm == "mis_ptv" else None)]
    scores, sa, kind = [], [], []
    for key, cols in blocks:
        if cols is None or len(d[key]) == 0:
            continue
        m = np.zeros((len(d[key]), 5), bool)
        m[:, cols] = True
        scores.append(d[key]); sa.append(m); kind += [key] * len(d[key])
    scores, sa = np.concatenate(scores), np.vstack(sa)
    keep_cols = sa.any(axis=0)
    names = [n for n, k in zip(NAMES, keep_cols) if k]
    return BasicScoreset(scores, sa[:, keep_cols], sample_names=names), np.array(kind)


def fit_bootstraps(ss, K, B, R):
    from src.assay_calibration.fit_utils.fit import Fit
    fitter = Fit(ss)
    fits, seeds = [], []
    for b in range(B):
        try:
            models, best, _ = fitter.run([K], bootstrap=True, bootstrap_seed=b, num_fits=R, master_seed=0,
                                         check_monotonic=True, core_limit=1, verbose=False)
            fits.append({"fit": models[best]}); seeds.append(b)
        except Exception:
            continue
    return fits, np.array(seeds)


def reweight_to_missense(ss, kind, fits, seeds):
    from analysis.missense_sensitivity.matched_subsample import training_rows
    from analysis.missense_sensitivity.reweight_plp_missense import _em_weights
    sa, sc = np.asarray(ss.sample_assignments), np.asarray(ss.scores)
    is_mis = kind == "mis_p"
    out = []
    for f, b in zip(fits, seeds):
        rows, onehot = training_rows(sa, sc, int(b))
        rp = rows[onehot[rows, 0] & is_mis[rows]]
        f = copy.deepcopy(f)
        if len(rp):
            f["fit"]["weights"][0] = _em_weights(sc[rp], f["fit"]["component_params"],
                                                 f["fit"]["weights"][0]).tolist()
        out.append(f)
    return out


def calibrate(ss, fits, seeds, K):
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.visualize import process_component_fits
    cfg = PipelineConfig(dataset_csv="", dataset_name="sim", output_dir="/tmp", components=[K],
                         benign_method="avg", use_median_prior=True, use_2c_equation=False,
                         point_values=POINTS, n_jobs=1)
    lg = logging.getLogger("compsim"); lg.setLevel(logging.ERROR)
    return process_component_fits(fits=fits, scoreset=ss, config=cfg, n_c=K, logger=lg, bootstrap_seeds=seeds)


def truth_points(log_lr, prior):
    from src.assay_calibration.fit_utils.fit import thresholds_from_prior
    lrP, lrB, _ = thresholds_from_prior(prior, POINTS)
    pts = np.zeros(len(log_lr), int)
    for pv, tp, tb in zip(POINTS, np.log(lrP), np.log(lrB)):
        pts[log_lr >= tp] = pv
        pts[log_lr <= tb] = -pv
    return pts


def evaluate(cal, p, pi, x_vus):
    from src.assay_calibration.fit_utils.fit import assign_points
    true_llr = np.log(mis_path_pdf(x_vus, p) / sps.norm.pdf(x_vus))
    sr = np.asarray(cal["score_range"], float)
    llr = np.asarray(cal["log_lr_plus"], float)
    p5 = np.interp(x_vus, sr, np.nanpercentile(llr, 5, axis=0))
    p95 = np.interp(x_vus, sr, np.nanpercentile(llr, 95, axis=0))
    pos, neg = true_llr > 0, true_llr < 0
    pts = assign_points(x_vus, {int(k): v for k, v in cal["point_ranges"].items()})
    tpts = truth_points(true_llr, pi)
    return dict(prior=float(cal["prior"]), prior_err=float(cal["prior"] - pi),
                overclaim_path=float((p5[pos] > true_llr[pos]).mean()),
                overclaim_benign=float((p95[neg] < true_llr[neg]).mean()),
                llr5_err_path=float(np.mean(p5[pos] - true_llr[pos])),
                pts_over=float((pts > np.maximum(tpts, 0)).mean()),
                pts_wrong_dir=float(((pts > 0) & (tpts < 0)).mean() + ((pts < 0) & (tpts > 0)).mean()),
                pts_mae=float(np.mean(np.abs(pts - tpts))),
                path_det=float((pts > 0).mean()), true_path_det=float((tpts > 0).mean()))


def run_cell(scen, n_mis, rep, K=2, B=30, R=4, pi=0.2, n_blb=30, n_gnomad=300, n_syn=200, n_ptv_unlab=150,
             ptv_share=0.5):
    p = SCENARIOS[scen]
    rng = np.random.default_rng([list(SCENARIOS).index(scen), n_mis, rep])
    n_ptv_plp = int(round(n_mis * ptv_share / (1 - ptv_share)))
    d = make_data(rng, p, n_mis, n_ptv_plp, n_blb, n_gnomad, n_syn, n_ptv_unlab, pi)
    g = rng.random(20000) < pi
    x_vus = np.where(g, draw_mis_path(rng, 20000, p), rng.standard_normal(20000))
    rows = []
    cache = {}
    for arm in ["all", "mis", "reweight", "mis_ptv"]:
        try:
            if arm == "reweight":
                ss, kind, fits, seeds = cache["all"]
                fits = reweight_to_missense(ss, kind, fits, seeds)
            else:
                ss, kind = build_scoreset(d, arm)
                fits, seeds = fit_bootstraps(ss, K, B, R)
                cache[arm] = (ss, kind, fits, seeds)
            cal = calibrate(ss, fits, seeds, K)
            m = evaluate(cal, p, pi, x_vus)
        except Exception as e:
            m = dict(error=repr(e)[:200])
        m.update(scenario=scen, n_mis=n_mis, n_ptv_plp=n_ptv_plp, rep=rep, arm=arm, K=K, B=B, R=R)
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=48)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--n-mis", type=int, nargs="+", default=[8, 30])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from joblib import Parallel, delayed
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(args.out or OUT / ("pilot.csv" if args.pilot else "results.csv"))
    tasks = [(s, n, r) for s in SCENARIOS for n in args.n_mis for r in range(args.reps)]
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_cell)(s, n, r, K=args.K, B=args.B, R=args.R) for s, n, r in tasks)
    df = pd.DataFrame([x for rs in res for x in rs])
    df.to_csv(out, index=False)
    cols = ["prior_err", "overclaim_path", "llr5_err_path", "pts_over", "pts_wrong_dir", "pts_mae", "path_det"]
    pd.set_option("display.width", 250)
    if "error" in df:
        print("errors:", df.error.notna().sum())
    print(df.groupby(["scenario", "n_mis", "arm"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
