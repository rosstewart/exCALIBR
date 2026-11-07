"""E6: how should variants that belong to several samples be handled in the fit?

Schemes (same data, same calibration code):
  onehot     production: Fit.run with the multi-label matrix; every bootstrap
             assigns each multi-label variant to one of its samples at random
             (makeOneHot), then resamples each sample independently.
  composite  composite (marginal) likelihood: each sample keeps ALL its members.
             Bootstrap = resample unique variants within each membership cell
             (cell sizes fixed); a drawn variant is duplicated once per sample it
             belongs to (one one-hot row per membership); Fit.run(bootstrap=False).
  inv_m      as composite, but each duplicated row has weight 1/m (m = number of
             samples the variant belongs to) in the component (theta) M-step; the
             mixing-weight step is an unweighted mean within each sample, so each
             sample keeps full membership for its weights while each variant counts
             once for the components.

Generative model (LoF assay; unit-variance normals; N mean 0, H mean -kappa_h,
L mean -kappa): pathogenic missense = (1-h) L + h H. Population (gnomAD)
pathogenic variants have hypomorph share h_pop; clinically ascertained P/LP not
seen in gnomAD have h_clin (more severe). A fraction of P/LP are ALSO gnomAD
variants (drawn from the population pathogenic pool), so the overlap cell
P/LP n gnomAD differs in composition from the rest of P/LP. B/LB are benign;
a fraction of B/LB are also gnomAD and a fraction also synonymous-like
(both benign, so overlap there is harmless by construction).

Truth for missense VUS = the population: lr+(x) = f_popP(x) / f_N(x), prior pi.
P/LP are biased (more severe) by design in every scheme; what differs between
schemes is only how the overlap is used.

Pilot:  python -m analysis.composition_simulation.overlap --pilot
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps

from analysis.composition_simulation.simulate import calibrate, truth_points, POINTS

warnings.filterwarnings("ignore")
OUT = Path("/data/ross/assay_calibration/composition_simulation")
NAMES = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "gnomAD", "Synonymous"]

SETTINGS = {
    "overlap_same": dict(h_pop=0.3, h_clin=0.3),     # overlap cell = rest of P/LP
    "overlap_milder": dict(h_pop=0.7, h_clin=0.15),  # population-observed P/LP are milder
}


def path_pdf(x, h, kappa=3.0, kappa_h=1.5):
    return (1 - h) * sps.norm.pdf(x, -kappa) + h * sps.norm.pdf(x, -kappa_h)


def draw_path(rng, n, h, kappa=3.0, kappa_h=1.5):
    return np.where(rng.random(n) < h, -kappa_h, -kappa) + rng.standard_normal(n)


def make_data(rng, s, n_plp=40, frac_plp_in_g=0.35, n_blb=40, frac_blb_in_g=0.5, n_g=400, n_syn=200, pi=0.2):
    """Unique variants: scores (n,) and membership (n, 4)."""
    rows, mem = [], []
    # gnomAD: pathogenic (population pool) + benign
    n_gp = rng.binomial(n_g, pi)
    g_path = draw_path(rng, n_gp, s["h_pop"])
    g_ben = rng.standard_normal(n_g - n_gp)
    # P/LP: some are gnomAD pathogenic variants, the rest clinical-only (more severe)
    n_pg = min(int(round(n_plp * frac_plp_in_g)), n_gp)
    pg_idx = rng.choice(n_gp, size=n_pg, replace=False)
    in_plp = np.zeros(n_gp, bool); in_plp[pg_idx] = True
    for x, p in zip(g_path, in_plp):
        rows.append(x); mem.append([p, False, True, False])
    # B/LB: some are gnomAD benign variants, the rest not in gnomAD
    n_bg = min(int(round(n_blb * frac_blb_in_g)), len(g_ben))
    bg_idx = set(rng.choice(len(g_ben), size=n_bg, replace=False).tolist())
    for i, x in enumerate(g_ben):
        rows.append(x); mem.append([False, i in bg_idx, True, False])
    for x in draw_path(rng, n_plp - n_pg, s["h_clin"]):
        rows.append(x); mem.append([True, False, False, False])
    for x in rng.standard_normal(n_blb - n_bg):
        rows.append(x); mem.append([False, True, False, False])
    for x in rng.standard_normal(n_syn):
        rows.append(x); mem.append([False, False, False, True])
    return np.array(rows), np.array(mem, bool)


def cell_bootstrap(rng, mem):
    """Resample unique variants within each membership cell; return indices with multiplicity."""
    keys = [tuple(r) for r in mem]
    cells = {}
    for i, k in enumerate(keys):
        cells.setdefault(k, []).append(i)
    out = []
    for idx in cells.values():
        idx = np.asarray(idx)
        out.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(out)


def expand(scores, mem, idx):
    """One one-hot row per (drawn variant, sample it belongs to); weight 1/m."""
    xs, sa, w = [], [], []
    for i in idx:
        cols = np.flatnonzero(mem[i])
        for c in cols:
            r = np.zeros(mem.shape[1], bool); r[c] = True
            xs.append(scores[i]); sa.append(r); w.append(1.0 / len(cols))
    return np.array(xs), np.array(sa), np.array(w)


def fit_scheme(scheme, scores, mem, K, B, R):
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    from src.assay_calibration.fit_utils.fit import Fit
    fits, seeds = [], []
    for b in range(B):
        try:
            if scheme == "onehot":
                ss = BasicScoreset(scores, mem, sample_names=NAMES)
                models, best, _ = Fit(ss).run([K], bootstrap=True, bootstrap_seed=b, num_fits=R, master_seed=0,
                                              check_monotonic=True, core_limit=1, verbose=False)
            else:
                rng = np.random.default_rng([b, 7])
                xs, sa, w = expand(scores, mem, cell_bootstrap(rng, mem))
                ss = BasicScoreset(xs, sa, sample_names=NAMES)
                kw = dict(observation_weights=w) if scheme == "inv_m" else {}
                models, best, _ = Fit(ss).run([K], bootstrap=False, bootstrap_seed=b, num_fits=R, master_seed=0,
                                              check_monotonic=True, core_limit=1, verbose=False, **kw)
            fits.append({"fit": models[best]}); seeds.append(b)
        except Exception:
            continue
    return fits, np.array(seeds)


def run_cell(setting, rep, K=2, B=30, R=4, pi=0.2):
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    from src.assay_calibration.fit_utils.fit import assign_points
    s = SETTINGS[setting]
    rng = np.random.default_rng([list(SETTINGS).index(setting), rep, 11])
    scores, mem = make_data(rng, s, pi=pi)
    ss_full = BasicScoreset(scores, mem, sample_names=NAMES)   # calibration: full memberships
    n = 20000
    g = rng.random(n) < pi
    x = np.where(g, draw_path(rng, n, s["h_pop"]), rng.standard_normal(n))
    true_llr = np.log(path_pdf(x, s["h_pop"]) / sps.norm.pdf(x))
    tpts = truth_points(true_llr, pi)
    out = []
    for scheme in ["onehot", "composite", "inv_m"]:
        try:
            fits, seeds = fit_scheme(scheme, scores, mem, K, B, R)
            cal = calibrate(ss_full, fits, seeds, K)
            sr = np.asarray(cal["score_range"], float)
            llr = np.asarray(cal["log_lr_plus"], float)
            p5 = np.interp(x, sr, np.nanpercentile(llr, 5, axis=0))
            p50 = np.interp(x, sr, np.nanpercentile(llr, 50, axis=0))
            p95 = np.interp(x, sr, np.nanpercentile(llr, 95, axis=0))
            pos = true_llr > 0
            pts = assign_points(x, {int(k): v for k, v in cal["point_ranges"].items()})
            W = np.array([f["fit"]["weights"][0] for f in fits])
            m = dict(prior=float(cal["prior"]), prior_err=float(cal["prior"] - pi),
                     llr50_err_path=float(np.mean(p50[pos] - true_llr[pos])),
                     overclaim_path=float((p5[pos] > true_llr[pos]).mean()),
                     band_width=float(np.mean(p95 - p5)),
                     coverage=float(((p5 <= true_llr) & (true_llr <= p95)).mean()),
                     pts_mae=float(np.mean(np.abs(pts - tpts))),
                     path_det=float((pts > 0).mean()), true_path_det=float((tpts > 0).mean()),
                     w_plp_abn_sd=float(W[:, int(np.argmin([p[0] for p in fits[0]["fit"]["component_params"]]))].std()))
        except Exception as e:
            m = dict(error=repr(e)[:200])
        m.update(setting=setting, rep=rep, scheme=scheme, K=K, B=B, R=R)
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--n-jobs", type=int, default=24)
    args = ap.parse_args()
    from joblib import Parallel, delayed
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / ("overlap_pilot.csv" if args.pilot else "overlap_results.csv")
    tasks = [(s, r) for s in SETTINGS for r in range(args.reps)]
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(run_cell)(s, r, B=args.B, R=args.R) for s, r in tasks)
    df = pd.DataFrame([x for rs in res for x in rs])
    df.to_csv(out, index=False)
    cols = ["prior_err", "llr50_err_path", "overclaim_path", "band_width", "coverage", "pts_mae", "path_det"]
    pd.set_option("display.width", 250)
    if "error" in df:
        print("errors:", df.error.notna().sum(), df.error.dropna().unique()[:3])
    print(df.groupby(["setting", "scheme"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
