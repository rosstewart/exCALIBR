"""Pathomechanism-aware joint prior P(Y=1, M=1) in the prior simulation (Supplementary Section "Theory and
simulation of prior estimation"): does the production estimate recover the true joint prior?

Same simulated datasets as test_priors.py (same settings, seeds and data generation as replicate_task, so the fitted
replicates are the main design's), restricted to the representative-control settings (1_clean: every regime, true
priors 0.05-0.40) and the mechanism-mix settings (3_mech_bias). Per fitted replicate and assay, each bootstrap fit
(production 2-component Gaussian fit, as in simulate_priors.fit_priors) gives

    P(Y=1)        get_fit_prior standard mode (with its discard rule), as in the main design,
    P(M=1|Y=1)    get_fit_prior(..., pathomechanism_method="boundary"), the production estimator
                  (point_ranges._pathomechanism_boundary_match_frac on the P/LP scores),

and the joint prior P(Y=1) * P(M=1|Y=1) (the production default, "product"); the replicate estimate is the median
over bootstraps, as visualize.py takes it. The true-component analogue uses the true class densities with the
sample-estimated class weights (as oracle_prior does) for both factors.

Truth: P(Y=1, M=1) = true prior x fraction of pathogenic population variants acting through a mechanism the assay
measures (simulate_priors.DETECTED: A1 0.25, A2 0.75, A3 1). In 3_mech_bias the P/LP sample's mechanism mix differs
from the population's, so the estimate is expected to track the P/LP sample's fraction instead.

Writes <out-dir>/pathomech_replicates.csv and pathomech_tests.csv. Nothing in test_priors.py's outputs is touched.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from analysis.prior_simulation.simulate_priors import (
    ASSAYS, B_IDX, CLASSES, DEFAULT_OUT_DIR, DETECTED, G_IDX, P_IDX, POP_M1_FRAC, S_IDX, class_density,
    class_weights, draw_classes, draw_latent, get_fit_prior, oracle_prior, readout,
)
from analysis.prior_simulation.test_priors import N_BOOT, N_REP_FIT, N_REP_ORACLE, N_RESTARTS, build_settings, \
    regime_counts, wilcoxon
from src.assay_calibration.fit_utils.point_ranges import _pathomechanism_boundary_match_frac

EXPERIMENTS = ("1_clean", "3_mech_bias")


def _benign_weights(samples, weights):
    """Benign-reference class weights, combined as oracle_prior combines them."""
    if len(samples[B_IDX]) == 0:
        return weights[S_IDX]
    if len(samples[S_IDX]) == 0:
        return weights[B_IDX]
    return (weights[B_IDX] + weights[S_IDX]) / 2


def oracle_match_frac(samples, assay, delta, rho):
    """P(M=1|Y=1) with the true class densities: the production boundary estimator on the P/LP scores."""
    weights = [class_weights(np.vstack([class_density(s, c, assay, delta, rho) for c in CLASSES]))
               if len(s) else None for s in samples]
    d = np.vstack([class_density(samples[P_IDX], c, assay, delta, rho) for c in CLASSES])
    return _pathomechanism_boundary_match_frac(_benign_weights(samples, weights) @ d, weights[P_IDX] @ d)


def fit_both(samples, seeds, num_fits):
    """Per bootstrap: (P(Y=1), P(M=1|Y=1)) from the production fit and get_fit_prior, as fit_priors fits."""
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    from src.assay_calibration.fit_utils.fit import Fit

    scores = np.concatenate(samples)
    sa = np.concatenate([np.full(len(s), str(i)) for i, s in enumerate(samples)])
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss = BasicScoreset(scores, sa)
        fitter = Fit(ss)
        if len(samples[B_IDX]):
            idx = dict(benign_idx=B_IDX, gnomad_idx=G_IDX, synonymous_idx=S_IDX if len(samples[S_IDX]) else None)
            benign_method = "avg"
        else:
            idx = dict(benign_idx=None, gnomad_idx=G_IDX - 1, synonymous_idx=S_IDX - 1)
            benign_method = "synonymous"
        for b in seeds:
            p, mf = np.nan, np.nan
            try:
                models, best, _ = fitter.run(
                    [2], bootstrap=True, bootstrap_seed=int(b), num_fits=num_fits, master_seed=0,
                    check_monotonic=True, force_gaussian=True, core_limit=1, verbose=False,
                )
                fit = {"fit": models[best]}
                kw = dict(pathogenic_idx=P_IDX, sample_assignments=ss.sample_assignments, **idx)
                p, _ = get_fit_prior(fit, ss.scores, benign_method, **kw)
                _, mf = get_fit_prior(fit, ss.scores, benign_method, pathomechanism_method="boundary",
                                      pathomechanism_boundary_joint_prior="product", **kw)
            except Exception:
                pass
            out.append((p, mf))
    return out


def task(setting, seed, rep, do_fit, n_boot, n_restarts):
    """One simulated dataset (generated exactly as test_priors.replicate_task), all three assays."""
    exp, regime, pi, params = setting
    rng = np.random.default_rng([seed, rep])
    n_p, n_b, n_g, n_s = regime_counts(regime)
    labs = [
        draw_classes(rng, n_p, 1 - params["eps_p"], params["lab_m1_frac"]),
        draw_classes(rng, n_b, params["delta_b"], POP_M1_FRAC),
        draw_classes(rng, n_g, pi, POP_M1_FRAC),
        draw_classes(rng, n_s, 0.0, POP_M1_FRAC),
    ]
    latent = [draw_latent(rng, lab, params["delta"], params["rho"]) for lab in labs]
    rows = []
    for assay in ASSAYS:
        samples = [readout(e1, e2, assay) for e1, e2 in latent]
        orc_p = oracle_prior(samples, samples[G_IDX], assay, params["delta"], params["rho"], discard=True)
        orc_mf = oracle_match_frac(samples, assay, params["delta"], params["rho"])
        row = {"experiment": exp, "regime": regime, "true_prior": pi, **params, "rep": rep, "assay": assay,
               "true_mech_frac": DETECTED[assay],
               "plp_mech_frac": (params["lab_m1_frac"] if assay.startswith("A1") else
                                 1 - params["lab_m1_frac"] if assay.startswith("A2") else 1.0),
               "true_joint": pi * DETECTED[assay],
               "oracle_prior": orc_p, "oracle_mech_frac": orc_mf, "oracle_joint": orc_p * orc_mf}
        if do_fit:
            pm = np.array(fit_both(samples, range(n_boot), n_restarts), dtype=float)
            joint = pm[:, 0] * pm[:, 1]
            ok = np.isfinite(joint)
            row.update(fit_prior=float(np.nanmedian(pm[:, 0])) if np.isfinite(pm[:, 0]).any() else np.nan,
                       fit_mech_frac=float(np.nanmedian(pm[:, 1])) if np.isfinite(pm[:, 1]).any() else np.nan,
                       fit_joint=float(np.median(joint[ok])) if ok.any() else np.nan,
                       fit_n_boot_ok=int(ok.sum()), fit_mech_frac_is_one=float(np.mean(pm[ok, 1] >= 1)) if ok.any()
                       else np.nan)
        rows.append(row)
    return rows


def tests(reps):
    """Per setting and assay: mean error of the joint prior (and of P(M=1|Y=1)) vs the truth, Wilcoxon p."""
    keys = ["experiment", "regime", "true_prior", "lab_m1_frac", "assay"]
    out = []
    for k, g in reps.groupby(keys):
        row = dict(zip(keys, k))
        for est in ("oracle", "fit"):
            if f"{est}_joint" not in g or g[f"{est}_joint"].isna().all():
                continue
            e = (g[f"{est}_joint"] - g["true_joint"]).dropna()
            m = (g[f"{est}_mech_frac"] - g["true_mech_frac"]).dropna()
            row.update({f"{est}_n": len(e), f"{est}_joint_mean": g[f"{est}_joint"].mean(),
                        f"{est}_joint_bias": e.mean(), f"{est}_joint_sd": g[f"{est}_joint"].std(),
                        f"{est}_joint_p": wilcoxon(e.to_numpy()),
                        f"{est}_mech_frac_mean": g[f"{est}_mech_frac"].mean(), f"{est}_mech_frac_bias": m.mean()})
        row["true_joint"] = g["true_joint"].iloc[0]
        row["true_mech_frac"] = g["true_mech_frac"].iloc[0]
        row["plp_mech_frac"] = g["plp_mech_frac"].iloc[0]
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=1)  # test_priors main design's seed (same datasets)
    ap.add_argument("--n-jobs", type=int, default=120)
    ap.add_argument("--n-rep-fit", type=int, default=N_REP_FIT)
    ap.add_argument("--n-rep-oracle", type=int, default=N_REP_ORACLE)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-restarts", type=int, default=N_RESTARTS)
    args = ap.parse_args()
    settings = build_settings()
    seeds = np.random.SeedSequence(args.seed).generate_state(len(settings))  # over the full list, as test_priors
    pairs = [(s, int(sd)) for s, sd in zip(settings, seeds) if s[0] in EXPERIMENTS]
    jobs = [delayed(task)(s, sd, r, True, args.n_boot, args.n_restarts) for s, sd in pairs
            for r in range(args.n_rep_fit)]
    jobs += [delayed(task)(s, sd, r, False, 0, 0) for s, sd in pairs for r in range(args.n_rep_fit, args.n_rep_oracle)]
    print(f"{len(pairs)} settings, {len(jobs)} tasks on {args.n_jobs} workers", flush=True)
    reps = pd.DataFrame([row for res in Parallel(n_jobs=args.n_jobs, verbose=5)(jobs) for row in res])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reps.to_csv(args.out_dir / "pathomech_replicates.csv", index=False, float_format="%.5f")
    tests(reps).to_csv(args.out_dir / "pathomech_tests.csv", index=False, float_format="%.4g")
    print("wrote", args.out_dir / "pathomech_replicates.csv", flush=True)


if __name__ == "__main__":
    main()
