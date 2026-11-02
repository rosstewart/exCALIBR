"""Paired significance tests for the prior-estimation simulation.

Uses the generative model, settings and estimators of simulate_priors.py, but
with the replicate dataset as the unit of analysis:

* 500 replicate datasets per setting for the true-component estimator, the
  first 50 of which also get the production fit (5 bootstraps x 8 restarts). Within a replicate, the three assays
  score the SAME simulated variants (one latent draw read out by A1, A2, A3),
  so assays are compared paired.
* Per replicate and assay, one prior per estimator:
    - oracle: fixed true components, weights from the full samples (point estimate);
    - fit: production 2-component Gaussian fit, 1 restart, median of the
      per-bootstrap priors over 10 bootstraps (production use_median_prior).
* Tests (two-sided Wilcoxon signed-rank, Holm-corrected within each
  experiment and test family):
    - bias: estimate - true prior, per arm and assay;
    - deviation from the large-sample expectation (noise and mechanism-bias
      experiments, where the expectation is not the true prior);
    - fit - oracle, paired, per assay;
    - between assays: paired differences of the estimate and of the absolute
      error, A1 vs A3, A2 vs A3, A1 vs A2.

Run:
    python -m analysis.prior_simulation.test_priors [--n-jobs 64]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps
from joblib import Parallel, delayed

from analysis.prior_simulation.simulate_priors import (
    ASSAYS, BASE, DEFAULT_OUT_DIR, G_IDX, POP_M1_FRAC, REGIMES,
    draw_classes, draw_latent, expected_prior, fit_priors, oracle_prior, readout,
)

N_REP_ORACLE = 500   # replicate datasets per setting, true-component estimator
N_REP_FIT = 50       # of those, the first N_REP_FIT also get the production fit (paired)
ORACLE_CHUNK = 50    # oracle-only replicates per task
N_BOOT = 5        # bootstraps per fitted replicate
N_RESTARTS = 8    # restarts per bootstrap (best by held-out log-likelihood, as in production)
ASSAY_PAIRS = [("A1 (M1 only)", "A3 (M1+M2)"), ("A2 (M2 only)", "A3 (M1+M2)"), ("A1 (M1 only)", "A2 (M2 only)")]


def build_settings():
    """Same grid as simulate_priors.build_cells, without the assay axis."""
    settings = []

    def add(exp, regime, pi, **over):
        settings.append((exp, regime, pi, {**BASE, **over}))

    for regime in REGIMES:
        for pi in (0.05, 0.10, 0.25, 0.40):
            add("1_clean", regime, pi)
    for eps in (0.05, 0.10, 0.20):
        add("2_noise_plp", "rich", 0.25, eps_p=eps)
    for db in (0.05, 0.10, 0.20):
        add("2_noise_blb", "rich", 0.25, delta_b=db)
    for m1 in (0.0, 0.25, 0.50, 0.75, 1.0):
        add("3_mech_bias", "rich", 0.25, lab_m1_frac=m1)
    for pi in (0.10, 0.25):
        for d in (1.5, 2.5, 4.0):
            add("4_separation", "rich", pi, delta=d)
        for d in (2.5, 4.0):
            for rho in (0.0, 0.3, 0.6):
                add("5_correlation", "rich", pi, delta=d, rho=rho)
    return settings


def replicate_task(setting, seed, rep, do_fit=True, n_boot=N_BOOT, n_restarts=N_RESTARTS):
    exp, regime, pi, params = setting
    rng = np.random.default_rng([seed, rep])
    n_p, n_b, n_g, n_s = REGIMES[regime]
    labs = [
        draw_classes(rng, n_p, 1 - params["eps_p"], params["lab_m1_frac"]),
        draw_classes(rng, n_b, params["delta_b"], POP_M1_FRAC),
        draw_classes(rng, n_g, pi, POP_M1_FRAC),
        draw_classes(rng, n_s, 0.0, POP_M1_FRAC),
    ]
    latent = [draw_latent(rng, lab, params["delta"], params["rho"]) for lab in labs]
    out = []
    for assay in ASSAYS:
        samples = [readout(e1, e2, assay) for e1, e2 in latent]
        orc = oracle_prior(samples, samples[G_IDX], assay, params["delta"], params["rho"])
        boots = np.asarray(fit_priors(samples, range(n_boot), num_fits=n_restarts) if do_fit else [], dtype=float)
        ok = boots[np.isfinite(boots)]
        out.append({
            "experiment": exp, "regime": regime, "true_prior": pi, **params, "rep": rep, "assay": assay,
            "oracle": orc,
            "fit": float(np.median(ok)) if len(ok) else np.nan,
            "fit_n_boot_ok": len(ok),
            "fit_n_boot": n_boot if do_fit else 0,
            "fit_n_restarts": n_restarts if do_fit else 0,
            "fit_frac_at_em_init": float(np.mean(np.abs(ok - 0.5) < 1e-3)) if len(ok) else np.nan,
        })
    return out


def oracle_chunk(setting, seed, r0, r1):
    return [row for r in range(r0, r1) for row in replicate_task(setting, seed, r, do_fit=False)]


def wilcoxon(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5 or np.allclose(x, 0):
        return len(x), np.nan
    return len(x), float(sps.wilcoxon(x).pvalue)


def summarize(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = sps.t.ppf(0.975, len(x) - 1) * se
    return m, m - h, m + h


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def tests(reps):
    keys = ["experiment", "regime", "true_prior", "delta", "rho", "eps_p", "delta_b", "lab_m1_frac"]
    rows = []
    for key, g in reps.groupby(keys, sort=False):
        setting = dict(zip(keys, key))
        wide = {a: g[g.assay == a].sort_values("rep").set_index("rep") for a in ASSAYS}
        for assay in ASSAYS:
            w = wide[assay]
            exp_p = w["expected_prior"].iloc[0]
            for arm in ("oracle", "fit"):
                for test, ref in (("bias_vs_truth", setting["true_prior"]), ("dev_vs_expectation", exp_p)):
                    if test == "dev_vs_expectation" and setting["experiment"] not in ("2_noise_plp", "2_noise_blb", "3_mech_bias"):
                        continue
                    d = w[arm] - ref
                    n, p = wilcoxon(d)
                    m, lo, hi = summarize(d)
                    rows.append({**setting, "test": test, "arm": arm, "assay": assay, "comparison": f"{arm} - reference",
                                 "reference": ref, "n": n, "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "p": p})
            d = w["fit"] - w["oracle"]
            n, p = wilcoxon(d)
            m, lo, hi = summarize(d)
            rows.append({**setting, "test": "fit_vs_oracle", "arm": "fit-oracle", "assay": assay,
                         "comparison": "fit - oracle", "n": n, "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "p": p})
        for arm in ("oracle", "fit"):
            for a, b in ASSAY_PAIRS:
                for test, f in (("assay_estimate", lambda s: s),
                                ("assay_abs_error", lambda s: (s - setting["true_prior"]).abs())):
                    d = f(wide[a][arm]) - f(wide[b][arm])
                    n, p = wilcoxon(d)
                    m, lo, hi = summarize(d)
                    rows.append({**setting, "test": test, "arm": arm, "assay": f"{a[:2]} vs {b[:2]}",
                                 "comparison": f"{a[:2]} - {b[:2]}", "n": n, "mean_diff": m,
                                 "ci_lo": lo, "ci_hi": hi, "p": p})
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for _, idx in df.groupby(["experiment", "test", "arm"]).groups.items():
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "p"].to_numpy())
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=64)
    ap.add_argument("--n-rep-oracle", type=int, default=N_REP_ORACLE)
    ap.add_argument("--n-rep-fit", type=int, default=N_REP_FIT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstraps per fitted replicate")
    ap.add_argument("--n-restarts", type=int, default=N_RESTARTS, help="restarts per bootstrap")
    args = ap.parse_args()

    settings = build_settings()
    seeds = np.random.SeedSequence(args.seed).generate_state(len(settings))
    tasks = [delayed(replicate_task)(s, int(seed), r, True, args.n_boot, args.n_restarts)
             for s, seed in zip(settings, seeds) for r in range(args.n_rep_fit)]
    tasks += [delayed(oracle_chunk)(s, int(seed), r0, min(r0 + ORACLE_CHUNK, args.n_rep_oracle))
              for s, seed in zip(settings, seeds)
              for r0 in range(args.n_rep_fit, args.n_rep_oracle, ORACLE_CHUNK)]
    print(f"{len(settings)} settings: {args.n_rep_fit} fitted + {args.n_rep_oracle - args.n_rep_fit} "
          f"true-component-only replicates each; {len(tasks)} tasks on {args.n_jobs} workers", flush=True)
    # Fit tasks first: they dominate the runtime.
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(tasks)
    reps = pd.DataFrame([row for res in results for row in res])
    ek = ["assay", "regime", "true_prior", "delta", "rho", "eps_p", "delta_b", "lab_m1_frac"]
    uniq = reps[ek].drop_duplicates().copy()
    uniq["expected_prior"] = [
        expected_prior(r.assay, r.true_prior, r.delta, r.rho, r.eps_p, r.delta_b, r.lab_m1_frac,
                       has_benign=REGIMES[r.regime][1] > 0)
        for r in uniq.itertuples()
    ]
    reps = reps.merge(uniq, on=ek, how="left")
    reps.to_csv(args.out_dir / "paired_replicates.csv", index=False, float_format="%.4f")
    stats = tests(reps)
    stats.to_csv(args.out_dir / "paired_tests.csv", index=False, float_format="%.4g")
    print(f"wrote {args.out_dir / 'paired_replicates.csv'} and paired_tests.csv ({len(stats)} tests)")


if __name__ == "__main__":
    main()
