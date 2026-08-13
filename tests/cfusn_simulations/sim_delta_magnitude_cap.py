#!/usr/bin/env python3
"""Does a defense-in-depth cap on get_Delta_update_cfusn's per-dimension
Delta row norm -- tied to that dimension's own LOCAL weighted variance,
computed from the same obs_z/residuals already available in the M-step, no
extra parameters needed -- eliminate the residual, non-catastrophic tail
left over after the two root-cause fixes upstream of it (Psi PSD-violation
clipping in _mc_truncated_mvn_moments, and the raised RIDGE_FLOOR in this
same function), WITHOUT clipping genuinely large true skew in clean data?

Context: those two fixes eliminated outright numerical blowups (was up to
~1.7e9 on real TP53 data), but a smaller, non-catastrophic tail remained
(real TP53 K=6 data: max Delta row-norm ~47, 13/960 checked fits exceeding
row-norm 10). This script tests the proposed cap -- max_norm_d =
SAFETY_FACTOR * sqrt(local weighted variance in dimension d) -- against
that same real-data scenario, plus a clean synthetic check that it doesn't
meaningfully cost real recovery at realistic skew levels.

Cap rationale (see get_Delta_update_cfusn's own docstring/comments for the
full version now shipped in production): Gamma[d,d] = cov[d,d] -
||Delta[d,:]||^2 must stay positive for Gamma to be PD at all, and more
importantly, a single active skewing direction's marginal distribution is
exactly a standard skew-normal, which has a hard ceiling on Pearson
skewness of ~0.995 (Azzalini's skew-normal, as the shape parameter ->
infinity) -- REGARDLESS of how large Delta gets. Past a moderate
magnitude, further increases in Delta buy almost no additional real,
identifiable skewness (a known skew-normal MLE flat-likelihood
pathology) while linearly claiming more of cov[d,d]'s variance, exactly
the poorly-conditioned regime this whole investigation traced the M-step
blowups back to.

Two-part test:
  (A) Real TP53 data, K=6, the exact fragile scenario found earlier --
      does the cap reduce/eliminate the remaining row-norm>10 tail?
  (B) Clean synthetic data (no missingness) with TRUE large skew
      (magnitude 0.7, this investigation's "large" regime throughout,
      plus a deliberately unrealistic "stress" case at 1.5, which implies
      ~41% of a standardized dimension's variance from skew alone) --
      does the cap avoid degrading recovery of REAL, plausible large
      skew, while it's fine/expected for it to hold back an implausible
      extreme?

Usage:
    python tests/cfusn_simulations/sim_delta_magnitude_cap.py
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hpc.prepare as prepare
from src.assay_calibration.fit_utils.fit import Fit, tryToFit
import src.assay_calibration.fit_utils.cfusn.update_steps as US
from tests.cfusn_simulations.sim_utils import sample_cfusn

RESULTS_DIR = Path(__file__).resolve().parent / "results"

_orig_get_Delta_update_cfusn = US.get_Delta_update_cfusn
SAFETY_FACTOR = 0.95

MU_TRUE = np.zeros(4)
GAMMA_TRUE = 0.5 * np.eye(4)
DIRECTION = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
N_OBS = 5000
N_RESTARTS = 4
N_BOOTSTRAPS_REAL = 20


def uncapped_get_Delta_update_cfusn(mu_new, observations, responsibilities, eta, Psi,
                                    sample_weights=None):
    """The current production formula MINUS the magnitude cap -- i.e. just
    the Psi-PSD-fix + ridge-floor fixes -- used here as the "without cap"
    comparison arm (production's actual get_Delta_update_cfusn already has
    the cap baked in, so this reimplements the pre-cap version deliberately
    for a clean A/B, rather than trying to monkeypatch pieces out of it).
    """
    obs = ~np.isnan(observations)
    x_fill = np.where(obs, observations, 0.0)
    z = responsibilities
    z_eff = z if sample_weights is None else z * sample_weights
    N, p = observations.shape
    q = eta.shape[1]

    obs_z = obs * z_eff[:, None]
    residuals = x_fill - mu_new
    numer = (obs_z * residuals).T @ eta
    Psi_sum = np.einsum('nd,nij->dij', obs_z, Psi)

    RIDGE_FLOOR = 1e-3
    Delta_new = np.zeros((p, q))
    for d in range(p):
        Ps = Psi_sum[d]
        eig = np.linalg.eigvalsh(Ps)
        if eig.min() < RIDGE_FLOOR:
            Ps = Ps + (RIDGE_FLOOR - eig.min() + RIDGE_FLOOR) * np.eye(q)
        try:
            Delta_new[d] = np.linalg.solve(Ps, numer[d])
        except np.linalg.LinAlgError:
            Delta_new[d] = numer[d] / np.maximum(np.diag(Ps), 1e-12)

    return Delta_new


def part_a_real_tp53(n_bootstraps):
    print(f"\n{'='*90}")
    print("  PART A: real TP53 K=6 fragile scenario -- does the cap reduce the residual tail?")
    print(f"{'='*90}")

    class _Args:
        kawoligo_seed = 0
        kawoligo_jitter_sigma = 0.1
        rpvs_all = False

    gene_ms_map = prepare._build_gene_set_ms_map("tp53", _Args())
    ms = gene_ms_map["TP53"]
    fitter = Fit(ms)

    rows = []
    for label, fn in [("without_cap", uncapped_get_Delta_update_cfusn),
                       ("with_cap", _orig_get_Delta_update_cfusn)]:
        US.get_Delta_update_cfusn = fn
        max_norm_overall = 0.0
        n_over_10 = 0
        n_total = 0
        for bootstrap_seed in range(n_bootstraps):
            jobs = fitter.generate_fit_jobs([6], bootstrap_seed=bootstrap_seed, num_fits=8,
                                            latent_q=2, min_overlap_rows=1, verbose_overlap=False)
            for j in jobs:
                result = tryToFit(
                    j["train_observations"], j["train_sample_assignments"], num_components=6,
                    constrained=False, init_method=j["init_method"],
                    init_constraint_adjustment=j["init_constraint_adjustment"], multivariate=True,
                    latent_q=2, check_monotonic=False, num_fits=1, fit_seed=j["kwargs"]["fit_seed"],
                    lambdaIndex=j["kwargs"]["lambdaIndex"], max_em_iters=300, verbose=False,
                    verbose_init=False,
                )
                params = result.get("component_params", [])
                for p in params:
                    if len(p) >= 2:
                        n = float(np.linalg.norm(np.asarray(p[1])))
                        n_total += 1
                        max_norm_overall = max(max_norm_overall, n)
                        if n > 10:
                            n_over_10 += 1
        print(f"  {label:>12}: checked {n_total}, max norm={max_norm_overall:.4f}, n>10: {n_over_10}")
        rows.append(dict(variant=label, n_checked=n_total, max_norm=max_norm_overall, n_over_10=n_over_10))
    US.get_Delta_update_cfusn = _orig_get_Delta_update_cfusn
    return rows


def part_b_synthetic_large_skew(n_seeds):
    print(f"\n{'='*90}")
    print("  PART B: clean synthetic data, TRUE large skew -- does the cap hurt real recovery?")
    print(f"{'='*90}")

    rows = []
    for magnitude, label in [(0.7, "large_0.7"), (1.5, "stress_1.5")]:
        Delta_true = DIRECTION * magnitude
        for cap_label, fn in [("without_cap", uncapped_get_Delta_update_cfusn),
                              ("with_cap", _orig_get_Delta_update_cfusn)]:
            US.get_Delta_update_cfusn = fn
            recovered = []
            for seed in range(n_seeds):
                rng = np.random.RandomState(seed)
                X, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
                sa = np.ones((N_OBS, 1), dtype=bool)
                best = None
                for i in range(N_RESTARTS):
                    result = tryToFit(
                        X, sa, num_components=1, constrained=False, init_method="kmeans",
                        init_constraint_adjustment="scale", multivariate=True, latent_q=2,
                        check_monotonic=False, num_fits=1, fit_seed=int(seed * 100 + i),
                        lambdaIndex=i, max_em_iters=300, verbose=False, verbose_init=False,
                    )
                    params = result.get("component_params", [])
                    if not params or len(params[0]) == 0:
                        continue
                    lls = list(result.get("likelihoods", []))
                    ll = lls[-1] if len(lls) else -np.inf
                    if best is None or ll > best[0]:
                        best = (ll, params)
                if best is not None:
                    Delta_fit = np.asarray(best[1][0][1])
                    recovered.append(float(np.linalg.norm(Delta_fit)))
            recovered = np.array(recovered)
            true_norm = float(np.linalg.norm(Delta_true))
            pct = 100 * recovered.mean() / true_norm if true_norm > 1e-9 else float("nan")
            print(f"  regime={label:>12}  {cap_label:>12}: mean fit_norm={recovered.mean():.4f} "
                  f"(true={true_norm:.4f}, recovered={pct:.0f}%)")
            rows.append(dict(regime=label, variant=cap_label, true_norm=true_norm,
                             mean_fit_norm=float(recovered.mean()), recovered_pct=pct))
        US.get_Delta_update_cfusn = _orig_get_Delta_update_cfusn
    return rows


def _write_csv(rows, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{ts}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-bootstraps-real", type=int, default=N_BOOTSTRAPS_REAL)
    parser.add_argument("--n-seeds-synthetic", type=int, default=8)
    args = parser.parse_args()

    rows_a = part_a_real_tp53(args.n_bootstraps_real)
    _write_csv(rows_a, "sim_delta_magnitude_cap_part_a")
    rows_b = part_b_synthetic_large_skew(args.n_seeds_synthetic)
    _write_csv(rows_b, "sim_delta_magnitude_cap_part_b")
    print()


if __name__ == "__main__":
    main()
