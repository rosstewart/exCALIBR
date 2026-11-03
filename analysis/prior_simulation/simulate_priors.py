"""Prior-estimation simulation: one gene, two disease mechanisms, three assays.

Reviewer 3 #6 says ExCALIBR priors reflect "pathogenic-like assay-score
mixture membership" rather than the fraction of population variants that are
pathogenic. This script tests whether the production prior estimator
recovers a known prior when assays measure only part of a gene's disease
mechanisms, and how label noise, mechanism-biased sampling of labelled
variants, assay separation and correlation between mechanisms move it.

Generative model
----------------
Each variant has a latent functional effect e = (e1, e2) on mechanisms M1 and
M2, e ~ N(mu_c, Sigma) with Sigma = [[1, rho], [rho, 1]]:

    benign                      mu = (0, 0)
    pathogenic, acting via M1   mu = (-delta, -rho * delta)
    pathogenic, acting via M2   mu = (-rho * delta, -delta)

so the unaffected mechanism's readout regresses on the affected one with slope
rho (rho = 0: orthogonal mechanisms). Pathogenic variants act via M1 with
probability 0.25 in the population. Assays: A1 = e1, A2 = e2,
A3 = min(e1, e2) (reports the worse of the two functions).

Two estimators, both ending in the production prior EM
-----------------------------------------------------
* oracle: no mixture fit. Each sample's density is a mixture of the exact
  class densities (benign / M1 / M2) under that assay, with the sample's class
  weights estimated by weight-only EM; the prior EM on gnomAD is the production
  core, ``point_ranges.estimate_prior_from_class_densities`` (what
  ``get_fit_prior`` calls), with ``benign_method="avg"`` and the same
  discard rule (prior <= 0.001 or >= 0.999 -> discarded).
* fit: the production fit, ``Fit.run`` with 2 Gaussian components
  (``force_gaussian``), 1 restart per bootstrap, then ``get_fit_prior`` on the
  full dataset, median over bootstraps (``use_median_prior``).

Run:
    python -m analysis.prior_simulation.simulate_priors [--out-dir DIR] [--n-jobs N]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as sps
from joblib import Parallel, delayed

from src.assay_calibration.fit_utils.point_ranges import (
    estimate_prior_from_class_densities, get_fit_prior,
)

DEFAULT_OUT_DIR = Path("/data/ross/assay_calibration/prior_simulation")

POP_M1_FRAC = 0.25
ASSAYS = ["A1 (M1 only)", "A2 (M2 only)", "A3 (M1+M2)"]
CLASSES = ["B", "M1", "M2"]

# (P/LP, B/LB, gnomAD, synonymous) counts.
REGIMES = {
    "rich": (100, 100, 500, 200),
    "TPK1-like": (8, 1, 50, 20),
    "KCNH2_Kozek-like": (3, 0, 10, 6),
}

P_IDX, B_IDX, G_IDX, S_IDX = 0, 1, 2, 3
BASE = dict(delta=4.0, rho=0.0, eps_p=0.0, delta_b=0.0, lab_m1_frac=POP_M1_FRAC)
N_REP_ORACLE, N_BOOT_ORACLE = 10, 100
N_REP_FIT, N_BOOT_FIT, BOOT_CHUNK = 3, 100, 20
LARGE_N = 200_000


# ── generative model ─────────────────────────────────────────────────────────

def class_means(delta, rho):
    return {"B": (0.0, 0.0), "M1": (-delta, -rho * delta), "M2": (-rho * delta, -delta)}


def draw_classes(rng, n, frac_pathogenic, m1_frac):
    is_path = rng.random(n) < frac_pathogenic
    is_m1 = rng.random(n) < m1_frac
    return np.where(~is_path, "B", np.where(is_m1, "M1", "M2"))


def draw_latent(rng, classes, delta, rho):
    """Latent effects (e1, e2) for each variant."""
    mu = class_means(delta, rho)
    m = np.array([mu[c] for c in classes]).reshape(-1, 2)
    z1 = rng.standard_normal(len(classes))
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(len(classes))
    return m[:, 0] + z1, m[:, 1] + z2


def readout(e1, e2, assay):
    return {"A1 (M1 only)": e1, "A2 (M2 only)": e2, "A3 (M1+M2)": np.minimum(e1, e2)}[assay]


def score(rng, classes, assay, delta, rho):
    return readout(*draw_latent(rng, classes, delta, rho), assay)


def class_density(x, cls, assay, delta, rho):
    """Exact density of the assay score for one variant class."""
    m1, m2 = class_means(delta, rho)[cls]
    if assay == "A1 (M1 only)":
        return sps.norm.pdf(x, m1, 1)
    if assay == "A2 (M2 only)":
        return sps.norm.pdf(x, m2, 1)
    # min(e1, e2): f(t) = phi(t-m1) P(e2 > t | e1 = t) + phi(t-m2) P(e1 > t | e2 = t)
    s = np.sqrt(1 - rho ** 2)
    a = sps.norm.pdf(x, m1, 1) * sps.norm.sf((x - (m2 + rho * (x - m1))) / s)
    b = sps.norm.pdf(x, m2, 1) * sps.norm.sf((x - (m1 + rho * (x - m2))) / s)
    return a + b


def make_samples(rng, assay, regime_n, pi, delta, rho, eps_p, delta_b, lab_m1_frac):
    n_p, n_b, n_g, n_s = regime_n
    labs = [
        draw_classes(rng, n_p, 1 - eps_p, lab_m1_frac),   # P/LP, benign contamination eps_p
        draw_classes(rng, n_b, delta_b, POP_M1_FRAC),     # B/LB, pathogenic contamination delta_b
        draw_classes(rng, n_g, pi, POP_M1_FRAC),          # gnomAD
        draw_classes(rng, n_s, 0.0, POP_M1_FRAC),         # synonymous
    ]
    return [score(rng, lab, assay, delta, rho) for lab in labs], labs


# ── oracle estimator ─────────────────────────────────────────────────────────

def class_weights(dens, tol=1e-10, max_iter=10_000):
    """Weight-only EM over fixed class densities (rows: classes, cols: points)."""
    k = dens.shape[0]
    w = np.full(k, 1 / k)
    if dens.shape[1] == 0:
        return w
    for _ in range(max_iter):
        num = w[:, None] * dens
        new_w = (num / np.maximum(num.sum(0), 1e-300)).mean(1)
        if np.abs(new_w - w).max() < tol:
            return new_w
        w = new_w
    return w


EM_STARTS = (0.5, 0.1)  # production starts the prior EM at 0.5; 0.1 is tested as an alternative


def prior_em(f_p, f_b, discard=True, initial_prior=0.5):
    """Production prior EM plus get_fit_prior's discard rule (standard mode); `discard=False` returns the raw EM
    estimate, so the effect of the rule itself can be measured. `initial_prior` is the EM starting value."""
    p = estimate_prior_from_class_densities(f_p, f_b, mode="standard", initial_prior=initial_prior)
    return np.nan if discard and (p <= 0.001 or p >= 0.999) else p


def oracle_prior(samples, gnomad, assay, delta, rho, weights=None, discard=True, initial_prior=0.5):
    """Weights from `samples` (or given); prior EM on the full gnomAD sample."""
    if weights is None:
        weights = []
        for s in samples:
            d = np.vstack([class_density(s, c, assay, delta, rho) for c in CLASSES])
            weights.append(class_weights(d))
    dg = np.vstack([class_density(gnomad, c, assay, delta, rho) for c in CLASSES])
    w_b = weights[S_IDX] if len(samples[B_IDX]) == 0 else (weights[B_IDX] + weights[S_IDX]) / 2
    return prior_em(weights[P_IDX] @ dg, w_b @ dg, discard=discard, initial_prior=initial_prior)


def expected_prior(assay, pi, delta, rho, eps_p, delta_b, lab_m1_frac, has_benign=True, seed=12345):
    """Infinite-control limit: true class weights, prior EM on a large gnomAD sample."""
    rng = np.random.default_rng(seed)
    gnomad = score(rng, draw_classes(rng, LARGE_N, pi, POP_M1_FRAC), assay, delta, rho)

    def comp(frac_path, m1):
        return np.array([1 - frac_path, frac_path * m1, frac_path * (1 - m1)])

    weights = [comp(1 - eps_p, lab_m1_frac), comp(delta_b, POP_M1_FRAC),
               comp(pi, POP_M1_FRAC), comp(0.0, POP_M1_FRAC)]
    samples = [None, np.zeros(1 if has_benign else 0), None, None]
    return oracle_prior(samples, gnomad, assay, delta, rho, weights=weights)


# ── production fit ───────────────────────────────────────────────────────────

def fit_gnomad_densities(model, scores, sample_assignments, benign_method, pathogenic_idx, benign_idx, gnomad_idx,
                         synonymous_idx):
    """The fitted pathogenic and benign-reference densities at the gnomAD scores, built exactly as get_fit_prior
    builds them in standard mode (default path: no LR-filtered pathogenic mask, no pathomechanism correction)."""
    from src.assay_calibration.fit_utils.cfusn import density_utils
    from src.assay_calibration.fit_utils.point_ranges import _get_benign_reference_weights
    if benign_idx is None:
        benign_idx = synonymous_idx
    if benign_method == "synonymous":
        benign_idx = synonymous_idx
    params, weights = model["component_params"], model["weights"]
    population = scores[sample_assignments[:, gnomad_idx]]
    f_p = density_utils.joint_densities(population, params, weights[pathogenic_idx]).sum(axis=0)
    w_b = _get_benign_reference_weights(weights, benign_method, benign_idx, synonymous_idx)
    f_b = density_utils.joint_densities(population, params, w_b).sum(axis=0)
    return f_p, f_b


def raw_fit_prior(*args, initial_prior=0.5):
    """The prior get_fit_prior computes in standard mode, BEFORE its discard rule (prior <= 0.001 or >= 0.999 -> NaN),
    from the same densities and EM, so alternative rules can be tested without touching production code."""
    return prior_em(*fit_gnomad_densities(*args), discard=False, initial_prior=initial_prior)


def fit_priors(samples, seeds, num_fits=1, return_raw=False):
    """Production 2-component Gaussian fit, `num_fits` restarts per bootstrap (best by
    held-out log-likelihood, as in production), one prior per bootstrap seed. With `return_raw`, returns
    (priors, raw_priors, densities): raw_priors[start] are the estimates before the discard rule with the EM started
    at each value in EM_STARTS, and densities[k] = (f_p, f_b) at the gnomAD scores (None if the fit failed), so any
    other prior rule can be applied later without refitting."""
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    from src.assay_calibration.fit_utils.fit import Fit

    scores = np.concatenate(samples)
    sa = np.concatenate([np.full(len(s), str(i)) for i, s in enumerate(samples)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss = BasicScoreset(scores, sa)
        fitter = Fit(ss)
        # Fit weights cover only non-empty samples. With no B/LB, production
        # (pipeline/visualize.py) shifts gnomAD/synonymous down one index and
        # switches benign_method from "avg" to "synonymous".
        if len(samples[B_IDX]):
            idx = dict(benign_idx=B_IDX, gnomad_idx=G_IDX, synonymous_idx=S_IDX)
            benign_method = "avg"
        else:
            idx = dict(benign_idx=None, gnomad_idx=G_IDX - 1, synonymous_idx=S_IDX - 1)
            benign_method = "synonymous"
        out, raw, dens = [], {s0: [] for s0 in EM_STARTS}, []
        for b in seeds:
            try:
                models, best, _ = fitter.run(
                    [2], bootstrap=True, bootstrap_seed=int(b), num_fits=num_fits, master_seed=0,
                    check_monotonic=True, force_gaussian=True, core_limit=1, verbose=False,
                )
                m = models[best]
                p, _ = get_fit_prior({"fit": m}, ss.scores, benign_method, pathogenic_idx=P_IDX,
                                     sample_assignments=ss.sample_assignments, **idx)
            except Exception:
                m, p = None, np.nan
            fd = None
            if return_raw and m is not None:
                try:
                    fd = fit_gnomad_densities(m, ss.scores, ss.sample_assignments, benign_method, P_IDX, **idx)
                except Exception:
                    fd = None
            for s0 in EM_STARTS:
                raw[s0].append(np.nan if fd is None else prior_em(*fd, discard=False, initial_prior=s0))
            out.append(p)
            dens.append(fd)
    return (out, raw, dens) if return_raw else out


# ── tasks ────────────────────────────────────────────────────────────────────

def _samples_for(cell, rep):
    exp, assay, regime, pi, params, cell_seed = cell
    rng = np.random.default_rng([cell_seed, rep])
    return make_samples(rng, assay, REGIMES[regime], pi, **params)[0]


def oracle_task(cell, rep):
    exp, assay, regime, pi, params, cell_seed = cell
    samples = _samples_for(cell, rep)
    d, r = params["delta"], params["rho"]
    point = oracle_prior(samples, samples[G_IDX], assay, d, r)
    rng = np.random.default_rng([cell_seed, rep, 1])
    boots = []
    for _ in range(N_BOOT_ORACLE):
        res = [s[rng.integers(0, len(s), len(s))] if len(s) else s for s in samples]
        boots.append(oracle_prior(res, samples[G_IDX], assay, d, r))
    return {"rep": rep, "point": point, "boots": boots}


def fit_task(cell, rep, chunk):
    samples = _samples_for(cell, rep)
    seeds = range(chunk * BOOT_CHUNK, (chunk + 1) * BOOT_CHUNK)
    return {"rep": rep, "chunk": chunk, "boots": fit_priors(samples, seeds)}


def _summ(vals, prefix):
    v = np.asarray(vals, dtype=float)
    ok = v[np.isfinite(v)]
    nan = np.nan
    return {
        f"{prefix}_median": float(np.median(ok)) if len(ok) else nan,
        f"{prefix}_p5": float(np.percentile(ok, 5)) if len(ok) else nan,
        f"{prefix}_p95": float(np.percentile(ok, 95)) if len(ok) else nan,
        f"{prefix}_frac_discarded": float(1 - len(ok) / len(v)) if len(v) else nan,
        # The prior EM starts at 0.5; with no pathogenic signal in the labelled
        # sample (f_P == f_B on gnomAD) it returns 0.5 unchanged.
        f"{prefix}_frac_at_em_init": float(np.mean(np.abs(ok - 0.5) < 1e-3)) if len(ok) else nan,
    }


def build_cells():
    cells = []

    def add(exp, assay, regime, pi, **over):
        cells.append((exp, assay, regime, pi, {**BASE, **over}))

    for regime in REGIMES:
        for assay in ASSAYS:
            for pi in (0.05, 0.10, 0.25, 0.40):
                add("1_clean", assay, regime, pi)
    for assay in ASSAYS:
        for eps in (0.05, 0.10, 0.20):
            add("2_noise_plp", assay, "rich", 0.25, eps_p=eps)
        for db in (0.05, 0.10, 0.20):
            add("2_noise_blb", assay, "rich", 0.25, delta_b=db)
        for m1 in (0.0, 0.25, 0.50, 0.75, 1.0):
            add("3_mech_bias", assay, "rich", 0.25, lab_m1_frac=m1)
        for pi in (0.10, 0.25):
            for d in (1.5, 2.5, 4.0):
                add("4_separation", assay, "rich", pi, delta=d)
            for d in (2.5, 4.0):
                for rho in (0.0, 0.3, 0.6):
                    add("5_correlation", assay, "rich", pi, delta=d, rho=rho)
    return cells


def check_oracle_matches_get_fit_prior():
    """For Gaussian assays the oracle path must equal get_fit_prior exactly."""
    rng = np.random.default_rng(7)
    samples = make_samples(rng, "A1 (M1 only)", REGIMES["rich"], 0.25, **{**BASE, "rho": 0.3})[0]
    mu = class_means(4.0, 0.3)
    params = [(0.0, mu[c][0], 1.0) for c in CLASSES]
    weights = [class_weights(np.vstack([class_density(s, c, "A1 (M1 only)", 4.0, 0.3) for c in CLASSES]))
               for s in samples]
    sa = np.zeros((len(samples[G_IDX]), 4), dtype=bool)
    sa[:, G_IDX] = True
    ref, _ = get_fit_prior({"fit": {"component_params": params, "weights": weights}},
                           samples[G_IDX], "avg", sample_assignments=sa)
    ours = oracle_prior(samples, samples[G_IDX], "A1 (M1 only)", 4.0, 0.3, weights=weights)
    assert np.isclose(ref, ours), (ref, ours)
    # min() density integrates to 1.
    x = np.linspace(-15, 8, 20001)
    for c in CLASSES:
        area = np.trapz(class_density(x, c, "A3 (M1+M2)", 2.5, 0.6), x)
        assert abs(area - 1) < 1e-4, (c, area)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=64)
    ap.add_argument("--no-fit", action="store_true", help="oracle arm only")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    check_oracle_matches_get_fit_prior()

    # One independent seed per cell, so results do not depend on scheduling.
    base_cells = build_cells()
    cell_seeds = np.random.SeedSequence(args.seed).generate_state(len(base_cells))
    cells = [c + (int(s),) for c, s in zip(base_cells, cell_seeds)]

    # Large-n sanity check: oracle with 2e5 variants per sample.
    sanity = []
    for assay in ASSAYS:
        for pi in (0.05, 0.25):
            rng = np.random.default_rng(99)
            s = make_samples(rng, assay, (LARGE_N,) * 4, pi, **BASE)[0]
            sanity.append({"assay": assay, "true_prior": pi,
                           "oracle_prior": oracle_prior(s, s[G_IDX], assay, 4.0, 0.0)})
    pd.DataFrame(sanity).to_csv(args.out_dir / "sanity_large_n.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(sanity).round(4).to_string(index=False), flush=True)

    tasks = [delayed(oracle_task)(c, r) for c in cells for r in range(N_REP_ORACLE)]
    tags = [("oracle", i) for i in range(len(cells)) for _ in range(N_REP_ORACLE)]
    if not args.no_fit:
        n_chunks = N_BOOT_FIT // BOOT_CHUNK
        tasks += [delayed(fit_task)(c, r, k) for c in cells for r in range(N_REP_FIT) for k in range(n_chunks)]
        tags += [("fit", i) for i in range(len(cells)) for _ in range(N_REP_FIT) for _ in range(n_chunks)]
    print(f"{len(cells)} cells, {len(tasks)} tasks", flush=True)
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(tasks)

    by_cell = {}
    for (arm, i), res in zip(tags, results):
        by_cell.setdefault(i, {"oracle": [], "fit": []})[arm].append(res)

    rows, rep_rows = [], []
    for i, (exp, assay, regime, pi, params, _) in enumerate(cells):
        orc = by_cell[i]["oracle"]
        fit = by_cell[i]["fit"]
        fit_reps = {}
        for r in fit:
            fit_reps.setdefault(r["rep"], []).extend(r["boots"])
        row = {
            "experiment": exp, "assay": assay, "regime": regime,
            "n_plp, n_blb, n_gnomad, n_syn": ", ".join(map(str, REGIMES[regime])),
            "true_prior": pi, **params,
            "expected_prior": expected_prior(assay, pi, has_benign=REGIMES[regime][1] > 0, **params),
            "oracle_point_median": float(np.nanmedian([r["point"] for r in orc])),
            **_summ([b for r in orc for b in r["boots"]], "oracle_boot"),
        }
        if fit:
            per_rep = {rep: (np.nanmedian(v) if np.isfinite(v).any() else np.nan)
                       for rep, v in fit_reps.items()}
            row.update({"fit_rep_medians": ";".join(f"{p:.3f}" for p in per_rep.values()),
                        **_summ([b for v in fit_reps.values() for b in v], "fit_boot")})
            for rep, p in per_rep.items():
                rep_rows.append({"cell": i, "experiment": exp, "assay": assay, "regime": regime,
                                 "true_prior": pi, **params, "rep": rep, "fit_prior_median": p})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "results.csv", index=False, float_format="%.4f")
    if rep_rows:
        pd.DataFrame(rep_rows).to_csv(args.out_dir / "fit_reps.csv", index=False, float_format="%.4f")
    print(f"wrote {args.out_dir / 'results.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()


# ── closed-form theory (Supplementary Section "Prior estimation: theory and simulations") ─────────────────────

# Fraction of pathogenic population variants each assay detects (rho = 0): M1-only reads M1, M2-only reads M2, the
# combined assay reads both. Pathogenic variants acting through an unmeasured mechanism score like benign variants.
DETECTED = {"A1 (M1 only)": POP_M1_FRAC, "A2 (M2 only)": 1 - POP_M1_FRAC, "A3 (M1+M2)": 1.0}


def predicted_prior(experiment, assay, pi, eps_p=0.0, delta_b=0.0, lab_m1_frac=POP_M1_FRAC):
    """Large-sample limit of the estimate from the mixture identity (rho = 0; benign reference = average of B/LB and
    synonymous). Solving p_G = a p_SP + (1 - a) p_ref for a, with each labeled density a mixture of the true class
    densities:
      benign variants among P/LP (eps):       a = pi / (1 - eps)
      pathogenic variants among B/LB (delta): a = (pi - delta/2) / (1 - delta/2)
      P/LP mechanism mix m (fraction via M1): a = pi * w_pop / w_lab for a single-mechanism assay (w = fraction of
                                              pathogenic variants it detects), and pi for the combined assay, whose
                                              score distribution is the same for M1 and M2 variants.
    NaN where the prior is not identified (w_lab = 0); values are clipped to [0, 1]."""
    if experiment == "2_noise_plp":
        a = pi / (1 - eps_p)
    elif experiment == "2_noise_blb":
        a = (pi - delta_b / 2) / (1 - delta_b / 2)
    elif experiment == "3_mech_bias":
        if assay == "A3 (M1+M2)":
            a = pi
        else:
            w_pop = DETECTED[assay]
            w_lab = lab_m1_frac if assay == "A1 (M1 only)" else 1 - lab_m1_frac
            if w_lab == 0:
                return np.nan
            a = pi * w_pop / w_lab
    else:
        a = pi
    return float(min(max(a, 0.0), 1.0))


def fisher_information(assay, pi, delta=4.0, rho=0.0):
    """I(pi) = integral of (p_P - p_B)^2 / p_G: the information per population variant about the prior when the
    class densities are known. p_P is the population mix of M1 and M2 pathogenic densities."""
    from scipy.integrate import quad
    d = lambda x, c: class_density(np.atleast_1d(x), c, assay, delta, rho)[0]
    p_p = lambda x: POP_M1_FRAC * d(x, "M1") + (1 - POP_M1_FRAC) * d(x, "M2")
    p_g = lambda x: pi * p_p(x) + (1 - pi) * d(x, "B")
    lo, hi = -delta - 10, 8
    return quad(lambda x: (p_p(x) - d(x, "B")) ** 2 / p_g(x), lo, hi, limit=400)[0]


def predicted_sd(assay, pi, n_gnomad, n_plp, delta=4.0, rho=0.0):
    """Asymptotic SD of the estimate: a population term 1/(n_G I) plus, for an assay detecting a fraction w of
    pathogenic variants, the delta-method term pi^2 (1 - w) / (n_P w) from estimating w from the P/LP sample."""
    w = DETECTED[assay]
    var = 1 / (n_gnomad * fisher_information(assay, pi, delta, rho)) + pi ** 2 * (1 - w) / (n_plp * w)
    return float(np.sqrt(var))


def predicted_plp_bias(assay, pi, n_plp):
    """Second-order (delta-method) bias from a small P/LP sample: the estimate behaves as pi * w / w_hat with
    w_hat ~ Binomial(n_P, w) / n_P, and E[1/w_hat] ~ (1/w)(1 + (1 - w)/(n_P w)), so bias ~ pi (1 - w) / (n_P w).
    An approximation, accurate when n_P w is not small."""
    w = DETECTED[assay]
    return float(pi * (1 - w) / (n_plp * w))
