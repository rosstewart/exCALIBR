#!/usr/bin/env python3
"""
EXPERIMENTAL, non-production fitting scheme: instead of one pooled/unsupervised EM
fitting all K skew-normal components jointly across every sample class, fit
component *shapes* in a class-restricted, staged way --

  1. Exactly 1 "functionally normal" component, fit using ONLY B/LB + Synonymous rows.
  2. The remaining (n-1) components, fit using ONLY P/LP rows (optionally augmented
     with gnomAD/population rows that are poorly explained by the already-fit benign
     component -- candidate "hidden pathogenic" variants, to help when P/LP counts
     are low).
  3. gnomAD (and every other class, including P/LP and B/LB/Synonymous themselves)
     then gets its own mixing-weight vector refit via a frozen-params E-step against
     the FULL combined n-component set, so scoring isn't hard-zero-padded -- a P/LP
     variant that actually looks benign-like can still pull weight toward the benign
     component.

Rationale: the production EM (src/assay_calibration/fit_utils/cfusn/fit.py::
single_fit) fits ALL components from a single pooled M-step across every class's
rows, regardless of K -- so a subtle, low-magnitude region where a handful of P/LP
variants genuinely (if faintly) separate from B/LB gets absorbed into whichever
component also explains the bulk of "normal" variation, no matter how many
components are added. This module dedicates model capacity specifically to disease
signal by never letting the benign bulk influence the pathogenic components' shape
(and vice versa), then reconciles the two via a shared mixing-weight refit.

NOT wired into hpc/prepare.py or any production job-generation path -- purely for
ad hoc/manual investigation.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.assay_calibration.fit_utils.cfusn.fit import single_fit
from src.assay_calibration.fit_utils.cfusn.initializations import (
    partial_distance_kmeans_labels, kmeans_init_mv,
)
from src.assay_calibration.fit_utils.cfusn.update_steps import get_sample_weights
from src.assay_calibration.fit_utils.cfusn.density_utils import get_likelihood
from src.assay_calibration.fit_utils.fit import (
    _unstandardize_component_params, pattern_stratified_bootstrap,
)
from src.assay_calibration.multivariate_analysis.mv_calibration import _sn_logpdf

ROLE_NAMES = ["P/LP", "B/LB", "gnomAD", "Synonymous"]


def _fit_one_group(observations, row_mask, n_components, init_strategy, latent_q,
                    num_fits=1, bootstrap_seed=None, fit_seed_base=None,
                    cluster_labels=None, initial_params=None, initial_weights=None,
                    init_fn=None):
    """Fits n_components on observations[row_mask] as a single merged
    "training pool" class (one indicator column, all True). If bootstrap_seed
    is given, resamples train/held-out rows first (pattern_stratified_bootstrap
    degenerates to a plain with-replacement bootstrap for a single class). If
    num_fits > 1, repeats with different fit_seeds and keeps the restart with
    the best held-out (or, if no bootstrap, training) log-likelihood -- mirrors
    hpc/prepare.py's NUM_FITS-restarts-pick-best-by-val_ll convention.

    ``cluster_labels``, if given (one int per row of observations[row_mask]),
    is used ONLY to build the stratification array for the bootstrap resample
    (a one-hot (n, K) matrix in place of the single merged column) -- this
    guarantees each cluster's row count is preserved (only which specific rows
    get redrawn varies) across every bootstrap replicate, so a small minority
    sub-cluster is never fully washed out. It is NOT used for the EM fit's own
    class indicator below, which stays a single pooled column regardless --
    the actual component fit is still one joint EM across all rows, letting
    soft responsibilities (not these pre-computed hard labels) decide which
    output component each row loads onto.

    ``initial_params``/``initial_weights``, if given, are a SINGLE fixed
    starting point reused identically for every restart -- since `single_fit`
    skips its own internal kmeans call whenever `initial_params` is supplied
    (cfusn/fit.py:109-121), this means `num_fits` restarts only vary via
    downstream RNG-dependent EM steps (e.g. the MC-truncated E-step under
    `latent_q>1`), NOT via genuinely different starting points -- a real bug
    found in `fit_all_assayed_mixture`'s staged_init path (restarts were
    silently not diverse; a `latent_q=1` fit showed 2 restarts failing at the
    IDENTICAL EM iteration, proving the trajectory was fully deterministic).

    ``init_fn``, if given, takes precedence over the fixed
    ``initial_params``/``initial_weights`` above: called once per restart as
    ``init_fn(i)`` -> ``(initial_params_i, initial_weights_i)``, so each
    restart gets an actually distinct starting point (e.g. a fresh
    kmeans_init_mv call seeded off the restart index). This is how
    restart-vs-bootstrap-replicate seed independence should be preserved:
    have the caller's `init_fn` derive its RNG seed from `i` alone (not from
    `bootstrap_seed`), so restart 0 is the same reference point across every
    bootstrap replicate (isolating resampling variability in the bootstrap
    distribution, the original intent) while restarts 0..num_fits-1 within
    any one replicate are genuinely different starting points.

    Returns (component_params, best_val_ll).
    """
    obs = observations[row_mask]
    n = obs.shape[0]
    indicators = np.ones((n, 1), dtype=bool)

    if bootstrap_seed is not None:
        if cluster_labels is not None:
            K = int(cluster_labels.max()) + 1
            strat = np.zeros((n, K), dtype=bool)
            strat[np.arange(n), cluster_labels] = True
        else:
            strat = indicators
        train_idx, val_idx = pattern_stratified_bootstrap(obs, strat, bootstrap_seed)
    else:
        train_idx, val_idx = np.arange(n), np.array([], dtype=int)

    base_fit_kwargs = dict(latent_q=latent_q, verbose=False)
    if init_fn is None and initial_params is not None:
        base_fit_kwargs["initial_params"] = initial_params
        base_fit_kwargs["initial_weights"] = initial_weights

    best_params, best_val_ll = None, -np.inf
    for i in range(num_fits):
        fit_seed = None if fit_seed_base is None else fit_seed_base * 10_000 + i
        fit_kwargs = dict(base_fit_kwargs)
        if init_fn is not None:
            init_params_i, init_weights_i = init_fn(i)
            fit_kwargs["initial_params"] = init_params_i
            fit_kwargs["initial_weights"] = init_weights_i
        try:
            result = single_fit(
                obs[train_idx], indicators[train_idx],
                n_components, False, init_strategy, "scale", multivariate=True,
                fit_seed=fit_seed, **fit_kwargs,
            )
        except Exception as e:
            # single_fit's default raise_on_error is now True -- a genuine
            # likelihood decrease/NaN fails loudly instead of silently
            # substituting a truncated pre-decrease iterate. A multi-restart
            # caller like this one should still treat one bad restart as
            # "try the next seed" rather than aborting the whole group's fit
            # (matching tryToFit's catch-and-skip convention in
            # src/assay_calibration/fit_utils/fit.py), so it's caught here.
            warnings.warn(f"single_fit restart {i} failed: {e}")
            continue
        params = result["component_params"]
        if not params or any(len(p) == 0 for p in params):
            continue
        eval_obs, eval_ind = (obs[val_idx], indicators[val_idx]) if len(val_idx) else (obs[train_idx], indicators[train_idx])
        val_ll = get_likelihood(eval_obs, eval_ind, params, result["weights"], multivariate=True)
        val_ll = val_ll / max(len(eval_obs), 1)
        if val_ll > best_val_ll:
            best_val_ll, best_params = val_ll, params

    if best_params is None:
        raise RuntimeError(
            f"All {num_fits} restart(s) failed to converge for this group "
            f"({row_mask.sum()} rows, K={n_components})."
        )
    return best_params, best_val_ll


def _converge_sample_weights(observations, sample_indicators, component_params,
                              seed_weights, tol=1e-6, max_iters=50):
    """Iterate get_sample_weights (a single frozen-params E-step) to
    convergence -- no existing loop-to-convergence wrapper for weights alone,
    since production EM only ever calls it once per M-step iteration inside
    the full EM loop. Returns (weights, n_iters_used).

    A class with zero rows (e.g. a gene with no labeled P/LP variants) has no
    evidence to inform its weight vector -- get_sample_weights (`update_steps.py:798`,
    `posts.mean(1)` over an empty array) raises "NaN weight" on it. Excluded
    from the iterative update below and left at its uniform seed value instead
    of erroring the whole reweighting step over one empty class."""
    n_classes = sample_indicators.shape[1]
    nonempty = np.array([sample_indicators[:, i].any() for i in range(n_classes)])
    if not nonempty.all():
        weights = seed_weights.copy()
        sub_weights, n_iters = _converge_sample_weights(
            observations, sample_indicators[:, nonempty], component_params,
            seed_weights[nonempty], tol=tol, max_iters=max_iters,
        )
        weights[nonempty] = sub_weights
        return weights, n_iters

    weights = seed_weights.copy()
    for it in range(max_iters):
        new_weights = get_sample_weights(
            observations, sample_indicators, component_params, weights, multivariate=True)
        delta = np.max(np.abs(new_weights - weights))
        weights = new_weights
        if delta < tol:
            return weights, it + 1
    return weights, max_iters


def _compact_weights_to_effective(weights, sa, n_roles):
    """Drop rows for roles with zero observations, matching the "effective
    index" convention `MVCalibrationAnalysis.__init__` expects (`mv_calibration.py:
    790-802`, `ms.sample_counts`-based -- empty-sample columns are dropped
    entirely, not zero-filled, so every later role's weight row shifts down by
    one for each earlier absent role). Our own weights matrix is always built
    in fixed 0..n_roles-1 role order (see `_converge_sample_weights`'s
    "leave empty classes at their uniform seed value" placeholder) -- without
    this compaction, a gene missing e.g. P/LP (row 0) would have every other
    role's weights read one row off from where the scoring code expects them,
    silently corrupting the NU/PU population-unmixing math (confirmed this
    session: manifested as an "invalid prior nan" on every bootstrap for
    zero-P/LP genes)."""
    present = [r for r in range(n_roles) if sa[:, r].any()]
    return weights[present]


def fit_staged_mixture(
    ms, n_components,
    benign_roles=(1, 3), pathogenic_role=0, population_role=2,
    outlier_density_percentile=10.0,
    latent_q=2, init_strategy="kmeans", seed=None,
    weight_max_iters=50, weight_tol=1e-6,
    num_fits=1, bootstrap_seed=None,
    cluster_stratify_pathogenic=False, cluster_seed=0,
):
    """Returns a fit_raw-shaped dict: {"component_params": [[mu, Delta, Gamma], ...]
    (raw units, n_components total, benign component first), "weights": (4, n)
    array in fixed role order [P/LP, B/LB, gnomAD, Synonymous], "latent_q": ...,
    plus diagnostic counts} -- directly consumable by MVCalibrationAnalysis once
    written to a small results.json.gz (see mv_analysis.report/gene_3d_evidence's
    existing fast_results_json/build_gene_set_analysis pattern for how to load it).

    ``n_components``: total components (1 benign + n_components-1 pathogenic).
    ``outlier_density_percentile``: population (gnomAD) rows scoring in the bottom
    X% by log-density under the fitted benign component are treated as candidate
    "hidden pathogenic" rows and unioned into the pathogenic-component training set
    (robustness fallback for low P/LP counts). Set to 0 to disable (pathogenic fit
    uses P/LP rows only).
    ``num_fits``: restarts per group (benign, pathogenic), best kept by held-out
    (or, if bootstrap_seed is None, training) log-likelihood -- mirrors
    hpc/prepare.py's multi-restart convention.
    ``bootstrap_seed``: if given, both group fits AND the final weight-
    reestimation step are run on a `pattern_stratified_bootstrap` resample of
    their respective rows (train split only for weights, matching how
    production EM only ever sees the bootstrap's train split) -- pass a
    different seed per bootstrap replicate to build a real multi-bootstrap
    ensemble (see `fit_staged_mixture_bootstrap`).
    ``cluster_stratify_pathogenic``: if True (and there are >=2 pathogenic
    components), partitions the pathogenic-fit row pool into `n_components-1`
    reference clusters via `partial_distance_kmeans_labels` (the same
    NaN-tolerant kmeans EM initialization already uses), computed ONCE from
    `cluster_seed` (fixed across every bootstrap replicate -- NOT derived from
    `bootstrap_seed`), and stratifies the pathogenic group's bootstrap
    resample by that cluster label (in addition to missingness pattern). This
    guarantees a small minority sub-cluster (e.g. a rare disease mechanism)
    keeps its full row count in every replicate -- only which specific rows
    within it get redrawn varies -- rather than risking being fully absent
    from some bootstrap replicates by chance.
    """
    if n_components < 2:
        raise ValueError("n_components must be >= 2 (1 benign + >=1 pathogenic)")

    sa = ms._sample_assignments.astype(bool)
    observations_raw = np.asarray(ms.scores, dtype=float)

    # Standardize once globally, matching hpc/prepare.py's Fit.generate_fit_jobs
    # convention (fit.py:735-740) -- undone at the very end via
    # _unstandardize_component_params so callers always get raw-unit params.
    scale_mean = np.nanmean(observations_raw, axis=0)
    scale_std = np.nanstd(observations_raw, axis=0)
    scale_std = np.where(scale_std < 1e-8, 1.0, scale_std)
    observations = (observations_raw - scale_mean) / scale_std

    seed_b = None if seed is None else seed
    seed_p = None if seed is None else seed + 1

    # ── 1. Benign component (K=1), B/LB + Synonymous rows only ──
    # single_fit requires a strictly one-hot indicator matrix per row (each row
    # sums to exactly 1) -- B/LB and Synonymous are NOT guaranteed mutually
    # exclusive (e.g. a variant can be both synonymous and separately observed
    # in gnomAD/other roles), so merge them into one training-pool column
    # rather than passing 2 separate columns (mirrors the pathogenic group's
    # single merged column below).
    benign_row_mask = sa[:, list(benign_roles)].any(axis=1)
    component_params_benign, benign_val_ll = _fit_one_group(
        observations, benign_row_mask, 1, init_strategy, latent_q,
        num_fits=num_fits, bootstrap_seed=bootstrap_seed, fit_seed_base=seed_b,
    )
    mu_b, Delta_b, Gamma_b = component_params_benign[0]

    # ── 2. Outlier scoring: population rows poorly explained by the benign fit ──
    pop_mask = sa[:, population_role]
    n_outlier_rows = 0
    outlier_global_idx = np.array([], dtype=int)
    if outlier_density_percentile > 0 and pop_mask.any():
        log_dens_pop = _sn_logpdf(observations[pop_mask], mu_b, Delta_b, Gamma_b)
        finite = np.isfinite(log_dens_pop)
        if finite.any():
            threshold = np.nanpercentile(log_dens_pop[finite], outlier_density_percentile)
            outlier_mask_within_pop = finite & (log_dens_pop <= threshold)
            outlier_global_idx = np.where(pop_mask)[0][outlier_mask_within_pop]
            n_outlier_rows = len(outlier_global_idx)

    # ── 3. Pathogenic components (K=n-1), P/LP rows UNION outlier population rows ──
    path_mask = sa[:, pathogenic_role].copy()
    n_true_path_rows = int(path_mask.sum())
    path_mask[outlier_global_idx] = True

    cluster_labels = None
    pathogenic_cluster_sizes = None
    if cluster_stratify_pathogenic and n_components - 1 >= 2:
        cluster_labels, _ = partial_distance_kmeans_labels(
            observations[path_mask], n_clusters=n_components - 1,
            rng=np.random.RandomState(cluster_seed),
        )
        pathogenic_cluster_sizes = np.bincount(cluster_labels, minlength=n_components - 1).tolist()

    component_params_pathogenic, path_val_ll = _fit_one_group(
        observations, path_mask, n_components - 1, init_strategy, latent_q,
        num_fits=num_fits, bootstrap_seed=bootstrap_seed, fit_seed_base=seed_p,
        cluster_labels=cluster_labels,
    )

    component_params = list(component_params_benign) + list(component_params_pathogenic)

    # ── 4. Reweight every class against the FULL combined n components ──
    n_roles = min(sa.shape[1], len(ROLE_NAMES))
    if bootstrap_seed is not None:
        # Resample rows the same way the group fits were, so weight
        # uncertainty across bootstrap replicates is real, not just inherited
        # from the (already-resampled) component shapes.
        train_idx, _ = pattern_stratified_bootstrap(observations, sa[:, :n_roles], bootstrap_seed)
        reweight_obs = observations[train_idx]
        all_role_indicators = sa[train_idx][:, :n_roles]
    else:
        reweight_obs = observations
        all_role_indicators = sa[:, :n_roles]
    seed_weights = np.ones((n_roles, n_components)) / n_components
    weights, n_weight_iters = _converge_sample_weights(
        reweight_obs, all_role_indicators, component_params, seed_weights,
        tol=weight_tol, max_iters=weight_max_iters,
    )

    component_params_raw = _unstandardize_component_params(component_params, scale_mean, scale_std)
    component_params_json = [
        [np.asarray(mu).tolist(), np.asarray(Delta).tolist(), np.asarray(Gamma).tolist()]
        for mu, Delta, Gamma in component_params_raw
    ]
    weights = _compact_weights_to_effective(weights, sa, n_roles)

    return {
        "component_params": component_params_json,
        "weights": weights.tolist(),
        "latent_q": latent_q,
        "n_benign_components": 1,
        "n_pathogenic_components": n_components - 1,
        "n_benign_rows": int(benign_row_mask.sum()),
        "n_true_pathogenic_rows": n_true_path_rows,
        "n_outlier_rows_added": n_outlier_rows,
        "n_pathogenic_rows_total": int(path_mask.sum()),
        "n_weight_reestimation_iters": n_weight_iters,
        "benign_val_ll": float(benign_val_ll),
        "pathogenic_val_ll": float(path_val_ll),
        "pathogenic_cluster_sizes": pathogenic_cluster_sizes,
    }


def fit_staged_mixture_bootstrap(ms, n_components, n_bootstraps=100, num_fits=8,
                                  master_seed=0, **kwargs):
    """Fits `fit_staged_mixture` across `n_bootstraps` independent bootstrap
    replicates (each with `num_fits` restarts per group, matching production
    convention), returning {str(seed): fit_raw} -- assemble into
    {dataset_label: {seed: {config_label: fit_raw}}} and write to a
    results.json.gz for MVCalibrationAnalysis to consume with proper
    bootstrap percentile statistics."""
    results = {}
    n_failed = 0
    for b in range(n_bootstraps):
        try:
            results[str(b)] = fit_staged_mixture(
                ms, n_components, num_fits=num_fits, bootstrap_seed=b,
                seed=master_seed * 1_000_000 + b, **kwargs,
            )
        except RuntimeError as e:
            n_failed += 1
            print(f"  bootstrap {b} failed: {e}")
    print(f"Completed {len(results)}/{n_bootstraps} bootstraps ({n_failed} failed)")
    return results


def fit_all_assayed_mixture(
    ms, n_components, all_assayed_role=4,
    latent_q=2, init_strategy="kmeans", seed=None,
    weight_max_iters=50, weight_tol=1e-6,
    num_fits=1, bootstrap_seed=None,
    staged_init=False, benign_roles=(1, 3), pathogenic_role=0, init_seed=0,
):
    """Fits n_components unsupervised on ms's 'all_assayed' regularization sample
    (every assayed variant regardless of clinical label -- requires ms built with
    `regularization_type='all_assayed'`, see `build_labelseq_multiscoresets`),
    then reweights the 4 canonical classes (P/LP, B/LB, gnomAD, Synonymous)
    post-hoc against those fixed components via a frozen-params E-step.

    ``staged_init``: if True, replaces the default kmeans-on-all_assayed-itself
    initialization with a staged-style one -- 1 component's initial (mu, Delta,
    Gamma) computed via `kmeans_init_mv(n_clusters=1)` on the B/LB+Synonymous
    rows, and the remaining `n_components - 1` via `kmeans_init_mv(n_clusters=
    n_components - 1)` on the P/LP rows alone -- then EM proceeds completely
    normally on the FULL (resampled) all_assayed pool, exactly as without this
    flag. This only changes the starting point handed to `single_fit` (via its
    `initial_params`/`initial_weights` override, `cfusn/fit.py:109-121`), not
    which rows the EM itself trains on. `init_seed` is the base seed for restart
    0's reference initialization and is fixed ACROSS bootstrap replicates
    (like `cluster_seed` in `fit_staged_mixture`) -- so for a given restart
    index i, every bootstrap replicate starts from the identical i-th
    reference point, isolating resampling variability in the bootstrap
    distribution. But restart index i itself now DOES get its own seed
    (`init_seed + i`, via `_fit_one_group`'s `init_fn` hook) -- previously
    `initial_params` was computed ONCE with the fixed `init_seed` and reused
    unchanged across every restart, so `num_fits` restarts within one
    bootstrap replicate were bitwise-identical starting points (a real bug:
    a `latent_q=1` sanity check showed 2 "different" restarts failing at the
    exact same EM iteration, since with no MC-integration noise the
    trajectory was fully deterministic from an identical start). Restarts
    now genuinely explore different initializations, while staying
    reproducible/comparable across bootstrap replicates.

    Unlike `fit_staged_mixture`, there is no benign/pathogenic split or outlier
    routing -- this tests a different hypothesis: that gnomAD's disproportionate
    row count *within the 4 labeled classes* dominates today's pooled EM's
    shared shape-fitting sums (confirmed: `em_iteration`'s M-step weights every
    row by responsibility alone, no per-class normalization by default), and
    that fitting shapes instead on the much larger, label-agnostic all_assayed
    pool might resolve subtle localized structure that gnomAD's presence
    otherwise washes out.
    """
    sa = ms._sample_assignments.astype(bool)
    observations_raw = np.asarray(ms.scores, dtype=float)

    scale_mean = np.nanmean(observations_raw, axis=0)
    scale_std = np.nanstd(observations_raw, axis=0)
    scale_std = np.where(scale_std < 1e-8, 1.0, scale_std)
    observations = (observations_raw - scale_mean) / scale_std

    init_fn = None
    if staged_init:
        benign_mask = sa[:, list(benign_roles)].any(axis=1)
        path_mask = sa[:, pathogenic_role]

        def init_fn(i, benign_mask=benign_mask, path_mask=path_mask):
            seed_i = init_seed + i
            benign_params, _ = kmeans_init_mv(
                observations[benign_mask], n_clusters=1, constrained=False,
                latent_q=latent_q, rng=np.random.RandomState(seed_i),
            )
            path_params, _ = kmeans_init_mv(
                observations[path_mask], n_clusters=n_components - 1, constrained=False,
                latent_q=latent_q, rng=np.random.RandomState(seed_i),
            )
            initial_params_i = list(benign_params) + list(path_params)
            initial_weights_i = np.ones((1, n_components)) / n_components
            return initial_params_i, initial_weights_i

    all_assayed_mask = sa[:, all_assayed_role]
    component_params, val_ll = _fit_one_group(
        observations, all_assayed_mask, n_components, init_strategy, latent_q,
        num_fits=num_fits, bootstrap_seed=bootstrap_seed, fit_seed_base=seed,
        init_fn=init_fn,
    )

    n_roles = min(sa.shape[1], len(ROLE_NAMES))
    if bootstrap_seed is not None:
        train_idx, _ = pattern_stratified_bootstrap(observations, sa[:, :n_roles], bootstrap_seed)
        reweight_obs = observations[train_idx]
        all_role_indicators = sa[train_idx][:, :n_roles]
    else:
        reweight_obs = observations
        all_role_indicators = sa[:, :n_roles]
    seed_weights = np.ones((n_roles, n_components)) / n_components
    weights, n_weight_iters = _converge_sample_weights(
        reweight_obs, all_role_indicators, component_params, seed_weights,
        tol=weight_tol, max_iters=weight_max_iters,
    )

    component_params_raw = _unstandardize_component_params(component_params, scale_mean, scale_std)
    component_params_json = [
        [np.asarray(mu).tolist(), np.asarray(Delta).tolist(), np.asarray(Gamma).tolist()]
        for mu, Delta, Gamma in component_params_raw
    ]
    weights = _compact_weights_to_effective(weights, sa, n_roles)

    return {
        "component_params": component_params_json,
        "weights": weights.tolist(),
        "latent_q": latent_q,
        "n_components": n_components,
        "n_all_assayed_rows": int(all_assayed_mask.sum()),
        "n_weight_reestimation_iters": n_weight_iters,
        "fit_val_ll": float(val_ll),
    }


def fit_all_assayed_mixture_bootstrap(ms, n_components, n_bootstraps=100, num_fits=8,
                                       master_seed=0, **kwargs):
    """Same bulk-bootstrap convention as `fit_staged_mixture_bootstrap`, for
    `fit_all_assayed_mixture`."""
    results = {}
    n_failed = 0
    for b in range(n_bootstraps):
        try:
            results[str(b)] = fit_all_assayed_mixture(
                ms, n_components, num_fits=num_fits, bootstrap_seed=b,
                seed=master_seed * 1_000_000 + b, **kwargs,
            )
        except RuntimeError as e:
            n_failed += 1
            print(f"  bootstrap {b} failed: {e}")
    print(f"Completed {len(results)}/{n_bootstraps} bootstraps ({n_failed} failed)")
    return results
