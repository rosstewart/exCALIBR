"""Glue between the existing job-pickle format (``fit.py``/``run_array_task.py``)
and the batched JAX fitters in ``batch_em.py`` / ``batch_em_cfusn.py``.

Initialization (kmeans / method-of-moments) is **not** batched — it stays on
CPU, one call per job via the existing NumPy routines in
``cfusn/initializations.py``, mirroring ``cfusn/fit.py::single_fit``'s init
block. It's cheap relative to the thousands of EM iterations that follow, so
this isn't a bottleneck; only the EM loop itself is moved to the GPU.

Only supports the common case each `prepare.py` invocation actually produces:
one `constrained`/`force_gaussian`/`multivariate` setting per (dataset,
num_components) group. `group_fit_specs` asserts this and raises if a group
turns out to be mixed (this would indicate the manifest was hand-edited or
a job dict is malformed) rather than silently mishandling it.
"""
from collections import defaultdict

import numpy as np
import jax.numpy as jnp

from ..cfusn.initializations import (
    kmeans_init, methodOfMomentsInit, kmeans_init_mv, kmeans_init_mv_anchored,
)
from ..cfusn.update_steps import get_sample_weights
from ..cfusn.fit import _ensure_cfusn_params
from . import batch_em, batch_em_cfusn

MAX_BATCH_UNIVARIATE = 20_000
MAX_BATCH_CFUSN = 500


# ---------------------------------------------------------------------------
# Grouping / chunking
# ---------------------------------------------------------------------------

def group_fit_specs(fit_specs, max_batch_univariate=MAX_BATCH_UNIVARIATE,
                     max_batch_cfusn=MAX_BATCH_CFUSN):
    """fit_specs: list of (full_job, bs_seed, label, save_dir) as built by
    run_array_task.py. Returns list of (multivariate, chunk) where chunk is a
    sub-list of fit_specs safe to run as one batched GPU call.
    """
    groups = defaultdict(list)
    for spec in fit_specs:
        full_job = spec[0]
        is_mv = bool(full_job.get("multivariate", False))
        key = (full_job["dataset_name"], spec[2], full_job["num_components"], is_mv)
        groups[key].append(spec)

    chunks = []
    for (dataset_name, label, num_components, is_mv), specs in groups.items():
        constrained_vals = {s[0]["constrained"] for s in specs}
        force_gaussian_vals = {s[0].get("kwargs", {}).get("force_gaussian", False) for s in specs}
        if len(constrained_vals) > 1 or len(force_gaussian_vals) > 1:
            raise ValueError(
                f"group_fit_specs: mixed constrained/force_gaussian within "
                f"({dataset_name}, {label}) — GPU batching requires a uniform "
                f"setting per (dataset, num_components) group."
            )
        max_batch = max_batch_cfusn if is_mv else max_batch_univariate
        for i in range(0, len(specs), max_batch):
            chunks.append((is_mv, specs[i:i + max_batch]))
    return chunks


# ---------------------------------------------------------------------------
# CPU-side initialization (per job, cheap — see module docstring)
# ---------------------------------------------------------------------------

def _init_univariate(full_job):
    kwargs = dict(full_job.get("kwargs", {}))
    kwargs["rng"] = np.random.RandomState(kwargs.get("fit_seed"))
    K = full_job["num_components"]
    constrained = full_job["constrained"]
    observations = np.asarray(full_job["train_observations"]).ravel()
    sample_assignments = full_job["train_sample_assignments"]

    initial_params = None
    if full_job["init_method"] == "method_of_moments":
        initial_params = methodOfMomentsInit(
            observations, K, constrained,
            init_constraint_adjustment=full_job["init_constraint_adjustment"], **kwargs
        )
    if initial_params is None:
        initial_params, _ = kmeans_init(
            observations, n_clusters=K, constrained=constrained,
            init_constraint_adjustment=full_job["init_constraint_adjustment"], **kwargs
        )
    if not initial_params or any(len(p) == 0 for p in initial_params):
        return None

    W = np.ones((sample_assignments.shape[1], K)) / K
    W = get_sample_weights(observations, sample_assignments, initial_params, W, multivariate=False)
    return initial_params, W


def _init_cfusn(full_job):
    kwargs = dict(full_job.get("kwargs", {}))
    kwargs["rng"] = np.random.RandomState(kwargs.get("fit_seed"))
    K = full_job["num_components"]
    constrained = full_job["constrained"]
    observations = full_job["train_observations"]
    sample_assignments = full_job["train_sample_assignments"]

    if full_job["init_method"] == "anchored":
        initial_params, _ = kmeans_init_mv_anchored(
            observations, sample_assignments, n_clusters=K, constrained=constrained,
            init_constraint_adjustment=full_job["init_constraint_adjustment"], **kwargs
        )
    else:
        initial_params, _ = kmeans_init_mv(
            observations, n_clusters=K, constrained=constrained,
            init_constraint_adjustment=full_job["init_constraint_adjustment"], **kwargs
        )
    if not initial_params or any(len(p) == 0 for p in initial_params):
        return None

    latent_q = kwargs.get("latent_q", 2)
    if latent_q > 1:
        initial_params = _ensure_cfusn_params(initial_params, latent_q)
    W = np.ones((sample_assignments.shape[1], K)) / K
    W = get_sample_weights(observations, sample_assignments, initial_params, W, multivariate=True)
    return initial_params, W


# ---------------------------------------------------------------------------
# Univariate batch: pack / run / unpack
# ---------------------------------------------------------------------------

def _run_univariate_chunk(specs):
    # NB: don't use `spec in valid_specs` anywhere here — specs are tuples
    # containing job dicts with ndarray values, and `==`/`in` on those raises
    # ValueError ("truth value of an array is ambiguous"). Track validity by
    # index instead.
    inits = []
    valid_specs = []
    results = []
    for spec in specs:
        full_job = spec[0]
        init = _init_univariate(full_job)
        if init is None:
            results.append(_failed_result(spec))  # init failure -> val_ll=-inf
            continue
        inits.append(init)
        valid_specs.append(spec)

    if not valid_specs:
        return results

    K = valid_specs[0][0]["num_components"]
    N = len(valid_specs[0][0]["train_observations"])
    S = valid_specs[0][0]["train_sample_assignments"].shape[1]
    constrained = bool(valid_specs[0][0]["constrained"])
    force_gaussian = bool(valid_specs[0][0].get("kwargs", {}).get("force_gaussian", False))

    obs = np.stack([np.asarray(s[0]["train_observations"]).ravel() for s in valid_specs])
    sample_idx = np.stack([
        np.argmax(s[0]["train_sample_assignments"], axis=1) for s in valid_specs
    ])
    a0 = np.stack([[p[0] for p in init[0]] for init in inits])
    loc0 = np.stack([[p[1] for p in init[0]] for init in inits])
    scale0 = np.stack([[p[2] for p in init[0]] for init in inits])
    W0 = np.stack([init[1] for init in inits])
    xmin = obs.min(axis=1)
    xmax = obs.max(axis=1)

    a, loc, scale, W, failed = batch_em.fit_batch(
        jnp.asarray(obs), jnp.asarray(sample_idx), S,
        jnp.asarray(a0), jnp.asarray(loc0), jnp.asarray(scale0), jnp.asarray(W0),
        jnp.asarray(xmin), jnp.asarray(xmax),
        constrained=constrained, force_gaussian=force_gaussian,
    )
    a, loc, scale, W, failed = map(np.asarray, (a, loc, scale, W, failed))

    for i, spec in enumerate(valid_specs):
        full_job, bs_seed, label, save_dir = spec
        if failed[i]:
            results.append(_failed_result(spec))
            continue
        component_params = [
            (float(a[i, k]), float(loc[i, k]), float(scale[i, k])) for k in range(K)
        ]
        weights = W[i]
        result = {"component_params": component_params, "weights": weights}
        val_ll = None
        if full_job.get("val_observations") is not None:
            val_ll = _val_ll_univariate(full_job, component_params, weights)
        results.append((bs_seed, label, save_dir, full_job.get("fit_idx"), {
            "dataset_name": full_job.get("dataset_name"),
            "bootstrap_seed": bs_seed,
            "num_components": K,
            "fit_idx": full_job.get("fit_idx"),
            "fit": result,
            "val_ll": val_ll if val_ll is not None else -np.inf,
            "calibrated_dims": full_job.get("calibrated_dims"),
        }))
    return results


def _val_ll_univariate(full_job, component_params, weights):
    from ..fit import _weighted_val_ll
    return _weighted_val_ll(
        full_job["val_observations"], full_job["val_sample_assignments"],
        component_params, weights, False, full_job.get("kwargs", {}),
    )


def _failed_result(spec):
    full_job, bs_seed, label, save_dir = spec
    return (bs_seed, label, save_dir, full_job.get("fit_idx"), {
        "dataset_name": full_job.get("dataset_name"),
        "bootstrap_seed": bs_seed,
        "num_components": full_job["num_components"],
        "fit_idx": full_job.get("fit_idx"),
        "fit": {"component_params": [[] for _ in range(full_job["num_components"])], "weights": None},
        "val_ll": -np.inf,
        "calibrated_dims": full_job.get("calibrated_dims"),
    })


# ---------------------------------------------------------------------------
# CFUSN batch: pack / run / unpack
# ---------------------------------------------------------------------------

def _run_cfusn_chunk(specs):
    # See _run_univariate_chunk's note: track validity by index, never `in`.
    inits = []
    valid_specs = []
    results = []
    for spec in specs:
        full_job = spec[0]
        init = _init_cfusn(full_job)
        if init is None:
            results.append(_failed_result(spec))
            continue
        inits.append(init)
        valid_specs.append(spec)

    if not valid_specs:
        return results

    K = valid_specs[0][0]["num_components"]
    S = valid_specs[0][0]["train_sample_assignments"].shape[1]
    p = np.asarray(valid_specs[0][0]["train_observations"]).shape[1]

    obs_raw = np.stack([np.asarray(s[0]["train_observations"], dtype=float) for s in valid_specs])
    obs_mask = ~np.isnan(obs_raw)
    obs = np.where(obs_mask, obs_raw, 0.0)
    sample_idx = np.stack([
        np.argmax(s[0]["train_sample_assignments"], axis=1) for s in valid_specs
    ])
    mu0 = np.stack([[p_[0] for p_ in init[0]] for init in inits])
    Delta0 = np.stack([[p_[1] for p_ in init[0]] for init in inits])
    Gamma0 = np.stack([[p_[2] for p_ in init[0]] for init in inits])
    W0 = np.stack([init[1] for init in inits])

    mu, Delta, Gamma, W, failed = batch_em_cfusn.fit_batch_cfusn(
        jnp.asarray(obs), jnp.asarray(obs_mask), jnp.asarray(sample_idx), S,
        jnp.asarray(mu0), jnp.asarray(Delta0), jnp.asarray(Gamma0), jnp.asarray(W0),
    )
    mu, Delta, Gamma, W, failed = map(np.asarray, (mu, Delta, Gamma, W, failed))

    for i, spec in enumerate(valid_specs):
        full_job, bs_seed, label, save_dir = spec
        if failed[i]:
            results.append(_failed_result(spec))
            continue
        component_params = [(mu[i, k], Delta[i, k], Gamma[i, k]) for k in range(K)]
        weights = W[i]
        result = {"component_params": component_params, "weights": weights}
        val_ll = None
        if full_job.get("val_observations") is not None:
            val_ll = _val_ll_cfusn(full_job, component_params, weights)
        results.append((bs_seed, label, save_dir, full_job.get("fit_idx"), {
            "dataset_name": full_job.get("dataset_name"),
            "bootstrap_seed": bs_seed,
            "num_components": K,
            "fit_idx": full_job.get("fit_idx"),
            "fit": result,
            "val_ll": val_ll if val_ll is not None else -np.inf,
            "calibrated_dims": full_job.get("calibrated_dims"),
        }))
    return results


def _val_ll_cfusn(full_job, component_params, weights):
    from ..fit import _weighted_val_ll
    return _weighted_val_ll(
        full_job["val_observations"], full_job["val_sample_assignments"],
        component_params, weights, True, full_job.get("kwargs", {}),
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_gpu(fit_specs):
    """fit_specs: the same flat list run_array_task.py builds for the CPU path.
    Returns a flat list of (bs_seed, label, save_dir, fit_idx, result) tuples,
    matching `_run_one_fit`'s contract, so the caller's existing
    best-fit-selection/save loop needs no changes.
    """
    out = []
    for is_mv, chunk in group_fit_specs(fit_specs):
        out.extend(_run_cfusn_chunk(chunk) if is_mv else _run_univariate_chunk(chunk))
    return out
