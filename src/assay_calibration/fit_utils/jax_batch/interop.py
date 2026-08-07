"""Glue between the existing job-pickle format (``fit.py``/``run_array_task.py``)
and the batched JAX fitters in ``batch_em.py`` / ``batch_em_cfusn.py``.

By default (``use_gpu_init=True`` in :func:`run_gpu`) initialization (k-means /
method-of-moments) is performed on the GPU via the batched JAX routines in
``init_jax.py``, eliminating the sequential CPU init loop that was responsible
for 97-99 % of total wall time (see ``tests/benchmark_scaling.py``).

Pass ``use_gpu_init=False`` to fall back to the original sequential CPU init
(``cfusn/initializations.py``) — used by ``tests/test_batch_em_parity.py`` to
keep the tight float64-parity assertions valid.

Only supports the common case each `prepare.py` invocation actually produces:
one `constrained`/`force_gaussian`/`multivariate` setting per (dataset,
num_components) group. `group_fit_specs` asserts this and raises if a group
turns out to be mixed (this would indicate the manifest was hand-edited or
a job dict is malformed) rather than silently mishandling it.
"""
from collections import defaultdict

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ..cfusn.initializations import (
    kmeans_init, methodOfMomentsInit, kmeans_init_mv, kmeans_init_mv_anchored,
)
from ..cfusn.update_steps import get_sample_weights
from ..cfusn.fit import _ensure_cfusn_params
from . import batch_em, batch_em_cfusn
from .init_jax import batch_init_univariate, batch_init_cfusn
from .batch_em import _log_pdfs, _gather_sample_weights
from jax.scipy.special import logsumexp as jax_logsumexp

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
# CPU-side initialization (legacy — only used when use_gpu_init=False)
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

def _run_univariate_chunk(specs, use_gpu_init=True, max_em_iters=None):
    # NB: don't use `spec in valid_specs` anywhere here — specs are tuples
    # containing job dicts with ndarray values, and `==`/`in` on those raises
    # ValueError ("truth value of an array is ambiguous"). Track validity by
    # index instead.
    if not specs:
        return []

    K = specs[0][0]["num_components"]
    S = specs[0][0]["train_sample_assignments"].shape[1]
    constrained = bool(specs[0][0]["constrained"])
    force_gaussian = bool(specs[0][0].get("kwargs", {}).get("force_gaussian", False))

    obs = jnp.asarray(np.stack([
        np.asarray(s[0]["train_observations"]).ravel() for s in specs
    ]))  # (batch, N)
    sample_idx = jnp.asarray(np.stack([
        np.argmax(s[0]["train_sample_assignments"], axis=1) for s in specs
    ]))  # (batch, N)
    xmin = obs.min(axis=1)
    xmax = obs.max(axis=1)

    if use_gpu_init:
        fit_seeds = [s[0].get("kwargs", {}).get("fit_seed") or 0 for s in specs]
        key = jax.random.PRNGKey(int(np.array(fit_seeds, dtype=np.uint32).sum()))
        fit_idx = jnp.asarray([s[0].get("fit_idx") or 0 for s in specs], dtype=jnp.int32)
        a0, loc0, scale0, W0, init_failed = batch_init_univariate(
            obs, sample_idx, S, K, constrained, xmin, xmax, key, fit_idx)
        results = []
        valid_mask = [True] * len(specs)  # all specs go to EM
    else:
        # Legacy sequential CPU init (used by parity tests)
        inits = []
        valid_specs = []
        results = []
        valid_mask = []
        for spec in specs:
            init = _init_univariate(spec[0])
            if init is None:
                results.append(_failed_result(spec))
                valid_mask.append(False)
                continue
            inits.append(init)
            valid_specs.append(spec)
            valid_mask.append(True)

        if not valid_specs:
            return results

        a0 = jnp.asarray(np.stack([[p[0] for p in init[0]] for init in inits]))
        loc0 = jnp.asarray(np.stack([[p[1] for p in init[0]] for init in inits]))
        scale0 = jnp.asarray(np.stack([[p[2] for p in init[0]] for init in inits]))
        W0 = jnp.asarray(np.stack([init[1] for init in inits]))
        obs = jnp.asarray(np.stack([
            np.asarray(s[0]["train_observations"]).ravel() for s in valid_specs
        ]))
        sample_idx = jnp.asarray(np.stack([
            np.argmax(s[0]["train_sample_assignments"], axis=1) for s in valid_specs
        ]))
        xmin = obs.min(axis=1)
        xmax = obs.max(axis=1)
        specs = valid_specs  # only run EM on valid inits

    fit_kw = dict(constrained=constrained, force_gaussian=force_gaussian)
    if max_em_iters is not None:
        fit_kw["max_em_iters"] = max_em_iters
    a, loc, scale, W, failed, it_final, done = batch_em.fit_batch(
        obs, sample_idx, S, a0, loc0, scale0, W0,
        xmin, xmax, **fit_kw,
    )
    a, loc, scale, W, failed, done = map(np.asarray, (a, loc, scale, W, failed, done))
    it_final_np = int(np.asarray(it_final))
    hit_cap = ~done
    n_hit_cap = int(hit_cap.sum())
    print(f"  [batch_em] iterations: {it_final_np} (batch={len(specs)}, cap_hits={n_hit_cap})", flush=True)
    if use_gpu_init:
        failed = failed | np.asarray(init_failed)

    # Batched GPU val_ll — one call for the whole chunk instead of one
    # scipy.stats.skewnorm.logpdf loop per fit (~18s/task on CPU).
    # All specs in a chunk share the same dataset so N_val is uniform.
    has_val = specs[0][0].get("val_observations") is not None
    batch_val_lls = None
    if has_val:
        val_obs_list, val_idx_list, n_orig_list = [], [], []
        for spec in specs:
            fj = spec[0]
            vo = np.asarray(fj["val_observations"]).ravel()
            vi = np.argmax(fj["val_sample_assignments"], axis=1)
            val_obs_list.append(vo)
            val_idx_list.append(vi)
            n_orig_list.append(len(vo))
        # Pad to max N_val with NaN so rows can be stacked; NaN obs are masked
        # in _val_ll_batch_uv. Denominator uses n_orig (unpadded length) to
        # match CPU's division by len(val_sample_assignments).
        max_n = max(n_orig_list)
        val_obs_np = np.full((len(specs), max_n), np.nan)
        val_idx_np = np.zeros((len(specs), max_n), dtype=int)
        for i, (vo, vi) in enumerate(zip(val_obs_list, val_idx_list)):
            val_obs_np[i, :len(vo)] = vo
            val_idx_np[i, :len(vi)] = vi
        n_orig_np = np.array(n_orig_list, dtype=float)
        batch_val_lls = _val_ll_batch_uv(val_obs_np, val_idx_np, a, loc, scale, W, n_orig_np)

    for i, spec in enumerate(specs):
        full_job, bs_seed, label, save_dir = spec
        if failed[i]:
            results.append(_failed_result(spec))
            continue
        component_params = [
            (float(a[i, k]), float(loc[i, k]), float(scale[i, k])) for k in range(K)
        ]
        weights = W[i]
        result = {
            "component_params": component_params,
            "weights": weights,
            "xlims": (float(xmin[i]), float(xmax[i])),
            "times_submerged": [],
        }
        val_ll = float(batch_val_lls[i]) if batch_val_lls is not None else None
        results.append((bs_seed, label, save_dir, full_job.get("fit_idx"), {
            "dataset_name": full_job.get("dataset_name"),
            "bootstrap_seed": bs_seed,
            "num_components": K,
            "fit_idx": full_job.get("fit_idx"),
            "fit": result,
            "val_ll": val_ll if val_ll is not None else -np.inf,
            "calibrated_dims": full_job.get("calibrated_dims"),
            "hit_cap": bool(hit_cap[i]),
        }))
    return results


def _val_ll_batch_uv(val_obs_np, val_sample_idx_np, a_np, loc_np, scale_np, W_np, n_orig_np):
    """Batched GPU val log-likelihood for univariate SN mixture.

    val_obs_np: (batch, N_val) — NaN-padded to a common max length;
    n_orig_np: (batch,) — original unpadded N_val per fit (used as denominator
               to match CPU's division by len(val_sample_assignments)).
    """
    val_obs = jnp.asarray(val_obs_np)
    sample_idx = jnp.asarray(val_sample_idx_np)
    a = jnp.asarray(a_np); loc = jnp.asarray(loc_np); scale = jnp.asarray(scale_np)
    W = jnp.asarray(W_np)

    valid = jnp.isfinite(val_obs)                                  # (batch, N_val)
    log_pdfs = _log_pdfs(val_obs, a, loc, scale)                   # (batch, K, N_val)
    log_pdfs_bnk = jnp.moveaxis(log_pdfs, 1, 2)                    # (batch, N_val, K)
    w_n = _gather_sample_weights(W, sample_idx)                    # (batch, N_val, K)
    log_w = jnp.where(w_n > 0, jnp.log(jnp.where(w_n > 0, w_n, 1.0)), -jnp.inf)
    log_mix = jax_logsumexp(log_pdfs_bnk + log_w, axis=-1)         # (batch, N_val)
    n_orig = jnp.asarray(n_orig_np, dtype=jnp.float64)
    return np.asarray((log_mix * valid).sum(-1) / n_orig)


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
        "fit": {"component_params": [[] for _ in range(full_job["num_components"])], "weights": None, "xlims": None, "times_submerged": []},
        "val_ll": -np.inf,
        "calibrated_dims": full_job.get("calibrated_dims"),
    })


# ---------------------------------------------------------------------------
# CFUSN batch: pack / run / unpack
# ---------------------------------------------------------------------------

def _run_cfusn_chunk(specs, use_gpu_init=True, max_em_iters=None):
    # See _run_univariate_chunk's note: track validity by index, never `in`.
    if not specs:
        return []

    K = specs[0][0]["num_components"]
    S = specs[0][0]["train_sample_assignments"].shape[1]

    obs_raw = np.stack([np.asarray(s[0]["train_observations"], dtype=float) for s in specs])
    obs_mask_np = ~np.isnan(obs_raw)
    obs_np = np.where(obs_mask_np, obs_raw, 0.0)

    obs = jnp.asarray(obs_np)
    obs_mask = jnp.asarray(obs_mask_np)
    sample_idx = jnp.asarray(np.stack([
        np.argmax(s[0]["train_sample_assignments"], axis=1) for s in specs
    ]))

    # Fall back to CPU init for anchored jobs (kmeans_init_mv_anchored too
    # complex to batch on GPU; anchored is a minority in practice)
    any_anchored = any(s[0].get("init_method") == "anchored" for s in specs)
    if use_gpu_init and not any_anchored:
        latent_q = int(specs[0][0].get("kwargs", {}).get("latent_q", 2))
        fit_seeds = [s[0].get("kwargs", {}).get("fit_seed") or 0 for s in specs]
        key = jax.random.PRNGKey(int(np.array(fit_seeds, dtype=np.uint32).sum()))
        mu0, Delta0, Gamma0, W0, init_failed = batch_init_cfusn(
            obs, obs_mask, sample_idx, S, K, latent_q, key)
        results = []
    else:
        # Legacy sequential CPU init
        inits = []
        valid_specs = []
        results = []
        for spec in specs:
            init = _init_cfusn(spec[0])
            if init is None:
                results.append(_failed_result(spec))
                continue
            inits.append(init)
            valid_specs.append(spec)

        if not valid_specs:
            return results

        p = np.asarray(valid_specs[0][0]["train_observations"]).shape[1]
        mu0 = jnp.asarray(np.stack([[p_[0] for p_ in init[0]] for init in inits]))
        Delta0 = jnp.asarray(np.stack([[p_[1] for p_ in init[0]] for init in inits]))
        Gamma0 = jnp.asarray(np.stack([[p_[2] for p_ in init[0]] for init in inits]))
        W0 = jnp.asarray(np.stack([init[1] for init in inits]))
        # Rebuild obs/obs_mask/sample_idx for valid specs only
        obs_raw_v = np.stack([np.asarray(s[0]["train_observations"], dtype=float) for s in valid_specs])
        obs_mask_v = ~np.isnan(obs_raw_v)
        obs = jnp.asarray(np.where(obs_mask_v, obs_raw_v, 0.0))
        obs_mask = jnp.asarray(obs_mask_v)
        sample_idx = jnp.asarray(np.stack([
            np.argmax(s[0]["train_sample_assignments"], axis=1) for s in valid_specs
        ]))
        specs = valid_specs

    cfusn_kw = {}
    if max_em_iters is not None:
        cfusn_kw["max_em_iters"] = max_em_iters
    mu, Delta, Gamma, W, failed, it_final, done = batch_em_cfusn.fit_batch_cfusn(
        obs, obs_mask, sample_idx, S,
        mu0, Delta0, Gamma0, W0,
        **cfusn_kw,
    )
    mu, Delta, Gamma, W, failed, done = map(np.asarray, (mu, Delta, Gamma, W, failed, done))
    it_final_np = int(np.asarray(it_final))
    hit_cap = ~done
    n_hit_cap = int(hit_cap.sum())
    print(f"  [batch_em_cfusn] iterations: {it_final_np} (batch={len(specs)}, cap_hits={n_hit_cap})", flush=True)
    if use_gpu_init and not any_anchored:
        failed = failed | np.asarray(init_failed)

    for i, spec in enumerate(specs):
        full_job, bs_seed, label, save_dir = spec
        if failed[i]:
            results.append(_failed_result(spec))
            continue
        component_params = [(mu[i, k], Delta[i, k], Gamma[i, k]) for k in range(K)]
        weights = W[i]
        train_obs = np.asarray(full_job["train_observations"])
        if train_obs.ndim == 2:
            xlims = tuple(
                (float(np.nanmin(train_obs[:, d])), float(np.nanmax(train_obs[:, d])))
                for d in range(train_obs.shape[1])
            )
        else:
            xlims = (float(np.nanmin(train_obs)), float(np.nanmax(train_obs)))
        result = {
            "component_params": component_params,
            "weights": weights,
            "xlims": xlims,
            "times_submerged": [],
        }
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
            "hit_cap": bool(hit_cap[i]),
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

def run_gpu(fit_specs, use_gpu_init=True, max_em_iters=None):
    """fit_specs: the same flat list run_array_task.py builds for the CPU path.
    Returns a flat list of (bs_seed, label, save_dir, fit_idx, result) tuples,
    matching `_run_one_fit`'s contract, so the caller's existing
    best-fit-selection/save loop needs no changes.

    use_gpu_init: if True (default), init is batched on GPU via init_jax.py —
        eliminates the CPU init bottleneck.  Set False for parity tests that
        require the same init as the NumPy path.
    max_em_iters: if set, overrides the default cap (500) passed to fit_batch.
        Pass a higher value in parity tests to ensure convergence is not
        truncated — parity tests should use max_em_iters=2000.
    """
    out = []
    chunks = group_fit_specs(fit_specs)
    total_fits = len(fit_specs)
    n_chunks = len(chunks)
    for i, (is_mv, chunk) in enumerate(chunks, start=1):
        fn = _run_cfusn_chunk if is_mv else _run_univariate_chunk
        out.extend(fn(chunk, use_gpu_init=use_gpu_init, max_em_iters=max_em_iters))
        # No joblib here to report progress for us (unlike the CPU path in
        # fit_bootstrap.py), so print our own cumulative counter -- one line
        # per chunk, not per fit, since a chunk is already a single batched
        # GPU call (up to MAX_BATCH_UNIVARIATE/MAX_BATCH_CFUSN fits at once).
        print(f"  [GPU] chunk {i}/{n_chunks} done -- {len(out)}/{total_fits} fits complete", flush=True)
    return out
