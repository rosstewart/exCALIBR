"""Tied population weights (EXPERIMENTAL, opt-in): population containment.

The calibration model reads the population sample (gnomAD) as a mixture of the
pathogenic and benign distributions,

    f_pop = pi * f_P + (1 - pi) * f_B,

and the prior pi is estimated from that identity after the fit. Fitting every
sample's mixing weights freely lets the P/LP component(s) hold almost no
population mass (a component fit to a handful of P/LP variants alone), which
the identity rules out: pi * f_P(x) <= f_pop(x) everywhere. This module builds
the identity into the fit by tying the population row of W,

    W[pop] = pi * W[P] + (1 - pi) * W[B],    W[b] = W[B] for every benign sample b,

so a pathogenic component must also explain the population variants in its
region. Samples outside the spec (phenotype groups, TP53 RPV) keep free rows.

The W step maximises the same Jensen minoriser as the free update,

    F(W) = sum_{s,k} nbar_sk log W_sk,
    nbar_sk = sum_{n in s} sw_n P(k | x_n; theta_new, W_old),

over tied W. The population term log(pi W_Pk + (1 - pi) W_Bk) is minorised
once more by the pathogenic/benign split of each component's population mass
at the current iterate, a_k = pi W_Pk / W_pop,k, which gives the closed form

    W_P  ∝ nbar_P + a * nbar_pop
    W_B  ∝ sum_b nbar_b + (1 - a) * nbar_pop
    pi   = sum_k a_k nbar_pop,k / sum_k nbar_pop,k

and never decreases F, so EM stays monotone. With no spec nothing here runs.
"""
from typing import Dict, Optional, Sequence

import numpy as np

PI_FLOOR = 1e-6


def resolve_tied_population(spec: Optional[Dict], n_samples: int) -> Optional[Dict]:
    """{"primary": i, "benign": [j, ...], "population": k} in the fit's sample
    (column) indices, validated; None when there is nothing to tie."""
    if not spec:
        return None
    p, pop = int(spec["primary"]), int(spec["population"])
    ben = [int(b) for b in spec["benign"]]
    idx = [p, pop] + ben
    if not ben or len(set(idx)) != len(idx) or min(idx) < 0 or max(idx) >= n_samples:
        raise ValueError(f"invalid tied_population spec {spec} for {n_samples} samples")
    return {"primary": p, "benign": ben, "population": pop}


def _benign_row(W, spec, counts=None):
    rows = np.asarray([W[b] for b in spec["benign"]], float)
    c = np.ones(len(rows)) if counts is None else np.asarray([counts[b] for b in spec["benign"]], float)
    wb = (c[:, None] * rows).sum(0) / c.sum()
    return wb / wb.sum()


def estimate_pi(W, spec) -> float:
    """pi with W[pop] closest to pi W[P] + (1 - pi) W[B] (exact for tied W)."""
    wp, wb, wpop = W[spec["primary"]], _benign_row(W, spec), W[spec["population"]]
    d = wp - wb
    denom = float(d @ d)
    if denom <= 0:
        return 0.5
    return float(np.clip((wpop - wb) @ d / denom, PI_FLOOR, 1 - PI_FLOOR))


def tie(W, spec, pi=None, counts=None):
    """Project free W onto the tied form (initialisation)."""
    W = np.array(W, float)
    wb = _benign_row(W, spec, counts)
    pi = estimate_pi(W, spec) if pi is None else float(pi)
    for b in spec["benign"]:
        W[b] = wb
    W[spec["population"]] = pi * W[spec["primary"]] + (1 - pi) * wb
    return W


def tied_weight_update(nbar, W_old, spec):
    """Closed-form tied W step (module docstring); free rows get the ordinary
    nbar_s / sum nbar_s update."""
    nbar = np.asarray(nbar, float)
    W = nbar / np.maximum(nbar.sum(1, keepdims=True), 1e-300)
    p, pop, ben = spec["primary"], spec["population"], spec["benign"]
    pi = estimate_pi(W_old, spec)
    wp_old, wb_old = W_old[p], _benign_row(W_old, spec)
    with np.errstate(invalid="ignore", divide="ignore"):
        mix = pi * wp_old + (1 - pi) * wb_old
        a = np.where(mix > 0, pi * wp_old / mix, 0.0)
    path_mass = nbar[pop] * a
    wp = nbar[p] + path_mass
    wb = nbar[ben].sum(0) + nbar[pop] * (1 - a)
    wp, wb = wp / wp.sum(), wb / wb.sum()
    pi_new = float(np.clip(path_mass.sum() / max(nbar[pop].sum(), 1e-300), PI_FLOOR, 1 - PI_FLOOR))
    W[p] = wp
    for b in ben:
        W[b] = wb
    W[pop] = pi_new * wp + (1 - pi_new) * wb
    return W


def spec_from_roles(roles: Sequence[str]) -> Optional[Dict]:
    """Spec from per-column roles (hpc/prepare.py sample_roles): primary P/LP,
    benign = B/LB + synonymous, population = gnomAD. None if any is missing."""
    prim = [i for i, r in enumerate(roles) if r == "primary"]
    pop = [i for i, r in enumerate(roles) if r == "population"]
    ben = [i for i, r in enumerate(roles) if r in ("benign", "synonymous")]
    if len(prim) != 1 or len(pop) != 1 or not ben:
        return None
    return {"primary": prim[0], "benign": ben, "population": pop[0]}
