"""Weight-divergence penalty on the mixing weights W (EXPERIMENTAL, opt-in).

Small samples (e.g. TP53's RPV) can only re-weight components that the large
samples shaped: their own row of W is all the model learns about them. This
regulariser rewards labeled samples' mixing distributions for being
DIFFERENT, so that components come to represent sample-specific subsets
rather than one shared majority profile:

    R(W) = sum_{(i, j) in cross-group pairs} JSD(W_i, W_j)
    JSD(p, q) = 1/2 KL(p || m) + 1/2 KL(q || m),   m = (p + q) / 2

Samples are given in GROUPS; only pairs from different groups are pushed
apart (e.g. B/LB and Synonymous share a group -- they describe the same
benign population). Samples in no group (e.g. gnomAD, a population mixture of
pathogenic and benign variants) are untouched: their W rows keep the
ordinary closed-form update exactly.

The fit maximises the penalised objective

    L_pen = [sum_n sw_n log sum_k W[s(n), k] f_k(x_n)] / N + priors / N + lam * R(W)

and stays a monotone (G)EM: R depends on W only, so every theta/categorical
M-step is untouched, and the W step below maximises the Jensen minoriser

    F(W) = sum_{s,k} nbar_sk log W_sk + lam * N * R(W),
    nbar_sk = sum_{n in s} sw_n P(k | x_n; theta_new, W_old)

(LL(theta_new, W) - F_LL(W) is minimised at W = W_old), and only ever
returns a W with F(W) >= F(W_old). With no spec, nothing here runs and every
fit is bit-identical to before.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MAX_INNER_ITERS = 50
INNER_REL_TOL = 1e-12


def resolve_weight_divergence(spec: Optional[Dict], sample_names: Optional[Sequence[str]],
                              n_samples: int) -> Optional[Dict]:
    """User spec {"lambda": float, "groups": [[name or index, ...], ...]} ->
    resolved {"lambda": float, "pairs": [(i, j), ...]} over sample columns, or
    None (plain path) when lambda <= 0 or fewer than two groups are present.
    Names absent from ``sample_names`` (or indices out of range) are dropped.
    """
    if not spec:
        return None
    if "pairs" in spec:                               # already resolved
        pairs = [tuple(int(v) for v in p) for p in spec["pairs"]]
        return {"lambda": float(spec["lambda"]), "pairs": pairs} if pairs and spec["lambda"] > 0 else None
    lam = float(spec.get("lambda", 0.0))
    if lam <= 0:
        return None
    names = list(sample_names) if sample_names is not None else []
    groups: List[List[int]] = []
    for group in spec.get("groups", []):
        idx = []
        for key in group:
            if isinstance(key, (int, np.integer)):
                if 0 <= int(key) < n_samples:
                    idx.append(int(key))
            elif key in names and names.index(key) < n_samples:
                idx.append(names.index(key))
        if idx:
            groups.append(sorted(set(idx)))
    seen = [i for g in groups for i in g]
    if len(seen) != len(set(seen)):
        raise ValueError(f"weight_divergence: a sample appears in more than one group: {spec['groups']}")
    pairs = [(i, j) for a in range(len(groups)) for b in range(a + 1, len(groups))
             for i in groups[a] for j in groups[b]]
    if not pairs:
        return None
    return {"lambda": lam, "pairs": sorted(pairs)}


def _xlogy_ratio(p, m):
    """p * log(p / m) with 0 log 0 = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0) / np.where(m > 0, m, 1.0)), 0.0)


def jsd(p, q):
    m = 0.5 * (p + q)
    return float(0.5 * _xlogy_ratio(p, m).sum() + 0.5 * _xlogy_ratio(q, m).sum())


def divergence(W, pairs: Sequence[Tuple[int, int]]) -> float:
    """R(W): summed cross-group pairwise JSD between rows of W."""
    W = np.asarray(W, dtype=float)
    return float(sum(jsd(W[i], W[j]) for i, j in pairs))


def divergence_grad(W, pairs: Sequence[Tuple[int, int]]):
    """dR/dW. For JSD(p, q): d/dp_k = 1/2 log(p_k / m_k) (0 where p_k = 0)."""
    W = np.asarray(W, dtype=float)
    G = np.zeros_like(W)
    for i, j in pairs:
        m = 0.5 * (W[i] + W[j])
        with np.errstate(divide="ignore", invalid="ignore"):
            G[i] += np.where(W[i] > 0, 0.5 * np.log(np.where(W[i] > 0, W[i], 1.0) / np.where(m > 0, m, 1.0)), 0.0)
            G[j] += np.where(W[j] > 0, 0.5 * np.log(np.where(W[j] > 0, W[j], 1.0) / np.where(m > 0, m, 1.0)), 0.0)
    return G


def minorizer(W, nbar, pairs, lam_total) -> float:
    """F(W) = sum nbar log W + lam_total * R(W); -inf if W puts zero mass where nbar > 0."""
    W = np.asarray(W, dtype=float)
    if np.any((nbar > 0) & (W <= 0)):
        return -np.inf
    with np.errstate(divide="ignore"):
        ll = float(np.sum(np.where(nbar > 0, nbar * np.log(np.where(W > 0, W, 1.0)), 0.0)))
    return ll + lam_total * divergence(W, pairs)


def _ascend(W, nbar, pairs, lam_total, rows, tot):
    """Exponentiated-gradient ascent on F from W (only rows in a pair move)."""
    f = minorizer(W, nbar, pairs, lam_total)
    scale = 1.0 / (tot[:, 0] + lam_total)   # log term's gradient ~n_s, penalty's ~lam_total
    for _ in range(MAX_INNER_ITERS):
        G = divergence_grad(W, pairs)
        with np.errstate(divide="ignore", invalid="ignore"):
            G = lam_total * G + np.where(W > 0, nbar / np.where(W > 0, W, 1.0), 0.0)
        step, f_try, W_try = 1.0, -np.inf, None
        while step > 1e-8:
            W_try = W.copy()
            for r in rows:
                pos = W[r] > 0
                if not pos.any():
                    continue
                g = G[r] * scale[r] * step
                g = np.where(pos, g - g[pos].max(), 0.0)
                w = W[r] * np.exp(g)                 # multiplicative: exact zeros stay zero
                W_try[r] = w / w.sum()
            f_try = minorizer(W_try, nbar, pairs, lam_total)
            if f_try > f:
                break
            step *= 0.5
        if not f_try > f:
            break
        rel = (f_try - f) / max(abs(f), 1.0)
        W, f = W_try, f_try
        if rel < INNER_REL_TOL:
            break
    return W, f


def penalized_weight_update(nbar, W_old, pairs, lam_total):
    """GEM W step: maximise F by exponentiated gradient on each penalised
    row's simplex (backtracking on F), started from both the closed-form
    (unpenalised) optimum and W_old; returns the better, so F(W_new) >=
    F(W_old) always holds.

    nbar      : (S, K) sw-weighted expected counts from the E-step at (theta_new, W_old)
    W_old     : (S, K) current weights (the minoriser's touching point)
    lam_total : lambda * N (the penalty in the same units as sum nbar log W)
    """
    nbar = np.asarray(nbar, dtype=float)
    W_old = np.asarray(W_old, dtype=float)
    rows = sorted({i for p in pairs for i in p})
    tot = nbar.sum(axis=1, keepdims=True)
    W_cf = np.where(tot > 0, nbar / np.where(tot > 0, tot, 1.0), W_old)
    best_W, best_f = _ascend(W_cf, nbar, pairs, lam_total, rows, tot)
    f_old = minorizer(W_old, nbar, pairs, lam_total)
    if best_f < f_old:
        W2, f2 = _ascend(W_old.copy(), nbar, pairs, lam_total, rows, tot)
        best_W, best_f = (W2, f2) if f2 >= f_old else (W_old.copy(), f_old)
    return best_W
