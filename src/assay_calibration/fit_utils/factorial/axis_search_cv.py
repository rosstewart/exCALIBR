"""Factorial model selection by K-fold cross-validated held-out likelihood
(EXPERIMENTAL; v3). Label-free: labels enter only as sample memberships
inside the likelihood, never as a selection target.

Every hyperparameter is chosen by the same score -- the mean held-out
log-likelihood per variant over K folds of ALL the data (each variant held
out exactly once; folds stratified by sample and missingness pattern):

  * axes            (co-measurement-constrained candidates from residual dependence)
  * states per axis (2..5; a single-state axis would carry no class information)
  * state kind      (gaussian / msn (q=1) / cfusn (q=2) -- searched directly,
                     not substituted after a Gaussian search)
  * Gamma prior     (prior_n0_frac: pseudo-variants per state, as a fraction of
                     the axis's measured variants)

Final choice: the simplest configuration within one standard error (over
folds) of the best: fewest cells, then fewest axes, then kind order
gaussian < msn < cfusn, then the stronger prior.
"""
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .model import FactorialMixture
from .axis_search import (comeasurement_ok, axis_allowed, residual_dependence, candidate_partitions, canon,
                          _axis_dicts, MAX_CELLS)

KINDS = ("gaussian", "msn", "cfusn")
PRIORS = (0.05, 0.02, 0.005)
CV_FIT_KW = dict(max_iter=500, tol=1e-6)


def stratified_folds(X, SA, k=5, seed=0):
    """Fold id per row, round-robin within (sample, missingness pattern) groups."""
    rng = np.random.default_rng(seed)
    obs = ~np.isnan(X)
    key = [(int(np.argmax(SA[i])) if SA[i].any() else -1, obs[i].tobytes()) for i in range(len(X))]
    groups = {}
    for i, g in enumerate(key):
        groups.setdefault(g, []).append(i)
    fold = np.zeros(len(X), int)
    offset = 0
    for g in sorted(groups, key=lambda t: (t[0], t[1])):
        idx = np.array(groups[g])
        rng.shuffle(idx)
        fold[idx] = (np.arange(len(idx)) + offset) % k
        offset += len(idx)
    return fold


def _cv_task(args):
    part, ns, kind, prior, f, X, SA, fold, cat_cols, levels, seeds, fm_kw = args
    tr, va = fold != f, fold == f
    best = -np.inf
    for seed in seeds:
        try:
            m = FactorialMixture(_axis_dicts(part, ns, cat_cols, levels, kind), seed=seed, prior_n0_frac=prior,
                                 **fm_kw).fit(X[tr], SA[tr], **CV_FIT_KW)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            continue
        best = max(best, m.loglik(X[va], SA[va]) / max(int(va.sum()), 1))
    return (canon(part), tuple(ns), kind, prior), f, best


class CVSearch:
    def __init__(self, X, SA, cat_cols, levels, labeled, k_folds=5, states_range=(2, 3, 4, 5), seeds=(0, 1),
                 workers=8, fm_kw=None, comeasure_kw=None, log=print, fold_seed=0):
        self.X, self.SA = X, np.asarray(SA, bool)
        self.cat_cols, self.levels = list(cat_cols), list(levels)
        self.k, self.states_range, self.seeds = k_folds, tuple(states_range), tuple(seeds)
        self.workers, self.fm_kw, self.log = workers, dict(fm_kw or {}), log
        self.ok = comeasurement_ok(X, labeled, **(comeasure_kw or {}))
        self.fold = stratified_folds(X, self.SA, k_folds, seed=fold_seed)
        self.scores = {}            # (canon part, states, kind, prior) -> per-fold held-out LL/row

    def _score(self, configs):
        todo = []
        for part, ns, kind, prior in configs:
            key = (canon(part), tuple(ns), kind, prior)
            if key in self.scores or int(np.prod(ns)) > MAX_CELLS or not all(axis_allowed(a, self.ok) for a in part):
                continue
            if key not in [t[0] for t in todo]:
                todo.append((key, [list(a) for a in part], list(ns), kind, prior))
        if not todo:
            return
        args = [(p, ns, kind, prior, f, self.X, self.SA, self.fold, self.cat_cols, self.levels, self.seeds,
                 self.fm_kw) for _, p, ns, kind, prior in todo for f in range(self.k)]
        res = {}
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            for key, f, val in ex.map(_cv_task, args):
                res.setdefault(key, [np.nan] * self.k)[f] = val
        self.scores.update(res)

    def mean(self, key):
        v = np.array(self.scores[key])
        return float(np.mean(v)) if np.all(np.isfinite(v)) else -np.inf

    def best(self, filt=None):
        keys = [k for k in self.scores if (filt is None or filt(k)) and np.isfinite(self.mean(k))]
        return max(keys, key=self.mean)

    def run(self, k_uniform=(2, 3, 4), max_sweeps=4):
        D = residual_dependence(self.X, self.SA, self.cat_cols, self.levels, self.ok, **self.fm_kw)
        self.D = D
        p = D.shape[0]
        parts = candidate_partitions(D, self.ok)
        if [list(range(p))] not in parts and axis_allowed(list(range(p)), self.ok):
            parts.append([list(range(p))])
        self.log(f"  {len(parts)} candidate partitions; cannot-link pairs {int((~self.ok).sum() // 2)}")
        # A: every candidate partition x uniform states x every kind (default prior)
        self._score([(pt, [k] * len(pt), kind, PRIORS[0]) for pt in parts for k in k_uniform for kind in KINDS])
        self.log(f"  A: {len(self.scores)} configs scored; best {self.mean(self.best()):.4f}")
        for sweep in range(max_sweeps):
            before = self.best()
            part, ns, kind, prior = before
            part = [list(a) for a in part]
            # B: states per axis
            cands = []
            for a in range(len(part)):
                for k in self.states_range:
                    ns2 = list(ns)
                    ns2[a] = k
                    cands.append((part, ns2, kind, prior))
            self._score(cands)
            part, ns, kind, prior = self.best()
            part = [list(a) for a in part]
            # C: greedy single-dim moves
            moves = []
            for a_i, axis in enumerate(part):
                for d in axis:
                    for b_i in range(len(part) + 1):
                        if b_i == a_i or (b_i == len(part) and len(axis) == 1):
                            continue
                        new = [list(x) for x in part] + [[]]
                        new[a_i].remove(d)
                        new[b_i].append(d)
                        new_ns = list(ns) + [2]
                        keep = [i for i, ax in enumerate(new) if ax]
                        moves.append(([new[i] for i in keep], [new_ns[i] for i in keep], kind, prior))
            self._score(moves)
            # D: kind and prior on the current best structure
            part, ns, kind, prior = self.best()
            self._score([([list(a) for a in part], list(ns), k2, p2) for k2 in KINDS for p2 in PRIORS])
            after = self.best()
            self.log(f"  sweep {sweep}: {self.mean(before):.4f} -> {self.mean(after):.4f}  ({len(self.scores)} configs)")
            if after == before:
                break
        return self.select()

    def select(self):
        best = self.best()
        v = np.array(self.scores[best])
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        order = {"gaussian": 0, "msn": 1, "cfusn": 2}
        ok = [k for k in self.scores if np.isfinite(self.mean(k)) and self.mean(k) >= self.mean(best) - se]
        return min(ok, key=lambda k: (int(np.prod(k[1])), len(k[0]), order[k[2]], -k[3]))

    def table(self, names=None):
        rows = []
        for key, vals in self.scores.items():
            part, ns, kind, prior = key
            rows.append({"partition": [[names[i] if names else i for i in a] for a in part], "states": list(ns),
                         "kind": kind, "prior_n0_frac": prior, "cells": int(np.prod(ns)),
                         "mean_cv_ll": self.mean(key), "per_fold": [float(x) for x in vals]})
        return sorted(rows, key=lambda r: -r["mean_cv_ll"])
