"""Automatic axis + state selection for the factorial mixture (EXPERIMENTAL).

Label-free: every choice is scored by held-out log-likelihood per variant,
averaged over several bootstrap OOB splits; P/LP / B/LB labels are used only
as sample memberships inside the likelihood (as in any fit), never as a
selection target, so a later label-based comparison stays honest.

Steps:
  1. co-measurement rule -> cannot-link pairs (dims rarely measured on the
     same variants can't share an axis: their joint states would be
     unidentified -- e.g. a DNA-binding-domain-only and an oligomerisation-
     domain-only assay)
  2. residual dependence (quick one-axis fit -> MAP state -> within-state
     residuals -> |Spearman| on co-observed rows)
  3. constrained average-linkage agglomeration -> candidate partitions
     (one per merge level, plus all-singletons and, if allowed, one axis)
  4. score candidates (uniform states per axis), then coordinate search over
     each axis's state count (>= 2: a single-state axis would carry no
     class information, silently dropping that assay from the evidence),
     then greedy single-dim moves
  5. among everything scored, the simplest configuration within one
     standard error of the best (fewest cells, then fewest axes)
  6. state kind (gaussian / msn / cfusn) chosen the same way on the result
"""
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import spearmanr

from .model import FactorialMixture

MAX_CELLS = 400


# ── 1. co-measurement rule ──────────────────────────────────────────────────
def comeasurement_ok(X, labeled, min_abs=20, min_frac=0.05, min_labeled=5):
    """(p, p) bool: may dims i and j share an axis?"""
    obs = ~np.isnan(X)
    n = len(X)
    thr = max(min_abs, int(np.ceil(min_frac * n)))
    co = obs.T.astype(int) @ obs.astype(int)
    co_lab = obs[labeled].T.astype(int) @ obs[labeled].astype(int)
    ok = (co >= thr) & (co_lab >= min_labeled)
    np.fill_diagonal(ok, True)
    return ok


def axis_allowed(axis, ok):
    return all(ok[i, j] for i, j in itertools.combinations(axis, 2))


# ── 2. residual dependence ──────────────────────────────────────────────────
def _axis_dicts(partition, n_states, cat_cols, levels, kind):
    lev = dict(zip(cat_cols, levels))
    return [dict(cont=[d for d in a if d not in lev], cats=[d for d in a if d in lev],
                 levels=[lev[d] for d in a if d in lev], n_states=k, kind=kind)
            for a, k in zip(partition, n_states)]


def residual_dependence(X, SA, cat_cols, levels, ok, K0=3, seed=0, min_co=20, **fm_kw):
    p = X.shape[1]
    # one-axis fit needs every pair co-measurable; otherwise fit per allowed block
    m = FactorialMixture(_axis_dicts([list(range(p))], [K0], cat_cols, levels, "gaussian"),
                         seed=seed, **fm_kw).fit(X, SA, max_iter=300)
    ax = m.axes[0]
    w = SA.astype(float) @ m.W
    state = np.argmax(m.cell_logpdfs(X) + np.log(np.maximum(w.T, 1e-300)), axis=0)
    Rz = np.full(X.shape, np.nan)
    for j in range(p):
        for k in range(K0):
            rows = state == k
            if j in ax.cont:
                jj = ax.cont.index(j)
                mu, D, G = ax.params[k]
                D = np.asarray(D).reshape(len(mu), -1)
                var = G[jj, jj] + float(D[jj] @ D[jj])
                Rz[rows, j] = (X[rows, j] - mu[jj]) / np.sqrt(var)
            else:
                jj = ax.cats.index(j)
                pi = ax.pi[jj][k]
                Rz[rows, j] = X[rows, j] - float((np.arange(len(pi)) * pi).sum())
    D = np.zeros((p, p))
    for i, j in itertools.combinations(range(p), 2):
        both = ~np.isnan(Rz[:, i]) & ~np.isnan(Rz[:, j])
        if ok[i, j] and both.sum() >= min_co and Rz[both, i].std() > 0 and Rz[both, j].std() > 0:
            D[i, j] = D[j, i] = abs(spearmanr(Rz[both, i], Rz[both, j]).correlation)
    return D


# ── 3. constrained agglomeration ────────────────────────────────────────────
def candidate_partitions(D, ok):
    p = D.shape[0]
    clusters = [[i] for i in range(p)]
    parts = [sorted(map(sorted, clusters))]
    while len(clusters) > 1:
        best, pair = -1.0, None
        for a, b in itertools.combinations(range(len(clusters)), 2):
            merged = clusters[a] + clusters[b]
            if not axis_allowed(merged, ok):
                continue
            sim = float(np.mean([D[i, j] for i in clusters[a] for j in clusters[b]]))
            if sim > best:
                best, pair = sim, (a, b)
        if pair is None:
            break
        a, b = pair
        clusters = [c for i, c in enumerate(clusters) if i not in pair] + [clusters[a] + clusters[b]]
        parts.append(sorted(map(sorted, clusters)))
    return parts


def canon(part):
    return tuple(sorted(tuple(sorted(a)) for a in part if a))


# ── 4. scoring ──────────────────────────────────────────────────────────────
SEARCH_FIT_KW = dict(max_iter=500, tol=1e-6)   # scoring fits only; final fits use the model defaults


def _score_task(args):
    """One (candidate, split): best held-out LL/row over seeds."""
    part, n_states, kind, split, cat_cols, levels, seeds, fm_kw = args
    Xtr, SAtr, Xva, SAva = split
    best = -np.inf
    for seed in seeds:
        try:
            m = FactorialMixture(_axis_dicts(part, n_states, cat_cols, levels, kind), seed=seed,
                                 **fm_kw).fit(Xtr, SAtr, **SEARCH_FIT_KW)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            continue
        best = max(best, m.loglik(Xva, SAva) / len(Xva))
    return canon(part), tuple(n_states), kind, best


class AxisSearch:
    """splits: list of (Xtr, SAtr, Xva, SAva) in the model's column space.
    X_all / labeled: all variants (any scale) for the co-measurement rule."""

    def __init__(self, splits, cat_cols, levels, X_all, labeled, kind="gaussian", states_range=(2, 3, 4, 5),
                 seeds=(0, 1), workers=8, fm_kw=None, comeasure_kw=None, log=print):
        self.splits, self.cat_cols, self.levels = splits, list(cat_cols), list(levels)
        self.kind, self.states_range, self.seeds = kind, tuple(states_range), tuple(seeds)
        self.workers, self.fm_kw, self.log = workers, dict(fm_kw or {}), log
        self.ok = comeasurement_ok(X_all, labeled, **(comeasure_kw or {}))
        self.scores = {}            # (canon part, states, kind) -> list of per-split held-out LL

    def _score(self, configs):
        todo = [(list(map(list, part)), list(ns), kind) for part, ns, kind in configs
                if (canon(part), tuple(ns), kind) not in self.scores
                and int(np.prod(ns)) <= MAX_CELLS and all(axis_allowed(a, self.ok) for a in part)]
        if not todo:
            return
        args = [(p, ns, k, sp, self.cat_cols, self.levels, self.seeds, self.fm_kw)
                for p, ns, k in todo for sp in self.splits]
        res = {}
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            for part, ns, kind, val in ex.map(_score_task, args):
                res.setdefault((part, ns, kind), []).append(val)
        self.scores.update(res)

    def _mean(self, key):
        v = np.array(self.scores[key])
        return float(np.mean(v)) if np.all(np.isfinite(v)) else -np.inf

    def _best(self, kind=None):
        keys = [k for k in self.scores if kind is None or k[2] == kind]
        return max(keys, key=self._mean)

    def run(self, k_uniform=(2, 3, 4), max_sweeps=4):
        Xtr, SAtr = self.splits[0][0], self.splits[0][1]
        D = residual_dependence(Xtr, SAtr, self.cat_cols, self.levels, self.ok, **self.fm_kw)
        self.D = D
        parts = candidate_partitions(D, self.ok)
        p = D.shape[0]
        if [list(range(p))] not in parts and axis_allowed(list(range(p)), self.ok):
            parts.append([list(range(p))])
        self.log(f"  {len(parts)} candidate partitions (co-measurement cannot-link pairs: "
                 f"{int((~self.ok).sum() // 2)})")
        self._score([(pt, [k] * len(pt), self.kind) for pt in parts for k in k_uniform])
        # coordinate search on states per axis, then greedy moves, repeated
        for sweep in range(max_sweeps):
            before = self._best(self.kind)
            part, ns, kind = before
            part = [list(a) for a in part]
            cands = []
            for a in range(len(part)):
                for k in self.states_range:
                    ns2 = list(ns)
                    ns2[a] = k
                    cands.append((part, ns2, kind))
            self._score(cands)
            part, ns, kind = self._best(self.kind)
            part = [list(a) for a in part]
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
                        moves.append(([new[i] for i in keep], [new_ns[i] for i in keep], kind))
            self._score(moves)
            after = self._best(self.kind)
            self.log(f"  sweep {sweep}: best held-out LL/row {self._mean(before):.4f} -> {self._mean(after):.4f}")
            if after == before:
                break
        # state kind on the selected structure
        part, ns, _ = self.select()
        self._score([(list(map(list, part)), list(ns), k) for k in ("gaussian", "msn", "cfusn")])
        return self.select(all_kinds=True)

    def select(self, all_kinds=False):
        """Simplest config within 1 SE of the best (fewest cells, then axes, then kind order)."""
        keys = [k for k in self.scores if all_kinds or k[2] == self.kind]
        keys = [k for k in keys if np.isfinite(self._mean(k))]
        best = max(keys, key=self._mean)
        v = np.array(self.scores[best])
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        order = {"gaussian": 0, "msn": 1, "cfusn": 2}
        ok = [k for k in keys if self._mean(k) >= self._mean(best) - se]
        return min(ok, key=lambda k: (int(np.prod(k[1])), len(k[0]), order[k[2]]))

    def shortlist(self, n=4):
        """Configurations within 1 SE of the best held-out LL (any kind), best
        first, at most n -- the label-free candidate set a final
        accuracy-based choice is made from."""
        keys = [k for k in self.scores if np.isfinite(self._mean(k))]
        best = max(keys, key=self._mean)
        v = np.array(self.scores[best])
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        ok = sorted([k for k in keys if self._mean(k) >= self._mean(best) - se], key=self._mean, reverse=True)
        return ok[:n]

    def table(self, names=None):
        rows = []
        for (part, ns, kind), vals in self.scores.items():
            rows.append({"partition": [[names[i] if names else i for i in a] for a in part], "states": list(ns),
                         "kind": kind, "cells": int(np.prod(ns)), "mean_val_ll": self._mean((part, ns, kind)),
                         "per_split": [float(x) for x in vals]})
        return sorted(rows, key=lambda r: -r["mean_val_ll"])

    def save(self, path, names=None):
        part, ns, kind = self.select(all_kinds=True)
        json.dump({"selected": {"partition": [[names[i] if names else i for i in a] for a in part],
                                "states": list(ns), "kind": kind},
                   "comeasurement_ok": self.ok.tolist(), "dependence": self.D.tolist(),
                   "dims": names, "scores": self.table(names)}, open(path, "w"), indent=1)
