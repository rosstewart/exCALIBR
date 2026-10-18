"""Export a fitted FactorialMixture to the production fit-dict format, and the
analysis-side adapter that evaluates it (EXPERIMENTAL, opt-in).

A cell is a product of independent per-axis skew-normal blocks, which is
itself an exact CFUSN over all continuous dims: block-diagonal Gamma, and a
block-diagonal Delta whose latent dimension is the sum of the axes' q
(Gaussian axes contribute none). So every cell is stored as an ordinary
(mu, Delta, Gamma) component -- any code that indexes component tuples gets
the exact joint density -- and the fit dict additionally carries a
``factorial`` block. When that block is present, categorical.wrap_params
returns a FactorialParams, and component_logpdf_full evaluates the cell as
the (much cheaper, identical) sum of per-axis block log-densities.

All parameters are stored in RAW units (generate_fit_jobs standardisation
undone, as execute_fit_job does for canonical fits).
"""
import numpy as np

from ..cfusn import categorical as _cat
from ..cfusn import density_utils as _du


def _raw_state(params, mean, std, kind):
    mu, D, G = (np.asarray(a, dtype=float) for a in params)
    mu_r = mu * std + mean
    D_r = D * std if D.ndim == 1 else D * std[:, None]
    G_r = G * np.outer(std, std)
    return mu_r, D_r, G_r


def to_fit_dict(model, scale_mean, scale_std, cat_spec=None):
    """Production-format fit dict (raw units) for a fitted FactorialMixture.

    ``cat_spec``: the job's categorical spec dict (positions / cont_positions
    in the model's column space), or None when the model has no categorical
    dims. Axis column indices are converted to continuous-column indices.
    """
    mean, std = np.asarray(scale_mean, dtype=float), np.asarray(scale_std, dtype=float)
    p_all = len(mean)
    cats = list(cat_spec["positions"]) if cat_spec else []
    cont = list(cat_spec["cont_positions"]) if cat_spec else list(range(p_all))
    axes_out = []
    for ax in model.axes:
        cidx = [cont.index(j) for j in ax.cont]
        states = [tuple(np.asarray(a).tolist() for a in _raw_state(prm, mean[ax.cont], std[ax.cont], ax.kind))
                  for prm in ax.params] if ax.cont else []
        axes_out.append({"cont": cidx, "cat": [cats.index(j) for j in ax.cats], "kind": ax.kind,
                         "n_states": ax.K, "states": states,
                         "pi": [[np.asarray(p).tolist() for p in pj] for pj in ax.pi]})
    comps = []
    q_cols = [ax.q if (ax.cont and ax.kind != "gaussian") else 0 for ax in model.axes]
    Q = sum(q_cols)
    for c in model.cells:
        mu = np.zeros(len(cont))
        G = np.zeros((len(cont), len(cont)))
        D = np.zeros((len(cont), max(Q, 1)))
        col = 0
        for a, k in enumerate(c):
            ax = axes_out[a]
            if not ax["cont"]:
                continue
            m_k, D_k, G_k = (np.asarray(v, dtype=float) for v in ax["states"][k])
            idx = ax["cont"]
            mu[idx] = m_k
            G[np.ix_(idx, idx)] = G_k
            if q_cols[a]:
                D[np.ix_(idx, list(range(col, col + q_cols[a])))] = D_k.reshape(len(idx), q_cols[a])
                col += q_cols[a]
        Dout = D[:, 0] if D.shape[1] == 1 else D
        comps.append([mu.tolist(), Dout.tolist(), G.tolist()])
    fit = {"component_params": comps, "weights": np.asarray(model.W).tolist(),
           "latent_q": int(max(Q, 1)), "xlims": None,
           "factorial": {"axes": axes_out, "cells": [list(c) for c in model.cells],
                         "tau": model.tau, "tau_by_sample": dict(model.tau_by_sample),
                         "alpha_w": model.alpha_w}}
    if cat_spec:
        cat_probs = []
        for j in range(len(cats)):
            a = next(i for i, ax in enumerate(axes_out) if j in ax["cat"])
            jj = axes_out[a]["cat"].index(j)
            cat_probs.append([axes_out[a]["pi"][jj][c[a]] for c in model.cells])
        fit["cat_spec"] = cat_spec
        fit["cat_probs"] = cat_probs
    return fit


class FactorialParams(_cat.MixtureParams):
    """Cell list (exact joint (mu, Delta, Gamma) per cell) + the axis blocks
    for fast evaluation. cat_spec/cat_probs as in MixtureParams (may be None)."""

    def __init__(self, components=(), cat_spec=None, cat_probs=None, factorial=None):
        super().__init__(components, cat_spec=cat_spec, cat_probs=cat_probs)
        self.factorial = factorial
        self._states = None
        if factorial is not None:
            self._states = [[tuple(np.asarray(v, dtype=float) for v in st) for st in ax["states"]]
                            for ax in factorial["axes"]]

    def __reduce__(self):
        return (FactorialParams, (list(self), self.cat_spec, self.cat_probs, self.factorial))

    def cont_logpdf(self, X_cont, c):
        """Continuous log-density of cell c on continuous-column rows X_cont."""
        X_cont = np.atleast_2d(np.asarray(X_cont, dtype=float))
        cell = self.factorial["cells"][c]
        out = np.zeros(len(X_cont))
        for a, ax in enumerate(self.factorial["axes"]):
            if not ax["cont"]:
                continue
            Xa = X_cont[:, ax["cont"]]
            rows = ~np.all(np.isnan(Xa), axis=1)
            if rows.any():
                out[rows] += np.atleast_1d(_du._single_component_logpdf(Xa[rows], self._states[a][cell[a]], True)).ravel()
        return out


def wrap_factorial(params, fit_dict):
    spec = fit_dict.get("cat_spec")
    if spec and not isinstance(spec, _cat.CatSpec):
        spec = _cat.CatSpec.from_dict(spec)
    probs = [np.asarray(P, dtype=float) for P in fit_dict["cat_probs"]] if fit_dict.get("cat_probs") else None
    return FactorialParams(params, cat_spec=spec or None, cat_probs=probs, factorial=fit_dict["factorial"])
