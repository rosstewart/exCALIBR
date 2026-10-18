"""Categorical (nominal) dimensions for the multivariate CFUSN mixture.

Location model (Olkin & Tate 1961): each mixture component factors as

    p(row | k) = p_CFUSN(x_cont | k) * prod_c Cat(x_c | pi_{k,c})

i.e. categorical and continuous dims are conditionally independent WITHIN a
component; dependence between them is carried entirely by which component a
row belongs to. The continuous block keeps its ordinary (mu, Delta, Gamma)
parameters over the continuous dims only; the categorical block is a
probability vector per (component, categorical dim), stored separately as
``cat_probs`` (a list over categorical dims of (K, L_c) arrays).

EXPERIMENTAL / opt-in: only active when a caller passes a categorical spec
(``categorical_dims``) -- with none, every fit is bit-identical to before.

Nominal, not ordinal: levels are treated as unordered. Some categorical
dims are really ordinal (e.g. TP53's KawOligo: Tetramer < Dimer < Monomer
in pathogenicity), and an ordinal model (shared cut-points plus a per-
component location) would use that ordering with fewer parameters; that is
a deliberate follow-up, not what this module fits.

Observation encoding: categorical columns stay in the full observation
matrix as float codes 0..L_c-1, NaN for missing. A missing categorical
value contributes nothing to the row's log-density (exact marginalisation
under MAR), exactly like a missing continuous dim.

M-step (closed form, MAP): with responsibilities z (times any per-row
sample_weights),

    pi_{k,c,l} = (alpha_{c,l} + sum_n z_nk 1[x_nc = l])
                 / (sum_l alpha_{c,l} + sum_n z_nk 1[x_nc observed])

which is the exact maximiser of the categorical part of Q plus the log-
prior sum_{k,c,l} alpha_{c,l} log pi_{k,c,l} (Dirichlet(alpha + 1)). Since
Q separates into a weights term, a continuous term and a categorical term,
this update composes with the existing continuous M-step without breaking
its (G)EM ascent. alpha_{c,l} = tau * ((1 - lam) * freq_{c,l} + lam / L_c)
shrinks each component toward the pooled category frequencies (so a
component with no categorical data contributes ~no evidence), and the
uniform floor lam keeps every probability strictly positive even when a
level never appears in a bootstrap's training rows.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_TAU = 3.0
DEFAULT_LAMBDA = 0.1


@dataclass
class CatSpec:
    """Resolved categorical-dimension spec for one observation matrix.

    names       : categorical dim names (as in dataset_names)
    positions   : their column indices in the observation matrix
    level_names : per categorical dim, the level names in code order
    cont_positions : the remaining (continuous) column indices, in order
    """
    names: List[str]
    positions: List[int]
    level_names: List[List[str]]
    cont_positions: List[int] = field(default_factory=list)

    @property
    def n_cat(self):
        return len(self.positions)

    @property
    def n_levels(self):
        return [len(levels) for levels in self.level_names]

    def to_dict(self):
        return {
            "names": list(self.names),
            "positions": [int(p) for p in self.positions],
            "level_names": [list(levels) for levels in self.level_names],
            "cont_positions": [int(p) for p in self.cont_positions],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(names=list(d["names"]), positions=list(d["positions"]),
                   level_names=[list(levels) for levels in d["level_names"]],
                   cont_positions=list(d.get("cont_positions", [])))


def resolve_cat_spec(categorical_dims: Optional[Dict], n_cols: int,
                     dim_names: Optional[Sequence[str]] = None) -> Optional[CatSpec]:
    """Turn a user-facing spec into a CatSpec for a matrix with ``n_cols`` columns.

    ``categorical_dims`` maps a dim NAME (resolved against ``dim_names``) or
    an integer column index to its list of level names (code order). Names
    absent from ``dim_names`` are dropped -- e.g. when overlap gating removed
    that dim -- and if nothing is left this returns None, so the caller
    falls back to the ordinary all-continuous path.
    """
    if not categorical_dims:
        return None
    names, positions, levels = [], [], []
    for key, level_names in categorical_dims.items():
        if isinstance(key, (int, np.integer)):
            pos = int(key)
            name = dim_names[pos] if dim_names is not None else str(pos)
        else:
            if dim_names is None or key not in list(dim_names):
                continue
            pos = list(dim_names).index(key)
            name = key
        if not 0 <= pos < n_cols:
            raise ValueError(f"categorical dim {key!r} -> column {pos} out of range [0, {n_cols})")
        if len(level_names) < 2:
            raise ValueError(f"categorical dim {name!r} needs >= 2 levels, got {level_names}")
        names.append(name)
        positions.append(pos)
        levels.append(list(level_names))
    if not positions:
        return None
    order = np.argsort(positions)
    names = [names[i] for i in order]
    positions = [positions[i] for i in order]
    levels = [levels[i] for i in order]
    pos_set = set(positions)
    cont = [j for j in range(n_cols) if j not in pos_set]
    if not cont:
        raise ValueError("categorical spec leaves no continuous dims; the CFUSN block needs >= 1")
    return CatSpec(names=names, positions=positions, level_names=levels, cont_positions=cont)


def split_observations(observations, spec: CatSpec):
    """Return (X_cont, labels, has_cont).

    X_cont   : (N, p_cont) continuous columns, unchanged
    labels   : (N, C) int, -1 where the categorical value is missing
    has_cont : (N,) bool, row has >= 1 observed continuous value
    """
    observations = np.asarray(observations, dtype=float)
    X_cont = observations[:, spec.cont_positions]
    raw = observations[:, spec.positions]
    labels = np.full(raw.shape, -1, dtype=int)
    for c, L in enumerate(spec.n_levels):
        col = raw[:, c]
        obs = ~np.isnan(col)
        vals = col[obs]
        if vals.size and (np.any(vals != np.round(vals)) or vals.min() < 0 or vals.max() > L - 1):
            raise ValueError(
                f"categorical dim {spec.names[c]!r} must hold integer codes in [0, {L}); "
                f"got values in [{vals.min()}, {vals.max()}]"
                + (" (non-integer codes -- is it still jittered?)"
                   if np.any(vals != np.round(vals)) else "")
            )
        labels[obs, c] = vals.astype(int)
    has_cont = ~np.all(np.isnan(X_cont), axis=1)
    return X_cont, labels, has_cont


def cat_alpha(labels, spec: CatSpec, tau=DEFAULT_TAU, lam=DEFAULT_LAMBDA):
    """Dirichlet pseudo-counts per categorical dim, from the pooled training frequencies."""
    alpha = []
    for c, L in enumerate(spec.n_levels):
        lab = labels[:, c]
        lab = lab[lab >= 0]
        counts = np.bincount(lab, minlength=L).astype(float)
        freq = counts / counts.sum() if counts.sum() > 0 else np.full(L, 1.0 / L)
        alpha.append(tau * ((1.0 - lam) * freq + lam / L))
    return alpha


def cat_log_pdfs(labels, probs):
    """(K, N) categorical log-pmf per component; 0 for missing values."""
    K = probs[0].shape[0]
    out = np.zeros((K, labels.shape[0]))
    for c, P in enumerate(probs):
        lab = labels[:, c]
        obs = lab >= 0
        if obs.any():
            with np.errstate(divide="ignore"):
                out[:, obs] += np.log(P)[:, lab[obs]]
    return out


def cat_mstep(labels, responsibilities, alpha, old_probs=None, frozen=(),
              sample_weights=None):
    """Closed-form MAP update; frozen components keep ``old_probs``."""
    z = np.asarray(responsibilities, dtype=float)
    if sample_weights is not None:
        z = z * np.asarray(sample_weights, dtype=float)[None, :]
    frozen = sorted(frozen) if frozen else []
    new = []
    for c, a in enumerate(alpha):
        L = len(a)
        lab = labels[:, c]
        obs = lab >= 0
        onehot = np.zeros((int(obs.sum()), L))
        onehot[np.arange(onehot.shape[0]), lab[obs]] = 1.0
        counts = z[:, obs] @ onehot                     # (K, L)
        P = (a[None, :] + counts) / (a.sum() + counts.sum(axis=1, keepdims=True))
        if frozen and old_probs is not None:
            P[frozen] = old_probs[c][frozen]
        new.append(P)
    return new


def cat_log_prior(probs, alpha):
    """sum_{k,c,l} alpha_{c,l} log pi_{k,c,l} -- the MAP penalty matching cat_mstep."""
    return float(sum(np.sum(a[None, :] * np.log(P)) for P, a in zip(probs, alpha)))


def cat_probs_from_mask(labels, mask, alpha, sample_weights=None):
    """Smoothed category frequencies over a row mask -- seeds a new component's pi."""
    z = np.asarray(mask, dtype=float)[None, :]
    return [P[0] for P in cat_mstep(labels, z, alpha, sample_weights=sample_weights)]


# ── Analysis-side helpers (fitted-mixture evaluation downstream) ─────────────

class MixtureParams(list):
    """A fitted mixture's continuous component tuples (a plain list of
    (mu, Delta, Gamma) over the CONTINUOUS dims) carrying the categorical
    block alongside, so code that indexes/iterates components keeps working
    and code that evaluates densities can reach the categorical part via
    component_logpdf_full. Only built for fits that have a categorical spec;
    all-continuous fits stay plain lists.
    """

    def __init__(self, components=(), cat_spec=None, cat_probs=None):
        super().__init__(components)
        self.cat_spec = cat_spec
        self.cat_probs = cat_probs

    def __reduce__(self):
        return (MixtureParams, (list(self), self.cat_spec, self.cat_probs))


def wrap_params(params, fit_dict):
    """Attach ``fit_dict``'s categorical block (if any) to a reconstituted
    params list; a fit without one returns ``params`` unchanged."""
    if (fit_dict or {}).get("factorial"):
        # EXPERIMENTAL factorial fits (fit_utils/factorial/export.py): cells
        # evaluated as sums of per-axis block densities.
        from ..factorial.export import wrap_factorial
        return wrap_factorial(params, fit_dict)
    spec = (fit_dict or {}).get("cat_spec")
    if not spec:
        return params
    spec = spec if isinstance(spec, CatSpec) else CatSpec.from_dict(spec)
    probs = [np.asarray(P, dtype=float) for P in fit_dict["cat_probs"]]
    return MixtureParams(params, cat_spec=spec, cat_probs=probs)


def get_cat_spec(params):
    return getattr(params, "cat_spec", None)


def total_dims(params):
    """Number of observation columns a fitted mixture covers (continuous + categorical)."""
    spec = get_cat_spec(params)
    return len(params[0][0]) + (spec.n_cat if spec is not None else 0)


def component_logpdf_full(x, params, c, base_logpdf):
    """Log-density of component ``c`` for full-width rows ``x``.

    ``base_logpdf(x, mu, Delta, Gamma)`` is the caller's continuous CFUSN
    log-pdf. For a MixtureParams with a categorical spec, ``x``'s
    categorical columns are split off (codes, NaN = missing) and their
    log-pmf added; otherwise this is exactly ``base_logpdf(x, *params[c])``.
    """
    spec = get_cat_spec(params)
    factorial = getattr(params, "factorial", None) is not None
    if spec is None:
        if factorial:
            return params.cont_logpdf(x, c)
        return base_logpdf(x, *params[c])
    x = np.atleast_2d(np.asarray(x, dtype=float))
    X_cont, labels, _ = split_observations(x, spec)
    if factorial:
        lp = params.cont_logpdf(X_cont, c)
    else:
        lp = np.atleast_1d(np.asarray(base_logpdf(X_cont, *params[c]), dtype=float))
    return lp + cat_log_pdfs(labels, [np.asarray(P)[c:c + 1] for P in params.cat_probs])[0]
