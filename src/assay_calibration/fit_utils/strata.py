"""
Consequence-stratified mixture: strata, composite samples and arms.

Every variant (row) sits in exactly one **stratum**: its membership pattern
over the standard samples (P/LP, B/LB, population, synonymous) x its
consequence class (missense, PTV, synonymous, other) x a splice flag
(SpliceAI > 0.2, which only survives in splice-detecting assays). The
mixture is fit with the strata as its samples -- shared components, one
weight vector per stratum -- and the standard samples are rebuilt as
**composite samples**, count-weighted mixtures of the strata they include:

    W_r = sum_c M[r, c] W_c / sum_c M[r, c],   M[r, c] = n_c if stratum c is in composite r.

Which strata enter which composite is the **arm** (ARMS). Calibration
strata are those entering the composites used for the LR and prior;
the rest are auxiliary (fitted, shaping the shared components only).

The dimension-free core of the univariate cell model
(analysis/membership_cells/cells.py) in production form; that module keeps
its own names.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROLES = ("P", "B", "g", "s")            # composite rows, standard sample order
ROLE_NAMES = ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "population", "Synonymous"]
CONSEQUENCES = ("missense", "PTV", "synonymous", "other")


def _role_labels(n):
    return list(ROLES) + [f"x{i}" for i in range(4, n)]


def _key(memb, csq, splice):
    m = "+".join(r for r, b in zip(_role_labels(len(memb)), memb) if b) or "none"
    return f"{m}|{csq}" + ("|splice" if splice else "")


# An arm says, per composite role, whether a stratum (membership, consequence,
# splice) is included. Membership in the role is always required.
def _incl_pooled(role, memb, csq, splice):
    if role == "s":  # synonymous reference: not splice-flagged, never P/LP-labelled
        return memb[3] and not splice and not memb[0]
    return memb[ROLES.index(role)]


def _incl_missense(role, memb, csq, splice):
    if role == "s":
        return memb[3] and not splice and not memb[0]
    return memb[ROLES.index(role)] and csq == "missense"


def _incl_pooled_no_splice_blb(role, memb, csq, splice):
    if role == "B" and splice:
        return False
    return _incl_pooled(role, memb, csq, splice)


def _incl_pooled_no_splice_syn_blb(role, memb, csq, splice):
    if role == "B" and splice and csq == "synonymous":
        return False
    return _incl_pooled(role, memb, csq, splice)


def _incl_pooled_benign_pool(role, memb, csq, splice):
    # one benign reference: B/LB and synonymous pooled, high-SpliceAI removed
    if role == "B":
        return (memb[1] or (memb[3] and not memb[0])) and not splice
    if role == "s":
        return False
    return _incl_pooled(role, memb, csq, splice)


def _no_conflict(fn):
    # a row that is both P/LP and B/LB is auxiliary: in neither label composite
    # nor the synonymous reference (population membership is unaffected)
    def wrapped(role, memb, csq, splice):
        if memb[0] and memb[1] and role in ("P", "B", "s"):
            return False
        if isinstance(role, int):  # extra sample column (e.g. TP53 RPV): its own members
            return bool(memb[role])
        return fn(role, memb, csq, splice)
    wrapped.__name__ = fn.__name__
    return wrapped


ARMS = {
    "pooled": _incl_pooled,                                # functional main
    "missense": _incl_missense,                            # combined main; functional consequence-specific
    "pooled_no_splice_blb": _incl_pooled_no_splice_blb,    # functional sensitivity
    "pooled_no_splice_syn_blb": _incl_pooled_no_splice_syn_blb,  # low-priority S2
    "pooled_benign_pool": _incl_pooled_benign_pool,        # low-priority S3
}
ARMS = {name: _no_conflict(fn) for name, fn in ARMS.items()}
# Arms whose composite B/LB already contains the synonymous reference.
ARM_BENIGN_METHOD = {"pooled_benign_pool": "benign"}


@dataclass
class Strata:
    names: List[str]                    # one per stratum (fit sample)
    keys: List[tuple]                   # (membership tuple, consequence, splice)
    row_stratum: np.ndarray             # (N,) stratum index per ms row
    counts: np.ndarray                  # (S,) rows per stratum
    merges: List[str] = field(default_factory=list)

    @property
    def onehot(self) -> np.ndarray:
        """(N, S); rows in no sample (row_stratum -1) are all-False, so the
        fitter leaves them out exactly as it does today."""
        oh = np.zeros((len(self.row_stratum), len(self.names)), dtype=bool)
        r = np.flatnonzero(self.row_stratum >= 0)
        oh[r, self.row_stratum[r]] = True
        return oh

    @property
    def n_roles(self) -> int:
        return len(self.keys[0][0]) if self.keys else 4

    def composition(self, arm: str) -> np.ndarray:
        """(R, S) count matrix M for ``arm`` (R = the ms's sample columns:
        P/LP, B/LB, population, synonymous, then any extra samples): M[r, c] =
        n_c if stratum c enters composite r."""
        incl = ARMS[arm]
        M = np.zeros((self.n_roles, len(self.names)))
        for c, (memb, csq, splice) in enumerate(self.keys):
            for r in range(self.n_roles):
                role = ROLES[r] if r < 4 else r
                if incl(role, memb, csq, splice):
                    M[r, c] = self.counts[c]
        return M

    def to_dict(self) -> dict:
        return dict(names=self.names, keys=[[list(map(bool, m)), c, bool(s)] for m, c, s in self.keys],
                    counts=self.counts.tolist(), merges=self.merges,
                    compositions={a: self.composition(a).tolist() for a in ARMS})


def _atom_key(a):
    """(gene, chrom, pos, ref, alt) normalised so the same SNV matches however
    the dataframe was read (chromosome 17 / "17" / 17.0 / "chr17", position
    as int or float)."""
    g, c, pos, ref, alt = a
    c = str(c)
    if c.endswith(".0"):
        c = c[:-2]
    if c.lower().startswith("chr"):
        c = c[3:]
    return (str(g), c, int(float(pos)), str(ref), str(alt))


def row_annotations(ms, df_integrated: Optional[pd.DataFrame] = None, gene: Optional[str] = None,
                    spliceai_threshold: float = 0.2) -> pd.DataFrame:
    """Per kept row of ``ms``: consequence class and splice flag.

    v2 combined rows carry ``ms.row_meta``. Functional MultiScoresets carry
    each row's member SNV atoms (``ms._row_atoms``): consequence and SpliceAI
    come from those SNVs in ``df_integrated``. Protein-id rows without either
    (TP53, TSC2, predictor sets) take the consequence from the id; their
    SpliceAI > threshold SNVs were already filtered."""
    from ..multivariate_data.rows import consequence_class, SPLICEAI_COLS
    n = len(ms.kept_variants)
    rm = getattr(ms, "row_meta", None)
    if rm is not None and len(rm) == n:
        sai = rm["spliceai_max"].to_numpy(dtype=float)
        return pd.DataFrame({"consequence": rm["consequence"].to_numpy(),
                             "splice": np.nan_to_num(sai, nan=0.0) > spliceai_threshold})
    atoms = getattr(ms, "_row_atoms", None)
    if atoms is not None and df_integrated is not None:
        d = (df_integrated if gene is None
             else df_integrated[df_integrated["Gene"].astype(str).str.upper() == str(gene).upper()])
        d = d[d["hg38_start"].notna()]
        key = [_atom_key(a) for a in zip(d["Gene"], d["Chrom"], d["hg38_start"], d["ref_allele"], d["alt_allele"])]
        sai = d[[c for c in SPLICEAI_COLS if c in d.columns]].max(axis=1).to_numpy()
        csq = d["simplified_consequence"].to_numpy()
        info = {}
        for k, s_, c_ in zip(key, sai, csq):
            prev = info.get(k)
            info[k] = (np.fmax(s_, prev[0]) if prev is not None else s_, c_ if prev is None else prev[1])
        out_c, out_s = [], []
        for row in atoms:
            cs, ss = [], []
            for a in row:
                a = _atom_key(tuple(a)) if len(a) == 5 else tuple(a)
                if a in info:
                    ss.append(info[a][0]); cs.append(info[a][1])
            classes = [consequence_class(None, c) if c == c else "other" for c in cs] or ["other"]
            # a merged row is one protein change: its most specific class
            for pref in ("missense", "PTV", "synonymous", "other"):
                if pref in classes:
                    out_c.append(pref)
                    break
            ss = [x for x in ss if x == x]
            out_s.append(bool(ss) and max(ss) > spliceai_threshold)
        return pd.DataFrame({"consequence": out_c, "splice": out_s})
    ids = [str(v) for v in ms.kept_variants]
    return pd.DataFrame({"consequence": [consequence_class(i) for i in ids], "splice": [False] * n})


def build_strata(ms, annotations: pd.DataFrame, min_stratum: int = 5,
                 row_mask: Optional[np.ndarray] = None) -> Strata:
    """Strata of ``ms``'s rows from their membership pattern (standard
    columns 0-3) and ``annotations`` (row_annotations). A stratum with fewer
    than ``min_stratum`` rows is merged into the largest stratum whose
    composite-role signature is identical in every arm -- pooling weights
    without changing any composite; strata with no such sibling are kept
    (and logged). Rows outside ``row_mask`` (default: rows with no observed
    readout) or in no sample get stratum -1 and are not fitted."""
    sa = np.asarray(ms._sample_assignments, dtype=bool)
    if sa.shape[1] < 4:
        sa = np.hstack([sa, np.zeros((sa.shape[0], 4 - sa.shape[1]), dtype=bool)])
    memb = [tuple(bool(x) for x in r) for r in sa]
    member = sa.any(axis=1)
    if row_mask is None:  # rows with at least one observed readout (the fitter's own rule)
        row_mask = ~np.isnan(np.asarray(ms.scores, dtype=float)).all(axis=1)
    member &= np.asarray(row_mask, dtype=bool)
    keys = [(m, c, bool(s)) for m, c, s in zip(memb, annotations["consequence"], annotations["splice"])]
    kept_keys = [k for k, m in zip(keys, member) if m]
    uniq = sorted(set(kept_keys), key=lambda k: (-kept_keys.count(k), _key(*k)))
    index = {k: i for i, k in enumerate(uniq)}
    row = np.array([index[k] if m else -1 for k, m in zip(keys, member)], dtype=int)
    counts = np.bincount(row[row >= 0], minlength=len(uniq))

    n_roles = sa.shape[1]

    def signature(k):
        return tuple(ARMS[a](ROLES[r] if r < 4 else r, *k) for a in sorted(ARMS) for r in range(n_roles))

    merges = []
    alive = list(range(len(uniq)))
    for c in sorted(alive, key=lambda i: counts[i]):
        if counts[c] >= min_stratum or counts[c] == 0:
            continue
        sib = [j for j in alive if j != c and counts[j] > 0 and signature(uniq[j]) == signature(uniq[c])]
        if not sib:
            merges.append(f"kept thin stratum {_key(*uniq[c])} (n={counts[c]}): no same-role sibling")
            continue
        j = max(sib, key=lambda i: counts[i])
        merges.append(f"merged {_key(*uniq[c])} (n={counts[c]}) into {_key(*uniq[j])}")
        row[row == c] = j
        counts[j] += counts[c]
        counts[c] = 0
    keep = [i for i in range(len(uniq)) if counts[i] > 0]
    remap = {old: new for new, old in enumerate(keep)}
    row = np.array([remap[r] if r >= 0 else -1 for r in row], dtype=int)
    return Strata(names=[_key(*uniq[i]) for i in keep], keys=[uniq[i] for i in keep], row_stratum=row,
                  counts=counts[keep].astype(int), merges=merges)


def strata_roles(strata: Strata) -> List[str]:
    """hpc/prepare.py sample_roles-style role per stratum (for the
    pathogenic-share weighting and noise floor): any P/LP member ->
    "primary"; else B/LB -> "benign"; synonymous -> "synonymous"; gnomAD ->
    "population"; none -> "other"."""
    out = []
    for memb, csq, splice in strata.keys:
        if memb[0]:
            out.append("primary")
        elif memb[1]:
            out.append("benign")
        elif memb[3]:
            out.append("synonymous")
        elif memb[2]:
            out.append("population")
        else:
            out.append("other")
    return out


class StrataView:
    """``ms`` seen by the fitter: same rows and scores, the strata as its
    samples. Everything else is delegated to ``ms``."""

    def __init__(self, ms, strata: Strata):
        self._ms = ms
        self.strata = strata
        self._sample_assignments = strata.onehot
        self.sample_counts = self._sample_assignments.sum(axis=0)
        self.strata_roles = strata_roles(strata)

    @property
    def sample_assignments(self):
        return self._sample_assignments

    @property
    def sample_names(self):
        return list(self.strata.names)

    @property
    def n_samples(self):
        return self._sample_assignments.shape[1]

    def __getattr__(self, name):
        return getattr(self._ms, name)


def compose_weights(W_strata: np.ndarray, M: np.ndarray) -> np.ndarray:
    """(R, K) composite weights from (S, K) stratum weights; a composite
    with no strata gets a NaN row."""
    W = np.asarray(W_strata, dtype=float)
    tot = M.sum(axis=1, keepdims=True)
    out = np.full((M.shape[0], W.shape[1]), np.nan)
    ok = tot[:, 0] > 0
    out[ok] = (M[ok] @ W) / tot[ok]
    return out


def apply_arm(ms, strata: Strata, arm: str, eval_arm: Optional[str] = None):
    """Rewrite ``ms``'s standard sample columns IN PLACE to the composites of
    ``arm`` (a row is in composite r iff its stratum enters r), so the prior
    (population rows), the LR roles and evaluation follow the arm. The
    evaluation controls stay those of ``eval_arm`` (default ``arm``):
    ms._eval_plp / ms._eval_blb, read by mv_analysis.report._eval_labels."""
    rs = strata.row_stratum
    member = rs >= 0

    def cols(M):
        out = np.zeros((len(rs), M.shape[0]), dtype=bool)
        out[member] = (M[:, rs[member]] > 0).T
        return out

    old = np.asarray(ms._sample_assignments, dtype=bool)
    new = cols(strata.composition(arm))
    if new.shape[1] < old.shape[1]:
        new = np.hstack([new, np.zeros((len(rs), old.shape[1] - new.shape[1]), dtype=bool)])
    ms._sample_assignments = new
    ms.sample_counts = ms._sample_assignments.sum(axis=0)
    ev = cols(strata.composition(eval_arm or arm))
    ms._eval_plp = ev[:, 0]
    ms._eval_blb = ev[:, 1]
    ms.strata_arm = arm
    return ms
