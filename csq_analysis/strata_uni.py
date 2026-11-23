"""Univariate consequence strata for the paper: keys, paper arms, thin-stratum merging, composition, arm scoresets.

A row's stratum key is (membership in P/LP, B/LB, gnomAD, synonymous; paper consequence class; SpliceAI flag in
splice-detecting assays; AF-proxy flag for gnomAD missense). Every arm role below is a predicate on these features
only, so each stratum is either wholly inside or wholly outside every composite (all-or-nothing composition, as in
fit_utils/strata.py). Thin strata (< min_stratum rows) are merged, smallest first, into the largest stratum with the
same role signature across ALL paper arms, which leaves every composite unchanged.

Composition uses fit_utils/strata.compose_weights. Arms are local to the paper so production strata.ARMS is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROLE_NAMES = {"P": "Pathogenic/Likely Pathogenic", "B": "Benign/Likely Benign", "g": "gnomAD", "s": "Synonymous"}
ROLES = ("P", "B", "g", "s")
PROXY_AF = 1e-4      # AF-based proxy benign (CanVIG Rec 4, population part only); sensitivity cutoffs set in pilots


# ── row features → stratum keys ──────────────────────────────────────────────────────────────────────────────────

def row_features(rows, proxy_af=PROXY_AF):
    """rows: csq_analysis.annotate rows of one dataset. Adds splice / proxy flags used by the keys and arms."""
    r = rows.copy()
    r["splice"] = r.detects_splice & (r.spliceai_max.fillna(0) > 0.2)
    r["proxy"] = r["pop"] & r.csq.eq("missense") & (r.gnomad_maf.fillna(0) >= proxy_af) & ~r.plp
    r["conflict"] = r.plp & r.blb
    return r


def keys(r):
    return list(zip(r.plp, r.blb, r["pop"], r.syn, r.csq, r.splice, r.proxy))


def key_name(k):
    m = "+".join(n for n, b in zip(("PLP", "BLB", "gnomAD", "Syn"), k[:4]) if b) or "assayed"
    return f"{m}|{k[4]}" + ("|splice" if k[5] else "") + ("|AFproxy" if k[6] else "")


# ── paper arms: role -> predicate on a feature row (pandas row or dict) ──────────────────────────────────────────

def _not_conflict(f):
    return not (f["plp"] and f["blb"])


def _syn_ref(f):
    """Synonymous reference: synonymous, not splice-flagged, never P/LP."""
    return bool(f["syn"]) and not f["splice"] and not f["plp"]


def _pred(P=None, B=None, g=None, s=None):
    def role(r, f):
        if r in ("P", "B", "s") and not _not_conflict(f):
            return False
        if r == "P":
            return bool(f["plp"]) and (P is None or P(f))
        if r == "B":
            return (bool(f["blb"]) and (B is None or B(f))) if B != "proxy" else \
                ((bool(f["blb"]) and f["csq"] == "missense") or bool(f["proxy"]))
        if r == "g":
            return bool(f["pop"]) and (g is None or g(f))
        if r == "s":
            return _syn_ref(f) if s is None else s(f)
        return False
    return role


def _csq(*c):
    return lambda f: f["csq"] in c


NONE = lambda f: False   # noqa: E731

# name: (role predicate, benign_method or None = production per-dataset[, options])
# options: g_all_rows -> the population composite's POINTS are every assayed row passing the g predicate (fitted or
#          not); its weights come from the fitted strata passing it. Valid for the standard (PN) prior, which uses only
#          f_P and f_B at the population points (point_ranges.get_fit_prior, mode 'standard').
#          manual_prior_from -> prior fixed at that arm's median prior (same fit config and mapping).
#          keep_extreme -> production keep_extreme_prior_fits (priors of exactly 0/1 kept, median clipped to
#          [0.001, 0.999]); used for PTV priors, which are legitimately ~1 when every PTV is LoF (BRCA1 SGE).
ASSAYED = dict(g_all_rows=True)
EXTREME = dict(keep_extreme=True)
ARMS = {
    # consequence-agnostic: ExCALIBR-equivalent composites
    "pool": (_pred(), None),
    # missense arm (user spec): missense P/LP vs missense B/LB, missense prior
    "mis": (_pred(P=_csq("missense"), B=_csq("missense"), g=_csq("missense")), "benign"),
    # missense benign-reference variants
    "mis_Ball": (_pred(P=_csq("missense"), g=_csq("missense")), None),
    "mis_Bmis_syn": (_pred(P=_csq("missense"), B=_csq("missense"), g=_csq("missense")), "avg"),
    "mis_Ball_syn": (_pred(P=_csq("missense"), g=_csq("missense")), "avg"),
    "mis_proxy": (_pred(P=_csq("missense"), B="proxy", g=_csq("missense")), "benign"),
    # missense prior variants
    "mis_gall": (_pred(P=_csq("missense"), B=_csq("missense")), "benign"),
    "mis_gassayed": (_pred(P=_csq("missense"), B=_csq("missense"), g=_csq("missense")), "benign", ASSAYED),
    # PTV arms (benign reference as production; prior options)
    "ptv_poolprior": (_pred(P=_csq("PTV")), None, dict(manual_prior_from="pool")),
    "ptv_gptv": (_pred(P=_csq("PTV"), g=_csq("PTV")), None, EXTREME),
    "ptv_gmisptv": (_pred(P=_csq("PTV"), g=_csq("missense", "PTV")), None, EXTREME),
    "ptv_gall": (_pred(P=_csq("PTV")), None),
    # mean posterior over all assayed PTVs (the standard-mode EM fixed point with assayed PTVs as the population)
    "ptv_gassayed": (_pred(P=_csq("PTV"), g=_csq("PTV")), None, dict(ASSAYED, **EXTREME)),
    # other consequences (production benign reference, pooled gnomAD prior)
    "syn": (_pred(P=_csq("synonymous")), None),
    "splice": (_pred(P=_csq("splice")), None),
    "intronic": (_pred(P=_csq("intronic")), None),
}


def arm_options(arm):
    a = ARMS[arm]
    return a[2] if len(a) > 2 else {}


def _role_in(arm, r, f, fitted=True):
    """Row f in role r of arm. For g_all_rows arms, an unfitted (assayed-only) row is in g if it passes g's
    consequence predicate (its membership in gnomAD is not required)."""
    role = ARMS[arm][0]
    if r == "g" and arm_options(arm).get("g_all_rows"):
        return role("g", dict(f, pop=True))
    return role(r, f) if fitted else False


def role_signature(f):
    return tuple(_role_in(a, r, f) for a in ARMS for r in ROLES)


# ── strata ───────────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class UniStrata:
    dataset: str
    rows: pd.DataFrame                 # feature rows (one per scoreset row)
    stratum: np.ndarray                # (N,) stratum index, -1 = not fitted (in no sample)
    names: list
    counts: np.ndarray
    merges: list = field(default_factory=list)

    @property
    def onehot(self):
        fitted = self.stratum >= 0
        m = np.zeros((fitted.sum(), len(self.names)), bool)
        m[np.arange(fitted.sum()), self.stratum[fitted]] = True
        return m

    @property
    def fitted(self):
        return self.stratum >= 0

    def composition(self, arm):
        """(4, S) count matrix: M[r, c] = n_c if stratum c enters role r of `arm`, else 0."""
        rep = self.rows[self.fitted].groupby(self.stratum[self.fitted]).head(1)
        rep = rep.assign(_s=self.stratum[self.fitted][self.rows[self.fitted].index.get_indexer(rep.index)])
        M = np.zeros((len(ROLES), len(self.names)))
        for f in rep.to_dict("records"):
            c = int(f["_s"])
            for i, r in enumerate(ROLES):
                if _role_in(arm, r, f):
                    M[i, c] = self.counts[c]
        return M


def build(dataset, rows, min_stratum=5, proxy_af=PROXY_AF):
    """Strata of one dataset's rows (csq_analysis.annotate rows). Rows in no sample are not fitted (-1)."""
    r = row_features(rows, proxy_af).reset_index(drop=True)
    in_sample = (r.plp | r.blb | r["pop"] | r.syn).to_numpy()
    ks = keys(r)
    uniq = sorted({k for k, s in zip(ks, in_sample) if s}, key=lambda k: (k[:4][::-1], k[4:]))
    index = {k: i for i, k in enumerate(uniq)}
    st = np.array([index[k] if s else -1 for k, s in zip(ks, in_sample)], int)
    counts = np.bincount(st[st >= 0], minlength=len(uniq))
    first = {c: i for i, c in reversed(list(enumerate(st))) if c >= 0}
    sig = {c: role_signature(r.iloc[i]) for c, i in first.items()}
    alive = set(range(len(uniq)))
    merges = []
    while True:
        thin = sorted([c for c in alive if 0 < counts[c] < min_stratum], key=lambda c: (counts[c], c))
        done = True
        for c in thin:
            sib = [d for d in alive if d != c and sig[d] == sig[c]]
            if not sib:
                continue
            d = max(sib, key=lambda x: (counts[x], -x))
            st[st == c] = d
            counts[d] += counts[c]
            counts[c] = 0
            alive.discard(c)
            merges.append((key_name(uniq[c]), key_name(uniq[d])))
            done = False
            break
        if done:
            break
    keep = sorted(alive)
    remap = {c: i for i, c in enumerate(keep)}
    st = np.array([remap[c] if c >= 0 else -1 for c in st], int)
    names = [key_name(uniq[c]) for c in keep]
    return UniStrata(dataset, r, st, names, np.asarray([counts[c] for c in keep], int), merges)


def compose(fit_record, M):
    """Fit record with stratum weights replaced by the arm's composite weights (empty composites dropped later)."""
    from src.assay_calibration.fit_utils.strata import compose_weights
    W = np.asarray(fit_record["fit"]["weights"], float)
    out = {"fit": dict(fit_record["fit"])}
    out["fit"]["weights"] = compose_weights(W, M)
    return out


def arm_scoreset(us, arm):
    """(BasicScoreset, row index, roles kept) for an arm: rows of any non-empty composite, columns = the non-empty
    composites only (process_component_fits indexes fit weights by sample-name position, while
    BasicScoreset.sample_assignments drops empty columns, so empty composites must not be columns at all).
    Compose the fit weights with composition(arm)[rows of the kept roles]."""
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    r = us.rows
    recs = r.to_dict("records")
    fitted = us.fitted
    mask = np.zeros((len(r), len(ROLES)), bool)
    for i, rr in enumerate(ROLES):
        mask[:, i] = [_role_in(arm, rr, f, fitted=bool(fi)) for f, fi in zip(recs, fitted)]
    roles = [i for i in range(len(ROLES)) if mask[:, i].any()]
    keep = mask[:, roles].any(1)
    ss = BasicScoreset(r.score.to_numpy(float)[keep], mask[keep][:, roles],
                       sample_names=[ROLE_NAMES[ROLES[i]] for i in roles])
    return ss, np.flatnonzero(keep), roles
