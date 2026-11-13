"""Choosing readouts (assays, predictors, both) before the joint model is fit
(plan P). Univariate AUC misses readouts that matter only together with
others, so a cheap multivariate stand-in for the joint model is fit on the
evaluation controls and its loss decomposed with DIP (Koenig et al. 2024,
arXiv:2410.23772): standalone + interaction + dependency per readout.

Stand-in: a class-balanced logistic model on the readouts, missing-value
indicators and pairwise readout products (_StandIn; its odds estimate the LR;
the additive model drops the cross-group products), cross-fitted
(repeated stratified 5-fold; each control's posterior averaged over repeats)
and scored by Cllr (review_common.cllr: prior-free, class-balanced, proper;
1 bit = no evidence). v(model) = 1 - Cllr(model), in bits. For a readout u against the rest R:
    standalone_u = v(u only)            standalone_rest = v(R only)
    additive     = v(u, R with no u x R products)
    full         = v(all readouts)
    interaction  = full - additive
    dependency   = additive - standalone_u - standalone_rest   (< 0: redundant)
    loco         = full - standalone_rest
Intervals: 200 stratified resamples of the controls on the cross-fitted
posteriors. In combined models also predictors vs assays.

Rule (decisions.csv):
  keep        unless both loco and standalone_u have 95% intervals reaching 0
              (a readout that matters only together with others is kept)
  redundant   pair {i, j} on its own: the better single readout explains it
              (gain of the pair over the better single has an interval
              reaching 0, and dependency <= -0.8 x the weaker standalone) ->
              keep the stronger
  too few     fewer than MIN_PER_CLASS controls of either class: keep all
  LOCO<=0     (gentler alternative, also reported) drop a readout only if
              the stand-in does at least as well without it (point estimate)
Honesty (nested.csv): each rule is chosen on one stratified half of the
controls and the stand-in with the chosen readouts vs all readouts is scored
on the other half, 20 times.
meta_predictors.csv: each predictor's per-gene terms combined across genes by a
random-effects meta-analysis, with the between-gene heterogeneity (Q, I^2):
results are pooled, never the data, so every gene keeps its own model.
All stand-in log LRs are capped at the model's Very Strong LR, as the joint
model's are.
agreement.csv compares the stand-in with the fitted joint model's own
decomposition (review/decomposition, review/assay_contribution) when present.

Usage: python -m mv_analysis.review_prefit_selection [--workers N] [--keys ...]
"""
import argparse
import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from mv_analysis import review_common as R

OUT = R.out_dir("prefit_selection")
REPS = 10
N_BOOT = 200
N_NESTED = 20
MIN_PER_CLASS = R.MIN_PER_CLASS
CAP = np.inf  # log LR cap of the model being analysed (set per model in _one)
PREDICTORS = ("REVEL", "MutPred2", "AlphaMissense")
EPS = 1e-6


class _StandIn:
    """Class-balanced logistic model (L2, C = 1) on standardized readouts,
    missing-value indicators (missing readouts imputed at the training mean)
    and pairwise products of readouts. cst (positions within the columns
    given, as interaction_cst in DIP's GAM) restricts the products to pairs
    within the same group: [[0], [1], ...] is fully additive, [u, R] has no
    u x R interaction. Classes weighted equally, so its odds estimate the LR.
    Chosen over gradient-boosted trees, whose LRs were overconfident on the
    smaller models (cross-fitted Cllr above 1 bit)."""

    def __init__(self, cst=None):
        self.cst = cst

    def _design(self, X):
        Z = (X - self.mu) / self.sd
        M = np.isnan(Z).astype(float)
        Z = np.nan_to_num(Z)
        return np.column_stack([Z, M] + [Z[:, i] * Z[:, j] for i, j in self.pairs])

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        d = X.shape[1]
        groups = [list(range(d))] if self.cst is None else self.cst
        self.pairs = [pr for g in groups for pr in itertools.combinations(sorted(g), 2)]
        with np.errstate(all="ignore"):
            self.mu = np.nan_to_num(np.nanmean(X, 0))
            self.sd = np.nan_to_num(np.nanstd(X, 0), nan=1.0) + 1e-9
        self.m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(self._design(X), y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(self._design(X))


def _hgb(n, cst=None):  # noqa: ARG001  (kept name: the stand-in factory)
    return _StandIn(cst)


def _loss(q, y):
    """Per-control Cllr terms (bits) scaled so that their mean over the
    controls (or over a stratified resample of them) is Cllr. q is the
    class-balanced stand-in's posterior, whose odds are the LR; the log LR is
    capped at the model's Very Strong LR (CAP) as the joint model's are."""
    q = np.clip(np.asarray(q, float), EPS, 1 - EPS)
    c, w = R.cllr_costs(np.clip(np.log(q / (1 - q)), -CAP, CAP), y)
    return c * w * len(c)


def _splits(y, reps, seed, groups=None):
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
    if groups is not None:
        yield from LeaveOneGroupOut().split(y, y, groups)
        return
    k = int(min(5, y.sum(), (~y).sum()))
    for r in range(reps):
        yield from StratifiedKFold(k, shuffle=True, random_state=seed + r).split(np.zeros(len(y)), y)


def crossfit(X, y, cols, cst=None, reps=None, seed=0, groups=None, train=None, test=None):
    """Cross-fitted posterior of the stand-in on columns `cols` (None: the
    constant). With train/test: fit on train, predict test (nested use)."""
    if train is not None:
        if cols is None:
            return np.full(len(test), 0.5)
        m = _hgb(len(train), cst).fit(X[np.ix_(train, cols)], y[train])
        return m.predict_proba(X[np.ix_(test, cols)])[:, 1]
    reps = REPS if reps is None else reps
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in _splits(y, reps, seed, groups):
        if cols is None:
            acc[te] += 0.5
        else:
            m = _hgb(len(tr), cst).fit(X[np.ix_(tr, cols)], y[tr])
            acc[te] += m.predict_proba(X[np.ix_(te, cols)])[:, 1]
        cnt[te] += 1
    return acc / cnt


def _boot_idx(y, rng, n=N_BOOT):
    return [np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (True, False)])
            for _ in range(n)]


def _terms(gain, idx):
    """DIP terms (point, lo, hi) from per-control loss reductions of the
    models base->{u, rest, additive, full}."""
    def t(g):
        return dict(standalone_u=g["u"], standalone_rest=g["rest"], additive=g["additive"], full=g["full"],
                    interaction=g["full"] - g["additive"], dependency=g["additive"] - g["u"] - g["rest"],
                    loco=g["full"] - g["rest"])
    point = t({k: v.mean() for k, v in gain.items()})
    boots = [t({k: v[ix].mean() for k, v in gain.items()}) for ix in idx]
    out = dict(point)
    for k in point:
        b = np.array([x[k] for x in boots])
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(b, 2.5), np.percentile(b, 97.5)
    return out


def dip_unit(X, y, u, rest, idx, groups=None, cache=None):
    cache = {} if cache is None else cache
    allc = sorted(u + rest)

    def post(cols, cst=None):
        k = (tuple(cols) if cols is not None else None, None if cst is None else tuple(map(tuple, cst)))
        if k not in cache:
            cache[k] = crossfit(X, y, cols, cst, groups=groups)
        return cache[k]
    base = _loss(post(None), y)
    # interaction_cst indexes the columns passed (positions within allc)
    pos = {c: i for i, c in enumerate(allc)}
    cst = [[pos[c] for c in u], [pos[c] for c in rest]]
    gain = {"u": base - _loss(post(sorted(u)), y), "rest": base - _loss(post(sorted(rest)), y),
            "additive": base - _loss(post(allc, cst), y), "full": base - _loss(post(allc), y)}
    return _terms(gain, idx)


def _data(key):
    d, m = R.prep(key)
    ev = np.asarray(d["eval_mask"], bool)
    X = np.asarray(d["X"], float)[ev]
    y = np.asarray(d["plp"], bool)[ev]
    return X, y, [str(n) for n in d["dims"]], m


def decide(X, y, names, rng, groups=None):
    """Per-readout DIP rows and the kept readout set."""
    n = len(names)
    idx = _boot_idx(y, rng)
    cache = {}
    rows = []
    for i in range(n):
        rest = [j for j in range(n) if j != i]
        if not rest:
            break
        r = dip_unit(X, y, [i], rest, idx, groups, cache)
        r["drop"] = r["loco_lo"] <= 0 and r["standalone_u_lo"] <= 0
        rows.append(dict(readout=names[i], **r))
    keep = {names[i] for i in range(n)} - {r["readout"] for r in rows if r["drop"]}
    if not keep:  # nothing distinguishable from zero: keep the best standalone readout
        keep = {max(rows, key=lambda r: r["standalone_u"])["readout"]}
    # fully redundant pairs among the kept readouts
    base = _loss(cache[(None, None)], y)
    single = {i: base - _loss(cache[((i,), None)], y) for i in range(n) if (((i,), None)) in cache}
    pairs = []
    for i, j in itertools.combinations(range(n), 2):
        if names[i] not in keep or names[j] not in keep:
            continue
        r = dip_unit(X, y, [i], [j], idx, groups, cache)
        si, sj = single[i].mean(), single[j].mean()
        better, weaker = (i, j) if si >= sj else (j, i)
        g_pair = base - _loss(cache[((i, j), None)], y)
        over = np.array([(g_pair - single[better])[ix].mean() for ix in idx])
        red = np.percentile(over, 2.5) <= 0 and r["dependency"] <= -0.8 * min(si, sj)
        pairs.append(dict(readout_a=names[i], readout_b=names[j], redundant=red, keep=names[better],
                          gain_over_better=(g_pair - single[better]).mean(), **r))
    for p in sorted(pairs, key=lambda p: p["dependency"]):
        if p["redundant"] and p["readout_a"] in keep and p["readout_b"] in keep and len(keep) > 1:
            keep.discard(p["readout_b"] if p["keep"] == p["readout_a"] else p["readout_a"])
    # gentler rule: drop only readouts the stand-in does at least as well
    # without (point estimate of loco <= 0); never all of them
    gentle = [names[i] for i in range(n) if not any(r["readout"] == names[i] and r["loco"] <= 0 for r in rows)]
    if not gentle:
        gentle = [max(rows, key=lambda r: r["loco"])["readout"]]
    return rows, pairs, sorted(keep, key=names.index), gentle


def _one(key):
    global CAP
    warnings.filterwarnings("ignore")
    X, y, names, m = _data(key)
    CAP = R.log_c(float(np.median(R.mv_npz(key)["priors"])))
    rng = np.random.default_rng(0)
    base = dict(key=key, panel=m["panel"], gene=m["gene"], n_readouts=len(names), n_plp=int(y.sum()),
                n_blb=int((~y).sum()))
    if len(names) < 2:
        return [], [], [], [dict(base, decision="single readout", kept=";".join(names))], []
    if min(y.sum(), (~y).sum()) < MIN_PER_CLASS:
        return [], [], [], [dict(base, decision="too few controls (keep all)", kept=";".join(names))], []
    rows, pairs, keep, gentle = decide(X, y, names, rng)
    rows = [dict(base, **r) for r in rows]
    pairs = [dict(base, **p) for p in pairs]
    grp = []
    if m["panel"] == "combined":
        pi = [i for i, n in enumerate(names) if n in PREDICTORS]
        ai = [i for i in range(len(names)) if i not in pi]
        if pi and ai:
            grp.append(dict(base, unit="predictors vs assays", **dip_unit(X, y, pi, ai, _boot_idx(y, rng))))
    dec = [dict(base, decision="selected", kept=";".join(keep),
                dropped=";".join(n for n in names if n not in keep),
                kept_loco_rule=";".join(gentle), dropped_loco_rule=";".join(n for n in names if n not in gentle))]
    # nested: choose on one half, score on the other
    nest = []
    for rep in range(N_NESTED):
        a = np.concatenate([rng.permutation(np.flatnonzero(y == c))[: (y == c).sum() // 2] for c in (True, False)])
        b = np.setdiff1d(np.arange(len(y)), a)
        global REPS
        old, REPS = REPS, 2
        try:
            _, _, ka, ga = decide(X[a], y[a], names, np.random.default_rng(rep))
        finally:
            REPS = old
        for rule, cols in (("all readouts", list(range(len(names)))),
                           ("CI rule", [names.index(n) for n in ka]),
                           ("LOCO <= 0 rule", [names.index(n) for n in ga])):
            q = crossfit(X, y, cols, train=a, test=b)
            nest.append(dict(base, rep=rep, rule=rule, kept=";".join(names[c] for c in cols), n_kept=len(cols),
                             loss=float(_loss(q, y[b]).mean()),
                             loss_constant=float(_loss(np.full(len(b), 0.5), y[b]).mean())))
    return rows, pairs, grp, dec, nest


def meta_predictors(dip):
    """Per-gene stand-in DIP terms of each predictor combined across genes by
    a DerSimonian-Laird random-effects meta-analysis (results pooled, not data:
    every gene keeps its own model). Standard errors from each gene's
    resampling interval; Cochran's Q and I^2 measure how much the genes
    differ. High I^2: the predictor's role is gene-specific and decisions stay
    per gene; low I^2: the pooled estimate is the better-supported one."""
    from scipy.stats import chi2
    out = []
    x = dip[(dip["panel"] == "predictor") & dip["readout"].isin(PREDICTORS)] if len(dip) else dip
    for (pred), g in (x.groupby("readout") if len(x) else []):
        for term in ("loco", "standalone_u", "dependency", "interaction"):
            est = g[term].to_numpy(float)
            se = ((g[f"{term}_hi"] - g[f"{term}_lo"]) / 3.92).to_numpy(float)
            ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
            est, se = est[ok], se[ok]
            k = len(est)
            if k < 2:
                continue
            w = 1 / se ** 2
            fe = (w * est).sum() / w.sum()
            Q = float((w * (est - fe) ** 2).sum())
            tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
            wr = 1 / (se ** 2 + tau2)
            re = (wr * est).sum() / wr.sum()
            se_re = np.sqrt(1 / wr.sum())
            out.append(dict(readout=pred, term=term, n_genes=k, pooled=re, lo=re - 1.96 * se_re, hi=re + 1.96 * se_re,
                            Q=Q, p_heterogeneity=float(chi2.sf(Q, k - 1)), I2=max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0,
                            tau=np.sqrt(tau2), genes=";".join(g["gene"].astype(str)[ok])))
    return pd.DataFrame(out)


def agreement(dip):
    """Stand-in vs the fitted joint model's own decomposition, per readout."""
    out = []
    f2 = R.REVIEW_DIR / "decomposition" / "dip.csv"
    do = R.REVIEW_DIR / "assay_contribution" / "drop_one.csv"
    if not dip.empty and f2.exists():
        j = pd.read_csv(f2)
        x = dip.merge(j, left_on=["key", "readout"], right_on=["key", "unit"], suffixes=("_proxy", "_joint"))
        for c in ("standalone_u", "interaction", "dependency", "loco"):
            if len(x) > 2:
                out.append(dict(source="decomposition", term=c, n=len(x),
                                spearman=x[f"{c}_proxy"].corr(x[f"{c}_joint"], method="spearman")))
    if not dip.empty and do.exists():
        d = pd.read_csv(do)
        x = dip.merge(d, left_on=["key", "readout"], right_on=["key", "dropped"])
        if len(x):
            col = "delta_cllr" if "delta_cllr" in x else "delta_log_loss"
            joint_drop = x[col] < 0
            out.append(dict(source="drop_one", term="drop decision agrees", n=len(x), score=col,
                            agreement=float((joint_drop == x["drop"]).mean()),
                            spearman=(-x["loco"]).corr(x[col], method="spearman")))
    return pd.DataFrame(out)


def run(workers=8, keys=None):
    warnings.filterwarnings("ignore")
    ks = [k for k, _ in R.paper_keys() if keys is None or k in keys]
    D, P, G, C, N = [], [], [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for k, (rows, pairs, grp, dec, nest) in zip(ks, ex.map(_one, ks)):
            D += rows
            P += pairs
            G += grp
            C += dec
            N += nest
            print(k, dec[0]["decision"], dec[0].get("kept"), "| dropped:", dec[0].get("dropped", ""), flush=True)
    dip = pd.DataFrame(D)
    dip.to_csv(OUT / "dip_proxy.csv", index=False)
    pd.DataFrame(P).to_csv(OUT / "pairs_proxy.csv", index=False)
    pd.DataFrame(G).to_csv(OUT / "groups_proxy.csv", index=False)
    pd.DataFrame(C).to_csv(OUT / "decisions.csv", index=False)
    nd = pd.DataFrame(N)
    nd.to_csv(OUT / "nested.csv", index=False)
    meta_predictors(dip).to_csv(OUT / "meta_predictors.csv", index=False)
    agreement(dip).to_csv(OUT / "agreement.csv", index=False)
    if len(nd):
        print(nd.groupby(["panel", "rule"])[["loss", "n_kept"]].median().round(4).to_string(), flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keys", nargs="*")
    a = ap.parse_args()
    run(a.workers, a.keys)
