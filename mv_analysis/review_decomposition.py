"""How much of the joint model's evidence each readout carries, and how much
comes from readouts acting together (plan F2). No refit: readout subsets are
marginalized out of the fitted mixtures (review_assay_contribution.score_masked,
cached under review/assay_contribution/cache/).

Value of a readout set S (Shapley "game"), in bits:
    v(S) = 1 - Cllr(S)
where Cllr (review_common.cllr; Brummer & du Preez 2006) scores the
evaluation controls' out-of-bag log LR+ under S at even prior odds with the
two classes weighted equally: LR only, proper, 1 bit with no evidence, so
v({}) = 0 and v is the share of the label uncertainty S removes (0-1).
Three versions of the log LR+ are scored (column `score`;
review_common.evidence_log_lrs), all capped at the Very Strong LR (8 points:
mixture-tail LRs beyond it never become evidence):
  median      the model's LR estimate (information; the primary game)
  percentile  the evidence rule's one-sided 5th/95th-percentile bound, so the
              bootstrap variance a readout adds counts against it
  points      the LR implied by the points assigned (evidence as applied)
Percentiles and points use the full model's out-of-bag prior per control for
every S. DIP and the per-variant split use the capped median.

Units are readouts, or readout groups for models with more than MAX_EXACT
readouts (TP53: Kato promoter PCs, Giacomelli, and in the combined model the
three predictors as one unit). Shapley values are exact over all 2^n unit
subsets.

  values.csv      v(S) for every unit subset
  shapley.csv     per model and unit: Shapley value phi (bits) and share =
                  phi / v(all units) (the credit, information shared by
                  several units split evenly among them); unique = v(all) -
                  v(all but the unit) (what only this unit adds) and its share;
                  Cllr of the unit alone and Cllr_min (after the optimal
                  monotone recalibration: discrimination only); 95% intervals
                  from 200 stratified resamples of the controls (fits fixed)
  pairs.csv       Shapley interaction index for every pair of units
                  (> 0 synergy: together worth more than apart; < 0 overlap)
  dip.csv         DIP decomposition (Koenig et al. 2024, arXiv:2410.23772)
                  of each unit u against the rest R, and in combined models of
                  the predictors against the assays, in bits (1 - Cllr). Its
                  four models are class-balanced logistic recalibrations of the
                  out-of-bag log LR+, cross-fitted on the controls (10 x 5-fold):
                    standalone u : log LR_u          standalone R : log LR_R
                    additive     : log LR_u, log LR_R (two weights; no interaction)
                    joint        : log LR_{u+R} (the fitted joint model)
                  interaction = v(joint) - v(additive)
                  dependency  = v(additive) - v(u) - v(R)   (< 0: redundant)
                  loco        = v(joint) - v(R) = v(u) + interaction + dependency
                  plus the uncalibrated values of the same sets and of the
                  plain sum log LR_u + log LR_R (what adding points assumes)
  per_variant/<key>.csv  every model row's median log LR+ (all fits) split into
                  per-unit Shapley terms that sum to it; for models with up to
                  4 units also the pairwise interaction terms
  fig_shares.png  shares per model

Usage: python -m mv_analysis.review_decomposition [--workers N] [--keys ...]
"""
import argparse
import itertools
import math
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from mv_analysis import review_assay_contribution as A
from mv_analysis import review_common as R

OUT = R.out_dir("decomposition")
MAX_EXACT = 8
N_BOOT = 200
SCORES = ("median", "percentile", "points")
PREDICTORS = ("REVEL", "MutPred2", "AlphaMissense")


def _tp53_unit(name, combined):
    if combined and name in PREDICTORS:
        return "predictors"
    if name.startswith("AIP1"):
        return "Kato (AIP1 PCs)"
    if name.startswith("Giac_"):
        return "Giacomelli"
    return name


def units(key):
    """[(unit name, (readouts...))] in the model's readout order."""
    names = [str(n) for n in R.prep(key)[0]["dims"]]  # = the fitted ms's dataset_names order
    if len(names) <= MAX_EXACT:
        return [(n, (n,)) for n in names], names
    if R.prep(key)[1]["gene"].upper() != "TP53":
        raise NotImplementedError(f"{key}: {len(names)} readouts and no grouping defined")
    comb = key.startswith("combined")
    groups = {}
    for n in names:
        groups.setdefault(_tp53_unit(n, comb), []).append(n)
    return [(u, tuple(r)) for u, r in groups.items()], names


def _masked(names, us, mask):
    kept = {r for i, (_, rs) in enumerate(us) if mask >> i & 1 for r in rs}
    return tuple(n for n in names if n not in kept)


def _controls(key):
    """Evaluation controls with out-of-bag fits (oob_eval's set), their labels
    and the full model's out-of-bag prior."""
    d, _ = R.prep(key)
    mv = R.mv_npz(key)
    rows = np.flatnonzero(d["eval_mask"])
    rows = rows[mv["oob"][:, rows].any(0)]
    oob = mv["oob"][:, rows].astype(bool)
    pri = np.array([np.median(mv["priors"][oob[:, i]]) for i in range(len(rows))])
    return rows, np.asarray(d["plp"], bool)[rows], pri


def _subset(key, masked):
    """With `masked` marginalized out: (median log LR+ over all fits per model
    row, capped log LR+ per control as review_common.evidence_log_lrs:
    {median, percentile, points}). Out-of-bag percentiles and points use the
    full model's out-of-bag prior per control, so subsets differ only in
    their LRs."""
    from mv_analysis.production_oob import _points
    A._single_threaded()
    warnings.filterwarnings("ignore")
    L, _, seeds = A.score_masked(key, masked)
    mv = R.mv_npz(key)
    rows, _, prior = _controls(key)
    pos = {int(s): i for i, s in enumerate(mv["seeds"])}
    idx = np.array([pos.get(int(s), -1) for s in seeds])
    ok = idx >= 0
    oob = mv["oob"][idx[ok]][:, rows].astype(bool)
    Lc = np.where(oob, L[ok][:, rows], np.nan)
    with np.errstate(all="ignore"):
        med_all = np.nanmedian(L, axis=0)
        med = np.nanmedian(Lc, axis=0)
        lo, hi = np.nanpercentile(Lc, 5, axis=0), np.nanpercentile(Lc, 95, axis=0)
    pts = np.array([_points(a, b, pr, mv["point_values"]) for a, b, pr in zip(lo, hi, prior)], float)
    return med_all, R.evidence_log_lrs(med, lo, hi, prior, pts)


def _job(key, mask, masked):
    try:
        return key, mask, _subset(key, masked)
    except Exception as e:  # noqa: BLE001
        return key, mask, f"FAILED {type(e).__name__}: {e}"


def _weights(n):
    return np.array([math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)])


def shapley(v, n):
    """Exact Shapley values from v indexed by bitmask (array of 2^n, v[0] = 0);
    v may carry trailing axes (resamples, rows)."""
    w = _weights(n)
    phi = []
    for i in range(n):
        acc = 0
        for S in range(1 << n):
            if not S >> i & 1:
                acc = acc + w[bin(S).count("1")] * (v[S | 1 << i] - v[S])
        phi.append(acc)
    return np.array(phi)


def interactions(v, n):
    """Shapley interaction index for every pair {i, j}."""
    out = {}
    if n < 2:
        return out
    w = [math.factorial(s) * math.factorial(n - s - 2) / math.factorial(n - 1) for s in range(n - 1)]
    for i, j in itertools.combinations(range(n), 2):
        acc = 0
        for S in range(1 << n):
            if not (S >> i & 1 or S >> j & 1):
                acc = acc + w[bin(S).count("1")] * (v[S | 1 << i | 1 << j] - v[S | 1 << i] - v[S | 1 << j] + v[S])
        out[(i, j)] = acc
    return out


def _crossfit(X, y, seed=0, reps=10):
    """Cross-fitted log LR from a class-balanced logistic recalibration (mild
    L2), averaged over repeated stratified folds: with the classes weighted
    equally the model's odds are the LR (no prior)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    k = int(min(5, y.sum(), (~y).sum()))
    if k < 2:
        return np.full(len(y), np.nan)
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    for r in range(reps):
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed + r).split(X, y):
            m = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
            acc[te] += m.decision_function(X[te])
            cnt[te] += 1
    return acc / cnt


def dip(lr_u, lr_r, lr_full, y):
    """DIP terms (bits) for unit u vs rest R from out-of-bag log LR+ per
    control. v = 1 - Cllr: the baseline (no readouts, LR = 1, also the
    class-balanced constant model) is 1 bit, so no baseline correction is
    needed."""
    f = lambda *c: np.nan_to_num(np.column_stack(c))  # noqa: E731
    vc = {name: 1 - R.cllr(_crossfit(X, y), y) for name, X in
          (("u", f(lr_u)), ("rest", f(lr_r)), ("joint", f(lr_full)), ("additive", f(lr_u, lr_r)))}
    vr = {name: 1 - R.cllr(lr, y) for name, lr in
          (("u", lr_u), ("rest", lr_r), ("joint", lr_full), ("naive_sum", np.nan_to_num(lr_u) + np.nan_to_num(lr_r)))}
    return dict(standalone_u=vc["u"], standalone_rest=vc["rest"], joint=vc["joint"], additive=vc["additive"],
                interaction=vc["joint"] - vc["additive"], dependency=vc["additive"] - vc["u"] - vc["rest"],
                loco=vc["joint"] - vc["rest"],
                **{f"raw_{k}": x for k, x in vr.items()})


def _game(key, m, us, lrs, y, idx, score):
    """Shapley values, unique contributions, pair interactions and subset
    values for one score (v = 1 - Cllr of that capped log LR+)."""
    n = len(us)
    full = (1 << n) - 1
    v = np.zeros(1 << n)
    vb = np.zeros((1 << n, len(idx)))
    for S, lr in lrs.items():
        v[S] = 1 - R.cllr(lr, y)
        c = R.cllr_costs(lr, y)[0]
        for b, ix in enumerate(idx):
            yy = y[ix]
            vb[S, b] = 1 - (c[ix] * np.where(yy, 0.5 / yy.sum(), 0.5 / (~yy).sum())).sum()
    phi, phib = shapley(v, n), shapley(vb, n)
    share_b = phib / np.where(vb[full] != 0, vb[full], np.nan)
    uniq_b = vb[full] - np.stack([vb[full & ~(1 << i)] for i in range(n)])
    pc = lambda x, q: float(np.nanpercentile(x, q))  # noqa: E731
    base = dict(key=key, panel=m["panel"], gene=m["gene"], score=score, n_units=n, n_controls=len(y),
                n_plp=int(y.sum()), few_controls=bool(min(y.sum(), (~y).sum()) < R.MIN_PER_CLASS),
                cllr_full=1 - float(v[full]), cllr_min_full=R.cllr_min(lrs[full], y), v_all=float(v[full]))
    sh = [dict(base, unit=us[i][0], readouts=";".join(us[i][1]), phi=phi[i],
               phi_lo=pc(phib[i], 2.5), phi_hi=pc(phib[i], 97.5),
               share=phi[i] / v[full] if v[full] else np.nan,
               share_lo=pc(share_b[i], 2.5), share_hi=pc(share_b[i], 97.5),
               # what only this unit adds: the loss from removing it
               unique=v[full] - v[full & ~(1 << i)], unique_lo=pc(uniq_b[i], 2.5), unique_hi=pc(uniq_b[i], 97.5),
               unique_share=(v[full] - v[full & ~(1 << i)]) / v[full] if v[full] else np.nan,
               v_alone=v[1 << i], v_without=v[full & ~(1 << i)],
               cllr_alone=1 - v[1 << i], cllr_min_alone=R.cllr_min(lrs[1 << i], y)) for i in range(n)]
    ii, iib = interactions(v, n), interactions(vb, n)
    pr = [dict(base, unit_a=us[i][0], unit_b=us[j][0], interaction=ii[(i, j)],
               lo=pc(iib[(i, j)], 2.5), hi=pc(iib[(i, j)], 97.5),
               v_a=v[1 << i], v_b=v[1 << j], v_ab=v[1 << i | 1 << j]) for (i, j) in ii]
    vals = [dict(key=key, score=score, subset=";".join(us[i][0] for i in range(n) if S >> i & 1),
                 n_kept=bin(S).count("1"), v=v[S], cllr=1 - v[S]) for S in range(1 << n)]
    return sh, pr, vals, base


def analyse(key, m, us, res, rng):
    n = len(us)
    rows, y, _ = _controls(key)
    d, _ = R.prep(key)
    full = (1 << n) - 1
    # stratified resamples of the controls (the fits held fixed)
    idx = [np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (True, False) if (y == c).any()])
           for _ in range(N_BOOT)]
    sh, pr, vals = [], [], []
    for score in SCORES:
        a, b, c, base = _game(key, m, us, {S: res[S][1][score] for S in range(1, 1 << n)}, y, idx, score)
        sh += a
        pr += b
        vals += c
        if score == "median":
            base_med = base
    lr = {S: res[S][1]["median"] for S in range(1, 1 << n)}
    base = base_med
    # DIP: each unit vs the rest; combined models also predictors vs assays
    pairs = [(us[i][0], 1 << i) for i in range(n)] if n > 1 else []
    if m["panel"] == "combined" and n > 1:
        pm = sum(1 << i for i in range(n) if set(us[i][1]) <= set(PREDICTORS))
        if 0 < pm < full:
            pairs.append(("predictors vs assays", pm))
    dp = [dict(base, unit=name, **dip(lr[S], lr[full & ~S], lr[full], y)) for name, S in pairs]
    # per variant: Shapley split of each row's median log LR+ (all fits)
    nr = len(d["variants"])
    va = np.zeros((1 << n, nr))
    cap = R.log_c(float(np.median(R.mv_npz(key)["priors"])))
    for S in range(1, 1 << n):
        va[S] = np.clip(np.nan_to_num(res[S][0]), -cap, cap)
    pv = pd.DataFrame(dict(variant=np.asarray(d["variants"]),
                           label=np.where(d["plp"], "P/LP", np.where(d["blb"], "B/LB",
                                          np.where(d["gnomad"], "gnomAD", "other"))),
                           eval_control=np.asarray(d["eval_mask"], bool), log_lr=va[full]))
    for i, x in enumerate(shapley(va, n)):
        pv[f"phi|{us[i][0]}"] = x
    if n <= 4:
        for (i, j), x in interactions(va, n).items():
            pv[f"pair|{us[i][0]}|{us[j][0]}"] = x
    assert np.allclose(pv.filter(like="phi|").sum(1), pv["log_lr"]), f"{key}: Shapley terms do not sum to log LR"
    return sh, pr, vals, dp, pv


def _figure(sh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(dict.fromkeys(sh["key"]))
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(keys))))
    for r, k in enumerate(keys):
        x = sh[sh.key == k]
        left_pos, left_neg = 0.0, 0.0
        for _, row in x.iterrows():
            s = row.share * 100
            if s >= 0:
                ax.barh(r, s, left=left_pos, color="0.55", edgecolor="white", lw=0.6)
                if s > 7:
                    ax.text(left_pos + s / 2, r, row.unit[:14], ha="center", va="center", fontsize=4.5, color="white")
                left_pos += s
            else:
                ax.barh(r, s, left=left_neg, color="#c0504d", edgecolor="white", lw=0.6)
                left_neg += s
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("share of the joint model's information (%)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_shares.png", dpi=200)
    plt.close(fig)


def run(workers=8, keys=None):
    warnings.filterwarnings("ignore")
    todo = [(k, m) for k, m in R.paper_keys() if keys is None or k in keys]
    spec = {}
    for k, m in todo:
        us, names = units(k)
        if len(us) > MAX_EXACT:
            raise NotImplementedError(f"{k}: {len(us)} units")
        spec[k] = (m, us, names)
        print(k, len(us), "units:", [u for u, _ in us], flush=True)
    res = {k: {} for k in spec}
    jobs = [(k, S, _masked(names, us, S)) for k, (m, us, names) in spec.items() for S in range(1, 1 << len(us))]
    # largest models first so they don't finish last
    jobs.sort(key=lambda j: -len(spec[j[0]][1]))
    print(len(jobs), "subsets", flush=True)
    failed = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            k, S, r = f.result()
            if isinstance(r, str):
                failed.append((k, S, r))
                print(k, S, r, flush=True)
            else:
                res[k][S] = r
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} subsets", flush=True)
    SH, PR, VA, DP = [], [], [], []
    (OUT / "per_variant").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for k, (m, us, names) in spec.items():
        if len(res[k]) != (1 << len(us)) - 1:
            print(f"{k}: missing subsets; left out", flush=True)
            continue
        sh, pr, vals, dp, pv = analyse(k, m, us, res[k], rng)
        SH += sh
        PR += pr
        VA += vals
        DP += dp
        pv.to_csv(OUT / "per_variant" / f"{k}.csv", index=False)
    sh = pd.DataFrame(SH)
    sh.to_csv(OUT / "shapley.csv", index=False)
    pd.DataFrame(PR).to_csv(OUT / "pairs.csv", index=False)
    pd.DataFrame(VA).to_csv(OUT / "values.csv", index=False)
    pd.DataFrame(DP).to_csv(OUT / "dip.csv", index=False)
    if len(sh):
        _figure(sh[sh.score == "median"])
    print("wrote", OUT, "failed subsets:", len(failed), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keys", nargs="*")
    a = ap.parse_args()
    run(a.workers, a.keys)
