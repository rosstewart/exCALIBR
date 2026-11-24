"""Which readouts to keep: backward elimination on a proper score, with
stability selection and nested (honest) evaluation. No refit: readouts are
removed by marginalizing them out of the fitted model
(review_assay_contribution.score_masked); refits confirm chosen subsets
later (cluster).

Per joint model with at least two readouts:
  score       Cllr (bits; review_common.cllr) of each evaluation control's
              out-of-bag log LR+ capped at the Very Strong LR: --score median
              (the model's LR: information; review/assay_selection/) or
              percentile (the evidence rule's one-sided bound, so variance a
              readout adds counts against it; review/assay_selection_percentile/).
              1 bit = no evidence, lower is better. Models with fewer than
              MIN_PER_CLASS controls of either class are flagged few_controls
              (no readout decision should be taken from them)
  eliminate   starting from all readouts, repeatedly drop the readout whose
              removal lowers the score most; stop when every removal raises it
              by more than TOL (relative)
  stability   repeat the elimination on N_STAB resamples of the controls
              (within class); keep the readouts retained in >= 50% of them
  nested      on N_NESTED random halvings of the controls (within class),
              choose on one half (elimination; the best single readout; the
              top-2 readouts by univariate separation) and evaluate on the
              other half: Cllr, MCC, determinate rate, errors,
              against keeping all readouts
  Calhoun     whether keeping all readouts beats the best single readout on
              the evaluation half (Calhoun et al. 2026: use the integrated
              score only if it improves on the best single assay)

Outputs (review/assay_selection/): path.csv, stability.csv, nested.csv,
summary.csv.

Usage: python -m mv_analysis.review_assay_selection [--workers N] [--keys ...]
"""
import argparse
import itertools
import functools
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from mv_analysis import review_common as R
from mv_analysis import review_assay_contribution as A

OUT = R.out_dir("assay_selection")
SCORE = "median"  # --score: which capped log LR+ Cllr scores (review_common.evidence_log_lrs)
TOL = 0.01
N_STAB = 100
N_NESTED = 20


class _Scorer:
    """Out-of-bag points and posteriors of the controls per kept readout set (cached)."""

    def __init__(self, key):
        self.key = key
        self.names = tuple(str(n) for n in A._ms(key).dataset_names)
        self._cache = {}
        self._lr = {}

    def get(self, kept):
        kept = tuple(sorted(kept))
        if kept not in self._cache:
            masked = tuple(n for n in self.names if n not in kept)
            pts, post, y, lr = A.oob_eval(self.key, *A.score_masked(self.key, masked), return_lr=True)
            self._cache[kept] = (pts, post, y)
            self._lr[kept] = lr
        return self._cache[kept]

    def loss(self, kept, idx):
        """Cllr (bits) on the controls idx of the capped out-of-bag log LR+
        chosen by SCORE (median: information; percentile: the evidence
        rule's bound, so bootstrap variance counts)."""
        _, _, y = self.get(kept)
        return R.cllr(self._lr[tuple(sorted(kept))][SCORE], y, idx)


def eliminate(sc, idx, record=None):
    cur = list(sc.names)
    base = sc.loss(cur, idx)
    while len(cur) > 1:
        trials = {r: sc.loss([x for x in cur if x != r], idx) for r in cur}
        r, best = min(trials.items(), key=lambda kv: kv[1])
        if record is not None:
            record.append(dict(n_kept=len(cur), loss=base, drop_candidate=r, loss_after=best))
        if best > base * (1 + TOL):
            break
        cur.remove(r)
        base = best
    return cur


def _strat_resample(y, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (True, False)])


def _strat_half(y, rng):
    a = []
    for c in (True, False):
        ix = rng.permutation(np.flatnonzero(y == c))
        a.append(ix[: len(ix) // 2])
    a = np.concatenate(a)
    return a, np.setdiff1d(np.arange(len(y)), a)


def _auc_rank(sc, idx):
    from sklearn.metrics import roc_auc_score
    d, _ = R.prep(sc.key)
    X = np.asarray(d["X"], float)
    dims = [str(n) for n in d["dims"]]
    pts, post, y = sc.get(sc.names)
    rows = np.flatnonzero(d["eval_mask"])
    # the controls oob_eval kept, in order (those with out-of-bag fits)
    mv = R.mv_npz(sc.key)
    rows = rows[mv["oob"][:, rows].any(0)]
    out = {}
    for n in sc.names:
        x = X[rows[idx], dims.index(n)]
        ok = ~np.isnan(x)
        yy = y[idx][ok]
        out[n] = abs(roc_auc_score(yy, x[ok]) - 0.5) if ok.sum() > 5 and 0 < yy.sum() < ok.sum() else 0
    return sorted(out, key=out.get, reverse=True)


def _eval(sc, kept, idx):
    from mv_analysis.review_priors import _metrics
    pts, post, y = sc.get(kept)
    m = _metrics(pts[idx], y[idx])
    return dict(loss=sc.loss(kept, idx), mcc=m["mcc"], det_rate=m["det_rate"], errors=m["errors"])


def run_key(key, seed=0):
    warnings.filterwarnings("ignore")
    A._single_threaded()
    sc = _Scorer(key)
    _, m = R.prep(key)
    _, _, y = sc.get(sc.names)
    allidx = np.arange(len(y))
    rng = np.random.default_rng(seed)
    path = []
    sel_full = eliminate(sc, allidx, record=path)
    path = [dict(key=key, **p) for p in path]
    counts = pd.Series(0, index=list(sc.names))
    for _ in range(N_STAB):
        counts[eliminate(sc, _strat_resample(y, rng))] += 1
    stab = [dict(key=key, readout=n, freq=counts[n] / N_STAB) for n in sc.names]
    stable = [n for n in sc.names if counts[n] / N_STAB >= 0.5] or [counts.idxmax()]
    nested = []
    for rep in range(N_NESTED):
        a, b = _strat_half(y, rng)
        singles = {n: sc.loss([n], a) for n in sc.names}
        choices = {"all readouts": list(sc.names),
                   "backward elimination": eliminate(sc, a),
                   "best single (by Cllr)": [min(singles, key=singles.get)],
                   "top-2 by univariate AUC": _auc_rank(sc, a)[:2]}
        for rule, kept in choices.items():
            nested.append(dict(key=key, rep=rep, rule=rule, kept=";".join(sorted(kept)), n_kept=len(kept),
                               **_eval(sc, kept, b)))
    nd = pd.DataFrame(nested)
    mean = nd.groupby("rule")[["loss", "mcc", "det_rate", "errors"]].mean()
    summ = dict(key=key, panel=m["panel"], gene=m["gene"], n_readouts=len(sc.names),
                selected_full=";".join(sel_full), stable=";".join(stable),
                **{f"{r}|{c}": mean.loc[r, c] for r in mean.index for c in mean.columns},
                all_beats_best_single=bool(mean.loc["all readouts", "loss"] < mean.loc["best single (by Cllr)", "loss"]),
                score=SCORE, few_controls=bool(min(y.sum(), (~y).sum()) < R.MIN_PER_CLASS))
    return path, stab, nested, summ


def _warm_one(key, masked):
    A._single_threaded()
    warnings.filterwarnings("ignore")
    try:
        A.score_masked(key, masked)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{key} {masked}: {type(e).__name__}: {e}"


def warm(keys, workers, depth=4, max_all=8):
    """Score in parallel, across all models at once, the readout subsets the
    elimination is likely to visit (all subsets for models with up to
    max_all readouts; otherwise every subset with up to `depth` readouts
    removed), so the per-model elimination below mostly reads the cache
    (review_assay_contribution.score_masked). Subsets already cached are
    skipped."""
    import hashlib
    jobs = []
    for k in keys:
        names = [str(n) for n in R.prep(k)[0]["dims"]]
        n = len(names)
        top = n - 1 if n <= max_all else min(depth, n - 1)
        for r in range(1, top + 1):
            for masked in itertools.combinations(sorted(names), r):
                tag = hashlib.md5("|".join(masked).encode()).hexdigest()[:12]
                if not (A.CACHE / k / f"{tag}.npz").exists():
                    jobs.append((n, k, masked))
    jobs.sort(key=lambda j: -j[0])  # largest models first
    print(f"warming {len(jobs)} subsets with {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_warm_one, k, m) for _, k, m in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            if f.result():
                print("FAILED", f.result(), flush=True)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)


def run(workers=6, keys=None, warm_workers=0):
    ks = [k for k, m in R.paper_keys()
          if (keys is None or k in keys) and len(R.prep(k)[0]["dims"]) >= 2]
    ks.sort(key=lambda k: -len(R.prep(k)[0]["dims"]))  # largest first: no long tail at the end
    if warm_workers:
        warm(ks, warm_workers)
    P, S, N, U = [], [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for path, stab, nested, summ in ex.map(run_key, ks):
            P += path
            S += stab
            N += nested
            U.append(summ)
            print(summ["key"], "stable:", summ["stable"], flush=True)
    pd.DataFrame(P).to_csv(OUT / "path.csv", index=False)
    pd.DataFrame(S).to_csv(OUT / "stability.csv", index=False)
    pd.DataFrame(N).to_csv(OUT / "nested.csv", index=False)
    pd.DataFrame(U).to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--score", choices=["median", "percentile"], default="median")
    ap.add_argument("--warm-workers", type=int, default=0,
                    help="first score likely subsets in parallel across models (see warm)")
    a = ap.parse_args()
    SCORE = a.score
    if SCORE != "median":
        OUT = R.out_dir(f"assay_selection_{SCORE}")
    run(a.workers, a.keys, a.warm_workers)
