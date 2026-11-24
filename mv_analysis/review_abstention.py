"""Is the joint model's accuracy gain over the independent comparator just
more abstention? (functional panel)

The paper compares classification metrics at each method's own operating
point, where a method that abstains more makes fewer errors. Here both are
scored with measures that do not reward abstaining, on the same out-of-bag
evaluation controls:

  Cllr        (bits, capped at the Very Strong LR; review_common.cllr) of the
              LR each method claims: abstaining is LR = 1 and costs
              information. Joint: points / 8 x log c(out-of-bag prior), at
              p = 5, 10 and 50, and the median log LR+ (the model's estimate).
              Comparator: each assay's points converted at that assay's own
              prior (the LR its calibration claims; review/priors/priors.csv,
              the joint gene prior where a readout has no entry), combined by
              the paper's non-conflicting rule on the log LRs (largest
              magnitude when all agree in sign, LR = 1 when they conflict).
              Also with every comparator point at the joint gene prior
              (sensitivity).
  matched coverage  expected errors among each method's most confident calls
              (ranked by |points|, ties broken at random, 200 draws), with
              both the joint (at p) and the comparator making the same number
              of calls per model: the smaller of their two call counts.
  plus the operating-point metrics (determinate rate, errors, accuracy of
  calls, MCC) for reference.

Caveat favouring the comparator: its calibrations use ClinVar controls that
include these (in sample); the joint model is scored out of bag.

Outputs (review/abstention/): per_model.csv, panel.csv
Usage: python -m mv_analysis.review_abstention
"""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis import review_common as R

OUT = R.out_dir("abstention")
N_DRAW = 200


def comparator_log_lr(key, rows, gene_prior):
    """(points, log LR at own priors, log LR at gene prior) of the comparator
    for model rows."""
    from mv_analysis import uv_agg, uv_sources
    from mv_analysis.report import _UV_DISPATCH_GENE_SET
    _, m = R.prep(key)
    with contextlib.redirect_stdout(io.StringIO()):
        ms = config.build_ms_for_gene_cluster(m["gene_set"], m["gene"], cluster_idx=m["cluster_idx"],
                                              results_json=config.PAPER_RESULTS_JSON)
        g = m["gene"].upper()
        uv = uv_sources.load_uv_points(g, ms, _UV_DISPATCH_GENE_SET.get(m["gene_set"], m["gene_set"]))
    if uv is None:
        return None
    names, mat = [str(n) for n in uv[0]], np.asarray(uv[1], float)[:, rows]
    pri = pd.read_csv(R.REVIEW_DIR / "priors" / "priors.csv")
    pri = dict(zip(pri[pri.key == key].dataset, pri[pri.key == key].assay_prior))
    own = np.array([pri.get(n, pri.get(f"{n}_clinvar_2018", np.nan)) for n in names])
    n_own = int(np.isfinite(own).sum())
    own = np.where(np.isfinite(own), own, gene_prior)
    llr_own = mat / 8 * np.array([R.log_c(p) for p in own])[:, None]
    llr_gene = mat / 8 * R.log_c(gene_prior)
    pts = uv_agg.aggregate_nonconflicting(mat)
    return pts, uv_agg.aggregate_nonconflicting(llr_own), uv_agg.aggregate_nonconflicting(llr_gene), \
        len(names), n_own


def _risk(pts, y, k, rng):
    """Expected error share among the k most confident calls (|points|, random ties)."""
    a = np.abs(np.nan_to_num(pts))
    if k == 0:
        return np.nan
    if (a > 0).sum() < k:
        return np.nan
    ok = np.where(pts > 0, y, ~y)
    e = []
    for _ in range(N_DRAW):
        o = np.lexsort((rng.random(len(a)), -a))[:k]
        e.append(float((~ok[o]).mean()))
    return float(np.mean(e))


def _ops(pts, y):
    p = np.nan_to_num(np.asarray(pts, float))
    det = p != 0
    err = ((p > 0) & ~y) | ((p < 0) & y)
    tp, fp, tn, fn = ((p > 0) & y).sum(), ((p > 0) & ~y).sum(), ((p < 0) & ~y).sum(), ((p < 0) & y).sum()
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return dict(det_rate=det.mean(), n_calls=int(det.sum()), errors=int(err.sum()),
                acc_calls=1 - err[det].mean() if det.any() else np.nan,
                mcc=(tp * tn - fp * fn) / den if den else np.nan)


def run():
    warnings.filterwarnings("ignore")
    from mv_analysis.review_percentile_info import model_lrs
    rng = np.random.default_rng(0)
    rows = []
    for key, m in R.paper_keys("functional"):
        d, _ = R.prep(key)
        mv = R.mv_npz(key)
        ev = np.flatnonzero(d["eval_mask"])
        ev = ev[mv["oob"][:, ev].any(0)]
        y = np.asarray(d["plp"], bool)[ev]
        lrs, y2 = model_lrs(key)  # same rows (eval controls with out-of-bag fits)
        assert np.array_equal(y, y2)
        gene_prior = float(m["mv_median_prior"])
        c = comparator_log_lr(key, ev, gene_prior)
        if c is None:
            print(key, "no comparator", flush=True)
            continue
        cpts, cllr_own, cllr_gene, n_read, n_own = c
        cap = R.log_c(gene_prior)
        meths = {}
        for p in (5, 10, 50):
            meths[f"joint p{p}"] = (R.oob_points(key, p).set_index("row").loc[ev, "points"].to_numpy(float),
                                    lrs[p]["points"])
        meths["joint median LR"] = (None, lrs[5]["median"])
        meths["comparator"] = (cpts, np.clip(np.nan_to_num(cllr_own), -cap, cap))
        meths["comparator (gene prior)"] = (cpts, np.clip(np.nan_to_num(cllr_gene), -cap, cap))
        ccalls = int((np.nan_to_num(cpts) != 0).sum())
        # matched coverage: each joint percentile vs the comparator at the
        # smaller of their two call counts (both can be ranked that far)
        common = {n: min(int((np.nan_to_num(meths[n][0]) != 0).sum()), ccalls) for n in meths
                  if n.startswith("joint p")}
        few = bool(min(y.sum(), (~y).sum()) < R.MIN_PER_CLASS)
        for name, (pts, llr) in meths.items():
            rec = dict(key=key, gene=m["gene"], method=name, n_plp=int(y.sum()), n_blb=int((~y).sum()),
                       few_controls=few, n_readouts=n_read, n_own_prior=n_own,
                       cllr=R.cllr(llr, y) if y.any() and (~y).any() else np.nan)
            if pts is not None:
                rec |= _ops(pts, y)
                for n, k in common.items():
                    if name in (n, "comparator"):
                        rec[f"errors_matched_{n.split()[1]}"] = _risk(pts, y, k, rng) * k
                        rec[f"calls_matched_{n.split()[1]}"] = k
            rows.append(rec)
        print(key, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "per_model.csv", index=False)
    ok = t[~t.few_controls]
    agg = []
    for name, x in t.groupby("method", sort=False):
        xo = ok[ok.method == name]
        agg.append(dict(method=name, n_models=len(xo), median_cllr=xo.cllr.median(),
                        pooled_calls=x.n_calls.sum() if "n_calls" in x else np.nan,
                        pooled_errors=x.errors.sum() if "errors" in x else np.nan,
                        pooled_det_rate=(x.n_calls.sum() / (x.n_plp + x.n_blb).sum()) if "n_calls" in x else np.nan,
                        median_mcc=xo.mcc.median() if "mcc" in xo else np.nan))
    a = pd.DataFrame(agg)
    for pp in ("p5", "p10", "p50"):
        x = t.set_index(["key", "method"])
        j = x.xs(f"joint {pp}", level="method")[f"errors_matched_{pp}"]
        c = x.xs("comparator", level="method")[f"errors_matched_{pp}"]
        k = x.xs("comparator", level="method")[f"calls_matched_{pp}"]
        a.loc[a.method == f"joint {pp}", "matched_calls"] = float(k.sum())
        a.loc[a.method == f"joint {pp}", "matched_errors_joint"] = float(j.sum())
        a.loc[a.method == f"joint {pp}", "matched_errors_comparator"] = float(c.sum())
        a.loc[a.method == f"joint {pp}", "models_joint_fewer_errors"] = int((j < c - 1e-9).sum())
        a.loc[a.method == f"joint {pp}", "models_joint_more_errors"] = int((j > c + 1e-9).sum())
    # paired per-model Cllr: joint - comparator
    piv = ok.pivot(index="key", columns="method", values="cllr")
    for name in ("joint p5", "joint p10", "joint p50", "joint median LR"):
        dlt = piv[name] - piv["comparator"]
        a.loc[a.method == name, "models_better_than_comparator"] = int((dlt < 0).sum())
        a.loc[a.method == name, "median_cllr_minus_comparator"] = float(dlt.median())
    a.to_csv(OUT / "panel.csv", index=False)
    pd.set_option("display.width", 250)
    print(a.round(3).to_string(index=False))
    cols = ["gene", "key", "method", "n_plp", "n_blb", "few_controls", "cllr", "det_rate", "errors", "acc_calls",
            "errors_matched_p5", "errors_matched_p50", "calls_matched_p5"]
    print(t[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    run()
